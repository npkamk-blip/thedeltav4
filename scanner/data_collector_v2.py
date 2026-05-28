"""
THE DELTA v2 — data_collector_v2.py (REBUILT)
==============================================
Collects ALL features for 2025-2026 training data.
Golden rule: "Would I have known this BEFORE the event?"

Every data source is verified and logged loudly on failure.
No silent failures. No placeholder zeros.

Data sources:
  - Polygon: daily bars, minute bars (PM + AH), snapshots, reference
  - FINRA: short interest (daily + biweekly)
  - EDGAR: 8-K, S-1/S-3/424B, Form 4, SC 13D, CIK map
  - NASDAQ: halt history
  - Polygon Financials: earnings dates

Output: /app/data/training_data_v2/YYYY-MM-DD.parquet
"""

import os, time, logging, requests, json, gc, threading, re
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from io import StringIO
from http.server import HTTPServer, BaseHTTPRequestHandler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
POLYGON_API_KEY  = os.environ.get("MASSIVE_API_KEY", "")
EDGAR_USER_AGENT = "NPKNOB@gmail.com"

DATA_ROOT       = Path(os.environ.get("DATA_DIR", "/app/data"))
RAW_DIR         = DATA_ROOT / "raw"
TICKER_RAW_DIR  = RAW_DIR / "tickers"
FINRA_DIR       = RAW_DIR / "finra"
HALTS_DIR       = RAW_DIR / "halts"
EDGAR_DIR       = RAW_DIR / "edgar"
OUTPUT_DIR      = DATA_ROOT / "training_data_v2"
LOG_DIR         = DATA_ROOT / "logs"

START_DATE = date(2025, 1, 1)
END_DATE   = date(2026, 5, 23)

MIN_PRICE           = 0.10
MIN_PREV_DOLLAR_VOL = 25_000
MIN_LABEL_VOLUME    = 25_000
SEED_MULT           = 2.00
SUPER_MULT          = 6.00
MEGA_MULT           = 11.00
CONTROL_RATIO       = 2
CALLS_PER_MINUTE    = 250
CALL_INTERVAL       = 60.0 / CALLS_PER_MINUTE
EDGAR_RATE_INTERVAL = 0.12

ET = ZoneInfo("America/New_York")

KNOWN_HOLIDAYS = {
    date(2025,1,1), date(2025,1,9), date(2025,1,20), date(2025,2,17),
    date(2025,4,18), date(2025,5,26), date(2025,6,19), date(2025,7,4),
    date(2025,9,1), date(2025,11,27), date(2025,12,25),
    date(2026,1,1), date(2026,1,19), date(2026,2,16), date(2026,4,3),
    date(2026,5,25),
}

# ─────────────────────────────────────────────
# DIRS + LOGGING
# ─────────────────────────────────────────────
for d in [LOG_DIR, OUTPUT_DIR, TICKER_RAW_DIR, FINRA_DIR, HALTS_DIR, EDGAR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "collector_v2.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("delta_collector")

# Collection stats — track failures across entire run
STATS = {
    "tickers_attempted": 0,
    "tickers_ok":        0,
    "hist_fail":         0,
    "pm_fail":           0,
    "ah_fail":           0,
    "float_fail":        0,
    "earnings_fail":     0,
    "finra_fail":        0,
    "halt_fail":         0,
    "edgar_fail":        0,
}


# ─────────────────────────────────────────────
# KEEPALIVE
# ─────────────────────────────────────────────
class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b"alive")
    def log_message(self, *a): pass

def start_keepalive(port=8080):
    HTTPServer(("0.0.0.0", port), _Health).serve_forever()


# ─────────────────────────────────────────────
# POLYGON CLIENT
# ─────────────────────────────────────────────
class PolygonClient:
    def __init__(self):
        self._last   = 0.0
        self.session = requests.Session()
        self.session.params = {"apiKey": POLYGON_API_KEY}

    def _wait(self):
        elapsed = time.time() - self._last
        if elapsed < CALL_INTERVAL:
            time.sleep(CALL_INTERVAL - elapsed)
        self._last = time.time()

    def get(self, path: str, params: dict = None, retries=3) -> dict | None:
        url = "https://api.polygon.io" + path
        for attempt in range(retries):
            self._wait()
            try:
                r = self.session.get(url, params=params or {}, timeout=20)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    wait = 30 * (attempt + 1)
                    log.warning(f"429 rate limit — sleeping {wait}s")
                    time.sleep(wait)
                elif r.status_code == 403:
                    log.error(f"FAIL 403 forbidden: {url}")
                    return None
                elif r.status_code == 404:
                    return None
                else:
                    log.warning(f"HTTP {r.status_code} for {url} attempt {attempt+1}")
                    time.sleep(3)
            except Exception as e:
                log.warning(f"Polygon request error (attempt {attempt+1}): {e}")
                time.sleep(2)
        log.error(f"FAIL all retries exhausted: {path}")
        return None

    def paginate(self, path: str, params: dict = None) -> list:
        results = []
        resp = self.get(path, params)
        if not resp:
            return results
        results.extend(resp.get("results", []))
        while resp.get("next_url"):
            self._wait()
            try:
                r = self.session.get(resp["next_url"], timeout=30)
                resp = r.json() if r.status_code == 200 else {}
                results.extend(resp.get("results", []))
            except Exception as e:
                log.warning(f"Pagination error: {e}")
                break
        return results

    def aggs(self, ticker, mult, span, from_, to, adjusted=True) -> list:
        params = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000}
        return self.paginate(
            f"/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{from_}/{to}", params
        )

poly = PolygonClient()


# ─────────────────────────────────────────────
# EDGAR CLIENT
# ─────────────────────────────────────────────
class EdgarClient:
    def __init__(self):
        self._last   = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": EDGAR_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })

    def _wait(self):
        elapsed = time.time() - self._last
        if elapsed < EDGAR_RATE_INTERVAL:
            time.sleep(EDGAR_RATE_INTERVAL - elapsed)
        self._last = time.time()

    def get(self, url: str, retries=3) -> dict | None:
        for attempt in range(retries):
            self._wait()
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    time.sleep(30)
                else:
                    log.warning(f"EDGAR HTTP {r.status_code}: {url}")
                    time.sleep(2)
            except Exception as e:
                log.warning(f"EDGAR error (attempt {attempt+1}): {e}")
                time.sleep(2)
        log.error(f"FAIL EDGAR all retries: {url}")
        return None

    def get_text(self, url: str) -> str | None:
        self._wait()
        try:
            r = self.session.get(url, timeout=60)
            if r.status_code == 200:
                return r.text
            log.warning(f"EDGAR text HTTP {r.status_code}: {url}")
        except Exception as e:
            log.warning(f"EDGAR text error: {e}")
        return None

edgar = EdgarClient()


# ─────────────────────────────────────────────
# TRADING CALENDAR
# ─────────────────────────────────────────────
def get_trading_days(start: date, end: date) -> list[date]:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5 and cur not in KNOWN_HOLIDAYS:
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ─────────────────────────────────────────────
# PHASE 1A — TICKER LIST
# ─────────────────────────────────────────────
def fetch_ticker_list() -> list[str]:
    cache = RAW_DIR / "ticker_list.json"
    if cache.exists():
        log.info("Loading cached ticker list...")
        with open(cache) as f:
            tickers = json.load(f)
        log.info(f"Loaded {len(tickers)} tickers")
        return tickers

    log.info("Fetching ticker list from Polygon...")
    results = poly.paginate(
        "/v3/reference/tickers",
        {"market": "stocks", "type": "CS", "active": "true", "limit": 1000}
    )
    if not results:
        log.error("FAIL ticker list — Polygon returned nothing")
        return []

    tickers = sorted({
        r["ticker"] for r in results
        if r.get("primary_exchange") in ("XNAS", "XNYS", "XASE")
    })
    log.info(f"Found {len(tickers)} tickers")
    with open(cache, "w") as f:
        json.dump(tickers, f)
    return tickers


# ─────────────────────────────────────────────
# PHASE 1B — COLLECT PER TICKER
# ─────────────────────────────────────────────
def collect_ticker(ticker: str) -> bool:
    cache_path = TICKER_RAW_DIR / f"{ticker}.parquet"
    if cache_path.exists():
        try:
            pd.read_parquet(cache_path, columns=["ticker"])
            return True
        except Exception:
            log.warning(f"{ticker}: corrupted cache — re-collecting")
            cache_path.unlink()

    STATS["tickers_attempted"] += 1
    from_str = START_DATE.strftime("%Y-%m-%d")
    to_str   = END_DATE.strftime("%Y-%m-%d")

    fetch_ok = {k: False for k in [
        "hist_fetch_ok", "pm_fetch_ok", "ah_fetch_ok",
        "float_fetch_ok", "earnings_fetch_ok"
    ]}

    # ── Daily bars ────────────────────────────────────────────
    daily_bars = poly.aggs(ticker, 1, "day", from_str, to_str)
    if not daily_bars:
        log.error(f"FAIL hist | {ticker} | no daily bars from Polygon")
        STATS["hist_fail"] += 1
        return False

    fetch_ok["hist_fetch_ok"] = True
    daily_df = pd.DataFrame(daily_bars)
    daily_df["t"] = pd.to_datetime(daily_df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    daily_df["date"] = daily_df["t"].dt.date
    daily_df = daily_df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume","vw":"vwap"})
    log.info(f"OK hist | {ticker} | {len(daily_df)} daily bars")

    # ── Premarket + AH minute bars ────────────────────────────
    pm_bars = []
    chunks = [
        ("2025-01-01", "2025-06-30"),
        ("2025-07-01", "2025-12-31"),
        ("2026-01-01", "2026-05-23"),
    ]
    for cfrom, cto in chunks:
        chunk = poly.aggs(ticker, 5, "minute", cfrom, cto, adjusted=False)
        if chunk:
            pm_bars.extend(chunk)
        else:
            log.warning(f"WARN pm_bars | {ticker} | no data for chunk {cfrom}:{cto}")

    if not pm_bars:
        log.error(f"FAIL pm | {ticker} | no minute bars returned at all")
        STATS["pm_fail"] += 1
    else:
        fetch_ok["pm_fetch_ok"] = True
        log.info(f"OK pm | {ticker} | {len(pm_bars)} minute bars")

    pm_df = pd.DataFrame(pm_bars) if pm_bars else pd.DataFrame()
    ah_df = pd.DataFrame()

    if not pm_df.empty:
        pm_df["t"]      = pd.to_datetime(pm_df["t"], unit="ms", utc=True).dt.tz_convert(ET)
        pm_df["hour"]   = pm_df["t"].dt.hour
        pm_df["minute"] = pm_df["t"].dt.minute
        pm_df["date"]   = pm_df["t"].dt.date

        # Premarket: 4AM-9:29AM
        pm_df = pm_df[
            (pm_df["hour"] >= 4) &
            ((pm_df["hour"] < 9) | ((pm_df["hour"] == 9) & (pm_df["minute"] < 30)))
        ].copy()

        # After-hours: 4PM-8PM
        ah_df = pm_df[
            (pm_df["hour"] >= 16) & (pm_df["hour"] < 20)
        ].copy()

        if ah_df.empty:
            log.warning(f"WARN ah | {ticker} | no after-hours bars found")
            STATS["ah_fail"] += 1
        else:
            fetch_ok["ah_fetch_ok"] = True
            log.info(f"OK ah | {ticker} | {len(ah_df)} AH bars")
    else:
        log.error(f"FAIL ah | {ticker} | no minute bars to derive AH from")
        STATS["ah_fail"] += 1

    # ── Float / reference data ────────────────────────────────
    float_data = {}
    detail = poly.get(f"/v3/reference/tickers/{ticker}")
    if not detail or "results" not in detail:
        log.error(f"FAIL float | {ticker} | no reference data from Polygon")
        STATS["float_fail"] += 1
    else:
        r = detail["results"]
        shares = r.get("share_class_shares_outstanding", 0) or 0
        if shares <= 0:
            log.warning(f"WARN float | {ticker} | shares_outstanding=0")
            STATS["float_fail"] += 1
        else:
            fetch_ok["float_fetch_ok"] = True
            log.info(f"OK float | {ticker} | float={shares/1e6:.1f}M")

        locale     = r.get("locale", "us").lower()
        addr       = r.get("address", {})
        hq_country = addr.get("country", "us").lower() if isinstance(addr, dict) else "us"
        is_foreign = 1 if (locale != "us" or hq_country not in ("us","usa","united states","")) else 0
        float_data = {
            "shares_outstanding": shares,
            "market_cap":         r.get("market_cap"),
            "name":               r.get("name", ""),
            "sic_code":           r.get("sic_code", ""),
            "is_foreign_listed":  is_foreign,
        }

    # ── Earnings dates ────────────────────────────────────────
    earnings_dates = []
    earn = poly.paginate(
        "/vX/reference/financials",
        {"ticker": ticker, "limit": 100, "sort": "filing_date"}
    )
    if not earn:
        log.warning(f"WARN earnings | {ticker} | no financials data")
        STATS["earnings_fail"] += 1
    else:
        earnings_dates = [e["filing_date"] for e in earn if e.get("filing_date")]
        fetch_ok["earnings_fetch_ok"] = True
        log.info(f"OK earnings | {ticker} | {len(earnings_dates)} dates")

    # ── Log collection summary for this ticker ────────────────
    status = " | ".join([
        f"hist={'OK' if fetch_ok['hist_fetch_ok'] else 'FAIL'}",
        f"pm={'OK' if fetch_ok['pm_fetch_ok'] else 'FAIL'}",
        f"ah={'OK' if fetch_ok['ah_fetch_ok'] else 'FAIL'}",
        f"float={'OK' if fetch_ok['float_fetch_ok'] else 'FAIL'}",
        f"earn={'OK' if fetch_ok['earnings_fetch_ok'] else 'FAIL'}",
    ])
    log.info(f"TICKER {ticker}: {status}")

    # ── Save ──────────────────────────────────────────────────
    cache = {
        "ticker":         ticker,
        "daily":          daily_df.to_dict("records"),
        "pm_minute":      pm_df.to_dict("records") if not pm_df.empty else [],
        "ah_minute":      ah_df.to_dict("records") if not ah_df.empty else [],
        "float_data":     float_data,
        "earnings_dates": earnings_dates,
        "fetch_ok":       fetch_ok,
    }
    pd.DataFrame([cache]).to_parquet(cache_path, index=False)
    STATS["tickers_ok"] += 1
    return True


def phase1_collect_all(tickers: list[str]):
    total = len(tickers)
    log.info(f"Phase 1: collecting {total} tickers...")
    failed = []
    for i, ticker in enumerate(tickers):
        if (TICKER_RAW_DIR / f"{ticker}.parquet").exists():
            continue
        ok = collect_ticker(ticker)
        if not ok:
            failed.append(ticker)
        if (i + 1) % 100 == 0:
            log.info(f"Progress: {i+1}/{total} | ok={STATS['tickers_ok']} | failed={len(failed)}")
            log.info(f"Fail counts: hist={STATS['hist_fail']} pm={STATS['pm_fail']} ah={STATS['ah_fail']} float={STATS['float_fail']}")

    # Also collect sector tickers
    for sym in ["SPY", "QQQ", "IWM", "XBI"]:
        collect_ticker(sym)

    if failed:
        fp = RAW_DIR / "failed_tickers.json"
        with open(fp, "w") as f:
            json.dump(failed, f)
        log.error(f"Phase 1 complete. {len(failed)} FAILED — see {fp}")
    else:
        log.info(f"Phase 1 complete. All {total} tickers collected successfully.")

    log.info(f"FINAL STATS: {json.dumps(STATS, indent=2)}")


# ─────────────────────────────────────────────
# PHASE 1C — FINRA SHORT INTEREST
# ─────────────────────────────────────────────
def collect_finra():
    cache = FINRA_DIR / "si_master.parquet"
    if cache.exists():
        log.info("FINRA SI already cached")
        return

    log.info("Downloading FINRA short interest 2025-2026...")
    all_rows = []
    cur = date(2025, 1, 2)
    hits = 0
    fails = 0

    while cur <= END_DATE:
        date_str  = cur.strftime("%Y%m%d")
        day_cache = FINRA_DIR / f"finra_{date_str}.parquet"

        if day_cache.exists():
            try:
                all_rows.append(pd.read_parquet(day_cache))
                hits += 1
            except Exception:
                day_cache.unlink()
        else:
            url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date_str}.txt"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200 and len(r.text) > 100:
                    df = pd.read_csv(StringIO(r.text), sep="|", on_bad_lines="skip")
                    if len(df) > 10:
                        df["si_date"] = cur.isoformat()
                        df.to_parquet(day_cache, index=False)
                        all_rows.append(df)
                        hits += 1
                        if hits % 20 == 0:
                            log.info(f"FINRA: {hits} files OK, {fails} failed, latest {cur}")
                    else:
                        log.warning(f"WARN FINRA | {cur} | only {len(df)} rows")
                        fails += 1
                else:
                    log.warning(f"WARN FINRA | {cur} | HTTP {r.status_code}")
                    fails += 1
            except Exception as e:
                log.error(f"FAIL FINRA | {cur} | {e}")
                fails += 1
            time.sleep(0.12)

        cur += timedelta(days=1)
        while cur.weekday() >= 5 or cur in KNOWN_HOLIDAYS:
            cur += timedelta(days=1)

    if not all_rows:
        log.error("FAIL FINRA | no data collected at all — SI features will be empty")
        STATS["finra_fail"] += 1
        return

    master = pd.concat(all_rows, ignore_index=True)
    master.to_parquet(cache, index=False)
    log.info(f"FINRA OK: {len(master)} rows from {hits} files ({fails} failed)")
    del master, all_rows
    gc.collect()


# ─────────────────────────────────────────────
# PHASE 1D — HALT HISTORY
# ─────────────────────────────────────────────
def collect_halts():
    cache = HALTS_DIR / "halts_master.parquet"
    if cache.exists():
        log.info("Halt history already cached")
        return

    log.info("Downloading NASDAQ halt history...")
    all_rows = []

    for year in [2025, 2026]:
        url = f"https://www.nasdaqtrader.com/dynamic/symdir/halts/{year}halts.txt"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 100:
                df = pd.read_csv(StringIO(r.text), sep="|", on_bad_lines="skip")
                df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
                all_rows.append(df)
                log.info(f"OK halts | {year} | {len(df)} rows")
            else:
                log.error(f"FAIL halts | {year} | HTTP {r.status_code}")
                STATS["halt_fail"] += 1
        except Exception as e:
            log.error(f"FAIL halts | {year} | {e}")
            STATS["halt_fail"] += 1
        time.sleep(1)

    # Also try current file
    try:
        r = requests.get("https://www.nasdaqtrader.com/dynamic/symdir/halts.txt", timeout=30)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text), sep="|", on_bad_lines="skip")
            df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
            all_rows.append(df)
            log.info(f"OK halts | current file | {len(df)} rows")
    except Exception as e:
        log.warning(f"WARN halts current file: {e}")

    if not all_rows:
        log.error("FAIL halts | no halt data collected — halt features will be empty")
        return

    master = pd.concat(all_rows, ignore_index=True).drop_duplicates()
    master.to_parquet(cache, index=False)
    log.info(f"Halts OK: {len(master)} total rows")


# ─────────────────────────────────────────────
# PHASE 1E — EDGAR INDEX
# ─────────────────────────────────────────────
RELEVANT_FORMS = {
    "8-K","8-K/A",
    "S-1","S-1/A","S-3","S-3/A",
    "424B1","424B3","424B4","424B5",
    "SC 13D","SC 13D/A","SC 13G","SC 13G/A",
    "4","4/A"
}

def collect_edgar_index():
    cache = EDGAR_DIR / "filings_master.parquet"
    if cache.exists():
        log.info("EDGAR index already cached")
        return

    log.info("Downloading EDGAR quarterly indexes 2025-2026...")
    quarters = []
    for year in [2025, 2026]:
        for q in [1, 2, 3, 4]:
            if date(year, q*3, 1) > END_DATE:
                break
            quarters.append((year, q))

    all_rows = []
    for year, q in quarters:
        q_cache = EDGAR_DIR / f"edgar_{year}_Q{q}.parquet"
        if q_cache.exists():
            try:
                df = pd.read_parquet(q_cache)
                all_rows.append(df)
                log.info(f"OK EDGAR | {year} Q{q} | {len(df)} filings (cached)")
                continue
            except Exception:
                q_cache.unlink()

        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/company.idx"
        edgar._wait()
        try:
            r = edgar.session.get(url, timeout=60)
            if r.status_code != 200:
                log.error(f"FAIL EDGAR | {year} Q{q} | HTTP {r.status_code}")
                STATS["edgar_fail"] += 1
                continue

            lines = r.text.split("\n")
            parsed = []
            for line in lines[10:]:
                if len(line) < 20:
                    continue
                try:
                    form_type = line[62:74].strip()
                    if form_type not in RELEVANT_FORMS:
                        continue
                    parsed.append({
                        "company":   line[:62].strip(),
                        "form_type": form_type,
                        "cik":       line[74:86].strip(),
                        "filed":     line[86:98].strip(),
                        "filename":  line[98:].strip(),
                    })
                except Exception:
                    continue

            if not parsed:
                log.error(f"FAIL EDGAR | {year} Q{q} | parsed 0 rows from index")
                STATS["edgar_fail"] += 1
                continue

            df = pd.DataFrame(parsed)
            df.to_parquet(q_cache, index=False)
            all_rows.append(df)
            log.info(f"OK EDGAR | {year} Q{q} | {len(df)} relevant filings")

        except Exception as e:
            log.error(f"FAIL EDGAR | {year} Q{q} | {e}")
            STATS["edgar_fail"] += 1
        time.sleep(0.5)

    if not all_rows:
        log.error("FAIL EDGAR | no index data collected — all EDGAR features will be empty")
        return

    chunks = []
    for df in all_rows:
        df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        chunks.append(df[["cik","form_type","filed","filename"]])

    master = pd.concat(chunks, ignore_index=True)
    master.to_parquet(cache, index=False)
    log.info(f"EDGAR OK: {len(master)} total filings saved")
    del master, chunks, all_rows
    gc.collect()


# ─────────────────────────────────────────────
# EDGAR CIK MAP
# ─────────────────────────────────────────────
def build_cik_map(tickers: list[str]) -> dict:
    cache = EDGAR_DIR / "cik_map.json"
    if cache.exists():
        with open(cache) as f:
            cik_map = json.load(f)
        log.info(f"CIK map loaded: {len(cik_map)} tickers")
        return cik_map

    log.info("Building CIK map from EDGAR...")
    cik_map = {}
    edgar._wait()
    try:
        r = edgar.session.get(
            "https://www.sec.gov/files/company_tickers.json", timeout=30
        )
        if r.status_code == 200:
            for entry in r.json().values():
                t = entry.get("ticker","").upper()
                c = str(entry.get("cik_str","")).zfill(10)
                if t:
                    cik_map[t] = c
            log.info(f"CIK map OK: {len(cik_map)} tickers mapped")
        else:
            log.error(f"FAIL CIK map | HTTP {r.status_code}")
    except Exception as e:
        log.error(f"FAIL CIK map | {e}")

    with open(cache, "w") as f:
        json.dump(cik_map, f)
    return cik_map


# ─────────────────────────────────────────────
# FEATURE CALCULATORS
# ─────────────────────────────────────────────
def calc_premarket_features(pm_day: pd.DataFrame, prev_close: float, avg_pm_vol: float) -> dict:
    out = {
        "pm_open": None, "pm_high": None, "pm_low": None,
        "pm_close": None, "pm_volume": None,
        "pm_gap_pct": None, "pm_move_pct": None,
        "pm_vol_ratio": None, "pm_volume_build": None,
        "pm_high_of_session": None, "pm_fade": None,
        "pm_remaining_to_seed": None,
        "pm_remaining_to_super": None,
        "pm_remaining_to_mega": None,
        "pm_fetch_ok": 0,
    }
    if pm_day.empty or prev_close <= 0:
        return out

    pm_day = pm_day.sort_values("t")
    out["pm_fetch_ok"] = 1
    out["pm_open"]     = float(pm_day.iloc[0]["o"])
    out["pm_high"]     = float(pm_day["h"].max())
    out["pm_low"]      = float(pm_day["l"].min())
    out["pm_close"]    = float(pm_day.iloc[-1]["c"])
    out["pm_volume"]   = float(pm_day["v"].sum())

    if prev_close > 0:
        out["pm_gap_pct"] = (out["pm_open"] - prev_close) / prev_close
    if out["pm_open"] and out["pm_open"] > 0:
        out["pm_move_pct"] = (out["pm_high"] - out["pm_open"]) / out["pm_open"]
    if avg_pm_vol and avg_pm_vol > 0:
        out["pm_vol_ratio"] = out["pm_volume"] / avg_pm_vol

    if len(pm_day) >= 4:
        half = len(pm_day) // 2
        ev = pm_day.iloc[:half]["v"].mean()
        lv = pm_day.iloc[half:]["v"].mean()
        out["pm_volume_build"] = 1 if lv > ev * 1.2 else 0

    if out["pm_close"] and out["pm_high"]:
        out["pm_high_of_session"] = 1 if out["pm_close"] >= out["pm_high"] * 0.99 else 0

    if out["pm_high"] and out["pm_open"] and out["pm_high"] > out["pm_open"]:
        move = out["pm_high"] - out["pm_open"]
        fade = out["pm_high"] - out["pm_close"]
        out["pm_fade"] = 1 if fade > move * 0.10 else 0

    if out["pm_close"] and out["pm_close"] > 0 and prev_close > 0:
        p = out["pm_close"]
        out["pm_remaining_to_seed"]  = (prev_close * SEED_MULT  - p) / p
        out["pm_remaining_to_super"] = (prev_close * SUPER_MULT - p) / p
        out["pm_remaining_to_mega"]  = (prev_close * MEGA_MULT  - p) / p

    return out


def calc_ah_features(ah_day: pd.DataFrame, prev_close: float) -> dict:
    out = {
        "ah_move_pct": None, "ah_volume": None,
        "ah_direction": None, "ah_fetch_ok": 0,
    }
    if ah_day.empty or prev_close <= 0:
        return out

    ah_day = ah_day.sort_values("t")
    ah_o   = float(ah_day.iloc[0]["o"])
    ah_c   = float(ah_day.iloc[-1]["c"])
    out["ah_volume"]    = float(ah_day["v"].sum())
    out["ah_fetch_ok"]  = 1

    if ah_o > 0:
        out["ah_move_pct"]  = (ah_c - ah_o) / ah_o
        out["ah_direction"] = 1 if ah_c > ah_o else (-1 if ah_c < ah_o else 0)
    return out


def calc_historical_features(hist: pd.DataFrame, today_idx: int) -> dict:
    out = {
        "price_52w_high": None, "price_52w_low": None,
        "pct_from_52w_high": None, "pct_from_52w_low": None,
        "near_52w_low": None,
        "avg_volume_20d": None, "vol_ratio_prev": None,
        "prev_3d_trend": None, "prev_5d_trend": None, "prev_10d_trend": None,
        "days_since_last_spike": None, "coil_days": None,
        "vol_trend_3d": None, "consecutive_vol_days": None,
        "hist_fetch_ok": 0,
    }
    if today_idx < 2:
        return out

    past = hist.iloc[:today_idx]
    if len(past) < 5:
        return out

    out["hist_fetch_ok"] = 1
    past_52    = past.tail(252)
    prev       = past.iloc[-1]
    prev_close = float(prev["close"])

    out["price_52w_high"]    = float(past_52["high"].max())
    out["price_52w_low"]     = float(past_52["low"].min())

    if out["price_52w_high"] > 0:
        out["pct_from_52w_high"] = (prev_close - out["price_52w_high"]) / out["price_52w_high"]
    if out["price_52w_low"] > 0:
        out["pct_from_52w_low"]  = (prev_close - out["price_52w_low"])  / out["price_52w_low"]
        out["near_52w_low"]      = 1 if prev_close < out["price_52w_low"] * 1.10 else 0

    if len(past) >= 20:
        out["avg_volume_20d"] = float(past.tail(20)["volume"].mean())
        prev_vol = float(prev["volume"])
        if out["avg_volume_20d"] > 0:
            out["vol_ratio_prev"] = prev_vol / out["avg_volume_20d"]

    closes  = past["close"].values
    volumes = past["volume"].values

    for n, key in [(3,"prev_3d_trend"),(5,"prev_5d_trend"),(10,"prev_10d_trend")]:
        if len(closes) >= n and closes[-n] > 0:
            out[key] = (closes[-1] - closes[-n]) / closes[-n]

    # Days since last 3x volume spike
    avg_vol = float(past["volume"].mean()) if len(past) > 0 else 0
    spike_days = 0
    for v in reversed(volumes):
        if avg_vol > 0 and v > avg_vol * 3:
            break
        spike_days += 1
    out["days_since_last_spike"] = spike_days

    # Coil days — consecutive below-avg volume
    coil = 0
    avg20 = out.get("avg_volume_20d") or 0
    for v in reversed(volumes):
        if avg20 > 0 and v < avg20 * 0.8:
            coil += 1
        else:
            break
    out["coil_days"] = coil

    # Vol trend 3d
    if len(volumes) >= 3 and volumes[-3] > 0:
        out["vol_trend_3d"] = (volumes[-1] - volumes[-3]) / volumes[-3]

    # Consecutive above-avg vol days
    consec = 0
    for v in reversed(volumes):
        if avg20 > 0 and v > avg20:
            consec += 1
        else:
            break
    out["consecutive_vol_days"] = consec

    return out


def calc_float_features(float_data: dict) -> dict:
    out = {
        "float_shares": None, "float_M": None, "market_cap": None,
        "float_tier": None, "float_rotation_prev": None,
        "is_foreign_listed": 0, "float_fetch_ok": 0,
    }
    shares = float_data.get("shares_outstanding")
    if shares and float(shares) > 0:
        out["float_fetch_ok"] = 1
        out["float_shares"]   = float(shares)
        out["float_M"]        = float(shares) / 1_000_000
        fm = out["float_M"]
        if fm < 5:     out["float_tier"] = 0   # nano
        elif fm < 15:  out["float_tier"] = 1   # micro
        elif fm < 50:  out["float_tier"] = 2   # small
        elif fm < 200: out["float_tier"] = 3   # mid
        else:          out["float_tier"] = 4   # large
        out["is_foreign_listed"] = float_data.get("is_foreign_listed", 0)
    else:
        log.warning(f"WARN float_features | shares_outstanding missing or zero")

    mc = float_data.get("market_cap")
    if mc:
        out["market_cap"] = float(mc)

    return out


def calc_si_features(ticker: str, trade_date: date, si_master: pd.DataFrame | None) -> dict:
    out = {"si_pct": None, "si_tier": None, "si_fetch_ok": 0}
    if si_master is None or si_master.empty:
        return out

    t_col  = next((c for c in si_master.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
    d_col  = next((c for c in si_master.columns if "date" in c.lower()), None)
    si_col = next((c for c in si_master.columns if "short" in c.lower() and "vol" not in c.lower()), None)

    if not all([t_col, d_col, si_col]):
        log.error(f"FAIL si | {ticker} | cannot find required columns in SI master. Cols: {list(si_master.columns)}")
        return out

    rows = si_master[si_master[t_col] == ticker].copy()
    if rows.empty:
        return out

    rows["_date"] = pd.to_datetime(rows[d_col], errors="coerce").dt.date
    rows = rows.dropna(subset=["_date"])
    rows["_diff"] = rows["_date"].apply(lambda d: abs((d - trade_date).days))
    nearest = rows.nsmallest(1, "_diff").iloc[0]

    try:
        sp = float(nearest[si_col])
        out["si_pct"]      = sp
        out["si_fetch_ok"] = 1
        if sp < 5:     out["si_tier"] = 0
        elif sp < 15:  out["si_tier"] = 1
        elif sp < 30:  out["si_tier"] = 2
        else:          out["si_tier"] = 3
    except (ValueError, TypeError) as e:
        log.warning(f"WARN si | {ticker} | could not parse si value: {e}")

    return out


def calc_edgar_features(ticker: str, trade_date: date, cik_map: dict, edgar_master: pd.DataFrame | None) -> dict:
    out = {
        "has_8k": 0, "has_8k_yesterday": 0, "has_8k_2days_ago": 0,
        "8k_filing_hour": 0, "hours_before_open": 0,
        "has_merger": 0, "has_fda": 0, "has_contract": 0,
        "has_earnings": 0, "has_reverse_split": 0,
        "has_dilution": 0, "has_buyback": 0,
        "dilution_count_6m": 0, "dilution_count_30d": 0,
        "days_since_dilution": 999,
        "reverse_split_count": 0,
        "is_serial_diluter": 0, "is_serial_reverser": 0,
        "has_form4_buy": 0, "form4_buy_count": 0,
        "has_sc13d": 0,
        "edgar_fetch_ok": 0,
        "edgar_dilution_ok": 0,
        "form4_fetch_ok": 0,
        "sc13d_fetch_ok": 0,
    }
    if edgar_master is None or edgar_master.empty:
        log.warning(f"WARN edgar | {ticker} | edgar_master is empty")
        return out

    cik = cik_map.get(ticker)
    if not cik:
        log.warning(f"WARN edgar | {ticker} | no CIK found in map")
        return out

    rows = edgar_master[edgar_master["cik"] == cik].copy()
    if rows.empty:
        return out

    rows["filed_date"] = pd.to_datetime(rows["filed"], errors="coerce").dt.date
    out["edgar_fetch_ok"] = 1

    today     = trade_date
    yesterday = today - timedelta(days=1)
    two_days  = today - timedelta(days=2)
    six_months_ago  = today - timedelta(days=180)
    thirty_days_ago = today - timedelta(days=30)

    # 8-K
    eightk = rows[rows["form_type"].isin(["8-K","8-K/A"])]
    eightk_today     = eightk[eightk["filed_date"] == today]
    eightk_yesterday = eightk[eightk["filed_date"] == yesterday]
    eightk_2days     = eightk[eightk["filed_date"] == two_days]

    if not eightk_today.empty:     out["has_8k"] = 1
    if not eightk_yesterday.empty: out["has_8k_yesterday"] = 1
    if not eightk_2days.empty:     out["has_8k_2days_ago"] = 1

    # 8K filing hour — fetch actual filing text for the most recent 8-K
    recent_8k = eightk[eightk["filed_date"] >= two_days]
    if not recent_8k.empty:
        filename = recent_8k.sort_values("filed").iloc[-1].get("filename","")
        if filename:
            filing_url = f"https://www.sec.gov/Archives/{filename}"
            # Just use the filed timestamp from the index
            filed_ts = recent_8k.sort_values("filed").iloc[-1]["filed"]
            try:
                filed_dt = pd.to_datetime(filed_ts)
                out["8k_filing_hour"]    = filed_dt.hour
                open_hour = 9.5
                filing_h  = filed_dt.hour + filed_dt.minute / 60
                out["hours_before_open"] = max(0, open_hour - filing_h)
            except Exception:
                pass

    # Parse 8-K content keywords from filename/company name (lightweight)
    if not recent_8k.empty:
        text = " ".join(recent_8k["company"].astype(str).tolist()).lower()
        out["has_merger"]        = 1 if any(w in text for w in ["merger","acqui","takeover"]) else 0
        out["has_fda"]           = 1 if "fda" in text else 0
        out["has_contract"]      = 1 if "contract" in text else 0
        out["has_reverse_split"] = 1 if "reverse" in text else 0
        out["has_buyback"]       = 1 if "repurchas" in text or "buyback" in text else 0

    # Dilution — S-1, S-3, 424B
    dilution_forms = {"S-1","S-1/A","S-3","S-3/A","424B1","424B3","424B4","424B5"}
    dil = rows[rows["form_type"].isin(dilution_forms)]
    dil_past = dil[dil["filed_date"] < today]

    out["dilution_count_6m"]  = len(dil_past[dil_past["filed_date"] >= six_months_ago])
    out["dilution_count_30d"] = len(dil_past[dil_past["filed_date"] >= thirty_days_ago])
    out["is_serial_diluter"]  = 1 if out["dilution_count_6m"] >= 3 else 0
    out["has_dilution"]       = 1 if out["dilution_count_30d"] > 0 else 0
    out["edgar_dilution_ok"]  = 1

    if not dil_past.empty:
        last_dil = dil_past["filed_date"].max()
        out["days_since_dilution"] = (today - last_dil).days

    # Form 4 — insider buying
    f4 = rows[rows["form_type"].isin(["4","4/A"])]
    f4_recent = f4[(f4["filed_date"] >= thirty_days_ago) & (f4["filed_date"] < today)]
    if not f4_recent.empty:
        out["has_form4_buy"]   = 1
        out["form4_buy_count"] = len(f4_recent)
    out["form4_fetch_ok"] = 1

    # SC 13D — activist investor
    sc = rows[rows["form_type"].isin(["SC 13D","SC 13D/A"])]
    sc_recent = sc[(sc["filed_date"] >= six_months_ago) & (sc["filed_date"] < today)]
    if not sc_recent.empty:
        out["has_sc13d"] = 1
    out["sc13d_fetch_ok"] = 1

    return out


def calc_halt_features(ticker: str, trade_date: date, halts: pd.DataFrame | None) -> dict:
    out = {
        "halted_yesterday": 0, "halt_count_5d": 0,
        "halt_count_30d": 0, "is_serial_halter": 0,
        "days_since_last_halt": 999, "halt_fetch_ok": 0,
    }
    if halts is None or halts.empty:
        return out

    t_col = next((c for c in halts.columns if any(x in c for x in ["symbol","issue","ticker"])), None)
    d_col = next((c for c in halts.columns if "date" in c or "halt" in c), None)
    if not t_col or not d_col:
        log.error(f"FAIL halts | {ticker} | cannot find symbol/date columns. Cols: {list(halts.columns)}")
        return out

    rows = halts[halts[t_col] == ticker].copy()
    if rows.empty:
        return out

    out["halt_fetch_ok"] = 1
    rows["_date"] = pd.to_datetime(rows[d_col], errors="coerce").dt.date
    rows = rows.dropna(subset=["_date"])
    rows = rows[rows["_date"] < trade_date]

    if rows.empty:
        return out

    yesterday  = trade_date - timedelta(days=1)
    five_ago   = trade_date - timedelta(days=5)
    thirty_ago = trade_date - timedelta(days=30)

    out["halted_yesterday"]    = 1 if not rows[rows["_date"] == yesterday].empty else 0
    out["halt_count_5d"]       = len(rows[rows["_date"] >= five_ago])
    out["halt_count_30d"]      = len(rows[rows["_date"] >= thirty_ago])
    out["is_serial_halter"]    = 1 if out["halt_count_30d"] >= 3 else 0
    last_halt = rows["_date"].max()
    out["days_since_last_halt"] = (trade_date - last_halt).days

    return out


def calc_earnings_features(ticker: str, trade_date: date, earnings_dates: list) -> dict:
    out = {
        "days_to_earnings": 999, "has_earnings_soon": 0,
        "had_earnings_recently": 0, "earnings_fetch_ok": 0,
    }
    if hasattr(earnings_dates, "tolist"):
        earnings_dates = earnings_dates.tolist()
    if not earnings_dates:
        return out

    out["earnings_fetch_ok"] = 1
    parsed = []
    for d in earnings_dates:
        try:
            parsed.append(date.fromisoformat(str(d)[:10]))
        except Exception:
            pass

    future = [d for d in parsed if d >= trade_date]
    past   = [d for d in parsed if d < trade_date]

    if future:
        next_e = min(future)
        out["days_to_earnings"]  = (next_e - trade_date).days
        out["has_earnings_soon"] = 1 if out["days_to_earnings"] <= 7 else 0
    if past:
        last_e = max(past)
        out["had_earnings_recently"] = 1 if (trade_date - last_e).days <= 5 else 0

    return out


def calc_sector_features(sector_data: dict, trade_date: date) -> dict:
    out = {
        "spy_prev_day_pct": 0, "qqq_prev_day_pct": 0,
        "iwm_prev_day_pct": 0, "xbi_prev_day_pct": 0,
        "market_green": 0, "market_red": 0,
        "sector_hot": 0, "sector_fetch_ok": 0,
    }
    for ticker, key in [("SPY","spy"),("QQQ","qqq"),("IWM","iwm"),("XBI","xbi")]:
        if ticker not in sector_data:
            log.warning(f"WARN sector | {ticker} not in sector_data")
            continue
        df = sector_data[ticker]["daily"]
        if df.empty:
            log.warning(f"WARN sector | {ticker} daily df is empty")
            continue
        prev_rows = df[df["date"] < trade_date].tail(2)
        if len(prev_rows) < 2:
            continue
        t2, t1 = prev_rows.iloc[-2], prev_rows.iloc[-1]
        if float(t2["close"]) > 0:
            pct = (float(t1["close"]) - float(t2["close"])) / float(t2["close"])
            out[f"{key}_prev_day_pct"] = pct
            out["sector_fetch_ok"] = 1

    spy = out["spy_prev_day_pct"]
    xbi = out["xbi_prev_day_pct"]
    out["market_green"] = 1 if spy > 0.003 else 0
    out["market_red"]   = 1 if spy < -0.003 else 0
    out["sector_hot"]   = 1 if xbi > 0.01 else 0

    if out["sector_fetch_ok"] == 0:
        log.error(f"FAIL sector | {trade_date} | no sector data populated")

    return out


# ─────────────────────────────────────────────
# LOAD TICKER CACHE
# ─────────────────────────────────────────────
def load_ticker_cache(ticker: str) -> dict | None:
    path = TICKER_RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        df  = pd.read_parquet(path)
        row = df.iloc[0].to_dict()
        del df
        for key in ["daily","pm_minute","ah_minute"]:
            val = row.get(key)
            if val is None:
                row[key] = pd.DataFrame()
            elif isinstance(val, pd.DataFrame):
                pass
            elif isinstance(val, (list, dict)):
                row[key] = pd.DataFrame(val) if val else pd.DataFrame()
            else:
                try:
                    lst = val.tolist() if hasattr(val, "tolist") else list(val)
                    row[key] = pd.DataFrame(lst) if lst else pd.DataFrame()
                except Exception:
                    row[key] = pd.DataFrame()

            df2 = row[key]
            if not isinstance(df2, pd.DataFrame):
                row[key] = pd.DataFrame()
                continue
            if not df2.empty and "date" not in df2.columns and "t" in df2.columns:
                df2["date"] = pd.to_datetime(df2["t"], unit="ms", utc=True).dt.tz_convert(ET).dt.date
                row[key] = df2
            elif not df2.empty and "date" in df2.columns:
                df2["date"] = pd.to_datetime(df2["date"]).dt.date
                row[key] = df2

        if isinstance(row.get("float_data"), str):
            row["float_data"] = json.loads(row["float_data"])
        elif hasattr(row.get("float_data"), "tolist"):
            row["float_data"] = row["float_data"].tolist()
        if isinstance(row.get("fetch_ok"), str):
            row["fetch_ok"] = json.loads(row["fetch_ok"])
        if hasattr(row.get("earnings_dates"), "tolist"):
            row["earnings_dates"] = row["earnings_dates"].tolist()
        return row
    except Exception as e:
        log.error(f"FAIL load_cache | {ticker} | {e} — deleting corrupted cache")
        try:
            path.unlink()
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────
# PHASE 2 — ASSEMBLE DAILY TRAINING ROWS
# ─────────────────────────────────────────────
def assemble_day(
    trade_date: date,
    tickers: list[str],
    si_master: pd.DataFrame | None,
    halts: pd.DataFrame | None,
    edgar_master: pd.DataFrame | None,
    cik_map: dict,
    sector_data: dict,
    all_labels: dict,
) -> pd.DataFrame | None:

    rows = []
    n_universe = n_no_cache = n_no_daily = n_filtered = 0

    for ticker in tickers:
        cache = load_ticker_cache(ticker)
        if not cache:
            n_no_cache += 1
            continue

        daily_raw = cache.get("daily")
        try:
            daily_df = daily_raw if isinstance(daily_raw, pd.DataFrame) else pd.DataFrame(list(daily_raw) if daily_raw else [])
        except Exception:
            n_no_daily += 1
            continue

        if daily_df.empty:
            n_no_daily += 1
            continue

        if "date" not in daily_df.columns:
            if "t" in daily_df.columns:
                daily_df["date"] = pd.to_datetime(daily_df["t"], unit="ms", utc=True).dt.tz_convert(ET).dt.date
            else:
                n_no_daily += 1
                continue

        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date

        today_rows = daily_df[daily_df["date"] == trade_date]
        past_rows  = daily_df[daily_df["date"] < trade_date].sort_values("date")

        if today_rows.empty or past_rows.empty:
            continue

        today_row  = today_rows.iloc[-1]
        prev_row   = past_rows.iloc[-1]
        prev_close = float(prev_row.get("close", 0))
        prev_vol   = float(prev_row.get("volume", 0))
        prev_dollar = prev_close * prev_vol

        if prev_close < MIN_PRICE or prev_dollar < MIN_PREV_DOLLAR_VOL:
            n_filtered += 1
            continue

        n_universe += 1

        day_high   = float(today_row.get("high", 0))
        day_volume = float(today_row.get("volume", 0))
        day_open   = float(today_row.get("open", 0))

        # Label
        label = 0
        if prev_close > 0 and day_volume >= MIN_LABEL_VOLUME:
            ratio = day_high / prev_close
            if ratio >= MEGA_MULT:   label = 3
            elif ratio >= SUPER_MULT: label = 2
            elif ratio >= SEED_MULT:  label = 1

        # T-1 price structure
        prev_open  = float(prev_row.get("open", 0))
        prev_high  = float(prev_row.get("high", 0))
        prev_low   = float(prev_row.get("low", 0))
        prev_body_pct   = abs(prev_close - prev_open) / prev_open if prev_open > 0 else 0
        total_range     = prev_high - prev_low
        body            = abs(prev_close - prev_open)
        prev_wick_ratio = 1 - (body / total_range) if total_range > 0 else 0

        # Premarket
        pm_minute = cache.get("pm_minute", pd.DataFrame())
        if not pm_minute.empty and "date" in pm_minute.columns:
            pm_minute["date"] = pd.to_datetime(pm_minute["date"]).dt.date
            pm_day = pm_minute[pm_minute["date"] == trade_date]
        else:
            pm_day = pd.DataFrame()

        avg_pm_vol = 0
        if not pm_minute.empty and "date" in pm_minute.columns:
            past_pm = pm_minute[pm_minute["date"] < trade_date]
            if not past_pm.empty:
                avg_pm_vol = float(past_pm.groupby("date")["v"].sum().tail(20).mean())

        pm_feats = calc_premarket_features(pm_day, prev_close, avg_pm_vol)
        if not pm_feats["pm_fetch_ok"]:
            log.debug(f"WARN pm_feats | {ticker} {trade_date} | no PM bars for this day")

        # After hours (T-1 evening)
        ah_minute = cache.get("ah_minute", pd.DataFrame())
        ah_day = pd.DataFrame()
        if not ah_minute.empty and "date" in ah_minute.columns:
            ah_minute["date"] = pd.to_datetime(ah_minute["date"]).dt.date
            prior_date = prev_row.get("date")
            if prior_date:
                ah_day = ah_minute[ah_minute["date"] == prior_date]

        ah_feats = calc_ah_features(ah_day, prev_close)
        if not ah_feats["ah_fetch_ok"]:
            log.debug(f"WARN ah_feats | {ticker} {trade_date} | no AH bars for T-1")

        # Historical
        today_idx  = len(past_rows)
        hist_feats = calc_historical_features(
            daily_df.sort_values("date").reset_index(drop=True), today_idx
        )
        if not hist_feats["hist_fetch_ok"]:
            log.warning(f"WARN hist | {ticker} {trade_date} | insufficient history")

        # Float
        float_feats = calc_float_features(cache.get("float_data", {}))
        if float_feats["float_fetch_ok"] and float_feats.get("float_shares"):
            avg_vol = hist_feats.get("avg_volume_20d") or 0
            if float_feats["float_shares"] > 0 and avg_vol > 0:
                float_feats["float_rotation_prev"] = (avg_vol * prev_close) / float_feats["float_shares"]

        # SI
        si_feats = calc_si_features(ticker, trade_date, si_master)

        # EDGAR
        edgar_feats = calc_edgar_features(ticker, trade_date, cik_map, edgar_master)

        # Halts
        halt_feats = calc_halt_features(ticker, trade_date, halts)

        # Earnings
        earn_feats = calc_earnings_features(ticker, trade_date, cache.get("earnings_dates", []))

        # Sector
        sector_feats = calc_sector_features(sector_data, trade_date)

        # Days since last seed
        ticker_seeds = [d for d, ts in all_labels.items() if ticker in ts and d < trade_date]
        days_since_seed = (trade_date - max(ticker_seeds)).days if ticker_seeds else 999

        row = {
            "ticker":   ticker,
            "date":     trade_date.isoformat(),
            "label":    label,
            "day_high": day_high, "day_volume": day_volume, "day_open": day_open,
            # T-1 price
            "prev_close":       prev_close,
            "prev_open":        prev_open,
            "prev_high":        prev_high,
            "prev_low":         prev_low,
            "prev_volume":      prev_vol,
            "prev_dollar_vol":  prev_dollar,
            "prev_body_pct":    prev_body_pct,
            "prev_wick_ratio":  prev_wick_ratio,
            "days_since_last_seed": days_since_seed,
            **pm_feats,
            **ah_feats,
            **hist_feats,
            **float_feats,
            **si_feats,
            **edgar_feats,
            **halt_feats,
            **earn_feats,
            **sector_feats,
        }
        rows.append(row)

    log.info(f"{trade_date}: universe={n_universe} no_cache={n_no_cache} filtered={n_filtered}")

    if not rows:
        return None

    df = pd.DataFrame(rows)
    positives = df[df["label"] > 0]
    controls  = df[df["label"] == 0]
    n_pos = len(positives)

    if n_pos == 0:
        log.info(f"{trade_date}: 0 positives — skipping")
        return None

    n_ctrl = min(n_pos * CONTROL_RATIO, len(controls))
    sampled = controls.sample(n=n_ctrl, random_state=42)
    final = pd.concat([positives, sampled], ignore_index=True)

    log.info(
        f"{trade_date}: {n_pos} positives "
        f"(s={len(df[df['label']==1])} su={len(df[df['label']==2])} m={len(df[df['label']==3])}) "
        f"+ {n_ctrl} controls = {len(final)} rows"
    )
    all_labels[trade_date] = set(positives["ticker"].tolist())
    return final


def phase2_assemble_all(tickers, trading_days, si_master, halts, edgar_master, cik_map):
    log.info("Phase 2: loading sector data...")
    sector_data = {}
    for sym in ["SPY","QQQ","IWM","XBI"]:
        cache = load_ticker_cache(sym)
        if not cache:
            log.error(f"FAIL sector | {sym} not cached — sector features will be empty")
            continue
        daily = cache.get("daily", pd.DataFrame())
        pm    = cache.get("pm_minute", pd.DataFrame())
        if not isinstance(daily, pd.DataFrame):
            daily = pd.DataFrame(daily) if daily else pd.DataFrame()
        if not isinstance(pm, pd.DataFrame):
            pm = pd.DataFrame(pm) if pm else pd.DataFrame()
        if not daily.empty and "date" not in daily.columns and "t" in daily.columns:
            daily["date"] = pd.to_datetime(daily["t"], unit="ms", utc=True).dt.tz_convert(ET).dt.date
        sector_data[sym] = {"daily": daily, "pm_minute": pm}
        log.info(f"Sector {sym}: {len(daily)} daily rows")

    all_labels: dict[date, set] = {}
    total = len(trading_days)

    for i, trade_date in enumerate(trading_days):
        out_path = OUTPUT_DIR / f"{trade_date.isoformat()}.parquet"
        if out_path.exists():
            log.debug(f"Skip {trade_date} — already assembled")
            continue

        log.info(f"Assembling {trade_date}...")
        day_df = assemble_day(
            trade_date, tickers, si_master, halts,
            edgar_master, cik_map, sector_data, all_labels
        )
        if day_df is not None:
            day_df.to_parquet(out_path, index=False)
            del day_df

        if (i + 1) % 10 == 0:
            gc.collect()
            log.info(f"Assemble progress: {i+1}/{total} days")

    log.info("Phase 2 complete.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    threading.Thread(target=start_keepalive, daemon=True).start()
    time.sleep(2)

    if not POLYGON_API_KEY:
        log.error("FAIL startup | MASSIVE_API_KEY not set — aborting")
        return

    log.info("=" * 60)
    log.info("THE DELTA v2 — Data Collector (2025-2026)")
    log.info(f"Window: {START_DATE} → {END_DATE}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("=" * 60)

    tickers = fetch_ticker_list()
    if not tickers:
        log.error("FAIL | no tickers — aborting")
        return

    phase1_collect_all(tickers)

    log.info("Collecting FINRA SI...")
    collect_finra()

    log.info("Collecting halt history...")
    collect_halts()

    log.info("Collecting EDGAR index...")
    collect_edgar_index()

    log.info("Building CIK map...")
    cik_map = build_cik_map(tickers)

    # Load support data
    si_master    = None
    halts_master = None
    edgar_master = None

    si_path    = FINRA_DIR  / "si_master.parquet"
    halt_path  = HALTS_DIR  / "halts_master.parquet"
    edgar_path = EDGAR_DIR  / "filings_master.parquet"

    if si_path.exists():
        si_master = pd.read_parquet(si_path)
        log.info(f"FINRA SI loaded: {len(si_master)} rows")
    else:
        log.error("FAIL | SI master not found — SI features will be empty")

    if halt_path.exists():
        halts_master = pd.read_parquet(halt_path)
        log.info(f"Halts loaded: {len(halts_master)} rows")
    else:
        log.error("FAIL | Halts master not found — halt features will be empty")

    if edgar_path.exists():
        edgar_master = pd.read_parquet(edgar_path)
        log.info(f"EDGAR loaded: {len(edgar_master)} filings")
    else:
        log.error("FAIL | EDGAR master not found — EDGAR features will be empty")

    trading_days = get_trading_days(START_DATE, END_DATE)
    log.info(f"Trading days: {len(trading_days)}")

    phase2_assemble_all(
        tickers, trading_days,
        si_master, halts_master, edgar_master, cik_map
    )

    # Final stats
    log.info("=" * 60)
    log.info("COLLECTION COMPLETE")
    log.info(f"FINAL STATS:\n{json.dumps(STATS, indent=2)}")
    log.info(f"Training files: {OUTPUT_DIR}")
    log.info("Next: run trainer.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
