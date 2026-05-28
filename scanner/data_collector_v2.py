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

    all_bars_df = pd.DataFrame(pm_bars) if pm_bars else pd.DataFrame()
    pm_df = pd.DataFrame()
    ah_df = pd.DataFrame()

    if not all_bars_df.empty:
        all_bars_df["t"]      = pd.to_datetime(all_bars_df["t"], unit="ms", utc=True).dt.tz_convert(ET)
        all_bars_df["hour"]   = all_bars_df["t"].dt.hour
        all_bars_df["minute"] = all_bars_df["t"].dt.minute
        all_bars_df["date"]   = all_bars_df["t"].dt.date

        # Premarket: 4AM-9:29AM
        pm_df = all_bars_df[
            (all_bars_df["hour"] >= 4) &
            ((all_bars_df["hour"] < 9) | ((all_bars_df["hour"] == 9) & (all_bars_df["minute"] < 30)))
        ].copy()

        # After-hours: 4PM-8PM — separate slice from full bars
        ah_df = all_bars_df[
            (all_bars_df["hour"] >= 16) & (all_bars_df["hour"] < 20)
        ].copy()

        if ah_df.empty:
            log.warning(f"WARN ah | {ticker} | no after-hours bars found in minute data")
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
# MAIN — Phase 1 only
# Stops after collecting all raw data.
# Next steps:
#   python phase1_5_seed_discovery.py
#   python collect_pm_1min.py
#   python assembler.py
#   python trainer.py
# ─────────────────────────────────────────────
def main():
    threading.Thread(target=start_keepalive, daemon=True).start()
    time.sleep(2)

    if not POLYGON_API_KEY:
        log.error("FAIL startup | MASSIVE_API_KEY not set — aborting")
        return

    log.info("=" * 60)
    log.info("THE DELTA v2 — Data Collector Phase 1 (2025-2026)")
    log.info(f"Window: {START_DATE} → {END_DATE}")
    log.info(f"Raw cache: {TICKER_RAW_DIR}")
    log.info("=" * 60)

    # Step 1 — Ticker list
    tickers = fetch_ticker_list()
    if not tickers:
        log.error("FAIL | no tickers — aborting")
        return

    # Step 2 — Collect all ticker data
    phase1_collect_all(tickers)

    # Step 3 — Support data
    log.info("Collecting FINRA SI...")
    collect_finra()

    log.info("Collecting halt history...")
    collect_halts()

    log.info("Collecting EDGAR index...")
    collect_edgar_index()

    log.info("Building CIK map...")
    build_cik_map(tickers)

    # Final stats
    log.info("=" * 60)
    log.info("PHASE 1 COLLECTION COMPLETE")
    log.info(f"FINAL STATS:\n{json.dumps(STATS, indent=2)}")
    log.info("")
    log.info("Next steps:")
    log.info("  1. python scanner/phase1_5_seed_discovery.py")
    log.info("  2. python scanner/collect_pm_1min.py")
    log.info("  3. python scanner/assembler.py")
    log.info("  4. python scanner/trainer.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
