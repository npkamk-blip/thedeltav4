"""
THE DELTA v2 — data_collector_v2.py
=====================================
Predictive historical data collector.
Golden rule: "Would I have known this BEFORE the event?"

Phase 1 — Collect: pull all history per ticker in bulk (fast, API-efficient)
Phase 2 — Assemble: slice by day, engineer features, label, sample controls, save parquet

Collection window: Jan 1 2024 → May 23 2026
Output: /app/data/training_data_v2/YYYY-MM-DD.parquet
Raw cache: /app/data/raw/tickers/<TICKER>.parquet
           /app/data/raw/finra/
           /app/data/raw/halts/
           /app/data/raw/edgar/
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
import json
import random
import gc
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ──────────────────────────────────────────────────────────────
# CONFIG — all values read from environment or defaults
# ──────────────────────────────────────────────────────────────
POLYGON_API_KEY     = os.environ.get("MASSIVE_API_KEY", "")
DATA_ROOT           = Path(os.environ.get("DATA_DIR", "/app/data"))
RAW_DIR             = DATA_ROOT / "raw"
TICKER_RAW_DIR      = RAW_DIR / "tickers"
FINRA_DIR           = RAW_DIR / "finra"
HALTS_DIR           = RAW_DIR / "halts"
EDGAR_DIR           = RAW_DIR / "edgar"
OUTPUT_DIR          = DATA_ROOT / "training_data_v2"
LOG_DIR             = DATA_ROOT / "logs"

START_DATE          = date(2024, 1, 1)
END_DATE            = date(2026, 5, 23)

# Universe filters
MIN_PRICE           = 0.20
MIN_PREV_DOLLAR_VOL = 25_000

# Label thresholds — gain above prev_close
SEED_MULT           = 2.00   # 100% gain  → day_high >= prev_close * 2.00
SUPER_MULT          = 6.00   # 500% gain  → day_high >= prev_close * 6.00
MEGA_MULT           = 11.00  # 1000% gain → day_high >= prev_close * 11.00
MIN_LABEL_VOLUME    = 25_000

# Control sampling
CONTROL_RATIO       = 2      # 2 controls per positive

# Polygon rate limit (calls per minute) — adjust if you hit 429s
CALLS_PER_MINUTE    = 300
CALL_INTERVAL       = 60.0 / CALLS_PER_MINUTE

# EDGAR
EDGAR_USER_AGENT    = "NPKNOB@gmail.com"
EDGAR_RATE_INTERVAL = 0.11   # ~10 req/sec max

ET = ZoneInfo("America/New_York")
POLYGON_BASE = "https://api.polygon.io"
EDGAR_BASE   = "https://data.sec.gov"

# Sector tickers we always collect
SECTOR_TICKERS = ["SPY", "QQQ", "IWM", "XBI"]


# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
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
log = logging.getLogger("delta_v2")


# ──────────────────────────────────────────────────────────────
# POLYGON CLIENT — rate-limited, retry, paginated
# ──────────────────────────────────────────────────────────────
class PolygonClient:
    def __init__(self):
        self.interval = CALL_INTERVAL
        self._last    = 0.0
        self.session  = requests.Session()
        self.session.params = {"apiKey": POLYGON_API_KEY}  # type: ignore

    def _wait(self):
        elapsed = time.time() - self._last
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last = time.time()

    def get(self, path: str, params: dict = None, retries: int = 2) -> dict | None:
        url = POLYGON_BASE + path
        for attempt in range(retries):
            self._wait()
            try:
                r = self.session.get(url, params=params or {}, timeout=15)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    wait = 30 * (attempt + 1)
                    log.warning(f"429 rate limit — sleeping {wait}s")
                    time.sleep(wait)
                elif r.status_code == 403:
                    log.error(f"403 forbidden: {url}")
                    return None
                elif r.status_code == 404:
                    return None
                else:
                    log.warning(f"HTTP {r.status_code} for {url} attempt {attempt+1}")
                    time.sleep(3)
            except Exception as e:
                log.warning(f"Request error (attempt {attempt+1}): {e}")
                time.sleep(2)
        return None

    def paginate(self, path: str, params: dict = None) -> list:
        """Follow Polygon next_url pagination, collect all results."""
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

    def aggs(self, ticker: str, multiplier: int, timespan: str,
             from_: str, to: str, adjusted: bool = True) -> list:
        """Pull aggregate bars for a ticker over a date range."""
        params = {
            "adjusted": str(adjusted).lower(),
            "sort":     "asc",
            "limit":    50000,
        }
        return self.paginate(
            f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to}",
            params
        )


poly = PolygonClient()


# ──────────────────────────────────────────────────────────────
# EDGAR CLIENT — rate-limited
# ──────────────────────────────────────────────────────────────
class EdgarClient:
    def __init__(self):
        self._last   = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent":      EDGAR_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })

    def _wait(self):
        elapsed = time.time() - self._last
        if elapsed < EDGAR_RATE_INTERVAL:
            time.sleep(EDGAR_RATE_INTERVAL - elapsed)
        self._last = time.time()

    def get(self, url: str, retries: int = 3) -> dict | None:
        for attempt in range(retries):
            self._wait()
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    time.sleep(30)
                else:
                    time.sleep(2)
            except Exception as e:
                log.warning(f"EDGAR error (attempt {attempt+1}): {e}")
                time.sleep(2)
        return None

    def get_csv(self, url: str) -> pd.DataFrame | None:
        self._wait()
        try:
            r = self.session.get(url, timeout=60)
            if r.status_code == 200:
                from io import StringIO
                return pd.read_csv(StringIO(r.text), sep="|", on_bad_lines="skip", low_memory=False)
        except Exception as e:
            log.warning(f"EDGAR CSV error: {e}")
        return None


edgar = EdgarClient()


# ──────────────────────────────────────────────────────────────
# KEEPALIVE — prevents Render from killing the process
# ──────────────────────────────────────────────────────────────
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"alive")
    def log_message(self, format, *args):
        pass  # silence HTTP logs

def start_keepalive(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log.info(f"Keepalive server running on port {port}")


# ──────────────────────────────────────────────────────────────
# TRADING CALENDAR
# ──────────────────────────────────────────────────────────────
KNOWN_HOLIDAYS = {
    # 2024
    date(2024,1,1), date(2024,1,15), date(2024,2,19), date(2024,3,29),
    date(2024,5,27), date(2024,6,19), date(2024,7,4), date(2024,9,2),
    date(2024,11,28), date(2024,12,25),
    # 2025
    date(2025,1,1), date(2025,1,9), date(2025,1,20), date(2025,2,17),
    date(2025,4,18), date(2025,5,26), date(2025,6,19), date(2025,7,4),
    date(2025,9,1), date(2025,11,27), date(2025,12,25),
    # 2026
    date(2026,1,1), date(2026,1,19), date(2026,2,16), date(2026,4,3),
    date(2026,5,25),
}

def get_trading_days(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5 and cur not in KNOWN_HOLIDAYS:
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ──────────────────────────────────────────────────────────────
# PHASE 1A — FETCH MASTER TICKER LIST
# ──────────────────────────────────────────────────────────────
def fetch_ticker_list() -> list[str]:
    """
    Pull all active common stocks on NASDAQ, NYSE, AMEX.
    No OTC, no ETFs, no preferreds, no warrants.
    Returns sorted list of tickers.
    """
    cache_path = RAW_DIR / "ticker_list.json"
    if cache_path.exists():
        log.info("Loading cached ticker list...")
        with open(cache_path) as f:
            tickers = json.load(f)
        log.info(f"Loaded {len(tickers)} tickers from cache.")
        return tickers

    log.info("Fetching master ticker list from Polygon...")
    results = poly.paginate(
        "/v3/reference/tickers",
        {
            "market":   "stocks",
            "type":     "CS",        # Common Stock only
            "active":   "true",
            "limit":    1000,
        }
    )

    tickers = []
    for r in results:
        exch = r.get("primary_exchange", "")
        # XNAS=NASDAQ, XNYS=NYSE, XASE=AMEX
        if exch in ("XNAS", "XNYS", "XASE"):
            tickers.append(r["ticker"])

    tickers = sorted(set(tickers))
    log.info(f"Found {len(tickers)} valid tickers.")

    with open(cache_path, "w") as f:
        json.dump(tickers, f)

    return tickers


# ──────────────────────────────────────────────────────────────
# PHASE 1B — BULK COLLECT PER TICKER
# ──────────────────────────────────────────────────────────────
def collect_ticker(ticker: str) -> bool:
    """
    Pull all bars for a single ticker covering our full date range.
    Saves to raw cache. Returns True if successful.
    Skips if cache already exists.
    """
    cache_path = TICKER_RAW_DIR / f"{ticker}.parquet"
    if cache_path.exists():
        # Verify file is not corrupted before skipping
        try:
            pd.read_parquet(cache_path, columns=["ticker"])
            return True  # valid cache
        except Exception:
            log.warning(f"{ticker}: corrupted cache — re-collecting")
            cache_path.unlink()

    from_str = START_DATE.strftime("%Y-%m-%d")
    to_str   = END_DATE.strftime("%Y-%m-%d")

    fetch_ok = {
        "hist_fetch_ok":    False,
        "pm_fetch_ok":      False,
        "ah_fetch_ok":      False,
        "float_fetch_ok":   False,
        "earnings_fetch_ok":False,
    }

    # ── Daily bars (T-1 structure + labeling) ──────────────────
    daily_bars = poly.aggs(ticker, 1, "day", from_str, to_str)
    if not daily_bars:
        log.warning(f"{ticker}: no daily bars — skipping")
        return False
    fetch_ok["hist_fetch_ok"] = True

    daily_df = pd.DataFrame(daily_bars)
    daily_df["t"] = pd.to_datetime(daily_df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    daily_df["date"] = daily_df["t"].dt.date
    daily_df = daily_df.rename(columns={
        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap"
    })

    # ── Premarket bars 4:00AM - 9:30AM ────────────────────────
    # Polygon extended hours: use "minute" bars, filter by time
    pm_bars = []
    # Pull 5-minute bars in quarterly chunks — small enough to avoid timeouts
    # 5-min resolution is sufficient for premarket feature engineering
    # Seeds get 1-min resolution in a separate targeted pass later
    for chunk_from, chunk_to in [
        ("2024-01-01", "2024-06-30"),
        ("2024-07-01", "2024-12-31"),
        ("2025-01-01", "2025-06-30"),
        ("2025-07-01", "2025-12-31"),
        ("2026-01-01", "2026-05-23"),
    ]:
        chunk = poly.aggs(ticker, 5, "minute", chunk_from, chunk_to, adjusted=False)
        if chunk:
            pm_bars.extend(chunk)
    if pm_bars:
        fetch_ok["pm_fetch_ok"] = True
    pm_df = pd.DataFrame(pm_bars) if pm_bars else pd.DataFrame()
    if not pm_df.empty:
        pm_df["t"] = pd.to_datetime(pm_df["t"], unit="ms", utc=True).dt.tz_convert(ET)
        pm_df["hour"]   = pm_df["t"].dt.hour
        pm_df["minute"] = pm_df["t"].dt.minute
        pm_df["date"]   = pm_df["t"].dt.date
        # Premarket: 4:00 AM to 9:29 AM ET
        pm_df = pm_df[
            (pm_df["hour"] >= 4) &
            ((pm_df["hour"] < 9) | ((pm_df["hour"] == 9) & (pm_df["minute"] < 30)))
        ]
        # After-hours prior day: 4:00 PM to 8:00 PM ET
        ah_df = pm_df[
            (pm_df["hour"] >= 16) & (pm_df["hour"] < 20)
        ].copy()
        fetch_ok["ah_fetch_ok"] = not ah_df.empty
    else:
        ah_df = pd.DataFrame()

    float_data = {}
    detail = poly.get(f"/v3/reference/tickers/{ticker}")
    if detail and "results" in detail:
        r = detail["results"]
        locale     = r.get("locale", "us").lower()
        addr       = r.get("address", {})
        hq_country = addr.get("country", "us").lower() if isinstance(addr, dict) else "us"
        is_foreign = 1 if (locale != "us" or hq_country not in ("us", "usa", "united states", "")) else 0
        float_data = {
            "shares_outstanding": r.get("share_class_shares_outstanding"),
            "market_cap":         r.get("market_cap"),
            "name":               r.get("name", ""),
            "sic_code":           r.get("sic_code", ""),
            "is_foreign_listed":  is_foreign,
        }
        fetch_ok["float_fetch_ok"] = True

    # ── Earnings dates ─────────────────────────────────────────
    earnings_dates = []
    earn_resp = poly.paginate(
        f"/vX/reference/financials",
        {"ticker": ticker, "limit": 100, "sort": "filing_date"}
    )
    if earn_resp:
        for e in earn_resp:
            fd = e.get("filing_date")
            if fd:
                earnings_dates.append(fd)
        fetch_ok["earnings_fetch_ok"] = True

    # ── Save raw cache ─────────────────────────────────────────
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
    return True


def phase1_collect_all_tickers(tickers: list[str]):
    """Bulk collect all tickers. Skips already-cached ones."""
    total   = len(tickers)
    done    = 0
    failed  = []

    log.info(f"Phase 1: collecting {total} tickers...")

    for i, ticker in enumerate(tickers):
        cache_path = TICKER_RAW_DIR / f"{ticker}.parquet"
        if cache_path.exists():
            done += 1
            continue

        ok = collect_ticker(ticker)
        if ok:
            done += 1
        else:
            failed.append(ticker)

        if (i + 1) % 100 == 0:
            log.info(f"Progress: {i+1}/{total} tickers | done={done} | failed={len(failed)}")

    # Also collect sector tickers
    for st in SECTOR_TICKERS:
        collect_ticker(st)

    # Save failed list for retry
    if failed:
        failed_path = RAW_DIR / "failed_tickers.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f)
        log.warning(f"Phase 1 complete. {len(failed)} tickers failed — saved to {failed_path}")
    else:
        log.info(f"Phase 1 complete. All {total} tickers collected.")


# ──────────────────────────────────────────────────────────────
# PHASE 1C — COLLECT FINRA SHORT INTEREST
# ──────────────────────────────────────────────────────────────
def collect_finra_short_interest():
    """
    Download FINRA short interest files for 2024-2026.
    FINRA publishes bi-monthly: first and middle of each month.
    Files stored in /app/data/raw/finra/
    """
    cache_path = FINRA_DIR / "si_master.parquet"
    if cache_path.exists():
        log.info("FINRA short interest already cached.")
        return

    log.info("Downloading FINRA short interest data...")

    all_rows = []
    # Pull daily FINRA short sale volume files
    # Confirmed working URL: https://cdn.finra.org/equity/regsho/daily/CNMSshvol20240102.txt
    cur = date(2024, 1, 2)
    hits = 0
    while cur <= END_DATE:
        date_str = cur.strftime("%Y%m%d")
        # Cache each file separately so restarts dont lose progress
        day_cache = FINRA_DIR / f"finra_{date_str}.parquet"
        if day_cache.exists():
            all_rows.append(pd.read_parquet(day_cache))
            hits += 1
        else:
            url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date_str}.txt"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200 and len(r.text) > 100:
                    from io import StringIO
                    df = pd.read_csv(StringIO(r.text), sep="|", on_bad_lines="skip")
                    if len(df) > 0:
                        df["si_date"] = cur.isoformat()
                        df.to_parquet(day_cache, index=False)
                        all_rows.append(df)
                        hits += 1
                        if hits % 50 == 0:
                            log.info(f"FINRA: {hits} files downloaded, latest {cur}")
            except Exception as e:
                log.debug(f"FINRA error for {cur}: {e}")
            time.sleep(0.1)

        cur += timedelta(days=1)
        while cur.weekday() >= 5:
            cur += timedelta(days=1)

    if all_rows:
        # Concat in chunks to keep memory manageable
        master = pd.concat(all_rows, ignore_index=True)
        del all_rows
        master.to_parquet(cache_path, index=False)
        log.info(f"FINRA: saved {len(master)} rows ({hits} daily files)")
        del master
        gc.collect()
    else:
        log.warning("FINRA: no data collected — si features will be null")


# ──────────────────────────────────────────────────────────────
# PHASE 1D — COLLECT NASDAQ HALT HISTORY
# ──────────────────────────────────────────────────────────────
def collect_halt_history():
    """
    Download NASDAQ halt history archives.
    nasdaqtrader.com provides yearly halt files.
    """
    cache_path = HALTS_DIR / "halts_master.parquet"
    if cache_path.exists():
        log.info("Halt history already cached.")
        return

    log.info("Downloading NASDAQ halt history...")

    all_rows = []
    for year in [2024, 2025, 2026]:
        url = f"https://www.nasdaqtrader.com/dynamic/symdir/halts/{year}halts.txt"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text), sep="|", on_bad_lines="skip")
                all_rows.append(df)
                log.info(f"Halts: got {year} ({len(df)} rows)")
            else:
                # Also try current file
                url2 = "https://www.nasdaqtrader.com/dynamic/symdir/halts.txt"
                r2 = requests.get(url2, timeout=30)
                if r2.status_code == 200:
                    from io import StringIO
                    df = pd.read_csv(StringIO(r2.text), sep="|", on_bad_lines="skip")
                    all_rows.append(df)
                    log.info(f"Halts: got current file ({len(df)} rows)")
        except Exception as e:
            log.warning(f"Halt download error for {year}: {e}")
        time.sleep(1)

    if all_rows:
        master = pd.concat(all_rows, ignore_index=True)
        # Standardize column names
        master.columns = [c.strip().lower().replace(" ", "_") for c in master.columns]
        master.to_parquet(cache_path, index=False)
        log.info(f"Halts: saved {len(master)} rows.")
    else:
        log.warning("Halts: no data collected — halt features will be null")


# ──────────────────────────────────────────────────────────────
# PHASE 1E — COLLECT EDGAR 8-K INDEX
# ──────────────────────────────────────────────────────────────
def collect_edgar_index():
    """
    Download SEC EDGAR quarterly filing indexes.
    Captures 8-K, S-1, S-3, 424B*, SC 13D, Form 4 existence.
    Existence only — no full text fetch.
    """
    cache_path = EDGAR_DIR / "filings_master.parquet"
    if cache_path.exists():
        log.info("EDGAR index already cached.")
        return

    log.info("Downloading EDGAR quarterly indexes...")

    quarters = []
    for year in [2024, 2025, 2026]:
        for q in [1, 2, 3, 4]:
            qdate = date(year, q * 3, 1)
            if qdate > END_DATE:
                break
            quarters.append((year, q))

    relevant = {"8-K", "8-K/A", "S-1", "S-1/A", "S-3", "S-3/A",
                "424B1", "424B3", "424B4", "424B5",
                "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
                "4", "4/A"}

    all_rows = []
    for year, q in quarters:
        # Cache each quarter separately so restarts dont lose progress
        q_cache = EDGAR_DIR / f"edgar_{year}_Q{q}.parquet"
        if q_cache.exists():
            try:
                df = pd.read_parquet(q_cache)
                all_rows.append(df)
                log.info(f"EDGAR: {year} Q{q} loaded from cache ({len(df)} filings)")
                continue
            except Exception:
                log.warning(f"EDGAR: corrupted cache for {year} Q{q} — re-downloading")
                q_cache.unlink()

        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/company.idx"
        edgar._wait()
        try:
            r = edgar.session.get(url, timeout=60)
            if r.status_code == 200:
                lines = r.text.split("\n")
                data_lines = [l for l in lines[10:] if len(l) > 20]
                parsed = []
                for line in data_lines:
                    try:
                        company   = line[:62].strip()
                        form_type = line[62:74].strip()
                        cik       = line[74:86].strip()
                        filed     = line[86:98].strip()
                        filename  = line[98:].strip()
                        parsed.append({
                            "company":   company,
                            "form_type": form_type,
                            "cik":       cik,
                            "filed":     filed,
                            "filename":  filename,
                        })
                    except Exception:
                        continue
                df = pd.DataFrame(parsed)
                df = df[df["form_type"].isin(relevant)]
                df.to_parquet(q_cache, index=False)  # save immediately
                all_rows.append(df)
                log.info(f"EDGAR: {year} Q{q} — {len(df)} filings")
            else:
                log.warning(f"EDGAR: HTTP {r.status_code} for {year} Q{q}")
        except Exception as e:
            log.warning(f"EDGAR index error {year} Q{q}: {e}")
        time.sleep(0.5)

    if all_rows:
        # Process in chunks to avoid memory spike
        chunks = []
        for df in all_rows:
            df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
            chunks.append(df[["cik", "form_type", "filed"]]) # keep only needed cols
        master = pd.concat(chunks, ignore_index=True)
        del chunks, all_rows
        master.to_parquet(cache_path, index=False)
        log.info(f"EDGAR: saved {len(master)} total relevant filings.")
        del master
    else:
        log.warning("EDGAR: no index data — edgar features will be null")


# ──────────────────────────────────────────────────────────────
# HELPER — build ticker→CIK mapping from EDGAR
# ──────────────────────────────────────────────────────────────
def build_cik_map(tickers: list[str]) -> dict:
    """Map ticker symbols to SEC CIK numbers using EDGAR company search."""
    cache_path = EDGAR_DIR / "cik_map.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    log.info("Building ticker→CIK map from EDGAR...")
    cik_map = {}

    # EDGAR provides a bulk company tickers JSON
    edgar._wait()
    try:
        r = edgar.session.get(
            "https://www.sec.gov/files/company_tickers.json", timeout=30
        )
        if r.status_code == 200:
            data = r.json()
            for entry in data.values():
                t = entry.get("ticker", "").upper()
                c = str(entry.get("cik_str", "")).zfill(10)
                if t:
                    cik_map[t] = c
            log.info(f"CIK map: {len(cik_map)} tickers mapped.")
    except Exception as e:
        log.warning(f"CIK map error: {e}")

    with open(cache_path, "w") as f:
        json.dump(cik_map, f)
    return cik_map


# ──────────────────────────────────────────────────────────────
# PHASE 2 — FEATURE ENGINEERING HELPERS
# ──────────────────────────────────────────────────────────────

def calc_premarket_features(pm_day: pd.DataFrame, prev_close: float,
                             avg_pm_vol_20d: float) -> dict:
    """Calculate all premarket features from minute bars for one day."""
    out = {
        "pm_open": None, "pm_high": None, "pm_low": None,
        "pm_close": None, "pm_volume": None,
        "pm_gap_pct": None, "pm_move_pct": None,
        "pm_vol_ratio": None, "pm_volume_build": None,
        "pm_high_of_session": None, "pm_fade": None,
        "pm_remaining_to_seed": None, "pm_remaining_to_super": None,
        "pm_remaining_to_mega": None,
        "pm_fetch_ok": False,
    }
    if pm_day.empty or prev_close <= 0:
        return out

    pm_day = pm_day.sort_values("t")
    out["pm_fetch_ok"]       = True
    out["pm_open"]           = float(pm_day.iloc[0]["o"])
    out["pm_high"]           = float(pm_day["h"].max())
    out["pm_low"]            = float(pm_day["l"].min())
    out["pm_close"]          = float(pm_day.iloc[-1]["c"])
    out["pm_volume"]         = float(pm_day["v"].sum())

    if prev_close > 0:
        out["pm_gap_pct"] = (out["pm_open"] - prev_close) / prev_close

    if out["pm_open"] and out["pm_open"] > 0:
        out["pm_move_pct"] = (out["pm_high"] - out["pm_open"]) / out["pm_open"]

    if avg_pm_vol_20d and avg_pm_vol_20d > 0:
        out["pm_vol_ratio"] = out["pm_volume"] / avg_pm_vol_20d

    # Volume build: was volume accelerating into open?
    if len(pm_day) >= 4:
        half = len(pm_day) // 2
        early_vol = pm_day.iloc[:half]["v"].mean()
        late_vol  = pm_day.iloc[half:]["v"].mean()
        out["pm_volume_build"] = 1 if (late_vol > early_vol * 1.2) else 0

    # Still running at open: close == high
    if out["pm_close"] and out["pm_high"]:
        out["pm_high_of_session"] = 1 if (out["pm_close"] >= out["pm_high"] * 0.99) else 0

    # Fade: closed more than 10% below high
    if out["pm_high"] and out["pm_open"] and out["pm_high"] > out["pm_open"]:
        move = out["pm_high"] - out["pm_open"]
        fade = out["pm_high"] - out["pm_close"]
        out["pm_fade"] = 1 if (fade > move * 0.10) else 0

    # Remaining upside at alert time — how much gas is left in the tank?
    # Uses pm_close as the current price at alert time
    if out["pm_close"] and out["pm_close"] > 0 and prev_close > 0:
        seed_target  = prev_close * SEED_MULT
        super_target = prev_close * SUPER_MULT
        mega_target  = prev_close * MEGA_MULT
        pm_price     = out["pm_close"]
        out["pm_remaining_to_seed"]  = (seed_target  - pm_price) / pm_price
        out["pm_remaining_to_super"] = (super_target - pm_price) / pm_price
        out["pm_remaining_to_mega"]  = (mega_target  - pm_price) / pm_price
    else:
        out["pm_remaining_to_seed"]  = None
        out["pm_remaining_to_super"] = None
        out["pm_remaining_to_mega"]  = None

    return out


def calc_ah_features(ah_day: pd.DataFrame, prev_close: float) -> dict:
    """Calculate after-hours features from T-1 evening bars."""
    out = {
        "ah_move_pct": None, "ah_volume": None,
        "ah_direction": None, "ah_fetch_ok": False,
    }
    if ah_day.empty or prev_close <= 0:
        return out

    out["ah_fetch_ok"] = True
    ah_open  = float(ah_day.iloc[0]["o"])
    ah_close = float(ah_day.iloc[-1]["c"])
    out["ah_volume"] = float(ah_day["v"].sum())
    if ah_open > 0:
        out["ah_move_pct"] = (ah_close - ah_open) / ah_open
    out["ah_direction"] = 1 if (ah_close > ah_open) else (-1 if ah_close < ah_open else 0)
    return out


def calc_historical_features(hist: pd.DataFrame, today_idx: int) -> dict:
    """
    Calculate 52-week history features.
    hist is sorted by date, today_idx is the index of the current trading day.
    All features are derived from data BEFORE today_idx (no leakage).
    """
    out = {
        "price_52w_high": None, "price_52w_low": None,
        "pct_from_52w_high": None, "pct_from_52w_low": None,
        "near_52w_low": None,
        "avg_volume_20d": None, "vol_ratio_prev": None,
        "prev_3d_trend": None, "prev_5d_trend": None, "prev_10d_trend": None,
        "days_since_last_spike": None, "coil_days": None,
        "vol_trend_3d": None, "consecutive_vol_days": None,
        "hist_fetch_ok": False,
    }
    if today_idx < 2:
        return out

    # Only use data before today
    past = hist.iloc[:today_idx]
    if len(past) < 5:
        return out

    out["hist_fetch_ok"] = True
    past_52 = past.tail(252)  # ~1 year of trading days

    prev = past.iloc[-1]  # T-1 row

    out["price_52w_high"] = float(past_52["high"].max())
    out["price_52w_low"]  = float(past_52["low"].min())
    prev_close = float(prev["close"])

    if out["price_52w_high"] > 0:
        out["pct_from_52w_high"] = (prev_close - out["price_52w_high"]) / out["price_52w_high"]
    if out["price_52w_low"] > 0:
        out["pct_from_52w_low"]  = (prev_close - out["price_52w_low"]) / out["price_52w_low"]
    if out["price_52w_low"] > 0 and prev_close > 0:
        out["near_52w_low"] = 1 if (prev_close < out["price_52w_low"] * 1.10) else 0

    if len(past) >= 20:
        out["avg_volume_20d"] = float(past.tail(20)["volume"].mean())
        prev_vol = float(prev["volume"])
        if out["avg_volume_20d"] > 0:
            out["vol_ratio_prev"] = prev_vol / out["avg_volume_20d"]

    # Price trends (slope of close over N days)
    for n, key in [(3, "prev_3d_trend"), (5, "prev_5d_trend"), (10, "prev_10d_trend")]:
        if len(past) >= n:
            closes = past.tail(n)["close"].values
            if closes[0] > 0:
                out[key] = (closes[-1] - closes[0]) / closes[0]

    # Days since last 2x spike (volume spike proxy)
    spike_threshold = 3.0
    vol_mean = float(past["volume"].mean()) if len(past) > 0 else 0
    spike_days = 0
    for i in range(len(past) - 1, -1, -1):
        row_vol = float(past.iloc[i]["volume"])
        if vol_mean > 0 and row_vol > vol_mean * spike_threshold:
            break
        spike_days += 1
    out["days_since_last_spike"] = spike_days

    # Coil days: consecutive days with below-average volume (compression)
    coil = 0
    for i in range(len(past) - 1, -1, -1):
        row_vol = float(past.iloc[i]["volume"])
        if out["avg_volume_20d"] and row_vol < out["avg_volume_20d"] * 0.8:
            coil += 1
        else:
            break
    out["coil_days"] = coil

    # Volume trend over last 3 days
    if len(past) >= 3:
        v3 = past.tail(3)["volume"].values
        if v3[0] > 0:
            out["vol_trend_3d"] = (v3[-1] - v3[0]) / v3[0]

    # Consecutive above-average volume days
    consec = 0
    for i in range(len(past) - 1, -1, -1):
        row_vol = float(past.iloc[i]["volume"])
        if out["avg_volume_20d"] and row_vol > out["avg_volume_20d"]:
            consec += 1
        else:
            break
    out["consecutive_vol_days"] = consec

    return out


def calc_float_features(float_data: dict) -> dict:
    """Derive float and market cap features from Polygon ticker details."""
    out = {
        "float_shares": None, "float_M": None, "market_cap": None,
        "float_tier": None, "float_rotation_prev": None,
        "is_foreign_listed": 0,
        "float_fetch_ok": False,
    }
    shares = float_data.get("shares_outstanding")
    mc     = float_data.get("market_cap")
    if shares:
        out["float_fetch_ok"]   = True
        out["float_shares"]     = float(shares)
        out["float_M"]          = float(shares) / 1_000_000
        out["is_foreign_listed"] = float_data.get("is_foreign_listed", 0)
        # Float tier classification
        fm = out["float_M"]
        if fm < 5:
            out["float_tier"] = "nano"
        elif fm < 15:
            out["float_tier"] = "micro"
        elif fm < 50:
            out["float_tier"] = "small"
        elif fm < 200:
            out["float_tier"] = "mid"
        else:
            out["float_tier"] = "large"
    if mc:
        out["market_cap"] = float(mc)
    return out


def calc_si_features(ticker: str, trade_date: date,
                     si_master: pd.DataFrame | None) -> dict:
    """Look up nearest FINRA short interest reading for this ticker/date."""
    out = {
        "si_pct": None, "si_tier": None,
        "days_to_cover": None, "si_fetch_ok": False,
    }
    if si_master is None or si_master.empty:
        return out

    # Find ticker rows and nearest date
    t_col = next((c for c in si_master.columns
                  if "symbol" in c.lower() or "ticker" in c.lower()), None)
    d_col = next((c for c in si_master.columns if "date" in c.lower()), None)
    si_col = next((c for c in si_master.columns
                   if "short" in c.lower() and "vol" not in c.lower()), None)

    if not all([t_col, d_col, si_col]):
        return out

    rows = si_master[si_master[t_col] == ticker].copy()
    if rows.empty:
        return out

    rows["_date"] = pd.to_datetime(rows[d_col], errors="coerce").dt.date
    rows = rows.dropna(subset=["_date"])
    rows["_diff"] = rows["_date"].apply(lambda d: abs((d - trade_date).days))
    nearest = rows.nsmallest(1, "_diff").iloc[0]

    si_val = nearest.get(si_col)
    if si_val:
        try:
            out["si_pct"]      = float(si_val)
            out["si_fetch_ok"] = True
            sp = out["si_pct"]
            if sp < 5:
                out["si_tier"] = "low"
            elif sp < 15:
                out["si_tier"] = "medium"
            elif sp < 30:
                out["si_tier"] = "high"
            else:
                out["si_tier"] = "extreme"
        except (ValueError, TypeError):
            pass
    return out


def calc_edgar_features(ticker: str, trade_date: date,
                         cik_map: dict,
                         edgar_master: pd.DataFrame | None) -> dict:
    """Look up EDGAR 8-K and related filings for this ticker/date."""
    out = {
        "has_8k": 0, "has_8k_yesterday": 0, "has_8k_2days_ago": 0,
        "8k_filing_hour": None, "hours_before_open": None,
        "8k_word_count": None,
        "has_merger": 0, "has_fda": 0, "has_contract": 0,
        "has_earnings": 0, "has_reverse_split": 0,
        "has_dilution": 0, "has_buyback": 0,
        "dilution_count_6m": 0, "dilution_count_30d": 0,
        "days_since_dilution": None,
        "reverse_split_count": 0,
        "is_serial_diluter": 0, "is_serial_reverser": 0,
        "has_form4_buy": 0, "form4_buy_count": 0,
        "has_sc13d": 0,
        "edgar_fetch_ok": False,
        "edgar_dilution_ok": False,
        "form4_fetch_ok": False,
        "sc13d_fetch_ok": False,
    }
    if edgar_master is None or edgar_master.empty:
        return out

    cik = cik_map.get(ticker)
    if not cik:
        return out

    ticker_filings = edgar_master[edgar_master["cik"] == cik].copy()
    if ticker_filings.empty:
        return out

    ticker_filings["filed_date"] = pd.to_datetime(
        ticker_filings["filed"], errors="coerce"
    ).dt.date

    # 8-K lookups
    eightk = ticker_filings[ticker_filings["form_type"].isin(["8-K", "8-K/A"])]
    today       = trade_date
    yesterday   = today - timedelta(days=1)
    two_days    = today - timedelta(days=2)

    if not eightk[eightk["filed_date"] == today].empty:
        out["has_8k"] = 1
    if not eightk[eightk["filed_date"] == yesterday].empty:
        out["has_8k_yesterday"] = 1
    if not eightk[eightk["filed_date"] == two_days].empty:
        out["has_8k_2days_ago"] = 1

    out["edgar_fetch_ok"] = True

    # Dilution: S-1, S-3, 424B*
    dilution_forms = {"S-1", "S-1/A", "S-3", "S-3/A",
                      "424B1", "424B3", "424B4", "424B5"}
    dilution = ticker_filings[ticker_filings["form_type"].isin(dilution_forms)]
    dilution_past = dilution[dilution["filed_date"] < today]

    six_months_ago  = today - timedelta(days=180)
    thirty_days_ago = today - timedelta(days=30)

    out["dilution_count_6m"]  = len(dilution_past[dilution_past["filed_date"] >= six_months_ago])
    out["dilution_count_30d"] = len(dilution_past[dilution_past["filed_date"] >= thirty_days_ago])
    out["is_serial_diluter"]  = 1 if out["dilution_count_6m"] >= 3 else 0

    if not dilution_past.empty:
        last_dil = dilution_past["filed_date"].max()
        out["days_since_dilution"] = (today - last_dil).days

    # Reverse splits — look for 8-Ks mentioning reverse split (simple proxy: count S-1 vs 424B)
    # Full text not fetched in v1; use count as proxy
    out["edgar_dilution_ok"] = True

    # Form 4 — insider buying
    form4 = ticker_filings[ticker_filings["form_type"].isin(["4", "4/A"])]
    form4_recent = form4[
        (form4["filed_date"] >= thirty_days_ago) &
        (form4["filed_date"] < today)
    ]
    if not form4_recent.empty:
        out["has_form4_buy"]   = 1
        out["form4_buy_count"] = len(form4_recent)
    out["form4_fetch_ok"] = True

    # SC 13D — activist investor
    sc13d = ticker_filings[ticker_filings["form_type"].isin(["SC 13D", "SC 13D/A"])]
    sc13d_recent = sc13d[
        (sc13d["filed_date"] >= six_months_ago) &
        (sc13d["filed_date"] < today)
    ]
    if not sc13d_recent.empty:
        out["has_sc13d"] = 1
    out["sc13d_fetch_ok"] = True

    return out


def calc_halt_features(ticker: str, trade_date: date,
                        halts_master: pd.DataFrame | None) -> dict:
    """Look up halt history for this ticker."""
    out = {
        "halted_yesterday": 0, "halt_count_5d": 0,
        "halt_count_30d": 0, "is_serial_halter": 0,
        "halt_fetch_ok": False,
    }
    if halts_master is None or halts_master.empty:
        return out

    # Find ticker column
    t_col = next((c for c in halts_master.columns
                  if "symbol" in c or "issue" in c or "ticker" in c), None)
    d_col = next((c for c in halts_master.columns
                  if "date" in c or "halt" in c), None)
    if not t_col or not d_col:
        return out

    rows = halts_master[halts_master[t_col] == ticker].copy()
    if rows.empty:
        return out

    out["halt_fetch_ok"] = True
    rows["_date"] = pd.to_datetime(rows[d_col], errors="coerce").dt.date
    rows = rows.dropna(subset=["_date"])
    rows = rows[rows["_date"] < trade_date]

    yesterday   = trade_date - timedelta(days=1)
    five_ago    = trade_date - timedelta(days=5)
    thirty_ago  = trade_date - timedelta(days=30)

    out["halted_yesterday"] = 1 if not rows[rows["_date"] == yesterday].empty else 0
    out["halt_count_5d"]    = len(rows[rows["_date"] >= five_ago])
    out["halt_count_30d"]   = len(rows[rows["_date"] >= thirty_ago])
    out["is_serial_halter"] = 1 if out["halt_count_30d"] >= 3 else 0
    return out


def calc_earnings_features(ticker: str, trade_date: date,
                            earnings_dates: list) -> dict:
    """Calculate earnings proximity features."""
    out = {
        "days_to_earnings": None, "has_earnings_soon": 0,
        "had_earnings_recently": 0, "earnings_fetch_ok": False,
    }
    # Handle numpy array
    if hasattr(earnings_dates, 'tolist'):
        earnings_dates = earnings_dates.tolist()
    if not earnings_dates:
        return out

    out["earnings_fetch_ok"] = True
    parsed = []
    for d in earnings_dates:
        try:
            parsed.append(date.fromisoformat(str(d)[:10]))
        except Exception:
            pass

    future = [d for d in parsed if d >= trade_date]
    past   = [d for d in parsed if d < trade_date]

    if future:
        next_earn = min(future)
        out["days_to_earnings"]  = (next_earn - trade_date).days
        out["has_earnings_soon"] = 1 if out["days_to_earnings"] <= 7 else 0

    if past:
        last_earn = max(past)
        days_since = (trade_date - last_earn).days
        out["had_earnings_recently"] = 1 if days_since <= 5 else 0

    return out


def calc_sector_features(sector_data: dict, trade_date: date) -> dict:
    """
    Calculate sector context features.
    CRITICAL: Only use PRIOR DAY data — no same-day leakage.
    """
    out = {
        "spy_prev_day_pct": None, "qqq_prev_day_pct": None,
        "iwm_prev_day_pct": None, "xbi_prev_day_pct": None,
        "pm_spy_pct": None,
        "market_green": None, "market_red": None,
        "sector_hot": None, "sector_fetch_ok": False,
    }

    for ticker, key in [("SPY", "spy"), ("QQQ", "qqq"),
                         ("IWM", "iwm"), ("XBI", "xbi")]:
        if ticker not in sector_data:
            continue
        df = sector_data[ticker]["daily"]
        if df.empty:
            continue
        # Find T-1 row: last close BEFORE trade_date
        prev_rows = df[df["date"] < trade_date].tail(2)
        if len(prev_rows) < 2:
            continue
        t2, t1 = prev_rows.iloc[-2], prev_rows.iloc[-1]
        if float(t2["close"]) > 0:
            pct = (float(t1["close"]) - float(t2["close"])) / float(t2["close"])
            out[f"{key}_prev_day_pct"] = pct
            out["sector_fetch_ok"] = True

    # SPY premarket move (4AM-9:30AM on trade_date — available before open)
    if "SPY" in sector_data:
        pm_df = sector_data["SPY"].get("pm_minute", pd.DataFrame())
        if not pm_df.empty:
            day_pm = pm_df[pm_df["date"] == trade_date]
            if not day_pm.empty:
                pm_open  = float(day_pm.iloc[0]["o"])
                pm_close = float(day_pm.iloc[-1]["c"])
                if pm_open > 0:
                    out["pm_spy_pct"] = (pm_close - pm_open) / pm_open

    # Derived flags from prior day
    spy_pct = out.get("spy_prev_day_pct")
    if spy_pct is not None:
        out["market_green"] = 1 if spy_pct > 0.003 else 0
        out["market_red"]   = 1 if spy_pct < -0.003 else 0

    xbi_pct = out.get("xbi_prev_day_pct")
    if xbi_pct is not None:
        out["sector_hot"] = 1 if xbi_pct > 0.01 else 0

    return out


def calc_time_to_event(ticker: str, trade_date: date,
                        all_labels: dict,
                        edgar_out: dict,
                        halt_out: dict) -> dict:
    """
    Calculate days-since-last-event features.
    All derived from pre-collected data — no leakage.
    """
    out = {
        "days_since_last_8k":       None,
        "days_since_last_halt":     halt_out.get("halt_count_30d"),
        "days_since_last_dilution": edgar_out.get("days_since_dilution"),
        "days_since_last_seed":     None,
    }

    # Days since last 8-K
    if edgar_out.get("has_8k_yesterday"):
        out["days_since_last_8k"] = 1
    elif edgar_out.get("has_8k_2days_ago"):
        out["days_since_last_8k"] = 2

    # Days since ticker last appeared as a seed/super/mega
    ticker_seeds = [
        d for d, tickers in all_labels.items()
        if ticker in tickers and d < trade_date
    ]
    if ticker_seeds:
        last_seed = max(ticker_seeds)
        out["days_since_last_seed"] = (trade_date - last_seed).days

    return out


# ──────────────────────────────────────────────────────────────
# PHASE 2 — ASSEMBLE DAILY TRAINING ROWS
# ──────────────────────────────────────────────────────────────
def load_ticker_cache(ticker: str) -> dict | None:
    """Load raw ticker cache from parquet — memory efficient."""
    cache_path = TICKER_RAW_DIR / f"{ticker}.parquet"
    if not cache_path.exists():
        return None
    try:
        df = pd.read_parquet(cache_path)
        row = df.iloc[0].to_dict()
        del df  # free memory immediately
        # Deserialize nested data
        for key in ["daily", "pm_minute", "ah_minute"]:
            val = row.get(key)
            # Convert whatever format to DataFrame
            if val is None:
                row[key] = pd.DataFrame()
            elif isinstance(val, pd.DataFrame):
                pass  # already good
            elif isinstance(val, (list, dict)):
                row[key] = pd.DataFrame(val) if val else pd.DataFrame()
            else:
                # numpy array — must use list() first
                try:
                    lst = val.tolist() if hasattr(val, "tolist") else list(val)
                    row[key] = pd.DataFrame(lst) if lst else pd.DataFrame()
                except Exception:
                    row[key] = pd.DataFrame()
            # Ensure date column exists
            df = row[key]
            if not isinstance(df, pd.DataFrame):
                row[key] = pd.DataFrame()
                df = row[key]
            if not df.empty and "date" not in df.columns and "t" in df.columns:
                df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET).dt.date
                row[key] = df
            elif not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                row[key] = df
        if isinstance(row.get("float_data"), str):
            row["float_data"] = json.loads(row["float_data"])
        elif hasattr(row.get("float_data"), "tolist"):
            row["float_data"] = row["float_data"].tolist()
        if isinstance(row.get("fetch_ok"), str):
            row["fetch_ok"] = json.loads(row["fetch_ok"])
        elif hasattr(row.get("fetch_ok"), "tolist"):
            row["fetch_ok"] = row["fetch_ok"].tolist()
        # Convert earnings_dates numpy array to list
        if hasattr(row.get("earnings_dates"), "tolist"):
            row["earnings_dates"] = row["earnings_dates"].tolist()
        return row
    except Exception as e:
        log.warning(f"Corrupted cache for {ticker} — deleting and will re-collect: {e}")
        try:
            cache_path.unlink()  # delete corrupted file
        except Exception:
            pass
        return None


def assemble_day(
    trade_date: date,
    tickers: list[str],
    si_master: pd.DataFrame | None,
    halts_master: pd.DataFrame | None,
    edgar_master: pd.DataFrame | None,
    cik_map: dict,
    sector_data: dict,
    all_labels: dict,
) -> pd.DataFrame | None:
    """
    Build one day's training rows.
    Returns DataFrame or None if no positives found.
    """
    rows = []
    universe_count = 0
    skip_no_cache = 0
    skip_no_daily = 0
    skip_filter = 0

    for ticker in tickers:
        cache = load_ticker_cache(ticker)
        if not cache:
            skip_no_cache += 1
            continue

        daily_raw = cache.get("daily", None)
        if daily_raw is None:
            continue
        try:
            if isinstance(daily_raw, pd.DataFrame):
                daily_df = daily_raw
            elif hasattr(daily_raw, "tolist"):
                daily_df = pd.DataFrame(daily_raw.tolist())
            else:
                daily_df = pd.DataFrame(list(daily_raw))
        except Exception:
            continue
        if daily_df.empty:
            continue

        # Ensure date column exists and is date type
        if "date" not in daily_df.columns:
            if "t" in daily_df.columns:
                daily_df["date"] = pd.to_datetime(daily_df["t"], unit="ms", utc=True).dt.tz_convert(ET).dt.date
            else:
                continue
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date

        # Ensure date column is Python date type for comparison
        try:
            daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
        except Exception:
            continue

        # Find today's and T-1 row
        today_rows = daily_df[daily_df["date"] == trade_date]
        past_rows  = daily_df[daily_df["date"] < trade_date].sort_values("date")

        if today_rows.empty or past_rows.empty:
            continue

        today_row = today_rows.iloc[-1]
        prev_row  = past_rows.iloc[-1]

        prev_close  = float(prev_row.get("close", 0))
        prev_vol    = float(prev_row.get("volume", 0))
        prev_dollar = prev_close * prev_vol

        # Universe filter
        if prev_close < MIN_PRICE or prev_dollar < MIN_PREV_DOLLAR_VOL:
            skip_filter += 1
            continue
        universe_count += 1

        day_high   = float(today_row.get("high",   0))
        day_volume = float(today_row.get("volume", 0))
        day_open   = float(today_row.get("open",   0))

        # ── LABEL ──────────────────────────────────────────────
        label = 0
        if prev_close > 0 and day_volume >= MIN_LABEL_VOLUME:
            ratio = day_high / prev_close
            if ratio >= MEGA_MULT:
                label = 3
            elif ratio >= SUPER_MULT:
                label = 2
            elif ratio >= SEED_MULT:
                label = 1
            # Debug: log big movers even if not seeds
            if ratio >= 1.5 and label == 0:
                log.debug(f"Near-seed: {ticker} {trade_date} ratio={ratio:.2f} high={day_high} prev_close={prev_close}")

        # ── T-1 PRICE FEATURES ────────────────────────────────
        prev_body_pct  = None
        prev_wick_ratio = None
        prev_open = float(prev_row.get("open", 0))
        prev_high = float(prev_row.get("high", 0))
        prev_low  = float(prev_row.get("low",  0))
        if prev_open > 0 and prev_close > 0:
            prev_body_pct = abs(prev_close - prev_open) / prev_open
        total_range = prev_high - prev_low
        body        = abs(prev_close - prev_open)
        if total_range > 0:
            prev_wick_ratio = 1 - (body / total_range)

        # ── PREMARKET FEATURES ────────────────────────────────
        pm_minute = cache.get("pm_minute", pd.DataFrame())
        if not pm_minute.empty and "date" in pm_minute.columns:
            pm_minute["date"] = pd.to_datetime(pm_minute["date"]).dt.date
            pm_day = pm_minute[pm_minute["date"] == trade_date]
        else:
            pm_day = pd.DataFrame()

        avg_pm_vol_20d = 0
        if not pm_minute.empty and "date" in pm_minute.columns:
            past_pm = pm_minute[pm_minute["date"] < trade_date]
            if not past_pm.empty:
                avg_pm_vol_20d = past_pm.groupby("date")["v"].sum().tail(20).mean()

        pm_feats = calc_premarket_features(pm_day, prev_close, avg_pm_vol_20d)

        # ── AFTER-HOURS FEATURES (T-1 evening) ────────────────
        ah_minute = cache.get("ah_minute", pd.DataFrame())
        if not ah_minute.empty and "date" in ah_minute.columns:
            ah_minute["date"] = pd.to_datetime(ah_minute["date"]).dt.date
            # AH for T-1 = prior date's evening
            prior_date = prev_row["date"] if "date" in prev_row else None
            ah_day = ah_minute[ah_minute["date"] == prior_date] if prior_date else pd.DataFrame()
        else:
            ah_day = pd.DataFrame()

        ah_feats = calc_ah_features(ah_day, prev_close)

        # ── HISTORICAL FEATURES ───────────────────────────────
        today_idx = len(past_rows)
        hist_feats = calc_historical_features(
            daily_df.sort_values("date").reset_index(drop=True),
            today_idx
        )

        # ── FLOAT FEATURES ────────────────────────────────────
        float_feats = calc_float_features(cache.get("float_data", {}))
        if float_feats.get("float_shares") and hist_feats.get("avg_volume_20d"):
            if float_feats["float_shares"] > 0:
                float_feats["float_rotation_prev"] = (
                    (hist_feats["avg_volume_20d"] * prev_close) /
                    float_feats["float_shares"]
                )

        # ── SHORT INTEREST ────────────────────────────────────
        si_feats = calc_si_features(ticker, trade_date, si_master)

        # ── EDGAR ─────────────────────────────────────────────
        edgar_feats = calc_edgar_features(
            ticker, trade_date, cik_map, edgar_master
        )

        # ── HALTS ─────────────────────────────────────────────
        halt_feats = calc_halt_features(ticker, trade_date, halts_master)

        # ── EARNINGS ──────────────────────────────────────────
        earn_feats = calc_earnings_features(
            ticker, trade_date, cache.get("earnings_dates", [])
        )

        # ── SECTOR ────────────────────────────────────────────
        sector_feats = calc_sector_features(sector_data, trade_date)

        # ── TIME-TO-EVENT ─────────────────────────────────────
        tte_feats = calc_time_to_event(
            ticker, trade_date, all_labels, edgar_feats, halt_feats
        )

        # ── ASSEMBLE ROW ──────────────────────────────────────
        row = {
            # Identity
            "ticker":     ticker,
            "date":       trade_date.isoformat(),
            "label":      label,

            # Outcome — DROP before training
            "day_high":   day_high,
            "day_volume": day_volume,
            "day_open":   day_open,

            # T-1 price structure
            "prev_close":      prev_close,
            "prev_open":       prev_open,
            "prev_high":       prev_high,
            "prev_low":        prev_low,
            "prev_volume":     prev_vol,
            "prev_dollar_vol": prev_dollar,
            "prev_body_pct":   prev_body_pct,
            "prev_wick_ratio": prev_wick_ratio,

            **pm_feats,
            **ah_feats,
            **hist_feats,
            **float_feats,
            **si_feats,
            **edgar_feats,
            **halt_feats,
            **earn_feats,
            **sector_feats,
            **tte_feats,
        }
        rows.append(row)

    log.info(f"{trade_date}: universe={universe_count} no_cache={skip_no_cache} no_daily={skip_no_daily} filtered={skip_filter}")

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # ── CONTROL SAMPLING ──────────────────────────────────────
    positives = df[df["label"] > 0]
    controls  = df[df["label"] == 0]

    n_pos = len(positives)
    if n_pos == 0:
        log.info(f"{trade_date}: 0 positives — skipping day")
        return None

    n_controls = min(n_pos * CONTROL_RATIO, len(controls))
    sampled_controls = controls.sample(n=n_controls, random_state=42)

    final = pd.concat([positives, sampled_controls], ignore_index=True)
    log.info(
        f"{trade_date}: {n_pos} positives "
        f"(seed={len(df[df['label']==1])} "
        f"super={len(df[df['label']==2])} "
        f"mega={len(df[df['label']==3])}) "
        f"+ {n_controls} controls → {len(final)} rows"
    )

    # Track which tickers were positives (for days_since_last_seed)
    all_labels[trade_date] = set(positives["ticker"].tolist())

    return final


def phase2_assemble_all(
    tickers: list[str],
    trading_days: list[date],
    si_master: pd.DataFrame | None,
    halts_master: pd.DataFrame | None,
    edgar_master: pd.DataFrame | None,
    cik_map: dict,
):
    """
    Assemble all training days.
    Loads sector data once, then iterates over trading days.
    """
    log.info("Phase 2: loading sector data...")
    sector_data = {}
    for st in SECTOR_TICKERS:
        log.info(f"Loading sector ticker: {st}")
        cache = load_ticker_cache(st)
        if not cache:
            log.warning(f"No cache for {st} — skipping")
            continue
        log.info(f"{st} cache loaded — processing daily bars")
        daily = cache.get("daily", pd.DataFrame())
        pm    = cache.get("pm_minute", pd.DataFrame())
        if not isinstance(daily, pd.DataFrame):
            daily = pd.DataFrame(daily) if daily is not None else pd.DataFrame()
        if not isinstance(pm, pd.DataFrame):
            pm = pd.DataFrame(pm) if pm is not None else pd.DataFrame()
        if not daily.empty and "date" not in daily.columns and "t" in daily.columns:
            daily["date"] = pd.to_datetime(daily["t"], unit="ms", utc=True).dt.tz_convert(ET).dt.date
        if not pm.empty and "date" not in pm.columns and "t" in pm.columns:
            pm["date"] = pd.to_datetime(pm["t"], unit="ms", utc=True).dt.tz_convert(ET).dt.date
        sector_data[st] = {"daily": daily, "pm_minute": pm}
        log.info(f"{st} loaded: {len(daily)} daily rows, {len(pm)} pm rows")

    all_labels: dict[date, set] = {}
    total_days = len(trading_days)

    for i, trade_date in enumerate(trading_days):
        out_path = OUTPUT_DIR / f"{trade_date.isoformat()}.parquet"
        if out_path.exists():
            log.debug(f"Skipping {trade_date} — already assembled")
            continue

        log.info(f"Assembling {trade_date}...")
        day_df = assemble_day(
            trade_date, tickers,
            si_master, halts_master, edgar_master,
            cik_map, sector_data, all_labels
        )

        if day_df is not None:
            day_df.to_parquet(out_path, index=False)
            del day_df

        # Force garbage collection every 20 days to keep memory low
        if (i + 1) % 20 == 0:
            gc.collect()
            log.info(f"Assemble progress: {i+1}/{total_days} days")

    log.info("Phase 2 complete.")


# ──────────────────────────────────────────────────────────────
# RETRY FAILED TICKERS
# ──────────────────────────────────────────────────────────────
def retry_failed_tickers():
    """
    Re-attempt any tickers that failed in Phase 1.
    Run this after the main collection finishes.
    Uses the fetch_ok flags to know which feature groups to re-pull.
    """
    failed_path = RAW_DIR / "failed_tickers.json"
    if not failed_path.exists():
        log.info("No failed tickers to retry.")
        return

    with open(failed_path) as f:
        failed = json.load(f)

    log.info(f"Retrying {len(failed)} failed tickers...")
    still_failed = []

    for ticker in failed:
        # Delete existing partial cache if any
        cache_path = TICKER_RAW_DIR / f"{ticker}.parquet"
        if cache_path.exists():
            cache_path.unlink()

        ok = collect_ticker(ticker)
        if not ok:
            still_failed.append(ticker)

    if still_failed:
        with open(failed_path, "w") as f:
            json.dump(still_failed, f)
        log.warning(f"Still failed after retry: {len(still_failed)} tickers")
    else:
        failed_path.unlink()
        log.info("All retries succeeded.")


def patch_missing_features(tickers: list[str],
                            si_master: pd.DataFrame | None,
                            halts_master: pd.DataFrame | None,
                            edgar_master: pd.DataFrame | None,
                            cik_map: dict):
    """
    Scan all cached tickers and identify those with fetch_ok=False
    for specific feature groups. Re-attempt those specific pulls only.
    This lets you fix partial failures without re-collecting everything.
    """
    log.info("Scanning for partial fetch failures...")
    needs_pm    = []
    needs_float = []
    needs_earn  = []

    for ticker in tickers:
        cache_path = TICKER_RAW_DIR / f"{ticker}.parquet"
        if not cache_path.exists():
            continue
        try:
            df  = pd.read_parquet(cache_path)
            row = df.iloc[0].to_dict()
            fo  = row.get("fetch_ok", {})
            if isinstance(fo, str):
                fo = json.loads(fo)
            if not fo.get("pm_fetch_ok"):
                needs_pm.append(ticker)
            if not fo.get("float_fetch_ok"):
                needs_float.append(ticker)
            if not fo.get("earnings_fetch_ok"):
                needs_earn.append(ticker)
        except Exception:
            continue

    log.info(
        f"Patch needed — PM: {len(needs_pm)} | "
        f"Float: {len(needs_float)} | Earnings: {len(needs_earn)}"
    )

    # Re-collect only what's missing
    for ticker in needs_pm + needs_float + needs_earn:
        cache_path = TICKER_RAW_DIR / f"{ticker}.parquet"
        if cache_path.exists():
            cache_path.unlink()
        collect_ticker(ticker)

    log.info("Patch complete.")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # Keep Render from killing the process — MUST be first
    start_keepalive()
    time.sleep(2)  # give health check time to bind before any work starts

    if not POLYGON_API_KEY:
        log.error("MASSIVE_API_KEY not set — aborting.")
        return

    log.info("=" * 60)
    log.info("THE DELTA v2 — Data Collector Starting")
    log.info(f"Window: {START_DATE} → {END_DATE}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("=" * 60)

    # ── Step 1: Get ticker universe ────────────────────────────
    tickers = fetch_ticker_list()
    if not tickers:
        log.error("No tickers — aborting.")
        return

    # ── Step 2: Bulk collect all tickers ──────────────────────
    phase1_collect_all_tickers(tickers)

    # ── Step 3: Collect support data ──────────────────────────
    collect_finra_short_interest()
    collect_halt_history()
    collect_edgar_index()

    # ── Step 4: Build CIK map ─────────────────────────────────
    cik_map = build_cik_map(tickers)

    # ── Step 5: Load support data ─────────────────────────────
    si_master     = None
    halts_master  = None
    edgar_master  = None

    si_path    = FINRA_DIR  / "si_master.parquet"
    halt_path  = HALTS_DIR  / "halts_master.parquet"
    edgar_path = EDGAR_DIR  / "filings_master.parquet"

    if si_path.exists():
        si_master = pd.read_parquet(si_path)
        log.info(f"Loaded FINRA SI: {len(si_master)} rows")
    if halt_path.exists():
        halts_master = pd.read_parquet(halt_path)
        log.info(f"Loaded halt history: {len(halts_master)} rows")
    if edgar_path.exists():
        edgar_master = pd.read_parquet(edgar_path)
        log.info(f"Loaded EDGAR index: {len(edgar_master)} filings")

    # ── Step 6: Get trading calendar ──────────────────────────
    trading_days = get_trading_days(START_DATE, END_DATE)
    log.info(f"Trading days: {len(trading_days)}")

    # ── Step 7: Assemble training data ────────────────────────
    phase2_assemble_all(
        tickers, trading_days,
        si_master, halts_master, edgar_master, cik_map
    )

    log.info("=" * 60)
    log.info("Collection complete.")
    log.info(f"Training files saved to: {OUTPUT_DIR}")
    log.info("Next step: run data_fixer.py, then trainer.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
