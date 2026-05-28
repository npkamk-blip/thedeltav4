"""
THE DELTA v2 — collect_pm_1min.py (Phase 1B)
=============================================
Runs AFTER phase1_5_seed_discovery.py.

Pulls 1-minute premarket bars (4:00AM - 9:30AM) for:
  - All seed tickers found in seed_registry.json
  - All control tickers found in control_registry.json

Why 1-min bars?
  - 5-min bars lose 5-15 minutes of a move waiting for confirmation
  - 1-min bars let us detect real multi-bar volume buildup vs single spikes
  - Required for reliable PM fake-move detection in morning model

Output: /app/data/raw/pm_1min/<TICKER>.parquet
  Each file contains all 1-min PM bars for that ticker
  across the full 2025-2026 window.
"""

import os, time, logging, requests, json, gc, threading
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from http.server import HTTPServer, BaseHTTPRequestHandler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
POLYGON_API_KEY = os.environ.get("MASSIVE_API_KEY", "")

DATA_ROOT      = Path(os.environ.get("DATA_DIR", "/app/data"))
RAW_DIR        = DATA_ROOT / "raw"
PM_1MIN_DIR    = RAW_DIR / "pm_1min"
LOG_DIR        = DATA_ROOT / "logs"

START_DATE = date(2025, 1, 1)
END_DATE   = date(2026, 5, 23)

CALLS_PER_MINUTE    = 250
CALL_INTERVAL       = 60.0 / CALLS_PER_MINUTE

ET = ZoneInfo("America/New_York")

KNOWN_HOLIDAYS = {
    date(2025,1,1), date(2025,1,9), date(2025,1,20), date(2025,2,17),
    date(2025,4,18), date(2025,5,26), date(2025,6,19), date(2025,7,4),
    date(2025,9,1), date(2025,11,27), date(2025,12,25),
    date(2026,1,1), date(2026,1,19), date(2026,2,16), date(2026,4,3),
    date(2026,5,25),
}

# Stats
STATS = {
    "attempted": 0,
    "ok":        0,
    "failed":    0,
    "skipped":   0,
    "total_bars": 0,
}

# ─────────────────────────────────────────────
# DIRS + LOGGING
# ─────────────────────────────────────────────
for d in [PM_1MIN_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "collect_pm_1min.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("pm_1min")


# ─────────────────────────────────────────────
# KEEPALIVE
# ─────────────────────────────────────────────
class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b"alive")
    def log_message(self, *a): pass

def start_keepalive(port=8080):
    from http.server import HTTPServer
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
                elif r.status_code == 404:
                    return None
                elif r.status_code == 403:
                    log.error(f"FAIL 403 forbidden: {url}")
                    return None
                else:
                    log.warning(f"HTTP {r.status_code} attempt {attempt+1}: {path}")
                    time.sleep(3)
            except Exception as e:
                log.warning(f"Request error (attempt {attempt+1}): {e}")
                time.sleep(2)
        log.error(f"FAIL all retries: {path}")
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

    def aggs(self, ticker, mult, span, from_, to, adjusted=False) -> list:
        params = {
            "adjusted": str(adjusted).lower(),
            "sort":     "asc",
            "limit":    50000,
        }
        return self.paginate(
            f"/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{from_}/{to}",
            params
        )

poly = PolygonClient()


# ─────────────────────────────────────────────
# LOAD CANDIDATES
# ─────────────────────────────────────────────
def load_candidates() -> list[str]:
    """Load PM 1-min candidates from phase 1.5 output."""
    candidates_path = RAW_DIR / "pm_1min_candidates.json"
    if not candidates_path.exists():
        log.error(f"FAIL | {candidates_path} not found — run phase1_5_seed_discovery.py first")
        return []

    with open(candidates_path) as f:
        candidates = json.load(f)

    log.info(f"Loaded {len(candidates)} PM 1-min candidates")
    return candidates


# ─────────────────────────────────────────────
# COLLECT 1-MIN PM BARS FOR ONE TICKER
# ─────────────────────────────────────────────
def collect_1min_pm(ticker: str) -> bool:
    """
    Pull 1-minute premarket bars for a single ticker.
    Saves to /app/data/raw/pm_1min/TICKER.parquet
    
    Only stores premarket bars: 4:00AM - 9:30AM ET
    This keeps storage manageable vs storing full day.
    """
    cache_path = PM_1MIN_DIR / f"{ticker}.parquet"
    if cache_path.exists():
        try:
            # Verify not corrupted
            df = pd.read_parquet(cache_path, columns=["t"])
            STATS["skipped"] += 1
            return True
        except Exception:
            log.warning(f"{ticker}: corrupted 1min cache — re-collecting")
            cache_path.unlink()

    STATS["attempted"] += 1
    from_str = START_DATE.strftime("%Y-%m-%d")
    to_str   = END_DATE.strftime("%Y-%m-%d")

    # Pull in chunks to avoid timeouts
    all_bars = []
    chunks = [
        ("2025-01-01", "2025-06-30"),
        ("2025-07-01", "2025-12-31"),
        ("2026-01-01", "2026-05-23"),
    ]

    for cfrom, cto in chunks:
        bars = poly.aggs(ticker, 1, "minute", cfrom, cto)
        if bars:
            all_bars.extend(bars)
        else:
            log.warning(f"WARN 1min | {ticker} | no bars for chunk {cfrom}:{cto}")

    if not all_bars:
        log.error(f"FAIL 1min | {ticker} | no 1-min bars returned at all")
        STATS["failed"] += 1
        return False

    # Parse and filter to premarket only
    df = pd.DataFrame(all_bars)
    df["t"]      = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    df["hour"]   = df["t"].dt.hour
    df["minute"] = df["t"].dt.minute
    df["date"]   = df["t"].dt.date

    # Premarket: 4:00AM to 9:29AM ET only
    pm_df = df[
        (df["hour"] >= 4) &
        ((df["hour"] < 9) | ((df["hour"] == 9) & (df["minute"] < 30)))
    ].copy()

    if pm_df.empty:
        log.warning(f"WARN 1min | {ticker} | no premarket bars found in {len(all_bars)} total bars")
        # Still save empty file so we don't re-collect
        pm_df.to_parquet(cache_path, index=False)
        STATS["failed"] += 1
        return False

    pm_df = pm_df.rename(columns={
        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap"
    })

    # Save
    pm_df.to_parquet(cache_path, index=False)
    STATS["ok"] += 1
    STATS["total_bars"] += len(pm_df)

    log.info(
        f"OK 1min | {ticker} | "
        f"{len(pm_df)} PM bars across "
        f"{pm_df['date'].nunique()} days"
    )
    return True


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
    log.info("THE DELTA v2 — Phase 1B: 1-Minute PM Bar Collection")
    log.info(f"Window: {START_DATE} → {END_DATE}")
    log.info(f"Output: {PM_1MIN_DIR}")
    log.info("=" * 60)

    # Load candidates from phase 1.5
    candidates = load_candidates()
    if not candidates:
        return

    total = len(candidates)
    log.info(f"Collecting 1-min PM bars for {total} tickers...")
    log.info("Estimated time: 2-4 hours depending on Polygon rate limits")

    for i, ticker in enumerate(candidates):
        collect_1min_pm(ticker)

        if (i + 1) % 50 == 0:
            log.info(
                f"Progress: {i+1}/{total} | "
                f"ok={STATS['ok']} | "
                f"failed={STATS['failed']} | "
                f"skipped={STATS['skipped']} | "
                f"total_bars={STATS['total_bars']:,}"
            )
            gc.collect()

    # Final summary
    log.info("=" * 60)
    log.info("PHASE 1B COMPLETE")
    log.info(f"Attempted:   {STATS['attempted']}")
    log.info(f"OK:          {STATS['ok']}")
    log.info(f"Failed:      {STATS['failed']}")
    log.info(f"Skipped:     {STATS['skipped']} (already cached)")
    log.info(f"Total bars:  {STATS['total_bars']:,}")

    if STATS["failed"] > 0:
        log.warning(
            f"{STATS['failed']} tickers failed — "
            f"check logs, re-run to retry"
        )

    # Disk usage estimate
    try:
        import shutil
        usage = shutil.disk_usage(PM_1MIN_DIR)
        log.info(
            f"Disk: {usage.used/1e9:.1f}GB used, "
            f"{usage.free/1e9:.1f}GB free"
        )
    except Exception:
        pass

    log.info("")
    log.info("Next step: python scanner/assembler.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
