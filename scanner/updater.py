"""
THE DELTA v2 — updater.py
==========================
Runs continuously on the data pull service.
Automatically keeps support data fresh:

  EDGAR — updates daily at 6PM ET
          downloads latest SEC filing index
          appends new filings to filings_master.parquet

  FINRA SI — updates bi-weekly (FINRA publishes Mon/Wed)
             downloads latest short interest file
             appends to si_master.parquet

No manual intervention needed. Run this alongside the scanner.
"""

import os, time, logging, threading, requests, json, gc
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from http.server import HTTPServer, BaseHTTPRequestHandler

ET = ZoneInfo("America/New_York")

DATA_ROOT  = Path(os.environ.get("DATA_DIR", "/app/data"))
RAW_DIR    = DATA_ROOT / "raw"
EDGAR_DIR  = RAW_DIR / "edgar"
FINRA_DIR  = RAW_DIR / "finra"
LOG_DIR    = DATA_ROOT / "logs"

for d in [EDGAR_DIR, FINRA_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "updater.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("updater")

EDGAR_USER_AGENT = os.environ.get("EDGAR_USER_AGENT", "NPKNOB@gmail.com")

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
# EDGAR DAILY UPDATE
# ─────────────────────────────────────────────
def get_edgar_quarter(d: date) -> str:
    """Return SEC quarter string like 2026q2"""
    q = (d.month - 1) // 3 + 1
    return f"{d.year}q{q}"


def download_edgar_index(quarter: str) -> pd.DataFrame | None:
    """Download SEC full-index for a quarter and return as DataFrame."""
    url = f"https://www.sec.gov/Archives/edgar/full-index/{quarter[:4]}/QTR{quarter[-1]}/company.idx"
    headers = {"User-Agent": EDGAR_USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code != 200:
            log.warning(f"EDGAR index HTTP {r.status_code}: {url}")
            return None

        lines = r.text.strip().split("\n")
        # Skip header lines (first 9 lines are headers/separator)
        data_lines = []
        for line in lines[9:]:
            if len(line) < 80:
                continue
            try:
                company   = line[:62].strip()
                form_type = line[62:74].strip()
                cik       = line[74:86].strip()
                filed     = line[86:98].strip()
                filename  = line[98:].strip()
                if cik and form_type and filed:
                    data_lines.append({
                        "cik":       cik,
                        "form_type": form_type,
                        "filed":     pd.to_datetime(filed, errors="coerce"),
                        "filename":  filename,
                    })
            except Exception:
                continue

        if not data_lines:
            log.warning(f"EDGAR index empty: {quarter}")
            return None

        df = pd.DataFrame(data_lines)
        df = df.dropna(subset=["filed"])
        log.info(f"EDGAR index {quarter}: {len(df):,} filings")
        return df

    except Exception as e:
        log.error(f"FAIL EDGAR index {quarter}: {e}")
        return None


def update_edgar():
    """
    Update EDGAR master with today's new filings.
    Downloads current quarter index and merges with existing master.
    """
    log.info("Updating EDGAR master...")

    master_path = EDGAR_DIR / "filings_master.parquet"
    today       = date.today()
    quarter     = get_edgar_quarter(today)

    # Download latest quarter index
    new_df = download_edgar_index(quarter)
    if new_df is None:
        log.warning("EDGAR update failed — keeping existing master")
        return False

    # If master exists, merge and deduplicate
    if master_path.exists():
        try:
            existing = pd.read_parquet(master_path)
            log.info(f"Existing master: {len(existing):,} filings")

            # Find truly new filings (filed today or yesterday)
            cutoff = pd.Timestamp(today - timedelta(days=2))
            new_recent = new_df[new_df["filed"] >= cutoff]

            if len(new_recent) == 0:
                log.info("No new filings since yesterday — master up to date")
                return True

            # Merge
            combined = pd.concat([existing, new_recent], ignore_index=True)
            combined = combined.drop_duplicates(subset=["cik","form_type","filed","filename"])
            combined = combined.sort_values("filed").reset_index(drop=True)

            combined.to_parquet(master_path, index=False)
            new_count = len(combined) - len(existing)
            log.info(f"EDGAR master updated: +{new_count} new filings ({len(combined):,} total)")
            return True

        except Exception as e:
            log.error(f"EDGAR merge error: {e}")
            # Rebuild from scratch
            new_df.to_parquet(master_path, index=False)
            log.info(f"EDGAR master rebuilt: {len(new_df):,} filings")
            return True
    else:
        # No existing master — save fresh
        new_df.to_parquet(master_path, index=False)
        log.info(f"EDGAR master created: {len(new_df):,} filings")
        return True


# ─────────────────────────────────────────────
# FINRA SI BI-WEEKLY UPDATE
# ─────────────────────────────────────────────
def get_latest_finra_url() -> str | None:
    """
    FINRA publishes short interest data twice a month:
      Settlement dates around 1st and 15th
      Published ~5 business days after settlement

    Try recent dates and find the latest available file.
    """
    base = "https://cdn.finra.org/equity/regsho/biweekly/FNRAshvol"
    today = date.today()

    # Try last 30 days
    for i in range(30):
        check = today - timedelta(days=i)
        if check.weekday() >= 5:  # skip weekends
            continue
        url = f"{base}{check.strftime('%Y%m%d')}.txt"
        try:
            r = requests.head(url, timeout=10)
            if r.status_code == 200:
                log.info(f"Found FINRA file: {check.strftime('%Y%m%d')}")
                return url, check
        except Exception:
            continue
    return None, None


def download_finra_si(url: str, si_date: date) -> pd.DataFrame | None:
    """Download and parse a FINRA short interest file."""
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            log.warning(f"FINRA HTTP {r.status_code}: {url}")
            return None

        lines = r.text.strip().split("\n")
        rows = []
        for line in lines[1:]:  # skip header
            parts = line.strip().split("|")
            if len(parts) >= 5:
                try:
                    rows.append({
                        "Date":             parts[3].strip() if len(parts) > 3 else str(si_date),
                        "Symbol":           parts[0].strip().upper(),
                        "ShortVolume":      float(parts[1]) if parts[1].strip() else 0,
                        "ShortExemptVolume": float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0,
                        "TotalVolume":      float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0,
                        "Market":           parts[4].strip() if len(parts) > 4 else "",
                        "si_date":          str(si_date),
                    })
                except Exception:
                    continue

        if not rows:
            log.warning(f"FINRA file empty: {url}")
            return None

        df = pd.DataFrame(rows)
        log.info(f"FINRA SI {si_date}: {len(df):,} rows")
        return df

    except Exception as e:
        log.error(f"FAIL FINRA download: {e}")
        return None


def update_si():
    """
    Update SI master with latest FINRA data.
    FINRA publishes bi-weekly — checks automatically.
    """
    log.info("Checking for FINRA SI update...")

    master_path = FINRA_DIR / "si_master.parquet"

    # Find latest FINRA file
    url, si_date = get_latest_finra_url()
    if not url:
        log.warning("No recent FINRA file found")
        return False

    # Check if we already have this date
    daily_path = FINRA_DIR / f"finra_{si_date.strftime('%Y%m%d')}.parquet"
    if daily_path.exists():
        log.info(f"FINRA {si_date} already cached — no update needed")
        return True

    # Download new file
    new_df = download_finra_si(url, si_date)
    if new_df is None:
        return False

    # Save daily file
    new_df.to_parquet(daily_path, index=False)
    log.info(f"Saved daily FINRA: {daily_path.name}")

    # Rebuild si_master from all daily files
    log.info("Rebuilding si_master from all daily files...")
    files = sorted(FINRA_DIR.glob("finra_*.parquet"))
    log.info(f"Found {len(files)} daily FINRA files")

    all_rows = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            all_rows.append(df)
        except Exception as e:
            log.warning(f"Bad FINRA file {f.name}: {e}")

    if all_rows:
        master = pd.concat(all_rows, ignore_index=True)
        master.to_parquet(master_path, index=False)
        log.info(f"SI master rebuilt: {len(master):,} rows from {len(files)} files")
        del master, all_rows
        gc.collect()
        return True

    return False


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    threading.Thread(target=start_keepalive, daemon=True).start()
    time.sleep(2)

    log.info("=" * 60)
    log.info("THE DELTA v2 — Auto Updater")
    log.info("EDGAR: daily at 6PM ET")
    log.info("FINRA SI: bi-weekly (auto-detected)")
    log.info("=" * 60)

    # Run immediately on startup to catch up
    log.info("Running initial update on startup...")
    try:
        update_edgar()
    except Exception as e:
        log.error(f"Startup EDGAR update error: {e}")
    try:
        update_si()
    except Exception as e:
        log.error(f"Startup SI update error: {e}")

    edgar_updated_today = False
    si_updated_today    = False
    last_date           = None

    while True:
        now    = datetime.now(ET)
        today  = now.date()
        hour   = now.hour
        minute = now.minute

        # Reset daily flags
        if last_date != today:
            edgar_updated_today = False
            si_updated_today    = False
            last_date = today
            log.info(f"New day: {today}")

        # EDGAR update — 6PM ET daily
        if hour == 18 and minute < 5 and not edgar_updated_today:
            log.info("6PM — running EDGAR update...")
            try:
                success = update_edgar()
                edgar_updated_today = True
                log.info(f"EDGAR update: {'OK' if success else 'FAILED'}")
            except Exception as e:
                log.error(f"EDGAR update error: {e}")

        # SI update — check at 8AM daily (FINRA publishes overnight)
        if hour == 8 and minute < 5 and not si_updated_today:
            log.info("8AM — checking FINRA SI update...")
            try:
                success = update_si()
                si_updated_today = True
                log.info(f"SI update: {'OK' if success else 'no new data'}")
            except Exception as e:
                log.error(f"SI update error: {e}")

        # Sleep 5 minutes between checks
        time.sleep(300)


if __name__ == "__main__":
    main()
