"""
THE DELTA v2 — fix_si_edgar.py
================================
Run this BEFORE re-running the assembler.

1. Rebuilds si_master.parquet from individual daily FINRA files
2. Checks EDGAR master column names and fixes if needed
3. Verifies both are readable and correct

Run from shell:
    python scanner/fix_si_edgar.py
"""

import pandas as pd
import json
import logging
from pathlib import Path

DATA_ROOT = Path("/app/data")
RAW_DIR   = DATA_ROOT / "raw"
FINRA_DIR = RAW_DIR / "finra"
EDGAR_DIR = RAW_DIR / "edgar"
LOG_DIR   = DATA_ROOT / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fix_si_edgar.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fix")


# ─────────────────────────────────────────────
# FIX 1 — Rebuild SI master
# ─────────────────────────────────────────────
def rebuild_si_master():
    log.info("=" * 50)
    log.info("Rebuilding si_master.parquet from daily files...")

    files = sorted(FINRA_DIR.glob("finra_*.parquet"))
    log.info(f"Found {len(files)} daily FINRA files")

    if not files:
        log.error("FAIL | no daily FINRA files found")
        return False

    all_rows = []
    bad = 0
    for f in files:
        try:
            df = pd.read_parquet(f)
            all_rows.append(df)
        except Exception as e:
            log.warning(f"WARN bad file {f.name}: {e}")
            bad += 1

    if not all_rows:
        log.error("FAIL | no readable FINRA files")
        return False

    log.info(f"Read {len(all_rows)} files ({bad} bad)")
    master = pd.concat(all_rows, ignore_index=True)

    # Save
    out_path = FINRA_DIR / "si_master.parquet"
    master.to_parquet(out_path, index=False)
    log.info(f"OK | si_master saved: {len(master):,} rows")
    log.info(f"Columns: {list(master.columns)}")

    # Verify it reads back
    try:
        verify = pd.read_parquet(out_path)
        log.info(f"VERIFY OK | {len(verify):,} rows readable")
        return True
    except Exception as e:
        log.error(f"FAIL verify: {e}")
        return False


# ─────────────────────────────────────────────
# FIX 2 — Check and fix EDGAR master
# ─────────────────────────────────────────────
def check_edgar_master():
    log.info("=" * 50)
    log.info("Checking EDGAR master...")

    master_path = EDGAR_DIR / "filings_master.parquet"
    if not master_path.exists():
        log.error("FAIL | filings_master.parquet not found")
        return None

    try:
        # Load just schema first
        df = pd.read_parquet(master_path, columns=None)
        log.info(f"EDGAR master columns: {list(df.columns)}")
        log.info(f"EDGAR master rows: {len(df):,}")

        # Show sample
        log.info(f"Sample row:\n{df.iloc[0].to_dict()}")

        return list(df.columns)

    except Exception as e:
        log.error(f"FAIL reading EDGAR master: {e}")

        # Try reading quarterly caches to rebuild
        log.info("Trying to rebuild from quarterly caches...")
        q_files = sorted(EDGAR_DIR.glob("edgar_*_Q*.parquet"))
        log.info(f"Found {len(q_files)} quarterly files")

        if not q_files:
            log.error("FAIL | no quarterly EDGAR files found")
            return None

        all_rows = []
        for f in q_files:
            try:
                df = pd.read_parquet(f)
                log.info(f"OK {f.name}: {len(df)} rows, cols={list(df.columns)}")
                all_rows.append(df[["cik","form_type","filed","filename"]])
            except Exception as e2:
                log.warning(f"WARN {f.name}: {e2}")

        if all_rows:
            master = pd.concat(all_rows, ignore_index=True)
            master.to_parquet(master_path, index=False)
            log.info(f"Rebuilt EDGAR master: {len(master):,} rows")
            log.info(f"Columns: {list(master.columns)}")
            return list(master.columns)

        return None


# ─────────────────────────────────────────────
# FIX 3 — Delete assembled parquets so assembler re-runs
# ─────────────────────────────────────────────
def clear_assembled():
    log.info("=" * 50)
    log.info("Clearing assembled training parquets...")

    midnight_dir = DATA_ROOT / "training_data_v2" / "midnight"
    morning_dir  = DATA_ROOT / "training_data_v2" / "morning"

    mid_files = list(midnight_dir.glob("*.parquet")) if midnight_dir.exists() else []
    mor_files = list(morning_dir.glob("*.parquet"))  if morning_dir.exists()  else []

    log.info(f"Midnight files to delete: {len(mid_files)}")
    log.info(f"Morning files to delete:  {len(mor_files)}")

    confirm = input("Delete all assembled parquets and re-assemble? (yes/no): ")
    if confirm.strip().lower() != "yes":
        log.info("Skipped — no files deleted")
        return

    for f in mid_files + mor_files:
        f.unlink()

    log.info(f"Deleted {len(mid_files) + len(mor_files)} files")
    log.info("Ready to re-run assembler")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("THE DELTA v2 — SI + EDGAR Fix Script")
    log.info("=" * 60)

    # Step 1: Rebuild SI
    si_ok = rebuild_si_master()

    # Step 2: Check EDGAR
    edgar_cols = check_edgar_master()

    # Step 3: Summary
    log.info("=" * 60)
    log.info("SUMMARY:")
    log.info(f"  SI master:    {'OK' if si_ok else 'FAILED'}")
    log.info(f"  EDGAR cols:   {edgar_cols}")

    if si_ok and edgar_cols:
        log.info("")
        log.info("Both SI and EDGAR look good.")
        log.info("Now update assembler.py to use correct EDGAR columns.")
        log.info("Then run clear_assembled() and re-deploy assembler.")
        log.info("")
        log.info("EDGAR columns available:")
        for col in (edgar_cols or []):
            log.info(f"  - {col}")

    # Step 4: Optionally clear assembled files
    log.info("=" * 60)
    clear_assembled()


if __name__ == "__main__":
    main()
