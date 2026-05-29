"""
THE DELTA v2 — push_models.py
===============================
Runs once on the data pull service.
Pushes model files from /app/data/models/ to GitHub.

Requires environment variable:
  GITHUB_TOKEN = your personal access token with repo scope

Run once then never again — models stay in GitHub permanently.
"""

import os, sys, json, base64, requests, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("push_models")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO  = os.environ.get("GITHUB_REPO", "npkamk-blip/thedeltav4")
MODEL_DIR    = Path(os.environ.get("DATA_DIR", "/app/data")) / "models"

FILES_TO_PUSH = [
    "midnight_seed_model.json",
    "midnight_super_model.json",
    "morning_seed_model.json",
    "morning_super_model.json",
    "feature_cols.json",
    "thresholds.json",
]

# Support data files — built from full parquets, small enough for GitHub
SUPPORT_FILES_TO_BUILD = True  # set False to skip rebuilding

def get_file_sha(path: str) -> str | None:
    """Get existing file SHA from GitHub (needed for updates)."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def push_file(local_path: Path, github_path: str) -> bool:
    """Push a single file to GitHub via API."""
    if not local_path.exists():
        log.error(f"File not found: {local_path}")
        return False

    # Read and base64 encode
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    # Check if file already exists
    sha = get_file_sha(github_path)

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{github_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "message": f"Update {github_path.split('/')[-1]}",
        "content": content,
    }
    if sha:
        payload["sha"] = sha  # required for updates

    r = requests.put(url, headers=headers, json=payload)
    if r.status_code in (200, 201):
        log.info(f"✅ Pushed: {github_path} ({local_path.stat().st_size:,} bytes)")
        return True
    else:
        log.error(f"❌ Failed: {github_path} — HTTP {r.status_code}: {r.text[:200]}")
        return False


def build_support_files():
    """
    Build lightweight support files from full parquets.
    These are small enough to push to GitHub.
    """
    import pandas as pd
    from datetime import date, timedelta

    data_root = Path(os.environ.get("DATA_DIR", "/app/data"))
    out_dir   = data_root / "support"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── SI lookup: most recent SI reading per ticker ──
    si_path = data_root / "raw/finra/si_master.parquet"
    if si_path.exists():
        try:
            si = pd.read_parquet(si_path, columns=["Date","Symbol","ShortVolume","TotalVolume"])
            si["_date"] = pd.to_datetime(si["Date"], errors="coerce").dt.date
            si = si.dropna(subset=["_date","Symbol"])
            # Keep only most recent reading per ticker
            latest = si.sort_values("_date").groupby("Symbol").last().reset_index()
            latest = latest[["Symbol","ShortVolume","TotalVolume","_date"]]
            latest["si_pct"] = (latest["ShortVolume"] / latest["TotalVolume"].replace(0, float("nan")) * 100).round(2)
            latest["si_tier"] = latest["si_pct"].apply(
                lambda x: 0 if x < 5 else (1 if x < 15 else (2 if x < 30 else 3)) if pd.notna(x) else -1
            )
            out = out_dir / "si_lookup.parquet"
            latest.to_parquet(out, index=False)
            log.info(f"Built si_lookup.parquet: {len(latest):,} tickers ({out.stat().st_size:,} bytes)")
        except Exception as e:
            log.error(f"FAIL build si_lookup: {e}")
    else:
        log.warning("SI master not found — skipping si_lookup")

    # ── EDGAR recent: last 90 days of filings ──
    edgar_path = data_root / "raw/edgar/filings_master.parquet"
    cik_path   = data_root / "raw/edgar/cik_map.json"
    if edgar_path.exists():
        try:
            edgar = pd.read_parquet(edgar_path, columns=["cik","form_type","filed","filename"])
            edgar["filed_date"] = pd.to_datetime(edgar["filed"], errors="coerce").dt.date
            cutoff = date.today() - timedelta(days=90)
            recent = edgar[edgar["filed_date"] >= cutoff].copy()
            out = out_dir / "edgar_recent.parquet"
            recent.to_parquet(out, index=False)
            log.info(f"Built edgar_recent.parquet: {len(recent):,} filings ({out.stat().st_size:,} bytes)")
        except Exception as e:
            log.error(f"FAIL build edgar_recent: {e}")
    else:
        log.warning("EDGAR master not found — skipping edgar_recent")

    # ── CIK map ──
    if cik_path.exists():
        import shutil
        shutil.copy(cik_path, out_dir / "cik_map.json")
        log.info(f"Copied cik_map.json ({cik_path.stat().st_size:,} bytes)")
    else:
        log.warning("cik_map.json not found")

    return out_dir


def main():
    log.info("=" * 50)
    log.info("THE DELTA v2 — Push Models to GitHub")
    log.info("=" * 50)

    if not GITHUB_TOKEN:
        log.error("FAIL | GITHUB_TOKEN not set")
        log.error("Add GITHUB_TOKEN to environment variables on this service")
        sys.exit(1)

    log.info(f"Repo:      {GITHUB_REPO}")
    log.info(f"Model dir: {MODEL_DIR}")

    ok = 0
    fail = 0

    # ── Push model files ──
    log.info("Pushing model files...")
    for filename in FILES_TO_PUSH:
        local_path  = MODEL_DIR / filename
        github_path = f"models/{filename}"
        if push_file(local_path, github_path):
            ok += 1
        else:
            fail += 1

    # ── Build and push support files ──
    log.info("Building support files...")
    support_dir = build_support_files()

    support_files = [
        ("si_lookup.parquet",    "support/si_lookup.parquet"),
        ("edgar_recent.parquet", "support/edgar_recent.parquet"),
        ("cik_map.json",         "support/cik_map.json"),
    ]

    log.info("Pushing support files...")
    for filename, github_path in support_files:
        local_path = support_dir / filename
        if local_path.exists():
            if push_file(local_path, github_path):
                ok += 1
            else:
                fail += 1
        else:
            log.warning(f"Support file not found: {filename}")

    log.info("=" * 50)
    log.info(f"Done: {ok} pushed, {fail} failed")
    if fail == 0:
        log.info("✅ All files pushed to GitHub!")
        log.info("Scanner will have models + SI + EDGAR on next deploy.")
        log.info("  MODEL_DIR:   /opt/render/project/src/models")
        log.info("  SUPPORT_DIR: /opt/render/project/src/support")
    else:
        log.error("Some files failed — check GITHUB_TOKEN permissions")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
