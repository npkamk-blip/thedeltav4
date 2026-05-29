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
    log.info(f"Files:     {FILES_TO_PUSH}")

    ok = 0
    fail = 0

    for filename in FILES_TO_PUSH:
        local_path  = MODEL_DIR / filename
        github_path = f"models/{filename}"
        if push_file(local_path, github_path):
            ok += 1
        else:
            fail += 1

    log.info("=" * 50)
    log.info(f"Done: {ok} pushed, {fail} failed")
    if fail == 0:
        log.info("✅ All models pushed to GitHub!")
        log.info("Scanner service will deploy them automatically on next push.")
        log.info("MODEL_DIR in scanner should be: /opt/render/project/src/models")
    else:
        log.error("Some files failed — check GITHUB_TOKEN permissions")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
