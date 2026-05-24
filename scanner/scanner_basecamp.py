"""
THE DELTA v2 — scanner_basecamp.py
=====================================
Overnight EDGAR watcher. Runs 8PM - 4AM ET.
Watches for new 8-K filings and scores them
against seed/super/mega profiles.
Fires early Pushover alerts for high-probability setups.
"""

import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import xgboost as xgb

ET = ZoneInfo("America/New_York")

# ── Config ────────────────────────────────────────────────────
POLYGON_API_KEY    = os.environ.get("MASSIVE_API_KEY", "")
PUSHOVER_USER_KEY  = os.environ.get("PUSHOVER_USER_KEY", "utvy26j5q66kae27ncwxsftfcuhi92")
PUSHOVER_APP_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN", "a3szzncpvgyevbck6z5z5yszm7nzg3")
EDGAR_USER_AGENT   = "NPKNOB@gmail.com"

MODEL_DIR  = Path(os.environ.get("MODEL_DIR", "/app/models"))
LOG_DIR    = Path(os.environ.get("LOG_DIR", "/app/logs"))
ALERT_DIR  = Path(os.environ.get("ALERT_DIR", "/app/alerts"))

for d in [LOG_DIR, ALERT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Thresholds for overnight alerts (more conservative)
BASECAMP_THRESHOLDS = {
    "seed":  0.70,
    "super": 0.60,
    "mega":  0.50,
}

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "basecamp.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("basecamp")

# ── Pushover ──────────────────────────────────────────────────
SOUNDS = {
    "seed":  "cashregister",
    "super": "siren",
    "mega":  "siren",
}

def send_pushover(title, message, alert_type="seed", priority=0):
    sound = SOUNDS.get(alert_type, "pushover")
    try:
        r = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token":    PUSHOVER_APP_TOKEN,
                "user":     PUSHOVER_USER_KEY,
                "title":    title,
                "message":  message,
                "sound":    sound,
                "priority": priority,
            },
            timeout=10
        )
        if r.status_code == 200:
            log.info(f"Pushover sent: {title}")
        else:
            log.warning(f"Pushover failed: {r.status_code}")
    except Exception as e:
        log.warning(f"Pushover error: {e}")

# ── Load models ───────────────────────────────────────────────
def load_models():
    models = {}
    feature_cols = []

    for name in ["seed", "super", "mega"]:
        path = MODEL_DIR / f"{name}_model.json"
        if path.exists():
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            models[name] = m
            log.info(f"Loaded {name}_model")

    feat_path = MODEL_DIR / "feature_cols.json"
    if feat_path.exists():
        with open(feat_path) as f:
            feature_cols = json.load(f)

    thresh_path = MODEL_DIR / "thresholds.json"
    thresholds = BASECAMP_THRESHOLDS.copy()
    if thresh_path.exists():
        with open(thresh_path) as f:
            saved = json.load(f)
            thresholds.update(saved)

    return models, feature_cols, thresholds

# ── Polygon client ────────────────────────────────────────────
class PolygonClient:
    def __init__(self):
        self._last = 0.0
        self.session = requests.Session()
        self.session.params = {"apiKey": POLYGON_API_KEY}

    def get(self, path, params=None):
        elapsed = time.time() - self._last
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)
        self._last = time.time()
        try:
            r = self.session.get(
                f"https://api.polygon.io{path}",
                params=params or {},
                timeout=15
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log.debug(f"Polygon error: {e}")
        return None

poly = PolygonClient()

# ── EDGAR client ──────────────────────────────────────────────
edgar_session = requests.Session()
edgar_session.headers.update({"User-Agent": EDGAR_USER_AGENT})

def get_recent_8k_filings(since_minutes=60):
    """Pull 8-K filings from EDGAR in the last N minutes."""
    filings = []
    try:
        # EDGAR RSS feed for recent filings
        r = edgar_session.get(
            "https://efts.sec.gov/LATEST/search-index?q=%228-K%22&dateRange=custom"
            f"&startdt={date.today().isoformat()}&forms=8-K",
            timeout=20
        )
        if r.status_code == 200:
            data = r.json()
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits:
                src = hit.get("_source", {})
                filed_at = src.get("file_date", "")
                filings.append({
                    "ticker":    src.get("ticker", ""),
                    "company":   src.get("display_names", [""])[0],
                    "cik":       src.get("entity_id", ""),
                    "filed_at":  filed_at,
                    "form_type": src.get("form_type", ""),
                })
    except Exception as e:
        log.debug(f"EDGAR RSS error: {e}")

    return filings

def get_ticker_snapshot(ticker):
    """Get current price and basic stats for a ticker."""
    data = poly.get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
    if not data or "ticker" not in data:
        return None
    t = data["ticker"]
    day = t.get("day", {})
    prev = t.get("prevDay", {})
    return {
        "ticker":      ticker,
        "price":       day.get("c", 0),
        "open":        day.get("o", 0),
        "high":        day.get("h", 0),
        "volume":      day.get("v", 0),
        "prev_close":  prev.get("c", 0),
        "change_pct":  t.get("todaysChangePerc", 0),
    }

def get_ticker_details(ticker):
    """Get float and market cap for a ticker."""
    data = poly.get(f"/v3/reference/tickers/{ticker}")
    if not data or "results" not in data:
        return {}
    r = data["results"]
    shares = r.get("share_class_shares_outstanding", 0) or 0
    return {
        "float_shares": shares,
        "float_M":      shares / 1_000_000 if shares else -1,
        "market_cap":   r.get("market_cap", -1) or -1,
        "is_foreign":   1 if r.get("locale", "us") != "us" else 0,
    }

# ── Build feature vector for scoring ─────────────────────────
def build_overnight_features(ticker, snapshot, details, filing, feature_cols):
    """
    Build a feature vector for overnight scoring.
    Many features will be -1/0 (not yet available).
    Available: T-1 price, EDGAR flags, float, market cap.
    """
    prev_close = snapshot.get("prev_close", 0) or 0
    float_M    = details.get("float_M", -1)

    # Classify float tier
    float_tier = -1
    if float_M > 0:
        if float_M < 5:   float_tier = 0   # nano
        elif float_M < 15: float_tier = 1  # micro
        elif float_M < 50: float_tier = 2  # small
        elif float_M < 200: float_tier = 3 # mid
        else:              float_tier = 4  # large

    # AH move if market is closed
    ah_pct = snapshot.get("change_pct", 0) / 100 if snapshot.get("change_pct") else 0

    # Build base features dict with defaults
    features = {col: 0 for col in feature_cols}

    # Fill what we know
    overrides = {
        "prev_close":       prev_close,
        "prev_volume":      snapshot.get("volume", 0),
        "float_shares":     details.get("float_shares", -1),
        "float_M":          float_M,
        "float_tier":       float_tier,
        "market_cap":       details.get("market_cap", -1),
        "is_foreign_listed": details.get("is_foreign", 0),
        "has_8k":           1,
        "has_8k_yesterday": 0,
        "has_8k_2days_ago": 0,
        "ah_move_pct":      ah_pct,
        "ah_direction":     1 if ah_pct > 0 else (-1 if ah_pct < 0 else 0),
        "ah_volume":        snapshot.get("volume", 0),
        "days_since_last_8k": 0,
        "si_pct":           -1,
        "si_tier":          -1,
        "days_to_cover":    -1,
        "days_since_last_seed": 999,
        "days_since_dilution":  999,
    }

    for k, v in overrides.items():
        if k in features:
            features[k] = v

    return np.array([features[col] for col in feature_cols]).reshape(1, -1)

# ── Score a ticker ────────────────────────────────────────────
def score_ticker(ticker, feature_vec, models, thresholds):
    results = {}
    for name, model in models.items():
        try:
            proba = model.predict_proba(feature_vec)[0][1]
            results[name] = round(float(proba), 4)
        except Exception:
            results[name] = 0.0

    # Check if any model fires
    alerts = []
    if results.get("mega", 0) >= thresholds.get("mega", 0.5):
        alerts.append(("mega", results["mega"]))
    elif results.get("super", 0) >= thresholds.get("super", 0.5):
        alerts.append(("super", results["super"]))
    elif results.get("seed", 0) >= thresholds.get("seed", 0.5):
        alerts.append(("seed", results["seed"]))

    return results, alerts

# ── Check if already alerted today ───────────────────────────
def already_alerted(ticker):
    today = date.today().isoformat()
    alert_file = ALERT_DIR / f"{today}_basecamp.json"
    if not alert_file.exists():
        return False
    with open(alert_file) as f:
        alerts = json.load(f)
    return ticker in alerts

def log_alert(ticker, alert_type, score):
    today = date.today().isoformat()
    alert_file = ALERT_DIR / f"{today}_basecamp.json"
    alerts = {}
    if alert_file.exists():
        with open(alert_file) as f:
            alerts = json.load(f)
    alerts[ticker] = {"type": alert_type, "score": score, "time": datetime.now(ET).isoformat()}
    with open(alert_file, "w") as f:
        json.dump(alerts, f, indent=2)

# ── Nightly summary ───────────────────────────────────────────
def send_nightly_summary():
    today = date.today().isoformat()
    alert_file = ALERT_DIR / f"{today}_basecamp.json"

    if not alert_file.exists():
        msg = "No alerts tonight. Clean slate for tomorrow."
    else:
        with open(alert_file) as f:
            alerts = json.load(f)
        lines = [f"Basecamp found {len(alerts)} setup(s) tonight:\n"]
        for ticker, info in alerts.items():
            lines.append(f"  {ticker}: {info['type'].upper()} (score {info['score']})")
        msg = "\n".join(lines)

    send_pushover("🏕️ Basecamp Nightly Summary", msg, "seed", priority=0)

# ── Main loop ─────────────────────────────────────────────────
def main():
    log.info("=" * 50)
    log.info("THE DELTA v2 — Scanner Basecamp Starting")
    log.info("=" * 50)

    models, feature_cols, thresholds = load_models()
    if not models:
        log.error("No models found — check MODEL_DIR")
        return

    alerted_today = set()
    last_summary_date = None

    while True:
        now = datetime.now(ET)
        hour = now.hour

        # Send nightly summary at 10 PM
        if hour == 22 and last_summary_date != now.date():
            send_nightly_summary()
            last_summary_date = now.date()
            alerted_today = set()

        # Only scan 8PM - 4AM ET
        if not (hour >= 20 or hour < 4):
            log.debug(f"Outside scan window ({hour}:00 ET) — sleeping 5 min")
            time.sleep(300)
            continue

        log.info(f"Scanning EDGAR for new 8-K filings...")
        filings = get_recent_8k_filings(since_minutes=60)
        log.info(f"Found {len(filings)} recent 8-K filings")

        for filing in filings:
            ticker = filing.get("ticker", "").strip().upper()
            if not ticker or ticker in alerted_today:
                continue
            if already_alerted(ticker):
                continue

            # Get snapshot and details
            snapshot = get_ticker_snapshot(ticker)
            if not snapshot or snapshot.get("prev_close", 0) < 0.10:
                continue

            details = get_ticker_details(ticker)

            # Build features and score
            feature_vec = build_overnight_features(
                ticker, snapshot, details, filing, feature_cols
            )
            scores, alerts = score_ticker(ticker, feature_vec, models, thresholds)

            if not alerts:
                continue

            alert_type, score = alerts[0]
            alerted_today.add(ticker)
            log_alert(ticker, alert_type, score)

            # Build notification
            prev_close = snapshot.get("prev_close", 0)
            float_M    = details.get("float_M", -1)
            ah_pct     = snapshot.get("change_pct", 0)

            title = f"🏕️ BASECAMP {alert_type.upper()} — {ticker}"
            msg = (
                f"8-K filed tonight\n"
                f"Prev close: ${prev_close:.2f}\n"
                f"AH move: {ah_pct:+.1f}%\n"
                f"Float: {float_M:.1f}M\n"
                f"Score: {score:.3f}\n"
                f"Seed: {scores.get('seed',0):.3f} | "
                f"Super: {scores.get('super',0):.3f} | "
                f"Mega: {scores.get('mega',0):.3f}"
            )

            priority = 1 if alert_type in ("super", "mega") else 0
            send_pushover(title, msg, alert_type, priority)
            log.info(f"ALERT: {ticker} {alert_type.upper()} score={score:.3f}")

        # Sleep 10 minutes between scans
        log.info("Scan complete — sleeping 10 minutes")
        time.sleep(600)

if __name__ == "__main__":
    main()
