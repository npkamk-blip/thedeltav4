"""
THE DELTA v2 — scanner.py
===========================
Main premarket scanner. Runs 4AM - 9:30AM ET.
Every minute:
  1. Pulls all stocks with premarket activity
  2. Builds live features
  3. Scores through seed/super/mega models
  4. Fires Pushover alerts
Also sends 5AM morning summary.
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
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import xgboost as xgb

ET = ZoneInfo("America/New_York")

# ── Config ────────────────────────────────────────────────────
POLYGON_API_KEY    = os.environ.get("MASSIVE_API_KEY", "")
PUSHOVER_USER_KEY  = os.environ.get("PUSHOVER_USER_KEY", "utvy26j5q66kae27ncwxsftfcuhi92")
PUSHOVER_APP_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN", "a3szzncpvgyevbck6z5z5yszm7nzg3")

MODEL_DIR  = Path(os.environ.get("MODEL_DIR", "/opt/render/project/src/models"))
LOG_DIR    = Path(os.environ.get("LOG_DIR", "/tmp/logs"))
ALERT_DIR  = Path(os.environ.get("ALERT_DIR", "/tmp/alerts"))

for d in [LOG_DIR, ALERT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Min premarket activity to consider a ticker
MIN_PM_VOLUME = 10_000
MIN_PM_GAP    = 0.05  # 5% gap minimum

# Scan interval in seconds
SCAN_INTERVAL = 60

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scanner.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("scanner")

# ── Keepalive ─────────────────────────────────────────────────
class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"alive")
    def log_message(self, *args): pass

def start_keepalive():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), _Health).serve_forever()

# ── Pushover ──────────────────────────────────────────────────
SOUNDS = {"seed": "cashregister", "super": "siren", "mega": "siren"}

def send_pushover(title, message, alert_type="seed", priority=0):
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token":    PUSHOVER_APP_TOKEN,
                "user":     PUSHOVER_USER_KEY,
                "title":    title,
                "message":  message,
                "sound":    SOUNDS.get(alert_type, "pushover"),
                "priority": priority,
                "retry":    600 if alert_type == "mega" else 0,
                "expire":   3600 if alert_type == "mega" else 0,
            },
            timeout=10
        )
        log.info(f"Pushover sent: {title}")
    except Exception as e:
        log.warning(f"Pushover error: {e}")

# ── Load models ───────────────────────────────────────────────
def load_models():
    models = {}
    for name in ["seed", "super", "mega"]:
        path = MODEL_DIR / f"{name}_model.json"
        if path.exists():
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            models[name] = m
            log.info(f"Loaded {name}_model")

    with open(MODEL_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    thresholds = {"seed": 0.70, "super": 0.60, "mega": 0.50}
    thresh_path = MODEL_DIR / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresholds.update(json.load(f))

    return models, feature_cols, thresholds

# ── Polygon ───────────────────────────────────────────────────
class PolygonClient:
    def __init__(self):
        self._last = 0.0
        self.session = requests.Session()
        self.session.params = {"apiKey": POLYGON_API_KEY}

    def _wait(self):
        elapsed = time.time() - self._last
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)
        self._last = time.time()

    def get(self, path, params=None):
        self._wait()
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

# ── Get premarket movers ──────────────────────────────────────
def get_premarket_movers():
    """
    Pull all stocks with significant premarket activity.
    Uses Polygon snapshot gainers + custom filter.
    """
    candidates = []

    # Top gainers snapshot
    data = poly.get("/v2/snapshot/locale/us/markets/stocks/gainers")
    if data and "tickers" in data:
        for t in data["tickers"]:
            ticker    = t.get("ticker", "")
            day       = t.get("day", {})
            prev      = t.get("prevDay", {})
            change    = t.get("todaysChangePerc", 0)
            pm_vol    = day.get("v", 0)
            prev_close = prev.get("c", 0)

            if prev_close < 0.10:
                continue
            if pm_vol < MIN_PM_VOLUME:
                continue
            if abs(change) / 100 < MIN_PM_GAP:
                continue

            candidates.append({
                "ticker":      ticker,
                "prev_close":  prev_close,
                "pm_open":     day.get("o", 0),
                "pm_high":     day.get("h", 0),
                "pm_low":      day.get("l", 0),
                "pm_close":    day.get("c", 0),
                "pm_volume":   pm_vol,
                "change_pct":  change / 100,
            })

    return candidates

# ── Get ticker details ────────────────────────────────────────
ticker_cache = {}

def get_ticker_details(ticker):
    if ticker in ticker_cache:
        return ticker_cache[ticker]

    data = poly.get(f"/v3/reference/tickers/{ticker}")
    details = {}
    if data and "results" in data:
        r = data["results"]
        shares = r.get("share_class_shares_outstanding", 0) or 0
        float_M = shares / 1_000_000 if shares else -1
        float_tier = -1
        if float_M > 0:
            if float_M < 5:    float_tier = 0
            elif float_M < 15: float_tier = 1
            elif float_M < 50: float_tier = 2
            elif float_M < 200: float_tier = 3
            else:              float_tier = 4

        details = {
            "float_shares":     shares,
            "float_M":          float_M,
            "float_tier":       float_tier,
            "market_cap":       r.get("market_cap", -1) or -1,
            "is_foreign_listed": 1 if r.get("locale", "us") != "us" else 0,
        }

    ticker_cache[ticker] = details
    return details

# ── Build feature vector ──────────────────────────────────────
def build_features(candidate, details, feature_cols):
    prev_close = candidate.get("prev_close", 0) or 0
    pm_close   = candidate.get("pm_close", 0) or 0
    pm_open    = candidate.get("pm_open", 0) or 0
    pm_high    = candidate.get("pm_high", 0) or 0
    pm_volume  = candidate.get("pm_volume", 0) or 0

    # Calculate PM features
    pm_gap_pct  = (pm_open - prev_close) / prev_close if prev_close > 0 else 0
    pm_move_pct = (pm_high - pm_open) / pm_open if pm_open > 0 else 0

    # Remaining upside
    seed_target  = prev_close * 2.0
    super_target = prev_close * 6.0
    mega_target  = prev_close * 11.0
    pm_remaining_to_seed  = (seed_target  - pm_close) / pm_close if pm_close > 0 else 0
    pm_remaining_to_super = (super_target - pm_close) / pm_close if pm_close > 0 else 0
    pm_remaining_to_mega  = (mega_target  - pm_close) / pm_close if pm_close > 0 else 0

    features = {col: 0 for col in feature_cols}

    overrides = {
        "prev_close":             prev_close,
        "pm_open":                pm_open,
        "pm_high":                pm_high,
        "pm_low":                 candidate.get("pm_low", 0),
        "pm_close":               pm_close,
        "pm_volume":              pm_volume,
        "pm_gap_pct":             pm_gap_pct,
        "pm_move_pct":            pm_move_pct,
        "pm_remaining_to_seed":   pm_remaining_to_seed,
        "pm_remaining_to_super":  pm_remaining_to_super,
        "pm_remaining_to_mega":   pm_remaining_to_mega,
        "pm_high_of_session":     1 if pm_close >= pm_high * 0.99 else 0,
        "pm_fade":                1 if (pm_high > pm_open and
                                        (pm_high - pm_close) > (pm_high - pm_open) * 0.10) else 0,
        "float_shares":           details.get("float_shares", -1),
        "float_M":                details.get("float_M", -1),
        "float_tier":             details.get("float_tier", -1),
        "market_cap":             details.get("market_cap", -1),
        "is_foreign_listed":      details.get("is_foreign_listed", 0),
        "si_pct":                 -1,
        "si_tier":                -1,
        "days_to_cover":          -1,
        "days_since_last_seed":   999,
        "days_since_last_8k":     999,
        "days_since_dilution":    999,
    }

    for k, v in overrides.items():
        if k in features:
            features[k] = v

    return np.array([features[col] for col in feature_cols]).reshape(1, -1)

# ── Score ticker ──────────────────────────────────────────────
def score_ticker(feature_vec, models, thresholds):
    scores = {}
    for name, model in models.items():
        try:
            scores[name] = round(float(model.predict_proba(feature_vec)[0][1]), 4)
        except Exception:
            scores[name] = 0.0

    alerts = []
    if scores.get("mega", 0) >= thresholds.get("mega", 0.5):
        alerts.append(("mega", scores["mega"]))
    elif scores.get("super", 0) >= thresholds.get("super", 0.5):
        alerts.append(("super", scores["super"]))
    elif scores.get("seed", 0) >= thresholds.get("seed", 0.5):
        alerts.append(("seed", scores["seed"]))

    return scores, alerts

# ── Alert tracking ────────────────────────────────────────────
def already_alerted(ticker):
    today = date.today().isoformat()
    f = ALERT_DIR / f"{today}_scanner.json"
    if not f.exists():
        return False
    with open(f) as fp:
        return ticker in json.load(fp)

def log_alert(ticker, alert_type, score, candidate, scores):
    today = date.today().isoformat()
    f = ALERT_DIR / f"{today}_scanner.json"
    alerts = {}
    if f.exists():
        with open(f) as fp:
            alerts = json.load(fp)
    alerts[ticker] = {
        "type":       alert_type,
        "score":      score,
        "time":       datetime.now(ET).isoformat(),
        "prev_close": candidate.get("prev_close"),
        "pm_close":   candidate.get("pm_close"),
        "pm_volume":  candidate.get("pm_volume"),
        "change_pct": candidate.get("change_pct"),
        "all_scores": scores,
    }
    with open(f, "w") as fp:
        json.dump(alerts, fp, indent=2)

# ── Morning summary ───────────────────────────────────────────
def send_morning_summary(models, feature_cols, thresholds):
    """5 AM summary of overnight setups and top premarket movers."""
    candidates = get_premarket_movers()
    log.info(f"Morning summary: {len(candidates)} premarket movers")

    hot = []
    for c in candidates[:20]:
        ticker  = c["ticker"]
        details = get_ticker_details(ticker)
        fvec    = build_features(c, details, feature_cols)
        scores, alerts = score_ticker(fvec, models, thresholds)
        if scores.get("seed", 0) > 0.40:
            hot.append((ticker, scores, c))

    if not hot:
        msg = f"Quiet morning. {len(candidates)} stocks moving premarket, none scoring high."
    else:
        lines = [f"☀️ {len(hot)} hot setup(s) at 5AM:\n"]
        for ticker, scores, c in sorted(hot, key=lambda x: x[1].get("seed", 0), reverse=True)[:5]:
            pct = c.get("change_pct", 0) * 100
            lines.append(
                f"{ticker}: +{pct:.1f}% | "
                f"Seed:{scores.get('seed',0):.2f} "
                f"Super:{scores.get('super',0):.2f}"
            )
        msg = "\n".join(lines)

    send_pushover("☀️ Delta v2 Morning Summary", msg, "seed", priority=0)

# ── Main scan loop ────────────────────────────────────────────
def scan_once(models, feature_cols, thresholds):
    candidates = get_premarket_movers()
    log.info(f"Scanning {len(candidates)} premarket movers...")

    fired = 0
    for candidate in candidates:
        ticker = candidate["ticker"]

        if already_alerted(ticker):
            continue

        details  = get_ticker_details(ticker)
        fvec     = build_features(candidate, details, feature_cols)
        scores, alerts = score_ticker(fvec, models, thresholds)

        if not alerts:
            continue

        alert_type, score = alerts[0]
        log_alert(ticker, alert_type, score, candidate, scores)

        prev_close = candidate.get("prev_close", 0)
        pm_close   = candidate.get("pm_close", 0)
        pm_volume  = candidate.get("pm_volume", 0)
        change_pct = candidate.get("change_pct", 0) * 100
        float_M    = details.get("float_M", -1)

        seed_target = prev_close * 2.0
        remaining   = ((seed_target - pm_close) / pm_close * 100) if pm_close > 0 else 0

        emoji = {"seed": "🌱", "super": "🚀", "mega": "💥"}.get(alert_type, "🌱")

        title = f"{emoji} {alert_type.upper()} — {ticker}"
        msg = (
            f"Price: ${pm_close:.2f} ({change_pct:+.1f}%)\n"
            f"Room to seed: {remaining:+.1f}%\n"
            f"Float: {float_M:.1f}M\n"
            f"PM Volume: {pm_volume:,.0f}\n"
            f"Score: {score:.3f}\n"
            f"Seed:{scores.get('seed',0):.3f} "
            f"Super:{scores.get('super',0):.3f} "
            f"Mega:{scores.get('mega',0):.3f}"
        )

        priority = 1 if alert_type in ("super", "mega") else 0
        send_pushover(title, msg, alert_type, priority)
        log.info(f"ALERT: {ticker} {alert_type.upper()} score={score:.3f} pm={change_pct:+.1f}%")
        fired += 1

    return fired

# ── Main ──────────────────────────────────────────────────────
def main():
    # Keepalive for Render
    t = threading.Thread(target=start_keepalive, daemon=True)
    t.start()

    log.info("=" * 50)
    log.info("THE DELTA v2 — Scanner Starting")
    log.info("=" * 50)

    models, feature_cols, thresholds = load_models()
    if not models:
        log.error("No models found")
        return

    morning_summary_sent = False
    last_scan_date = None

    while True:
        now  = datetime.now(ET)
        hour = now.hour
        minute = now.minute

        # Reset daily state
        if last_scan_date != now.date():
            morning_summary_sent = False
            ticker_cache.clear()
            last_scan_date = now.date()
            log.info(f"New trading day: {now.date()}")

        # 5 AM morning summary
        if hour == 5 and minute < 5 and not morning_summary_sent:
            send_morning_summary(models, feature_cols, thresholds)
            morning_summary_sent = True

        # Active scan window: 4 AM - 9:30 AM ET
        in_window = (hour >= 4) and (hour < 9 or (hour == 9 and minute < 30))

        if in_window:
            fired = scan_once(models, feature_cols, thresholds)
            log.info(f"Scan complete: {fired} alerts fired")
            time.sleep(SCAN_INTERVAL)
        else:
            # Outside window — sleep longer
            if hour >= 9 and minute >= 30 and hour < 20:
                log.debug("Market hours — scanner resting until 4AM")
                time.sleep(1800)  # 30 min
            else:
                log.debug("Overnight — scanner resting")
                time.sleep(300)   # 5 min

if __name__ == "__main__":
    main()
