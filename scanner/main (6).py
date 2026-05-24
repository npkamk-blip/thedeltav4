"""
THE DELTA v2 — main.py
========================
Single entry point. Runs 24/7 on Render.
Automatically switches between modes based on ET time:

  8PM - 4AM:   Basecamp mode — watches EDGAR for 8-K filings
  4AM - 9:30AM: Scanner mode — scans premarket movers every 60s
  9:30AM - 8PM: Resting — sleeps, sends nightly summary at 10PM
"""

import os
import time
import json
import logging
import requests
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
EDGAR_USER_AGENT   = "NPKNOB@gmail.com"

MODEL_DIR  = Path(os.environ.get("MODEL_DIR",  "/opt/render/project/src/models"))
LOG_DIR    = Path(os.environ.get("LOG_DIR",    "/tmp/logs"))
ALERT_DIR  = Path(os.environ.get("ALERT_DIR",  "/tmp/alerts"))

for d in [LOG_DIR, ALERT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MIN_PM_VOLUME = 10_000
MIN_PM_GAP    = 0.05
SCAN_INTERVAL = 60

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "delta_v2.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("delta_v2")

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
        r = requests.post(
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
        if r.status_code == 200:
            log.info(f"Pushover sent: {title}")
        else:
            log.warning(f"Pushover failed: {r.status_code}")
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
ticker_cache = {}

# ── EDGAR ─────────────────────────────────────────────────────
edgar_session = requests.Session()
edgar_session.headers.update({"User-Agent": EDGAR_USER_AGENT})

def get_recent_8k_filings():
    filings = []
    try:
        r = edgar_session.get(
            "https://efts.sec.gov/LATEST/search-index?q=%228-K%22"
            f"&dateRange=custom&startdt={date.today().isoformat()}&forms=8-K",
            timeout=20
        )
        if r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            for hit in hits:
                src = hit.get("_source", {})
                filings.append({
                    "ticker":  src.get("ticker", "").strip().upper(),
                    "company": src.get("display_names", [""])[0],
                    "cik":     src.get("entity_id", ""),
                    "filed_at": src.get("file_date", ""),
                })
    except Exception as e:
        log.debug(f"EDGAR error: {e}")
    return filings

# ── Ticker helpers ────────────────────────────────────────────
def get_snapshot(ticker):
    data = poly.get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
    if not data or "ticker" not in data:
        return None
    t = data["ticker"]
    day  = t.get("day", {})
    prev = t.get("prevDay", {})
    return {
        "ticker":     ticker,
        "price":      day.get("c", 0),
        "open":       day.get("o", 0),
        "high":       day.get("h", 0),
        "low":        day.get("l", 0),
        "volume":     day.get("v", 0),
        "prev_close": prev.get("c", 0),
        "change_pct": t.get("todaysChangePerc", 0),
        "vw":         day.get("vw", 0),
    }

def get_details(ticker):
    if ticker in ticker_cache:
        return ticker_cache[ticker]
    data = poly.get(f"/v3/reference/tickers/{ticker}")
    details = {"float_shares": -1, "float_M": -1, "float_tier": -1,
               "market_cap": -1, "is_foreign_listed": 0}
    if data and "results" in data:
        r = data["results"]
        shares  = r.get("share_class_shares_outstanding", 0) or 0
        float_M = shares / 1_000_000 if shares else -1
        ft = -1
        if float_M > 0:
            if float_M < 5:    ft = 0
            elif float_M < 15: ft = 1
            elif float_M < 50: ft = 2
            elif float_M < 200: ft = 3
            else:              ft = 4
        details = {
            "float_shares":     shares,
            "float_M":          float_M,
            "float_tier":       ft,
            "market_cap":       r.get("market_cap", -1) or -1,
            "is_foreign_listed": 1 if r.get("locale", "us") != "us" else 0,
        }
    ticker_cache[ticker] = details
    return details

# ── Feature builder ───────────────────────────────────────────
def build_features(snap, details, feature_cols, has_8k=0):
    prev_close = snap.get("prev_close", 0) or 0
    pm_open    = snap.get("open", 0) or 0
    pm_high    = snap.get("high", 0) or 0
    pm_low     = snap.get("low", 0) or 0
    pm_close   = snap.get("price", 0) or 0
    pm_volume  = snap.get("volume", 0) or 0
    change_pct = (snap.get("change_pct", 0) or 0) / 100

    pm_gap_pct  = (pm_open - prev_close) / prev_close if prev_close > 0 else 0
    pm_move_pct = (pm_high - pm_open) / pm_open if pm_open > 0 else 0

    seed_tgt  = prev_close * 2.0
    super_tgt = prev_close * 6.0
    mega_tgt  = prev_close * 11.0
    rem_seed  = (seed_tgt  - pm_close) / pm_close if pm_close > 0 else 0
    rem_super = (super_tgt - pm_close) / pm_close if pm_close > 0 else 0
    rem_mega  = (mega_tgt  - pm_close) / pm_close if pm_close > 0 else 0

    features = {col: 0 for col in feature_cols}
    overrides = {
        "prev_close":             prev_close,
        "pm_open":                pm_open,
        "pm_high":                pm_high,
        "pm_low":                 pm_low,
        "pm_close":               pm_close,
        "pm_volume":              pm_volume,
        "pm_gap_pct":             pm_gap_pct,
        "pm_move_pct":            pm_move_pct,
        "pm_remaining_to_seed":   rem_seed,
        "pm_remaining_to_super":  rem_super,
        "pm_remaining_to_mega":   rem_mega,
        "pm_high_of_session":     1 if pm_close >= pm_high * 0.99 else 0,
        "pm_fade":                1 if (pm_high > pm_open and
                                  (pm_high - pm_close) > (pm_high - pm_open) * 0.10) else 0,
        "float_shares":           details.get("float_shares", -1),
        "float_M":                details.get("float_M", -1),
        "float_tier":             details.get("float_tier", -1),
        "market_cap":             details.get("market_cap", -1),
        "is_foreign_listed":      details.get("is_foreign_listed", 0),
        "has_8k":                 has_8k,
        "ah_move_pct":            change_pct if change_pct != 0 else 0,
        "ah_direction":           1 if change_pct > 0 else (-1 if change_pct < 0 else 0),
        "si_pct":                 -1,
        "si_tier":                -1,
        "days_to_cover":          -1,
        "days_since_last_seed":   999,
        "days_since_last_8k":     0 if has_8k else 999,
        "days_since_dilution":    999,
    }
    for k, v in overrides.items():
        if k in features:
            features[k] = v

    return np.array([features[col] for col in feature_cols]).reshape(1, -1)

# ── Scorer ────────────────────────────────────────────────────
def score(fvec, models, thresholds):
    scores = {}
    for name, model in models.items():
        try:
            scores[name] = round(float(model.predict_proba(fvec)[0][1]), 4)
        except Exception:
            scores[name] = 0.0

    alerts = []
    if scores.get("mega", 0)  >= thresholds.get("mega",  0.5): alerts.append(("mega",  scores["mega"]))
    elif scores.get("super", 0) >= thresholds.get("super", 0.5): alerts.append(("super", scores["super"]))
    elif scores.get("seed", 0)  >= thresholds.get("seed",  0.5): alerts.append(("seed",  scores["seed"]))
    return scores, alerts

# ── Alert tracking ────────────────────────────────────────────
def already_alerted(ticker, mode="scanner"):
    f = ALERT_DIR / f"{date.today().isoformat()}_{mode}.json"
    if not f.exists(): return False
    with open(f) as fp: return ticker in json.load(fp)

def log_alert(ticker, alert_type, score_val, scores, snap, mode="scanner"):
    f = ALERT_DIR / f"{date.today().isoformat()}_{mode}.json"
    alerts = {}
    if f.exists():
        with open(f) as fp: alerts = json.load(fp)
    alerts[ticker] = {
        "type": alert_type, "score": score_val,
        "time": datetime.now(ET).isoformat(),
        "prev_close": snap.get("prev_close"),
        "price": snap.get("price"),
        "change_pct": snap.get("change_pct"),
        "all_scores": scores,
    }
    with open(f, "w") as fp: json.dump(alerts, fp, indent=2)

# ── Format alert message ──────────────────────────────────────
def format_alert(ticker, alert_type, score_val, scores, snap, details):
    emoji  = {"seed": "🌱", "super": "🚀", "mega": "💥"}.get(alert_type, "🌱")
    price  = snap.get("price", 0)
    pct    = snap.get("change_pct", 0)
    prev   = snap.get("prev_close", 0)
    vol    = snap.get("volume", 0)
    flt    = details.get("float_M", -1)
    rem    = ((prev * 2 - price) / price * 100) if price > 0 and prev > 0 else 0

    title = f"{emoji} {alert_type.upper()} — {ticker}"
    msg = (
        f"Price: ${price:.2f} ({pct:+.1f}%)\n"
        f"Room to seed: {rem:+.1f}%\n"
        f"Float: {flt:.1f}M\n"
        f"Volume: {vol:,.0f}\n"
        f"Score: {score_val:.3f}\n"
        f"S:{scores.get('seed',0):.2f} "
        f"Su:{scores.get('super',0):.2f} "
        f"M:{scores.get('mega',0):.2f}"
    )
    return title, msg

# ══════════════════════════════════════════════════════════════
# BASECAMP MODE — 8PM to 4AM
# ══════════════════════════════════════════════════════════════
def run_basecamp_scan(models, feature_cols, thresholds):
    log.info("Basecamp: scanning EDGAR for 8-K filings...")
    filings = get_recent_8k_filings()
    log.info(f"Basecamp: {len(filings)} filings found")

    for filing in filings:
        ticker = filing.get("ticker", "").strip().upper()
        if not ticker or already_alerted(ticker, "basecamp"):
            continue

        snap = get_snapshot(ticker)
        if not snap or (snap.get("prev_close") or 0) < 0.10:
            continue

        details = get_details(ticker)
        fvec    = build_features(snap, details, feature_cols, has_8k=1)
        scores, alerts = score(fvec, models, thresholds)

        if not alerts:
            continue

        alert_type, score_val = alerts[0]
        log_alert(ticker, alert_type, score_val, scores, snap, "basecamp")

        title, msg = format_alert(ticker, alert_type, score_val, scores, snap, details)
        title = "🏕️ " + title + " (overnight)"
        priority = 1 if alert_type in ("super", "mega") else 0
        send_pushover(title, msg, alert_type, priority)
        log.info(f"BASECAMP ALERT: {ticker} {alert_type.upper()} score={score_val:.3f}")

# ══════════════════════════════════════════════════════════════
# SCANNER MODE — 4AM to 9:30AM
# ══════════════════════════════════════════════════════════════
def get_premarket_movers():
    candidates = []
    data = poly.get("/v2/snapshot/locale/us/markets/stocks/gainers")
    if data and "tickers" in data:
        for t in data["tickers"]:
            day  = t.get("day", {})
            prev = t.get("prevDay", {})
            pct  = t.get("todaysChangePerc", 0)
            vol  = day.get("v", 0)
            pc   = prev.get("c", 0)
            if pc < 0.10 or vol < MIN_PM_VOLUME or abs(pct)/100 < MIN_PM_GAP:
                continue
            candidates.append({
                "ticker":     t.get("ticker", ""),
                "prev_close": pc,
                "open":       day.get("o", 0),
                "high":       day.get("h", 0),
                "low":        day.get("l", 0),
                "price":      day.get("c", 0),
                "volume":     vol,
                "change_pct": pct,
            })
    return candidates

def run_scanner_scan(models, feature_cols, thresholds):
    candidates = get_premarket_movers()
    log.info(f"Scanner: {len(candidates)} premarket movers")

    fired = 0
    for snap in candidates:
        ticker = snap["ticker"]
        if already_alerted(ticker, "scanner"):
            continue

        details = get_details(ticker)
        fvec    = build_features(snap, details, feature_cols)
        scores, alerts = score(fvec, models, thresholds)

        if not alerts:
            continue

        alert_type, score_val = alerts[0]
        log_alert(ticker, alert_type, score_val, scores, snap, "scanner")

        title, msg = format_alert(ticker, alert_type, score_val, scores, snap, details)
        priority = 1 if alert_type in ("super", "mega") else 0
        send_pushover(title, msg, alert_type, priority)
        log.info(f"SCANNER ALERT: {ticker} {alert_type.upper()} score={score_val:.3f}")
        fired += 1

    return fired

# ══════════════════════════════════════════════════════════════
# SUMMARIES
# ══════════════════════════════════════════════════════════════
def send_morning_summary(models, feature_cols, thresholds):
    candidates = get_premarket_movers()
    hot = []
    for snap in candidates[:20]:
        details = get_details(snap["ticker"])
        fvec    = build_features(snap, details, feature_cols)
        scores, _ = score(fvec, models, thresholds)
        if scores.get("seed", 0) > 0.40:
            hot.append((snap["ticker"], scores, snap))

    if not hot:
        msg = f"Quiet morning. {len(candidates)} movers, none scoring high."
    else:
        lines = [f"☀️ {len(hot)} hot setup(s) at 5AM:\n"]
        for ticker, scores, snap in sorted(hot, key=lambda x: x[1].get("seed",0), reverse=True)[:5]:
            pct = snap.get("change_pct", 0)
            lines.append(f"{ticker}: {pct:+.1f}% | S:{scores.get('seed',0):.2f} Su:{scores.get('super',0):.2f}")
        msg = "\n".join(lines)

    send_pushover("☀️ Delta v2 Morning Report", msg, "seed", priority=0)

def send_nightly_summary():
    today = date.today().isoformat()
    scanner_file  = ALERT_DIR / f"{today}_scanner.json"
    basecamp_file = ALERT_DIR / f"{today}_basecamp.json"

    scanner_alerts  = json.load(open(scanner_file))  if scanner_file.exists()  else {}
    basecamp_alerts = json.load(open(basecamp_file)) if basecamp_file.exists() else {}

    total = len(scanner_alerts) + len(basecamp_alerts)
    if total == 0:
        msg = "No alerts today."
    else:
        lines = [f"📊 Delta v2 Daily Summary — {total} alert(s)\n"]
        if scanner_alerts:
            lines.append(f"Premarket ({len(scanner_alerts)}):")
            for t, info in scanner_alerts.items():
                lines.append(f"  {t}: {info['type'].upper()} score={info['score']:.3f}")
        if basecamp_alerts:
            lines.append(f"\nOvernight ({len(basecamp_alerts)}):")
            for t, info in basecamp_alerts.items():
                lines.append(f"  {t}: {info['type'].upper()} score={info['score']:.3f}")
        msg = "\n".join(lines)

    send_pushover("📊 Delta v2 Daily Summary", msg, "seed", priority=0)

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    threading.Thread(target=start_keepalive, daemon=True).start()
    time.sleep(2)

    log.info("=" * 50)
    log.info("THE DELTA v2 — Starting")
    log.info("=" * 50)

    models, feature_cols, thresholds = load_models()
    if not models:
        log.error("No models found — check MODEL_DIR")
        return

    morning_summary_sent = False
    nightly_summary_sent = False
    last_basecamp_scan   = None
    last_date            = None

    while True:
        now    = datetime.now(ET)
        hour   = now.hour
        minute = now.minute

        # Reset daily flags
        if last_date != now.date():
            morning_summary_sent = False
            nightly_summary_sent = False
            last_basecamp_scan   = None
            ticker_cache.clear()
            last_date = now.date()
            log.info(f"New day: {now.date()}")

        # 10 PM nightly summary
        if hour == 22 and minute < 5 and not nightly_summary_sent:
            send_nightly_summary()
            nightly_summary_sent = True

        # 5 AM morning summary
        if hour == 5 and minute < 5 and not morning_summary_sent:
            send_morning_summary(models, feature_cols, thresholds)
            morning_summary_sent = True

        # BASECAMP: 8PM - 4AM
        in_basecamp = hour >= 20 or hour < 4
        if in_basecamp:
            # Scan every 10 minutes
            if last_basecamp_scan is None or (now - last_basecamp_scan).seconds >= 600:
                run_basecamp_scan(models, feature_cols, thresholds)
                last_basecamp_scan = now
            time.sleep(60)
            continue

        # SCANNER: 4AM - 9:30AM
        in_scanner = hour >= 4 and (hour < 9 or (hour == 9 and minute < 30))
        if in_scanner:
            fired = run_scanner_scan(models, feature_cols, thresholds)
            log.info(f"Scan done: {fired} alerts fired")
            time.sleep(SCAN_INTERVAL)
            continue

        # REST: 9:30AM - 8PM
        log.debug(f"Resting ({hour}:{minute:02d} ET) — next window at 8PM or 4AM")
        time.sleep(300)

if __name__ == "__main__":
    main()
