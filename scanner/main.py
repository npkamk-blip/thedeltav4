"""
THE DELTA v2 — main.py (v2)
============================
Single entry point. Runs 24/7 on Render.

Uses BULK snapshot — one API call gets ALL 8,000 stocks.
Catches stocks at +5-8% not +35% like gainers endpoint.

Schedule (ET):
  4AM  - 9:30AM:  Premarket scanner (bulk snapshot every 60s)
  9:30AM - 11AM:  Early market scanner (bulk snapshot every 60s)
  11AM - 4PM:     Sleep
  4PM  - 8PM:     AH scanner (bulk snapshot every 60s)
  8PM  - 4AM:     Basecamp (EDGAR watch every 10min)
  5AM:            Morning summary
  10PM:           Nightly summary
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

# Filters
MIN_PRICE     = 0.10
MIN_VOLUME    = 10_000
MIN_GAP       = 0.05   # 5% minimum move from prev_close
SCAN_INTERVAL = 60     # seconds between scans

# Thresholds
THRESHOLDS = {"seed": 0.70, "super": 0.60, "mega": 0.50}

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

    thresholds = THRESHOLDS.copy()
    thresh_path = MODEL_DIR / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresholds.update(json.load(f))

    return models, feature_cols, thresholds

# ── Polygon ───────────────────────────────────────────────────
poly_session = requests.Session()
poly_session.params = {"apiKey": POLYGON_API_KEY}
_last_call = 0.0

def poly_get(path, params=None):
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 0.2:
        time.sleep(0.2 - elapsed)
    _last_call = time.time()
    try:
        r = poly_session.get(
            f"https://api.polygon.io{path}",
            params=params or {},
            timeout=20
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.debug(f"Polygon error: {e}")
    return None

# ── Get real prev close (handles holidays) ───────────────────
_prev_close_cache = {}

def get_prev_close(ticker):
    """
    Get the most recent valid close price for a ticker.
    Falls back to fetching daily bars if prevDay.c is 0.
    """
    if ticker in _prev_close_cache:
        return _prev_close_cache[ticker]

    # Walk back up to 5 days to find last valid close
    check_date = date.today() - timedelta(days=1)
    for _ in range(5):
        if check_date.weekday() < 5:  # weekday
            data = poly_get(f"/v1/open-close/{ticker}/{check_date.isoformat()}")
            if data and data.get("status") == "OK":
                close = data.get("close", 0) or 0
                if close > 0:
                    _prev_close_cache[ticker] = close
                    return close
        check_date -= timedelta(days=1)

    return 0

# ── BULK SNAPSHOT — the key function ─────────────────────────
def get_bulk_candidates(use_today_close=False):
    """
    ONE API call → ALL ~8,000 stocks snapshot.
    Filter locally for gap + volume.
    Returns candidates at +5% not +35%.
    use_today_close=True for AH mode (compare to today close, not yesterday)
    """
    data = poly_get(
        "/v2/snapshot/locale/us/markets/stocks/tickers",
        {"include_otc": "false"}
    )

    if not data or "tickers" not in data:
        log.warning("Bulk snapshot returned no data")
        return []

    candidates = []
    for t in data["tickers"]:
        ticker = t.get("ticker", "")
        day    = t.get("day", {})
        prev   = t.get("prevDay", {})

        prev_close     = prev.get("c", 0) or 0
        pm_close       = day.get("c", 0) or 0
        pm_open        = day.get("o", 0) or 0
        pm_high        = day.get("h", 0) or 0
        pm_low         = day.get("l", 0) or 0
        volume         = day.get("v", 0) or 0
        vwap           = day.get("vw", 0) or 0
        change_perc    = t.get("todaysChangePerc", 0) or 0

        # Apply filters
        if volume < MIN_VOLUME:
            continue

        # AH mode — compare to today's close not yesterday's
        if use_today_close:
            today_close = day.get("c", 0) or 0
            ref_price = today_close if today_close > 0 else prev_close
        else:
            ref_price = prev_close

        # If ref_price is 0 (holiday/weekend), fetch real prev close
        if ref_price <= 0 and pm_close > 0:
            ref_price = get_prev_close(ticker)

        if ref_price <= 0 or ref_price < MIN_PRICE:
            continue

        # Calculate gap from reference price
        gap = (pm_close - ref_price) / ref_price
        if gap < MIN_GAP:
            continue

        prev_close = ref_price  # use as prev_close for feature building

        candidates.append({
            "ticker":     ticker,
            "prev_close": prev_close,
            "pm_open":    pm_open,
            "pm_high":    pm_high,
            "pm_low":     pm_low,
            "pm_close":   pm_close,
            "volume":     volume,
            "vwap":       vwap,
            "gap_pct":    gap,
            "change_pct": t.get("todaysChangePerc", 0),
        })

    # Sort by gap descending
    candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
    return candidates

# ── Ticker details cache ──────────────────────────────────────
ticker_cache = {}

def get_details(ticker):
    if ticker in ticker_cache:
        return ticker_cache[ticker]
    data = poly_get(f"/v3/reference/tickers/{ticker}")
    details = {"float_shares": -1, "float_M": -1, "float_tier": -1,
               "market_cap": -1, "is_foreign_listed": 0}
    if data and "results" in data:
        r = data["results"]
        shares  = r.get("share_class_shares_outstanding", 0) or 0
        float_M = shares / 1_000_000 if shares else -1
        ft = -1
        if float_M > 0:
            if float_M < 5:     ft = 0
            elif float_M < 15:  ft = 1
            elif float_M < 50:  ft = 2
            elif float_M < 200: ft = 3
            else:               ft = 4
        details = {
            "float_shares":      shares,
            "float_M":           float_M,
            "float_tier":        ft,
            "market_cap":        r.get("market_cap", -1) or -1,
            "is_foreign_listed": 1 if r.get("locale", "us") != "us" else 0,
        }
    ticker_cache[ticker] = details
    return details

# ── Build features ────────────────────────────────────────────
def build_features(snap, details, feature_cols, has_8k=0):
    prev_close = snap.get("prev_close", 0) or 0
    pm_open    = snap.get("pm_open", 0) or 0
    pm_high    = snap.get("pm_high", 0) or 0
    pm_low     = snap.get("pm_low", 0) or 0
    pm_close   = snap.get("pm_close", 0) or 0
    volume     = snap.get("volume", 0) or 0
    gap_pct    = snap.get("gap_pct", 0) or 0

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
        "pm_volume":              volume,
        "pm_gap_pct":             gap_pct,
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
        "days_since_last_8k":     0 if has_8k else 999,
        "si_pct":                 -1,
        "si_tier":                -1,
        "days_to_cover":          -1,
        "days_since_last_seed":   999,
        "days_since_dilution":    999,
    }

    for k, v in overrides.items():
        if k in features:
            features[k] = v

    return np.array([features[col] for col in feature_cols]).reshape(1, -1)

# ── Score ─────────────────────────────────────────────────────
def score(fvec, models, thresholds):
    scores = {}
    for name, model in models.items():
        try:
            scores[name] = round(float(model.predict_proba(fvec)[0][1]), 4)
        except Exception:
            scores[name] = 0.0

    alerts = []
    if scores.get("mega", 0)  >= thresholds.get("mega",  0.5):
        alerts.append(("mega",  scores["mega"]))
    elif scores.get("super", 0) >= thresholds.get("super", 0.5):
        alerts.append(("super", scores["super"]))
    elif scores.get("seed", 0)  >= thresholds.get("seed",  0.5):
        alerts.append(("seed",  scores["seed"]))

    return scores, alerts

# ── Alert tracking ────────────────────────────────────────────
def already_alerted(ticker, mode="scanner"):
    f = ALERT_DIR / f"{date.today().isoformat()}_{mode}.json"
    if not f.exists():
        return False
    with open(f) as fp:
        return ticker in json.load(fp)

def log_alert(ticker, alert_type, score_val, scores, snap, mode="scanner"):
    f = ALERT_DIR / f"{date.today().isoformat()}_{mode}.json"
    alerts = {}
    if f.exists():
        with open(f) as fp:
            alerts = json.load(fp)
    alerts[ticker] = {
        "type":       alert_type,
        "score":      score_val,
        "time":       datetime.now(ET).isoformat(),
        "prev_close": snap.get("prev_close"),
        "price":      snap.get("pm_close"),
        "gap_pct":    snap.get("gap_pct"),
        "volume":     snap.get("volume"),
        "all_scores": scores,
    }
    with open(f, "w") as fp:
        json.dump(alerts, fp, indent=2)

# ── Format alert ──────────────────────────────────────────────
def format_alert(ticker, alert_type, score_val, scores, snap, details, mode=""):
    emoji  = {"seed": "🌱", "super": "🚀", "mega": "💥"}.get(alert_type, "🌱")
    price  = snap.get("pm_close", 0)
    gap    = snap.get("gap_pct", 0) * 100
    prev   = snap.get("prev_close", 0)
    vol    = snap.get("volume", 0)
    flt    = details.get("float_M", -1)
    rem    = ((prev * 2 - price) / price * 100) if price > 0 and prev > 0 else 0

    mode_tag = f" ({mode})" if mode else ""
    title = f"{emoji} {alert_type.upper()} — {ticker}{mode_tag}"
    msg = (
        f"Price: ${price:.2f} ({gap:+.1f}% from close)\n"
        f"Room to 100%: {rem:+.1f}%\n"
        f"Float: {flt:.1f}M\n"
        f"Volume: {vol:,.0f}\n"
        f"Score: {score_val:.3f}\n"
        f"S:{scores.get('seed',0):.2f} "
        f"Su:{scores.get('super',0):.2f} "
        f"M:{scores.get('mega',0):.2f}"
    )
    return title, msg

# ══════════════════════════════════════════════════════════════
# MAIN SCAN — used for premarket, early market, and AH
# ══════════════════════════════════════════════════════════════
def run_scan(models, feature_cols, thresholds, mode="premarket", use_today_close=False):
    candidates = get_bulk_candidates(use_today_close=use_today_close)
    log.info(f"{mode}: {len(candidates)} candidates (gap >= {MIN_GAP*100:.0f}%)")

    fired = 0
    for snap in candidates:
        ticker = snap["ticker"]

        if already_alerted(ticker, mode):
            continue

        details = get_details(ticker)
        fvec    = build_features(snap, details, feature_cols)
        scores, alerts = score(fvec, models, thresholds)

        if not alerts:
            continue

        alert_type, score_val = alerts[0]
        log_alert(ticker, alert_type, score_val, scores, snap, mode)

        title, msg = format_alert(
            ticker, alert_type, score_val, scores, snap, details, mode
        )
        priority = 1 if alert_type in ("super", "mega") else 0
        send_pushover(title, msg, alert_type, priority)

        log.info(
            f"ALERT [{mode}]: {ticker} {alert_type.upper()} "
            f"score={score_val:.3f} gap={snap['gap_pct']*100:+.1f}%"
        )
        fired += 1

    return fired

# ══════════════════════════════════════════════════════════════
# BASECAMP — EDGAR overnight watch
# ══════════════════════════════════════════════════════════════
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
                ticker = src.get("ticker", "").strip().upper()
                if ticker:
                    filings.append({
                        "ticker":   ticker,
                        "filed_at": src.get("file_date", ""),
                    })
    except Exception as e:
        log.debug(f"EDGAR error: {e}")
    return filings

def run_basecamp(models, feature_cols, thresholds):
    log.info("Basecamp: scanning EDGAR...")
    filings = get_recent_8k_filings()
    log.info(f"Basecamp: {len(filings)} 8-K filings today")

    fired = 0
    for filing in filings:
        ticker = filing["ticker"]
        if already_alerted(ticker, "basecamp"):
            continue

        # Get snapshot
        data = poly_get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
        if not data or "ticker" not in data:
            continue

        t    = data["ticker"]
        day  = t.get("day", {})
        prev = t.get("prevDay", {})

        prev_close = prev.get("c", 0) or 0
        if prev_close < MIN_PRICE:
            continue

        snap = {
            "ticker":     ticker,
            "prev_close": prev_close,
            "pm_open":    day.get("o", 0),
            "pm_high":    day.get("h", 0),
            "pm_low":     day.get("l", 0),
            "pm_close":   day.get("c", 0) or prev_close,
            "volume":     day.get("v", 0),
            "gap_pct":    t.get("todaysChangePerc", 0) / 100,
        }

        details = get_details(ticker)
        fvec    = build_features(snap, details, feature_cols, has_8k=1)
        scores, alerts = score(fvec, models, thresholds)

        if not alerts:
            continue

        alert_type, score_val = alerts[0]
        log_alert(ticker, alert_type, score_val, scores, snap, "basecamp")

        title, msg = format_alert(
            ticker, alert_type, score_val, scores, snap, details, "overnight 8-K"
        )
        priority = 1 if alert_type in ("super", "mega") else 0
        send_pushover(title, msg, alert_type, priority)

        log.info(
            f"BASECAMP ALERT: {ticker} {alert_type.upper()} score={score_val:.3f}"
        )
        fired += 1

    return fired

# ══════════════════════════════════════════════════════════════
# SUMMARIES
# ══════════════════════════════════════════════════════════════
def send_morning_summary(models, feature_cols, thresholds):
    candidates = get_bulk_candidates()
    hot = []
    for snap in candidates[:50]:
        details = get_details(snap["ticker"])
        fvec    = build_features(snap, details, feature_cols)
        scores, _ = score(fvec, models, thresholds)
        if scores.get("seed", 0) > 0.40:
            hot.append((snap["ticker"], scores, snap))

    if not hot:
        msg = (f"Quiet morning. {len(candidates)} stocks "
               f"gapping 5%+, none scoring high yet.")
    else:
        lines = [f"☀️ {len(hot)} setup(s) scoring above 0.40:\n"]
        for ticker, sc, snap in sorted(
            hot, key=lambda x: x[1].get("seed", 0), reverse=True
        )[:5]:
            gap = snap.get("gap_pct", 0) * 100
            lines.append(
                f"{ticker}: {gap:+.1f}% | "
                f"S:{sc.get('seed',0):.2f} Su:{sc.get('super',0):.2f}"
            )
        msg = "\n".join(lines)

    send_pushover("☀️ Delta v2 Morning Report", msg, "seed", priority=0)

def send_nightly_summary():
    today   = date.today().isoformat()
    all_alerts = {}

    for mode in ["premarket", "early_market", "ah", "basecamp"]:
        f = ALERT_DIR / f"{today}_{mode}.json"
        if f.exists():
            with open(f) as fp:
                all_alerts.update(json.load(fp))

    if not all_alerts:
        msg = "No alerts today."
    else:
        lines = [f"📊 {len(all_alerts)} alert(s) today:\n"]
        for ticker, info in sorted(
            all_alerts.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True
        ):
            lines.append(
                f"  {ticker}: {info['type'].upper()} "
                f"score={info['score']:.3f} "
                f"gap={info.get('gap_pct', 0)*100:+.1f}%"
            )
        msg = "\n".join(lines)

    send_pushover("📊 Delta v2 Daily Summary", msg, "seed", priority=0)

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    threading.Thread(target=start_keepalive, daemon=True).start()
    time.sleep(2)

    log.info("=" * 50)
    log.info("THE DELTA v2 — Starting (bulk snapshot mode)")
    log.info("=" * 50)

    models, feature_cols, thresholds = load_models()
    if not models:
        log.error("No models found")
        return

    morning_summary_sent = False
    nightly_summary_sent = False
    last_basecamp_scan   = None
    last_date            = None

    while True:
        now    = datetime.now(ET)
        hour   = now.hour
        minute = now.minute

        # Reset daily flags at midnight
        if last_date != now.date():
            morning_summary_sent = False
            nightly_summary_sent = False
            last_basecamp_scan   = None
            ticker_cache.clear()
            last_date = now.date()
            log.info(f"New day: {now.date()}")

        # ── 10 PM nightly summary ──────────────────────────
        if hour == 22 and minute < 5 and not nightly_summary_sent:
            send_nightly_summary()
            nightly_summary_sent = True

        # ── 5 AM morning summary ───────────────────────────
        if hour == 5 and minute < 5 and not morning_summary_sent:
            send_morning_summary(models, feature_cols, thresholds)
            morning_summary_sent = True

        # ── PREMARKET: 4AM - 9:30AM ────────────────────────
        in_premarket = hour >= 4 and (hour < 9 or (hour == 9 and minute < 30))
        if in_premarket:
            fired = run_scan(models, feature_cols, thresholds, "premarket")
            log.info(f"Premarket scan: {fired} alerts")
            time.sleep(SCAN_INTERVAL)
            continue

        # ── EARLY MARKET: 9:30AM - 11AM ───────────────────
        in_early = (hour == 9 and minute >= 30) or hour == 10 or (hour == 11 and minute == 0)
        if in_early:
            fired = run_scan(models, feature_cols, thresholds, "early_market")
            log.info(f"Early market scan: {fired} alerts")
            time.sleep(SCAN_INTERVAL)
            continue

        # ── AH: 4PM - 8PM ─────────────────────────────────
        in_ah = hour >= 16 and hour < 20
        if in_ah:
            fired = run_scan(models, feature_cols, thresholds, "ah",
                           use_today_close=True)
            log.info(f"AH scan: {fired} alerts")
            time.sleep(SCAN_INTERVAL)
            continue

        # ── BASECAMP: 8PM - 4AM ────────────────────────────
        in_basecamp = hour >= 20 or hour < 4
        if in_basecamp:
            if (last_basecamp_scan is None or
                    (now - last_basecamp_scan).seconds >= 600):
                run_basecamp(models, feature_cols, thresholds)
                last_basecamp_scan = now
            time.sleep(60)
            continue

        # ── REST: 11AM - 4PM ──────────────────────────────
        log.debug(f"Resting ({hour}:{minute:02d} ET)")
        time.sleep(300)

if __name__ == "__main__":
    main()
