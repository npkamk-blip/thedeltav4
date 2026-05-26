"""
THE DELTA v2 — main.py (v3)
============================
Two-phase premarket scanner:

Phase 1 — Universe builder (runs at midnight + startup):
  Loads ALL tickers from grouped daily bars
  Filters by price > $0.10, prev vol > $10k
  Saves watchlist of ~5,000 candidates

Phase 2 — Scanner (every 60s):
  Pulls specific ticker snapshots in batches of 100
  This endpoint populates premarket price AND volume
  Scores through models, fires alerts

Schedule (ET):
  4AM  - 9:30AM:  Premarket scanner
  9:30AM - 11AM:  Early market scanner
  11AM - 4PM:     Sleep
  4PM  - 8PM:     AH scanner
  8PM  - 4AM:     Basecamp EDGAR watch
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

MIN_PRICE      = 0.10
MIN_PREV_VOL   = 10_000
MIN_GAP        = 0.05
BATCH_SIZE     = 100
SCAN_INTERVAL  = 60
THRESHOLDS     = {"seed": 0.90, "super": 0.80, "mega": 0.70}

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
            log.warning(f"Pushover failed: {r.status_code} {r.text}")
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
    if elapsed < 0.21:
        time.sleep(0.21 - elapsed)
    _last_call = time.time()
    try:
        r = poly_session.get(
            f"https://api.polygon.io{path}",
            params=params or {},
            timeout=20
        )
        if r.status_code == 200:
            return r.json()
        else:
            log.debug(f"Polygon {r.status_code}: {path}")
    except Exception as e:
        log.debug(f"Polygon error: {e}")
    return None

# ── Holiday list ──────────────────────────────────────────────
HOLIDAYS = {
    date(2024,1,1),date(2024,1,15),date(2024,2,19),date(2024,3,29),
    date(2024,5,27),date(2024,6,19),date(2024,7,4),date(2024,9,2),
    date(2024,11,28),date(2024,12,25),
    date(2025,1,1),date(2025,1,9),date(2025,1,20),date(2025,2,17),
    date(2025,4,18),date(2025,5,26),date(2025,6,19),date(2025,7,4),
    date(2025,9,1),date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

def get_last_trading_day():
    check = date.today() - timedelta(days=1)
    for _ in range(7):
        if check.weekday() < 5 and check not in HOLIDAYS:
            return check
        check -= timedelta(days=1)
    return check

# ══════════════════════════════════════════════════════════════
# PHASE 1 — UNIVERSE BUILDER
# ══════════════════════════════════════════════════════════════
_watchlist      = []   # list of {ticker, prev_close, prev_vol}
_watchlist_date = None

def build_watchlist():
    """
    Load ALL tickers from grouped daily bars.
    Filter by price and volume.
    Runs once per day at startup.
    """
    global _watchlist, _watchlist_date

    today = date.today()
    if _watchlist_date == today and _watchlist:
        return

    last_day = get_last_trading_day()
    log.info(f"Building watchlist from {last_day}...")

    data = poly_get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{last_day.isoformat()}",
        {"adjusted": "false"}
    )

    if not data or "results" not in data:
        log.warning(f"Could not load grouped bars for {last_day}")
        return

    watchlist = []
    for bar in data["results"]:
        ticker  = bar.get("T", "")
        close   = bar.get("c", 0) or 0
        volume  = bar.get("v", 0) or 0
        dollar_vol = close * volume

        if not ticker:
            continue
        if close < MIN_PRICE:
            continue
        if close > 2.00:          # seeds almost always under $2
            continue
        if dollar_vol < 25_000:   # need real liquidity
            continue
        # Skip warrants, units, rights, ETFs
        if len(ticker) > 5:
            continue
        if ticker.endswith("W") or ticker.endswith("R") or ticker.endswith("U"):
            continue
        if "." in ticker:         # skip preferred shares etc
            continue

        watchlist.append({
            "ticker":     ticker,
            "prev_close": close,
            "prev_vol":   volume,
        })

    _watchlist      = watchlist
    _watchlist_date = today
    log.info(f"Watchlist built: {len(_watchlist)} tickers from {last_day}")

# ══════════════════════════════════════════════════════════════
# PHASE 2 — SPECIFIC TICKER SNAPSHOT (populates premarket data)
# ══════════════════════════════════════════════════════════════
def get_batch_snapshots(tickers):
    """
    Pull snapshots for specific tickers.
    This endpoint populates premarket price AND volume.
    Returns list of snapshot dicts.
    """
    if not tickers:
        return []

    ticker_str = ",".join(tickers)
    data = poly_get(
        "/v2/snapshot/locale/us/markets/stocks/tickers",
        {"tickers": ticker_str, "include_otc": "false"}
    )

    if not data or "tickers" not in data:
        return []

    return data["tickers"]

def get_candidates(use_today_close=False):
    """
    Scan watchlist in batches.
    Returns candidates that pass gap + volume filters.
    """
    if not _watchlist:
        build_watchlist()
        if not _watchlist:
            return []

    candidates = []
    tickers = [w["ticker"] for w in _watchlist]
    prev_map = {w["ticker"]: w for w in _watchlist}

    # Process in batches of BATCH_SIZE
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        snaps = get_batch_snapshots(batch)

        for t in snaps:
            ticker = t.get("ticker", "")
            day    = t.get("day", {})
            prev   = t.get("prevDay", {})
            min_bar = t.get("min", {})

            # Price — use day.c first, fall back to last minute bar
            pm_close = day.get("c", 0) or 0
            if pm_close <= 0:
                pm_close = min_bar.get("c", 0) or 0

            pm_open   = day.get("o", 0) or pm_close
            pm_high   = day.get("h", 0) or pm_close
            pm_low    = day.get("l", 0) or pm_close

            # Volume — use day.v first, fall back to last minute bar
            volume = day.get("v", 0) or 0
            if volume <= 0:
                volume = min_bar.get("v", 0) or 0

            # Prev close — use our watchlist map (from grouped bars)
            prev_data  = prev_map.get(ticker, {})
            prev_close = prev_data.get("prev_close", 0)
            prev_vol   = prev_data.get("prev_vol", 0)

            # Fallback to snapshot prevDay if needed
            if prev_close <= 0:
                prev_close = prev.get("c", 0) or 0
            if prev_close <= 0 or prev_close < MIN_PRICE:
                continue

            # If still no current price, use changePerc to back-calculate
            change_perc = t.get("todaysChangePerc", 0) or 0
            if pm_close <= 0 and change_perc != 0:
                pm_close = prev_close * (1 + change_perc / 100)
                pm_open = pm_high = pm_low = pm_close

            if pm_close <= 0:
                continue

            # Reference price for gap calculation
            if use_today_close:
                ref_price = day.get("c", 0) or prev_close
            else:
                ref_price = prev_close

            if ref_price <= 0:
                continue

            gap = (pm_close - ref_price) / ref_price
            if gap < 0.05:        # minimum 5% gap
                continue

            # Must have room to reach 100% — this is the real filter
            remaining = (ref_price * 2.0 - pm_close) / pm_close
            if remaining < 0.40:  # less than 40% room = too late
                continue

            # Volume filter
            if volume > 0:
                if volume < 50_000:
                    continue
                # Volume acceleration — today's volume should be
                # on track to exceed prev day (annualize current vol)
                # Market open ~6.5 hours = 390 minutes
                # If we're 30 min in and have 50k vol = 650k annualized
                now_et = datetime.now(ET)
                market_open = now_et.replace(hour=9, minute=30, second=0)
                minutes_open = max((now_et - market_open).seconds / 60, 1)
                projected_vol = volume * (390 / minutes_open)
                vol_ratio = projected_vol / prev_vol if prev_vol > 0 else 1
                if vol_ratio < 1.5:  # projected volume < 1.5x prev day = weak
                    continue
            else:
                # Premarket — use prev day volume as proxy
                if prev_vol < 25_000:
                    continue

            candidates.append({
                "ticker":     ticker,
                "prev_close": prev_close,
                "pm_open":    pm_open,
                "pm_high":    pm_high,
                "pm_low":     pm_low,
                "pm_close":   pm_close,
                "volume":     volume if volume > 0 else prev_vol,
                "gap_pct":    gap,
                "change_pct": change_perc,
            })

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
_alerted_today = set()

def already_alerted(ticker):
    # Check memory first (fast)
    if ticker in _alerted_today:
        return True
    # Check all alert files (survives deploys)
    today = date.today().isoformat()
    for mode in ["premarket", "early_market", "ah", "basecamp"]:
        f = ALERT_DIR / f"{today}_{mode}.json"
        if f.exists():
            try:
                with open(f) as fp:
                    if ticker in json.load(fp):
                        _alerted_today.add(ticker)  # cache it
                        return True
            except Exception:
                pass
    return False

def mark_alerted(ticker):
    _alerted_today.add(ticker)

def log_alert(ticker, alert_type, score_val, scores, snap, mode):
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
# MAIN SCAN
# ══════════════════════════════════════════════════════════════
def run_scan(models, feature_cols, thresholds, mode="premarket", use_today_close=False):
    candidates = get_candidates(use_today_close=use_today_close)
    log.info(f"{mode}: {len(candidates)} candidates (gap >= {MIN_GAP*100:.0f}%)")

    fired = 0
    # Sort by model score descending before alerting
    scored = []
    for snap in candidates:
        ticker = snap["ticker"]
        if already_alerted(ticker):
            continue
        details = get_details(ticker)
        fvec    = build_features(snap, details, feature_cols)
        scores, alerts = score(fvec, models, thresholds)
        if alerts:
            scored.append((snap, details, scores, alerts))
    
    # Sort by seed score descending, take top 3 per scan
    scored.sort(key=lambda x: x[2].get("seed", 0), reverse=True)
    scored = scored[:3]

    for snap, details, scores, alerts in scored:
        if fired >= 3:
            break
        ticker = snap["ticker"]

        alert_type, score_val = alerts[0]
        log_alert(ticker, alert_type, score_val, scores, snap, mode)
        mark_alerted(ticker)

        title, msg = format_alert(
            ticker, alert_type, score_val, scores, snap, details, mode
        )
        priority = 1 if alert_type in ("super", "mega") else 0
        send_pushover(title, msg, alert_type, priority)

        log.info(
            f"ALERT [{mode}]: {ticker} {alert_type.upper()} "
            f"score={score_val:.3f} gap={snap['gap_pct']*100:+.1f}% "
            f"vol={snap['volume']:,.0f}"
        )
        fired += 1

    return fired

# ══════════════════════════════════════════════════════════════
# BASECAMP
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
        if already_alerted(ticker):
            continue

        snaps = get_batch_snapshots([ticker])
        if not snaps:
            continue

        t    = snaps[0]
        day  = t.get("day", {})
        prev = t.get("prevDay", {})

        prev_close = prev.get("c", 0) or 0
        if prev_close < MIN_PRICE:
            # Try watchlist
            wl = next((w for w in _watchlist if w["ticker"] == ticker), None)
            if wl:
                prev_close = wl["prev_close"]
        if prev_close < MIN_PRICE:
            continue

        pm_close = day.get("c", 0) or prev_close
        volume   = day.get("v", 0) or 0

        snap = {
            "ticker":     ticker,
            "prev_close": prev_close,
            "pm_open":    day.get("o", 0) or pm_close,
            "pm_high":    day.get("h", 0) or pm_close,
            "pm_low":     day.get("l", 0) or pm_close,
            "pm_close":   pm_close,
            "volume":     volume,
            "gap_pct":    (pm_close - prev_close) / prev_close if prev_close > 0 else 0,
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
        log.info(f"BASECAMP ALERT: {ticker} {alert_type.upper()} score={score_val:.3f}")
        fired += 1

    return fired

# ══════════════════════════════════════════════════════════════
# SUMMARIES
# ══════════════════════════════════════════════════════════════
def send_morning_summary(models, feature_cols, thresholds):
    candidates = get_candidates()[:50]
    hot = []
    for snap in candidates:
        details = get_details(snap["ticker"])
        fvec    = build_features(snap, details, feature_cols)
        scores, _ = score(fvec, models, thresholds)
        if scores.get("seed", 0) > 0.40:
            hot.append((snap["ticker"], scores, snap))

    if not hot:
        msg = f"Quiet morning. {len(candidates)} gapping 5%+, none scoring high."
    else:
        lines = [f"☀️ {len(hot)} setup(s) scoring above 0.40:\n"]
        for ticker, sc, snap in sorted(hot, key=lambda x: x[1].get("seed",0), reverse=True)[:5]:
            gap = snap.get("gap_pct", 0) * 100
            vol = snap.get("volume", 0)
            lines.append(
                f"{ticker}: {gap:+.1f}% vol={vol:,.0f} | "
                f"S:{sc.get('seed',0):.2f} Su:{sc.get('super',0):.2f}"
            )
        msg = "\n".join(lines)

    send_pushover("☀️ Delta v2 Morning Report", msg, "seed", priority=0)

def send_nightly_summary():
    today = date.today().isoformat()
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
        for ticker, info in sorted(all_alerts.items(),
                                   key=lambda x: x[1].get("score",0), reverse=True):
            lines.append(
                f"  {ticker}: {info['type'].upper()} "
                f"score={info['score']:.3f} "
                f"gap={info.get('gap_pct',0)*100:+.1f}%"
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
    log.info("THE DELTA v2 — Starting (two-phase scanner)")
    log.info("=" * 50)

    models, feature_cols, thresholds = load_models()
    if not models:
        log.error("No models found")
        return

    # Build watchlist at startup
    build_watchlist()

    morning_summary_sent = False
    nightly_summary_sent = False
    last_basecamp_scan   = None
    last_date            = None

    while True:
        now    = datetime.now(ET)
        hour   = now.hour
        minute = now.minute

        # Reset daily state
        if last_date != now.date():
            morning_summary_sent = False
            nightly_summary_sent = False
            last_basecamp_scan   = None
            ticker_cache.clear()
            _alerted_today.clear()
            last_date = now.date()
            log.info(f"New day: {now.date()}")
            build_watchlist()

        # 10 PM nightly summary
        if hour == 22 and minute < 5 and not nightly_summary_sent:
            send_nightly_summary()
            nightly_summary_sent = True

        # 5 AM morning summary
        if hour == 5 and minute < 5 and not morning_summary_sent:
            send_morning_summary(models, feature_cols, thresholds)
            morning_summary_sent = True

        # PREMARKET: 4AM - 9:30AM
        in_premarket = hour >= 4 and (hour < 9 or (hour == 9 and minute < 30))
        if in_premarket:
            fired = run_scan(models, feature_cols, thresholds, "premarket")
            log.info(f"Premarket scan done: {fired} alerts")
            time.sleep(SCAN_INTERVAL)
            continue

        # EARLY MARKET: 9:30AM - 11AM
        in_early = (hour == 9 and minute >= 30) or hour == 10 or (hour == 11 and minute == 0)
        if in_early:
            fired = run_scan(models, feature_cols, thresholds, "early_market")
            log.info(f"Early market scan done: {fired} alerts")
            time.sleep(SCAN_INTERVAL)
            continue

        # AH: 4PM - 8PM
        in_ah = hour >= 16 and hour < 20
        if in_ah:
            fired = run_scan(models, feature_cols, thresholds, "ah",
                           use_today_close=True)
            log.info(f"AH scan done: {fired} alerts")
            time.sleep(SCAN_INTERVAL)
            continue

        # BASECAMP: 8PM - 4AM
        in_basecamp = hour >= 20 or hour < 4
        if in_basecamp:
            if (last_basecamp_scan is None or
                    (now - last_basecamp_scan).seconds >= 600):
                run_basecamp(models, feature_cols, thresholds)
                last_basecamp_scan = now
            time.sleep(60)
            continue

        # REST: 11AM - 4PM
        log.debug(f"Resting ({hour}:{minute:02d} ET)")
        time.sleep(300)

if __name__ == "__main__":
    main()
