"""
THE DELTA v2 — main.py (v4)
============================
Predictive scanner. Knows BEFORE the move.

Architecture:
  Midnight:     Overnight scorer — scores ALL 2,000 candidates
                using all 106 features. Saves top 20 watchlist.
  
  4AM-9:30AM:   Premarket scanner — watches top 20 only
                Confirms gap + volume → fires alert
  
  4PM-8PM:      AH scanner — watches top 20 for AH moves
                Catches QTEX-style overnight setups
  
  8PM-4AM:      Basecamp — EDGAR watch every 10min
                New 8-K → scores ticker → adds to watchlist
  
  NO live market scanning (9:30AM-4PM) — model not trained on it

Test endpoint: GET /test?ticker=QTEX
  → Scores any ticker through all 3 models
  → Returns full feature vector + scores

Schedule:
  Midnight:  Overnight score → build top 20 watchlist
  4AM:       Premarket scanner starts
  5AM:       Morning summary
  4PM:       AH scanner starts
  8PM:       Basecamp starts
  10PM:      Nightly summary
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
from urllib.parse import urlparse, parse_qs
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
DATA_DIR   = Path(os.environ.get("DATA_DIR",   "/tmp/data"))

for d in [LOG_DIR, ALERT_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Universe filters
MIN_PRICE       = 0.10
MAX_PRICE       = 2.00
MIN_DOLLAR_VOL  = 25_000
MAX_WATCHLIST   = 20       # top N to watch each day

# Scanner filters (premarket confirmation)
MIN_GAP         = 0.05     # 5% minimum gap — must be activating
MAX_GAP         = 1.00     # 100% max — already moved = reactive, skip
MIN_VOLUME      = 10_000   # minimum PM volume
MIN_ROOM        = 0.20     # 20% room to seed minimum
MAX_FLOAT_M     = 50.0     # max float — micro/small cap only

# Thresholds
THRESHOLDS = {"seed": 0.90, "super": 0.80, "mega": 0.70}

SCAN_INTERVAL = 60

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

# ── Global state ──────────────────────────────────────────────
_models       = {}
_feature_cols = []
_thresholds   = {}
_watchlist    = []   # top 20 for today
_alerted      = set()
_ticker_cache = {}   # float/details cache

# ── Polygon ───────────────────────────────────────────────────
poly_session = requests.Session()
poly_session.params = {"apiKey": POLYGON_API_KEY}
_last_poly = 0.0

def poly_get(path, params=None):
    global _last_poly
    elapsed = time.time() - _last_poly
    if elapsed < 0.21:
        time.sleep(0.21 - elapsed)
    _last_poly = time.time()
    try:
        r = poly_session.get(
            f"https://api.polygon.io{path}",
            params=params or {},
            timeout=20
        )
        if r.status_code == 200:
            return r.json()
        log.debug(f"Polygon {r.status_code}: {path}")
    except Exception as e:
        log.debug(f"Polygon error: {e}")
    return None

# ── EDGAR ─────────────────────────────────────────────────────
edgar_session = requests.Session()
edgar_session.headers.update({"User-Agent": EDGAR_USER_AGENT})

def edgar_get(url):
    try:
        r = edgar_session.get(url, timeout=20)
        if r.status_code == 200:
            return r
    except Exception as e:
        log.debug(f"EDGAR error: {e}")
    return None

# ── Helpers ───────────────────────────────────────────────────
def get_last_trading_day():
    check = date.today() - timedelta(days=1)
    for _ in range(7):
        if check.weekday() < 5 and check not in HOLIDAYS:
            return check
        check -= timedelta(days=1)
    return check

def get_ticker_details(ticker):
    if ticker in _ticker_cache:
        return _ticker_cache[ticker]
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
    _ticker_cache[ticker] = details
    return details

# ── Load models ───────────────────────────────────────────────
def load_models():
    global _models, _feature_cols, _thresholds
    for name in ["seed", "super", "mega"]:
        path = MODEL_DIR / f"{name}_model.json"
        if path.exists():
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            _models[name] = m
            log.info(f"Loaded {name}_model")

    with open(MODEL_DIR / "feature_cols.json") as f:
        _feature_cols = json.load(f)

    _thresholds = THRESHOLDS.copy()
    thresh_path = MODEL_DIR / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            _thresholds.update(json.load(f))

# ── Feature builder ───────────────────────────────────────────
def build_features(snap, details, has_8k=0, edgar_features=None):
    """
    Build full 106-feature vector for a ticker.
    snap: price/volume data
    details: float/market cap
    edgar_features: optional dict with EDGAR signals
    """
    ef = edgar_features or {}

    prev_close = snap.get("prev_close", 0) or 0
    pm_open    = snap.get("pm_open", 0) or 0
    pm_high    = snap.get("pm_high", 0) or 0
    pm_low     = snap.get("pm_low", 0) or 0
    pm_close   = snap.get("pm_close", 0) or prev_close
    volume     = snap.get("volume", 0) or 0
    gap_pct    = snap.get("gap_pct", 0) or 0
    prev_vol   = snap.get("prev_vol", 0) or 0

    pm_move_pct = (pm_high - pm_open) / pm_open if pm_open > 0 else 0

    seed_tgt  = prev_close * 2.0
    super_tgt = prev_close * 6.0
    mega_tgt  = prev_close * 11.0
    rem_seed  = (seed_tgt  - pm_close) / pm_close if pm_close > 0 else 0
    rem_super = (super_tgt - pm_close) / pm_close if pm_close > 0 else 0
    rem_mega  = (mega_tgt  - pm_close) / pm_close if pm_close > 0 else 0

    vol_ratio = volume / prev_vol if prev_vol > 0 and volume > 0 else 0

    features = {col: 0 for col in _feature_cols}

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
        "pm_vol_ratio":           vol_ratio,
        "float_shares":           details.get("float_shares", -1),
        "float_M":                details.get("float_M", -1),
        "float_tier":             details.get("float_tier", -1),
        "market_cap":             details.get("market_cap", -1),
        "is_foreign_listed":      details.get("is_foreign_listed", 0),
        "has_8k":                 has_8k,
        "days_since_last_8k":     ef.get("days_since_last_8k", 999),
        "has_8k_yesterday":       ef.get("has_8k_yesterday", 0),
        "has_merger":             ef.get("has_merger", 0),
        "has_fda":                ef.get("has_fda", 0),
        "has_contract":           ef.get("has_contract", 0),
        "has_dilution":           ef.get("has_dilution", 0),
        "days_since_dilution":    ef.get("days_since_dilution", 999),
        "si_pct":                 snap.get("si_pct", -1),
        "si_tier":                snap.get("si_tier", -1),
        "days_to_cover":          -1,
        "days_since_last_seed":   snap.get("days_since_last_seed", 999),
        "prev_volume":            prev_vol,
        "vol_ratio_prev":         vol_ratio,
        "avg_volume_20d":         snap.get("avg_volume_20d", 0),
        "prev_close":             prev_close,
        "pct_from_52w_high":      snap.get("pct_from_52w_high", 0),
        "pct_from_52w_low":       snap.get("pct_from_52w_low", 0),
        "near_52w_low":           snap.get("near_52w_low", 0),
        "coil_days":              snap.get("coil_days", 0),
        "prev_3d_trend":          snap.get("prev_3d_trend", 0),
        "prev_5d_trend":          snap.get("prev_5d_trend", 0),
        "ah_move_pct":            snap.get("ah_move_pct", 0),
        "ah_direction":           snap.get("ah_direction", 0),
        "ah_volume":              snap.get("ah_volume", 0),
    }

    for k, v in overrides.items():
        if k in features:
            features[k] = v

    return np.array([features[col] for col in _feature_cols]).reshape(1, -1)

# ── Setup quality pre-filter ─────────────────────────────────
def passes_setup_filter(snap, wl):
    """Hard rules checked BEFORE scoring. Fast rejection of bad setups."""
    prev    = snap.get("prev_close", 0) or 0
    price   = snap.get("pm_close", 0) or 0
    volume  = snap.get("volume", 0) or 0
    gap     = snap.get("gap_pct", 0) or 0
    float_M = wl.get("float_M", -1)

    if prev <= 0 or price <= 0:
        return False, "no price data"
    if gap < MIN_GAP:
        return False, f"gap too small ({gap*100:.1f}%)"
    if gap > MAX_GAP:
        return False, f"already extended ({gap*100:.0f}%) — reactive"
    if volume > 0 and volume < MIN_VOLUME:
        return False, f"PM volume too low ({volume:,.0f})"
    if float_M > 0 and float_M > MAX_FLOAT_M:
        return False, f"float too large ({float_M:.0f}M)"
    room = (prev * 2 - price) / price if price > 0 else 0
    if room < MIN_ROOM:
        return False, f"room to seed too small ({room*100:.0f}%)"
    return True, "ok"

# ── Score ─────────────────────────────────────────────────────
def score_ticker(fvec):
    scores = {}
    for name, model in _models.items():
        try:
            scores[name] = round(float(model.predict_proba(fvec)[0][1]), 4)
        except Exception:
            scores[name] = 0.0

    alerts = []
    if scores.get("mega", 0)  >= _thresholds.get("mega",  0.7):
        alerts.append(("mega",  scores["mega"]))
    elif scores.get("super", 0) >= _thresholds.get("super", 0.8):
        alerts.append(("super", scores["super"]))
    elif scores.get("seed", 0)  >= _thresholds.get("seed",  0.9):
        alerts.append(("seed",  scores["seed"]))

    return scores, alerts

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

# ── Alert tracking ────────────────────────────────────────────
def already_alerted(ticker):
    if ticker in _alerted:
        return True
    today = date.today().isoformat()
    for mode in ["premarket", "ah", "basecamp"]:
        f = ALERT_DIR / f"{today}_{mode}.json"
        if f.exists():
            try:
                with open(f) as fp:
                    if ticker in json.load(fp):
                        _alerted.add(ticker)
                        return True
            except Exception:
                pass
    return False

def log_alert(ticker, alert_type, score_val, scores, snap, mode):
    f = ALERT_DIR / f"{date.today().isoformat()}_{mode}.json"
    alerts = {}
    if f.exists():
        with open(f) as fp:
            try:
                alerts = json.load(fp)
            except Exception:
                pass
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
    _alerted.add(ticker)

def format_alert(ticker, alert_type, score_val, scores, snap, details, mode=""):
    emoji  = {"seed": "🌱", "super": "🚀", "mega": "💥"}.get(alert_type, "🌱")
    price  = snap.get("pm_close", 0) or 0
    gap    = snap.get("gap_pct", 0) * 100
    prev   = snap.get("prev_close", 0) or 0
    vol    = snap.get("volume", 0) or 0
    flt    = details.get("float_M", -1)
    rem    = ((prev * 2 - price) / price * 100) if price > 0 and prev > 0 else 0

    mode_tag = f" ({mode})" if mode else ""
    title = f"{emoji} {alert_type.upper()} — {ticker}{mode_tag}"
    msg = (
        f"Price: ${price:.3f} ({gap:+.1f}% from close)\n"
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
# OVERNIGHT SCORER — builds top 20 watchlist
# ══════════════════════════════════════════════════════════════
def get_si_data():
    """Download FINRA short interest for today."""
    today = date.today()
    for delta in range(5):
        check = today - timedelta(days=delta)
        if check.weekday() >= 5 or check in HOLIDAYS:
            continue
        url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{check.strftime('%Y%m%d')}.txt"
        r = edgar_get(url)
        if r and r.status_code == 200:
            si_map = {}
            for line in r.text.strip().split("\n")[1:]:
                parts = line.split("|")
                if len(parts) >= 4:
                    sym = parts[1].strip()
                    try:
                        sv = float(parts[2])
                        tv = float(parts[4]) if len(parts) > 4 else 0
                        if tv > 0:
                            si_map[sym] = round(sv / tv * 100, 2)
                    except Exception:
                        pass
            log.info(f"Loaded SI data: {len(si_map)} tickers from {check}")
            return si_map
    return {}

def get_edgar_8k_today():
    """Get tickers with 8-K filings today or recently."""
    today = date.today()
    filings = {}
    try:
        r = edgar_get(
            "https://efts.sec.gov/LATEST/search-index?q=%228-K%22"
            f"&dateRange=custom&startdt={(today - timedelta(days=3)).isoformat()}"
            f"&enddt={today.isoformat()}&forms=8-K"
        )
        if r and r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            for hit in hits:
                src    = hit.get("_source", {})
                ticker = src.get("ticker", "").strip().upper()
                filed  = src.get("file_date", "")
                form   = src.get("form_type", "")
                if ticker:
                    if ticker not in filings:
                        filings[ticker] = []
                    filings[ticker].append({"filed": filed, "form": form})
    except Exception as e:
        log.debug(f"EDGAR error: {e}")
    log.info(f"EDGAR: {len(filings)} tickers with recent filings")
    return filings

def score_universe():
    """
    Overnight scorer — runs at midnight.
    Scores ALL candidates using full feature set.
    Returns top MAX_WATCHLIST ranked by seed score.
    """
    global _watchlist
    log.info("=" * 50)
    log.info("Overnight scorer starting...")

    last_day = get_last_trading_day()

    # Load grouped daily bars
    data = poly_get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{last_day.isoformat()}",
        {"adjusted": "false"}
    )
    if not data or "results" not in data:
        log.warning("Could not load grouped daily bars")
        return

    # Load support data
    si_map       = get_si_data()
    edgar_8k     = get_edgar_8k_today()

    # Load sector data
    sector_data = {}
    for sym in ["SPY", "QQQ", "IWM", "XBI"]:
        sd = poly_get(f"/v2/aggs/ticker/{sym}/range/1/day/{last_day}/{last_day}")
        if sd and sd.get("results"):
            r = sd["results"][0]
            sector_data[sym] = (r.get("c", 0) - r.get("o", 0)) / r.get("o", 1)

    candidates = []
    total = len(data["results"])
    log.info(f"Scoring {total} tickers...")

    for bar in data["results"]:
        ticker  = bar.get("T", "")
        close   = bar.get("c", 0) or 0
        volume  = bar.get("v", 0) or 0
        high    = bar.get("h", 0) or 0
        low     = bar.get("l", 0) or 0
        open_   = bar.get("o", 0) or 0
        dollar_vol = close * volume

        # Hard filters
        if not ticker or len(ticker) > 5:
            continue
        if ticker.endswith("W") or ticker.endswith("R") or "." in ticker:
            continue
        if close < MIN_PRICE or close > MAX_PRICE:
            continue
        if dollar_vol < MIN_DOLLAR_VOL:
            continue

        # Get details (float etc) — use cache
        details = get_ticker_details(ticker)

        # Get historical data for 52w features
        hist = poly_get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{(last_day - timedelta(days=365)).isoformat()}/{last_day.isoformat()}",
            {"adjusted": "false", "limit": 365}
        )
        hist_results = hist.get("results", []) if hist else []

        price_52w_high = max([r.get("h", 0) for r in hist_results], default=close)
        price_52w_low  = min([r.get("l", 0) for r in hist_results], default=close)
        pct_52w_high   = (close - price_52w_high) / price_52w_high if price_52w_high > 0 else 0
        pct_52w_low    = (close - price_52w_low)  / price_52w_low  if price_52w_low  > 0 else 0
        near_52w_low   = 1 if close < price_52w_low * 1.10 else 0

        # Volume trend
        recent_vols = [r.get("v", 0) for r in hist_results[-20:]] if hist_results else []
        avg_vol_20d = np.mean(recent_vols) if recent_vols else 0
        vol_ratio   = volume / avg_vol_20d if avg_vol_20d > 0 else 0

        # Trend features
        closes = [r.get("c", 0) for r in hist_results]
        trend_3d = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 and closes[-4] > 0 else 0
        trend_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0

        # Coil days — days of tightening range
        coil = 0
        if len(hist_results) >= 5:
            ranges = [r.get("h", 0) - r.get("l", 0) for r in hist_results[-10:]]
            for i in range(len(ranges)-1, 0, -1):
                if ranges[i] <= ranges[i-1]:
                    coil += 1
                else:
                    break

        # SI data
        si_pct  = si_map.get(ticker, -1)
        si_tier = -1
        if si_pct >= 0:
            if si_pct < 5:    si_tier = 0
            elif si_pct < 15: si_tier = 1
            elif si_pct < 30: si_tier = 2
            else:             si_tier = 3

        # EDGAR features
        ef = {"days_since_last_8k": 999, "has_8k_yesterday": 0,
              "has_merger": 0, "has_fda": 0, "has_contract": 0,
              "has_dilution": 0, "days_since_dilution": 999}

        if ticker in edgar_8k:
            filings = edgar_8k[ticker]
            latest  = max(filings, key=lambda x: x.get("filed", ""))  # max = most recent
            try:
                filed_date = date.fromisoformat(latest["filed"][:10])
                ef["days_since_last_8k"] = (last_day - filed_date).days
                ef["has_8k_yesterday"]   = 1 if ef["days_since_last_8k"] <= 1 else 0
            except Exception:
                pass

        # Build snap for feature vector
        snap = {
            "prev_close":        close,
            "pm_open":           open_,
            "pm_high":           high,
            "pm_low":            low,
            "pm_close":          close,
            "volume":            volume,
            "prev_vol":          volume,
            "gap_pct":           0,
            "si_pct":            si_pct,
            "si_tier":           si_tier,
            "pct_from_52w_high": pct_52w_high,
            "pct_from_52w_low":  pct_52w_low,
            "near_52w_low":      near_52w_low,
            "coil_days":         coil,
            "prev_3d_trend":     trend_3d,
            "prev_5d_trend":     trend_5d,
            "avg_volume_20d":    avg_vol_20d,
            "ah_move_pct":       0,
            "ah_direction":      0,
            "ah_volume":         0,
            "spy_prev_day_pct":  sector_data.get("SPY", 0),
            "qqq_prev_day_pct":  sector_data.get("QQQ", 0),
            "iwm_prev_day_pct":  sector_data.get("IWM", 0),
            "xbi_prev_day_pct":  sector_data.get("XBI", 0),
            "market_green":      1 if sector_data.get("SPY", 0) > 0 else 0,
            "market_red":        1 if sector_data.get("SPY", 0) < 0 else 0,
            "sector_hot":        1 if sector_data.get("XBI", 0) > 0.01 else 0,
        }

        has_8k = 1 if ef["days_since_last_8k"] <= 3 else 0
        fvec   = build_features(snap, details, has_8k=has_8k, edgar_features=ef)
        scores, _ = score_ticker(fvec)

        candidates.append({
            "ticker":           ticker,
            "prev_close":       close,
            "prev_vol":         volume,
            "seed_score":       scores.get("seed", 0),
            "super_score":      scores.get("super", 0),
            "mega_score":       scores.get("mega", 0),
            "float_M":          details.get("float_M", -1),
            "si_pct":           si_pct,
            "near_52w_low":     near_52w_low,
            "coil_days":        coil,
            "has_8k":           has_8k,
            "days_since_8k":    ef["days_since_last_8k"],
            "ef":               ef,
            "snap_base":        snap,
            "details":          details,
        })

    # Sort by seed score, take top MAX_WATCHLIST
    candidates.sort(key=lambda x: x["seed_score"], reverse=True)
    _watchlist = candidates[:MAX_WATCHLIST]

    # Save watchlist
    wl_file = DATA_DIR / f"{date.today().isoformat()}_watchlist.json"
    with open(wl_file, "w") as f:
        json.dump([{
            "ticker":      w["ticker"],
            "prev_close":  w["prev_close"],
            "seed_score":  w["seed_score"],
            "super_score": w["super_score"],
            "mega_score":  w["mega_score"],
            "float_M":     w["float_M"],
            "si_pct":      w["si_pct"],
            "has_8k":      w["has_8k"],
            "near_52w_low": w["near_52w_low"],
        } for w in _watchlist], f, indent=2)

    log.info(f"Overnight score complete. Top {len(_watchlist)} watchlist:")
    for w in _watchlist[:5]:
        log.info(f"  {w['ticker']}: seed={w['seed_score']:.3f} "
                 f"super={w['super_score']:.3f} "
                 f"float={w['float_M']:.1f}M "
                 f"si={w['si_pct']:.1f}% "
                 f"8k={w['has_8k']}")

# ══════════════════════════════════════════════════════════════
# PREMARKET / AH SCANNER — watches top 20 only
# ══════════════════════════════════════════════════════════════
def get_watchlist_snapshots(use_today_close=False):
    """
    Pull live snapshots for watchlist tickers only.
    Returns candidates that pass confirmation filters.
    """
    if not _watchlist:
        log.warning("Watchlist empty — skipping scan")
        return []

    tickers    = [w["ticker"] for w in _watchlist]
    ticker_str = ",".join(tickers)

    data = poly_get(
        "/v2/snapshot/locale/us/markets/stocks/tickers",
        {"tickers": ticker_str, "include_otc": "false"}
    )
    if not data or "tickers" not in data:
        return []

    # Build lookup
    wl_map = {w["ticker"]: w for w in _watchlist}

    candidates = []
    for t in data["tickers"]:
        ticker  = t.get("ticker", "")
        day     = t.get("day", {})
        prev    = t.get("prevDay", {})
        min_bar = t.get("min", {})
        wl      = wl_map.get(ticker, {})

        prev_close = wl.get("prev_close", 0) or prev.get("c", 0) or 0
        if prev_close <= 0:
            continue

        # Current price
        pm_close = day.get("c", 0) or min_bar.get("c", 0) or 0
        change   = t.get("todaysChangePerc", 0) or 0
        if pm_close <= 0 and change != 0:
            pm_close = prev_close * (1 + change / 100)

        if pm_close <= 0:
            continue

        pm_open = day.get("o", 0) or pm_close
        pm_high = day.get("h", 0) or pm_close
        pm_low  = day.get("l", 0) or pm_close
        volume  = day.get("v", 0) or min_bar.get("v", 0) or 0

        # Reference price
        ref_price = day.get("c", 0) or prev_close if use_today_close else prev_close

        gap = (pm_close - ref_price) / ref_price if ref_price > 0 else 0

        # Confirmation filters
        if gap < MIN_GAP:
            continue
        if volume > 0 and volume < MIN_VOLUME:
            continue

        # Room to seed
        remaining = (prev_close * 2.0 - pm_close) / pm_close if pm_close > 0 else 0
        if remaining < MIN_ROOM:
            continue

        candidates.append({
            "ticker":     ticker,
            "prev_close": prev_close,
            "prev_vol":   wl.get("prev_vol", 0),
            "pm_open":    pm_open,
            "pm_high":    pm_high,
            "pm_low":     pm_low,
            "pm_close":   pm_close,
            "volume":     volume,
            "gap_pct":    gap,
            "wl_data":    wl,
        })

    return candidates

def run_scan(mode="premarket", use_today_close=False):
    candidates = get_watchlist_snapshots(use_today_close)
    log.info(f"{mode}: {len(candidates)}/{len(_watchlist)} watchlist stocks active")

    fired = 0
    # Sort by OVERNIGHT score — not live score
    # Overnight score is clean, predictive, not reactive
    sorted_candidates = sorted(
        candidates,
        key=lambda x: x["wl_data"].get("seed_score", 0),
        reverse=True
    )

    for snap in sorted_candidates:
        ticker = snap["ticker"]
        if already_alerted(ticker):
            continue

        wl = snap["wl_data"]

        # ── SETUP QUALITY GATE ────────────────────────────────
        passes, reason = passes_setup_filter(snap, wl)
        if not passes:
            log.debug(f"{ticker}: skip — {reason}")
            continue

        # ── USE OVERNIGHT SCORE AS PRIMARY SIGNAL ─────────────
        # Don't re-score with live data — overnight score is more predictive
        # Live premarket data already confirmed by passes_setup_filter
        overnight_seed  = wl.get("seed_score", 0)
        overnight_super = wl.get("super_score", 0)
        overnight_mega  = wl.get("mega_score", 0)

        # Determine alert tier from overnight scores only
        if overnight_mega >= _thresholds.get("mega", 0.70):
            alert_type, score_val = "mega", overnight_mega
        elif overnight_super >= _thresholds.get("super", 0.80):
            alert_type, score_val = "super", overnight_super
        elif overnight_seed >= _thresholds.get("seed", 0.90):
            alert_type, score_val = "seed", overnight_seed
        else:
            log.debug(f"{ticker}: overnight scores too low — skip")
            continue

        scores = {
            "seed":  overnight_seed,
            "super": overnight_super,
            "mega":  overnight_mega,
        }

        details = wl.get("details", get_ticker_details(ticker))
        log_alert(ticker, alert_type, score_val, scores, snap, mode)

        title, msg = format_alert(
            ticker, alert_type, score_val, scores, snap, details, mode
        )
        priority = 1 if alert_type in ("super", "mega") else 0
        send_pushover(title, msg, alert_type, priority)

        gap  = snap.get("gap_pct", 0) * 100
        vol  = snap.get("volume", 0)
        prev = snap.get("prev_close", 0)
        pm   = snap.get("pm_close", 0)
        room = (prev * 2 - pm) / pm * 100 if pm > 0 else 0

        log.info(
            f"ALERT [{mode}]: {ticker} {alert_type.upper()} "
            f"overnight={score_val:.3f} gap={gap:+.1f}% "
            f"vol={vol:,.0f} room={room:+.1f}%"
        )
        fired += 1

    return fired

# ══════════════════════════════════════════════════════════════
# BASECAMP — EDGAR overnight watch
# ══════════════════════════════════════════════════════════════
def run_basecamp():
    log.info("Basecamp: scanning EDGAR...")
    try:
        r = edgar_get(
            "https://efts.sec.gov/LATEST/search-index?q=%228-K%22"
            f"&dateRange=custom&startdt={date.today().isoformat()}&forms=8-K"
        )
        if not r or r.status_code != 200:
            return
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception:
        return

    log.info(f"Basecamp: {len(hits)} 8-K filings")
    fired = 0

    for hit in hits:
        src    = hit.get("_source", {})
        ticker = src.get("ticker", "").strip().upper()
        if not ticker or already_alerted(ticker):
            continue
        if len(ticker) > 5 or ticker.endswith("W") or "." in ticker:
            continue

        # Get snapshot
        snaps = poly_get(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            {"tickers": ticker}
        )
        if not snaps or "tickers" not in snaps or not snaps["tickers"]:
            continue

        t    = snaps["tickers"][0]
        day  = t.get("day", {})
        prev = t.get("prevDay", {})

        prev_close = prev.get("c", 0) or 0
        if prev_close < MIN_PRICE or prev_close > MAX_PRICE:
            continue

        pm_close = day.get("c", 0) or prev_close
        volume   = day.get("v", 0) or 0
        gap      = (pm_close - prev_close) / prev_close if prev_close > 0 else 0

        details = get_ticker_details(ticker)
        snap = {
            "ticker":     ticker,
            "prev_close": prev_close,
            "pm_open":    day.get("o", 0) or pm_close,
            "pm_high":    day.get("h", 0) or pm_close,
            "pm_low":     day.get("l", 0) or pm_close,
            "pm_close":   pm_close,
            "volume":     volume,
            "gap_pct":    gap,
            "prev_vol":   prev.get("v", 0),
        }

        ef = {
            "days_since_last_8k": 0,
            "has_8k_yesterday":   1,
            "has_merger":         1 if "merger" in str(src).lower() else 0,
            "has_fda":            1 if "fda" in str(src).lower() else 0,
        }

        fvec = build_features(snap, details, has_8k=1, edgar_features=ef)
        scores, alerts = score_ticker(fvec)

        if not alerts:
            continue

        alert_type, score_val = alerts[0]
        log_alert(ticker, alert_type, score_val, scores, snap, "basecamp")

        title, msg = format_alert(
            ticker, alert_type, score_val, scores, snap, details, "overnight 8-K"
        )
        priority = 1 if alert_type in ("super", "mega") else 0
        send_pushover(title, msg, alert_type, priority)
        log.info(f"BASECAMP: {ticker} {alert_type.upper()} score={score_val:.3f}")

        # Also add to watchlist for tomorrow
        if ticker not in [w["ticker"] for w in _watchlist]:
            _watchlist.append({
                "ticker":      ticker,
                "prev_close":  prev_close,
                "prev_vol":    prev.get("v", 0),
                "seed_score":  scores.get("seed", 0),
                "super_score": scores.get("super", 0),
                "mega_score":  scores.get("mega", 0),
                "float_M":     details.get("float_M", -1),
                "si_pct":      -1,
                "has_8k":      1,
                "near_52w_low": 0,
                "ef":          ef,
                "snap_base":   snap,
                "details":     details,
            })
            log.info(f"Added {ticker} to watchlist from basecamp")

        fired += 1

# ══════════════════════════════════════════════════════════════
# DAILY VALIDATION — runs at 4PM after market close
# ══════════════════════════════════════════════════════════════
def run_daily_validation():
    """
    Pull today's top gainers from Polygon.
    Compare against watchlist.
    Calculate recall — how many seeds did we have on watchlist?
    Send summary to Pushover.
    """
    log.info("Running daily validation...")

    last_day = date.today()
    data = poly_get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{last_day.isoformat()}",
        {"adjusted": "false"}
    )
    if not data or "results" not in data:
        log.warning("Could not load today's bars for validation")
        return

    # Find all stocks that hit 100%+ today
    wl_tickers = [w["ticker"] for w in _watchlist]
    seeds_today = []

    for bar in data["results"]:
        ticker = bar.get("T", "")
        close  = bar.get("c", 0) or 0
        high   = bar.get("h", 0) or 0
        open_  = bar.get("o", 0) or 0
        prev   = bar.get("o", 0) or 0  # use open as proxy if no prev

        # Get prev close from watchlist or skip
        wl = next((w for w in _watchlist if w["ticker"] == ticker), None)
        prev_close = wl["prev_close"] if wl else 0

        if prev_close > 0 and high >= prev_close * 2.0:
            seeds_today.append(ticker)

    # Calculate recall
    caught = [t for t in seeds_today if t in wl_tickers]
    missed = [t for t in seeds_today if t not in wl_tickers]
    recall = len(caught) / len(seeds_today) if seeds_today else 0

    log.info(f"Validation: {len(seeds_today)} seeds today | "
             f"caught={len(caught)} | missed={len(missed)} | "
             f"recall={recall:.0%}")

    # Build message
    lines = [f"📊 Daily Validation — {last_day}"]
    lines.append(f"Seeds today: {len(seeds_today)}")
    lines.append(f"On watchlist: {len(caught)} ({recall:.0%} recall)")
    if caught:
        lines.append(f"Caught: {', '.join(caught)}")
    if missed:
        lines.append(f"Missed: {', '.join(missed[:5])}")
    if not seeds_today:
        lines.append("No seeds today")

    msg = "\n".join(lines)
    send_pushover("📊 Delta v2 Validation", msg, "seed", priority=0)

    # Save to file for tracking
    val_file = ALERT_DIR / f"{last_day.isoformat()}_validation.json"
    with open(val_file, "w") as f:
        json.dump({
            "date":        last_day.isoformat(),
            "seeds_today": seeds_today,
            "caught":      caught,
            "missed":      missed,
            "recall":      round(recall, 3),
            "watchlist":   wl_tickers,
        }, f, indent=2)

# ══════════════════════════════════════════════════════════════
# SUMMARIES
# ══════════════════════════════════════════════════════════════
def send_morning_summary():
    if not _watchlist:
        msg = "No watchlist built yet."
    else:
        lines = [f"☀️ Today's watchlist ({len(_watchlist)} stocks):\n"]
        for w in _watchlist[:10]:
            lines.append(
                f"  {w['ticker']}: S={w['seed_score']:.2f} "
                f"Su={w['super_score']:.2f} "
                f"float={w['float_M']:.1f}M "
                f"8k={'✅' if w['has_8k'] else '❌'} "
                f"52wL={'✅' if w['near_52w_low'] else '❌'}"
            )
        msg = "\n".join(lines)
    send_pushover("☀️ Delta v2 Morning Report", msg, "seed", priority=0)

def send_nightly_summary():
    today = date.today().isoformat()
    all_alerts = {}
    for mode in ["premarket", "ah", "basecamp"]:
        f = ALERT_DIR / f"{today}_{mode}.json"
        if f.exists():
            try:
                with open(f) as fp:
                    all_alerts.update(json.load(fp))
            except Exception:
                pass

    if not all_alerts:
        msg = "No alerts today."
    else:
        lines = [f"📊 {len(all_alerts)} alert(s) today:\n"]
        for ticker, info in sorted(all_alerts.items(),
                                   key=lambda x: x[1].get("score", 0), reverse=True):
            lines.append(
                f"  {ticker}: {info['type'].upper()} "
                f"score={info['score']:.3f} "
                f"gap={info.get('gap_pct', 0)*100:+.1f}%"
            )
        msg = "\n".join(lines)

    send_pushover("📊 Delta v2 Daily Summary", msg, "seed", priority=0)

# ══════════════════════════════════════════════════════════════
# TEST ENDPOINT — score any ticker on demand
# ══════════════════════════════════════════════════════════════
def score_ticker_on_demand(ticker):
    """Score any ticker through all models. Returns JSON."""
    ticker = ticker.upper().strip()
    log.info(f"Test scoring: {ticker}")

    # Get snapshot
    snaps = poly_get(
        "/v2/snapshot/locale/us/markets/stocks/tickers",
        {"tickers": ticker}
    )
    if not snaps or "tickers" not in snaps or not snaps["tickers"]:
        return {"error": f"No snapshot data for {ticker}"}

    t    = snaps["tickers"][0]
    day  = t.get("day", {})
    prev = t.get("prevDay", {})
    min_bar = t.get("min", {})

    prev_close = prev.get("c", 0) or 0
    pm_close   = day.get("c", 0) or min_bar.get("c", 0) or 0
    change     = t.get("todaysChangePerc", 0) or 0
    if pm_close <= 0 and change != 0 and prev_close > 0:
        pm_close = prev_close * (1 + change / 100)

    volume = day.get("v", 0) or min_bar.get("v", 0) or 0
    gap    = (pm_close - prev_close) / prev_close if prev_close > 0 else 0

    details = get_ticker_details(ticker)

    snap = {
        "ticker":     ticker,
        "prev_close": prev_close,
        "pm_open":    day.get("o", 0) or pm_close,
        "pm_high":    day.get("h", 0) or pm_close,
        "pm_low":     day.get("l", 0) or pm_close,
        "pm_close":   pm_close,
        "volume":     volume,
        "gap_pct":    gap,
        "prev_vol":   prev.get("v", 0),
    }

    fvec = build_features(snap, details)
    scores, alerts = score_ticker(fvec)

    remaining = (prev_close * 2 - pm_close) / pm_close * 100 if pm_close > 0 else 0

    return {
        "ticker":      ticker,
        "prev_close":  prev_close,
        "price":       pm_close,
        "gap_pct":     round(gap * 100, 2),
        "volume":      volume,
        "float_M":     details.get("float_M", -1),
        "room_to_seed": round(remaining, 2),
        "scores": {
            "seed":  scores.get("seed", 0),
            "super": scores.get("super", 0),
            "mega":  scores.get("mega", 0),
        },
        "alert": alerts[0][0] if alerts else "none",
        "on_watchlist": ticker in [w["ticker"] for w in _watchlist],
    }

# ══════════════════════════════════════════════════════════════
# HTTP SERVER — keepalive + test endpoint
# ══════════════════════════════════════════════════════════════
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        # Health check
        if parsed.path == "/" or parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            wl_tickers = [w["ticker"] for w in _watchlist]
            self.wfile.write(
                f"Delta v2 alive\nWatchlist ({len(_watchlist)}): {', '.join(wl_tickers)}\n"
                f"Alerted today: {len(_alerted)}\n".encode()
            )
            return

        # Test endpoint: /test?ticker=QTEX
        if parsed.path == "/test":
            params = parse_qs(parsed.query)
            ticker = params.get("ticker", [""])[0]
            if not ticker:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing ticker param. Use /test?ticker=QTEX")
                return

            result = score_ticker_on_demand(ticker)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, indent=2).encode())
            return

        # Watchlist endpoint: /watchlist
        if parsed.path == "/watchlist":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            wl = [{
                "ticker":      w["ticker"],
                "prev_close":  w["prev_close"],
                "seed_score":  w["seed_score"],
                "super_score": w["super_score"],
                "mega_score":  w["mega_score"],
                "float_M":     w["float_M"],
                "si_pct":      w["si_pct"],
                "has_8k":      w["has_8k"],
                "near_52w_low": w["near_52w_low"],
            } for w in _watchlist]
            self.wfile.write(json.dumps(wl, indent=2).encode())
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, *args): pass

def start_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(2)

    log.info("=" * 50)
    log.info("THE DELTA v2 — Starting (predictive scorer)")
    log.info("=" * 50)

    load_models()
    if not _models:
        log.error("No models found")
        return

    # Build watchlist immediately at startup
    log.info("Building initial watchlist...")
    try:
        score_universe()
    except Exception as e:
        log.error(f"Overnight scorer failed: {e}")

    morning_summary_sent      = False
    nightly_summary_sent      = False
    confirmation_score_sent   = False
    last_basecamp_scan        = None
    last_score_date           = date.today()
    last_date                 = None

    while True:
        now    = datetime.now(ET)
        hour   = now.hour
        minute = now.minute

        # Reset daily state at midnight
        if last_date != now.date():
            morning_summary_sent    = False
            nightly_summary_sent    = False
            confirmation_score_sent = False
            last_basecamp_scan      = None
            _alerted.clear()
            _ticker_cache.clear()
            last_date = now.date()
            log.info(f"New day: {now.date()}")

        # Midnight overnight score
        if hour == 0 and minute < 5 and last_score_date != now.date():
            log.info("Midnight — running overnight scorer...")
            try:
                score_universe()
                last_score_date = now.date()
            except Exception as e:
                log.error(f"Overnight scorer error: {e}")

        # 10 PM nightly summary
        if hour == 22 and minute < 5 and not nightly_summary_sent:
            send_nightly_summary()
            nightly_summary_sent = True

        # 5 AM morning summary
        if hour == 5 and minute < 5 and not morning_summary_sent:
            send_morning_summary()
            morning_summary_sent = True

        # 6:30AM confirmation window — one-time deep scan
        # Most predictive window: gap established, not yet fully played
        in_confirmation = hour == 6 and 30 <= minute <= 59
        if in_confirmation and not confirmation_score_sent:
            log.info("6:30AM confirmation window — running focused scan")
            fired = run_scan("premarket_confirmation")
            log.info(f"Confirmation scan: {fired} alerts")
            confirmation_score_sent = True

        # PREMARKET: 4AM - 9:30AM only
        in_premarket = hour >= 4 and (hour < 9 or (hour == 9 and minute < 30))
        if in_premarket:
            fired = run_scan("premarket")
            log.info(f"Premarket scan done: {fired} alerts")
            time.sleep(SCAN_INTERVAL)
            continue

        # 4PM daily validation — how many seeds did we catch?
        if hour == 16 and minute < 5:
            try:
                run_daily_validation()
            except Exception as e:
                log.error(f"Validation error: {e}")

        # AH: 4PM - 8PM only
        in_ah = hour >= 16 and hour < 20
        if in_ah:
            fired = run_scan("ah", use_today_close=True)
            log.info(f"AH scan done: {fired} alerts")
            time.sleep(SCAN_INTERVAL)
            continue

        # BASECAMP: 8PM - 4AM
        in_basecamp = hour >= 20 or hour < 4
        if in_basecamp:
            if (last_basecamp_scan is None or
                    (now - last_basecamp_scan).seconds >= 600):
                try:
                    run_basecamp()
                except Exception as e:
                    log.error(f"Basecamp error: {e}")
                last_basecamp_scan = now
            time.sleep(60)
            continue

        # REST: 9:30AM - 4PM — no live market scanning
        log.debug(f"Resting ({hour}:{minute:02d} ET) — market hours, not scanning")
        time.sleep(300)

if __name__ == "__main__":
    main()
