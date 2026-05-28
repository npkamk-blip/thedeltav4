"""
THE DELTA v2 — main.py (v5)
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
POLYGON_API_KEY    = os.environ.get("MASSIVE_API_KEY", "")
PUSHOVER_USER_KEY  = os.environ.get("PUSHOVER_USER_KEY", "utvy26j5q66kae27ncwxsftfcuhi92")
PUSHOVER_APP_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN", "a3szzncpvgyevbck6z5z5yszm7nzg3")
EDGAR_USER_AGENT   = "NPKNOB@gmail.com"
MODEL_DIR  = Path(os.environ.get("MODEL_DIR",  "/opt/render/project/src/models"))
LOG_DIR    = Path(os.environ.get("LOG_DIR",    "/data/logs"))
ALERT_DIR  = Path(os.environ.get("ALERT_DIR",  "/data/alerts"))
DATA_DIR   = Path(os.environ.get("DATA_DIR",   "/data/data"))
for d in [LOG_DIR, ALERT_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
MIN_PRICE       = 0.10
MAX_PRICE       = 3.00
MIN_DOLLAR_VOL  = 25_000
MAX_WATCHLIST   = 100
MIN_GAP         = 0.05
MAX_GAP         = 1.00
MIN_VOLUME      = 10_000
MIN_ROOM        = 0.20
MAX_FLOAT_M     = 50.0
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "delta_v2.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("delta_v2")
_models       = {}
_feature_cols = []
_thresholds   = {}
_watchlist    = []
_alerted      = set()
_ticker_cache = {}
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
    details = {"float_shares": -1, "float_M": -1, "float_tier": "unknown",
               "market_cap": -1, "is_foreign_listed": 0}
    if data and "results" in data:
        r = data["results"]
        shares  = r.get("share_class_shares_outstanding", 0) or 0
        float_M = shares / 1_000_000 if shares else -1
        # Match training code string values exactly
        ft = "unknown"
        if float_M > 0:
            if float_M < 5:     ft = "nano"
            elif float_M < 15:  ft = "micro"
            elif float_M < 50:  ft = "small"
            elif float_M < 200: ft = "mid"
            else:               ft = "large"
        locale     = r.get("locale", "us").lower()
        addr       = r.get("address", {})
        hq_country = addr.get("country", "us").lower() if isinstance(addr, dict) else "us"
        is_foreign = 1 if (locale != "us" or hq_country not in ("us", "usa", "united states", "")) else 0
        details = {
            "float_shares":      shares,
            "float_M":           float_M,
            "float_tier":        ft,
            "market_cap":        r.get("market_cap", -1) or -1,
            "is_foreign_listed": is_foreign,
        }
    _ticker_cache[ticker] = details
    return details
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
def build_features(snap, details, has_8k=0, edgar_features=None):
    ef = edgar_features or {}
    prev_close  = snap.get("prev_close", 0) or 0
    pm_open     = snap.get("pm_open", 0) or 0
    pm_high     = snap.get("pm_high", 0) or 0
    pm_low      = snap.get("pm_low", 0) or 0
    pm_close    = snap.get("pm_close", 0) or prev_close
    volume      = snap.get("pm_volume", 0) or snap.get("volume", 0) or 0
    gap_pct     = snap.get("pm_gap_pct", 0) or snap.get("gap_pct", 0) or 0
    prev_vol    = snap.get("prev_volume", 0) or snap.get("prev_vol", 0) or 0
    avg_vol_20d = snap.get("avg_volume_20d", 0) or 0

    # PM derived
    pm_move_pct = (pm_high - pm_open) / pm_open if pm_open > 0 else 0
    seed_tgt    = prev_close * 2.0
    super_tgt   = prev_close * 6.0
    mega_tgt    = prev_close * 11.0
    rem_seed    = (seed_tgt  - pm_close) / pm_close if pm_close > 0 else 0
    rem_super   = (super_tgt - pm_close) / pm_close if pm_close > 0 else 0
    rem_mega    = (mega_tgt  - pm_close) / pm_close if pm_close > 0 else 0
    pm_vol_ratio = volume / avg_vol_20d if avg_vol_20d > 0 and volume > 0 else 0
    vol_ratio_prev = prev_vol / avg_vol_20d if avg_vol_20d > 0 and prev_vol > 0 else 0
    pm_volume_build = snap.get("pm_volume_build", 0) or 0

    # T-1 price structure — match training calc exactly
    prev_open  = snap.get("prev_open", 0) or 0
    prev_high  = snap.get("prev_high", 0) or 0
    prev_low   = snap.get("prev_low", 0) or 0
    prev_dollar_vol = prev_close * prev_vol if prev_close > 0 and prev_vol > 0 else 0
    prev_body_pct = abs(prev_close - prev_open) / prev_open if prev_open > 0 else 0
    total_range = prev_high - prev_low
    body = abs(prev_close - prev_open)
    prev_wick_ratio = 1 - (body / total_range) if total_range > 0 else 0

    # Float rotation — float_M denominator in training used float_shares
    float_shares = details.get("float_shares", 0) or 0
    float_rotation_prev = (avg_vol_20d * prev_close / float_shares) if float_shares > 0 and prev_close > 0 and avg_vol_20d > 0 else 0

    # SI — match training string tiers exactly
    si_pct  = snap.get("si_pct", -1)
    si_tier = "unknown"
    if si_pct is not None and si_pct >= 0:
        if si_pct < 5:    si_tier = "low"
        elif si_pct < 15: si_tier = "medium"
        elif si_pct < 30: si_tier = "high"
        else:             si_tier = "extreme"

    # EDGAR — all fields from training
    has_8k_val          = has_8k
    has_8k_yesterday    = ef.get("has_8k_yesterday", 0)
    has_8k_2days_ago    = ef.get("has_8k_2days_ago", 0)
    days_since_last_8k  = ef.get("days_since_last_8k", 999)
    has_merger          = ef.get("has_merger", 0)
    has_fda             = ef.get("has_fda", 0)
    has_contract        = ef.get("has_contract", 0)
    has_dilution        = ef.get("has_dilution", 0)
    has_earnings        = ef.get("has_earnings", 0)
    has_reverse_split   = ef.get("has_reverse_split", 0)
    has_buyback         = ef.get("has_buyback", 0)
    dilution_count_6m   = ef.get("dilution_count_6m", 0)
    dilution_count_30d  = ef.get("dilution_count_30d", 0)
    days_since_dilution = ef.get("days_since_dilution", 999)
    reverse_split_count = ef.get("reverse_split_count", 0)
    is_serial_diluter   = ef.get("is_serial_diluter", 0)
    is_serial_reverser  = ef.get("is_serial_reverser", 0)
    has_form4_buy       = ef.get("has_form4_buy", 0)
    form4_buy_count     = ef.get("form4_buy_count", 0)
    has_sc13d           = ef.get("has_sc13d", 0)
    edgar_fetch_ok      = ef.get("edgar_fetch_ok", 0)
    edgar_dilution_ok   = ef.get("edgar_dilution_ok", 0)
    form4_fetch_ok      = ef.get("form4_fetch_ok", 0)
    sc13d_fetch_ok      = ef.get("sc13d_fetch_ok", 0)
    filing_hour         = ef.get("8k_filing_hour", 0)
    hours_before_open   = ef.get("hours_before_open", 0)

    # Earnings
    days_to_earnings       = snap.get("days_to_earnings", 999) or 999
    has_earnings_soon      = snap.get("has_earnings_soon", 0)
    had_earnings_recently  = snap.get("had_earnings_recently", 0)
    earnings_fetch_ok      = snap.get("earnings_fetch_ok", 0)

    # Halt features
    halted_yesterday  = snap.get("halted_yesterday", 0)
    halt_count_5d     = snap.get("halt_count_5d", 0)
    halt_count_30d    = snap.get("halt_count_30d", 0)
    is_serial_halter  = snap.get("is_serial_halter", 0)
    halt_fetch_ok     = snap.get("halt_fetch_ok", 0)
    days_since_halt   = snap.get("days_since_last_halt", 999) or 999

    # Sector
    spy_pct    = snap.get("spy_prev_day_pct", 0) or 0
    qqq_pct    = snap.get("qqq_prev_day_pct", 0) or 0
    iwm_pct    = snap.get("iwm_prev_day_pct", 0) or 0
    xbi_pct    = snap.get("xbi_prev_day_pct", 0) or 0
    market_green = 1 if spy_pct > 0.003 else 0
    market_red   = 1 if spy_pct < -0.003 else 0
    sector_hot   = 1 if xbi_pct > 0.01 else 0
    sector_fetch_ok = snap.get("sector_fetch_ok", 1)

    features = {col: 0 for col in _feature_cols}
    overrides = {
        # T-1 price structure
        "prev_close":           prev_close,
        "prev_open":            prev_open,
        "prev_high":            prev_high,
        "prev_low":             prev_low,
        "prev_volume":          prev_vol,
        "prev_dollar_vol":      prev_dollar_vol,
        "prev_body_pct":        prev_body_pct,
        "prev_wick_ratio":      prev_wick_ratio,
        # PM features
        "pm_open":              pm_open,
        "pm_high":              pm_high,
        "pm_low":               pm_low,
        "pm_close":             pm_close,
        "pm_volume":            volume,
        "pm_gap_pct":           gap_pct,
        "pm_move_pct":          pm_move_pct,
        "pm_vol_ratio":         pm_vol_ratio,
        "pm_volume_build":      pm_volume_build,
        "pm_remaining_to_seed":  rem_seed,
        "pm_remaining_to_super": rem_super,
        "pm_remaining_to_mega":  rem_mega,
        "pm_high_of_session":   1 if pm_close >= pm_high * 0.99 and pm_high > 0 else 0,
        "pm_fade":              1 if (pm_high > pm_open and pm_high > 0 and
                                (pm_high - pm_close) > (pm_high - pm_open) * 0.10) else 0,
        "pm_fetch_ok":          1 if pm_open > 0 else 0,
        # AH features
        "ah_move_pct":          snap.get("ah_move_pct", 0) or 0,
        "ah_direction":         snap.get("ah_direction", 0) or 0,
        "ah_volume":            snap.get("ah_volume", 0) or 0,
        "ah_fetch_ok":          snap.get("ah_fetch_ok", 0) or 0,
        # Historical
        "price_52w_high":       snap.get("price_52w_high", 0) or 0,
        "price_52w_low":        snap.get("price_52w_low", 0) or 0,
        "pct_from_52w_high":    snap.get("pct_from_52w_high", 0) or 0,
        "pct_from_52w_low":     snap.get("pct_from_52w_low", 0) or 0,
        "near_52w_low":         snap.get("near_52w_low", 0),
        "avg_volume_20d":       avg_vol_20d,
        "vol_ratio_prev":       vol_ratio_prev,
        "prev_3d_trend":        snap.get("prev_3d_trend", 0) or 0,
        "prev_5d_trend":        snap.get("prev_5d_trend", 0) or 0,
        "prev_10d_trend":       snap.get("prev_10d_trend", 0) or 0,
        "days_since_last_spike": snap.get("days_since_last_spike", 999) or 999,
        "coil_days":            snap.get("coil_days", 0),
        "vol_trend_3d":         snap.get("vol_trend_3d", 0) or 0,
        "consecutive_vol_days": snap.get("consecutive_vol_days", 0),
        "hist_fetch_ok":        1 if snap.get("avg_volume_20d", 0) > 0 else 0,
        # Float
        "float_shares":         float_shares,
        "float_M":              details.get("float_M", -1),
        "float_tier":           details.get("float_tier", "unknown"),
        "market_cap":           details.get("market_cap", -1),
        "is_foreign_listed":    details.get("is_foreign_listed", 0),
        "float_rotation_prev":  float_rotation_prev,
        "float_fetch_ok":       1 if float_shares > 0 else 0,
        # SI
        "si_pct":               si_pct if si_pct is not None else -1,
        "si_tier":              si_tier,
        "si_fetch_ok":          1 if (si_pct is not None and si_pct >= 0) else 0,
        # Days since events
        "days_since_last_seed":     snap.get("days_since_last_seed", 999) or 999,
        "days_since_last_8k":       days_since_last_8k,
        "days_since_last_halt":     days_since_halt,
        "days_since_last_dilution": days_since_dilution,
        # EDGAR
        "has_8k":               has_8k_val,
        "has_8k_yesterday":     has_8k_yesterday,
        "has_8k_2days_ago":     has_8k_2days_ago,
        "8k_filing_hour":       filing_hour,
        "hours_before_open":    hours_before_open,
        "has_merger":           has_merger,
        "has_fda":              has_fda,
        "has_contract":         has_contract,
        "has_dilution":         has_dilution,
        "has_earnings":         has_earnings,
        "has_reverse_split":    has_reverse_split,
        "has_buyback":          has_buyback,
        "dilution_count_6m":    dilution_count_6m,
        "dilution_count_30d":   dilution_count_30d,
        "reverse_split_count":  reverse_split_count,
        "is_serial_diluter":    is_serial_diluter,
        "is_serial_reverser":   is_serial_reverser,
        "has_form4_buy":        has_form4_buy,
        "form4_buy_count":      form4_buy_count,
        "has_sc13d":            has_sc13d,
        "edgar_fetch_ok":       edgar_fetch_ok,
        "edgar_dilution_ok":    edgar_dilution_ok,
        "form4_fetch_ok":       form4_fetch_ok,
        "sc13d_fetch_ok":       sc13d_fetch_ok,
        # Halts
        "halted_yesterday":     halted_yesterday,
        "halt_count_5d":        halt_count_5d,
        "halt_count_30d":       halt_count_30d,
        "is_serial_halter":     is_serial_halter,
        "halt_fetch_ok":        halt_fetch_ok,
        # Earnings
        "days_to_earnings":     days_to_earnings,
        "has_earnings_soon":    has_earnings_soon,
        "had_earnings_recently": had_earnings_recently,
        "earnings_fetch_ok":    earnings_fetch_ok,
        # Sector
        "spy_prev_day_pct":     spy_pct,
        "qqq_prev_day_pct":     qqq_pct,
        "iwm_prev_day_pct":     iwm_pct,
        "xbi_prev_day_pct":     xbi_pct,
        "market_green":         market_green,
        "market_red":           market_red,
        "sector_hot":           sector_hot,
        "sector_fetch_ok":      sector_fetch_ok,
    }
    for k, v in overrides.items():
        if k in features:
            features[k] = v
    return np.array([features[col] for col in _feature_cols]).reshape(1, -1)
def passes_setup_filter(snap, wl):
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
def get_si_data():
    today = date.today()
    check = today
    for _ in range(30):
        if check.day in [1,2,3,14,15,16,17] and check.weekday() < 5 and check not in HOLIDAYS:
            url = f"https://cdn.finra.org/equity/regsho/biweekly/FNRAshvol{check.strftime('%Y%m%d')}.txt"
            r = edgar_get(url)
            if r and r.status_code == 200 and len(r.text) > 1000:
                si_map = {}
                for line in r.text.strip().split("\n")[1:]:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        sym = parts[0].strip().upper()
                        try:
                            short_int = float(parts[1])
                            avg_vol   = float(parts[2]) if len(parts) > 2 else 0
                            if avg_vol > 0:
                                si_map[sym] = round(short_int / avg_vol, 2)
                        except Exception:
                            pass
                if si_map:
                    log.info(f"Loaded biweekly SI: {len(si_map)} tickers from {check}")
                    return si_map
        check -= timedelta(days=1)
    log.info("Biweekly SI not found, using daily short volume")
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
            log.info(f"Loaded daily SI: {len(si_map)} tickers from {check}")
            return si_map
    return {}
def get_edgar_8k_today():
    today = date.today()
    filings = {}
    start_date = (today - timedelta(days=7)).isoformat()
    try:
        r = edgar_get(
            "https://efts.sec.gov/LATEST/search-index?q=%228-K%22"
            f"&dateRange=custom&startdt={start_date}"
            f"&enddt={today.isoformat()}&forms=8-K"
        )
        if r and r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            log.info(f"EDGAR hits: {len(hits)}")
            import re as _re
            for hit in hits:
                src    = hit.get("_source", {})
                filed  = src.get("file_date", today.isoformat())
                ticker = ""
                display = src.get("display_names", [])
                for name in (display if isinstance(display, list) else [display]):
                    match = _re.search(r"\(([A-Z]{1,5})[,)]", str(name))
                    if match:
                        ticker = match.group(1)
                        break
                if ticker and len(ticker) <= 5 and not ticker.endswith("W"):
                    if ticker not in filings:
                        filings[ticker] = []
                    filings[ticker].append({"filed": filed, "form": "8-K"})
            log.info(f"EDGAR: resolved {len(filings)} tickers from {len(hits)} hits")
        else:
            status = r.status_code if r else "no response"
            log.warning(f"EDGAR failed: {status}")
    except Exception as e:
        log.warning(f"EDGAR error: {e}")
    log.info(f"EDGAR: {len(filings)} tickers with recent filings")
    return filings
def score_universe():
    global _watchlist
    log.info("=" * 50)
    log.info("Overnight scorer starting...")
    last_day = get_last_trading_day()
    data = poly_get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{last_day.isoformat()}",
        {"adjusted": "false"}
    )
    if not data or "results" not in data:
        log.warning("Could not load grouped daily bars")
        return
    si_map   = get_si_data()
    edgar_8k = get_edgar_8k_today()

    # Sector: pull 2 days so we get T-1 change exactly like training
    sector_data = {}
    two_days_ago = last_day - timedelta(days=5)  # go back far enough to find 2 trading days
    for sym in ["SPY", "QQQ", "IWM", "XBI"]:
        sd = poly_get(
            f"/v2/aggs/ticker/{sym}/range/1/day/{two_days_ago}/{last_day}",
            {"adjusted": "false", "limit": 5}
        )
        if sd and sd.get("results") and len(sd["results"]) >= 2:
            r0 = sd["results"][-2]  # T-2
            r1 = sd["results"][-1]  # T-1
            if r0.get("c", 0) > 0:
                sector_data[sym] = (r1.get("c", 0) - r0.get("c", 0)) / r0.get("c", 1)
            else:
                sector_data[sym] = 0
        elif sd and sd.get("results"):
            r = sd["results"][-1]
            sector_data[sym] = (r.get("c", 0) - r.get("o", 0)) / r.get("o", 1)

    spy_pct = sector_data.get("SPY", 0)
    qqq_pct = sector_data.get("QQQ", 0)
    iwm_pct = sector_data.get("IWM", 0)
    xbi_pct = sector_data.get("XBI", 0)

    candidates = []
    total = len(data["results"])
    log.info(f"Scoring {total} tickers...")

    for bar in data["results"]:
        ticker     = bar.get("T", "")
        close      = bar.get("c", 0) or 0
        volume     = bar.get("v", 0) or 0
        high       = bar.get("h", 0) or 0
        low        = bar.get("l", 0) or 0
        open_      = bar.get("o", 0) or 0
        dollar_vol = close * volume

        if not ticker or len(ticker) > 5:
            continue
        if ticker.endswith("W") or ticker.endswith("R") or "." in ticker:
            continue
        if close < MIN_PRICE or close > MAX_PRICE:
            continue
        if dollar_vol < MIN_DOLLAR_VOL:
            continue
        if volume < 5_000:
            continue

        details = get_ticker_details(ticker)
        if details.get("float_M", -1) > 100:
            continue

        # ── Historical features ───────────────────────────────
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

        recent_vols = [r.get("v", 0) for r in hist_results[-20:]] if hist_results else []
        avg_vol_20d = float(np.mean(recent_vols)) if recent_vols else 0
        vol_ratio_prev = volume / avg_vol_20d if avg_vol_20d > 0 else 0

        closes = [r.get("c", 0) for r in hist_results]
        volumes_hist = [r.get("v", 0) for r in hist_results]

        trend_3d  = (closes[-1]-closes[-4])/closes[-4]  if len(closes)>=4  and closes[-4]>0  else 0
        trend_5d  = (closes[-1]-closes[-6])/closes[-6]  if len(closes)>=6  and closes[-6]>0  else 0
        trend_10d = (closes[-1]-closes[-11])/closes[-11] if len(closes)>=11 and closes[-11]>0 else 0

        # Vol trend 3d
        vol_trend_3d = 0
        if len(volumes_hist) >= 3:
            v3 = volumes_hist[-3:]
            if v3[0] > 0:
                vol_trend_3d = (v3[-1] - v3[0]) / v3[0]

        # Consecutive above-avg vol days
        consec_vol = 0
        for v in reversed(volumes_hist):
            if avg_vol_20d > 0 and v > avg_vol_20d:
                consec_vol += 1
            else:
                break

        # Days since last volume spike (3x avg)
        spike_days = 0
        for r in reversed(hist_results):
            if avg_vol_20d > 0 and r.get("v", 0) > avg_vol_20d * 3:
                break
            spike_days += 1

        # Coil days — consecutive below-avg volume compression
        coil = 0
        if len(hist_results) >= 5:
            for r in reversed(hist_results):
                if avg_vol_20d > 0 and r.get("v", 0) < avg_vol_20d * 0.8:
                    coil += 1
                else:
                    break

        # T-1 price structure (bar = last_day bar)
        prev_body_pct  = abs(close - open_) / open_ if open_ > 0 else 0
        total_range    = high - low
        body           = abs(close - open_)
        prev_wick_ratio = 1 - (body / total_range) if total_range > 0 else 0
        prev_dollar_vol = close * volume

        # ── AH features from Polygon snapshot ────────────────
        ah_move_pct  = 0
        ah_direction = 0
        ah_volume    = 0
        ah_fetch_ok  = 0
        snap_data = poly_get(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            {"tickers": ticker, "include_otc": "false"}
        )
        if snap_data and snap_data.get("tickers"):
            t_snap = snap_data["tickers"][0]
            ah = t_snap.get("afterHours", {})
            if ah and ah.get("c") and ah.get("o"):
                ah_o = float(ah.get("o", 0))
                ah_c = float(ah.get("c", 0))
                ah_v = float(ah.get("v", 0))
                if ah_o > 0:
                    ah_move_pct  = (ah_c - ah_o) / ah_o
                    ah_direction = 1 if ah_c > ah_o else (-1 if ah_c < ah_o else 0)
                    ah_volume    = ah_v
                    ah_fetch_ok  = 1

        # ── SI features ───────────────────────────────────────
        si_pct  = si_map.get(ticker, -1)
        si_tier = "unknown"
        if si_pct >= 0:
            if si_pct < 5:    si_tier = "low"
            elif si_pct < 15: si_tier = "medium"
            elif si_pct < 30: si_tier = "high"
            else:             si_tier = "extreme"

        # ── EDGAR features ────────────────────────────────────
        ef = {
            "days_since_last_8k":   999,
            "has_8k_yesterday":     0,
            "has_8k_2days_ago":     0,
            "8k_filing_hour":       0,
            "hours_before_open":    0,
            "has_merger":           0,
            "has_fda":              0,
            "has_contract":         0,
            "has_dilution":         0,
            "has_earnings":         0,
            "has_reverse_split":    0,
            "has_buyback":          0,
            "dilution_count_6m":    0,
            "dilution_count_30d":   0,
            "days_since_dilution":  999,
            "reverse_split_count":  0,
            "is_serial_diluter":    0,
            "is_serial_reverser":   0,
            "has_form4_buy":        0,
            "form4_buy_count":      0,
            "has_sc13d":            0,
            "edgar_fetch_ok":       0,
            "edgar_dilution_ok":    0,
            "form4_fetch_ok":       0,
            "sc13d_fetch_ok":       0,
        }
        if ticker in edgar_8k:
            filings = edgar_8k[ticker]
            latest  = max(filings, key=lambda x: x.get("filed", ""))
            try:
                filed_str  = latest["filed"]
                filed_date = date.fromisoformat(filed_str[:10])
                days_since = (last_day - filed_date).days
                ef["days_since_last_8k"]  = days_since
                ef["has_8k_yesterday"]    = 1 if days_since <= 1 else 0
                ef["has_8k_2days_ago"]    = 1 if days_since == 2 else 0
                ef["edgar_fetch_ok"]      = 1
                # Filing hour — parse if time is in the filed string
                if "T" in filed_str:
                    try:
                        filed_dt = datetime.fromisoformat(filed_str)
                        ef["8k_filing_hour"]    = filed_dt.hour
                        # Hours before 9:30 AM open
                        open_hour = 9.5
                        filing_hour_et = filed_dt.hour + filed_dt.minute / 60
                        ef["hours_before_open"] = max(0, open_hour - filing_hour_et)
                    except Exception:
                        pass
                # Parse filing text for keywords
                text = str(latest).lower()
                ef["has_merger"]        = 1 if any(w in text for w in ["merger", "acqui", "takeover"]) else 0
                ef["has_fda"]           = 1 if "fda" in text else 0
                ef["has_contract"]      = 1 if "contract" in text else 0
                ef["has_dilution"]      = 1 if any(w in text for w in ["dilut", "offering", "424b"]) else 0
                ef["has_reverse_split"] = 1 if "reverse split" in text else 0
                ef["has_buyback"]       = 1 if "buyback" in text or "repurchas" in text else 0
            except Exception:
                pass

        has_8k = 1 if ef["days_since_last_8k"] <= 3 else 0

        snap = {
            # T-1 price
            "prev_close":           close,
            "prev_open":            open_,
            "prev_high":            high,
            "prev_low":             low,
            "prev_volume":          volume,
            "prev_dollar_vol":      prev_dollar_vol,
            "prev_body_pct":        prev_body_pct,
            "prev_wick_ratio":      prev_wick_ratio,
            # PM — zeroed at midnight, filled at 4AM rescore
            "pm_open":              0,
            "pm_high":              0,
            "pm_low":               0,
            "pm_close":             close,
            "pm_volume":            0,
            "pm_gap_pct":           0,
            # AH from snapshot
            "ah_move_pct":          ah_move_pct,
            "ah_direction":         ah_direction,
            "ah_volume":            ah_volume,
            "ah_fetch_ok":          ah_fetch_ok,
            # SI
            "si_pct":               si_pct,
            "si_tier":              si_tier,
            # Historical
            "price_52w_high":       price_52w_high,
            "price_52w_low":        price_52w_low,
            "pct_from_52w_high":    pct_52w_high,
            "pct_from_52w_low":     pct_52w_low,
            "near_52w_low":         near_52w_low,
            "avg_volume_20d":       avg_vol_20d,
            "vol_ratio_prev":       vol_ratio_prev,
            "prev_3d_trend":        trend_3d,
            "prev_5d_trend":        trend_5d,
            "prev_10d_trend":       trend_10d,
            "vol_trend_3d":         vol_trend_3d,
            "consecutive_vol_days": consec_vol,
            "days_since_last_spike": spike_days,
            "coil_days":            coil,
            # Sector
            "spy_prev_day_pct":     spy_pct,
            "qqq_prev_day_pct":     qqq_pct,
            "iwm_prev_day_pct":     iwm_pct,
            "xbi_prev_day_pct":     xbi_pct,
            "sector_fetch_ok":      1 if sector_data else 0,
        }

        fvec = build_features(snap, details, has_8k=has_8k, edgar_features=ef)
        scores, _ = score_ticker(fvec)

        candidates.append({
            "ticker":        ticker,
            "prev_close":    close,
            "prev_vol":      volume,
            "seed_score":    scores.get("seed", 0),
            "super_score":   scores.get("super", 0),
            "mega_score":    scores.get("mega", 0),
            "float_M":       details.get("float_M", -1),
            "si_pct":        si_pct,
            "near_52w_low":  near_52w_low,
            "coil_days":     coil,
            "has_8k":        has_8k,
            "days_since_8k": ef["days_since_last_8k"],
            "ef":            ef,
            "snap_base":     snap,
            "details":       details,
        })

    candidates.sort(key=lambda x: x["seed_score"], reverse=True)
    _watchlist = candidates[:MAX_WATCHLIST]

    wl_file = DATA_DIR / f"{date.today().isoformat()}_watchlist.json"
    with open(wl_file, "w") as f:
        json.dump([{
            "ticker":       w["ticker"],
            "prev_close":   w["prev_close"],
            "seed_score":   w["seed_score"],
            "super_score":  w["super_score"],
            "mega_score":   w["mega_score"],
            "float_M":      w["float_M"],
            "si_pct":       w["si_pct"],
            "has_8k":       w["has_8k"],
            "near_52w_low": w["near_52w_low"],
        } for w in _watchlist], f, indent=2)

    log.info(f"Overnight score complete. Top {len(_watchlist)} watchlist:")
    for w in _watchlist[:5]:
        log.info(f"  {w['ticker']}: seed={w['seed_score']:.3f} "
                 f"super={w['super_score']:.3f} "
                 f"float={w['float_M']:.1f}M "
                 f"si={w['si_pct']:.1f}% "
                 f"8k={w['has_8k']}")
def get_watchlist_snapshots(use_today_close=False):
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
        # Use lastTrade/lastQuote for pre-market price if day.c is stale
        last_trade = t.get("lastTrade", {})
        last_quote = t.get("lastQuote", {})
        pm_close = (
            day.get("c") or
            min_bar.get("c") or
            last_trade.get("p") or
            last_quote.get("P") or
            0
        )
        change   = t.get("todaysChangePerc", 0) or 0
        if pm_close <= 0 and change != 0:
            pm_close = prev_close * (1 + change / 100)
        if pm_close <= 0:
            continue
        pm_open = day.get("o", 0) or pm_close
        pm_high = day.get("h", 0) or pm_close
        pm_low  = day.get("l", 0) or pm_close
        volume  = day.get("v", 0) or min_bar.get("v", 0) or 0
        prev_vol_day = prev.get("v", 0) or 0
        if volume == 0 and prev_vol_day > 0:
            volume = prev_vol_day
        ref_price = day.get("c", 0) or prev_close if use_today_close else prev_close
        gap = (pm_close - ref_price) / ref_price if ref_price > 0 else 0
        if gap < MIN_GAP:
            continue
        if volume > 0 and volume < MIN_VOLUME:
            continue
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
        passes, reason = passes_setup_filter(snap, wl)
        if not passes:
            log.debug(f"{ticker}: skip — {reason}")
            continue
        overnight_seed  = wl.get("seed_score", 0)
        overnight_super = wl.get("super_score", 0)
        overnight_mega  = wl.get("mega_score", 0)
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
    log.info(f"Basecamp: {len(hits)} 8-K filings found")
    fired = 0
    import re as _re_bc
    for hit in hits:
        src = hit.get("_source", {})
        ticker = ""
        display = src.get("display_names", [])
        for name in (display if isinstance(display, list) else [display]):
            match = _re_bc.search(r"\(([A-Z]{1,5})[,)]", str(name))
            if match:
                ticker = match.group(1)
                break
        if not ticker or already_alerted(ticker):
            continue
        if len(ticker) > 5 or ticker.endswith("W") or "." in ticker:
            continue
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
            log.debug(f"Basecamp {ticker}: price ${prev_close:.2f} outside range")
            continue
        details = get_ticker_details(ticker)
        float_M = details.get("float_M", -1)
        if float_M > 100:
            log.debug(f"Basecamp {ticker}: float {float_M:.0f}M too large")
            continue
        pm_close = day.get("c", 0) or prev_close
        volume   = day.get("v", 0) or 0
        gap      = (pm_close - prev_close) / prev_close if prev_close > 0 else 0
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
            "has_merger":         1 if "merger"   in str(src).lower() else 0,
            "has_fda":            1 if "fda"      in str(src).lower() else 0,
            "has_contract":       1 if "contract" in str(src).lower() else 0,
            "has_dilution":       1 if "dilut"    in str(src).lower() else 0,
        }
        fvec = build_features(snap, details, has_8k=1, edgar_features=ef)
        scores, alerts = score_ticker(fvec)
        if not alerts:
            log.debug(f"Basecamp {ticker}: below threshold seed={scores.get('seed',0):.3f}")
            continue
        alert_type, score_val = alerts[0]
        log_alert(ticker, alert_type, score_val, scores, snap, "basecamp")
        title, msg = format_alert(
            ticker, alert_type, score_val, scores, snap, details, "8-K tonight"
        )
        priority = 1 if alert_type in ("super", "mega") else 0
        send_pushover(title, msg, alert_type, priority)
        log.info(
            f"BASECAMP: {ticker} {alert_type.upper()} "
            f"score={score_val:.3f} float={float_M:.1f}M gap={gap*100:+.1f}%"
        )
        fired += 1
    log.info(f"Basecamp scan complete: {fired} alerts fired")
def run_daily_validation():
    log.info("Running 4PM daily validation + training data collection...")
    today    = date.today()
    last_day = get_last_trading_day()
    wl_tickers = [w["ticker"] for w in _watchlist]
    training_records = []
    caught = []
    missed = []
    log.info("Path 1: pulling intraday highs for watchlist...")
    if _watchlist:
        ticker_str = ",".join(wl_tickers)
        snaps = poly_get(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            {"tickers": ticker_str}
        )
        if snaps and "tickers" in snaps:
            for t in snaps["tickers"]:
                ticker        = t.get("ticker", "")
                day           = t.get("day", {})
                intraday_high = day.get("h", 0) or 0
                wl = next((w for w in _watchlist if w["ticker"] == ticker), None)
                if not wl:
                    continue
                prev_close = wl.get("prev_close", 0) or 0
                if prev_close <= 0:
                    continue
                actual_pct = ((intraday_high - prev_close) / prev_close * 100) if prev_close > 0 else 0
                hit_seed   = 1 if intraday_high >= prev_close * 2.0  else 0
                hit_super  = 1 if intraday_high >= prev_close * 6.0  else 0
                hit_mega   = 1 if intraday_high >= prev_close * 11.0 else 0
                record = {
                    "date":            today.isoformat(),
                    "ticker":          ticker,
                    "path":            "watchlist",
                    "prev_close":      prev_close,
                    "intraday_high":   intraday_high,
                    "actual_pct":      round(actual_pct, 2),
                    "hit_seed":        hit_seed,
                    "hit_super":       hit_super,
                    "hit_mega":        hit_mega,
                    "overnight_score": wl.get("seed_score", 0),
                    "float_M":         wl.get("float_M", -1),
                    "si_pct":          wl.get("si_pct", -1),
                    "has_8k":          wl.get("has_8k", 0),
                    "near_52w_low":    wl.get("near_52w_low", 0),
                    "features":        wl.get("snap_base", {}),
                }
                training_records.append(record)
                log.info(f"Path1 {ticker}: high={intraday_high:.2f} pct={actual_pct:.0f}% seed={hit_seed}")
    log.info("Path 2: pulling gainers for full feature training...")
    gainers = poly_get(
        "/v2/snapshot/locale/us/markets/stocks/gainers",
        {"include_otc": "false"}
    )
    if gainers and "tickers" in gainers:
        for t in gainers["tickers"]:
            ticker        = t.get("ticker", "")
            day           = t.get("day", {})
            prev          = t.get("prevDay", {})
            prev_close    = prev.get("c", 0) or 0
            intraday_high = day.get("h", 0) or 0
            if not ticker or prev_close <= 0:
                continue
            if len(ticker) > 5 or ticker.endswith("W") or "." in ticker:
                continue
            if prev_close < MIN_PRICE or prev_close > MAX_PRICE:
                continue
            if intraday_high < prev_close * 2.0:
                continue
            actual_pct = (intraday_high - prev_close) / prev_close * 100
            if ticker in wl_tickers:
                if ticker not in caught:
                    caught.append(ticker)
            else:
                if ticker not in missed:
                    missed.append(ticker)
            details = get_ticker_details(ticker)
            float_M = details.get("float_M", -1)
            if float_M > 100:
                continue
            hist = poly_get(
                f"/v2/aggs/ticker/{ticker}/range/1/day/"
                f"{(last_day - timedelta(days=365)).isoformat()}/{last_day.isoformat()}",
                {"adjusted": "false", "limit": 365}
            )
            hist_results   = hist.get("results", []) if hist else []
            price_52w_high = max([r.get("h",0) for r in hist_results], default=prev_close)
            price_52w_low  = min([r.get("l",0) for r in hist_results], default=prev_close)
            pct_52w_high   = (prev_close-price_52w_high)/price_52w_high if price_52w_high > 0 else 0
            pct_52w_low    = (prev_close-price_52w_low)/price_52w_low   if price_52w_low  > 0 else 0
            near_52w_low   = 1 if prev_close < price_52w_low * 1.10 else 0
            recent_vols    = [r.get("v",0) for r in hist_results[-20:]] if hist_results else []
            avg_vol_20d    = float(np.mean(recent_vols)) if recent_vols else 0
            prev_volume    = prev.get("v", 0) or 0
            vol_ratio      = prev_volume / avg_vol_20d if avg_vol_20d > 0 else 0
            closes         = [r.get("c",0) for r in hist_results]
            trend_3d = (closes[-1]-closes[-4])/closes[-4] if len(closes)>=4 and closes[-4]>0 else 0
            trend_5d = (closes[-1]-closes[-6])/closes[-6] if len(closes)>=6 and closes[-6]>0 else 0
            coil = 0
            if len(hist_results) >= 5:
                ranges = [r.get("h",0)-r.get("l",0) for r in hist_results[-10:]]
                for i in range(len(ranges)-1, 0, -1):
                    if ranges[i] <= ranges[i-1]: coil += 1
                    else: break
            path = "caught" if ticker in wl_tickers else "miss"
            record = {
                "date":            today.isoformat(),
                "ticker":          ticker,
                "path":            path,
                "prev_close":      prev_close,
                "intraday_high":   intraday_high,
                "actual_pct":      round(actual_pct, 2),
                "hit_seed":        1,
                "hit_super":       1 if intraday_high >= prev_close * 6.0  else 0,
                "hit_mega":        1 if intraday_high >= prev_close * 11.0 else 0,
                "overnight_score": 0,
                "float_M":         float_M,
                "features": {
                    "prev_close":        prev_close,
                    "prev_volume":       prev_volume,
                    "avg_volume_20d":    avg_vol_20d,
                    "vol_ratio_prev":    vol_ratio,
                    "prev_3d_trend":     trend_3d,
                    "prev_5d_trend":     trend_5d,
                    "float_M":           float_M,
                    "float_tier":        details.get("float_tier", -1),
                    "float_shares":      details.get("float_shares", -1),
                    "market_cap":        details.get("market_cap", -1),
                    "is_foreign_listed": details.get("is_foreign_listed", 0),
                    "pct_from_52w_high": pct_52w_high,
                    "pct_from_52w_low":  pct_52w_low,
                    "near_52w_low":      near_52w_low,
                    "price_52w_high":    price_52w_high,
                    "price_52w_low":     price_52w_low,
                    "coil_days":         coil,
                    "si_pct":            -1,
                    "has_8k":            0,
                    "days_since_last_8k": 999,
                },
            }
            training_records.append(record)
            log.info(f"Path2 {path.upper()}: {ticker} pct={actual_pct:.0f}% float={float_M:.1f}M 52wL={near_52w_low}")
    if training_records:
        outcomes_file = DATA_DIR / "training_outcomes.jsonl"
        with open(outcomes_file, "a") as f:
            for record in training_records:
                f.write(json.dumps(record) + "\n")
        log.info(f"Saved {len(training_records)} training records to {outcomes_file}")
    total_seeds = len(caught) + len(missed)
    recall      = len(caught) / total_seeds if total_seeds > 0 else 0
    log.info(f"Validation: {total_seeds} seeds | caught={len(caught)} | missed={len(missed)} | recall={recall:.0%}")
    lines = [f"📊 Daily Report — {today}"]
    lines.append(f"Seeds today: {total_seeds}")
    lines.append(f"Caught: {len(caught)} ({recall:.0%} recall)")
    if caught:
        lines.append(f"✅ {', '.join(caught)}")
    if missed:
        lines.append(f"❌ Missed: {', '.join(missed[:5])}")
    lines.append(f"Training records: {len(training_records)}")
    send_pushover("📊 Delta v2 Daily Report", "\n".join(lines), "seed", priority=0)
    val_file = ALERT_DIR / f"{today.isoformat()}_validation.json"
    with open(val_file, "w") as f:
        json.dump({
            "date":             today.isoformat(),
            "caught":           caught,
            "missed":           missed,
            "recall":           round(recall, 3),
            "total_seeds":      total_seeds,
            "training_records": len(training_records),
            "watchlist":        wl_tickers,
        }, f, indent=2)
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
def score_ticker_on_demand(ticker):
    ticker = ticker.upper().strip()
    log.info(f"Test scoring: {ticker}")
    snaps = poly_get(
        "/v2/snapshot/locale/us/markets/stocks/tickers",
        {"tickers": ticker}
    )
    if not snaps or "tickers" not in snaps or not snaps["tickers"]:
        return {"error": f"No snapshot data for {ticker}"}
    t       = snaps["tickers"][0]
    day     = t.get("day", {})
    prev    = t.get("prevDay", {})
    min_bar = t.get("min", {})
    prev_close = prev.get("c", 0) or 0
    pm_close   = day.get("c", 0) or min_bar.get("c", 0) or 0
    change     = t.get("todaysChangePerc", 0) or 0
    if pm_close <= 0 and change != 0 and prev_close > 0:
        pm_close = prev_close * (1 + change / 100)
    volume = day.get("v", 0) or min_bar.get("v", 0) or 0
    if volume == 0:
        volume = prev.get("v", 0) or 0
    gap     = (pm_close - prev_close) / prev_close if prev_close > 0 else 0
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
        "ticker":       ticker,
        "prev_close":   prev_close,
        "price":        pm_close,
        "gap_pct":      round(gap * 100, 2),
        "volume":       volume,
        "float_M":      details.get("float_M", -1),
        "room_to_seed": round(remaining, 2),
        "scores": {
            "seed":  scores.get("seed", 0),
            "super": scores.get("super", 0),
            "mega":  scores.get("mega", 0),
        },
        "alert":        alerts[0][0] if alerts else "none",
        "on_watchlist": ticker in [w["ticker"] for w in _watchlist],
    }
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            wl_tickers = [w["ticker"] for w in _watchlist]
            self.wfile.write(
                f"Delta v2 alive\nWatchlist ({len(_watchlist)}): {', '.join(wl_tickers)}\n"
                f"Alerted today: {len(_alerted)}\n".encode()
            )
            return
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
        if parsed.path == "/watchlist":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            wl = [{
                "ticker":       w["ticker"],
                "prev_close":   w["prev_close"],
                "seed_score":   w["seed_score"],
                "super_score":  w["super_score"],
                "mega_score":   w["mega_score"],
                "float_M":      w["float_M"],
                "si_pct":       w["si_pct"],
                "has_8k":       w["has_8k"],
                "near_52w_low": w["near_52w_low"],
            } for w in _watchlist]
            self.wfile.write(json.dumps(wl, indent=2).encode())
            return
        if parsed.path == "/features":
            params = parse_qs(parsed.query)
            ticker = params.get("ticker", [""])[0].upper().strip()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if not ticker:
                # No ticker — summary of all watchlist feature coverage
                summary = []
                for w in _watchlist:
                    snap = w.get("snap_base", {})
                    summary.append({
                        "ticker":          w["ticker"],
                        "seed_score":      w["seed_score"],
                        "super_score":     w["super_score"],
                        "gap_pct":         snap.get("gap_pct", 0),
                        "volume":          snap.get("volume", 0),
                        "avg_vol_20d":     snap.get("avg_volume_20d", 0),
                        "ah_move_pct":     snap.get("ah_move_pct", 0),
                        "ah_volume":       snap.get("ah_volume", 0),
                        "si_pct":          snap.get("si_pct", -1),
                        "near_52w_low":    snap.get("near_52w_low", 0),
                        "coil_days":       snap.get("coil_days", 0),
                        "prev_3d_trend":   snap.get("prev_3d_trend", 0),
                        "prev_5d_trend":   snap.get("prev_5d_trend", 0),
                        "has_8k":          w.get("has_8k", 0),
                        "days_since_8k":   w.get("days_since_8k", 999),
                        "float_M":         w.get("float_M", -1),
                    })
                self.wfile.write(json.dumps(summary, indent=2).encode())
                return
            # Specific ticker — full feature vector breakdown
            wl = next((w for w in _watchlist if w["ticker"] == ticker), None)
            if not wl:
                self.wfile.write(json.dumps({
                    "error": f"{ticker} not on watchlist",
                    "watchlist_size": len(_watchlist),
                    "watchlist_tickers": [w["ticker"] for w in _watchlist],
                }).encode())
                return
            snap    = wl.get("snap_base", {})
            details = wl.get("details", {})
            ef      = wl.get("ef", {})
            fvec    = build_features(snap, details,
                                     has_8k=wl.get("has_8k", 0),
                                     edgar_features=ef)
            feature_dict = {
                col: round(float(fvec[0][i]), 6)
                for i, col in enumerate(_feature_cols)
            }
            zeros    = [k for k, v in feature_dict.items() if v == 0]
            nonzeros = {k: v for k, v in feature_dict.items() if v != 0}
            self.wfile.write(json.dumps({
                "ticker":        ticker,
                "seed_score":    wl["seed_score"],
                "super_score":   wl["super_score"],
                "mega_score":    wl["mega_score"],
                "snap_inputs":   snap,
                "filled":        nonzeros,
                "zeroed_out":    zeros,
                "zero_count":    len(zeros),
                "filled_count":  len(nonzeros),
                "total_features": len(_feature_cols),
            }, indent=2).encode())
            return
        self.send_response(404)
        self.end_headers()
    def log_message(self, *args): pass
def start_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
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
    log.info("Building initial watchlist...")
    try:
        score_universe()
    except Exception as e:
        log.error(f"Overnight scorer failed: {e}")
    morning_summary_sent    = False
    nightly_summary_sent    = False
    confirmation_score_sent = False
    last_basecamp_scan      = None
    last_score_date         = date.today()
    last_date               = None
    while True:
        now    = datetime.now(ET)
        hour   = now.hour
        minute = now.minute
        if last_date != now.date():
            morning_summary_sent    = False
            nightly_summary_sent    = False
            confirmation_score_sent = False
            last_basecamp_scan      = None
            _alerted.clear()
            _ticker_cache.clear()
            last_date = now.date()
            log.info(f"New day: {now.date()}")
        if hour == 0 and minute < 5 and last_score_date != now.date():
            log.info("Midnight — running overnight scorer...")
            try:
                score_universe()
                last_score_date = now.date()
            except Exception as e:
                log.error(f"Overnight scorer error: {e}")
        if hour == 22 and minute < 5 and not nightly_summary_sent:
            send_nightly_summary()
            nightly_summary_sent = True
        if hour == 5 and minute < 5 and not morning_summary_sent:
            send_morning_summary()
            morning_summary_sent = True
        in_confirmation = hour == 6 and 30 <= minute <= 59
        if in_confirmation and not confirmation_score_sent:
            log.info("6:30AM confirmation window — running focused scan")
            fired = run_scan("premarket_confirmation")
            log.info(f"Confirmation scan: {fired} alerts")
            confirmation_score_sent = True
        in_premarket = hour >= 4 and (hour < 9 or (hour == 9 and minute < 30))
        if in_premarket:
            fired = run_scan("premarket")
            log.info(f"Premarket scan done: {fired} alerts")
            time.sleep(SCAN_INTERVAL)
            continue
        if hour == 16 and minute < 5:
            try:
                run_daily_validation()
            except Exception as e:
                log.error(f"Validation error: {e}")
        in_post_market = hour >= 16 and hour < 20
        if in_post_market:
            time.sleep(300)
            continue
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
        log.debug(f"Resting ({hour}:{minute:02d} ET)")
        time.sleep(300)
if __name__ == "__main__":
    main()
