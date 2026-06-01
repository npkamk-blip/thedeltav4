"""
THE DELTA v2 — main.py
=======================
Two-model architecture:
  11PM  — midnight model scores universe -> builds watchlist
  5:30AM — morning model scores watchlist -> fires alerts

Models loaded from: /app/data/models/
Data pulled from:   Polygon API
"""

import os, time, json, logging, threading, requests, gc
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from http.server import HTTPServer, BaseHTTPRequestHandler
import xgboost as xgb

ET = ZoneInfo("America/New_York")

POLYGON_API_KEY    = os.environ.get("MASSIVE_API_KEY", "")
PUSHOVER_USER_KEY  = os.environ.get("PUSHOVER_USER_KEY", "utvy26j5q66kae27ncwxsftfcuhi92")
PUSHOVER_APP_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN", "a3szzncpvgyevbck6z5z5yszm7nzg3")

MODEL_DIR   = Path(os.environ.get("MODEL_DIR",   "/opt/render/project/src/models"))
SUPPORT_DIR = Path(os.environ.get("SUPPORT_DIR", "/opt/render/project/src/support"))
LOG_DIR     = Path(os.environ.get("LOG_DIR",     "/data/logs"))
ALERT_DIR   = Path(os.environ.get("ALERT_DIR",   "/data/alerts"))

for d in [LOG_DIR, ALERT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MIN_PRICE      = 0.10
MAX_PRICE      = 5.00
MIN_DOLLAR_VOL = 25_000
MAX_WATCHLIST  = 150
MAX_FLOAT_M    = 200.0

MIDNIGHT_THRESHOLD      = 0.45
MORNING_SEED_THRESHOLD  = 0.55
MORNING_SUPER_THRESHOLD = 0.50

MORNING_SCORE_HOUR = 5
MORNING_SCORE_MIN  = 30
SCAN_INTERVAL      = 60

HOLIDAYS = {
    date(2025,1,1),date(2025,1,9),date(2025,1,20),date(2025,2,17),
    date(2025,4,18),date(2025,5,26),date(2025,6,19),date(2025,7,4),
    date(2025,9,1),date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

SOUNDS = {"seed": "cashregister", "super": "siren", "midnight": "echo"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scanner.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("scanner")

_models       = {}
_feature_cols = {}
_thresholds   = {}
_watchlist    = {}
_alerted      = set()


def download_models():
    """Download model files from data service if not already present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        "midnight_seed_model.json",
        "midnight_super_model.json",
        "morning_seed_model.json",
        "morning_super_model.json",
        "feature_cols.json",
        "thresholds.json",
    ]
    all_ok = True
    for filename in files:
        dest = MODEL_DIR / filename
        if dest.exists():
            log.info(f"Model already exists: {filename}")
            continue
        url = f"{DATA_SERVICE_URL}/models/{filename}"
        log.info(f"Downloading: {url}")
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(r.content)
                log.info(f"Downloaded: {filename} ({len(r.content):,} bytes)")
            else:
                log.error(f"FAIL download {filename}: HTTP {r.status_code}")
                all_ok = False
        except Exception as e:
            log.error(f"FAIL download {filename}: {e}")
            all_ok = False
    return all_ok


def load_models():
    global _models, _feature_cols, _thresholds
    for name in ["midnight_seed_model","midnight_super_model",
                 "morning_seed_model","morning_super_model"]:
        path = MODEL_DIR / f"{name}.json"
        if path.exists():
            try:
                m = xgb.XGBClassifier()
                m.load_model(str(path))
                _models[name] = m
                log.info(f"Loaded: {name}")
            except Exception as e:
                log.error(f"FAIL load {name}: {e}")
        else:
            log.warning(f"Model not found: {path}")
    fc = MODEL_DIR / "feature_cols.json"
    if fc.exists():
        with open(fc) as f:
            _feature_cols = json.load(f)
    th = MODEL_DIR / "thresholds.json"
    if th.exists():
        with open(th) as f:
            _thresholds = json.load(f)
    log.info(f"Models: {list(_models.keys())}")
    log.info(f"Features: { {k:len(v) for k,v in _feature_cols.items()} }")


class PolygonClient:
    CALL_INTERVAL = 60.0 / 250

    def __init__(self):
        self._last   = 0.0
        self.session = requests.Session()
        self.session.params = {"apiKey": POLYGON_API_KEY}

    def _wait(self):
        elapsed = time.time() - self._last
        if elapsed < self.CALL_INTERVAL:
            time.sleep(self.CALL_INTERVAL - elapsed)
        self._last = time.time()

    def get(self, path, params=None, retries=3):
        url = "https://api.polygon.io" + path
        for attempt in range(retries):
            self._wait()
            try:
                r = self.session.get(url, params=params or {}, timeout=20)
                if r.status_code == 200:   return r.json()
                if r.status_code == 429:   time.sleep(30*(attempt+1))
                if r.status_code == 404:   return None
            except Exception as e:
                log.warning(f"HTTP error {path}: {e}")
                time.sleep(2)
        return None

    def paginate(self, path, params=None):
        results = []
        resp = self.get(path, params)
        if not resp: return results
        results.extend(resp.get("results", []))
        while resp.get("next_url"):
            self._wait()
            try:
                r = self.session.get(resp["next_url"], timeout=30)
                resp = r.json() if r.status_code == 200 else {}
                results.extend(resp.get("results", []))
            except Exception: break
        return results

    def aggs(self, ticker, mult, span, frm, to, adjusted=False):
        return self.paginate(
            f"/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{frm}/{to}",
            {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000}
        )

    def get_tickers(self):
        results = self.paginate("/v3/reference/tickers", {
            "market": "stocks",
            "active": "true", "type": "CS", "limit": 1000,
        })
        tickers = []
        for r in results:
            t = r.get("ticker","")
            if not t or len(t) > 5: continue
            if any(t.endswith(x) for x in ["W","R","U"]): continue
            tickers.append(t)
        log.info(f"Universe: {len(tickers)} tickers")
        return tickers

poly = PolygonClient()


def get_prev_close(ticker):
    now = datetime.now(ET)
    bars = poly.aggs(ticker, 1, "day",
                     (now-timedelta(days=10)).strftime("%Y-%m-%d"),
                     (now-timedelta(days=1)).strftime("%Y-%m-%d"))
    if not bars: return 0, 0
    bars.sort(key=lambda x: x.get("t",0))
    last = bars[-1]
    return float(last.get("c",0) or 0), float(last.get("v",0) or 0)


def get_pm_bars(ticker):
    import pandas as pd
    now   = datetime.now(ET)
    today = now.strftime("%Y-%m-%d")
    bars  = poly.aggs(ticker, 1, "minute", today, today)
    pm = []
    for b in bars:
        t = pd.Timestamp(b["t"], unit="ms", tz="UTC").tz_convert(ET)
        h, m = t.hour, t.minute
        if 4 <= h < 9 or (h == 9 and m < 30):
            pm.append({**b, "hour": h, "minute": m})
    return sorted(pm, key=lambda x: x["t"])


def get_ah_bars(ticker):
    import pandas as pd
    now  = datetime.now(ET)
    yest = (now-timedelta(days=1)).strftime("%Y-%m-%d")
    bars = poly.aggs(ticker, 1, "minute", yest, yest)
    ah = []
    for b in bars:
        t = pd.Timestamp(b["t"], unit="ms", tz="UTC").tz_convert(ET)
        h = t.hour
        if h >= 16:
            ah.append({**b, "hour": h, "minute": t.minute})
    return sorted(ah, key=lambda x: x["t"])


def get_float(ticker):
    resp = poly.get(f"/v3/reference/tickers/{ticker}")
    if not resp or not resp.get("results"): return -1, 0, 0
    d  = resp["results"]
    sh = float(d.get("share_class_shares_outstanding",0) or 0)
    mc = float(d.get("market_cap",0) or 0)
    is_foreign = 1 if d.get("locale","us") != "us" else 0
    if sh > 0: return sh/1_000_000, mc, is_foreign
    if mc > 0:
        pc, _ = get_prev_close(ticker)
        if pc > 0: return mc/pc/1_000_000, mc, is_foreign
    return -1, mc, is_foreign


# SI lookup cache — loaded once on startup
_si_lookup = {}

def load_si_lookup():
    global _si_lookup
    si_path = SUPPORT_DIR / "si_lookup.parquet"
    if not si_path.exists():
        log.warning(f"SI lookup not found: {si_path}")
        return
    try:
        import pandas as pd
        df = pd.read_parquet(si_path)
        for _, row in df.iterrows():
            sym = str(row.get("Symbol",""))
            if sym:
                _si_lookup[sym] = {
                    "si_pct":  float(row.get("si_pct", -1) or -1),
                    "si_tier": int(row.get("si_tier", -1) or -1),
                }
        log.info(f"SI lookup loaded: {len(_si_lookup):,} tickers")
    except Exception as e:
        log.error(f"FAIL load SI lookup: {e}")


def get_si(ticker):
    entry = _si_lookup.get(ticker)
    if entry:
        return entry["si_pct"], entry["si_tier"]
    return -1, -1


def get_edgar_features(ticker):
    out = {
        "has_8k":0,"has_8k_yesterday":0,"has_8k_2days_ago":0,
        "8k_filing_hour":0,"hours_before_open":0,
        "has_dilution":0,"dilution_count_6m":0,"dilution_count_30d":0,
        "is_serial_diluter":0,"has_form4_buy":0,"has_sc13d":0,
        "days_since_dilution":999,"form4_buy_count":0,
        "has_merger":0,"has_fda":0,"has_contract":0,
        "has_reverse_split":0,"has_buyback":0,
    }
    ep = SUPPORT_DIR / "edgar_recent.parquet"
    cp = SUPPORT_DIR / "cik_map.json"
    if not ep.exists() or not cp.exists(): return out
    try:
        import pandas as pd
        with open(cp) as f: cik_map = json.load(f)
        cik = cik_map.get(ticker)
        if not cik: return out
        df   = pd.read_parquet(ep, columns=["cik","form_type","filed","filename"])
        rows = df[df["cik"]==cik].copy()
        if rows.empty: return out
        rows["fd"] = pd.to_datetime(rows["filed"],errors="coerce").dt.date
        today = date.today()
        yest  = today - timedelta(days=1)
        two   = today - timedelta(days=2)
        six_m = today - timedelta(days=180)
        thirty= today - timedelta(days=30)
        ek = rows[rows["form_type"].isin(["8-K","8-K/A"])]
        if not ek[ek["fd"]==today].empty:  out["has_8k"]=1
        if not ek[ek["fd"]==yest].empty:   out["has_8k_yesterday"]=1
        if not ek[ek["fd"]==two].empty:    out["has_8k_2days_ago"]=1
        rec = ek[ek["fd"]>=two]
        if not rec.empty:
            try:
                ft = pd.to_datetime(rec.sort_values("filed").iloc[-1]["filed"])
                out["8k_filing_hour"]    = ft.hour
                out["hours_before_open"] = max(0, 9.5-(ft.hour+ft.minute/60))
            except Exception: pass
        dil_forms = {"S-1","S-1/A","S-3","S-3/A","424B1","424B3","424B4","424B5"}
        dil  = rows[rows["form_type"].isin(dil_forms)]
        dp   = dil[dil["fd"]<today]
        out["dilution_count_6m"]  = len(dp[dp["fd"]>=six_m])
        out["dilution_count_30d"] = len(dp[dp["fd"]>=thirty])
        out["has_dilution"]       = 1 if out["dilution_count_30d"]>0 else 0
        out["is_serial_diluter"]  = 1 if out["dilution_count_6m"]>=3 else 0
        if not dp.empty:
            out["days_since_dilution"] = (today - dp["fd"].max()).days
        f4 = rows[rows["form_type"].isin(["4","4/A"])]
        f4_recent = f4[(f4["fd"]>=thirty)&(f4["fd"]<today)]
        if not f4_recent.empty:
            out["has_form4_buy"]=1
        out["form4_buy_count"] = len(f4_recent)
        sc = rows[rows["form_type"].isin(["SC 13D","SC 13D/A"])]
        if not sc[(sc["fd"]>=six_m)&(sc["fd"]<today)].empty:
            out["has_sc13d"]=1
        # Text-based features — not available from EDGAR index, default 0
        out["has_merger"]        = 0
        out["has_fda"]           = 0
        out["has_contract"]      = 0
        out["has_reverse_split"] = 0
        out["has_buyback"]       = 0
    except Exception as e:
        log.warning(f"EDGAR {ticker}: {e}")
    return out


def calc_ah_features(ah_bars, prev_close):
    out = {"ah_move_pct":0,"ah_direction":0,"ah_volume":0,
           "ah_start_hour":0,"ah_sustained":0}
    if not ah_bars or prev_close<=0: return out
    out["ah_volume"]     = sum(b.get("v",0) for b in ah_bars)
    out["ah_start_hour"] = ah_bars[0].get("hour",16)
    ao = float(ah_bars[0].get("o",0) or 0)
    ac = float(ah_bars[-1].get("c",0) or 0)
    if ao>0:
        out["ah_move_pct"]  = (ac-ao)/ao
        out["ah_direction"] = 1 if ac>ao else (-1 if ac<ao else 0)
    if len(ah_bars)>=4:
        half = len(ah_bars)//2
        fc = float(ah_bars[half-1].get("c",0) or 0)
        lc = float(ah_bars[-1].get("c",0) or 0)
        fo = float(ah_bars[0].get("o",0) or 0)
        if fo>0 and fc>fo:
            out["ah_sustained"] = 1 if lc>=fc*0.90 else 0
    return out


def calc_pm_features(pm_bars, prev_close, avg_pm_vol=0):
    out = {
        "pm_open":0,"pm_high":0,"pm_low":0,"pm_close":0,"pm_volume":0,
        "pm_gap_pct":0,"pm_move_pct":0,"pm_vol_ratio":0,"pm_volume_build":0,
        "pm_high_of_session":0,"pm_fade":0,"pm_remaining_to_seed":0,
        "pm_remaining_to_super":0,"pm_consecutive_vol_bars":0,
        "pm_volume_consistency":0,"pm_gap_held":0,"pm_active_bars":0,
        "pm_vol_acceleration":0,"pm_gap_start_hour":0,
    }
    if not pm_bars or prev_close<=0: return out
    out["pm_open"]   = float(pm_bars[0].get("o",0) or 0)
    out["pm_high"]   = max(float(b.get("h",0) or 0) for b in pm_bars)
    out["pm_low"]    = min(float(b.get("l",0) or 0) for b in pm_bars if b.get("l",0)>0) if pm_bars else 0
    out["pm_close"]  = float(pm_bars[-1].get("c",0) or 0)
    out["pm_volume"] = sum(float(b.get("v",0) or 0) for b in pm_bars)
    if prev_close>0 and out["pm_open"]>0:
        out["pm_gap_pct"]  = (out["pm_open"]-prev_close)/prev_close
    if out["pm_open"]>0 and out["pm_high"]>0:
        out["pm_move_pct"] = (out["pm_high"]-out["pm_open"])/out["pm_open"]
    if avg_pm_vol>0:
        out["pm_vol_ratio"] = out["pm_volume"]/avg_pm_vol
    vols = [float(b.get("v",0) or 0) for b in pm_bars]
    if len(vols)>=4:
        h  = len(vols)//2
        ev = np.mean(vols[:h]); lv = np.mean(vols[h:])
        out["pm_volume_build"] = 1 if lv>ev*1.2 else 0
    if out["pm_close"] and out["pm_high"]:
        out["pm_high_of_session"] = 1 if out["pm_close"]>=out["pm_high"]*0.99 else 0
    if out["pm_high"] and out["pm_open"] and out["pm_high"]>out["pm_open"]:
        move = out["pm_high"]-out["pm_open"]
        fade = out["pm_high"]-out["pm_close"]
        out["pm_fade"] = 1 if fade>move*0.10 else 0
    if out["pm_close"]>0:
        out["pm_remaining_to_seed"]  = (prev_close*2.0-out["pm_close"])/out["pm_close"]
        out["pm_remaining_to_super"] = (prev_close*3.5-out["pm_close"])/out["pm_close"]
    if len(vols)>1:
        c=0
        for i in range(1,len(vols)):
            c = c+1 if vols[i]>=vols[i-1] else 0
        out["pm_consecutive_vol_bars"] = c
    if len(vols)>1 and np.mean(vols)>0:
        out["pm_volume_consistency"] = 1-min(1.0,np.std(vols)/np.mean(vols))
    out["pm_active_bars"] = int(sum(1 for v in vols if v>100))
    if out["pm_gap_pct"]>0.02:
        for b in pm_bars:
            bc = float(b.get("c",0) or 0)
            if prev_close>0 and bc>0 and (bc-prev_close)/prev_close>=0.02:
                out["pm_gap_start_hour"] = b.get("hour",0)
                break
    if prev_close>0 and out["pm_gap_pct"]>0:
        closes = [float(b.get("c",0) or 0) for b in pm_bars]
        gaps   = [(c-prev_close)/prev_close for c in closes if c>0]
        if gaps:
            mg = max(gaps)
            if mg>0:
                out["pm_gap_held"] = 1 if gaps[-1]>=mg*0.80 else 0
    if len(vols)>=3:
        ac = sum(1 for i in range(1,len(vols)) if vols[i]>vols[i-1])
        out["pm_vol_acceleration"] = ac/(len(vols)-1)
    return out


def get_hist_features(ticker, prev_close):
    out = {
        "price_52w_high":prev_close,"price_52w_low":prev_close,
        "pct_from_52w_high":0,"pct_from_52w_low":0,"near_52w_low":0,
        "avg_volume_20d":0,"vol_ratio_prev":0,
        "prev_3d_trend":0,"prev_5d_trend":0,"prev_10d_trend":0,
        "vol_trend_3d":0,"consecutive_vol_days":0,
        "days_since_last_spike":999,"coil_days":0,
        "prev_body_pct":0,"prev_wick_ratio":0,
        "prev_open":prev_close,"prev_high":prev_close,"prev_low":prev_close,
        "days_since_last_seed":999,
        "days_to_earnings":999,"has_earnings_soon":0,"had_earnings_recently":0,
        "halted_yesterday":0,"halt_count_5d":0,"halt_count_30d":0,
        "is_serial_halter":0,"days_since_last_halt":999,
        "spy_prev_day_pct":0,"qqq_prev_day_pct":0,
        "iwm_prev_day_pct":0,"xbi_prev_day_pct":0,
        "market_green":0,"market_red":0,"sector_hot":0,
        "hist_fetch_ok":0,"si_fetch_ok":0,
        "ah_fetch_ok":0,"float_fetch_ok":0,"earnings_fetch_ok":0,
        "edgar_fetch_ok":0,"halt_fetch_ok":0,"sector_fetch_ok":0,"pm_fetch_ok":0,
        "days_since_dilution":999,
    }
    try:
        now = datetime.now(ET)
        bars = poly.aggs(ticker, 1, "day",
                         (now-timedelta(days=380)).strftime("%Y-%m-%d"),
                         (now-timedelta(days=1)).strftime("%Y-%m-%d"))
        if not bars: return out
        bars.sort(key=lambda x: x.get("t",0))
        out["hist_fetch_ok"] = 1
        closes  = [float(b.get("c",0) or 0) for b in bars]
        volumes = [float(b.get("v",0) or 0) for b in bars]
        highs   = [float(b.get("h",0) or 0) for b in bars]
        lows    = [float(b.get("l",0) or 0) for b in bars]
        opens   = [float(b.get("o",0) or 0) for b in bars]
        if bars:
            out["prev_open"] = opens[-1]; out["prev_high"] = highs[-1]; out["prev_low"] = lows[-1]
            if opens[-1]>0: out["prev_body_pct"] = abs(closes[-1]-opens[-1])/opens[-1]
            rng = highs[-1]-lows[-1]; body = abs(closes[-1]-opens[-1])
            if rng>0: out["prev_wick_ratio"] = 1-(body/rng)
        p52 = bars[-252:] if len(bars)>=252 else bars
        h52 = max(float(b.get("h",0) or 0) for b in p52)
        l52 = min(float(b.get("l",0) or 0) for b in p52 if b.get("l",0)>0)
        out["price_52w_high"]=h52; out["price_52w_low"]=l52
        if h52>0: out["pct_from_52w_high"]=(prev_close-h52)/h52
        if l52>0:
            out["pct_from_52w_low"]=(prev_close-l52)/l52
            out["near_52w_low"]=1 if prev_close<l52*1.10 else 0
        if len(volumes)>=20:
            out["avg_volume_20d"]=float(np.mean(volumes[-20:]))
            if out["avg_volume_20d"]>0: out["vol_ratio_prev"]=volumes[-1]/out["avg_volume_20d"]
        for n,key in [(3,"prev_3d_trend"),(5,"prev_5d_trend"),(10,"prev_10d_trend")]:
            if len(closes)>=n and closes[-n]>0: out[key]=(closes[-1]-closes[-n])/closes[-n]
        if len(volumes)>=3 and volumes[-3]>0: out["vol_trend_3d"]=(volumes[-1]-volumes[-3])/volumes[-3]
        av = np.mean(volumes) if volumes else 0
        sd=0
        for v in reversed(volumes):
            if av>0 and v>av*3: break
            sd+=1
        out["days_since_last_spike"]=sd
        co=0
        for v in reversed(volumes):
            if av>0 and v<av*0.8: co+=1
            else: break
        out["coil_days"]=co
        cv=0
        for v in reversed(volumes):
            if av>0 and v>av: cv+=1
            else: break
        out["consecutive_vol_days"]=cv
        for sym,key in [("SPY","spy"),("QQQ","qqq"),("IWM","iwm"),("XBI","xbi")]:
            try:
                sb = poly.aggs(sym, 1, "day",
                               (now-timedelta(days=5)).strftime("%Y-%m-%d"),
                               (now-timedelta(days=1)).strftime("%Y-%m-%d"))
                if sb and len(sb)>=2:
                    c1=float(sb[-1].get("c",0) or 0); c2=float(sb[-2].get("c",0) or 0)
                    if c2>0:
                        pct=(c1-c2)/c2; out[f"{key}_prev_day_pct"]=pct
                        out["sector_fetch_ok"]=1
            except Exception: pass
        spy=out["spy_prev_day_pct"]; xbi=out["xbi_prev_day_pct"]
        out["market_green"]=1 if spy>0.003 else 0
        out["market_red"]  =1 if spy<-0.003 else 0
        out["sector_hot"]  =1 if xbi>0.01 else 0
    except Exception as e:
        log.warning(f"Hist {ticker}: {e}")
    return out


def build_feature_vector(cols, prev_close, prev_vol, float_M, mc, is_foreign,
                          si_pct, si_tier, ah_feats, edgar_feats, pm_feats, hist_feats):
    all_f = {
        "prev_close": prev_close, "prev_volume": prev_vol,
        "prev_dollar_vol": prev_close*prev_vol,
        "float_M": float_M, "float_shares": float_M*1_000_000 if float_M>0 else -1,
        "market_cap": mc, "is_foreign_listed": is_foreign, "is_estimated_float": 0,
        "float_tier": (0 if 0<float_M<5 else 1 if float_M<15 else 2 if float_M<50 else 3 if float_M<200 else 4) if float_M>0 else -1,
        "float_rotation_prev": (prev_close*prev_vol)/(float_M*1_000_000) if float_M>0 and prev_close>0 else 0,
        "si_pct": si_pct, "si_tier": si_tier,
        **ah_feats, **edgar_feats, **pm_feats, **hist_feats,
    }
    vec = np.array([float(all_f.get(c,-1) or -1) for c in cols]).reshape(1,-1)
    return np.nan_to_num(vec, nan=-1, posinf=-1, neginf=-1)


def score_midnight(ticker, prev_close, prev_vol):
    scores = {"midnight_seed": 0.0, "midnight_super": 0.0}
    if "midnight_seed_model" not in _models: return scores
    float_M, mc, is_foreign = get_float(ticker)
    if float_M<=0 or float_M>MAX_FLOAT_M: return scores
    si_pct, si_tier = get_si(ticker)
    ah_feats  = calc_ah_features(get_ah_bars(ticker), prev_close)
    edgar     = get_edgar_features(ticker)
    hist      = get_hist_features(ticker, prev_close)
    pm_feats  = {k:0 for k in [
        "pm_open","pm_high","pm_low","pm_close","pm_volume","pm_gap_pct",
        "pm_move_pct","pm_vol_ratio","pm_volume_build","pm_high_of_session",
        "pm_fade","pm_remaining_to_seed","pm_remaining_to_super",
        "pm_consecutive_vol_bars","pm_volume_consistency","pm_gap_held",
        "pm_active_bars","pm_vol_acceleration","pm_gap_start_hour","pm_fetch_ok",
    ]}
    for mk,ck in [("midnight_seed_model","midnight_seed"),("midnight_super_model","midnight_super")]:
        if mk not in _models: continue
        cols = _feature_cols.get(ck,[])
        if not cols: continue
        vec = build_feature_vector(cols, prev_close, prev_vol, float_M, mc, is_foreign,
                                   si_pct, si_tier, ah_feats, edgar, pm_feats, hist)
        try: scores[ck] = round(float(_models[mk].predict_proba(vec)[0][1]),4)
        except Exception as e: log.warning(f"Score {ticker} {mk}: {e}")
    return scores


def score_morning(ticker, prev_close, prev_vol, midnight_scores):
    scores = dict(midnight_scores)
    scores.update({"morning_seed":0.0,"morning_super":0.0})
    if "morning_seed_model" not in _models: return scores
    float_M, mc, is_foreign = get_float(ticker)
    if float_M<=0: return scores
    si_pct, si_tier = get_si(ticker)
    ah_feats = calc_ah_features(get_ah_bars(ticker), prev_close)
    edgar    = get_edgar_features(ticker)
    hist     = get_hist_features(ticker, prev_close)
    pm_feats = calc_pm_features(get_pm_bars(ticker), prev_close)
    for mk,ck in [("morning_seed_model","morning_seed"),("morning_super_model","morning_super")]:
        if mk not in _models: continue
        cols = _feature_cols.get(ck,[])
        if not cols: continue
        vec = build_feature_vector(cols, prev_close, prev_vol, float_M, mc, is_foreign,
                                   si_pct, si_tier, ah_feats, edgar, pm_feats, hist)
        try: scores[ck] = round(float(_models[mk].predict_proba(vec)[0][1]),4)
        except Exception as e: log.warning(f"Score {ticker} {mk}: {e}")
    return scores


def send_pushover(title, message, alert_type="seed", priority=0):
    try:
        r = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token":    PUSHOVER_APP_TOKEN,
                "user":     PUSHOVER_USER_KEY,
                "title":    title,
                "message":  message,
                "sound":    SOUNDS.get(alert_type,"pushover"),
                "priority": priority,
            },
            timeout=10,
        )
        if r.status_code==200: log.info(f"Pushover sent: {title}")
        else: log.warning(f"Pushover failed: {r.status_code}")
    except Exception as e:
        log.warning(f"Pushover error: {e}")


def already_alerted(ticker):
    if ticker in _alerted: return True
    f = ALERT_DIR / f"{date.today().isoformat()}_alerts.json"
    if f.exists():
        try:
            with open(f) as fp:
                if ticker in json.load(fp):
                    _alerted.add(ticker); return True
        except Exception: pass
    return False


def log_alert(ticker, alert_type, scores, price, gap):
    f = ALERT_DIR / f"{date.today().isoformat()}_alerts.json"
    alerts = {}
    if f.exists():
        try:
            with open(f) as fp: alerts=json.load(fp)
        except Exception: pass
    alerts[ticker] = {"type":alert_type,"time":datetime.now(ET).isoformat(),
                      "price":price,"gap":gap,"scores":scores}
    with open(f,"w") as fp: json.dump(alerts,fp,indent=2)
    _alerted.add(ticker)


def format_alert(ticker, alert_type, scores, price, gap, float_M, si_pct, has_8k, mode=""):
    emoji  = {"seed":"🌱","super":"🚀","midnight":"🌙"}.get(alert_type,"🌱")
    prev   = price/(1+gap) if (1+gap)!=0 else price
    room   = ((prev*2.0-price)/price*100) if price>0 and prev>0 else 0
    tag    = f" [{mode}]" if mode else ""
    title  = f"{emoji} {alert_type.upper()} — {ticker}{tag}"
    msg = (
        f"Price: ${price:.3f} ({gap*100:+.1f}% gap)\n"
        f"Room to 100%: {room:+.1f}%\n"
        f"Float: {float_M:.1f}M | SI: {si_pct:.1f}%\n"
        f"8K: {'YES' if has_8k else 'no'}\n"
        f"Seed: {scores.get('morning_seed',0):.2f} | "
        f"Super: {scores.get('morning_super',0):.2f}\n"
        f"Mid: {scores.get('midnight_seed',0):.2f}"
    )
    return title, msg


def run_midnight_scan():
    global _watchlist
    log.info("="*50)
    log.info("MIDNIGHT SCAN — building watchlist")
    tickers = poly.get_tickers()
    if not tickers:
        log.error("No tickers"); return
    candidates = {}
    processed = skipped = 0
    for ticker in tickers:
        try:
            pc, pv = get_prev_close(ticker)
            if pc<MIN_PRICE or pc>MAX_PRICE or pc*pv<MIN_DOLLAR_VOL:
                skipped+=1; continue
            scores = score_midnight(ticker, pc, pv)
            if scores.get("midnight_seed",0) >= MIDNIGHT_THRESHOLD:
                candidates[ticker] = {**scores,"prev_close":pc,"prev_vol":pv}
            processed+=1
            if processed%100==0:
                log.info(f"Midnight: {processed}/{len(tickers)} | candidates={len(candidates)}")
        except Exception as e:
            log.warning(f"Midnight {ticker}: {e}")
    _watchlist = dict(sorted(candidates.items(),
                             key=lambda x:x[1].get("midnight_seed",0),
                             reverse=True)[:MAX_WATCHLIST])
    log.info(f"Watchlist built: {len(_watchlist)} stocks")
    top5 = list(_watchlist.items())[:5]
    lines = [f"Watchlist: {len(_watchlist)} stocks\n"]
    for t,s in top5:
        lines.append(f"{t}: mid={s.get('midnight_seed',0):.2f} close=${s.get('prev_close',0):.2f}")
    send_pushover("🌙 Delta v2 Watchlist","\n".join(lines),"midnight",priority=0)
    gc.collect()


def run_morning_scan():
    if not _watchlist:
        log.warning("Watchlist empty"); return 0
    log.info(f"MORNING SCAN — {len(_watchlist)} stocks")
    fired = 0
    for ticker, wl in _watchlist.items():
        try:
            if already_alerted(ticker): continue
            pc = wl.get("prev_close",0); pv = wl.get("prev_vol",0)
            if pc<=0: continue
            scores = score_morning(ticker, pc, pv, wl)
            ms = scores.get("morning_seed",0)
            mu = scores.get("morning_super",0)
            if ms < MORNING_SEED_THRESHOLD: continue
            alert_type = "super" if mu>=MORNING_SUPER_THRESHOLD else "seed"
            pm = get_pm_bars(ticker)
            price = float(pm[-1].get("c",pc) or pc) if pm else pc
            gap   = (price-pc)/pc if pc>0 else 0
            float_M,_,_ = get_float(ticker)
            si_pct,_    = get_si(ticker)
            edgar       = get_edgar_features(ticker)
            has_8k      = edgar.get("has_8k",0)
            title,msg   = format_alert(ticker,alert_type,scores,price,gap,
                                       float_M,si_pct,has_8k,mode="morning")
            send_pushover(title,msg,alert_type,1 if alert_type=="super" else 0)
            log_alert(ticker,alert_type,scores,price,gap)
            fired+=1
            log.info(f"ALERT {alert_type.upper()} {ticker} seed={ms:.3f} super={mu:.3f} gap={gap*100:+.1f}%")
        except Exception as e:
            log.warning(f"Morning {ticker}: {e}")
    log.info(f"Morning scan: {fired} alerts")
    return fired


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        if path=="/health":
            self.wfile.write(json.dumps({
                "status":"ok","models":list(_models.keys()),
                "watchlist":len(_watchlist),"alerted":len(_alerted),
                "time_et":datetime.now(ET).isoformat(),
            }).encode())
        elif path=="/watchlist":
            out = {t:{"midnight_seed":round(v.get("midnight_seed",0),3),
                      "midnight_super":round(v.get("midnight_super",0),3),
                      "prev_close":v.get("prev_close",0)}
                   for t,v in sorted(_watchlist.items(),
                                     key=lambda x:x[1].get("midnight_seed",0),reverse=True)}
            self.wfile.write(json.dumps(out,indent=2).encode())
        elif path=="/alerts":
            f = ALERT_DIR/f"{date.today().isoformat()}_alerts.json"
            data = {}
            if f.exists():
                try:
                    with open(f) as fp: data=json.load(fp)
                except Exception: pass
            self.wfile.write(json.dumps(data,indent=2).encode())
        else:
            self.wfile.write(b"{}")
    def log_message(self,*a): pass


def start_server():
    HTTPServer(("0.0.0.0",int(os.environ.get("PORT",8080))),Handler).serve_forever()


def main():
    threading.Thread(target=start_server,daemon=True).start()
    time.sleep(2)
    log.info("="*60)
    log.info("THE DELTA v2 — Scanner (two-model architecture)")
    log.info("="*60)
    log.info("Downloading models from data service...")
    if not download_models():
        log.warning("Some models failed to download — will retry on next startup")
    load_models()
    load_si_lookup()
    if not _models:
        log.error("No models loaded — check DATA_SERVICE_URL and model files")
    midnight_done = morning_done = False
    last_date = None
    while True:
        now    = datetime.now(ET)
        today  = now.date()
        hour   = now.hour
        minute = now.minute
        if last_date != today:
            midnight_done = morning_done = False
            _alerted.clear()
            last_date = today
            log.info(f"New day: {today}")
        if hour==23 and minute<5 and not midnight_done:
            log.info("11PM — midnight scan")
            try: run_midnight_scan(); midnight_done=True
            except Exception as e: log.error(f"Midnight error: {e}")
        if hour==MORNING_SCORE_HOUR and minute>=MORNING_SCORE_MIN and not morning_done:
            log.info("5:30AM — morning scan")
            try: run_morning_scan(); morning_done=True
            except Exception as e: log.error(f"Morning error: {e}")
        in_pm = (hour>=4) and (hour<9 or (hour==9 and minute<30))
        if in_pm and morning_done:
            try: run_morning_scan()
            except Exception as e: log.warning(f"PM rescan: {e}")
            time.sleep(SCAN_INTERVAL)
            continue
        log.debug(f"Resting ({hour}:{minute:02d} ET)")
        time.sleep(60)

if __name__=="__main__":
    main()
