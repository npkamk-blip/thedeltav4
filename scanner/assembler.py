"""
THE DELTA v2 — assembler.py
============================
Builds two training datasets from raw collected data:

1. MIDNIGHT dataset — features known at 11PM night before
   Label: did this stock seed the NEXT trading day?
   Used to train: midnight_seed_model, midnight_super_model

2. MORNING dataset — midnight features + premarket features
   Label: did this stock seed TODAY?
   Used to train: morning_seed_model, morning_super_model

Inputs:
  /app/data/raw/tickers/       ← phase 1 raw data
  /app/data/raw/pm_1min/       ← phase 1B 1-min PM bars
  /app/data/raw/finra/         ← SI data
  /app/data/raw/edgar/         ← filing data
  /app/data/raw/halts/         ← halt history
  /app/data/raw/seed_registry.json    ← phase 1.5
  /app/data/raw/control_registry.json ← phase 1.5

Outputs:
  /app/data/training_data_v2/midnight/ ← one parquet per trading day
  /app/data/training_data_v2/morning/  ← one parquet per trading day
"""

import os, json, gc, logging, time, threading
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from http.server import HTTPServer, BaseHTTPRequestHandler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT      = Path(os.environ.get("DATA_DIR", "/app/data"))
RAW_DIR        = DATA_ROOT / "raw"
TICKER_RAW_DIR = RAW_DIR / "tickers"
PM_1MIN_DIR    = RAW_DIR / "pm_1min"
FINRA_DIR      = RAW_DIR / "finra"
HALTS_DIR      = RAW_DIR / "halts"
EDGAR_DIR      = RAW_DIR / "edgar"
OUTPUT_DIR     = DATA_ROOT / "training_data_v2"
MIDNIGHT_DIR   = OUTPUT_DIR / "midnight"
MORNING_DIR    = OUTPUT_DIR / "morning"
LOG_DIR        = DATA_ROOT / "logs"

START_DATE = date(2025, 1, 1)
END_DATE   = date(2026, 5, 23)

SEED_MULT  = 2.00   # 100% gain
SUPER_MULT = 3.50   # 250% gain

MIN_PRICE        = 0.10
MAX_PRICE        = 10.00
MIN_DOLLAR_VOL   = 10_000
MIN_LABEL_VOLUME = 25_000

ET = ZoneInfo("America/New_York")

KNOWN_HOLIDAYS = {
    date(2025,1,1), date(2025,1,9), date(2025,1,20), date(2025,2,17),
    date(2025,4,18), date(2025,5,26), date(2025,6,19), date(2025,7,4),
    date(2025,9,1), date(2025,11,27), date(2025,12,25),
    date(2026,1,1), date(2026,1,19), date(2026,2,16), date(2026,4,3),
    date(2026,5,25),
}

# ─────────────────────────────────────────────
# DIRS + LOGGING
# ─────────────────────────────────────────────
for d in [MIDNIGHT_DIR, MORNING_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "assembler.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("assembler")


# ─────────────────────────────────────────────
# KEEPALIVE
# ─────────────────────────────────────────────
class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b"alive")
    def log_message(self, *a): pass

def start_keepalive(port=8080):
    HTTPServer(("0.0.0.0", port), _Health).serve_forever()


# ─────────────────────────────────────────────
# TRADING CALENDAR
# ─────────────────────────────────────────────
def get_trading_days(start: date, end: date) -> list[date]:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5 and cur not in KNOWN_HOLIDAYS:
            days.append(cur)
        cur += timedelta(days=1)
    return days

def get_next_trading_day(d: date) -> date | None:
    nxt = d + timedelta(days=1)
    for _ in range(10):
        if nxt.weekday() < 5 and nxt not in KNOWN_HOLIDAYS:
            return nxt
        nxt += timedelta(days=1)
    return None


# ─────────────────────────────────────────────
# LOAD SUPPORT DATA
# ─────────────────────────────────────────────
def load_si_master() -> pd.DataFrame | None:
    path = FINRA_DIR / "si_master.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        log.info(f"SI master loaded: {len(df)} rows")
        return df
    log.warning("SI master not found — SI features will be -1")
    return None


def load_halts_master() -> pd.DataFrame | None:
    path = HALTS_DIR / "halts_master.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
        log.info(f"Halts master loaded: {len(df)} rows")
        return df
    log.warning("Halts master not found — halt features will be 0")
    return None


def load_edgar_master() -> tuple[pd.DataFrame | None, dict]:
    master_path = EDGAR_DIR / "filings_master.parquet"
    cik_path    = EDGAR_DIR / "cik_map.json"
    master = None
    cik_map = {}
    if master_path.exists():
        master = pd.read_parquet(master_path)
        master["filed_date"] = pd.to_datetime(master["filed"], errors="coerce").dt.date
        log.info(f"EDGAR master loaded: {len(master)} filings")
    else:
        log.warning("EDGAR master not found — EDGAR features will be 0")
    if cik_path.exists():
        with open(cik_path) as f:
            cik_map = json.load(f)
        log.info(f"CIK map loaded: {len(cik_map)} tickers")
    return master, cik_map


# ─────────────────────────────────────────────
# LOAD TICKER CACHE
# ─────────────────────────────────────────────
def load_ticker_cache(ticker: str) -> dict | None:
    path = TICKER_RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        df  = pd.read_parquet(path)
        row = df.iloc[0].to_dict()
        del df

        for key in ["daily", "ah_minute"]:
            val = row.get(key)
            if val is None:
                row[key] = pd.DataFrame()
            elif isinstance(val, pd.DataFrame):
                pass
            else:
                try:
                    lst = val.tolist() if hasattr(val, "tolist") else list(val)
                    row[key] = pd.DataFrame(lst) if lst else pd.DataFrame()
                except Exception:
                    row[key] = pd.DataFrame()

            df2 = row[key]
            if not isinstance(df2, pd.DataFrame):
                row[key] = pd.DataFrame()
                continue
            if not df2.empty and "date" not in df2.columns and "t" in df2.columns:
                df2["date"] = pd.to_datetime(
                    df2["t"], unit="ms", utc=True
                ).dt.tz_convert(ET).dt.date
                row[key] = df2
            elif not df2.empty and "date" in df2.columns:
                df2["date"] = pd.to_datetime(df2["date"]).dt.date
                row[key] = df2

        if isinstance(row.get("float_data"), str):
            row["float_data"] = json.loads(row["float_data"])
        elif hasattr(row.get("float_data"), "tolist"):
            row["float_data"] = {}

        if isinstance(row.get("fetch_ok"), str):
            row["fetch_ok"] = json.loads(row["fetch_ok"])
        elif hasattr(row.get("fetch_ok"), "tolist"):
            row["fetch_ok"] = {}

        if hasattr(row.get("earnings_dates"), "tolist"):
            row["earnings_dates"] = row["earnings_dates"].tolist()

        return row
    except Exception as e:
        log.warning(f"WARN load_cache | {ticker} | {e}")
        return None


def load_pm_1min(ticker: str) -> pd.DataFrame:
    path = PM_1MIN_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        if not df.empty and "date" not in df.columns and "t" in df.columns:
            df["t"]    = pd.to_datetime(df["t"], utc=True).dt.tz_convert(ET)
            df["date"] = df["t"].dt.date
            df["hour"] = df["t"].dt.hour
            df["minute"] = df["t"].dt.minute
        elif not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    except Exception as e:
        log.warning(f"WARN pm_1min | {ticker} | {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# FEATURE CALCULATORS
# ─────────────────────────────────────────────
def calc_historical_features(daily_df: pd.DataFrame, today_idx: int) -> dict:
    out = {
        "price_52w_high": None, "price_52w_low": None,
        "pct_from_52w_high": None, "pct_from_52w_low": None,
        "near_52w_low": 0,
        "avg_volume_20d": None, "vol_ratio_prev": None,
        "prev_3d_trend": None, "prev_5d_trend": None, "prev_10d_trend": None,
        "days_since_last_spike": 999, "coil_days": 0,
        "vol_trend_3d": None, "consecutive_vol_days": 0,
        "hist_fetch_ok": 0,
    }
    if today_idx < 2:
        return out

    past = daily_df.iloc[:today_idx]
    if len(past) < 5:
        return out

    out["hist_fetch_ok"] = 1
    past_52    = past.tail(252)
    prev       = past.iloc[-1]
    prev_close = float(prev.get("close", 0) or 0)

    out["price_52w_high"] = float(past_52["high"].max()) if "high" in past_52.columns else None
    out["price_52w_low"]  = float(past_52["low"].min())  if "low"  in past_52.columns else None

    if out["price_52w_high"] and out["price_52w_high"] > 0:
        out["pct_from_52w_high"] = (prev_close - out["price_52w_high"]) / out["price_52w_high"]
    if out["price_52w_low"] and out["price_52w_low"] > 0:
        out["pct_from_52w_low"]  = (prev_close - out["price_52w_low"])  / out["price_52w_low"]
        out["near_52w_low"]      = 1 if prev_close < out["price_52w_low"] * 1.10 else 0

    if len(past) >= 20 and "volume" in past.columns:
        out["avg_volume_20d"] = float(past.tail(20)["volume"].mean())
        prev_vol = float(prev.get("volume", 0) or 0)
        if out["avg_volume_20d"] > 0:
            out["vol_ratio_prev"] = prev_vol / out["avg_volume_20d"]

    closes  = past["close"].values  if "close"  in past.columns else []
    volumes = past["volume"].values if "volume" in past.columns else []

    for n, key in [(3,"prev_3d_trend"),(5,"prev_5d_trend"),(10,"prev_10d_trend")]:
        if len(closes) >= n and closes[-n] > 0:
            out[key] = (closes[-1] - closes[-n]) / closes[-n]

    # Vol trend 3d
    if len(volumes) >= 3 and volumes[-3] > 0:
        out["vol_trend_3d"] = (volumes[-1] - volumes[-3]) / volumes[-3]

    # Days since last 3x spike
    avg_vol = float(past["volume"].mean()) if len(past) > 0 and "volume" in past.columns else 0
    spike_days = 0
    for v in reversed(volumes):
        if avg_vol > 0 and v > avg_vol * 3:
            break
        spike_days += 1
    out["days_since_last_spike"] = spike_days

    # Coil days
    avg20 = out.get("avg_volume_20d") or 0
    coil = 0
    for v in reversed(volumes):
        if avg20 > 0 and v < avg20 * 0.8:
            coil += 1
        else:
            break
    out["coil_days"] = coil

    # Consecutive above-avg vol days
    consec = 0
    for v in reversed(volumes):
        if avg20 > 0 and v > avg20:
            consec += 1
        else:
            break
    out["consecutive_vol_days"] = consec

    return out


def calc_ah_features(ah_df: pd.DataFrame, trade_date: date, prev_close: float) -> dict:
    """AH features from T-1 evening (night before trade_date)."""
    out = {
        "ah_move_pct": 0, "ah_direction": 0, "ah_volume": 0,
        "ah_start_hour": 0, "ah_sustained": 0, "ah_fetch_ok": 0,
    }
    if ah_df.empty or prev_close <= 0:
        return out

    # Get T-1 date
    prev_date = trade_date - timedelta(days=1)
    while prev_date.weekday() >= 5 or prev_date in KNOWN_HOLIDAYS:
        prev_date -= timedelta(days=1)

    ah_day = ah_df[ah_df["date"] == prev_date] if "date" in ah_df.columns else pd.DataFrame()
    if ah_day.empty:
        return out

    ah_day = ah_day.sort_values("t") if "t" in ah_day.columns else ah_day
    out["ah_fetch_ok"] = 1
    out["ah_volume"]   = float(ah_day["v"].sum()) if "v" in ah_day.columns else 0

    if "o" in ah_day.columns and "c" in ah_day.columns:
        ah_o = float(ah_day.iloc[0]["o"])
        ah_c = float(ah_day.iloc[-1]["c"])
        if ah_o > 0:
            out["ah_move_pct"]  = (ah_c - ah_o) / ah_o
            out["ah_direction"] = 1 if ah_c > ah_o else (-1 if ah_c < ah_o else 0)

    # What hour did AH activity start?
    if "hour" in ah_day.columns:
        out["ah_start_hour"] = int(ah_day["hour"].min())

    # Was move sustained? Compare first half vs second half
    if len(ah_day) >= 4:
        half = len(ah_day) // 2
        first_close  = float(ah_day.iloc[half-1]["c"]) if "c" in ah_day.columns else 0
        second_close = float(ah_day.iloc[-1]["c"]) if "c" in ah_day.columns else 0
        first_open   = float(ah_day.iloc[0]["o"]) if "o" in ah_day.columns else 0
        if first_open > 0 and first_close > first_open:
            out["ah_sustained"] = 1 if second_close >= first_close * 0.90 else 0

    return out


def calc_float_features(float_data: dict, prev_close: float, avg_vol_20d: float) -> dict:
    out = {
        "float_shares": -1, "float_M": -1, "market_cap": -1,
        "float_tier": -1, "float_rotation_prev": 0,
        "is_foreign_listed": 0, "float_fetch_ok": 0,
        "is_estimated_float": 0,
    }
    if not isinstance(float_data, dict):
        return out

    shares = float_data.get("shares_outstanding", 0) or 0
    mc     = float_data.get("market_cap", 0) or 0
    is_foreign = float_data.get("is_foreign_listed", 0)

    if shares and float(shares) > 0:
        out["float_fetch_ok"]    = 1
        out["float_shares"]      = float(shares)
        out["float_M"]           = float(shares) / 1_000_000
        out["is_foreign_listed"] = is_foreign
    elif mc and float(mc) > 0 and prev_close > 0:
        # Estimate float from market cap
        est_shares = float(mc) / prev_close
        out["float_shares"]      = est_shares
        out["float_M"]           = est_shares / 1_000_000
        out["float_fetch_ok"]    = 1
        out["is_estimated_float"] = 1
        out["is_foreign_listed"] = is_foreign
        log.debug(f"Estimated float from MC: {out['float_M']:.1f}M")

    if mc:
        out["market_cap"] = float(mc)

    fm = out["float_M"]
    if fm > 0:
        if fm < 5:     out["float_tier"] = 0   # nano
        elif fm < 15:  out["float_tier"] = 1   # micro
        elif fm < 50:  out["float_tier"] = 2   # small
        elif fm < 200: out["float_tier"] = 3   # mid
        else:          out["float_tier"] = 4   # large

    if out["float_shares"] > 0 and avg_vol_20d > 0 and prev_close > 0:
        out["float_rotation_prev"] = (avg_vol_20d * prev_close) / out["float_shares"]

    return out


def calc_si_features(ticker: str, trade_date: date, si_master: pd.DataFrame | None) -> dict:
    out = {"si_pct": -1, "si_tier": -1, "si_fetch_ok": 0}
    if si_master is None or si_master.empty:
        return out

    t_col  = next((c for c in si_master.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
    d_col  = next((c for c in si_master.columns if "date" in c.lower()), None)
    si_col = next((c for c in si_master.columns if "short" in c.lower() and "vol" not in c.lower()), None)

    if not all([t_col, d_col, si_col]):
        return out

    rows = si_master[si_master[t_col] == ticker].copy()
    if rows.empty:
        return out

    rows["_date"] = pd.to_datetime(rows[d_col], errors="coerce").dt.date
    rows = rows.dropna(subset=["_date"])
    rows["_diff"] = rows["_date"].apply(lambda d: abs((d - trade_date).days))
    nearest = rows.nsmallest(1, "_diff").iloc[0]

    try:
        sp = float(nearest[si_col])
        out["si_pct"]      = sp
        out["si_fetch_ok"] = 1
        if sp < 5:     out["si_tier"] = 0
        elif sp < 15:  out["si_tier"] = 1
        elif sp < 30:  out["si_tier"] = 2
        else:          out["si_tier"] = 3
    except Exception:
        pass
    return out


def calc_edgar_features(ticker: str, trade_date: date,
                         cik_map: dict, edgar_master: pd.DataFrame | None) -> dict:
    out = {
        "has_8k": 0, "has_8k_yesterday": 0, "has_8k_2days_ago": 0,
        "8k_filing_hour": 0, "hours_before_open": 0,
        "has_merger": 0, "has_fda": 0, "has_contract": 0,
        "has_dilution": 0, "has_reverse_split": 0, "has_buyback": 0,
        "dilution_count_6m": 0, "dilution_count_30d": 0,
        "days_since_dilution": 999, "is_serial_diluter": 0,
        "has_form4_buy": 0, "form4_buy_count": 0,
        "has_sc13d": 0, "edgar_fetch_ok": 0,
    }
    if edgar_master is None or edgar_master.empty:
        return out

    cik = cik_map.get(ticker)
    if not cik:
        return out

    rows = edgar_master[edgar_master["cik"] == cik].copy()
    if rows.empty:
        return out

    out["edgar_fetch_ok"] = 1
    yesterday  = trade_date - timedelta(days=1)
    two_days   = trade_date - timedelta(days=2)
    six_months = trade_date - timedelta(days=180)
    thirty     = trade_date - timedelta(days=30)

    eightk = rows[rows["form_type"].isin(["8-K","8-K/A"])]
    if not eightk[eightk["filed_date"] == trade_date].empty:    out["has_8k"] = 1
    if not eightk[eightk["filed_date"] == yesterday].empty:     out["has_8k_yesterday"] = 1
    if not eightk[eightk["filed_date"] == two_days].empty:      out["has_8k_2days_ago"] = 1

    # Filing hour from most recent 8-K
    recent = eightk[eightk["filed_date"] >= two_days]
    if not recent.empty:
        try:
            filed_ts = recent.sort_values("filed").iloc[-1]["filed"]
            filed_dt = pd.to_datetime(filed_ts)
            out["8k_filing_hour"]    = filed_dt.hour
            out["hours_before_open"] = max(0, 9.5 - (filed_dt.hour + filed_dt.minute / 60))
        except Exception:
            pass
        text = " ".join(recent["company"].astype(str).tolist()).lower()
        out["has_merger"]        = 1 if any(w in text for w in ["merger","acqui"]) else 0
        out["has_fda"]           = 1 if "fda" in text else 0
        out["has_contract"]      = 1 if "contract" in text else 0
        out["has_reverse_split"] = 1 if "reverse" in text else 0
        out["has_buyback"]       = 1 if "repurchas" in text else 0

    # Dilution
    dil_forms = {"S-1","S-1/A","S-3","S-3/A","424B1","424B3","424B4","424B5"}
    dil = rows[rows["form_type"].isin(dil_forms)]
    dil_past = dil[dil["filed_date"] < trade_date]
    out["dilution_count_6m"]  = len(dil_past[dil_past["filed_date"] >= six_months])
    out["dilution_count_30d"] = len(dil_past[dil_past["filed_date"] >= thirty])
    out["has_dilution"]       = 1 if out["dilution_count_30d"] > 0 else 0
    out["is_serial_diluter"]  = 1 if out["dilution_count_6m"] >= 3 else 0
    if not dil_past.empty:
        out["days_since_dilution"] = (trade_date - dil_past["filed_date"].max()).days

    # Form 4
    f4 = rows[rows["form_type"].isin(["4","4/A"])]
    f4r = f4[(f4["filed_date"] >= thirty) & (f4["filed_date"] < trade_date)]
    if not f4r.empty:
        out["has_form4_buy"]   = 1
        out["form4_buy_count"] = len(f4r)

    # SC 13D
    sc = rows[rows["form_type"].isin(["SC 13D","SC 13D/A"])]
    scr = sc[(sc["filed_date"] >= six_months) & (sc["filed_date"] < trade_date)]
    if not scr.empty:
        out["has_sc13d"] = 1

    return out


def calc_halt_features(ticker: str, trade_date: date, halts: pd.DataFrame | None) -> dict:
    out = {
        "halted_yesterday": 0, "halt_count_5d": 0,
        "halt_count_30d": 0, "is_serial_halter": 0,
        "days_since_last_halt": 999, "halt_fetch_ok": 0,
    }
    if halts is None or halts.empty:
        return out

    t_col = next((c for c in halts.columns if any(x in c for x in ["symbol","issue","ticker"])), None)
    d_col = next((c for c in halts.columns if "date" in c or "halt" in c), None)
    if not t_col or not d_col:
        return out

    rows = halts[halts[t_col] == ticker].copy()
    if rows.empty:
        return out

    out["halt_fetch_ok"] = 1
    rows["_date"] = pd.to_datetime(rows[d_col], errors="coerce").dt.date
    rows = rows.dropna(subset=["_date"])
    rows = rows[rows["_date"] < trade_date]
    if rows.empty:
        return out

    yesterday  = trade_date - timedelta(days=1)
    five_ago   = trade_date - timedelta(days=5)
    thirty_ago = trade_date - timedelta(days=30)

    out["halted_yesterday"]    = 1 if not rows[rows["_date"] == yesterday].empty else 0
    out["halt_count_5d"]       = len(rows[rows["_date"] >= five_ago])
    out["halt_count_30d"]      = len(rows[rows["_date"] >= thirty_ago])
    out["is_serial_halter"]    = 1 if out["halt_count_30d"] >= 3 else 0
    out["days_since_last_halt"] = (trade_date - rows["_date"].max()).days
    return out


def calc_earnings_features(earnings_dates: list, trade_date: date) -> dict:
    out = {
        "days_to_earnings": 999, "has_earnings_soon": 0,
        "had_earnings_recently": 0, "earnings_fetch_ok": 0,
    }
    if hasattr(earnings_dates, "tolist"):
        earnings_dates = earnings_dates.tolist()
    if not earnings_dates:
        return out

    out["earnings_fetch_ok"] = 1
    parsed = []
    for d in earnings_dates:
        try:
            parsed.append(date.fromisoformat(str(d)[:10]))
        except Exception:
            pass

    future = [d for d in parsed if d >= trade_date]
    past   = [d for d in parsed if d < trade_date]

    if future:
        nxt = min(future)
        out["days_to_earnings"]  = (nxt - trade_date).days
        out["has_earnings_soon"] = 1 if out["days_to_earnings"] <= 7 else 0
    if past:
        last = max(past)
        out["had_earnings_recently"] = 1 if (trade_date - last).days <= 5 else 0

    return out


def calc_pm_features_1min(pm_df: pd.DataFrame, trade_date: date,
                            prev_close: float, avg_pm_vol: float) -> dict:
    """Calculate PM features from 1-minute bars for the morning model."""
    out = {
        "pm_open": None, "pm_high": None, "pm_low": None,
        "pm_close": None, "pm_volume": None,
        "pm_gap_pct": None, "pm_move_pct": None,
        "pm_vol_ratio": None, "pm_volume_build": None,
        "pm_high_of_session": None, "pm_fade": None,
        "pm_remaining_to_seed": None, "pm_remaining_to_super": None,
        "pm_consecutive_vol_bars": 0,
        "pm_volume_consistency": 0,
        "pm_gap_held": 0,
        "pm_active_bars": 0,
        "pm_vol_acceleration": 0,
        "pm_gap_start_hour": 0,
        "pm_fetch_ok": 0,
    }
    if pm_df.empty or prev_close <= 0:
        return out

    if "date" in pm_df.columns:
        pm_day = pm_df[pm_df["date"] == trade_date].copy()
    else:
        return out

    if pm_day.empty:
        return out

    # Sort by time
    if "t" in pm_day.columns:
        pm_day = pm_day.sort_values("t")
    elif "hour" in pm_day.columns and "minute" in pm_day.columns:
        pm_day = pm_day.sort_values(["hour","minute"])

    out["pm_fetch_ok"] = 1

    col_o = "o" if "o" in pm_day.columns else "open"
    col_h = "h" if "h" in pm_day.columns else "high"
    col_l = "l" if "l" in pm_day.columns else "low"
    col_c = "c" if "c" in pm_day.columns else "close"
    col_v = "v" if "v" in pm_day.columns else "volume"

    out["pm_open"]   = float(pm_day.iloc[0][col_o])
    out["pm_high"]   = float(pm_day[col_h].max())
    out["pm_low"]    = float(pm_day[col_l].min())
    out["pm_close"]  = float(pm_day.iloc[-1][col_c])
    out["pm_volume"] = float(pm_day[col_v].sum())

    if prev_close > 0:
        out["pm_gap_pct"] = (out["pm_open"] - prev_close) / prev_close

    if out["pm_open"] and out["pm_open"] > 0:
        out["pm_move_pct"] = (out["pm_high"] - out["pm_open"]) / out["pm_open"]

    if avg_pm_vol and avg_pm_vol > 0:
        out["pm_vol_ratio"] = out["pm_volume"] / avg_pm_vol

    # Volume build — second half bigger than first
    if len(pm_day) >= 4:
        half = len(pm_day) // 2
        ev = pm_day.iloc[:half][col_v].mean()
        lv = pm_day.iloc[half:][col_v].mean()
        out["pm_volume_build"] = 1 if lv > ev * 1.2 else 0

    # High of session
    if out["pm_close"] and out["pm_high"]:
        out["pm_high_of_session"] = 1 if out["pm_close"] >= out["pm_high"] * 0.99 else 0

    # Fade
    if out["pm_high"] and out["pm_open"] and out["pm_high"] > out["pm_open"]:
        move = out["pm_high"] - out["pm_open"]
        fade = out["pm_high"] - out["pm_close"]
        out["pm_fade"] = 1 if fade > move * 0.10 else 0

    # Remaining upside
    if out["pm_close"] and out["pm_close"] > 0:
        p = out["pm_close"]
        out["pm_remaining_to_seed"]  = (prev_close * SEED_MULT  - p) / p
        out["pm_remaining_to_super"] = (prev_close * SUPER_MULT - p) / p

    # Consecutive bars with rising volume (fake move detector)
    volumes = pm_day[col_v].values
    consec = 0
    for i in range(1, len(volumes)):
        if volumes[i] >= volumes[i-1]:
            consec += 1
        else:
            consec = 0
    out["pm_consecutive_vol_bars"] = consec

    # Volume consistency (low std dev = real move)
    if len(volumes) > 1 and volumes.mean() > 0:
        out["pm_volume_consistency"] = 1 - min(1.0, volumes.std() / volumes.mean())

    # Active bars (bars with real volume > 100 shares)
    out["pm_active_bars"] = int((volumes > 100).sum())

    # Gap held — did gap stay above 80% of max gap?
    if out["pm_gap_pct"] and out["pm_gap_pct"] > 0:
        closes = pm_day[col_c].values
        if prev_close > 0:
            gaps = (closes - prev_close) / prev_close
            max_gap = gaps.max()
            if max_gap > 0:
                out["pm_gap_held"] = 1 if gaps[-1] >= max_gap * 0.80 else 0

    # Gap start hour
    if out["pm_gap_pct"] and out["pm_gap_pct"] >= 0.02 and "hour" in pm_day.columns:
        # Find first bar where gap exceeded 2%
        for _, bar in pm_day.iterrows():
            bar_gap = (float(bar[col_c]) - prev_close) / prev_close
            if bar_gap >= 0.02:
                out["pm_gap_start_hour"] = int(bar.get("hour", 0))
                break

    # Volume acceleration — is each bar bigger than previous?
    if len(volumes) >= 3:
        accel = sum(1 for i in range(1, len(volumes)) if volumes[i] > volumes[i-1])
        out["pm_vol_acceleration"] = accel / (len(volumes) - 1)

    return out


def calc_sector_features(sector_data: dict, trade_date: date) -> dict:
    out = {
        "spy_prev_day_pct": 0, "qqq_prev_day_pct": 0,
        "iwm_prev_day_pct": 0, "xbi_prev_day_pct": 0,
        "market_green": 0, "market_red": 0,
        "sector_hot": 0, "sector_fetch_ok": 0,
    }
    for sym, key in [("SPY","spy"),("QQQ","qqq"),("IWM","iwm"),("XBI","xbi")]:
        if sym not in sector_data:
            continue
        df = sector_data[sym]
        if df.empty:
            continue
        prev_rows = df[df["date"] < trade_date].tail(2)
        if len(prev_rows) < 2:
            continue
        t2, t1 = prev_rows.iloc[-2], prev_rows.iloc[-1]
        c2 = float(t2.get("close", 0) or 0)
        c1 = float(t1.get("close", 0) or 0)
        if c2 > 0:
            pct = (c1 - c2) / c2
            out[f"{key}_prev_day_pct"] = pct
            out["sector_fetch_ok"] = 1

    spy = out["spy_prev_day_pct"]
    xbi = out["xbi_prev_day_pct"]
    out["market_green"] = 1 if spy > 0.003 else 0
    out["market_red"]   = 1 if spy < -0.003 else 0
    out["sector_hot"]   = 1 if xbi > 0.01 else 0
    return out


# ─────────────────────────────────────────────
# LOAD SECTOR DATA
# ─────────────────────────────────────────────
def load_sector_data() -> dict:
    sector = {}
    for sym in ["SPY","QQQ","IWM","XBI"]:
        cache = load_ticker_cache(sym)
        if not cache:
            log.warning(f"WARN sector | {sym} not cached")
            continue
        daily = cache.get("daily", pd.DataFrame())
        if not isinstance(daily, pd.DataFrame):
            try:
                daily = pd.DataFrame(list(daily)) if daily else pd.DataFrame()
            except Exception:
                daily = pd.DataFrame()
        if not daily.empty:
            if "date" not in daily.columns and "t" in daily.columns:
                daily["date"] = pd.to_datetime(
                    daily["t"], unit="ms", utc=True
                ).dt.tz_convert(ET).dt.date
            elif "date" in daily.columns:
                daily["date"] = pd.to_datetime(daily["date"]).dt.date
            col_map = {"o":"open","h":"high","l":"low","c":"close","v":"volume"}
            daily = daily.rename(columns={k:v for k,v in col_map.items() if k in daily.columns})
        sector[sym] = daily
        log.info(f"Sector {sym}: {len(daily)} daily bars")
    return sector


# ─────────────────────────────────────────────
# ASSEMBLE ONE TICKER ONE DAY
# ─────────────────────────────────────────────
def assemble_ticker_day(
    ticker: str,
    trade_date: date,
    label_seed: int,
    label_super: int,
    daily_df: pd.DataFrame,
    ah_df: pd.DataFrame,
    pm_1min_df: pd.DataFrame,
    float_data: dict,
    earnings_dates: list,
    si_master: pd.DataFrame | None,
    halts: pd.DataFrame | None,
    edgar_master: pd.DataFrame | None,
    cik_map: dict,
    sector_data: dict,
    days_since_last_seed: int,
    avg_pm_vol: float,
) -> tuple[dict | None, dict | None]:
    """
    Build midnight and morning feature rows for one ticker on one day.
    Returns (midnight_row, morning_row) or (None, None) if invalid.
    """
    # Get T-1 bar
    past_rows = daily_df[daily_df["date"] < trade_date].sort_values("date")
    if past_rows.empty:
        return None, None

    prev_row   = past_rows.iloc[-1]
    prev_close = float(prev_row.get("close", 0) or 0)
    prev_vol   = float(prev_row.get("volume", 0) or 0)
    prev_open  = float(prev_row.get("open", 0) or 0)
    prev_high  = float(prev_row.get("high", 0) or 0)
    prev_low   = float(prev_row.get("low", 0) or 0)
    dollar_vol = prev_close * prev_vol

    # Drop checks
    if prev_close < MIN_PRICE or prev_close > MAX_PRICE:
        return None, None
    if dollar_vol < MIN_DOLLAR_VOL:
        return None, None

    # T-1 price structure
    prev_body_pct   = abs(prev_close - prev_open) / prev_open if prev_open > 0 else 0
    total_range     = prev_high - prev_low
    body            = abs(prev_close - prev_open)
    prev_wick_ratio = 1 - (body / total_range) if total_range > 0 else 0
    prev_dollar_vol = prev_close * prev_vol

    # Historical features
    today_idx  = len(past_rows)
    hist_feats = calc_historical_features(
        daily_df.sort_values("date").reset_index(drop=True), today_idx
    )

    # Float features
    avg_vol_20d = hist_feats.get("avg_volume_20d") or 0
    float_feats = calc_float_features(float_data, prev_close, avg_vol_20d)

    # Drop if no float at all
    if float_feats["float_M"] <= 0:
        return None, None

    # AH features
    ah_feats = calc_ah_features(ah_df, trade_date, prev_close)

    # SI features
    si_feats = calc_si_features(ticker, trade_date, si_master)

    # EDGAR features
    edgar_feats = calc_edgar_features(ticker, trade_date, cik_map, edgar_master)

    # Halt features
    halt_feats = calc_halt_features(ticker, trade_date, halts)

    # Earnings features
    earn_feats = calc_earnings_features(earnings_dates, trade_date)

    # Sector features
    sector_feats = calc_sector_features(sector_data, trade_date)

    # Base row shared by both models
    base = {
        "ticker":   ticker,
        "date":     trade_date.isoformat(),
        "label_seed":  label_seed,
        "label_super": label_super,
        # T-1 price structure
        "prev_close":       prev_close,
        "prev_open":        prev_open,
        "prev_high":        prev_high,
        "prev_low":         prev_low,
        "prev_volume":      prev_vol,
        "prev_dollar_vol":  prev_dollar_vol,
        "prev_body_pct":    prev_body_pct,
        "prev_wick_ratio":  prev_wick_ratio,
        "days_since_last_seed": days_since_last_seed,
        **hist_feats,
        **float_feats,
        **ah_feats,
        **si_feats,
        **edgar_feats,
        **halt_feats,
        **earn_feats,
        **sector_feats,
    }

    midnight_row = dict(base)

    # Morning row adds PM features
    pm_feats = calc_pm_features_1min(pm_1min_df, trade_date, prev_close, avg_pm_vol)
    morning_row = {**base, **pm_feats}

    return midnight_row, morning_row


# ─────────────────────────────────────────────
# MAIN ASSEMBLY LOOP
# ─────────────────────────────────────────────
def assemble_all(
    seed_registry: dict,
    control_registry: dict,
    si_master: pd.DataFrame | None,
    halts: pd.DataFrame | None,
    edgar_master: pd.DataFrame | None,
    cik_map: dict,
    sector_data: dict,
    trading_days: list[date],
):
    log.info("Starting assembly...")
    log.info(f"Seed days: {len(seed_registry)}")

    # Track which tickers seeded on which days for days_since_last_seed
    seed_history: dict[str, list[date]] = {}

    total_midnight = 0
    total_morning  = 0
    days_done      = 0

    for trade_date in trading_days:
        date_str = trade_date.isoformat()

        # Check if already assembled
        mid_path = MIDNIGHT_DIR / f"{date_str}.parquet"
        mor_path = MORNING_DIR  / f"{date_str}.parquet"
        if mid_path.exists() and mor_path.exists():
            days_done += 1
            continue

        # Get seeds and controls for this day
        seed_tickers    = seed_registry.get(date_str, [])
        day_controls    = control_registry.get(date_str, {})

        # Build full set of tickers to process today
        tickers_today = set(seed_tickers)
        for controls in day_controls.values():
            tickers_today.update(controls)

        if not tickers_today:
            days_done += 1
            continue

        # Determine labels for each ticker
        # Midnight model label: did it seed on the NEXT trading day?
        next_date     = get_next_trading_day(trade_date)
        next_date_str = next_date.isoformat() if next_date else None
        next_seeds    = set(seed_registry.get(next_date_str, [])) if next_date_str else set()

        midnight_rows = []
        morning_rows  = []

        for ticker in tickers_today:
            # Load ticker data
            cache = load_ticker_cache(ticker)
            if not cache:
                continue

            daily_raw = cache.get("daily")
            try:
                if isinstance(daily_raw, pd.DataFrame):
                    daily_df = daily_raw
                else:
                    lst = daily_raw.tolist() if hasattr(daily_raw, "tolist") else list(daily_raw)
                    daily_df = pd.DataFrame(lst) if lst else pd.DataFrame()
            except Exception:
                continue

            if daily_df.empty:
                continue

            if "date" not in daily_df.columns:
                if "t" in daily_df.columns:
                    daily_df["date"] = pd.to_datetime(
                        daily_df["t"], unit="ms", utc=True
                    ).dt.tz_convert(ET).dt.date
                else:
                    continue

            daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
            col_map = {"o":"open","h":"high","l":"low","c":"close","v":"volume"}
            daily_df = daily_df.rename(columns={k:v for k,v in col_map.items() if k in daily_df.columns})

            ah_df = cache.get("ah_minute", pd.DataFrame())
            if not isinstance(ah_df, pd.DataFrame):
                try:
                    lst = ah_df.tolist() if hasattr(ah_df, "tolist") else list(ah_df)
                    ah_df = pd.DataFrame(lst) if lst else pd.DataFrame()
                except Exception:
                    ah_df = pd.DataFrame()

            if not ah_df.empty and "date" not in ah_df.columns and "t" in ah_df.columns:
                ah_df["t"]    = pd.to_datetime(ah_df["t"], unit="ms", utc=True).dt.tz_convert(ET)
                ah_df["date"] = ah_df["t"].dt.date
                ah_df["hour"] = ah_df["t"].dt.hour

            # 1-min PM bars
            pm_1min_df = load_pm_1min(ticker)

            # Average PM volume for vol ratio
            avg_pm_vol = 0
            if not pm_1min_df.empty and "date" in pm_1min_df.columns:
                col_v = "v" if "v" in pm_1min_df.columns else "volume"
                past_pm = pm_1min_df[pm_1min_df["date"] < trade_date]
                if not past_pm.empty:
                    avg_pm_vol = float(past_pm.groupby("date")[col_v].sum().tail(20).mean())

            float_data     = cache.get("float_data", {})
            earnings_dates = cache.get("earnings_dates", [])
            if hasattr(earnings_dates, "tolist"):
                earnings_dates = earnings_dates.tolist()

            # Determine labels
            is_seed_today  = ticker in seed_tickers
            is_super_today = False
            if is_seed_today:
                # Check in seed_details if it was a super
                pass  # label_super set below

            # Midnight label: will it seed TOMORROW?
            midnight_label_seed  = 1 if ticker in next_seeds else 0
            midnight_label_super = 0  # simplified for now

            # Morning label: did it seed TODAY?
            morning_label_seed  = 1 if is_seed_today else 0
            morning_label_super = 0

            # Days since last seed
            ticker_seeds = seed_history.get(ticker, [])
            if ticker_seeds:
                days_since = (trade_date - max(ticker_seeds)).days
            else:
                days_since = 999

            # Assemble rows
            mid_row, mor_row = assemble_ticker_day(
                ticker=ticker,
                trade_date=trade_date,
                label_seed=midnight_label_seed,
                label_super=midnight_label_super,
                daily_df=daily_df,
                ah_df=ah_df,
                pm_1min_df=pm_1min_df,
                float_data=float_data if isinstance(float_data, dict) else {},
                earnings_dates=earnings_dates,
                si_master=si_master,
                halts=halts,
                edgar_master=edgar_master,
                cik_map=cik_map,
                sector_data=sector_data,
                days_since_last_seed=days_since,
                avg_pm_vol=avg_pm_vol,
            )

            if mid_row:
                # Override labels with correct morning labels
                mid_row["label_seed"]  = midnight_label_seed
                mor_row["label_seed"]  = morning_label_seed
                midnight_rows.append(mid_row)
                morning_rows.append(mor_row)

            # Update seed history
            if is_seed_today:
                if ticker not in seed_history:
                    seed_history[ticker] = []
                seed_history[ticker].append(trade_date)

        if midnight_rows:
            pd.DataFrame(midnight_rows).to_parquet(mid_path, index=False)
            total_midnight += len(midnight_rows)

        if morning_rows:
            pd.DataFrame(morning_rows).to_parquet(mor_path, index=False)
            total_morning += len(morning_rows)

        days_done += 1
        seeds_today = len(seed_tickers)
        if days_done % 20 == 0 or seeds_today > 0:
            log.info(
                f"{date_str}: {seeds_today} seeds | "
                f"{len(tickers_today)} tickers | "
                f"mid_rows={len(midnight_rows)} mor_rows={len(morning_rows)} | "
                f"days={days_done}/{len(trading_days)}"
            )

        gc.collect()

    log.info("=" * 60)
    log.info("ASSEMBLY COMPLETE")
    log.info(f"Total midnight rows: {total_midnight:,}")
    log.info(f"Total morning rows:  {total_morning:,}")
    log.info(f"Days processed:      {days_done}")
    log.info(f"Midnight files:      {len(list(MIDNIGHT_DIR.glob('*.parquet')))}")
    log.info(f"Morning files:       {len(list(MORNING_DIR.glob('*.parquet')))}")
    log.info("Next step: python scanner/trainer.py")
    log.info("=" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    threading.Thread(target=start_keepalive, daemon=True).start()
    time.sleep(2)

    log.info("=" * 60)
    log.info("THE DELTA v2 — Assembler")
    log.info(f"Window: {START_DATE} → {END_DATE}")
    log.info("=" * 60)

    # Load registries
    seed_reg_path = RAW_DIR / "seed_registry.json"
    ctrl_reg_path = RAW_DIR / "control_registry.json"

    if not seed_reg_path.exists():
        log.error("FAIL | seed_registry.json not found — run phase1_5 first")
        return
    if not ctrl_reg_path.exists():
        log.error("FAIL | control_registry.json not found — run phase1_5 first")
        return

    with open(seed_reg_path) as f:
        seed_registry = json.load(f)
    with open(ctrl_reg_path) as f:
        control_registry = json.load(f)

    log.info(f"Seed registry: {len(seed_registry)} days")
    log.info(f"Control registry: {len(control_registry)} days")

    # Load support data
    si_master            = load_si_master()
    halts                = load_halts_master()
    edgar_master, cik_map = load_edgar_master()

    # Load sector data
    log.info("Loading sector data...")
    sector_data = load_sector_data()

    # Trading calendar
    trading_days = get_trading_days(START_DATE, END_DATE)
    log.info(f"Trading days: {len(trading_days)}")

    # Run assembly
    assemble_all(
        seed_registry=seed_registry,
        control_registry=control_registry,
        si_master=si_master,
        halts=halts,
        edgar_master=edgar_master,
        cik_map=cik_map,
        sector_data=sector_data,
        trading_days=trading_days,
    )


if __name__ == "__main__":
    main()
