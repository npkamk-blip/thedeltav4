"""
THE DELTA v2 — data_patcher.py
================================
Patches missing features in training data:
1. Re-parses AH bars from raw ticker cache
2. Joins FINRA SI by ticker/date
3. Calculates 8-K timing from EDGAR index
4. Recalculates derived PM features
5. Fills days_since_last_seed = -1 for first-timers
6. Drops 8k_word_count
7. Saves patched training files
"""

import os
import gc
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

DATA_ROOT      = Path(os.environ.get("DATA_DIR", "/app/data")).parent
TRAINING_DIR   = DATA_ROOT / "training_data_v2"
RAW_DIR        = DATA_ROOT / "raw"
TICKER_RAW_DIR = RAW_DIR / "tickers"
FINRA_DIR      = RAW_DIR / "finra"
EDGAR_DIR      = RAW_DIR / "edgar"

EDGAR_USER_AGENT = "NPKNOB@gmail.com"

# ── Load support data ─────────────────────────────────────────
def load_support_data():
    si_master    = None
    edgar_master = None
    cik_map      = {}

    si_path = FINRA_DIR / "si_master.parquet"
    if si_path.exists():
        si_master = pd.read_parquet(si_path)
        # Normalize column names
        si_master.columns = [c.strip().lower() for c in si_master.columns]
        print(f"Loaded FINRA SI: {len(si_master)} rows")
        print(f"SI columns: {si_master.columns.tolist()[:6]}")

    edgar_path = EDGAR_DIR / "filings_master.parquet"
    if edgar_path.exists():
        edgar_master = pd.read_parquet(edgar_path)
        print(f"Loaded EDGAR: {len(edgar_master)} filings")

    cik_path = EDGAR_DIR / "cik_map.json"
    if cik_path.exists():
        with open(cik_path) as f:
            cik_map = json.load(f)
        print(f"Loaded CIK map: {len(cik_map)} tickers")

    return si_master, edgar_master, cik_map

# ── Get SI for a ticker on a date ────────────────────────────
def get_si(ticker, trade_date, si_master):
    if si_master is None:
        return {"si_pct": -1, "si_tier": "unknown", "days_to_cover": -1}

    # Find ticker column
    t_col = next((c for c in si_master.columns
                  if "symbol" in c or "ticker" in c), None)
    if not t_col:
        return {"si_pct": -1, "si_tier": "unknown", "days_to_cover": -1}

    rows = si_master[si_master[t_col] == ticker]
    if rows.empty:
        return {"si_pct": -1, "si_tier": "unknown", "days_to_cover": -1}

    # Find nearest date
    date_col = next((c for c in si_master.columns if "date" in c), None)
    if not date_col:
        return {"si_pct": -1, "si_tier": "unknown", "days_to_cover": -1}

    rows = rows.copy()
    rows["_date"] = pd.to_datetime(rows[date_col], errors="coerce").dt.date
    rows = rows.dropna(subset=["_date"])
    rows["_diff"] = rows["_date"].apply(lambda d: abs((d - trade_date).days))
    nearest = rows.nsmallest(1, "_diff").iloc[0]

    # Find short volume column
    sv_col = next((c for c in si_master.columns
                   if "short" in c and "vol" in c and "exempt" not in c), None)
    tv_col = next((c for c in si_master.columns
                   if "total" in c and "vol" in c), None)

    si_pct = -1
    if sv_col and tv_col:
        try:
            sv = float(nearest[sv_col])
            tv = float(nearest[tv_col])
            if tv > 0:
                si_pct = round((sv / tv) * 100, 2)
        except Exception:
            pass

    si_tier = "unknown"
    if si_pct >= 0:
        if si_pct < 5:
            si_tier = "low"
        elif si_pct < 15:
            si_tier = "medium"
        elif si_pct < 30:
            si_tier = "high"
        else:
            si_tier = "extreme"

    return {"si_pct": si_pct, "si_tier": si_tier, "days_to_cover": -1}

# ── Get AH features from raw cache ───────────────────────────
def get_ah_features(ticker, prev_date):
    cache_path = TICKER_RAW_DIR / f"{ticker}.parquet"
    if not cache_path.exists():
        return {"ah_move_pct": 0, "ah_volume": 0, "ah_direction": 0}

    try:
        df = pd.read_parquet(cache_path)
        row = df.iloc[0].to_dict()
        ah_raw = row.get("ah_minute")

        if ah_raw is None:
            return {"ah_move_pct": 0, "ah_volume": 0, "ah_direction": 0}

        # Convert numpy array to DataFrame
        if hasattr(ah_raw, "tolist"):
            ah_df = pd.DataFrame(ah_raw.tolist())
        elif isinstance(ah_raw, list):
            ah_df = pd.DataFrame(ah_raw)
        else:
            ah_df = pd.DataFrame(list(ah_raw))

        if ah_df.empty:
            return {"ah_move_pct": 0, "ah_volume": 0, "ah_direction": 0}

        # Get date column
        if "date" not in ah_df.columns and "t" in ah_df.columns:
            ah_df["date"] = pd.to_datetime(
                ah_df["t"], unit="ms", utc=True
            ).dt.tz_convert(ET).dt.date
        elif "date" in ah_df.columns:
            ah_df["date"] = pd.to_datetime(ah_df["date"]).dt.date

        # Filter to prev_date AH (4PM-8PM)
        if "t" in ah_df.columns:
            ah_df["hour"] = pd.to_datetime(
                ah_df["t"], unit="ms", utc=True
            ).dt.tz_convert(ET).dt.hour
            ah_day = ah_df[
                (ah_df["date"] == prev_date) &
                (ah_df["hour"] >= 16) &
                (ah_df["hour"] < 20)
            ]
        else:
            ah_day = ah_df[ah_df["date"] == prev_date]

        if ah_day.empty:
            return {"ah_move_pct": 0, "ah_volume": 0, "ah_direction": 0}

        ah_open  = float(ah_day.iloc[0].get("o", ah_day.iloc[0].get("open", 0)))
        ah_close = float(ah_day.iloc[-1].get("c", ah_day.iloc[-1].get("close", 0)))
        ah_vol   = float(ah_day.get("v", ah_day.get("volume", pd.Series([0]))).sum())

        move = 0
        if ah_open > 0:
            move = (ah_close - ah_open) / ah_open

        direction = 1 if ah_close > ah_open else (-1 if ah_close < ah_open else 0)

        return {
            "ah_move_pct": round(move, 4),
            "ah_volume":   ah_vol,
            "ah_direction": direction,
        }
    except Exception as e:
        return {"ah_move_pct": 0, "ah_volume": 0, "ah_direction": 0}

# ── Get 8K timing from EDGAR ──────────────────────────────────
def get_8k_timing(ticker, trade_date, edgar_master, cik_map):
    out = {
        "8k_filing_hour":   -1,
        "hours_before_open": -1,
        "days_since_last_8k": 999,
        "days_since_dilution": 999,
    }

    if edgar_master is None:
        return out

    cik = cik_map.get(ticker)
    if not cik:
        return out

    ticker_filings = edgar_master[edgar_master["cik"] == cik].copy()
    if ticker_filings.empty:
        return out

    ticker_filings["filed_date"] = pd.to_datetime(
        ticker_filings["filed"], errors="coerce"
    ).dt.date

    # 8-K on trade_date
    eightk = ticker_filings[ticker_filings["form_type"].isin(["8-K", "8-K/A"])]
    today_8k = eightk[eightk["filed_date"] == trade_date]

    if not today_8k.empty:
        # Try to get hour from filed timestamp
        try:
            filed_ts = pd.to_datetime(today_8k.iloc[0]["filed"])
            if hasattr(filed_ts, "hour"):
                out["8k_filing_hour"] = filed_ts.hour
                # Hours before market open (9:30 AM)
                if filed_ts.hour < 9 or (filed_ts.hour == 9 and filed_ts.minute < 30):
                    out["hours_before_open"] = round(
                        (9.5 - filed_ts.hour - filed_ts.minute/60), 2
                    )
                else:
                    out["hours_before_open"] = 0
        except Exception:
            pass

    # Days since last 8-K
    past_8k = eightk[eightk["filed_date"] < trade_date]
    if not past_8k.empty:
        last_8k = past_8k["filed_date"].max()
        out["days_since_last_8k"] = (trade_date - last_8k).days

    # Days since last dilution
    dilution_forms = {"S-1", "S-1/A", "S-3", "S-3/A", "424B1", "424B3", "424B4", "424B5"}
    dilution = ticker_filings[
        (ticker_filings["form_type"].isin(dilution_forms)) &
        (ticker_filings["filed_date"] < trade_date)
    ]
    if not dilution.empty:
        last_dil = dilution["filed_date"].max()
        out["days_since_dilution"] = (trade_date - last_dil).days

    return out

# ── Patch one parquet file ────────────────────────────────────
def patch_file(filepath, si_master, edgar_master, cik_map):
    df = pd.read_parquet(filepath)

    # Drop 8k_word_count
    if "8k_word_count" in df.columns:
        df = df.drop(columns=["8k_word_count"])

    # Fill days_since_last_seed = -1 for nulls
    if "days_since_last_seed" in df.columns:
        df["days_since_last_seed"] = df["days_since_last_seed"].fillna(-1)

    # Fill PM features = 0 for nulls
    pm_cols = ["pm_volume_build", "pm_fade", "pm_vol_ratio",
               "pm_remaining_to_seed", "pm_remaining_to_super",
               "pm_remaining_to_mega", "pm_high_of_session"]
    for col in pm_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill float features
    if "float_tier" in df.columns:
        df["float_tier"] = df["float_tier"].fillna("unknown")
    if "float_rotation_prev" in df.columns:
        df["float_rotation_prev"] = df["float_rotation_prev"].fillna(0)

    # Fill earnings
    if "days_to_earnings" in df.columns:
        df["days_to_earnings"] = df["days_to_earnings"].fillna(-1)

    # Patch SI, AH, 8K per row
    for i, row in df.iterrows():
        ticker     = row["ticker"]
        trade_date = pd.to_datetime(row["date"]).date()
        prev_date  = trade_date - timedelta(days=1)
        while prev_date.weekday() >= 5:
            prev_date -= timedelta(days=1)

        # SI
        if pd.isna(row.get("si_pct")) or row.get("si_pct") == -1:
            si = get_si(ticker, trade_date, si_master)
            for k, v in si.items():
                if k in df.columns:
                    df.at[i, k] = v

        # AH
        if pd.isna(row.get("ah_move_pct")):
            ah = get_ah_features(ticker, prev_date)
            for k, v in ah.items():
                if k in df.columns:
                    df.at[i, k] = v

        # 8K timing
        if row.get("days_since_last_8k") is None or pd.isna(row.get("days_since_last_8k", 0)):
            timing = get_8k_timing(ticker, trade_date, edgar_master, cik_map)
            for k, v in timing.items():
                if k in df.columns:
                    df.at[i, k] = v

    # Final fills for any remaining nulls
    df["si_pct"]          = df["si_pct"].fillna(-1)
    df["si_tier"]         = df["si_tier"].fillna("unknown")
    df["days_to_cover"]   = df["days_to_cover"].fillna(-1)
    df["ah_move_pct"]     = df["ah_move_pct"].fillna(0)
    df["ah_volume"]       = df["ah_volume"].fillna(0)
    df["ah_direction"]    = df["ah_direction"].fillna(0)
    df["8k_filing_hour"]  = df["8k_filing_hour"].fillna(-1)
    df["hours_before_open"] = df["hours_before_open"].fillna(-1)
    df["days_since_last_8k"] = df["days_since_last_8k"].fillna(999)
    df["days_since_dilution"] = df["days_since_dilution"].fillna(999)

    df.to_parquet(filepath, index=False)
    return len(df)

# ── Main ──────────────────────────────────────────────────────
def main():
    print("THE DELTA v2 — Data Patcher")
    print("=" * 50)

    si_master, edgar_master, cik_map = load_support_data()

    files = sorted(TRAINING_DIR.glob("*.parquet"))
    print(f"Patching {len(files)} training files...")

    total_rows = 0
    for i, fp in enumerate(files):
        rows = patch_file(fp, si_master, edgar_master, cik_map)
        total_rows += rows
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(files)} files patched")
        gc.collect()

    print(f"\nDone. {total_rows} total rows patched.")
    print("Next step: run data_fixer.py")

if __name__ == "__main__":
    main()
