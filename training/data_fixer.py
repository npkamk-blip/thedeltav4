"""
THE DELTA v2 — data_fixer.py
==============================
Cleans and validates the patched training data.
Fills remaining nulls with appropriate sentinel values.
Saves a clean master training file ready for trainer.py
"""

import os
import gc
import pandas as pd
import numpy as np
import glob
from pathlib import Path

DATA_ROOT    = Path(os.environ.get("DATA_DIR", "/app/data")).parent
TRAINING_DIR = DATA_ROOT / "training_data_v2"
OUTPUT_FILE  = DATA_ROOT / "training_master.parquet"

def main():
    print("THE DELTA v2 — Data Fixer")
    print("=" * 50)

    # ── Load all training files ───────────────────────────────
    files = sorted(TRAINING_DIR.glob("*.parquet"))
    print(f"Loading {len(files)} training files...")

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Skipping corrupted file: {f.name} — {e}")

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    print(f"Total rows loaded: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # ── Label distribution before fixing ─────────────────────
    print("\nLabel distribution:")
    print(f"  Controls: {len(df[df['label']==0])}")
    print(f"  Seeds:    {len(df[df['label']==1])}")
    print(f"  Supers:   {len(df[df['label']==2])}")
    print(f"  Megas:    {len(df[df['label']==3])}")

    # ── Drop columns we don't need ────────────────────────────
    drop_cols = ["8k_word_count"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped: {col}")

    # ── Fill nulls by group ───────────────────────────────────

    # PM features — 0 means no premarket activity (valid info)
    pm_zero_cols = [
        "pm_open", "pm_high", "pm_low", "pm_close", "pm_volume",
        "pm_gap_pct", "pm_move_pct", "pm_vol_ratio",
        "pm_volume_build", "pm_fade", "pm_high_of_session",
        "pm_remaining_to_seed", "pm_remaining_to_super", "pm_remaining_to_mega",
    ]
    for col in pm_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # AH features — 0 means no AH activity
    ah_zero_cols = ["ah_move_pct", "ah_volume", "ah_direction"]
    for col in ah_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # SI features — -1 means unknown
    si_neg_cols = ["si_pct", "days_to_cover"]
    for col in si_neg_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
    if "si_tier" in df.columns:
        df["si_tier"] = df["si_tier"].fillna("unknown")

    # Float features — -1 means unknown
    float_neg_cols = ["float_shares", "float_M", "float_rotation_prev"]
    for col in float_neg_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
    if "float_tier" in df.columns:
        df["float_tier"] = df["float_tier"].fillna("unknown")
    if "market_cap" in df.columns:
        df["market_cap"] = df["market_cap"].fillna(-1)

    # Historical features — 0 for new listings
    hist_zero_cols = [
        "avg_volume_20d", "vol_ratio_prev",
        "prev_3d_trend", "prev_5d_trend", "prev_10d_trend",
        "days_since_last_spike", "coil_days", "vol_trend_3d",
        "consecutive_vol_days", "near_52w_low",
        "price_52w_high", "price_52w_low",
        "pct_from_52w_high", "pct_from_52w_low",
        "prev_wick_ratio", "prev_body_pct",
    ]
    for col in hist_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Time-to-event — 999 means "very long ago / never"
    tte_999_cols = [
        "days_since_last_8k", "days_since_last_halt",
        "days_since_last_dilution", "days_since_dilution",
        "days_since_last_seed",
    ]
    for col in tte_999_cols:
        if col in df.columns:
            df[col] = df[col].fillna(999)

    # Earnings — -1 means unknown
    earn_cols = ["days_to_earnings"]
    for col in earn_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
    if "has_earnings_soon" in df.columns:
        df["has_earnings_soon"] = df["has_earnings_soon"].fillna(0)
    if "had_earnings_recently" in df.columns:
        df["had_earnings_recently"] = df["had_earnings_recently"].fillna(0)

    # Sector features — 0 means neutral/unknown
    sector_zero_cols = [
        "spy_prev_day_pct", "qqq_prev_day_pct",
        "iwm_prev_day_pct", "xbi_prev_day_pct",
        "pm_spy_pct", "market_green", "market_red", "sector_hot",
    ]
    for col in sector_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # EDGAR flags — 0 means no filing
    edgar_zero_cols = [
        "has_8k", "has_8k_yesterday", "has_8k_2days_ago",
        "has_merger", "has_fda", "has_contract", "has_earnings",
        "has_reverse_split", "has_dilution", "has_buyback",
        "dilution_count_6m", "dilution_count_30d",
        "reverse_split_count", "is_serial_diluter", "is_serial_reverser",
        "has_form4_buy", "form4_buy_count", "has_sc13d",
        "halted_yesterday", "halt_count_5d", "halt_count_30d",
        "is_serial_halter", "is_foreign_listed",
    ]
    for col in edgar_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 8K timing — -1 means no filing
    timing_cols = ["8k_filing_hour", "hours_before_open"]
    for col in timing_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # Fetch ok flags — 0 means failed
    fetch_cols = [c for c in df.columns if "fetch_ok" in c]
    for col in fetch_cols:
        df[col] = df[col].fillna(0)

    # ── Encode categorical columns ────────────────────────────
    if "si_tier" in df.columns:
        tier_map = {"unknown": -1, "low": 0, "medium": 1, "high": 2, "extreme": 3}
        df["si_tier"] = df["si_tier"].map(tier_map).fillna(-1)

    if "float_tier" in df.columns:
        float_map = {"unknown": -1, "nano": 0, "micro": 1, "small": 2, "mid": 3, "large": 4}
        df["float_tier"] = df["float_tier"].map(float_map).fillna(-1)

    # ── Final null check ──────────────────────────────────────
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]

    if len(remaining_nulls) > 0:
        print(f"\nRemaining nulls after fixing:")
        print(remaining_nulls)
    else:
        print("\nNo nulls remaining ✅")

    # ── Data quality report ───────────────────────────────────
    print("\nData quality report:")
    print(f"  Total rows:     {len(df)}")
    print(f"  Total columns:  {len(df.columns)}")
    print(f"  Controls:       {len(df[df['label']==0])}")
    print(f"  Seeds:          {len(df[df['label']==1])}")
    print(f"  Supers:         {len(df[df['label']==2])}")
    print(f"  Megas:          {len(df[df['label']==3])}")
    print(f"  Date range:     {df['date'].min()} → {df['date'].max()}")
    print(f"  Unique tickers: {df['ticker'].nunique()}")

    # Seeds with PM data
    seeds = df[df["label"] >= 1]
    pm_coverage = (seeds["pm_volume"] > 0).sum() / len(seeds) * 100
    print(f"  Seeds with PM data: {pm_coverage:.1f}%")

    # ── Save master file ──────────────────────────────────────
    print(f"\nSaving master training file to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"Saved: {size_mb:.1f} MB")
    print("\nNext step: run trainer.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
