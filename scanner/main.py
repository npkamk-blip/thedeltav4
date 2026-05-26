"""
Analyze training data to find optimal hard filters for scanner.
"""
import pandas as pd
import numpy as np

df = pd.read_parquet("/home/knobp/delta_v2/data/training_master.parquet")

seeds    = df[df["label"] >= 1]
controls = df[df["label"] == 0]
supers   = df[df["label"] >= 2]
megas    = df[df["label"] >= 3]

print(f"Seeds: {len(seeds)} | Controls: {len(controls)} | Supers: {len(supers)} | Megas: {len(megas)}")
print()

key_features = [
    "prev_close", "pm_gap_pct", "pm_volume",
    "pm_remaining_to_seed", "float_M", "pm_move_pct",
    "vol_ratio_prev", "pm_vol_ratio"
]

for feat in key_features:
    if feat not in df.columns:
        continue
    s = seeds[feat].replace(-1, np.nan).dropna()
    c = controls[feat].replace(-1, np.nan).dropna()
    if len(s) == 0:
        continue
    print(f"── {feat} ──")
    print(f"  SEEDS    p10={s.quantile(0.10):.4f} p25={s.quantile(0.25):.4f} "
          f"p50={s.quantile(0.50):.4f} p75={s.quantile(0.75):.4f} p90={s.quantile(0.90):.4f}")
    print(f"  CONTROLS p10={c.quantile(0.10):.4f} p25={c.quantile(0.25):.4f} "
          f"p50={c.quantile(0.50):.4f} p75={c.quantile(0.75):.4f} p90={c.quantile(0.90):.4f}")
    print()

# Find threshold that captures 90% of seeds
print("── OPTIMAL HARD FILTERS (capture 90% of seeds) ──")
for feat, direction in [
    ("prev_close", "max"),
    ("pm_gap_pct", "min"),
    ("pm_volume", "min"),
    ("pm_remaining_to_seed", "min"),
    ("float_M", "max"),
]:
    if feat not in seeds.columns:
        continue
    s = seeds[feat].replace(-1, np.nan).dropna()
    if direction == "max":
        val = s.quantile(0.90)
        pct_captured = (s <= val).mean() * 100
    else:
        val = s.quantile(0.10)
        pct_captured = (s >= val).mean() * 100
    print(f"  {feat} {direction} {val:.4f} → captures {pct_captured:.1f}% of seeds")
