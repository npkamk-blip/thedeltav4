"""
Quick re-run of just Experiment 2 gap bucket analysis
Saves to /app/data/logs/backtest_exp2.txt
"""
import os, json, sys
import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT   = Path(os.environ.get("DATA_DIR", "/app/data"))
MORNING_DIR = DATA_ROOT / "training_data_v2" / "morning"
MODEL_DIR   = DATA_ROOT / "models"
LOG_DIR     = DATA_ROOT / "logs"

from xgboost import XGBClassifier

with open(MODEL_DIR / "feature_cols.json") as f:
    feature_cols_map = json.load(f)

midnight_model = XGBClassifier()
midnight_model.load_model(str(MODEL_DIR / "midnight_seed_model.json"))
midnight_cols = feature_cols_map.get("midnight_seed", [])

morning_model = XGBClassifier()
morning_model.load_model(str(MODEL_DIR / "morning_seed_model.json"))
morning_cols = feature_cols_map.get("morning_seed", [])

# Load test data
files = sorted(MORNING_DIR.glob("*.parquet"))
test_files = [f for f in files if f.stem >= "2026-01-01"]
all_rows = [pd.read_parquet(f) for f in test_files]
test_df = pd.concat(all_rows, ignore_index=True)
test_df["date"] = test_df["date"].astype(str)

# Score
mid_avail = [c for c in midnight_cols if c in test_df.columns]
X_mid = test_df[mid_avail].fillna(-1).replace([np.inf, -np.inf], -1)
test_df["midnight_score"] = midnight_model.predict_proba(X_mid)[:, 1]

mor_avail = [c for c in morning_cols if c in test_df.columns]
X_mor = test_df[mor_avail].fillna(-1).replace([np.inf, -np.inf], -1)
test_df["morning_score"] = morning_model.predict_proba(X_mor)[:, 1]

lines = []
lines.append("=" * 60)
lines.append("EXPERIMENT 1: MIDNIGHT MODEL — threshold analysis")
lines.append("=" * 60)
total_seeds = int(test_df["label_seed"].sum())
lines.append(f"{'Threshold':>10} {'Alerts':>8} {'Seeds':>8} {'HitRate':>9} {'Recall':>9} {'Alerts/day':>11}")
lines.append("-" * 60)
for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    flagged = test_df[test_df["midnight_score"] >= thresh]
    if len(flagged) == 0:
        continue
    seeds = int(flagged["label_seed"].sum())
    hit = seeds / len(flagged)
    recall = seeds / total_seeds
    n_days = len(flagged["date"].unique())
    apd = len(flagged) / n_days
    lines.append(f"{thresh:>10.2f} {len(flagged):>8,} {seeds:>8,} {hit:>8.1%} {recall:>8.1%} {apd:>10.1f}")

lines.append("")
lines.append("=" * 60)
lines.append("EXPERIMENT 2: MORNING MODEL — gap bucket analysis")
lines.append("For SEED stocks only — at what PM gap does model fire?")
lines.append("=" * 60)
seeds_only = test_df[test_df["label_seed"] == 1].copy()
gap_buckets = [
    ("flat(<2%)",     -0.99,  0.02),
    ("small(2-10%)",   0.02,  0.10),
    ("med(10-25%)",    0.10,  0.25),
    ("large(25-50%)",  0.25,  0.50),
    ("huge(50-100%)",  0.50,  1.00),
    ("seeded(100%+)",  1.00,  99.0),
]
lines.append(f"{'Gap Range':>15} {'Count':>7} {'AvgScore':>10} {'%Alert@0.55':>13} {'%Alert@0.70':>13}")
lines.append("-" * 63)
for label, lo, hi in gap_buckets:
    b = seeds_only[(seeds_only["pm_gap_pct"] >= lo) & (seeds_only["pm_gap_pct"] < hi)]
    if len(b) == 0:
        continue
    avg = b["morning_score"].mean()
    p55 = (b["morning_score"] >= 0.55).mean()
    p70 = (b["morning_score"] >= 0.70).mean()
    lines.append(f"{label:>15} {len(b):>7,} {avg:>10.3f} {p55:>12.1%} {p70:>12.1%}")

lines.append("")
lines.append("=" * 60)
lines.append("EXPERIMENT 2B: Score distribution at early gaps")
lines.append("Seeds with pm_gap < 10% — what scores do they get?")
lines.append("=" * 60)
early = seeds_only[seeds_only["pm_gap_pct"] < 0.10].copy()
lines.append(f"Total seeds with PM gap < 10%: {len(early)}")
lines.append(f"  Score >= 0.55: {(early['morning_score'] >= 0.55).sum()} ({(early['morning_score'] >= 0.55).mean():.1%})")
lines.append(f"  Score >= 0.70: {(early['morning_score'] >= 0.70).sum()} ({(early['morning_score'] >= 0.70).mean():.1%})")
lines.append(f"  Score >= 0.80: {(early['morning_score'] >= 0.80).sum()} ({(early['morning_score'] >= 0.80).mean():.1%})")
lines.append(f"  Avg score:     {early['morning_score'].mean():.3f}")
lines.append(f"  Avg midnight:  {early['midnight_score'].mean():.3f}")
lines.append("")
lines.append("Top 20 early catches (gap < 10%, label=1, sorted by morning score):")
show_cols = ["ticker","date","morning_score","midnight_score","pm_gap_pct","pm_move_pct","float_M","ah_move_pct","has_8k"]
avail = [c for c in show_cols if c in early.columns]
top_early = early.nlargest(20, "morning_score")[avail]
lines.append(top_early.to_string(index=False))

output = "\n".join(lines)
print(output)

out_file = LOG_DIR / "backtest_exp2.txt"
with open(out_file, "w") as f:
    f.write(output)
print(f"\nSaved to {out_file}")
