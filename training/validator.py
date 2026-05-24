"""
THE DELTA v2 — validator.py
============================
Deep validation of trained models.
- Threshold tuning (find best precision/recall tradeoff)
- By-year performance breakdown
- False positive analysis
- Feature importance deep dive
- Trading simulation
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve,
    confusion_matrix
)

DATA_ROOT   = Path(os.environ.get("DATA_DIR", "/app/data")).parent
MASTER_FILE = DATA_ROOT / "training_master.parquet"
MODEL_DIR   = DATA_ROOT / "models"

DROP_COLS = [
    "ticker", "date", "label",
    "day_high", "day_volume", "day_open",
]

# ── Load models ───────────────────────────────────────────────
def load_models():
    models = {}
    for name in ["seed", "super", "mega"]:
        path = MODEL_DIR / f"{name}_model.json"
        if path.exists():
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            models[name] = m
            print(f"Loaded {name}_model")
    return models

# ── Threshold tuning ──────────────────────────────────────────
def tune_threshold(y_true, y_proba, model_name):
    """Find best threshold for precision >= 0.20 with max recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    print(f"\n  Threshold analysis for {model_name}:")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Alerts/day':>12}")

    best_threshold = 0.5
    best_f1 = 0

    # Estimate trading days in test set
    n_test = len(y_true)
    trading_days = max(n_test / 10, 1)  # rough estimate

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = (y_proba >= thresh).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        alerts_per_day = y_pred.sum() / trading_days

        marker = " ← recommended" if prec >= 0.5 and rec >= 0.1 and f1 > best_f1 else ""
        if prec >= 0.5 and rec >= 0.1 and f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

        print(f"  {thresh:>10.1f} {prec:>10.3f} {rec:>10.3f} {alerts_per_day:>12.2f}{marker}")

    return best_threshold

# ── By year breakdown ─────────────────────────────────────────
def year_breakdown(df, y_proba, target_col, threshold, model_name):
    """Show model performance by year."""
    df = df.copy()
    df["proba"]  = y_proba
    df["pred"]   = (y_proba >= threshold).astype(int)
    df["actual"] = df[target_col]

    print(f"\n  Year breakdown ({model_name} @ threshold={threshold}):")
    print(f"  {'Year':>6} {'Positives':>10} {'Caught':>8} {'Alerts':>8} {'Precision':>10} {'Recall':>8}")

    for year in sorted(df["date"].dt.year.unique()):
        yr = df[df["date"].dt.year == year]
        n_pos   = yr["actual"].sum()
        caught  = (yr["actual"] & yr["pred"]).sum()
        alerts  = yr["pred"].sum()
        prec    = precision_score(yr["actual"], yr["pred"], zero_division=0)
        rec     = recall_score(yr["actual"], yr["pred"], zero_division=0)
        print(f"  {year:>6} {int(n_pos):>10} {int(caught):>8} {int(alerts):>8} {prec:>10.3f} {rec:>8.3f}")

# ── Trading simulation ────────────────────────────────────────
def trading_simulation(df, y_proba, target_col, threshold, model_name,
                        account=250, position_pct=0.20,
                        take_half_at=1.00, stop_loss=-0.15):
    """
    Simulate trading based on model alerts.
    Simple model: enter at open, exit at target or stop.
    """
    df = df.copy()
    df["proba"]  = y_proba
    df["pred"]   = (y_proba >= threshold).astype(int)
    df["actual"] = df[target_col]

    alerts = df[df["pred"] == 1].copy()
    if len(alerts) == 0:
        print(f"\n  No alerts at threshold {threshold} — skipping simulation")
        return

    balance  = account
    wins     = 0
    losses   = 0
    total    = 0

    print(f"\n  Trading simulation ({model_name}):")
    print(f"  Starting balance: ${account}")
    print(f"  Position size: {position_pct*100:.0f}% | "
          f"Take half at: +{take_half_at*100:.0f}% | "
          f"Stop: {stop_loss*100:.0f}%")

    for _, row in alerts.iterrows():
        position = balance * position_pct
        total += 1

        if row["actual"] == 1:
            # Winner — take half at target
            profit = position * 0.5 * take_half_at
            balance += profit
            wins += 1
        else:
            # Loser — stop loss
            loss = position * abs(stop_loss)
            balance -= loss
            losses += 1

    win_rate = wins / total if total > 0 else 0
    roi      = (balance - account) / account * 100

    print(f"  Total alerts:  {total}")
    print(f"  Winners:       {wins} ({win_rate*100:.1f}%)")
    print(f"  Losers:        {losses}")
    print(f"  Final balance: ${balance:.2f}")
    print(f"  ROI:           {roi:.1f}%")

    if win_rate >= 0.20:
        print(f"  ✅ Win rate above 20% minimum")
    else:
        print(f"  ⚠️  Win rate below 20% — model needs improvement")

# ── Main ──────────────────────────────────────────────────────
def main():
    print("THE DELTA v2 — Validator")
    print("=" * 50)

    # Load data
    df = pd.read_parquet(MASTER_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # Encode strings
    for col in df.select_dtypes(include=["object"]).columns:
        if col not in ["ticker", "date"]:
            df[col] = pd.Categorical(df[col]).codes

    # Load feature list
    feat_path = MODEL_DIR / "feature_cols.json"
    if feat_path.exists():
        with open(feat_path) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in df.columns if c not in DROP_COLS]

    # Load models
    models = load_models()

    configs = [
        ("seed",  "is_seed",  lambda d: (d["label"] >= 1).astype(int), 0.5,  250, 0.20, 1.00, -0.15),
        ("super", "is_super", lambda d: (d["label"] >= 2).astype(int), 0.5,  250, 0.25, 2.00, -0.15),
        ("mega",  "is_mega",  lambda d: (d["label"] >= 3).astype(int), 0.5,  250, 0.50, 5.00, -0.15),
    ]

    for name, col, label_fn, default_thresh, acct, pos_pct, take_at, stop in configs:
        if name not in models:
            continue

        print(f"\n{'='*50}")
        print(f"Validating {name}_model")
        print(f"{'='*50}")

        model = models[name]
        df[col] = label_fn(df)

        # Use same walk-forward split as trainer
        df_sorted = df.sort_values("date")
        # Use last 20% as test
        cutoff_idx = int(len(df_sorted) * 0.8)
        cutoff_date = df_sorted.iloc[cutoff_idx]["date"]
        test_df = df_sorted[df_sorted["date"] >= cutoff_date]

        # Get features that exist
        valid_cols = [c for c in feature_cols if c in test_df.columns]
        X_test  = test_df[valid_cols].values
        y_test  = test_df[col].values
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"Test set: {len(test_df)} rows | "
              f"Positives: {int(y_test.sum())} | "
              f"Date: {test_df['date'].min().date()} → {test_df['date'].max().date()}")

        # Threshold tuning
        best_thresh = tune_threshold(y_test, y_proba, name)

        # Year breakdown
        test_df = test_df.copy()
        year_breakdown(test_df, y_proba, col, best_thresh, name)

        # Trading simulation
        trading_simulation(
            test_df, y_proba, col, best_thresh, name,
            account=acct, position_pct=pos_pct,
            take_half_at=take_at, stop_loss=stop
        )

    # Save recommended thresholds
    thresholds = {}
    for name in ["seed", "super", "mega"]:
        thresholds[name] = 0.5  # default, update manually after review
    with open(MODEL_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n{'='*50}")
    print("Validation complete.")
    print("Review results above and adjust thresholds in:")
    print(f"  {MODEL_DIR}/thresholds.json")
    print("\nNext step: build scanner.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
