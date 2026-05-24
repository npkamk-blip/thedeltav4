"""
THE DELTA v2 — trainer.py
==========================
Trains three independent XGBoost models:
  seed_model:  predicts 100%+ same-day movers
  super_model: predicts 500%+ same-day movers
  mega_model:  predicts 1000%+ same-day movers

Walk-forward validation:
  Walks back from end of dataset until minimum positives in holdout
  Trains on everything before that cutoff
  Tests on holdout period

Sample weights:
  2026 = 4x, 2025 = 3x, 2024 = 2x
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    import xgboost as xgb
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run(["pip3", "install", "xgboost", "scikit-learn", "--break-system-packages", "-q"])
    import xgboost as xgb
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )

DATA_ROOT   = Path(os.environ.get("DATA_DIR", "/app/data")).parent
MASTER_FILE = DATA_ROOT / "training_master.parquet"
MODEL_DIR   = DATA_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Minimum positives required in holdout
MIN_HOLDOUT = {
    "seed":  50,
    "super": 20,
    "mega":  10,
}

# Sample weights by year
YEAR_WEIGHTS = {2024: 2, 2025: 3, 2026: 4}

# Columns to drop before training
DROP_COLS = [
    "ticker", "date", "label",
    "day_high", "day_volume", "day_open",  # outcome leakage
]

# ── Find walk-forward cutoff date ─────────────────────────────
def find_cutoff(df, target_col, min_positives):
    """
    Walk back from end of dataset until we have
    min_positives in the holdout period.
    Returns (train_df, test_df)
    """
    df = df.sort_values("date").reset_index(drop=True)
    dates = sorted(df["date"].unique())

    for i in range(len(dates) - 1, 0, -1):
        cutoff = dates[i]
        test_df  = df[df["date"] >= cutoff]
        n_pos    = test_df[target_col].sum()
        if n_pos >= min_positives:
            train_df = df[df["date"] < cutoff]
            print(f"  Cutoff: {cutoff.date()} | "
                  f"Train: {len(train_df)} rows | "
                  f"Test: {len(test_df)} rows ({int(n_pos)} positives)")
            return train_df, test_df

    # Fallback: 80/20 split
    cutoff = dates[int(len(dates) * 0.8)]
    train_df = df[df["date"] < cutoff]
    test_df  = df[df["date"] >= cutoff]
    print(f"  Fallback 80/20 split at {cutoff.date()}")
    return train_df, test_df

# ── Build sample weights ───────────────────────────────────────
def build_weights(df):
    weights = df["date"].dt.year.map(YEAR_WEIGHTS).fillna(1)
    return weights.values

# ── Train one model ───────────────────────────────────────────
def train_model(df, model_name, target_col, min_holdout):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")

    # Positive counts
    n_pos = df[target_col].sum()
    n_neg = len(df) - n_pos
    print(f"Total: {len(df)} rows | Positives: {int(n_pos)} | Negatives: {int(n_neg)}")

    if n_pos < 10:
        print(f"Not enough positives to train {model_name} — skipping")
        return None, None

    # Walk-forward split
    train_df, test_df = find_cutoff(df, target_col, min_holdout)

    # Features
    feature_cols = [c for c in df.columns
                    if c not in DROP_COLS and c != target_col]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target_col].values

    # Sample weights
    w_train = build_weights(train_df)

    # Scale pos weight
    n_pos_train = y_train.sum()
    n_neg_train = len(y_train) - n_pos_train
    scale_pos_weight = np.sqrt(n_neg_train / max(n_pos_train, 1))
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        auc = 0.0

    print(f"\n  Results:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  AUC:       {auc:.3f}")

    if precision < 0.20:
        print(f"  ⚠️  Precision below 20% minimum — model needs review")
    else:
        print(f"  ✅ Precision above 20% minimum")

    # Top features
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 15 features:")
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']:<35} {row['importance']:.4f}")

    # Save metrics
    metrics = {
        "model_name":       model_name,
        "trained_at":       datetime.now().isoformat(),
        "n_train":          len(train_df),
        "n_test":           len(test_df),
        "n_positives_test": int(y_test.sum()),
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1":               round(f1, 4),
        "auc":              round(auc, 4),
        "top_features":     importance.head(20)["feature"].tolist(),
    }

    return model, metrics

# ── Main ──────────────────────────────────────────────────────
def main():
    print("THE DELTA v2 — Trainer")
    print("=" * 50)

    # Load master file
    print(f"Loading {MASTER_FILE}...")
    df = pd.read_parquet(MASTER_FILE)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Encode any remaining string columns
    for col in df.select_dtypes(include=["object"]).columns:
        if col not in ["ticker", "date"]:
            df[col] = pd.Categorical(df[col]).codes

    all_metrics = {}

    # ── Seed model ────────────────────────────────────────────
    seed_df = df.copy()
    seed_df["is_seed"] = (seed_df["label"] >= 1).astype(int)
    seed_model, seed_metrics = train_model(
        seed_df, "seed_model", "is_seed", MIN_HOLDOUT["seed"]
    )
    if seed_model:
        seed_model.save_model(str(MODEL_DIR / "seed_model.json"))
        all_metrics["seed"] = seed_metrics
        print(f"\n  Saved: {MODEL_DIR}/seed_model.json")

    # ── Super model ───────────────────────────────────────────
    super_df = df.copy()
    super_df["is_super"] = (super_df["label"] >= 2).astype(int)
    super_model, super_metrics = train_model(
        super_df, "super_model", "is_super", MIN_HOLDOUT["super"]
    )
    if super_model:
        super_model.save_model(str(MODEL_DIR / "super_model.json"))
        all_metrics["super"] = super_metrics
        print(f"\n  Saved: {MODEL_DIR}/super_model.json")

    # ── Mega model ────────────────────────────────────────────
    mega_df = df.copy()
    mega_df["is_mega"] = (mega_df["label"] >= 3).astype(int)
    mega_model, mega_metrics = train_model(
        mega_df, "mega_model", "is_mega", MIN_HOLDOUT["mega"]
    )
    if mega_model:
        mega_model.save_model(str(MODEL_DIR / "mega_model.json"))
        all_metrics["mega"] = mega_metrics
        print(f"\n  Saved: {MODEL_DIR}/mega_model.json")

    # ── Save feature list ─────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in DROP_COLS + ["is_seed", "is_super", "is_mega"]]
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    # ── Save all metrics ──────────────────────────────────────
    with open(MODEL_DIR / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*50}")
    print("Training complete.")
    print(f"Models saved to: {MODEL_DIR}")
    print("\nSummary:")
    for name, m in all_metrics.items():
        print(f"  {name}: precision={m['precision']:.3f} recall={m['recall']:.3f} auc={m['auc']:.3f}")
    print("\nNext step: run validator.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
