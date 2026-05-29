"""
THE DELTA v2 — trainer.py
==========================
Trains 4 XGBoost models from assembled training data:

  midnight_seed_model  — predicts 2x+ next day (label_seed=1)
  midnight_super_model — predicts 3.5x+ next day (label_super=1)
  morning_seed_model   — predicts 2x+ today (label_seed=1)
  morning_super_model  — predicts 3.5x+ today (label_super=1)

Train/test split: time-based
  Train: 2025-01-01 → 2025-12-31
  Test:  2026-01-01 → 2026-05-23

Outputs:
  /app/data/models/midnight_seed_model.json
  /app/data/models/midnight_super_model.json
  /app/data/models/morning_seed_model.json
  /app/data/models/morning_super_model.json
  /app/data/models/feature_cols.json
  /app/data/models/thresholds.json
  /app/data/models/training_report.json
"""

import os, json, gc, logging, time, threading, sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from http.server import HTTPServer, BaseHTTPRequestHandler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT    = Path(os.environ.get("DATA_DIR", "/app/data"))
MIDNIGHT_DIR = DATA_ROOT / "training_data_v2" / "midnight"
MORNING_DIR  = DATA_ROOT / "training_data_v2" / "morning"
MODEL_DIR    = DATA_ROOT / "models"
LOG_DIR      = DATA_ROOT / "logs"

TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"

# XGBoost hyperparameters — FULL POWER
# More trees + early stopping = learns complex patterns without overfitting
XGB_PARAMS_BASE = {
    "n_estimators":       1000,   # up to 1000 trees, early stopping cuts it short
    "max_depth":          6,      # deeper trees = more complex pattern recognition
    "learning_rate":      0.01,   # slow learning = better generalization
    "subsample":          0.8,    # row sampling = prevents overfitting
    "colsample_bytree":   0.7,    # feature sampling per tree = forces diversity
    "colsample_bylevel":  0.7,    # feature sampling per level = more diversity
    "min_child_weight":   3,      # lower = allows more splits on rare events
    "gamma":              0.05,   # minimum split gain — low = more splits
    "reg_alpha":          0.05,   # L1 regularization — light
    "reg_lambda":         0.5,    # L2 regularization — light
    "random_state":       42,
    "eval_metric":        "aucpr",  # optimize for area under precision-recall curve
                                    # better than logloss for imbalanced data
    "early_stopping_rounds": 50,  # stop if no improvement for 50 rounds
    "use_label_encoder":  False,
}

# Features to NEVER use as inputs (labels, identifiers, leakage)
EXCLUDE_COLS = {
    "ticker", "date",
    "label_seed", "label_super",
    "hist_fetch_ok", "pm_fetch_ok", "ah_fetch_ok",
    "float_fetch_ok", "earnings_fetch_ok", "si_fetch_ok",
    "edgar_fetch_ok", "halt_fetch_ok", "sector_fetch_ok",
}

# ─────────────────────────────────────────────
# DIRS + LOGGING
# ─────────────────────────────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "trainer.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("trainer")


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
# LOAD TRAINING DATA
# ─────────────────────────────────────────────
def load_dataset(data_dir: Path, label_col: str) -> pd.DataFrame:
    """
    Load all parquet files from a directory and combine into one DataFrame.
    Filter to only rows that have the requested label column populated.
    """
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        log.error(f"FAIL | no parquet files in {data_dir}")
        return pd.DataFrame()

    all_dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        except Exception as e:
            log.warning(f"WARN load | {f.name}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Loaded {len(combined):,} rows from {len(files)} files ({data_dir.name})")

    # Ensure date column is string for filtering
    combined["date"] = combined["date"].astype(str)

    # Check label column exists
    if label_col not in combined.columns:
        log.error(f"FAIL | label column '{label_col}' not found")
        log.info(f"Available columns: {list(combined.columns[:20])}")
        return pd.DataFrame()

    return combined


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: train 2025, test 2026."""
    train = df[df["date"] <= TRAIN_END].copy()
    test  = df[df["date"] >= TEST_START].copy()
    log.info(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get all numeric feature columns, excluding labels and identifiers."""
    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if df[col].dtype in [object, str, "object"]:
            continue
        if df[col].dtype == bool:
            continue
        feature_cols.append(col)
    return sorted(feature_cols)


def prepare_xy(df: pd.DataFrame, feature_cols: list[str], label_col: str):
    """Prepare X and y for training/testing."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()

    # Fill NaN with -1 (means "missing/unknown")
    X = X.fillna(-1)

    # Replace inf with -1
    X = X.replace([np.inf, -np.inf], -1)

    y = df[label_col].astype(int)

    return X, y, available


# ─────────────────────────────────────────────
# TRAIN ONE MODEL
# ─────────────────────────────────────────────
def train_model(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    scale_pos_weight: float,
) -> tuple[object, list[str], dict]:
    """
    Train one XGBoost model and return (model, feature_cols, metrics).
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        log.error("FAIL | xgboost not installed — run: pip install xgboost")
        return None, [], {}

    log.info(f"\n{'='*60}")
    log.info(f"Training: {name}")
    log.info(f"{'='*60}")

    # Prepare data
    X_train, y_train, feat_cols = prepare_xy(train_df, feature_cols, label_col)
    X_test,  y_test,  _         = prepare_xy(test_df,  feature_cols, label_col)

    if len(X_train) == 0:
        log.error(f"FAIL {name} | no training data")
        return None, [], {}

    # Label stats
    n_pos_train = int(y_train.sum())
    n_neg_train = int((y_train == 0).sum())
    n_pos_test  = int(y_test.sum())
    n_neg_test  = int((y_test == 0).sum())

    log.info(f"Train: {n_pos_train} positive, {n_neg_train} negative ({n_pos_train/(len(y_train)+1e-6)*100:.1f}% positive)")
    log.info(f"Test:  {n_pos_test} positive, {n_neg_test} negative ({n_pos_test/(len(y_test)+1e-6)*100:.1f}% positive)")
    log.info(f"Features: {len(feat_cols)}")
    log.info(f"scale_pos_weight: {scale_pos_weight:.1f}")

    if n_pos_train < 10:
        log.warning(f"WARN {name} | only {n_pos_train} positive examples — model may be unreliable")

    # Train
    params = dict(XGB_PARAMS_BASE)
    params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)

    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100,  # print every 100 rounds so we can see learning
        )
        log.info(f"Best iteration: {model.best_iteration}")
    except Exception as e:
        log.error(f"FAIL {name} training: {e}")
        return None, [], {}

    # Evaluate
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score,
        f1_score, average_precision_score, confusion_matrix
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_05    = (y_pred_proba >= 0.50).astype(int)
    y_pred_07    = (y_pred_proba >= 0.70).astype(int)
    y_pred_08    = (y_pred_proba >= 0.80).astype(int)

    auc  = roc_auc_score(y_test, y_pred_proba) if n_pos_test > 0 else 0
    ap   = average_precision_score(y_test, y_pred_proba) if n_pos_test > 0 else 0

    def safe_metrics(y_true, y_pred, threshold):
        if y_pred.sum() == 0:
            return {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except Exception:
            tn, fp, fn, tp = 0, 0, 0, 0
        return {
            "threshold": threshold,
            "precision": round(float(prec), 3),
            "recall":    round(float(rec), 3),
            "f1":        round(float(f1), 3),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
            "alerts_per_day": round(float(y_pred.sum()) / max(1, len(test_df["date"].unique())), 2)
                if "date" in test_df.columns else 0,
        }

    metrics_05 = safe_metrics(y_test, y_pred_05, 0.50)
    metrics_07 = safe_metrics(y_test, y_pred_07, 0.70)
    metrics_08 = safe_metrics(y_test, y_pred_08, 0.80)

    log.info(f"\nTest Results for {name}:")
    log.info(f"  AUC-ROC:  {auc:.3f}")
    log.info(f"  Avg Prec: {ap:.3f}")
    log.info(f"\n  At threshold 0.50:")
    log.info(f"    Precision: {metrics_05['precision']:.3f} | Recall: {metrics_05['recall']:.3f} | F1: {metrics_05['f1']:.3f}")
    log.info(f"    TP={metrics_05['tp']} FP={metrics_05['fp']} FN={metrics_05['fn']} | {metrics_05['alerts_per_day']:.1f} alerts/day")
    log.info(f"\n  At threshold 0.70:")
    log.info(f"    Precision: {metrics_07['precision']:.3f} | Recall: {metrics_07['recall']:.3f} | F1: {metrics_07['f1']:.3f}")
    log.info(f"    TP={metrics_07['tp']} FP={metrics_07['fp']} FN={metrics_07['fn']} | {metrics_07['alerts_per_day']:.1f} alerts/day")
    log.info(f"\n  At threshold 0.80:")
    log.info(f"    Precision: {metrics_08['precision']:.3f} | Recall: {metrics_08['recall']:.3f} | F1: {metrics_08['f1']:.3f}")
    log.info(f"    TP={metrics_08['tp']} FP={metrics_08['fp']} FN={metrics_08['fn']} | {metrics_08['alerts_per_day']:.1f} alerts/day")

    # Feature importance — top 20
    importance = model.feature_importances_
    feat_importance = sorted(
        zip(feat_cols, importance),
        key=lambda x: x[1],
        reverse=True
    )

    log.info(f"\n  Top 20 features:")
    for i, (feat, imp) in enumerate(feat_importance[:20]):
        log.info(f"    {i+1:2d}. {feat:<35} {imp:.4f}")

    # Find optimal threshold — maximize precision while keeping recall >= 0.3
    thresholds = np.arange(0.40, 0.95, 0.05)
    best_threshold = 0.70
    best_f1 = 0
    threshold_sweep = []

    for t in thresholds:
        y_t = (y_pred_proba >= t).astype(int)
        if y_t.sum() == 0:
            continue
        prec = precision_score(y_test, y_t, zero_division=0)
        rec  = recall_score(y_test, y_t, zero_division=0)
        f1   = f1_score(y_test, y_t, zero_division=0)
        alerts = y_t.sum()
        threshold_sweep.append({
            "threshold": round(float(t), 2),
            "precision": round(float(prec), 3),
            "recall":    round(float(rec), 3),
            "f1":        round(float(f1), 3),
            "alerts":    int(alerts),
        })
        if f1 > best_f1 and rec >= 0.25:
            best_f1 = f1
            best_threshold = float(t)

    log.info(f"\n  Optimal threshold: {best_threshold:.2f} (F1={best_f1:.3f})")

    # Check if model is useful
    if auc < 0.55:
        log.warning(f"WARN {name} | AUC={auc:.3f} — barely better than random. Model may not be reliable.")
    elif auc < 0.65:
        log.warning(f"WARN {name} | AUC={auc:.3f} — moderate performance. Consider more data.")
    else:
        log.info(f"OK {name} | AUC={auc:.3f} — good performance")

    # Save model
    model_path = MODEL_DIR / f"{name}.json"
    model.save_model(str(model_path))
    log.info(f"Saved: {model_path}")

    metrics = {
        "name":            name,
        "auc":             round(float(auc), 3),
        "avg_precision":   round(float(ap), 3),
        "n_train_pos":     n_pos_train,
        "n_train_neg":     n_neg_train,
        "n_test_pos":      n_pos_test,
        "n_test_neg":      n_neg_test,
        "n_features":      len(feat_cols),
        "optimal_threshold": round(best_threshold, 2),
        "at_05":           metrics_05,
        "at_07":           metrics_07,
        "at_08":           metrics_08,
        "threshold_sweep": threshold_sweep,
        "top_20_features": [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in feat_importance[:20]
        ],
    }

    return model, feat_cols, metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    threading.Thread(target=start_keepalive, daemon=True).start()
    time.sleep(2)

    log.info("=" * 60)
    log.info("THE DELTA v2 — Trainer")
    log.info(f"Train: 2025-01-01 → {TRAIN_END}")
    log.info(f"Test:  {TEST_START} → 2026-05-23")
    log.info("=" * 60)

    # Check XGBoost available
    try:
        import xgboost
        from sklearn.metrics import roc_auc_score
        log.info(f"XGBoost version: {xgboost.__version__}")
    except ImportError as e:
        log.error(f"FAIL | missing library: {e}")
        log.error("Install with: pip install xgboost scikit-learn")
        sys.exit(1)

    # ─────────────────────────────────────────────
    # LOAD MIDNIGHT DATASET
    # ─────────────────────────────────────────────
    log.info("\nLoading midnight dataset...")
    midnight_df = load_dataset(MIDNIGHT_DIR, "label_seed")
    if midnight_df.empty:
        log.error("FAIL | midnight dataset empty")
        sys.exit(1)

    midnight_train, midnight_test = train_test_split(midnight_df)

    # ─────────────────────────────────────────────
    # LOAD MORNING DATASET
    # ─────────────────────────────────────────────
    log.info("\nLoading morning dataset...")
    morning_df = load_dataset(MORNING_DIR, "label_seed")
    if morning_df.empty:
        log.error("FAIL | morning dataset empty")
        sys.exit(1)

    morning_train, morning_test = train_test_split(morning_df)

    # ─────────────────────────────────────────────
    # GET FEATURE COLUMNS
    # ─────────────────────────────────────────────
    midnight_features = get_feature_cols(midnight_df)
    morning_features  = get_feature_cols(morning_df)

    log.info(f"\nMidnight features: {len(midnight_features)}")
    log.info(f"Morning features:  {len(morning_features)}")

    # ─────────────────────────────────────────────
    # CLASS WEIGHTS
    # ─────────────────────────────────────────────
    def calc_weight(df, label_col):
        n_pos = int(df[label_col].sum())
        n_neg = int((df[label_col] == 0).sum())
        return round(n_neg / max(n_pos, 1), 1)

    mid_seed_weight  = calc_weight(midnight_train, "label_seed")
    mid_super_weight = calc_weight(midnight_train, "label_super")
    mor_seed_weight  = calc_weight(morning_train, "label_seed")
    mor_super_weight = calc_weight(morning_train, "label_super")

    log.info(f"\nClass weights:")
    log.info(f"  midnight seed:  {mid_seed_weight}")
    log.info(f"  midnight super: {mid_super_weight}")
    log.info(f"  morning seed:   {mor_seed_weight}")
    log.info(f"  morning super:  {mor_super_weight}")

    # ─────────────────────────────────────────────
    # TRAIN 4 MODELS
    # ─────────────────────────────────────────────
    all_metrics = {}
    feature_cols_map = {}

    # 1. Midnight seed
    model_ms, cols_ms, metrics_ms = train_model(
        name="midnight_seed_model",
        train_df=midnight_train,
        test_df=midnight_test,
        feature_cols=midnight_features,
        label_col="label_seed",
        scale_pos_weight=mid_seed_weight,
    )
    if model_ms:
        all_metrics["midnight_seed"] = metrics_ms
        feature_cols_map["midnight_seed"] = cols_ms

    gc.collect()

    # 2. Midnight super
    n_super_train = int(midnight_train["label_super"].sum()) if "label_super" in midnight_train.columns else 0
    if n_super_train >= 10:
        model_msu, cols_msu, metrics_msu = train_model(
            name="midnight_super_model",
            train_df=midnight_train,
            test_df=midnight_test,
            feature_cols=midnight_features,
            label_col="label_super",
            scale_pos_weight=mid_super_weight,
        )
        if model_msu:
            all_metrics["midnight_super"] = metrics_msu
            feature_cols_map["midnight_super"] = cols_msu
    else:
        log.warning(f"SKIP midnight_super | only {n_super_train} super events in training — falling back to seed model")
        all_metrics["midnight_super"] = {"skipped": True, "reason": f"only {n_super_train} events"}

    gc.collect()

    # 3. Morning seed
    model_mos, cols_mos, metrics_mos = train_model(
        name="morning_seed_model",
        train_df=morning_train,
        test_df=morning_test,
        feature_cols=morning_features,
        label_col="label_seed",
        scale_pos_weight=mor_seed_weight,
    )
    if model_mos:
        all_metrics["morning_seed"] = metrics_mos
        feature_cols_map["morning_seed"] = cols_mos

    gc.collect()

    # 4. Morning super
    n_super_mor = int(morning_train["label_super"].sum()) if "label_super" in morning_train.columns else 0
    if n_super_mor >= 10:
        model_mosu, cols_mosu, metrics_mosu = train_model(
            name="morning_super_model",
            train_df=morning_train,
            test_df=morning_test,
            feature_cols=morning_features,
            label_col="label_super",
            scale_pos_weight=mor_super_weight,
        )
        if model_mosu:
            all_metrics["morning_super"] = metrics_mosu
            feature_cols_map["morning_super"] = cols_mosu
    else:
        log.warning(f"SKIP morning_super | only {n_super_mor} super events in training")
        all_metrics["morning_super"] = {"skipped": True, "reason": f"only {n_super_mor} events"}

    gc.collect()

    # ─────────────────────────────────────────────
    # SAVE FEATURE COLS + THRESHOLDS
    # ─────────────────────────────────────────────
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols_map, f, indent=2)
    log.info("Saved feature_cols.json")

    thresholds = {}
    for model_name, metrics in all_metrics.items():
        if isinstance(metrics, dict) and "optimal_threshold" in metrics:
            thresholds[model_name] = metrics["optimal_threshold"]
        else:
            thresholds[model_name] = 0.70  # default

    with open(MODEL_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info("Saved thresholds.json")

    with open(MODEL_DIR / "training_report.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    log.info("Saved training_report.json")

    # ─────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)
    log.info("\nModel Performance Summary:")

    for model_name, metrics in all_metrics.items():
        if metrics.get("skipped"):
            log.info(f"  {model_name}: SKIPPED — {metrics['reason']}")
            continue
        auc = metrics.get("auc", 0)
        opt = metrics.get("optimal_threshold", 0.70)
        at_opt = metrics.get("at_07", {})
        status = "✅ GOOD" if auc >= 0.65 else ("⚠️  WEAK" if auc >= 0.55 else "❌ POOR")
        log.info(f"\n  {model_name}: {status}")
        log.info(f"    AUC: {auc:.3f} | Optimal threshold: {opt:.2f}")
        log.info(f"    At threshold 0.70: P={at_opt.get('precision',0):.3f} R={at_opt.get('recall',0):.3f} F1={at_opt.get('f1',0):.3f}")
        log.info(f"    Alerts/day: {at_opt.get('alerts_per_day',0):.1f}")

    log.info("\n" + "=" * 60)
    log.info("Next steps:")
    log.info("  1. Review model performance above")
    log.info("  2. Copy models to scanner service:")
    log.info("     /app/data/models/ → scanner/models/")
    log.info("  3. Update main.py to use new models")
    log.info("=" * 60)

    # Sleep forever so Render doesn't restart it
    log.info("\nTrainer finished. Sleeping to prevent restart...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
