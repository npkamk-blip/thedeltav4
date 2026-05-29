"""
THE DELTA v2 — test_leakage.py
================================
Tests if the morning model is predictive or reactive by 
removing the top PM features and measuring AUC drop.

Three scenarios:
  A) Full model — all features including pm_move_pct, pm_volume
  B) Early PM only — remove reactive features, keep early signals
  C) Zero PM — midnight features only (AH, float, SI, EDGAR)

If AUC stays high in B and C — model is genuinely predictive.
If AUC collapses without pm_move_pct — model is reactive.
"""

import os, json, sys, time, threading
import pandas as pd
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

DATA_ROOT   = Path(os.environ.get("DATA_DIR", "/app/data"))
MORNING_DIR = DATA_ROOT / "training_data_v2" / "morning"
MODEL_DIR   = DATA_ROOT / "models"
LOG_DIR     = DATA_ROOT / "logs"

class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b"alive")
    def log_message(self, *a): pass

def start_keepalive():
    HTTPServer(("0.0.0.0", 8080), _Health).serve_forever()

XGB_PARAMS = {
    "n_estimators":          1000,
    "max_depth":             6,
    "learning_rate":         0.01,
    "subsample":             0.8,
    "colsample_bytree":      0.7,
    "colsample_bylevel":     0.7,
    "min_child_weight":      3,
    "gamma":                 0.05,
    "reg_alpha":             0.05,
    "reg_lambda":            0.5,
    "random_state":          42,
    "eval_metric":           "aucpr",
    "early_stopping_rounds": 50,
}

# Reactive features — things that only exist AFTER the move starts
REACTIVE_FEATURES = [
    "pm_move_pct",          # how much it already moved — REACTIVE
    "pm_high",              # premarket high — REACTIVE
    "pm_close",             # premarket close — REACTIVE
    "pm_high_of_session",   # at session high — REACTIVE
    "pm_remaining_to_seed", # how close to 2x — REACTIVE
    "pm_remaining_to_super",# how close to 3.5x — REACTIVE
    "pm_fade",              # did it fade — REACTIVE
]

# Early PM features — available in first 15-30 min at 4AM
# These are genuinely early signals
EARLY_PM_FEATURES = [
    "pm_open",              # opening print
    "pm_gap_pct",           # gap from prev close
    "pm_volume",            # early volume
    "pm_vol_ratio",         # vs historical
    "pm_active_bars",       # consistent bars
    "pm_volume_build",      # volume building
    "pm_consecutive_vol_bars", # consecutive bars
    "pm_volume_consistency",   # steady volume
    "pm_gap_held",          # gap holding
    "pm_gap_start_hour",    # when gap started
    "pm_vol_acceleration",  # accelerating
]

EXCLUDE_COLS = {
    "ticker", "date", "label_seed", "label_super",
    "hist_fetch_ok", "pm_fetch_ok", "ah_fetch_ok",
    "float_fetch_ok", "earnings_fetch_ok", "si_fetch_ok",
    "edgar_fetch_ok", "halt_fetch_ok", "sector_fetch_ok",
}

def load_data():
    files = sorted(MORNING_DIR.glob("*.parquet"))
    all_dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = df["date"].astype(str)
    train = df[df["date"] <= "2025-12-31"].copy()
    test  = df[df["date"] >= "2026-01-01"].copy()
    return train, test

def get_features(df, exclude_extra=None):
    exclude = set(EXCLUDE_COLS)
    if exclude_extra:
        exclude.update(exclude_extra)
    cols = []
    for c in df.columns:
        if c in exclude: continue
        if df[c].dtype == object: continue
        if df[c].dtype == bool: continue
        cols.append(c)
    return sorted(cols)

def train_and_eval(name, train_df, test_df, features, label="label_seed"):
    avail = [c for c in features if c in train_df.columns]
    X_train = train_df[avail].fillna(-1).replace([np.inf, -np.inf], -1)
    X_test  = test_df[avail].fillna(-1).replace([np.inf, -np.inf], -1)
    y_train = train_df[label].astype(int)
    y_test  = test_df[label].astype(int)

    n_pos = int(y_train.sum())
    n_neg = int((y_train==0).sum())
    spw   = round(n_neg / max(n_pos, 1), 1)

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = spw

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    scores = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, scores)
    ap     = average_precision_score(y_test, scores)

    y_55 = (scores >= 0.55).astype(int)
    y_70 = (scores >= 0.70).astype(int)

    p55 = precision_score(y_test, y_55, zero_division=0)
    r55 = recall_score(y_test, y_55, zero_division=0)
    p70 = precision_score(y_test, y_70, zero_division=0)
    r70 = recall_score(y_test, y_70, zero_division=0)

    # Gap bucket analysis
    test_copy = test_df.copy()
    test_copy["score"] = scores
    gap_results = {}
    if "pm_gap_pct" in test_copy.columns:
        seeds = test_copy[test_copy["label_seed"] == 1]
        for lbl, lo, hi in [
            ("flat<2%",   -0.99, 0.02),
            ("2-10%",      0.02, 0.10),
            ("10-25%",     0.10, 0.25),
            ("25%+",       0.25, 99.0),
        ]:
            b = seeds[(seeds["pm_gap_pct"] >= lo) & (seeds["pm_gap_pct"] < hi)]
            if len(b) > 0:
                gap_results[lbl] = {
                    "n": len(b),
                    "alerted_55": f"{(b['score'] >= 0.55).mean():.1%}",
                    "alerted_70": f"{(b['score'] >= 0.70).mean():.1%}",
                    "avg_score":  f"{b['score'].mean():.3f}",
                }

    return {
        "name":        name,
        "n_features":  len(avail),
        "n_train_pos": n_pos,
        "best_iter":   model.best_iteration,
        "auc":         round(auc, 3),
        "avg_prec":    round(ap, 3),
        "p@0.55":      round(p55, 3),
        "r@0.55":      round(r55, 3),
        "p@0.70":      round(p70, 3),
        "r@0.70":      round(r70, 3),
        "gap_buckets": gap_results,
        "top_features": sorted(
            zip(avail, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:15],
    }

def print_result(r):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {r['name']}")
    print(f"{'='*60}")
    print(f"Features: {r['n_features']} | Train positives: {r['n_train_pos']} | Best iter: {r['best_iter']}")
    print(f"AUC-ROC:  {r['auc']}  |  Avg Precision: {r['avg_prec']}")
    print(f"At 0.55:  P={r['p@0.55']}  R={r['r@0.55']}")
    print(f"At 0.70:  P={r['p@0.70']}  R={r['r@0.70']}")
    print(f"\nGap bucket alert rates:")
    print(f"  {'Gap':>10} {'N':>6} {'Score':>8} {'Alert@55':>10} {'Alert@70':>10}")
    for gap_lbl, gd in r["gap_buckets"].items():
        print(f"  {gap_lbl:>10} {gd['n']:>6} {gd['avg_score']:>8} {gd['alerted_55']:>10} {gd['alerted_70']:>10}")
    print(f"\nTop 15 features:")
    for i, (feat, imp) in enumerate(r["top_features"]):
        print(f"  {i+1:2d}. {feat:<40} {imp:.4f}")

def main():
    threading.Thread(target=start_keepalive, daemon=True).start()
    time.sleep(2)

    print("=" * 60)
    print("THE DELTA v2 — Leakage Test")
    print("Removing reactive features to test true predictive power")
    print("=" * 60)

    train_df, test_df = load_data()
    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    print(f"Train seeds: {int(train_df['label_seed'].sum())} | Test seeds: {int(test_df['label_seed'].sum())}")

    results = []

    # Scenario A — Full model
    print("\nScenario A: FULL MODEL (all features)")
    all_feats = get_features(train_df)
    r_a = train_and_eval("A_full_model", train_df, test_df, all_feats)
    print_result(r_a)
    results.append(r_a)

    # Scenario B — Remove reactive features
    print("\nScenario B: EARLY PM ONLY (remove reactive pm features)")
    early_feats = get_features(train_df, exclude_extra=REACTIVE_FEATURES)
    r_b = train_and_eval("B_early_pm_only", train_df, test_df, early_feats)
    print_result(r_b)
    results.append(r_b)

    # Scenario C — Zero PM features (midnight equivalent)
    print("\nScenario C: ZERO PM (midnight features only)")
    all_pm = [c for c in train_df.columns if c.startswith("pm_")]
    zero_feats = get_features(train_df, exclude_extra=all_pm)
    r_c = train_and_eval("C_zero_pm", train_df, test_df, zero_feats)
    print_result(r_c)
    results.append(r_c)

    # Comparison summary
    print("\n" + "=" * 60)
    print("LEAKAGE TEST SUMMARY")
    print("=" * 60)
    print(f"\n{'Scenario':<25} {'Features':>10} {'AUC':>8} {'AvgPrec':>10} {'P@0.55':>8} {'R@0.55':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['n_features']:>10} {r['auc']:>8} {r['avg_prec']:>10} {r['p@0.55']:>8} {r['r@0.55']:>8}")

    auc_a = results[0]["auc"]
    auc_b = results[1]["auc"]
    auc_c = results[2]["auc"]
    drop_ab = auc_a - auc_b
    drop_ac = auc_a - auc_c

    print(f"\nAUC drop removing reactive features: {drop_ab:.3f}")
    print(f"AUC drop removing ALL pm features:   {drop_ac:.3f}")

    print("\nVERDICT:")
    if drop_ab < 0.05:
        print("✅ VERY PREDICTIVE — reactive features add <5% AUC")
        print("   Model works from early PM signals alone")
    elif drop_ab < 0.10:
        print("✅ MOSTLY PREDICTIVE — reactive features add 5-10% AUC")
        print("   Model has real edge at early entry")
    elif drop_ab < 0.20:
        print("⚠️  MIXED — reactive features add 10-20% AUC")
        print("   Model has some edge but also some reactivity")
    else:
        print("❌ REACTIVE — reactive features add >20% AUC")
        print("   Model relies heavily on seeing the move first")

    if auc_c >= 0.75:
        print(f"\n✅ MIDNIGHT MODEL SOLID — AUC={auc_c} without any PM data")
        print("   Strong overnight signals alone")
    elif auc_c >= 0.65:
        print(f"\n⚠️  MIDNIGHT MODEL MODERATE — AUC={auc_c}")
    else:
        print(f"\n❌ MIDNIGHT MODEL WEAK — AUC={auc_c}")

    # Save results
    out = {r["name"]: {k: v for k, v in r.items() if k != "top_features"} for r in results}
    with open(LOG_DIR / "leakage_test.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved: /app/data/logs/leakage_test.json")

    print("\nLeakage test complete. Sleeping...")
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
