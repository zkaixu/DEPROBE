#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE: Traditional ML Baselines on 12D Physical Priors
========================================================
Trains multiple traditional ML models on the same 12D physics
features used by DEPROBE, without any sequence information or
deep learning. This establishes the marginal contribution of
the neural network architecture beyond handcrafted features.

Models tested:
    1. Linear Regression (simplest baseline)
    2. Ridge Regression (L2 regularized)
    3. Decision Tree
    4. Random Forest
    5. Gradient Boosting (XGBoost-like)

Usage:
    # Default: train master vs val master (between-replicate consistency)
    python baseline_traditional_ml.py

    # Real-probe validation (apples-to-apples with DEPROBE main number)
    python baseline_traditional_ml.py \\
        --val_h5 data/data_factory/final/probe_validation/deprobe_probe_val_master.h5 \\
        --output_suffix _real_probe
"""

import os
import sys
import argparse
import h5py
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_TRAIN_H5 = f"{PROJECT_ROOT}/data/data_factory/final/train/deprobe_train_master.h5"
DEFAULT_VAL_H5 = f"{PROJECT_ROOT}/data/data_factory/final/val/deprobe_val_master.h5"

# Canonical Paper-1 output destinations.
TABLES_DIR = f"{PROJECT_ROOT}/results/tables"
JSON_DIR = f"{PROJECT_ROOT}/results/json"

PRIOR_NAMES = [
    'Tm', 'GC_pct', 'dG', 'Hairpin_Tm', 'Dimer_Tm', 'Yield', 'Norm_Len',
    'GC_Skew', 'Entropy', 'dG_5p', 'dG_3p', 'Collision_Penalty'
]


def _suffix(name: str, suffix: str) -> str:
    """Insert suffix before file extension. Empty suffix → unchanged name."""
    if not suffix:
        return name
    base, ext = os.path.splitext(name)
    return f"{base}{suffix}{ext}"

# Performance is reported relative to a random-prediction baseline.
# All "% variance recovered above floor" reporting is dropped; we keep only
# MSE / R² (vs random baseline) / Spearman / Pearson / top-K precision.


def top_k_precision(y_true, y_pred, k_pct):
    n = len(y_true)
    k = max(1, int(n * k_pct / 100))
    pred_top = set(np.argsort(y_pred)[-k:])
    true_top = set(np.argsort(y_true)[-k:])
    return len(pred_top & true_top) / k


def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    """Train, predict, and compute all metrics for one model."""
    print(f"\n  Training {name}...", flush=True)
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_val)
    pred_time = time.time() - start - train_time

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    pearson_r, _ = stats.pearsonr(y_pred, y_val)
    spearman_rho, _ = stats.spearmanr(y_pred, y_val)
    top1 = top_k_precision(y_val, y_pred, 1)
    top5 = top_k_precision(y_val, y_pred, 5)
    top10 = top_k_precision(y_val, y_pred, 10)

    random_mse = np.var(y_val)
    r2_vs_random = 1.0 - mse / random_mse if random_mse > 0 else 0.0

    # Feature importance
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)

    result = {
        'name': name,
        'MSE': mse,
        'MAE': mae,
        'Pearson': pearson_r,
        'Spearman': spearman_rho,
        'Top1': top1,
        'Top5': top5,
        'Top10': top10,
        'R2_vs_random': r2_vs_random,
        'Train_time': train_time,
        'importances': importances,
    }

    print(f"  {name}: MSE={mse:.4f} | R²={r2_vs_random:.4f} | "
          f"Spearman={spearman_rho:.4f} | Top-10%={top10:.1%} | Time={train_time:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train traditional ML baselines on the 12D physics priors.")
    parser.add_argument("--train_h5", default=DEFAULT_TRAIN_H5,
                        help=f"Training H5 (default: {DEFAULT_TRAIN_H5})")
    parser.add_argument("--val_h5", default=DEFAULT_VAL_H5,
                        help=f"Validation H5 (default: {DEFAULT_VAL_H5}). "
                             "Pass `data/data_factory/final/probe_validation/"
                             "deprobe_probe_val_master.h5` for the real-probe "
                             "test set.")
    parser.add_argument("--output_suffix", default="",
                        help="Suffix appended to output file names before "
                             "extension. e.g. '_real_probe' → "
                             "baselines_traditional_ml_real_probe.csv. Empty "
                             "preserves the legacy filename.")
    parser.add_argument("--max_train", type=int, default=500000,
                        help="Subsample size for tree-based models (default: 500K).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling (default: 42).")
    args = parser.parse_args()

    print("=" * 70)
    print("  DEPROBE: TRADITIONAL ML BASELINE COMPARISON")
    print("=" * 70)
    print(f"  Train H5     : {args.train_h5}")
    print(f"  Val   H5     : {args.val_h5}")
    print(f"  Output suffix: '{args.output_suffix}'")
    print("=" * 70)

    # Load data
    print("\n  Loading data...")
    with h5py.File(args.train_h5, 'r') as f:
        X_train_full = f['priors'][:]
        y_train_full = f['efficiency'][:]

    with h5py.File(args.val_h5, 'r') as f:
        X_val = f['priors'][:]
        y_val = f['efficiency'][:]

    print(f"  Train: {X_train_full.shape[0]:,} x {X_train_full.shape[1]}D")
    print(f"  Val:   {X_val.shape[0]:,} x {X_val.shape[1]}D")

    # Subsample for tree-based models
    max_train = min(args.max_train, X_train_full.shape[0])
    if max_train < X_train_full.shape[0]:
        print(f"\n  Subsampling to {max_train:,} for tractability...")
        np.random.seed(args.seed)
        idx = np.random.choice(X_train_full.shape[0], max_train, replace=False)
        X_train = X_train_full[idx]
        y_train = y_train_full[idx]
    else:
        X_train, y_train = X_train_full, y_train_full

    # Standardize for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Define models
    models = [
        ("Linear Regression",
         LinearRegression(),
         X_train_scaled, X_val_scaled),

        ("Ridge Regression",
         Ridge(alpha=1.0),
         X_train_scaled, X_val_scaled),

        ("Decision Tree",
         DecisionTreeRegressor(max_depth=12, min_samples_leaf=100, random_state=42),
         X_train, X_val),

        ("Random Forest",
         RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=50,
                               n_jobs=-1, random_state=42, verbose=0),
         X_train, X_val),

        ("Gradient Boosting",
         GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, min_samples_leaf=50, random_state=42,
                                   verbose=0),
         X_train, X_val),
    ]

    # Run all models
    results = []
    for name, model, X_tr, X_va in models:
        r = evaluate_model(name, model, X_tr, y_train, X_va, y_val)
        results.append(r)

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 70}")

    header = f"  {'Model':<22s} {'MSE':>7s} {'R²':>7s} {'Spearman':>9s} {'Top-10%':>8s} {'Time':>7s}"
    print(header)
    print(f"  {'-' * 63}")

    for r in results:
        print(f"  {r['name']:<22s} {r['MSE']:>7.4f} {r['R2_vs_random']:>7.4f} "
              f"{r['Spearman']:>9.4f} {r['Top10']:>7.1%} {r['Train_time']:>6.1f}s")

    print(f"  {'-' * 63}")
    random_mse = np.var(y_val)
    print(f"  {'Random (predict mean)':<22s} {random_mse:>7.4f} {'0.000':>7s} "
          f"{'0.000':>9s} {'10.0%':>8s} {'--':>7s}")

    # Best traditional model. DEPROBE comparison numbers come from
    # results/json/main_metrics.json (run evaluate_model.py with the same
    # --val_h5). We deliberately don't hardcode DEPROBE numbers here so the
    # comparison stays consistent across val sets.
    best = max(results, key=lambda r: r['Spearman'])
    print(f"\n  Best traditional baseline: {best['name']} "
          f"(Spearman={best['Spearman']:.4f}, Top-10%={best['Top10']:.1%})")

    # Feature importance from best tree model
    best_tree = [r for r in results if r['importances'] is not None and 'Forest' in r['name']]
    if best_tree:
        imp = best_tree[0]['importances']
        sorted_idx = np.argsort(imp)[::-1]
        print(f"\n  Feature Importance (Random Forest):")
        print(f"  {'Rank':<6s} {'Feature':<20s} {'Importance':>12s}")
        print(f"  {'-' * 38}")
        for rank, i in enumerate(sorted_idx):
            print(f"  {rank+1:<6d} {PRIOR_NAMES[i]:<20s} {imp[i]:>12.4f}")

    # ================================================================
    # Paper Table 3. Traditional-ML baseline rows
    # CSV : results/tables/baselines_traditional_ml.csv
    # JSON: results/json/baselines_traditional_ml.json
    # (BiGRU and DEPROBE rows live in their own files; the paper Table 3
    #  is built by stacking these three CSVs.)
    # ================================================================
    import json as _json  # local import to avoid touching the original header
    import pandas as _pd

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

    table_rows = [{
        'Model': r['name'],
        'MSE': round(float(r['MSE']), 4),
        'MAE': round(float(r['MAE']), 4),
        'Pearson_r': round(float(r['Pearson']), 4),
        'Spearman_rho': round(float(r['Spearman']), 4),
        'R2_vs_random_baseline': round(float(r['R2_vs_random']), 4),
        'Top1_pct_precision': round(float(r['Top1']) * 100, 2),
        'Top5_pct_precision': round(float(r['Top5']) * 100, 2),
        'Top10_pct_precision': round(float(r['Top10']) * 100, 2),
        'Train_time_sec': round(float(r['Train_time']), 1),
    } for r in results]

    csv_path = os.path.join(TABLES_DIR,
                            _suffix('baselines_traditional_ml.csv', args.output_suffix))
    _pd.DataFrame(table_rows).to_csv(csv_path, index=False)

    json_path = os.path.join(JSON_DIR,
                             _suffix('baselines_traditional_ml.json', args.output_suffix))
    json_payload = {
        'train_h5': os.path.abspath(args.train_h5),
        'val_h5': os.path.abspath(args.val_h5),
        'val_n_samples': int(len(y_val)),
        'val_random_mse': float(random_mse),
        'rows': table_rows,
    }
    if best_tree:
        json_payload['random_forest_feature_importance'] = {
            PRIOR_NAMES[i]: float(best_tree[0]['importances'][i])
            for i in range(len(PRIOR_NAMES))
        }
    with open(json_path, 'w') as fh:
        _json.dump(json_payload, fh, indent=2)

    print(f"\n  Table CSV: {csv_path}")
    print(f"  JSON dump: {json_path}")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
