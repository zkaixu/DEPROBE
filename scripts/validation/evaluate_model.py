#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: Model Evaluation Suite
===================================
Evaluates trained model on held-out validation data with metrics
relevant to both academic publication and product deployment:

1. Regression metrics: MSE, MAE, Pearson r, Spearman rho
2. Ranking metrics: Top-K selection accuracy, NDCG
3. Comparison against noise floor and random baseline
4. Per-region analysis for panel design relevance

Usage:
    python evaluate_model.py --checkpoint <path_to_best_pth> --data <val_h5>

    Example:
    python evaluate_model.py \
        --checkpoint ../../models/phase1_pure_physics/deprobe_best_internal.pth \
        --data ../../data/data_factory/final/val/deprobe_val_master.h5
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import DataLoader

# Add model directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'model')
sys.path.insert(0, MODEL_DIR)

# Canonical Paper-1 output destinations.
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables')
JSON_DIR = os.path.join(PROJECT_ROOT, 'results', 'json')

from dataset import PanMolecularProbeDataset
from model import DEPROBE


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    model = DEPROBE(num_platforms=10, prior_dim=12, num_modalities=5, d_model=256).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    weights = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(weights, strict=False)
    model.eval()

    prior_mean = ckpt.get('prior_mean', None)
    prior_std = ckpt.get('prior_std', None)
    if prior_mean is not None:
        prior_mean = prior_mean.cpu()
    if prior_std is not None:
        prior_std = prior_std.cpu()

    return model, prior_mean, prior_std


def predict_all(model, dataloader, device):
    """Run inference on entire dataset."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['anchor'].to(device)
            priors = batch['priors'].to(device)
            mod = batch['modality'].to(device)
            mask = batch['anchor_mask'].to(device)
            eff = batch['efficiency']

            with torch.amp.autocast('cuda'):
                _, pred, _ = model(x, priors, mod, pad_mask=mask, alpha=0.0)

            all_preds.append(pred.squeeze().cpu().numpy())
            all_labels.append(eff.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def top_k_accuracy(y_true, y_pred, k_percent):
    """
    Top-K selection accuracy: among the top K% probes selected by the model,
    what fraction are truly in the top K% by ground truth?

    This is the metric that matters for panel design: if you select the
    top 10% of candidates, how many are actually good?
    """
    n = len(y_true)
    k = max(1, int(n * k_percent / 100))

    # Indices of top-K by prediction and by ground truth
    pred_top_k = set(np.argsort(y_pred)[-k:])
    true_top_k = set(np.argsort(y_true)[-k:])

    overlap = len(pred_top_k & true_top_k)
    precision = overlap / k
    return precision


def ndcg_at_k(y_true, y_pred, k_percent):
    """
    Normalized Discounted Cumulative Gain at K%.
    Measures ranking quality with position-weighted scoring.
    """
    n = len(y_true)
    k = max(1, int(n * k_percent / 100))

    # Sort by predicted score descending
    pred_order = np.argsort(y_pred)[::-1][:k]
    # Relevance = true efficiency (higher is more relevant)
    # Shift to non-negative for NDCG
    relevance = y_true - y_true.min()

    dcg = np.sum(relevance[pred_order] / np.log2(np.arange(2, k + 2)))

    # Ideal DCG: sort by true relevance
    ideal_order = np.argsort(y_true)[::-1][:k]
    idcg = np.sum(relevance[ideal_order] / np.log2(np.arange(2, k + 2)))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def main():
    parser = argparse.ArgumentParser(description="DEPROBE-DNA Model Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data", required=True, help="Path to evaluation H5 file")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for output plots (default: <project>/results/plots)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    # Defensive: results/plots is the canonical figure home; create it if missing.
    output_dir = args.output_dir or os.path.join(project_root, 'results', 'plots')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  DEPROBE-DNA: MODEL EVALUATION SUITE")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data:       {args.data}")
    print(f"  Device:     {device}")

    # Load model
    model, prior_mean, prior_std = load_model(args.checkpoint, device)

    # Load dataset with inherited statistics
    dataset = PanMolecularProbeDataset(
        h5_path=args.data,
        prior_mean=prior_mean,
        prior_std=prior_std
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True, persistent_workers=True)

    print(f"  Samples:    {len(dataset):,}")

    # Run inference
    print("\n  Running inference...")
    y_pred, y_true = predict_all(model, loader, device)
    n = len(y_true)

    # ================================================================
    # Regression Metrics
    # ================================================================
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    pearson_r, p_pearson = stats.pearsonr(y_pred, y_true)
    spearman_rho, p_spearman = stats.spearmanr(y_pred, y_true)
    r_squared = pearson_r ** 2

    print(f"\n{'=' * 70}")
    print(f"  REGRESSION METRICS")
    print(f"{'=' * 70}")
    print(f"    MSE:          {mse:.4f}")
    print(f"    MAE:          {mae:.4f}")
    print(f"    Pearson r:    {pearson_r:.4f}")
    print(f"    R-squared:    {r_squared:.4f}")
    print(f"    Spearman rho: {spearman_rho:.4f}")

    # ================================================================
    # Context: random baseline only.
    # ================================================================
    # Historical note: this script previously reported a "noise floor"
    # derived from MSE_between (between-replicate MSE / 2). After review
    # we found that:
    #   (a) Quantile normalisation of the labels breaks the i.i.d.
    #       additive-noise assumption that the formula relies on.
    #   (b) MSE_between is dominated by extreme-bin (|z|>3.12) rank-flip
    #       noise (per-bin MSE up to 9.62), while the model's predictions
    #       concentrate in the mid-range (97.5 % of probes have |z|<2.08).
    #   (c) The converged model achieves MSE 0.15-0.17, well below both
    #       MSE_between/2 = 0.4003 and MSE_between/4 = 0.2002, direct
    #       empirical refutation of either as a hard lower bound.
    # We now report only R² (vs random baseline) and Spearman ρ. The
    # between-replicate descriptive statistics live in
    # estimate_noise_floor.py / results/json/noise_floor.json as a
    # consistency diagnostic, not as a model floor.
    random_mse = np.var(y_true)  # predicting mean gives MSE = variance
    r2_vs_random = 1.0 - mse / random_mse if random_mse > 0 else 0.0

    print(f"\n{'=' * 70}")
    print(f"  CONTEXT")
    print(f"{'=' * 70}")
    print(f"    Random baseline MSE:           {random_mse:.4f}")
    print(f"    Model MSE:                     {mse:.4f}")
    print(f"    R² vs random baseline:         {r2_vs_random:.4f}")

    # ================================================================
    # Ranking Metrics (Product Relevance)
    # ================================================================
    k_values = [1, 5, 10, 20, 50]

    print(f"\n{'=' * 70}")
    print(f"  RANKING METRICS (Panel Design Relevance)")
    print(f"{'=' * 70}")
    print(f"    {'Top-K%':<10s} {'Precision':>12s} {'NDCG':>12s} {'Meaning':>30s}")
    print(f"    {'-' * 64}")

    precisions = []
    ndcgs = []
    for k in k_values:
        prec = top_k_accuracy(y_true, y_pred, k)
        ndcg = ndcg_at_k(y_true, y_pred, k)
        precisions.append(prec)
        ndcgs.append(ndcg)

        if k <= 5:
            meaning = "Best probes for critical targets"
        elif k <= 20:
            meaning = "Typical panel selection range"
        else:
            meaning = "Broad panel coverage"

        print(f"    Top {k:>2d}%    {prec:>11.1%}  {ndcg:>11.4f}  {meaning:>30s}")

    # Random baseline for comparison
    print(f"\n    Random baseline Top-K precision: {k_values[0]}% for all K")
    print(f"    (by definition, random selection has K% chance of overlap)")

    # ================================================================
    # Spearman rho interpretation
    # ================================================================
    # Technical replicate Spearman from noise floor analysis
    replicate_spearman = 0.5924  # from estimate_noise_floor.py

    spearman_fraction = spearman_rho / replicate_spearman * 100 if replicate_spearman > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  SPEARMAN RANKING ANALYSIS")
    print(f"{'=' * 70}")
    print(f"    Model Spearman rho:              {spearman_rho:.4f}")
    print(f"    Technical replicate Spearman:     {replicate_spearman:.4f}")
    print(f"    Ranking quality vs replicate:     {spearman_fraction:.1f}%")
    print(f"    (100% = model ranks as well as a biological replicate)")

    # ================================================================
    # Plots
    # ================================================================

    # Semantic color palette, consistent across ALL figures
    # Blue    = DEPROBE model / model predictions
    # Green   = Noise floor / technical replicate
    # Coral   = Random baseline / warning
    # Orange  = Reference lines (identity, zero)
    # Top-journal unified palette (consistent across all paper figures).
    # Blue family = DEPROBE model. Green family = replicate / noise floor.
    # Gray = random baseline (neutral). Tomato red = reference lines.
    C_MODEL = '#1F4E79'       # deep navy blue
    C_MODEL_LIGHT = '#5B9BD5' # medium blue for histogram fill
    C_FLOOR = '#1F7A48'       # deep forest green
    C_RANDOM = '#7F7F7F'      # medium gray (random baseline)
    C_REF = '#D62828'         # tomato red (reference lines)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Predicted vs Actual scatter, blue = model predictions
    ax = axes[0, 0]
    subsample = np.random.choice(n, min(50000, n), replace=False)
    ax.scatter(y_true[subsample], y_pred[subsample], s=1, alpha=0.1,
               c=C_MODEL, rasterized=True)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color=C_REF, linestyle='--', linewidth=1.2, label='Identity line (y = x)')
    ax.set_xlabel('True Efficiency (z-score)', fontsize=11)
    ax.set_ylabel('Predicted Efficiency (z-score)', fontsize=11)
    ax.set_title('Predicted vs True', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')

    # Panel B: Residual distribution, light blue fill + green noise floor ref
    ax = axes[0, 1]
    residuals = y_pred - y_true
    ax.hist(residuals, bins=100, density=True, color=C_MODEL_LIGHT,
            alpha=0.75, edgecolor='none')
    ax.axvline(0, color=C_REF, linestyle='--', linewidth=1.2, label='Zero residual')
    ax.axvline(np.mean(residuals), color=C_MODEL, linestyle='-', linewidth=1.0,
               label=f'Mean residual ({np.mean(residuals):.3f})')
    ax.set_xlabel('Residual (Predicted - True)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Prediction Residuals', fontsize=11)
    ax.legend(fontsize=9)

    # Panel C: Top-K Precision, dark blue bars + coral random baseline
    ax = axes[1, 0]
    ax.bar(range(len(k_values)), precisions, color=C_MODEL, alpha=0.85,
           edgecolor='white', label='DEPROBE precision')
    for i, k in enumerate(k_values):
        ax.plot([i - 0.4, i + 0.4], [k / 100, k / 100], color=C_RANDOM,
                linestyle='--', linewidth=1.5,
                label='Random baseline' if i == 0 else None)
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f'Top {k}%' for k in k_values])
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Top-K Selection Accuracy', fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

    # Panel D: MSE context, coral = random baseline, blue = current model
    ax = axes[1, 1]
    metrics = ['Random\nBaseline', 'Current\nModel']
    values = [random_mse, mse]
    bar_colors = [C_RANDOM, C_MODEL]
    bars = ax.bar(metrics, values, color=bar_colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('MSE Context', fontsize=11)
    ax.set_ylim(0, max(values) * 1.2)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_evaluation.png')
    # Defensive: re-ensure directory exists at write time.
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ================================================================
    # Headline DEPROBE metrics on the held-out set.
    # CSV : results/tables/main_metrics.csv  (one row)
    # JSON: results/json/main_metrics.json   (full dump)
    # ================================================================
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

    table1_row = {
        'N_samples': int(n),
        'MSE': round(float(mse), 4),
        'MAE': round(float(mae), 4),
        'Pearson_r': round(float(pearson_r), 4),
        'Spearman_rho': round(float(spearman_rho), 4),
        'R2': round(float(r_squared), 4),
        'Random_MSE': round(float(random_mse), 4),
        'R2_vs_random_baseline': round(float(r2_vs_random), 4),
        'Top1_pct_precision': round(float(precisions[0]) * 100, 2),
        'Top5_pct_precision': round(float(precisions[1]) * 100, 2),
        'Top10_pct_precision': round(float(precisions[2]) * 100, 2),
        'Top20_pct_precision': round(float(precisions[3]) * 100, 2),
        'Top50_pct_precision': round(float(precisions[4]) * 100, 2),
        'NDCG_at_10pct': round(float(ndcgs[2]), 4),
        'Replicate_Spearman': round(float(replicate_spearman), 4),
        'Ranking_vs_Replicate_pct': round(float(spearman_fraction), 2),
    }
    table1_csv = os.path.join(TABLES_DIR, 'main_metrics.csv')
    pd.DataFrame([table1_row]).to_csv(table1_csv, index=False)

    table1_json = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'data': os.path.abspath(args.data),
        'metrics': table1_row,
        'topk_breakdown': [
            {'K_percent': k, 'Precision': float(p), 'NDCG': float(d)}
            for k, p, d in zip(k_values, precisions, ndcgs)
        ],
    }
    table1_json_path = os.path.join(JSON_DIR, 'main_metrics.json')
    with open(table1_json_path, 'w') as fh:
        json.dump(table1_json, fh, indent=2)

    print(f"\n  CSV : {table1_csv}")
    print(f"  JSON: {table1_json_path}")

    # ================================================================
    # Summary for Paper
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  PAPER-READY SUMMARY")
    print(f"{'=' * 70}")
    print(f"    DEPROBE achieved a Spearman rank correlation of {spearman_rho:.3f}")
    print(f"    between predicted and observed capture efficiency on the held-out")
    print(f"    validation set, with MSE = {mse:.4f} (R² = {r2_vs_random:.3f} vs the")
    print(f"    random-prediction baseline of {random_mse:.4f}). For panel design,")
    print(f"    selecting the top 10% of candidates by model score yielded")
    print(f"    {precisions[2]:.0%} precision — a {precisions[2] / 0.10:.1f}-fold enrichment over random.")

    print(f"\n  Plot saved to: {plot_path}")
    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
