#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE: ROC / PR Curve Analysis
================================
Converts the DEPROBE regression output into a ranking-based binary
classification problem and reports AUROC + AUPRC across the standard
top-K probe-selection thresholds (top-10 %, top-20 %, top-50 %).

Why this matters for a regression model
---------------------------------------
DEPROBE predicts a *continuous* Capture_Efficiency. AUROC is a binary
classification metric, so it is only well-defined once we fix a notion
of "positive class". The natural choice is the model's actual
deployment scenario: **selecting the top-K % of candidate probes**.

For each top-K threshold we binarise the held-out labels into
{positive = top-K %, negative = rest} and use the continuous model
output as the ranking score. AUROC then answers:

    "What is the probability that a randomly chosen truly-top-K probe
     receives a higher model score than a randomly chosen non-top-K
     probe?"

AUPRC is reported alongside because top-10 % is a 1:9 imbalance. AUROC
is somewhat optimistic in that regime, while AUPRC measures
precision-recall trade-off directly and is the harsher,
deployment-relevant metric.

Outputs
-------
    results/plots/roc_curves.png      # 2 rows (ROC, PR) × N split cols
    results/tables/roc_metrics.csv    # flat (split, topK, AUROC, AUPRC, lift)
    results/json/roc_metrics.json     # full curve points for re-plotting

Usage
-----
    # Default: Phase 1 12D model on Int/Ext/Probe val.
    python scripts/validation/roc_analysis.py \\
        --checkpoint models/phase1_pure_physics/deprobe_best_internal.pth

    # Custom split list (skip probe-val), explicit top-K grid:
    python scripts/validation/roc_analysis.py \\
        --checkpoint models/phase1_pure_physics/deprobe_best_internal.pth \\
        --int_data data/data_factory/final/train/deprobe_train_master.h5 \\
        --ext_data data/data_factory/final/val/deprobe_val_master.h5 \\
        --probe_data "" \\
        --topk 0.05 0.10 0.20 0.50
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------
# Paths and imports. Match evaluate_model.py conventions exactly.
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'model')
sys.path.insert(0, MODEL_DIR)

PLOTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'plots')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables')
JSON_DIR = os.path.join(PROJECT_ROOT, 'results', 'json')

from dataset import PanMolecularProbeDataset  # noqa: E402
from model import DEPROBE  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("roc")


# ----------------------------------------------------------------------
# Model loading & inference (mirrors evaluate_model.py).
# ----------------------------------------------------------------------
def load_model(checkpoint_path: str, prior_dim: int, device: torch.device):
    """Load DEPROBE checkpoint and inherit prior normalisation stats."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint missing: {checkpoint_path}")
    model = DEPROBE(
        num_platforms=10, prior_dim=prior_dim,
        num_modalities=5, d_model=256,
    ).to(device)
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


@torch.no_grad()
def predict_split(model: DEPROBE, h5_path: str, prior_mean, prior_std,
                  device: torch.device, batch_size: int = 2048,
                  num_workers: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on one H5 split. Returns (y_true, y_pred)."""
    dataset = PanMolecularProbeDataset(
        h5_path=h5_path, prior_mean=prior_mean, prior_std=prior_std,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    preds, labels = [], []
    for batch in loader:
        x = batch['anchor'].to(device, non_blocking=True)
        priors = batch['priors'].to(device, non_blocking=True)
        mod = batch['modality'].to(device, non_blocking=True)
        mask = batch['anchor_mask'].to(device, non_blocking=True)
        eff = batch['efficiency']
        with torch.amp.autocast('cuda'):
            _, pred, _ = model(x, priors, mod, pad_mask=mask, alpha=0.0)
        preds.append(pred.squeeze().float().cpu().numpy())
        labels.append(eff.numpy())
    return np.concatenate(labels), np.concatenate(preds)


# ----------------------------------------------------------------------
# Metric core: binarise at top-K, compute ROC + PR curves.
# ----------------------------------------------------------------------
def topk_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 topk_frac: float) -> Dict:
    """Binarise at top-K threshold and compute AUROC + AUPRC + curve points.

    Returns dict with scalars (auroc, auprc, baseline_auprc, lift, threshold,
    n_pos, n_neg) and full curve arrays (fpr, tpr, precision, recall).
    """
    if not 0.0 < topk_frac < 1.0:
        raise ValueError(f"topk_frac must be in (0, 1), got {topk_frac}")
    threshold = float(np.quantile(y_true, 1.0 - topk_frac))
    y_bin = (y_true >= threshold).astype(int)
    n_pos = int(y_bin.sum())
    n_neg = int(len(y_bin) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {
            'topk_frac': topk_frac, 'threshold': threshold,
            'n_pos': n_pos, 'n_neg': n_neg,
            'auroc': float('nan'), 'auprc': float('nan'),
            'baseline_auprc': float('nan'), 'lift': float('nan'),
            'fpr': [], 'tpr': [], 'precision': [], 'recall': [],
        }
    fpr, tpr, _ = roc_curve(y_bin, y_pred)
    prec, rec, _ = precision_recall_curve(y_bin, y_pred)
    auroc = float(roc_auc_score(y_bin, y_pred))
    auprc = float(average_precision_score(y_bin, y_pred))
    baseline = n_pos / (n_pos + n_neg)
    lift = auprc / baseline if baseline > 0 else float('nan')
    return {
        'topk_frac': float(topk_frac), 'threshold': threshold,
        'n_pos': n_pos, 'n_neg': n_neg,
        'auroc': auroc, 'auprc': auprc,
        'baseline_auprc': float(baseline), 'lift': float(lift),
        # Subsample curve points to keep JSON small (~500 points each is plenty)
        'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
        'precision': prec.tolist(), 'recall': rec.tolist(),
    }


def _downsample_curve(x: List[float], y: List[float], n_max: int = 500
                      ) -> Tuple[List[float], List[float]]:
    """Even-stride downsample for compact JSON / smooth plotting."""
    if len(x) <= n_max:
        return list(x), list(y)
    idx = np.linspace(0, len(x) - 1, n_max).astype(int)
    return [x[i] for i in idx], [y[i] for i in idx]


# ----------------------------------------------------------------------
# Plot: 2 rows (ROC, PR) × N split cols.
# ----------------------------------------------------------------------
def render_plot(per_split_results: Dict[str, Dict[float, Dict]],
                split_order: List[str], topk_grid: List[float],
                out_path: str):
    """Two-row figure: top = ROC curves, bottom = PR curves.
    Each column is a split; each panel overlays one curve per top-K.
    """
    n_splits = len(split_order)
    fig, axes = plt.subplots(
        2, n_splits, figsize=(5.2 * n_splits, 9),
        squeeze=False,
    )
    # Color per top-K threshold (deeper colour = stricter threshold)
    palette = ['#08306B', '#2171B5', '#6BAED6']  # ColorBrewer sequential blue (dark→light)
    if len(topk_grid) > 3:
        # Fall back to viridis for >3 thresholds
        cmap = plt.get_cmap('viridis')
        palette = [cmap(i / max(1, len(topk_grid) - 1)) for i in range(len(topk_grid))]

    for col, split_name in enumerate(split_order):
        ax_roc = axes[0, col]
        ax_pr = axes[1, col]
        topk_results = per_split_results[split_name]

        for i, k in enumerate(topk_grid):
            r = topk_results.get(k)
            if r is None or not r['fpr']:
                continue
            fpr, tpr = _downsample_curve(r['fpr'], r['tpr'])
            prec, rec = _downsample_curve(r['precision'], r['recall'])

            ax_roc.plot(fpr, tpr, color=palette[i], linewidth=1.8,
                        label=f"top-{int(k * 100)}%  AUROC={r['auroc']:.3f}")
            ax_pr.plot(rec, prec, color=palette[i], linewidth=1.8,
                       label=f"top-{int(k * 100)}%  AUPRC={r['auprc']:.3f} "
                             f"(lift {r['lift']:.2f}×)")
            # Random PR baseline for this top-K
            ax_pr.axhline(r['baseline_auprc'], color=palette[i],
                          linestyle=':', linewidth=0.8, alpha=0.55)

        # ROC reference: diagonal = random
        ax_roc.plot([0, 1], [0, 1], color='#888888', linestyle='--',
                    linewidth=1.0, alpha=0.7, label='Random')
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1.005)
        ax_roc.set_xlabel('False Positive Rate', fontsize=11)
        ax_roc.set_ylabel('True Positive Rate', fontsize=11)
        ax_roc.grid(alpha=0.3)
        ax_roc.legend(loc='lower right', fontsize=8)

        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1.005)
        ax_pr.set_xlabel('Recall', fontsize=11)
        ax_pr.set_ylabel('Precision', fontsize=11)
        ax_pr.grid(alpha=0.3)
        ax_pr.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"figure: {out_path}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="ROC / PR curve analysis for DEPROBE regression output.")
    p.add_argument('--checkpoint', required=True,
                   help="Trained DEPROBE checkpoint (.pth).")
    p.add_argument('--int_data', default=os.path.join(
        PROJECT_ROOT, 'data', 'data_factory', 'final', 'train',
        'deprobe_train_master.h5'),
                   help="Internal validation H5.")
    p.add_argument('--ext_data', default=os.path.join(
        PROJECT_ROOT, 'data', 'data_factory', 'final', 'val',
        'deprobe_val_master.h5'),
                   help="External validation H5.")
    p.add_argument('--probe_data', default=os.path.join(
        PROJECT_ROOT, 'data', 'data_factory', 'final', 'probe_validation',
        'deprobe_probe_val_master.h5'),
                   help='Real-probe validation H5. Pass empty string "" to skip.')
    p.add_argument('--topk', type=float, nargs='+', default=[0.10, 0.20, 0.50],
                   help='Top-K fractions for binarisation (default: 0.10 0.20 0.50).')
    p.add_argument('--prior_dim', type=int, default=12,
                   help='Prior dimension.')
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--out_png', default=os.path.join(PLOTS_DIR, 'roc_curves.png'))
    p.add_argument('--out_csv', default=os.path.join(TABLES_DIR, 'roc_metrics.csv'))
    p.add_argument('--out_json', default=os.path.join(JSON_DIR, 'roc_metrics.json'))
    args = p.parse_args()

    # Validate top-K
    args.topk = sorted(set(round(float(k), 4) for k in args.topk))
    for k in args.topk:
        if not 0.0 < k < 1.0:
            raise ValueError(f"--topk values must be in (0, 1); got {k}")

    # Build the split list, skipping any with empty/missing path.
    candidate_splits = [
        ('Internal Val',  args.int_data),
        ('External Val',  args.ext_data),
        ('Real-Probe Val', args.probe_data),
    ]
    splits: List[Tuple[str, str]] = []
    for name, path in candidate_splits:
        if not path:
            logger.info(f"skipping {name} (path empty).")
            continue
        if not os.path.exists(path):
            logger.warning(f"skipping {name} — file not found: {path}")
            continue
        splits.append((name, path))
    if not splits:
        logger.error("no valid splits — aborting.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("=" * 70)
    logger.info(f"  Checkpoint : {args.checkpoint}")
    logger.info(f"  Prior dim  : {args.prior_dim}")
    logger.info(f"  Top-K grid : {args.topk}")
    logger.info(f"  Splits     : {[s[0] for s in splits]}")
    logger.info(f"  Device     : {device}")
    logger.info("=" * 70)

    model, prior_mean, prior_std = load_model(
        args.checkpoint, args.prior_dim, device,
    )

    # Per-split inference + metric loop.
    per_split_results: Dict[str, Dict[float, Dict]] = {}
    flat_rows: List[Dict] = []
    for name, h5_path in splits:
        logger.info(f"[{name}] running inference: {h5_path}")
        y_true, y_pred = predict_split(
            model, h5_path, prior_mean, prior_std, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        logger.info(f"  n={len(y_true):,}  "
                    f"y_true [{y_true.min():.3f}, {y_true.max():.3f}] mean={y_true.mean():.3f}  "
                    f"y_pred [{y_pred.min():.3f}, {y_pred.max():.3f}] mean={y_pred.mean():.3f}")

        per_split_results[name] = {}
        for k in args.topk:
            r = topk_metrics(y_true, y_pred, k)
            per_split_results[name][k] = r
            flat_rows.append({
                'Split': name,
                'TopK_Pct': round(k * 100, 2),
                'N_Total': r['n_pos'] + r['n_neg'],
                'N_Positive': r['n_pos'],
                'Threshold_y_true': round(r['threshold'], 4),
                'AUROC': round(r['auroc'], 4),
                'AUPRC': round(r['auprc'], 4),
                'Baseline_AUPRC': round(r['baseline_auprc'], 4),
                'AUPRC_Lift': round(r['lift'], 3),
            })
            logger.info(
                f"  top-{int(k * 100):>3d}%  AUROC={r['auroc']:.4f}  "
                f"AUPRC={r['auprc']:.4f}  baseline={r['baseline_auprc']:.3f}  "
                f"lift={r['lift']:.2f}x"
            )

    # ------------------------------------------------------------------
    # Persist outputs (defensive os.makedirs everywhere).
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_png) or '.', exist_ok=True)

    pd.DataFrame(flat_rows).to_csv(args.out_csv, index=False)
    logger.info(f"CSV : {args.out_csv}")

    # JSON: full curve data, downsampled for size.
    json_payload = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'prior_dim': int(args.prior_dim),
        'topk_grid': args.topk,
        'splits': {
            name: {
                f"top_{int(k * 100)}pct": {
                    'auroc': r['auroc'],
                    'auprc': r['auprc'],
                    'baseline_auprc': r['baseline_auprc'],
                    'lift': r['lift'],
                    'threshold_y_true': r['threshold'],
                    'n_total': r['n_pos'] + r['n_neg'],
                    'n_positive': r['n_pos'],
                    'roc_curve': dict(zip(
                        ['fpr', 'tpr'],
                        _downsample_curve(r['fpr'], r['tpr']),
                    )),
                    'pr_curve': dict(zip(
                        ['recall', 'precision'],
                        _downsample_curve(r['recall'], r['precision']),
                    )),
                }
                for k, r in per_split_results[name].items()
            }
            for name in per_split_results
        },
    }
    with open(args.out_json, 'w') as fh:
        json.dump(json_payload, fh, indent=2)
    logger.info(f"JSON: {args.out_json}")

    render_plot(per_split_results, [s[0] for s in splits], args.topk, args.out_png)

    # ------------------------------------------------------------------
    # Paper-ready summary line.
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("  PAPER-READY SUMMARY")
    logger.info("=" * 70)
    for name in per_split_results:
        line_parts = []
        for k in args.topk:
            r = per_split_results[name][k]
            line_parts.append(
                f"top-{int(k * 100)}% AUROC={r['auroc']:.3f}/AUPRC={r['auprc']:.3f}"
            )
        logger.info(f"  {name:<16s} | " + "  ".join(line_parts))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
