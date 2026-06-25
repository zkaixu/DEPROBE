#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: Position-Matched Label Correlation Diagnostic
==========================================================
Reads the dual-label master Parquet produced by compute_dual_labels.py and
quantifies the agreement between two capture kits' efficiency labels at
IDENTICAL genomic positions.

The narrative leverage: if Pearson r is low even at matched positions, the
cross-platform gap is not a covariate-shift problem (which DANN can fix),
it is a fundamental incompatibility between the two label functions
(which DANN, by construction, cannot bridge).

Outputs:
    results/json/matched_bed_label_correlation.json     # all stats
    results/tables/matched_bed_label_correlation.csv    # 1 row, paper-ready
    results/tables/matched_bed_label_per_chromosome.csv # per-chr breakdown
    results/plots/matched_bed_label_scatter.png         # 3-panel figure

Usage:
    python label_correlation_diagnostic.py \\
        --input data/data_factory/staging/matched_bed_dual/dual_labels.parquet \\
        --label_a_col Nextera_label \\
        --label_b_col TruSeq_label
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("label-corr")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables')
JSON_DIR = os.path.join(PROJECT_ROOT, 'results', 'json')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'plots')


def chromosome_from_probe_id(probe_id: str) -> str:
    """Probe IDs follow 'region_<chrom>_<start>_<end>_len<L>_pos<w>'.
    Anything between the first 'region_' and the next underscore-number tail is
    the chromosome string. We split conservatively on '_' to be robust to
    chromosome names that themselves contain digits.
    """
    if not isinstance(probe_id, str) or not probe_id.startswith('region_'):
        return 'unknown'
    parts = probe_id.split('_')
    # parts[0]='region', parts[1]=<chrom>, parts[2]=<start>, parts[3]=<end>, ...
    return parts[1] if len(parts) > 1 else 'unknown'


def main():
    parser = argparse.ArgumentParser(
        description="Position-matched label correlation diagnostic.")
    parser.add_argument("--input", required=True,
                        help="dual_labels.parquet from compute_dual_labels.py")
    parser.add_argument("--label_a_col", required=True,
                        help="Column name for kit A's label (e.g. Nextera_label).")
    parser.add_argument("--label_b_col", required=True,
                        help="Column name for kit B's label (e.g. TruSeq_label).")
    parser.add_argument("--scatter_subsample", type=int, default=50000,
                        help="Max points in the scatter (default: 50,000).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input not found: {args.input}")
        raise SystemExit(1)

    logger.info(f"Loading dual-label master: {args.input}")
    df = pd.read_parquet(args.input, columns=['Probe_ID', args.label_a_col, args.label_b_col])

    # Drop any NaN-bearing row (rare).
    df = df.dropna(subset=[args.label_a_col, args.label_b_col]).reset_index(drop=True)
    n = len(df)
    if n < 100:
        logger.error(f"Too few matched positions ({n}). Cannot compute meaningful correlation.")
        raise SystemExit(1)

    a = df[args.label_a_col].values.astype(np.float64)
    b = df[args.label_b_col].values.astype(np.float64)

    # ----------------------------------------------------------------
    # Global agreement statistics
    # ----------------------------------------------------------------
    pearson_r, pearson_p = stats.pearsonr(a, b)
    spearman_rho, spearman_p = stats.spearmanr(a, b)
    mse_ab = float(np.mean((a - b) ** 2))
    mae_ab = float(np.mean(np.abs(a - b)))
    r_squared = float(pearson_r ** 2)

    logger.info("=" * 60)
    logger.info(f"GLOBAL  N = {n:,}  ({args.label_a_col} vs {args.label_b_col})")
    logger.info(f"  Pearson r   = {pearson_r:.4f}  (p < {pearson_p:.2e})")
    logger.info(f"  Spearman ρ  = {spearman_rho:.4f}  (p < {spearman_p:.2e})")
    logger.info(f"  R²          = {r_squared:.4f}")
    logger.info(f"  MSE         = {mse_ab:.4f}")
    logger.info(f"  MAE         = {mae_ab:.4f}")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # Per-chromosome breakdown: does any chromosome buck the trend?
    # ----------------------------------------------------------------
    df['Chromosome'] = df['Probe_ID'].apply(chromosome_from_probe_id)
    per_chr_rows = []
    for chrom, sub in df.groupby('Chromosome', sort=False):
        if len(sub) < 50:
            continue
        a_c = sub[args.label_a_col].values
        b_c = sub[args.label_b_col].values
        rho, _ = stats.spearmanr(a_c, b_c)
        pr, _ = stats.pearsonr(a_c, b_c)
        per_chr_rows.append({
            'Chromosome': chrom,
            'N': int(len(sub)),
            'Pearson_r': round(float(pr), 4),
            'Spearman_rho': round(float(rho), 4),
        })

    # Stable sort: numeric chromosomes first, then alphabetical.
    def _chrom_sort_key(name: str):
        s = name.replace('chr', '')
        return (0, int(s)) if s.isdigit() else (1, s)
    per_chr_rows.sort(key=lambda r: _chrom_sort_key(r['Chromosome']))

    # ----------------------------------------------------------------
    # Defensive output dir creation
    # ----------------------------------------------------------------
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # CSV (single-row paper table)
    # ----------------------------------------------------------------
    csv_global = os.path.join(TABLES_DIR, 'matched_bed_label_correlation.csv')
    pd.DataFrame([{
        'Label_A': args.label_a_col,
        'Label_B': args.label_b_col,
        'N': int(n),
        'Pearson_r': round(float(pearson_r), 4),
        'Spearman_rho': round(float(spearman_rho), 4),
        'R2': round(r_squared, 4),
        'MSE_AB': round(mse_ab, 4),
        'MAE_AB': round(mae_ab, 4),
    }]).to_csv(csv_global, index=False)
    logger.info(f"CSV (global) : {csv_global}")

    csv_per_chr = os.path.join(TABLES_DIR, 'matched_bed_label_per_chromosome.csv')
    pd.DataFrame(per_chr_rows).to_csv(csv_per_chr, index=False)
    logger.info(f"CSV (per-chr): {csv_per_chr}")

    # ----------------------------------------------------------------
    # JSON dump
    # ----------------------------------------------------------------
    json_path = os.path.join(JSON_DIR, 'matched_bed_label_correlation.json')
    with open(json_path, 'w') as fh:
        json.dump({
            'input': os.path.abspath(args.input),
            'label_a': args.label_a_col,
            'label_b': args.label_b_col,
            'n': int(n),
            'global': {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_rho': float(spearman_rho),
                'spearman_p': float(spearman_p),
                'r_squared': r_squared,
                'mse_ab': mse_ab,
                'mae_ab': mae_ab,
            },
            'per_chromosome': per_chr_rows,
        }, fh, indent=2)
    logger.info(f"JSON         : {json_path}")

    # ----------------------------------------------------------------
    # Three-panel figure: scatter + residual hist + per-chr bar
    # ----------------------------------------------------------------
    # Okabe-Ito colourblind-safe palette, consistent across all paper figures.
    C_DATA = '#0072B2'   # blue (matched-position scatter, residual fill, mean line)
    C_REF  = '#444444'   # dark grey (identity, zero, global reference lines)
    C_BAR  = '#0072B2'   # blue (per-chromosome bars)

    rng = np.random.default_rng(42)
    if n > args.scatter_subsample:
        idx = rng.choice(n, size=args.scatter_subsample, replace=False)
        a_plot, b_plot = a[idx], b[idx]
    else:
        a_plot, b_plot = a, b

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: scatter at matched positions
    ax = axes[0]
    ax.scatter(a_plot, b_plot, s=1, alpha=0.1, c=C_DATA, rasterized=True)
    lims = [min(a.min(), b.min()), max(a.max(), b.max())]
    ax.plot(lims, lims, color=C_REF, linestyle='--', linewidth=1.2, label='Identity (y = x)')
    ax.set_xlabel(args.label_a_col.replace('_label', ''), fontsize=11)
    ax.set_ylabel(args.label_b_col.replace('_label', ''), fontsize=11)
    ax.set_title(f'Position-Matched Labels (N={n:,})', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal')

    # Panel B: residual histogram (label_b minus label_a)
    ax = axes[1]
    residuals = b - a
    ax.hist(residuals, bins=100, density=True, color=C_DATA, alpha=0.6, edgecolor='none')
    ax.axvline(0, color=C_REF, linestyle='--', linewidth=1.2, label='Zero residual')
    ax.axvline(float(np.mean(residuals)), color=C_DATA, linestyle='-', linewidth=1.0,
               label=f'Mean ({float(np.mean(residuals)):.3f})')
    ax.set_xlabel('Residual', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Between-Kit Residuals', fontsize=11)
    ax.legend(fontsize=9)

    # Panel C: per-chromosome Pearson bar (22 autosomes; chrX, chrY excluded)
    WITHIN_KIT_R = 0.601  # within-kit Pearson baseline, NIST7086 vs NIST7035 (Fig 4 / estimate_noise_floor.py)
    ax = axes[2]
    auto_rows = [r for r in per_chr_rows if r['Chromosome'].replace('chr', '').isdigit()]
    if auto_rows:
        labels = [r['Chromosome'] for r in auto_rows]
        prs    = [r['Pearson_r'] for r in auto_rows]
        ax.bar(range(len(labels)), prs, color=C_BAR, alpha=0.85, edgecolor='white', width=0.6)
        ax.axhline(WITHIN_KIT_R, color=C_REF, linestyle='--', linewidth=1.5,
                   label=f'Within-kit replicate r = {WITHIN_KIT_R}')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=70)
        ax.set_ylabel('Pearson r', fontsize=11)
        ax.set_title('Per-Chromosome Agreement', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0.0, max(0.7, max(prs) + 0.1))

    plt.tight_layout()
    fig_path = os.path.join(PLOTS_DIR, 'matched_bed_label_scatter.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figure       : {fig_path}")


if __name__ == "__main__":
    main()
