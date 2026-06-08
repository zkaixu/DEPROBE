#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: In-Silico Probe Validation Suite
=============================================
6 analysis modules:

1. Global Metrics: Spearman, Top-K, NDCG (vs sliding-window baseline)
2. Per-Region Best-Probe ID: can the model pick the best probe per target?
3. Simulated Panel Redesign: would model-guided selection improve the panel?
4. Per-Chromosome Spearman: consistency across the genome
5. Error Analysis: failure modes by GC%, Entropy, Tm
6. Summary Report: paper-ready numbers + multi-panel figure

Usage:
    python probe_validation_suite.py \
        --checkpoint <phase1_best.pth> \
        --h5 <probe_val.h5> \
        --staging_csv <probe_val_master.csv> \
        --probe_mapping <nextera_probe_mapping.csv> \
        --target_bed <expandedexome_targetedregions.bed> \
        --output_dir <results/plots>   # default if omitted
"""

import os
import sys
import json
import argparse
import bisect
import numpy as np
import pandas as pd
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from torch.utils.data import DataLoader

# Canonical output destinations.
SCRIPT_DIR_BOOTSTRAP = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR_BOOTSTRAP, '..', '..'))
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables')
JSON_DIR = os.path.join(PROJECT_ROOT, 'results', 'json')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'model')
sys.path.insert(0, MODEL_DIR)

from dataset import PanMolecularProbeDataset
from model import DEPROBE

# Semantic color palette (consistent across figures).
C_MODEL = '#4878D0'
C_MODEL_LIGHT = '#7EB0E0'
C_FLOOR = '#6ACC64'
C_RANDOM = '#D65F5F'
C_REF = '#EE854A'
C_ACCENT = '#956CB4'

# Sliding-window reference numbers (for comparison table).
SW_SPEARMAN = 0.735
SW_MSE = 0.414
SW_TOP10 = 0.601
SW_NDCG10 = 0.949


# ======================================================================
# Utilities
# ======================================================================

def load_model(checkpoint_path, device, prior_dim=12):
    model = DEPROBE(num_platforms=10, prior_dim=prior_dim, num_modalities=5, d_model=256).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    weights = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(weights, strict=False)
    model.eval()
    prior_mean = ckpt.get('prior_mean', None)
    prior_std = ckpt.get('prior_std', None)
    if prior_mean is not None: prior_mean = prior_mean.cpu()
    if prior_std is not None: prior_std = prior_std.cpu()
    return model, prior_mean, prior_std


def predict_all(model, dataloader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['anchor'].to(device)
            priors = batch['priors'].to(device)
            mod = batch['modality'].to(device)
            mask = batch['anchor_mask'].to(device)
            with torch.amp.autocast('cuda'):
                _, pred, _ = model(x, priors, mod, pad_mask=mask, alpha=0.0)
            all_preds.append(pred.squeeze().cpu().numpy())
            all_labels.append(batch['efficiency'].numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def top_k_accuracy(y_true, y_pred, k_percent):
    n = len(y_true)
    k = max(1, int(n * k_percent / 100))
    pred_top_k = set(np.argsort(y_pred)[-k:])
    true_top_k = set(np.argsort(y_true)[-k:])
    return len(pred_top_k & true_top_k) / k


def ndcg_at_k(y_true, y_pred, k_percent):
    n = len(y_true)
    k = max(1, int(n * k_percent / 100))
    pred_order = np.argsort(y_pred)[::-1][:k]
    relevance = y_true - y_true.min()
    dcg = np.sum(relevance[pred_order] / np.log2(np.arange(2, k + 2)))
    ideal_order = np.argsort(y_true)[::-1][:k]
    idcg = np.sum(relevance[ideal_order] / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def load_target_regions(bed_path):
    """Load target regions BED, return dict: chrom → sorted list of (start, end, idx)."""
    regions = defaultdict(list)
    idx = 0
    with open(bed_path) as f:
        for line in f:
            cols = line.strip().split('\t')
            chrom, start, end = cols[0], int(cols[1]), int(cols[2])
            regions[chrom].append((start, end, idx))
            idx += 1
    for chrom in regions:
        regions[chrom].sort()
    return regions, idx


def assign_probes_to_regions(probe_df, target_regions):
    """Assign each probe to its overlapping target region using coordinate overlap."""
    assignments = []
    for _, row in probe_df.iterrows():
        chrom = row['Chromosome']
        mid = (row['Probe_Start'] + row['Probe_End']) // 2
        region_idx = -1
        if chrom in target_regions:
            intervals = target_regions[chrom]
            starts = [iv[0] for iv in intervals]
            i = bisect.bisect_right(starts, mid) - 1
            if i >= 0 and intervals[i][0] <= mid <= intervals[i][1]:
                region_idx = intervals[i][2]
        assignments.append(region_idx)
    return assignments


def link_h5_to_probes(staging_csv, probe_mapping_csv, prior_cols):
    """
    Link H5 rows to original probe names via staging CSV and probe mapping.
    Returns DataFrame with columns: Probe_Name, Chromosome, Probe_Start, Probe_End,
    Centered_Start, Centered_End, GC_pct, Entropy, Tm + Capture_Efficiency.
    Rows are aligned with H5 (after filtering NaN/invalid same as build_h5.py).
    """
    # Read staging CSV
    df_stage = pd.read_csv(staging_csv)

    # Filter same as build_h5.py: drop rows with NaN in priors or efficiency
    keep_cols = prior_cols + ['Capture_Efficiency']
    valid_mask = df_stage[keep_cols].notna().all(axis=1)
    df_stage = df_stage[valid_mask].reset_index(drop=True)

    # Parse coordinates from Probe_ID: region_{chr}_{start}_{end}_len120_pos{start}
    def parse_probe_id(pid):
        parts = pid.split('_')
        # region_chr1_14301_14421_len120_pos14301
        chrom = parts[1]  # chr1
        c_start = int(parts[2])
        c_end = int(parts[3])
        return chrom, c_start, c_end

    coords = df_stage['Probe_ID'].apply(parse_probe_id)
    df_stage['_chrom'] = [c[0] for c in coords]
    df_stage['_c_start'] = [c[1] for c in coords]
    df_stage['_c_end'] = [c[2] for c in coords]

    # Build centered BED → probe name lookup from mapping CSV
    df_map = pd.read_csv(probe_mapping_csv)
    map_lookup = {}
    for _, row in df_map.iterrows():
        key = (row['Chromosome'], int(row['Centered_Start']), int(row['Centered_End']))
        map_lookup[key] = (row['Probe_Name'], int(row['Probe_Start']),
                           int(row['Probe_End']), int(row['Probe_Length']))

    # Match
    probe_names = []
    probe_starts = []
    probe_ends = []
    matched = 0
    for _, row in df_stage.iterrows():
        key = (row['_chrom'], row['_c_start'], row['_c_end'])
        if key in map_lookup:
            pn, ps, pe, pl = map_lookup[key]
            probe_names.append(pn)
            probe_starts.append(ps)
            probe_ends.append(pe)
            matched += 1
        else:
            probe_names.append(None)
            probe_starts.append(0)
            probe_ends.append(0)

    print(f"  Probe name matching: {matched}/{len(df_stage)} ({100*matched/len(df_stage):.1f}%)")

    result = pd.DataFrame({
        'Probe_Name': probe_names,
        'Chromosome': df_stage['_chrom'].values,
        'Probe_Start': probe_starts,
        'Probe_End': probe_ends,
        'GC_pct': df_stage['GC_pct'].values,
        'Entropy': df_stage['Entropy'].values,
        'Tm': df_stage['Tm'].values,
        'Capture_Efficiency': df_stage['Capture_Efficiency'].values,
    })
    return result


# ======================================================================
# Analysis Modules
# ======================================================================

def module_1_global_metrics(y_pred, y_true):
    """Global regression and ranking metrics with sliding-window comparison."""
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    pearson_r, _ = stats.pearsonr(y_pred, y_true)
    spearman_rho, _ = stats.spearmanr(y_pred, y_true)

    k_values = [1, 5, 10, 20]
    precs = {k: top_k_accuracy(y_true, y_pred, k) for k in k_values}
    ndcgs = {k: ndcg_at_k(y_true, y_pred, k) for k in k_values}

    print(f"\n{'='*70}")
    print(f"  MODULE 1: GLOBAL METRICS")
    print(f"{'='*70}")
    print(f"  {'Metric':<25s} {'Real Probes':>12s} {'Sliding Win':>12s}")
    print(f"  {'-'*49}")
    print(f"  {'Spearman rho':<25s} {spearman_rho:>12.4f} {SW_SPEARMAN:>12.4f}")
    print(f"  {'MSE':<25s} {mse:>12.4f} {SW_MSE:>12.4f}")
    print(f"  {'Top-10% precision':<25s} {precs[10]:>11.1%} {SW_TOP10:>11.1%}")
    print(f"  {'NDCG@10%':<25s} {ndcgs[10]:>12.4f} {SW_NDCG10:>12.4f}")
    print(f"  {'Pearson r':<25s} {pearson_r:>12.4f} {'—':>12s}")
    print(f"  {'MAE':<25s} {mae:>12.4f} {'—':>12s}")

    for k in k_values:
        enrich = precs[k] / (k / 100)
        print(f"  Top-{k}% precision: {precs[k]:.1%} ({enrich:.1f}x enrichment)")

    return {'spearman': spearman_rho, 'mse': mse, 'pearson': pearson_r,
            'precs': precs, 'ndcgs': ndcgs, 'y_pred': y_pred, 'y_true': y_true}


def module_2_per_region(y_pred, y_true, probe_df, target_regions, min_probes=3):
    """Per-region best-probe identification analysis."""
    region_assignments = assign_probes_to_regions(probe_df, target_regions)
    probe_df = probe_df.copy()
    probe_df['region_idx'] = region_assignments
    probe_df['y_pred'] = y_pred
    probe_df['y_true'] = y_true

    # Filter to assigned probes and regions with >= min_probes
    assigned = probe_df[probe_df['region_idx'] >= 0]
    region_counts = assigned['region_idx'].value_counts()
    valid_regions = region_counts[region_counts >= min_probes].index
    assigned = assigned[assigned['region_idx'].isin(valid_regions)]

    n_regions = len(valid_regions)
    print(f"\n{'='*70}")
    print(f"  MODULE 2: PER-REGION BEST-PROBE IDENTIFICATION")
    print(f"{'='*70}")
    print(f"  Regions with >= {min_probes} probes: {n_regions}")
    print(f"  Total probes in these regions: {len(assigned)}")
    print(f"  Mean probes per region: {len(assigned)/n_regions:.1f}")

    hit_at_1 = 0
    hit_at_3 = 0
    rank_percentiles = []
    region_sizes = []
    random_hit1_expected = 0

    for reg_idx in valid_regions:
        group = assigned[assigned['region_idx'] == reg_idx]
        n = len(group)
        region_sizes.append(n)

        true_ranking = group['y_true'].values.argsort()[::-1]  # descending
        pred_ranking = group['y_pred'].values.argsort()[::-1]

        # Model's top-1 pick
        model_best_idx = pred_ranking[0]
        # True rank of model's best pick (0-indexed, 0 = best)
        true_rank = np.where(true_ranking == model_best_idx)[0][0]

        if true_rank == 0:
            hit_at_1 += 1
        if true_rank < 3:
            hit_at_3 += 1

        # Percentile (1.0 = best, 0.0 = worst)
        rank_percentiles.append(1.0 - true_rank / (n - 1) if n > 1 else 1.0)
        random_hit1_expected += 1.0 / n

    hit1_rate = hit_at_1 / n_regions
    hit3_rate = hit_at_3 / n_regions
    random_hit1_rate = random_hit1_expected / n_regions
    median_pct = np.median(rank_percentiles)

    print(f"\n  Results:")
    print(f"    Hit@1 (exact best):     {hit1_rate:.1%}  (random: {random_hit1_rate:.1%}, "
          f"{hit1_rate/random_hit1_rate:.1f}x)")
    print(f"    Hit@3 (in top 3):       {hit3_rate:.1%}")
    print(f"    Median rank percentile: {median_pct:.1%}  (1.0 = perfect, 0.5 = random)")
    print(f"    Mean region size:       {np.mean(region_sizes):.1f} probes")

    return {'hit1': hit1_rate, 'hit3': hit3_rate, 'random_hit1': random_hit1_rate,
            'median_pct': median_pct, 'rank_pcts': rank_percentiles,
            'n_regions': n_regions, 'region_sizes': region_sizes,
            'assigned_df': assigned}


def module_3_panel_redesign(assigned_df):
    """Simulated panel redesign: model-recommended vs positionally central probe."""
    print(f"\n{'='*70}")
    print(f"  MODULE 3: SIMULATED PANEL REDESIGN")
    print(f"{'='*70}")

    gains = []
    n_improved = 0
    n_degraded = 0
    n_same = 0

    for reg_idx in assigned_df['region_idx'].unique():
        group = assigned_df[assigned_df['region_idx'] == reg_idx].copy()
        if len(group) < 3:
            continue

        # "Default" probe: positionally central (middle by genomic coordinate)
        group = group.sort_values('Probe_Start')
        mid_idx = len(group) // 2
        default_eff = group.iloc[mid_idx]['y_true']

        # Model's recommended: highest predicted efficiency
        model_best = group.loc[group['y_pred'].idxmax()]
        model_eff = model_best['y_true']

        gain = model_eff - default_eff
        gains.append(gain)
        if gain > 0.01:
            n_improved += 1
        elif gain < -0.01:
            n_degraded += 1
        else:
            n_same += 1

    gains = np.array(gains)
    n_total = len(gains)

    print(f"  Regions analyzed: {n_total}")
    print(f"  Model improves:   {n_improved} ({100*n_improved/n_total:.1f}%)")
    print(f"  Model degrades:   {n_degraded} ({100*n_degraded/n_total:.1f}%)")
    print(f"  No change (±0.01): {n_same} ({100*n_same/n_total:.1f}%)")
    print(f"  Mean gain:        {gains.mean():.4f} (z-score units)")
    print(f"  Median gain:      {np.median(gains):.4f}")
    print(f"  Gain > 0.5 sigma: {(gains > 0.5).sum()} regions")

    return {'gains': gains, 'n_improved': n_improved, 'n_degraded': n_degraded,
            'n_total': n_total}


def module_4_per_chromosome(y_pred, y_true, probe_df):
    """Per-chromosome Spearman correlation."""
    print(f"\n{'='*70}")
    print(f"  MODULE 4: PER-CHROMOSOME SPEARMAN")
    print(f"{'='*70}")

    chrom_results = {}
    for chrom in sorted(probe_df['Chromosome'].unique(),
                        key=lambda c: (0, int(c.replace('chr', '')))
                        if c.replace('chr', '').isdigit()
                        else (1, c)):
        mask = probe_df['Chromosome'].values == chrom
        if mask.sum() < 50:
            continue
        rho, _ = stats.spearmanr(y_pred[mask], y_true[mask])
        chrom_results[chrom] = (rho, mask.sum())
        print(f"  {chrom:<8s}  rho={rho:.4f}  n={mask.sum():>6d}")

    rhos = [v[0] for v in chrom_results.values()]
    print(f"\n  Mean: {np.mean(rhos):.4f}  Min: {np.min(rhos):.4f}  Max: {np.max(rhos):.4f}")

    return chrom_results


def module_5_error_analysis(y_pred, y_true, probe_df):
    """Error analysis by probe GC%, Entropy, Tm."""
    print(f"\n{'='*70}")
    print(f"  MODULE 5: ERROR ANALYSIS BY PROBE FEATURES")
    print(f"{'='*70}")

    results = {}
    for feature in ['GC_pct', 'Entropy', 'Tm']:
        vals = probe_df[feature].values
        valid = ~np.isnan(vals)
        if valid.sum() < 100:
            continue
        quantiles = np.quantile(vals[valid], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bin_rhos = []
        print(f"\n  {feature}:")
        for i in range(5):
            lo, hi = quantiles[i], quantiles[i + 1]
            if i < 4:
                mask = valid & (vals >= lo) & (vals < hi)
            else:
                mask = valid & (vals >= lo) & (vals <= hi)
            if mask.sum() < 50:
                continue
            rho, _ = stats.spearmanr(y_pred[mask], y_true[mask])
            bin_rhos.append(rho)
            print(f"    [{lo:.1f}, {hi:.1f})  n={mask.sum():>6d}  rho={rho:.4f}")
        results[feature] = bin_rhos

    return results


# ======================================================================
# Plotting
# ======================================================================

def generate_figure(m1, m2, m3, m4, m5, output_path):
    """Generate 6-panel publication figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel A: Predicted vs Actual scatter
    ax = axes[0, 0]
    n = len(m1['y_true'])
    sub = np.random.choice(n, min(50000, n), replace=False)
    ax.scatter(m1['y_true'][sub], m1['y_pred'][sub], s=1, alpha=0.1,
               c=C_MODEL, rasterized=True)
    lims = [min(m1['y_true'].min(), m1['y_pred'].min()),
            max(m1['y_true'].max(), m1['y_pred'].max())]
    ax.plot(lims, lims, color=C_REF, linestyle='--', linewidth=1.2)
    ax.set_xlabel('True Efficiency')
    ax.set_ylabel('Predicted Efficiency')
    ax.set_title(f'A. Predicted vs True (Real Probes)\n'
                 f'Spearman={m1["spearman"]:.3f}, Pearson={m1["pearson"]:.3f}')

    # Panel B: Per-region rank percentile histogram
    ax = axes[0, 1]
    ax.hist(m2['rank_pcts'], bins=20, color=C_MODEL, alpha=0.8, edgecolor='white', density=True)
    ax.axvline(0.5, color=C_RANDOM, linestyle='--', linewidth=1.5, label='Random (0.5)')
    ax.axvline(m2['median_pct'], color=C_REF, linestyle='-', linewidth=2,
               label=f'Median ({m2["median_pct"]:.2f})')
    ax.set_xlabel('Rank Percentile of Model Top-1\n(1.0 = best probe)')
    ax.set_ylabel('Density')
    ax.set_title(f'B. Per-Region Probe Selection\n'
                 f'Hit@1={m2["hit1"]:.1%}, Hit@3={m2["hit3"]:.1%} '
                 f'({m2["n_regions"]} regions)')
    ax.legend(fontsize=9)

    # Panel C: Panel redesign gain distribution
    ax = axes[0, 2]
    ax.hist(m3['gains'], bins=50, color=C_MODEL_LIGHT, alpha=0.8, edgecolor='white')
    ax.axvline(0, color=C_RANDOM, linestyle='--', linewidth=1.5, label='No change')
    ax.axvline(np.median(m3['gains']), color=C_REF, linewidth=2,
               label=f'Median gain ({np.median(m3["gains"]):.3f})')
    pct_improved = 100 * m3['n_improved'] / m3['n_total']
    ax.set_xlabel('Efficiency Gain (model pick - default)')
    ax.set_ylabel('Count (regions)')
    ax.set_title(f'C. Simulated Panel Redesign\n'
                 f'{pct_improved:.0f}% regions improved')
    ax.legend(fontsize=9)

    # Panel D: Per-chromosome Spearman
    ax = axes[1, 0]
    chroms = list(m4.keys())
    rhos = [m4[c][0] for c in chroms]
    labels = [c.replace('chr', '') for c in chroms]
    colors = [C_MODEL if r > 0.3 else C_RANDOM for r in rhos]
    ax.bar(range(len(chroms)), rhos, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(chroms)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.axhline(np.mean(rhos), color=C_REF, linestyle='--',
               label=f'Mean ({np.mean(rhos):.3f})')
    ax.set_ylabel('Spearman rho')
    ax.set_title('D. Per-Chromosome Consistency')
    ax.legend(fontsize=9)

    # Panel E: Spearman by GC% bins
    ax = axes[1, 1]
    if 'GC_pct' in m5 and len(m5['GC_pct']) == 5:
        gc_labels = ['Q1\n(low)', 'Q2', 'Q3', 'Q4', 'Q5\n(high)']
        ax.bar(range(5), m5['GC_pct'], color=C_MODEL, alpha=0.85, edgecolor='white')
        ax.set_xticks(range(5))
        ax.set_xticklabels(gc_labels)
        ax.axhline(np.mean(m5['GC_pct']), color=C_REF, linestyle='--')
        ax.set_ylabel('Spearman rho')
        ax.set_title('E. Spearman by GC% Quintile')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

    # Panel F: Top-K precision comparison
    ax = axes[1, 2]
    k_values = [1, 5, 10, 20]
    probe_precs = [m1['precs'][k] for k in k_values]
    x = np.arange(len(k_values))
    w = 0.35
    ax.bar(x - w/2, probe_precs, w, color=C_MODEL, alpha=0.85, label='Real probes')
    sw_precs = [None, None, SW_TOP10, None]  # only have Top-10% for sliding windows
    # Show random baseline
    for i, k in enumerate(k_values):
        ax.plot([i - 0.4, i + 0.4], [k/100, k/100], color=C_RANDOM,
                linestyle='--', linewidth=1.5,
                label='Random' if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top {k}%' for k in k_values])
    ax.set_ylabel('Precision')
    ax.set_ylim(0, 1)
    ax.set_title('F. Top-K Selection Accuracy')
    ax.legend(fontsize=9)

    plt.tight_layout()
    # Defensive: re-ensure parent directory exists at write time.
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {output_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="DEPROBE-DNA: Probe Validation Suite")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--staging_csv", required=True)
    parser.add_argument("--probe_mapping", required=True)
    parser.add_argument("--target_bed", required=True)
    # Default output_dir to <project>/results/plots (auto-created).
    _default_output_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'results', 'plots'))
    parser.add_argument("--output_dir", default=_default_output_dir,
                        help=f"Directory for output plot (default: {_default_output_dir})")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--prior_dim", type=int, default=12, help="Prior dimension (12 or 18)")
    parser.add_argument("--output_suffix", default="",
                        help="Output filename suffix (e.g. for separating runs).")
    args = parser.parse_args()

    def _sfx(filename: str) -> str:
        """Insert args.output_suffix before file extension. Pass-through if empty."""
        if not args.output_suffix:
            return filename
        base, ext = os.path.splitext(filename)
        return f"{base}{args.output_suffix}{ext}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Defensive: auto-create the output directory tree if missing.
    os.makedirs(args.output_dir, exist_ok=True)

    prior_cols = [
        'Tm', 'GC_pct', 'dG', 'Hairpin_Tm', 'Dimer_Tm', 'Yield', 'Norm_Len',
        'GC_Skew', 'Entropy', 'dG_5p', 'dG_3p', 'Collision_Penalty'
    ]

    print("=" * 70)
    print("  DEPROBE-DNA: PROBE VALIDATION SUITE")
    print("=" * 70)
    print(f"  Checkpoint:    {os.path.basename(args.checkpoint)}")
    print(f"  Probe H5:     {os.path.basename(args.h5)}")
    print(f"  Device:        {device}")

    # --- Load model and run inference ---
    print("\n  Loading model and running inference...")
    model, prior_mean, prior_std = load_model(args.checkpoint, device, prior_dim=args.prior_dim)
    dataset = PanMolecularProbeDataset(args.h5, prior_mean=prior_mean, prior_std=prior_std)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True, persistent_workers=True)
    print(f"  Samples: {len(dataset):,}")
    y_pred, y_true = predict_all(model, loader, device)

    # --- Link H5 rows to probe metadata ---
    print("\n  Building probe traceability map...")
    probe_df = link_h5_to_probes(args.staging_csv, args.probe_mapping, prior_cols)

    # Verify alignment
    assert len(probe_df) == len(y_pred), \
        f"Row count mismatch: probe_df={len(probe_df)}, predictions={len(y_pred)}"

    # --- Load target regions ---
    target_regions, n_target = load_target_regions(args.target_bed)
    print(f"  Target regions loaded: {n_target}")

    # ================================================================
    # Run all 6 modules
    # ================================================================
    m1 = module_1_global_metrics(y_pred, y_true)
    m2 = module_2_per_region(y_pred, y_true, probe_df, target_regions, min_probes=3)
    m3 = module_3_panel_redesign(m2['assigned_df'])
    m4 = module_4_per_chromosome(y_pred, y_true, probe_df)
    m5 = module_5_error_analysis(y_pred, y_true, probe_df)

    # ================================================================
    # Module 6: Summary Report
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  MODULE 6: PAPER-READY SUMMARY")
    print(f"{'='*70}")
    print(f"""
  DEPROBE-DNA was validated on {len(y_pred):,} real Nextera Expanded Exome
  probes (94 bp, Illumina) using an independent technical replicate (NIST7035)
  for ground-truth efficiency labels.

  GLOBAL: The model achieved Spearman rho = {m1['spearman']:.3f} between
  predicted and observed capture efficiency, with Top-10% precision of
  {m1['precs'][10]:.1%} ({m1['precs'][10]/0.10:.1f}x enrichment over random).

  PER-REGION: Across {m2['n_regions']:,} target regions (>= 3 probes each),
  the model identified the single best probe {m2['hit1']:.1%} of the time
  ({m2['hit1']/m2['random_hit1']:.1f}x vs random), and placed its top pick
  within the true top 3 in {m2['hit3']:.1%} of regions. The median rank
  percentile of the model's selection was {m2['median_pct']:.2f} (1.0 = perfect).

  PANEL REDESIGN: In a simulated redesign where the model's top-ranked probe
  replaced the positionally central probe, {100*m3['n_improved']/m3['n_total']:.0f}% of
  regions showed improved capture efficiency (median gain: {np.median(m3['gains']):.3f}).

  GENOME-WIDE: Per-chromosome Spearman ranged from {min(v[0] for v in m4.values()):.3f}
  to {max(v[0] for v in m4.values()):.3f} (mean {np.mean([v[0] for v in m4.values()]):.3f}),
  confirming consistent performance across the genome.
""")

    # Generate figure
    fig_path = os.path.join(args.output_dir, _sfx('probe_validation_suite.png'))
    generate_figure(m1, m2, m3, m4, m5, fig_path)

    # ================================================================
    # CSV/JSON exports
    #   results/tables/probe_global_metrics.csv     (Module 1, single row)
    #   results/tables/probe_topk_precision.csv     (Module 1, K rows)
    #   results/tables/probe_per_region.csv         (Module 2, single row)
    #   results/tables/probe_panel_redesign.csv     (Module 3, single row)
    #   results/tables/probe_per_chromosome.csv     (Module 4, chr rows)
    #   results/tables/probe_per_feature.csv        (Module 5, feature×bin rows)
    #   results/json/probe_validation_full.json     (Everything in one dump)
    # ================================================================
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

    # Module 1: global headline metrics
    pd.DataFrame([{
        'N_probes': int(len(y_pred)),
        'Spearman_rho': round(float(m1['spearman']), 4),
        'Pearson_r': round(float(m1['pearson']), 4),
        'MSE': round(float(m1['mse']), 4),
        'Top1_pct_precision': round(float(m1['precs'][1]) * 100, 2),
        'Top5_pct_precision': round(float(m1['precs'][5]) * 100, 2),
        'Top10_pct_precision': round(float(m1['precs'][10]) * 100, 2),
        'Top20_pct_precision': round(float(m1['precs'][20]) * 100, 2),
        'NDCG_at_10pct': round(float(m1['ndcgs'][10]), 4),
    }]).to_csv(os.path.join(TABLES_DIR, _sfx('probe_global_metrics.csv')), index=False)

    # Module 1: top-K precision long-format (one row per K)
    pd.DataFrame([
        {'K_percent': k,
         'Precision_pct': round(float(m1['precs'][k]) * 100, 2),
         'Enrichment_x': round(float(m1['precs'][k]) / (k / 100.0), 2),
         'NDCG': round(float(m1['ndcgs'][k]), 4)}
        for k in sorted(m1['precs'].keys())
    ]).to_csv(os.path.join(TABLES_DIR, _sfx('probe_topk_precision.csv')), index=False)

    # Module 2: per-region best-probe identification
    pd.DataFrame([{
        'N_regions': int(m2['n_regions']),
        'Hit1_rate_pct': round(float(m2['hit1']) * 100, 2),
        'Hit3_rate_pct': round(float(m2['hit3']) * 100, 2),
        'Random_Hit1_rate_pct': round(float(m2['random_hit1']) * 100, 2),
        'Median_rank_percentile': round(float(m2['median_pct']), 4),
        'Mean_region_size': round(float(np.mean(m2['region_sizes'])), 2),
    }]).to_csv(os.path.join(TABLES_DIR, _sfx('probe_per_region.csv')), index=False)

    # Module 3: simulated panel redesign
    gains_arr = np.asarray(m3['gains'])
    pd.DataFrame([{
        'N_regions': int(m3['n_total']),
        'N_improved': int(m3['n_improved']),
        'N_degraded': int(m3['n_degraded']),
        'Improved_pct': round(100 * m3['n_improved'] / m3['n_total'], 2) if m3['n_total'] else 0.0,
        'Mean_gain': round(float(gains_arr.mean()), 4) if len(gains_arr) else 0.0,
        'Median_gain': round(float(np.median(gains_arr)), 4) if len(gains_arr) else 0.0,
        'Gain_above_0p5_sigma': int((gains_arr > 0.5).sum()) if len(gains_arr) else 0,
    }]).to_csv(os.path.join(TABLES_DIR, _sfx('probe_panel_redesign.csv')), index=False)

    # Module 4: per-chromosome Spearman
    pd.DataFrame([
        {'Chromosome': chrom, 'Spearman': round(float(rho), 4), 'N': int(n)}
        for chrom, (rho, n) in m4.items()
    ]).to_csv(os.path.join(TABLES_DIR, _sfx('probe_per_chromosome.csv')), index=False)

    # Module 5: per-feature stratification (long format: feature × bin index)
    feat_rows = []
    for feature, bin_rhos in m5.items():
        for bin_idx, rho in enumerate(bin_rhos):
            feat_rows.append({
                'Feature': feature,
                'Bin_index': int(bin_idx),
                'Spearman': round(float(rho), 4) if rho == rho else None,
            })
    pd.DataFrame(feat_rows).to_csv(
        os.path.join(TABLES_DIR, _sfx('probe_per_feature.csv')), index=False)

    # Master JSON: everything in one file (numpy-coerced)
    def _coerce(obj):
        if isinstance(obj, dict):
            return {str(k): _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_coerce(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    full_json = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'probe_h5': os.path.abspath(args.h5),
        'n_probes': int(len(y_pred)),
        'module_1_global': {k: _coerce(v) for k, v in m1.items() if k not in ('y_pred', 'y_true')},
        'module_2_per_region': {k: _coerce(v) for k, v in m2.items() if k != 'assigned_df'},
        'module_3_panel_redesign': {
            'n_total': int(m3['n_total']),
            'n_improved': int(m3['n_improved']),
            'n_degraded': int(m3['n_degraded']),
            'mean_gain': float(gains_arr.mean()) if len(gains_arr) else 0.0,
            'median_gain': float(np.median(gains_arr)) if len(gains_arr) else 0.0,
        },
        'module_4_per_chromosome': {chrom: {'spearman': float(r), 'n': int(n)}
                                    for chrom, (r, n) in m4.items()},
        'module_5_per_feature': _coerce(m5),
    }
    full_json_path = os.path.join(JSON_DIR, _sfx('probe_validation_full.json'))
    with open(full_json_path, 'w') as fh:
        json.dump(full_json, fh, indent=2)

    print(f"\n  Tables written to: {TABLES_DIR}")
    print(f"  Full JSON dump : {full_json_path}")

    print(f"{'='*70}")
    print(f"  SUITE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
