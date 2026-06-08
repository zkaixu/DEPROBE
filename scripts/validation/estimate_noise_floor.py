#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE: Between-Replicate Variance Floor (post-QN labels)
==========================================================
Compares the quantile-normalised capture-efficiency labels between
two BAMs of the SAME individual (NA12878):

    Train H5 ← derived from NIST7086 BAM (flow cell H7AP8ADXX, lane 2,
               barcode CGTACTAG), @RG SM:NIST7086
    Val   H5 ← derived from NIST7035 BAM (flow cell H7AP8ADXX, lane 2,
               barcode TAAGGCGA), @RG SM:NIST7035

Verified shared properties (from @RG PU tag):
  - Same flow cell  : H7AP8ADXX
  - Same lane       : 2
  - Same individual : NA12878 (HG001 / Genome in a Bottle)

Different properties:
  - Different barcodes (CGTACTAG vs TAAGGCGA)
  - Different @RG SM tags (NIST7086 vs NIST7035)

NOT verifiable from @RG:
  - Whether the two BAMs come from the same library prep batch
    (@RG LB is the generic placeholder "library" in both, no info).
  - Whether the capture kit is identical (this would have to be
    confirmed from project metadata, not from the BAM header).

Probe sequences are identical between the two H5 files. Row i in
Train and row i in Val target the same genomic position. Only the
efficiency *label* differs.

What this script measures
-------------------------
    MSE_between = E[(y_train - y_val)^2]    over identical probes
    Per-measurement variance ≈ MSE_between / 2

This is a **between-replicate variance estimate after quantile
normalisation**, not a hard lower bound on what a model can achieve.
The labels in each H5 have already been mapped to a standard normal
distribution via QuantileTransformer (output_distribution='normal'),
which compresses raw count noise. A regression model trained on one
replicate can output predictions whose MSE against the OTHER replicate
falls BELOW this between-replicate variance because:

  (a) Train and Val H5 contain the SAME probe sequences. Row i in
      both files refers to the same genomic position. The model has
      already seen those sequences during training (with replicate-1
      labels).
  (b) Quantile normalisation maps both replicates onto an identical
      marginal distribution, compressing raw count noise into rank-
      perturbation residuals.
  (c) Both BAMs share systematic effects (same flow cell, lane, library
      recipe) that are partly determined by sequence and therefore
      learnable.

Use this number as a *reference scale for between-replicate
consistency*, not as an "irreducible noise floor" in the strict sense.
For the paper, true cross-kit / cross-panel generalisation is measured
by `probe_validation_suite.py` on the 344K real Nextera Expanded Exome
probes (different sequences, different panel design).

Outputs
-------
    results/plots/noise_floor_scatter.png      # 3-panel diagnostic
    results/tables/noise_floor_by_bin.csv      # per-bin breakdown
    results/json/noise_floor.json              # canonical record

Usage
-----
    python estimate_noise_floor.py
"""

import os
import sys
import json
import h5py
import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ====================================================================
# Configuration
# ====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TRAIN_H5 = f"{PROJECT_ROOT}/data/data_factory/final/train/deprobe_train_master.h5"
VAL_H5 = f"{PROJECT_ROOT}/data/data_factory/final/val/deprobe_val_master.h5"

OUTPUT_DIR = f"{PROJECT_ROOT}/results/plots"
PLOT_PATH = f"{OUTPUT_DIR}/noise_floor_scatter.png"
TABLES_DIR = f"{PROJECT_ROOT}/results/tables"
JSON_DIR = f"{PROJECT_ROOT}/results/json"
# Defensive: auto-create output directory if missing.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Both H5 files use the same BED regions with the same sliding window,
# so row i in Train corresponds to the same genomic position as row i in Val.
# This is guaranteed by the identical extract_sequences.py + calc_efficiency.py
# pipeline run on the same BED file with the same parameters.

# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 70)
    print("  DEPROBE: BETWEEN-REPLICATE VARIANCE FLOOR (post-QN)")
    print("  NA12878 / NIST7086 vs NIST7035 (same flow cell, different barcode)")
    print("=" * 70)

    # Load efficiency labels
    for path, name in [(TRAIN_H5, "Train"), (VAL_H5, "Val")]:
        if not os.path.exists(path):
            print(f"[FATAL] {name} H5 not found: {path}")
            sys.exit(1)

    with h5py.File(TRAIN_H5, 'r') as f:
        eff_train = f['efficiency'][:]
    with h5py.File(VAL_H5, 'r') as f:
        eff_val = f['efficiency'][:]

    n_train = len(eff_train)
    n_val = len(eff_val)

    print(f"\n  Train (NIST7086): {n_train:,} probes")
    print(f"  Val   (NIST7035): {n_val:,} probes")

    if n_train != n_val:
        print(f"\n  [WARN] Row counts differ ({n_train} vs {n_val}).")
        print(f"  Using first {min(n_train, n_val):,} rows for comparison.")
        n = min(n_train, n_val)
        eff_train = eff_train[:n]
        eff_val = eff_val[:n]
    else:
        n = n_train
        print(f"  Row counts match: {n:,}")

    # ================================================================
    # Core Statistics
    # ================================================================
    residuals = eff_train - eff_val
    mse_between = np.mean(residuals ** 2)
    mae_between = np.mean(np.abs(residuals))
    pearson_r, p_value = stats.pearsonr(eff_train, eff_val)
    r_squared = pearson_r ** 2
    spearman_rho, _ = stats.spearmanr(eff_train, eff_val)

    # Noise floor: half of the between-replicate MSE
    # Rationale: if two replicates each have noise variance sigma^2,
    # then Var(y1 - y2) = 2 * sigma^2, so sigma^2 = MSE_between / 2.
    # This sigma^2 is the irreducible noise per single measurement.
    noise_floor_mse = mse_between / 2.0

    # Variance decomposition
    total_var = np.var(eff_train)
    signal_var = total_var - noise_floor_mse
    signal_fraction = signal_var / total_var if total_var > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  Between-Replicate Statistics:")
    print(f"    MSE(Train vs Val):         {mse_between:.4f}")
    print(f"    MAE(Train vs Val):         {mae_between:.4f}")
    print(f"    Pearson r:                 {pearson_r:.4f}  (p < {p_value:.2e})")
    print(f"    R-squared:                 {r_squared:.4f}")
    print(f"    Spearman rho:              {spearman_rho:.4f}")

    print(f"\n  Between-Replicate Variance (post-QN):")
    print(f"    Between-replicate MSE:     {mse_between:.4f}")
    print(f"    Per-measurement variance:  {noise_floor_mse:.4f}  (MSE_between / 2)")
    print(f"    Total label variance:      {total_var:.4f}")
    print(f"    Variance not explained by between-rep diff: {signal_var:.4f}")
    print(f"    Fraction NOT in between-rep diff:           {signal_fraction:.1%}")

    print(f"\n  INTERPRETATION CAVEATS:")
    print(f"    {noise_floor_mse:.4f} is the per-measurement variance AFTER labels")
    print(f"    have been quantile-normalised to a standard-normal distribution.")
    print(f"    It is NOT a hard lower bound on achievable model MSE — a model")
    print(f"    can produce MSE below this number because:")
    print(f"      (a) Train and Val H5 contain the SAME probe sequences (row i in")
    print(f"          both files = same genomic position).")
    print(f"      (b) Quantile normalisation maps both replicates onto an")
    print(f"          identical marginal, compressing raw count noise.")
    print(f"      (c) The two BAMs share at least flow cell + lane (verified")
    print(f"          via @RG PU tag); any sequence-predictable systematic")
    print(f"          effect from those is learnable. Whether they share the")
    print(f"          same library prep batch is NOT verifiable from @RG.")
    print(f"    Use this number as a *between-replicate consistency reference*,")
    print(f"    NOT as an irreducible-noise lower bound. For true cross-kit")
    print(f"    generalisation, see probe_validation_suite.py output.")

    # ================================================================
    # Percentile Analysis
    # ================================================================
    abs_res = np.abs(residuals)
    percentiles = [50, 75, 90, 95, 99]
    print(f"\n  Absolute Residual Percentiles:")
    print(f"    {'Percentile':<15s} {'|Train - Val|':>15s}")
    print(f"    {'-' * 30}")
    for p in percentiles:
        val = np.percentile(abs_res, p)
        print(f"    {p}th{'':<10s} {val:>15.4f}")

    # ================================================================
    # Binned Analysis: noise varies by efficiency level
    # ================================================================
    n_bins = 10
    bin_edges = np.linspace(
        min(eff_train.min(), eff_val.min()),
        max(eff_train.max(), eff_val.max()),
        n_bins + 1
    )
    mean_eff = (eff_train + eff_val) / 2

    print(f"\n  Noise by Efficiency Bin:")
    print(f"    {'Bin Range':<20s} {'Count':>8s} {'MSE':>8s} {'MAE':>8s} {'r':>8s}")
    print(f"    {'-' * 52}")

    bin_mses = []
    bin_centers = []
    for i in range(n_bins):
        mask = (mean_eff >= bin_edges[i]) & (mean_eff < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (mean_eff >= bin_edges[i]) & (mean_eff <= bin_edges[i + 1])
        count = mask.sum()
        if count > 10:
            bin_mse = np.mean(residuals[mask] ** 2)
            bin_mae = np.mean(np.abs(residuals[mask]))
            bin_r, _ = stats.pearsonr(eff_train[mask], eff_val[mask])
            bin_mses.append(bin_mse)
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            print(f"    [{bin_edges[i]:>6.2f}, {bin_edges[i+1]:>6.2f})"
                  f" {count:>8,} {bin_mse:>8.4f} {bin_mae:>8.4f} {bin_r:>8.4f}")
        else:
            print(f"    [{bin_edges[i]:>6.2f}, {bin_edges[i+1]:>6.2f})"
                  f" {count:>8,}      (too few samples)")

    # ================================================================
    # Reference Numbers
    #   No hardcoded model MSE here. Model performance is read from
    #   results/json/main_metrics.json (produced by evaluate_model.py)
    #   to keep this script's output decoupled from training state.
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  REFERENCE NUMBERS")
    print(f"{'=' * 70}")
    print(f"\n    {'Quantity':<42s} {'Value':>10s}")
    print(f"    {'-' * 54}")
    print(f"    {'Random baseline MSE (predict mean)':<42s} {total_var:>10.4f}")
    print(f"    {'Between-replicate per-meas variance':<42s} {noise_floor_mse:>10.4f}")
    print(f"    {'Between-replicate raw MSE':<42s} {mse_between:>10.4f}")
    print(f"\n    For model performance numbers, see:")
    print(f"      results/json/main_metrics.json"
          f"     ← run evaluate_model.py first")
    print(f"      results/json/probe_validation_full.json"
          f" ← run go_probe_suite_12d.sh first")

    # ================================================================
    # Scatter Plot
    # ================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Semantic color palette, top-journal grade saturated forest-green family
    # for technical replicate / noise floor data, vivid tomato red for reference lines.
    # Deep saturated tones replace the previous pastel set to improve contrast at
    # low scatter alpha and to align with Nature/NAR figure conventions.
    C_MODEL = '#4878D0'
    C_FLOOR = '#1F7A48'         # deep forest green for replicate scatter
    C_FLOOR_LIGHT = '#74C69D'   # spring green for histogram fill (visible at alpha 0.8)
    C_FLOOR_WARM = '#40916C'    # medium forest green for bar chart
    C_RANDOM = '#D65F5F'
    C_REF = '#D62828'           # tomato red, vivid contrast to forest-green data

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Scatter, green = technical replicate data points
    ax = axes[0]
    subsample = np.random.choice(n, min(50000, n), replace=False)
    ax.scatter(eff_train[subsample], eff_val[subsample],
               s=1, alpha=0.1, c=C_FLOOR, rasterized=True)
    lims = [min(eff_train.min(), eff_val.min()), max(eff_train.max(), eff_val.max())]
    ax.plot(lims, lims, color=C_REF, linestyle='--', linewidth=1.2,
            label='Identity line (y = x)')
    ax.set_xlabel('Efficiency', fontsize=11)
    ax.set_ylabel('Efficiency', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal')

    # Panel B: Residual histogram, light green fill, distinct from Panel A scatter
    ax = axes[1]
    ax.hist(residuals, bins=100, density=True, color=C_FLOOR_LIGHT,
            alpha=0.8, edgecolor='none')
    ax.axvline(0, color=C_REF, linestyle='--', linewidth=1.2,
               label='Zero residual')
    ax.axvline(np.mean(residuals), color=C_FLOOR, linestyle='-', linewidth=1.0,
               label=f'Mean residual ({np.mean(residuals):.3f})')
    ax.set_xlabel('Residual', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9)

    # Panel C: Noise by bin, warm green bars, orange global floor line
    ax = axes[2]
    if bin_centers and bin_mses:
        ax.bar(range(len(bin_centers)), [m / 2 for m in bin_mses],
               color=C_FLOOR_WARM, alpha=0.8, edgecolor='white')
        ax.set_xticks(range(len(bin_centers)))
        ax.set_xticklabels([f'{c:.1f}' for c in bin_centers], fontsize=8, rotation=45)
        ax.axhline(noise_floor_mse, color=C_REF, linestyle='--', linewidth=1.5,
                   label=f'Global per-meas variance = {noise_floor_mse:.4f}')
        ax.set_xlabel('Efficiency bin centre', fontsize=11)
        ax.set_ylabel('Per-measurement variance', fontsize=11)
        ax.legend(fontsize=9)

    plt.tight_layout()
    # Defensive: re-ensure parent directory exists at write time.
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n  Plot saved to: {PLOT_PATH}")
    plt.close()

    # ================================================================
    # Paper noise-floor record. Single number every other script reads.
    # JSON: results/json/noise_floor.json
    # CSV : results/tables/noise_floor_by_bin.csv
    # ================================================================
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    json_payload = {
        'train_h5': os.path.abspath(TRAIN_H5),
        'val_h5': os.path.abspath(VAL_H5),
        'n_positions': int(n),
        'sample': 'NA12878 (HG001 / Genome in a Bottle)',
        'replicate_setup_verified_from_RG_tag': {
            'train_bam_id': 'NIST7086',
            'val_bam_id': 'NIST7035',
            'flow_cell': 'H7AP8ADXX',
            'lane': 2,
            'barcodes': ['CGTACTAG', 'TAAGGCGA'],
            'platform': 'Illumina',
        },
        'replicate_setup_unverified_from_RG_tag': {
            'same_library_prep_batch': 'unknown (LB tag is placeholder "library" in both)',
            'capture_kit': 'not in @RG; assumed Nextera Rapid Capture from project metadata',
        },
        'sequence_overlap_pct': 100.0,  # row i in Train and Val target same genomic position
        'between_replicate': {
            'MSE': float(mse_between),
            'MAE': float(mae_between),
            'Pearson_r': float(pearson_r),
            'R2': float(r_squared),
            'Spearman_rho': float(spearman_rho),
        },
        'noise_floor_mse_per_measurement': float(noise_floor_mse),
        'total_label_variance': float(total_var),
        'estimated_signal_variance': float(signal_var),
        'estimated_signal_fraction': float(signal_fraction),
        'absolute_residual_percentiles': {
            f'p{p}': float(np.percentile(abs_res, p)) for p in percentiles
        },
        'caveats': (
            "Labels are quantile-normalised to a standard-normal distribution "
            "before this comparison. The reported per-measurement variance is "
            "NOT a hard lower bound on model MSE. A regression model can "
            "score below it because (1) Train and Val contain the same probe "
            "sequences, (2) QN compresses raw count noise, and (3) the two "
            "BAMs share systematic effects that are sequence-predictable. Use "
            "this as a between-replicate consistency reference, not as an "
            "irreducible noise floor."
        ),
    }
    json_path = os.path.join(JSON_DIR, 'noise_floor.json')
    with open(json_path, 'w') as fh:
        json.dump(json_payload, fh, indent=2)

    bin_csv_path = os.path.join(TABLES_DIR, 'noise_floor_by_bin.csv')
    pd.DataFrame([
        {'Bin_center_zscore': round(float(c), 4),
         'Per_measurement_noise_MSE': round(float(m / 2.0), 4),
         'Between_replicate_MSE': round(float(m), 4)}
        for c, m in zip(bin_centers, bin_mses)
    ]).to_csv(bin_csv_path, index=False)

    print(f"  JSON: {json_path}")
    print(f"  CSV : {bin_csv_path}")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
