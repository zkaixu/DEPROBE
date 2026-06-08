#!/usr/bin/env python3
"""
DEPROBE-DNA: Probe-Level Replicate Spearman + Top-25 Genomic Distribution
=========================================================================
Two supplementary analyses:
1. Compute between-replicate Spearman at REAL probe positions
   (NIST7086 vs NIST7035) to establish the noise ceiling
2. Check genomic distribution of model's Top-25 and Bottom-25 probes

Usage:
    python probe_replicate_check.py \
        --checkpoint <path> --h5 <probe_val.h5> \
        --staging_csv <csv> --probe_mapping <csv> \
        --bam_7086 <bam> --bam_7035 <bam>
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import pysam
import torch
from scipy import stats
from collections import Counter
from torch.utils.data import DataLoader

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
    model = DEPROBE(num_platforms=10, prior_dim=12, num_modalities=5, d_model=256).to(device)
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


def compute_depth_at_probes(bam_path, probe_df, sample_size=None):
    """Compute mean depth at each probe position from a BAM file."""
    bam = pysam.AlignmentFile(bam_path, "rb")

    if sample_size and sample_size < len(probe_df):
        indices = np.random.choice(len(probe_df), sample_size, replace=False)
        probe_df = probe_df.iloc[indices].reset_index(drop=True)

    depths = []
    for _, row in probe_df.iterrows():
        chrom = row['Chromosome']
        start = int(row['Probe_Start'])
        end = int(row['Probe_End'])
        try:
            # Mean depth across probe region
            cols = bam.count_coverage(chrom, start, end, quality_threshold=20)
            total = np.array(cols).sum(axis=0)
            depths.append(total.mean() if len(total) > 0 else 0)
        except (ValueError, KeyError):
            depths.append(0)

    bam.close()
    return np.array(depths), probe_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--staging_csv", required=True)
    parser.add_argument("--probe_mapping", required=True)
    parser.add_argument("--bam_7086", required=True)
    parser.add_argument("--bam_7035", required=True)
    parser.add_argument("--replicate_sample", type=int, default=20000,
                        help="Sample size for replicate analysis (full BAM scan is slow)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior_cols = [
        'Tm', 'GC_pct', 'dG', 'Hairpin_Tm', 'Dimer_Tm', 'Yield', 'Norm_Len',
        'GC_Skew', 'Entropy', 'dG_5p', 'dG_3p', 'Collision_Penalty'
    ]

    print("=" * 70)
    print("  SUPPLEMENTARY ANALYSIS: REPLICATE SPEARMAN + TOP-25 DISTRIBUTION")
    print("=" * 70)

    # --- Build probe metadata (same logic as probe_validation_suite.py) ---
    print("\n  Building probe traceability map...")
    df_stage = pd.read_csv(args.staging_csv)
    valid_mask = df_stage[prior_cols + ['Capture_Efficiency']].notna().all(axis=1)
    df_stage = df_stage[valid_mask].reset_index(drop=True)

    def parse_probe_id(pid):
        parts = pid.split('_')
        return parts[1], int(parts[2]), int(parts[3])

    coords = df_stage['Probe_ID'].apply(parse_probe_id)
    df_stage['_chrom'] = [c[0] for c in coords]
    df_stage['_c_start'] = [c[1] for c in coords]
    df_stage['_c_end'] = [c[2] for c in coords]

    df_map = pd.read_csv(args.probe_mapping)
    map_lookup = {}
    for _, row in df_map.iterrows():
        key = (row['Chromosome'], int(row['Centered_Start']), int(row['Centered_End']))
        map_lookup[key] = (row['Probe_Name'], int(row['Probe_Start']),
                           int(row['Probe_End']))

    probe_names, probe_starts, probe_ends, chroms = [], [], [], []
    for _, row in df_stage.iterrows():
        key = (row['_chrom'], row['_c_start'], row['_c_end'])
        if key in map_lookup:
            pn, ps, pe = map_lookup[key]
            probe_names.append(pn)
            probe_starts.append(ps)
            probe_ends.append(pe)
            chroms.append(row['_chrom'])
        else:
            probe_names.append(None)
            probe_starts.append(0)
            probe_ends.append(0)
            chroms.append(row['_chrom'])

    probe_df = pd.DataFrame({
        'Probe_Name': probe_names,
        'Chromosome': chroms,
        'Probe_Start': probe_starts,
        'Probe_End': probe_ends,
    })
    print(f"  Probes matched: {(probe_df['Probe_Name'].notna()).sum()}/{len(probe_df)}")

    # ================================================================
    # Analysis 1: Replicate Spearman at probe positions
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 1: REPLICATE SPEARMAN AT PROBE POSITIONS")
    print(f"{'='*70}")
    print(f"  Sampling {args.replicate_sample} probes for BAM depth comparison...")
    print(f"  (Full 344K would take too long; 20K gives stable Spearman estimate)")

    print(f"\n  Computing depth from NIST7086 BAM...")
    depth_7086, sampled_df = compute_depth_at_probes(
        args.bam_7086, probe_df, sample_size=args.replicate_sample)

    print(f"  Computing depth from NIST7035 BAM...")
    depth_7035, _ = compute_depth_at_probes(
        args.bam_7035, sampled_df)

    # Filter out zero-depth positions
    valid = (depth_7086 > 0) & (depth_7035 > 0)
    d86 = depth_7086[valid]
    d35 = depth_7035[valid]
    print(f"  Valid probe positions (both BAMs > 0): {valid.sum()}/{len(valid)}")

    rep_spearman = None
    rep_pearson = None
    if valid.sum() > 50:
        rep_spearman, rep_p = stats.spearmanr(d86, d35)
        rep_pearson, _ = stats.pearsonr(d86, d35)
        print(f"\n  Between-replicate Spearman (at probe positions): {rep_spearman:.4f}")
        print(f"  Between-replicate Pearson  (at probe positions): {rep_pearson:.4f}")
        print(f"  (Compare: model Spearman on these probes = 0.489)")
        print(f"  (Compare: sliding-window replicate Spearman = 0.5924)")

        ratio = 0.489 / rep_spearman if rep_spearman > 0 else float('inf')
        print(f"\n  Model achieves {ratio:.1%} of probe-level replicate ceiling")
        if ratio > 0.8:
            print(f"  → Model is near the noise ceiling at probe positions")
        elif ratio > 0.6:
            print(f"  → Model captures most of the reproducible signal")
        else:
            print(f"  → Room for improvement exists")
    else:
        print(f"  [WARN] Too few valid positions for Spearman")

    # ================================================================
    # Analysis 2: Top-25 / Bottom-25 genomic distribution
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 2: TOP-25 / BOTTOM-25 GENOMIC DISTRIBUTION")
    print(f"{'='*70}")

    # Run model inference
    print(f"\n  Running model inference...")
    model, prior_mean, prior_std = load_model(args.checkpoint, device)
    dataset = PanMolecularProbeDataset(args.h5, prior_mean=prior_mean, prior_std=prior_std)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False,
                        num_workers=8, pin_memory=True, persistent_workers=True)
    y_pred, y_true = predict_all(model, loader, device)

    for label, indices in [("TOP-25", np.argsort(y_pred)[-25:]),
                           ("BOTTOM-25", np.argsort(y_pred)[:25])]:
        chroms_sel = probe_df.iloc[indices]['Chromosome'].values
        chrom_counts = Counter(chroms_sel)
        n_chroms = len(chrom_counts)
        print(f"\n  {label}:")
        print(f"    Spread across {n_chroms} chromosomes")
        for ch in sorted(chrom_counts.keys(),
                         key=lambda c: (0, int(c.replace('chr', '')))
                         if c.replace('chr', '').isdigit() else (1, c)):
            print(f"      {ch}: {chrom_counts[ch]} probes")

        # Show actual efficiency stats
        true_eff = y_true[indices]
        pred_eff = y_pred[indices]
        print(f"    Predicted efficiency: mean={pred_eff.mean():.3f}, SD={pred_eff.std():.3f}")
        print(f"    Actual efficiency:    mean={true_eff.mean():.3f}, SD={true_eff.std():.3f}")

    # Check: are top and bottom overlapping in chromosomes?
    top_chroms = set(probe_df.iloc[np.argsort(y_pred)[-25:]]['Chromosome'])
    bot_chroms = set(probe_df.iloc[np.argsort(y_pred)[:25]]['Chromosome'])
    shared = top_chroms & bot_chroms
    print(f"\n  Chromosomes shared between Top and Bottom: {len(shared)}")
    if len(top_chroms) >= 5 and len(bot_chroms) >= 5:
        print(f"  → Both groups are genomically distributed (not clustered)")
    else:
        print(f"  → WARNING: One group may be clustered — check probe selection diversity")

    # ================================================================
    # Paper supplement. Replicate consistency at probe positions.
    # JSON: results/json/replicate_consistency.json
    # CSV : results/tables/replicate_consistency.csv
    # ================================================================
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    payload = {
        'bam_7086': os.path.abspath(args.bam_7086),
        'bam_7035': os.path.abspath(args.bam_7035),
        'n_sampled_probes': int(args.replicate_sample),
        'n_valid_positions': int(valid.sum()),
        'between_replicate_spearman_at_probes': float(rep_spearman) if rep_spearman is not None else None,
        'between_replicate_pearson_at_probes': float(rep_pearson) if rep_pearson is not None else None,
    }
    with open(os.path.join(JSON_DIR, 'replicate_consistency.json'), 'w') as fh:
        json.dump(payload, fh, indent=2)

    pd.DataFrame([{
        'N_valid_positions': int(valid.sum()),
        'Replicate_Spearman_at_probes': round(float(rep_spearman), 4) if rep_spearman is not None else None,
        'Replicate_Pearson_at_probes': round(float(rep_pearson), 4) if rep_pearson is not None else None,
    }]).to_csv(os.path.join(TABLES_DIR, 'replicate_consistency.csv'), index=False)

    print(f"\n  JSON: {os.path.join(JSON_DIR, 'replicate_consistency.json')}")
    print(f"  CSV : {os.path.join(TABLES_DIR, 'replicate_consistency.csv')}")

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
