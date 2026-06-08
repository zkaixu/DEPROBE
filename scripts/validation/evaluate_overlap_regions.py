#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: Overlapping Region Cross-Platform Evaluation
=========================================================
Evaluates model predictions ONLY at genomic positions covered by
BOTH the source and target capture kits. This isolates the true
cross-platform transferability from the confounding effect of
non-overlapping BED regions.

If Spearman is much higher in the overlap region than on the full
dataset, it means the model HAS learned transferable biophysical
features, but the full-dataset evaluation was diluted by positions
unique to one platform.

Usage:
    python evaluate_overlap_regions.py \
        --checkpoint <model.pth> \
        --source_h5 <source.h5> --source_bed <source.bed> \
        --target_h5 <target.h5> --target_bed <target.bed>
"""

import os
import sys
import argparse
import numpy as np
import torch
import h5py
from scipy import stats
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'model')
sys.path.insert(0, MODEL_DIR)

from dataset import PanMolecularProbeDataset
from model import DEPROBE
from torch.utils.data import DataLoader


# ====================================================================
# BED Overlap Logic
# ====================================================================

def load_bed_intervals(bed_path):
    """Load BED file as dict of chrom -> sorted list of (start, end)."""
    intervals = defaultdict(list)
    with open(bed_path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('track'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            chrom = parts[0].replace('chr', '')
            start, end = int(parts[1]), int(parts[2])
            intervals[chrom].append((start, end))
    for c in intervals:
        intervals[c].sort()
    return dict(intervals)


def build_overlap_set(bed1, bed2):
    """
    Find overlapping genomic intervals between two BED files.
    Returns a set of (chrom, overlap_start, overlap_end) tuples.
    """
    overlaps = []
    for chrom in bed1:
        if chrom not in bed2:
            continue
        list2 = bed2[chrom]
        j = 0
        for s1, e1 in bed1[chrom]:
            while j < len(list2) and list2[j][1] <= s1:
                j += 1
            k = j
            while k < len(list2) and list2[k][0] < e1:
                s2, e2 = list2[k]
                ov_start = max(s1, s2)
                ov_end = min(e1, e2)
                if ov_end > ov_start:
                    overlaps.append((chrom, ov_start, ov_end))
                k += 1
    return overlaps


def extract_probe_positions_from_h5(h5_path):
    """
    Extract (chrom, probe_start) for each probe from Probe_ID in the
    staging CSV. Since H5 doesn't store Probe_ID, we reconstruct
    positions from the sequence data by reading the original CSV.

    Fallback: use the H5 row index as-is (positions not recoverable).
    """
    # H5 files don't store Probe_ID. We need to reconstruct from
    # the staging CSV if available, or use a different approach.
    # For now, return None to signal that position-based filtering
    # needs the staging CSV.
    return None


def find_probes_in_overlap(csv_path, overlap_intervals):
    """
    Given a staging CSV with Probe_ID column, find which row indices
    fall within the overlap intervals.

    Probe_ID format: region_{chrom}_{start}_{end}_len{L}_pos{w_start}
    """
    import pandas as pd

    overlap_set = defaultdict(list)
    for chrom, start, end in overlap_intervals:
        overlap_set[chrom].append((start, end))
    for c in overlap_set:
        overlap_set[c].sort()

    indices = []
    reader = pd.read_csv(csv_path, usecols=['Probe_ID'], chunksize=500000)
    current_idx = 0

    for chunk in reader:
        for pid in chunk['Probe_ID']:
            pid_str = str(pid)
            try:
                if '_pos' in pid_str and '_len' in pid_str:
                    w_start = int(pid_str.rsplit('_pos', 1)[1])
                    base = pid_str.rsplit('_len', 1)[0]
                    l_str = pid_str.rsplit('_len', 1)[1].split('_')[0]
                    probe_len = int(l_str)
                    if base.startswith('region_'):
                        chrom = base[7:].split('_')[0].replace('chr', '')
                    else:
                        current_idx += 1
                        continue

                    # Check if this probe falls within any overlap interval
                    if chrom in overlap_set:
                        for ov_s, ov_e in overlap_set[chrom]:
                            if ov_s > w_start + probe_len:
                                break
                            if w_start >= ov_s and w_start + probe_len <= ov_e:
                                indices.append(current_idx)
                                break
            except Exception:
                pass
            current_idx += 1

    return np.array(indices, dtype=np.int64)


# ====================================================================
# Model Inference
# ====================================================================

def load_model(checkpoint_path, device):
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


def predict_at_indices(model, dataset, indices, device, batch_size=2048):
    """Run inference only at specified indices."""
    all_preds = []
    all_labels = []

    # Process in batches
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_items = [dataset[int(i)] for i in batch_idx]

        # Collate
        anchor = torch.stack([b['anchor'] for b in batch_items]).to(device)
        mask = torch.stack([b['anchor_mask'] for b in batch_items]).to(device)
        priors = torch.stack([b['priors'] for b in batch_items]).to(device)
        mod = torch.stack([b['modality'] for b in batch_items]).to(device)
        eff = torch.stack([b['efficiency'] for b in batch_items])

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                _, pred, _ = model(anchor, priors, mod, pad_mask=mask, alpha=0.0)

        all_preds.append(pred.squeeze().cpu().numpy())
        all_labels.append(eff.numpy())

        if (start // batch_size) % 50 == 0:
            print(f"    Processed {start + len(batch_idx):,} / {len(indices):,} probes", flush=True)

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DEPROBE-DNA: Cross-Platform Evaluation on Overlapping BED Regions"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_h5", required=True)
    parser.add_argument("--source_bed", required=True)
    parser.add_argument("--source_csv", required=True,
                        help="Staging CSV with Probe_ID column for source")
    parser.add_argument("--target_h5", required=True)
    parser.add_argument("--target_bed", required=True)
    parser.add_argument("--target_csv", required=True,
                        help="Staging CSV with Probe_ID column for target")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  DEPROBE-DNA: OVERLAPPING REGION CROSS-PLATFORM EVALUATION")
    print("=" * 70)

    # 1. Load BED files and compute overlap
    print("\n  Loading BED files...")
    source_bed = load_bed_intervals(args.source_bed)
    target_bed = load_bed_intervals(args.target_bed)

    overlaps = build_overlap_set(source_bed, target_bed)
    total_overlap_bp = sum(e - s for _, s, e in overlaps)
    print(f"  Overlap regions: {len(overlaps):,} intervals, {total_overlap_bp:,} bp ({total_overlap_bp/1e6:.1f} Mbp)")

    if total_overlap_bp == 0:
        print("  [FATAL] No overlapping regions found. Check BED chromosome naming.")
        sys.exit(1)

    # 2. Find probes within overlap regions
    print("\n  Finding source probes in overlap regions...")
    source_idx = find_probes_in_overlap(args.source_csv, overlaps)
    print(f"  Source probes in overlap: {len(source_idx):,}")

    print("  Finding target probes in overlap regions...")
    target_idx = find_probes_in_overlap(args.target_csv, overlaps)
    print(f"  Target probes in overlap: {len(target_idx):,}")

    if len(source_idx) == 0 or len(target_idx) == 0:
        print("  [FATAL] No probes found in overlap regions.")
        sys.exit(1)

    # 3. Load model
    print("\n  Loading model...")
    model, prior_mean, prior_std = load_model(args.checkpoint, device)

    # 4. Load datasets
    source_dataset = PanMolecularProbeDataset(args.source_h5, prior_mean=prior_mean, prior_std=prior_std)
    target_dataset = PanMolecularProbeDataset(args.target_h5, prior_mean=prior_mean, prior_std=prior_std)

    # 5. Predict on overlap probes
    print("\n  Predicting on SOURCE overlap probes...")
    src_pred, src_true = predict_at_indices(model, source_dataset, source_idx, device)

    print("  Predicting on TARGET overlap probes...")
    tgt_pred, tgt_true = predict_at_indices(model, target_dataset, target_idx, device)

    # 6. Compute metrics
    src_spearman, _ = stats.spearmanr(src_pred, src_true)
    src_pearson, _ = stats.pearsonr(src_pred, src_true)
    src_mse = np.mean((src_pred - src_true) ** 2)

    tgt_spearman, _ = stats.spearmanr(tgt_pred, tgt_true)
    tgt_pearson, _ = stats.pearsonr(tgt_pred, tgt_true)
    tgt_mse = np.mean((tgt_pred - tgt_true) ** 2)

    # 7. Also get full-dataset metrics for comparison
    print("\n  Predicting on FULL target dataset for comparison...")
    full_tgt_dataset = PanMolecularProbeDataset(args.target_h5, prior_mean=prior_mean, prior_std=prior_std)
    full_loader = DataLoader(full_tgt_dataset, batch_size=2048, shuffle=False,
                             num_workers=4, pin_memory=True)
    full_preds, full_labels = [], []
    with torch.no_grad():
        for batch in full_loader:
            x = batch['anchor'].to(device)
            p = batch['priors'].to(device)
            m = batch['modality'].to(device)
            mask = batch['anchor_mask'].to(device)
            with torch.amp.autocast('cuda'):
                _, pred, _ = model(x, p, m, pad_mask=mask, alpha=0.0)
            full_preds.append(pred.squeeze().cpu().numpy())
            full_labels.append(batch['efficiency'].numpy())
    full_preds = np.concatenate(full_preds)
    full_labels = np.concatenate(full_labels)
    full_spearman, _ = stats.spearmanr(full_preds, full_labels)
    full_mse = np.mean((full_preds - full_labels) ** 2)

    # 8. Report
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  Source (in overlap):  {len(source_idx):,} probes")
    print(f"  Target (in overlap): {len(target_idx):,} probes")
    print(f"  Target (full):       {len(full_labels):,} probes")

    print(f"\n  {'Metric':<25s} {'Source(overlap)':>15s} {'Target(overlap)':>15s} {'Target(full)':>15s}")
    print(f"  {'-' * 70}")
    print(f"  {'MSE':<25s} {src_mse:>15.4f} {tgt_mse:>15.4f} {full_mse:>15.4f}")
    print(f"  {'Pearson r':<25s} {src_pearson:>15.4f} {tgt_pearson:>15.4f} {'--':>15s}")
    print(f"  {'Spearman rho':<25s} {src_spearman:>15.4f} {tgt_spearman:>15.4f} {full_spearman:>15.4f}")

    print(f"\n  KEY COMPARISON:")
    print(f"    Target Spearman (overlap only): {tgt_spearman:.4f}")
    print(f"    Target Spearman (full dataset): {full_spearman:.4f}")
    delta = tgt_spearman - full_spearman
    if delta > 0.05:
        print(f"    Delta: +{delta:.4f} — overlap regions show BETTER transfer")
        print(f"    → Non-overlapping regions were diluting the signal")
    elif delta < -0.05:
        print(f"    Delta: {delta:.4f} — overlap regions show WORSE transfer")
        print(f"    → Problem is fundamental, not just BED coverage mismatch")
    else:
        print(f"    Delta: {delta:.4f} — no significant difference")
        print(f"    → BED overlap is not the main factor")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
