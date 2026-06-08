#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: H5 Dataset Verification Suite
==========================================
Pre-training checks on H5 files.
Verifies structure, dimensions, data integrity, label quality,
and cross-file consistency for DANN training readiness.

Usage:
    python3 verify_h5.py
"""

import sys
import os
import h5py
import numpy as np

# ====================================================================
# Configuration
# ====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
H5_FILES = {
    "Train (Nextera 7086)":     f"{PROJECT_ROOT}/data/data_factory/final/train/deprobe_train_master.h5",
    "Val (Nextera 7035)":       f"{PROJECT_ROOT}/data/data_factory/final/val/deprobe_val_master.h5",
    "DANN TruSeq":              f"{PROJECT_ROOT}/data/data_factory/final/dann/truseq/deprobe_truseq_master.h5",
}

EXPECTED_DATASETS = ['sequences', 'priors', 'efficiency', 'modalities', 'platforms']
VALID_PRIOR_DIMS = [12, 18]
PRIOR_NAMES_12 = [
    'Tm', 'GC_pct', 'dG', 'Hairpin_Tm', 'Dimer_Tm', 'Yield', 'Norm_Len',
    'GC_Skew', 'Entropy', 'dG_5p', 'dG_3p', 'Collision_Penalty'
]
PRIOR_NAMES_18 = PRIOR_NAMES_12 + [
    'Longest_Homopolymer', 'Trinuc_Entropy', 'Dinuc_Repeat_Frac',
    'Off_Target_Count', 'Max_Off_Target_Score', 'Mean_Off_Target_Identity'
]


# ====================================================================
# Per-File Checks
# ====================================================================

def check_file(name, path):
    """Run all checks on a single H5 file. Returns (pass_count, fail_count, warnings)."""
    passed, failed, warnings = 0, 0, []

    def ok(msg):
        nonlocal passed
        passed += 1
        print(f"    [PASS] {msg}")

    def fail(msg):
        nonlocal failed
        failed += 1
        print(f"    [FAIL] {msg}")

    def warn(msg):
        warnings.append(msg)
        print(f"    [WARN] {msg}")

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"  {path}")
    print(f"{'=' * 70}")

    # --- 1. File existence & readability ---
    if not os.path.exists(path):
        fail(f"File not found: {path}")
        return passed, failed, warnings

    file_mb = os.path.getsize(path) / 1e6
    print(f"    File size: {file_mb:.1f} MB")

    try:
        f = h5py.File(path, 'r')
    except Exception as e:
        fail(f"Cannot open H5: {e}")
        return passed, failed, warnings

    # --- 2. Dataset presence ---
    keys = list(f.keys())
    for ds in EXPECTED_DATASETS:
        if ds in keys:
            ok(f"Dataset '{ds}' present")
        else:
            fail(f"Dataset '{ds}' MISSING")

    if 'neg_indices' in keys:
        warn("'neg_indices' found — should have been removed (RNC doesn't need it)")

    unexpected = set(keys) - set(EXPECTED_DATASETS) - {'neg_indices'}
    if unexpected:
        warn(f"Unexpected datasets: {unexpected}")

    # --- 3. Shape consistency ---
    n = f['sequences'].shape[0]
    print(f"    Total probes: {n:,}")

    if n == 0:
        fail("Dataset is EMPTY (0 rows)")
        f.close()
        return passed, failed, warnings
    else:
        ok(f"Non-empty: {n:,} rows")

    for ds in ['efficiency', 'modalities', 'platforms']:
        if ds in keys:
            if f[ds].shape[0] == n:
                ok(f"'{ds}' row count matches ({n:,})")
            else:
                fail(f"'{ds}' row count mismatch: {f[ds].shape[0]} vs expected {n}")

    # --- 4. Priors dimensionality ---
    if 'priors' in keys:
        prior_shape = f['priors'].shape
        prior_dim = prior_shape[1] if len(prior_shape) == 2 else 0
        if prior_dim in VALID_PRIOR_DIMS:
            ok(f"Priors shape: {prior_shape} ({prior_dim}D)")
        else:
            fail(f"Priors shape: {prior_shape} (expected dim in {VALID_PRIOR_DIMS})")

        if prior_shape[0] != n:
            fail(f"Priors row count mismatch: {prior_shape[0]} vs {n}")

        # Select correct prior names for this file
        PRIOR_NAMES = PRIOR_NAMES_18 if prior_dim == 18 else PRIOR_NAMES_12

    # --- 5. Data type checks ---
    if f['sequences'].dtype.kind == 'S':
        ok(f"Sequences dtype: {f['sequences'].dtype} (byte string)")
    else:
        fail(f"Sequences dtype: {f['sequences'].dtype} (expected byte string S*)")

    if 'priors' in keys and f['priors'].dtype == np.float32:
        ok("Priors dtype: float32")
    elif 'priors' in keys:
        warn(f"Priors dtype: {f['priors'].dtype} (expected float32)")

    # --- 6. Sequence content check ---
    n_sample = min(100, n)
    sample_seqs = [f['sequences'][i] for i in range(n_sample)]
    bad_seqs = 0
    empty_seqs = 0
    lengths = []

    for s in sample_seqs:
        decoded = s.decode('ascii').rstrip('\x00')
        lengths.append(len(decoded))
        if len(decoded) == 0:
            empty_seqs += 1
        else:
            for c in decoded:
                if c not in 'ACGTN':
                    bad_seqs += 1
                    break

    if empty_seqs == 0:
        ok(f"No empty sequences in sample (first {n_sample})")
    else:
        fail(f"{empty_seqs} empty sequences in first {n_sample} rows")

    if bad_seqs == 0:
        ok("All sampled sequences contain only A/C/G/T/N")
    else:
        warn(f"{bad_seqs} sequences with unexpected characters")

    unique_lens = sorted(set(lengths))
    print(f"    Sequence lengths in sample: {unique_lens}")

    # --- 7. Efficiency label analysis ---
    eff = f['efficiency'][:]
    eff_min, eff_max = eff.min(), eff.max()
    eff_mean, eff_std = eff.mean(), eff.std()
    eff_nan = np.isnan(eff).sum()
    eff_inf = np.isinf(eff).sum()

    print(f"    Efficiency: min={eff_min:.4f}, max={eff_max:.4f}, "
          f"mean={eff_mean:.4f}, std={eff_std:.4f}")

    if eff_nan > 0:
        fail(f"Efficiency has {eff_nan} NaN values!")
    else:
        ok("No NaN in efficiency")

    if eff_inf > 0:
        fail(f"Efficiency has {eff_inf} Inf values!")
    else:
        ok("No Inf in efficiency")

    if eff_std < 0.01:
        fail(f"Efficiency std={eff_std:.6f} — nearly constant, model can't learn")
    elif eff_std < 0.05:
        warn(f"Efficiency std={eff_std:.4f} — low variance, training may be difficult")
    else:
        ok(f"Efficiency variance healthy (std={eff_std:.4f})")

    eff_zeros = (eff == 0).sum()
    eff_ones = (eff >= 0.999).sum()
    zero_pct = eff_zeros / n * 100
    if zero_pct > 50:
        warn(f"Efficiency: {zero_pct:.1f}% are exactly 0 (heavy class imbalance)")
    else:
        print(f"    Efficiency: {zero_pct:.1f}% zeros, {eff_ones / n * 100:.1f}% near 1.0")

    # --- 8. Priors sanity checks ---
    if 'priors' in keys:
        priors = f['priors'][:]
        nan_count = np.isnan(priors).sum()
        inf_count = np.isinf(priors).sum()

        if nan_count > 0:
            fail(f"Priors have {nan_count} NaN values!")
            nan_cols = np.isnan(priors).any(axis=0)
            for i, has_nan in enumerate(nan_cols):
                if has_nan:
                    print(f"        NaN in column {i} ({PRIOR_NAMES[i]})")
        else:
            ok("No NaN in priors")

        if inf_count > 0:
            fail(f"Priors have {inf_count} Inf values!")
        else:
            ok("No Inf in priors")

        # Per-column statistics
        print(f"\n    {'Prior':<20s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s} {'Zeros%':>8s}")
        print(f"    {'-' * 68}")
        for i, col_name in enumerate(PRIOR_NAMES):
            col = priors[:, i]
            z_pct = (col == 0).sum() / n * 100
            print(f"    {col_name:<20s} {col.min():>10.3f} {col.max():>10.3f} "
                  f"{col.mean():>10.3f} {col.std():>10.3f} {z_pct:>7.1f}%")

            if col.std() == 0:
                if col_name == 'Norm_Len':
                    warn(f"Prior '{col_name}' is constant (expected for fixed 120bp probes)")
                else:
                    fail(f"Prior '{col_name}' is constant (std=0) — provides no information")

        del priors

    # --- 9. Platform / Modality checks ---
    if 'platforms' in keys:
        plats = f['platforms'][:]
        unique_plats = np.unique(plats)
        print(f"\n    Platform IDs: {unique_plats.tolist()}")
        if len(unique_plats) == 1:
            ok(f"Single platform: {unique_plats[0]}")
        else:
            warn(f"Multiple platforms in one H5: {unique_plats.tolist()}")

    if 'modalities' in keys:
        mods = f['modalities'][:]
        unique_mods = np.unique(mods)
        print(f"    Modality IDs: {unique_mods.tolist()}")

    # --- 10. Efficiency distribution histogram ---
    hist, edges = np.histogram(eff, bins=10, range=(0, 1))
    print(f"\n    Efficiency Distribution:")
    max_bar = 40
    max_count = hist.max() if hist.max() > 0 else 1
    for i in range(len(hist)):
        bar_len = int(hist[i] / max_count * max_bar)
        pct = hist[i] / n * 100
        print(f"    [{edges[i]:.1f}-{edges[i+1]:.1f}] {'#' * bar_len} {hist[i]:>8,} ({pct:.1f}%)")

    f.close()
    del eff
    return passed, failed, warnings


# ====================================================================
# Cross-File Consistency Checks
# ====================================================================

def cross_file_checks():
    """Verify consistency across all H5 files for DANN training."""
    print(f"\n{'=' * 70}")
    print(f"  CROSS-FILE CONSISTENCY CHECKS")
    print(f"{'=' * 70}")

    dims = {}
    plats = {}
    row_counts = {}

    for name, path in H5_FILES.items():
        if not os.path.exists(path):
            print(f"    [SKIP] {name}: file not found")
            continue
        with h5py.File(path, 'r') as f:
            if 'priors' in f:
                dims[name] = f['priors'].shape[1]
            if 'platforms' in f:
                plats[name] = np.unique(f['platforms'][:]).tolist()
            row_counts[name] = f['sequences'].shape[0]

    # All files should have same prior dimension
    unique_dims = set(dims.values())
    if len(unique_dims) == 1:
        print(f"    [PASS] All H5 files have consistent prior dimension: {unique_dims.pop()}D")
    elif len(unique_dims) > 1:
        print(f"    [FAIL] Inconsistent prior dimensions: {dims}")
    else:
        print(f"    [SKIP] No files to compare")

    # Platform IDs should be distinct across DANN domains
    print()
    for name, p in plats.items():
        cnt = row_counts.get(name, 0)
        print(f"    {name}: Platform={p}, Rows={cnt:,}")

    # Check that DANN target platforms differ from each other and from source
    # Train and Val SHOULD share the same platform (same source domain)
    dann_plat_ids = set()
    for name, p in plats.items():
        if "DANN" in name:
            dann_plat_ids.update(p)

    source_plat_ids = set()
    for name, p in plats.items():
        if "Train" in name or "Val" in name:
            source_plat_ids.update(p)

    overlap = dann_plat_ids & source_plat_ids
    if not overlap and len(dann_plat_ids) == len([p for n, p in plats.items() if "DANN" in n]):
        print(f"    [PASS] DANN platform IDs {dann_plat_ids} are distinct from source {source_plat_ids}")
    elif overlap:
        print(f"    [FAIL] DANN shares platform IDs {overlap} with source — DANN has no signal!")

    # Train and Val should have the same platform
    train_plat = plats.get("Train (Nextera 7086)", [])
    val_plat = plats.get("Val (Nextera 7035)", [])
    if train_plat and val_plat:
        if train_plat == val_plat:
            print(f"    [PASS] Train and Val share Platform={train_plat} (same source domain)")
        else:
            print(f"    [WARN] Train Platform={train_plat} vs Val Platform={val_plat}")

    # DANN targets should differ from source
    for dann_name in ["DANN TruSeq"]:
        dann_plat = plats.get(dann_name, [])
        if dann_plat and train_plat:
            if dann_plat != train_plat:
                print(f"    [PASS] {dann_name} Platform={dann_plat} differs from Source={train_plat}")
            else:
                print(f"    [FAIL] {dann_name} Platform={dann_plat} same as Source — DANN has no signal!")


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  DEPROBE-DNA: H5 DATASET VERIFICATION SUITE")
    print("=" * 70)

    total_passed, total_failed, total_warnings = 0, 0, 0

    for name, path in H5_FILES.items():
        p, f, w = check_file(name, path)
        total_passed += p
        total_failed += f
        total_warnings += len(w)

    cross_file_checks()

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"    Passed:   {total_passed}")
    print(f"    Failed:   {total_failed}")
    print(f"    Warnings: {total_warnings}")

    if total_failed == 0:
        print(f"\n    READY FOR TRAINING.")
    else:
        print(f"\n    {total_failed} FAILURE(S) DETECTED — fix before training.")

    sys.exit(total_failed)
