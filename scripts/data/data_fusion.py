#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: 12D data fusion and alignment.

Joins sequence, prior, and efficiency tables on Probe_ID with a strict 1:1
inner join, producing the master per-sample table consumed downstream.

Features carried through (12D): Tm, GC_pct, dG, Hairpin_Tm, Dimer_Tm, Yield,
Norm_Len, GC_Skew, Entropy, dG_5p, dG_3p, Collision_Penalty.

Notes:
    - Explicit gc.collect() after each merge to release intermediate dataframes.
    - Output format routes to .to_parquet() or .to_csv() based on extension.
"""

import pandas as pd
import sys
import argparse
import logging
import os
import gc

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Data-Fusion-12D")


def fuse_12d_dataset(
        eff_file: str,
        prior_file: str,
        platform_id: int,
        mode: str,
        offtarget_file: str = None
) -> pd.DataFrame:
    """
    Executes the multi-modal fusion logic with memory-efficient 1:1 joining.
    """
    try:
        logger.info(f"Loading Priors binary stream: {prior_file}")
        df_priors = pd.read_parquet(prior_file)
        logger.info(f"Statistics: Prior_Rows={len(df_priors)}")
    except Exception as e:
        logger.error(f"Critical I/O Failure during Priors load: {e}")
        sys.exit(1)

    logger.info("Executing Memory-Optimized Strict Inner Join (1:1 Pixel-Perfect)...")

    if not eff_file:
        logger.info("[INFERENCE MODE] No Efficiency file provided. Injecting Dummy targets (0.0).")
        df_merged = df_priors.copy()
        df_merged['Capture_Efficiency'] = 0.0
    else:
        try:
            logger.info(f"Loading Efficiency binary stream: {eff_file}")
            df_eff = pd.read_parquet(eff_file)
        except Exception as e:
            logger.error(f"Critical I/O Failure during Efficiency load: {e}")
            sys.exit(1)

        if 'Efficiency' in df_eff.columns:
            df_eff = df_eff.rename(columns={'Efficiency': 'Capture_Efficiency'})

        cols_to_keep = ['Probe_ID', 'Capture_Efficiency']
        eff_subset = df_eff[cols_to_keep]

        logger.info("Executing Memory-Optimized Strict Inner Join (1:1 Pixel-Perfect)...")
        df_merged = pd.merge(df_priors, eff_subset, on='Probe_ID', how='inner')

        del df_eff, df_priors, eff_subset
        gc.collect()

    if df_merged.empty:
        logger.error("Join Collision: No matches found. Ensure calc_efficiency and extract_sequences use identical window params.")
        sys.exit(1)

    # Merge off-target features if provided.
    if offtarget_file:
        logger.info(f"Loading off-target features: {offtarget_file}")
        try:
            df_ot = pd.read_parquet(offtarget_file) if offtarget_file.endswith('.parquet') else pd.read_csv(offtarget_file)
            df_merged = pd.merge(df_merged, df_ot[['Probe_ID', 'Off_Target_Count',
                                                     'Max_Off_Target_Score', 'Mean_Off_Target_Identity']],
                                  on='Probe_ID', how='left')
            # Fill NaN for probes without off-target data
            for col in ['Off_Target_Count', 'Max_Off_Target_Score', 'Mean_Off_Target_Identity']:
                df_merged[col] = df_merged[col].fillna(0)
            del df_ot
            gc.collect()
            logger.info(f"Off-target features merged. {(df_merged['Off_Target_Count'] > 0).sum()} probes have off-target hits.")
        except Exception as e:
            logger.error(f"Failed to load off-target features: {e}")
            sys.exit(1)

    # Apply Metadata Tags for Domain Adversarial Training (DANN)
    df_merged['Platform'] = pd.Series(platform_id, index=df_merged.index, dtype='int8')
    df_merged['Modality'] = pd.Series(1 if mode == 'rna' else 0, index=df_merged.index, dtype='int8')

    # Column Ordering (12D base + optional extended features)
    core_columns = [
        'Probe_ID', 'Sequence', 'Capture_Efficiency',
        # Original 12D
        'Tm', 'GC_pct', 'dG', 'Hairpin_Tm', 'Dimer_Tm', 'Yield', 'Norm_Len',
        'GC_Skew', 'Entropy', 'dG_5p', 'dG_3p', 'Collision_Penalty',
        # 3 complexity (from calc_priors --extended)
        'Longest_Homopolymer', 'Trinuc_Entropy', 'Dinuc_Repeat_Frac',
        # 3 off-target (from calc_offtarget)
        'Off_Target_Count', 'Max_Off_Target_Score', 'Mean_Off_Target_Identity',
        'Platform', 'Modality'
    ]

    available_cols = [c for c in core_columns if c in df_merged.columns]
    return df_merged[available_cols]


def main():
    parser = argparse.ArgumentParser(description="DEPROBE-DNA 12D data fusion.")
    parser.add_argument("--eff", required=False, help="Path to normalized efficiency parquet")
    parser.add_argument("--prior", required=True, help="Path to physical priors parquet")
    parser.add_argument("--out", required=True, help="Path to output master CSV/Parquet")
    parser.add_argument("--mode", choices=['dna', 'rna'], default='dna')
    parser.add_argument("--platform", type=int, default=0, help="Platform ID")
    parser.add_argument("--offtarget", required=False, default=None,
                        help="Path to off-target features parquet (optional).")

    args = parser.parse_args()

    dim_label = "extended" if args.offtarget else "12D"
    logger.info(f"--- {dim_label} Fusion Engine Initiated (Mode: {args.mode.upper()}) ---")

    final_dataset = fuse_12d_dataset(args.eff, args.prior, args.platform, args.mode,
                                      offtarget_file=args.offtarget)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Flashing {len(final_dataset)} synchronized samples to {args.out}...")

    if args.out.endswith('.parquet'):
        final_dataset.to_parquet(args.out, index=False, compression='snappy')
    else:
        final_dataset.to_csv(args.out, index=False)

    logger.info("--- Fusion Successfully Completed ---")


if __name__ == "__main__":
    main()
