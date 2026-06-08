#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: Matched-position Dual-Label H5 Splitter
===============================================
Reads the dual-label master Parquet produced by compute_dual_labels.py
and emits TWO HDF5 datasets, one per kit. Both H5s share IDENTICAL
sequences and priors row-for-row; only the `efficiency` field differs.

This pairing is exactly what plain Phase-2 DANN consumes:
    --data        = matched_bed_<kit_a>_master.h5  (source, labelled)
    --target_data = matched_bed_<kit_b>_master.h5  (target, labels are
                                                    ignored during DANN
                                                    training but used at
                                                    evaluation time)

Because both H5s come from the SAME genomic positions, any residual gap
the DANN cannot close is direct evidence that the gap is not covariate
shift but a fundamental incompatibility between the two label
functions.

Output schema (matches the trainer's PanMolecularProbeDataset):
    sequences  (S120)     : 120-bp probe sequence (utf-8 bytes)
    priors     (float32)  : (N, 12)
    efficiency (float32)  : (N,)
    modalities (int8)     : (N,), all zeros (dna)
    platforms  (int8)     : (N,), kit-specific tag

Usage:
    python build_matched_bed_h5.py \\
        --input  data/data_factory/staging/matched_bed_dual/dual_labels.parquet \\
        --label_a_col Nextera_label --tag_a nextera --platform_a 1 \\
        --label_b_col TruSeq_label  --tag_b truseq  --platform_b 2 \\
        --out_dir data/data_factory/final/matched_bed
"""

import os
import argparse
import logging
import h5py
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("matched-bed-h5")

PRIOR_COLS_12 = [
    'Tm', 'GC_pct', 'dG', 'Hairpin_Tm', 'Dimer_Tm', 'Yield', 'Norm_Len',
    'GC_Skew', 'Entropy', 'dG_5p', 'dG_3p', 'Collision_Penalty',
]


def write_h5(out_path: str, df: pd.DataFrame, label_col: str, platform_id: int):
    """Stream a DataFrame to a 12D-priors HDF5 file with a single label column."""
    # Defensive: ensure parent dir exists at write time.
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    n = len(df)
    logger.info(f"  -> {out_path}  (rows={n:,}, label='{label_col}', platform={platform_id})")

    with h5py.File(out_path, 'w') as h5f:
        h5f.create_dataset('sequences',  (n,),     dtype='S120',   compression='lzf')
        h5f.create_dataset('priors',     (n, 12),  dtype='float32', compression='lzf')
        h5f.create_dataset('efficiency', (n,),     dtype='float32', compression='lzf')
        h5f.create_dataset('modalities', (n,),     dtype='int8',    compression='lzf')
        h5f.create_dataset('platforms',  (n,),     dtype='int8',    compression='lzf')

        h5f['sequences'][:]  = df['Sequence'].astype(str).str.upper().values.astype('S120')
        h5f['priors'][:]     = df[PRIOR_COLS_12].values.astype(np.float32)
        h5f['efficiency'][:] = df[label_col].values.astype(np.float32)
        h5f['modalities'][:] = np.zeros(n, dtype=np.int8)              # 0 = DNA
        h5f['platforms'][:]  = np.full(n, platform_id, dtype=np.int8)


def main():
    parser = argparse.ArgumentParser(description="Split dual-label Parquet into two matched-position H5s.")
    parser.add_argument("--input", required=True,
                        help="dual_labels.parquet (output of compute_dual_labels.py)")
    parser.add_argument("--label_a_col", required=True,
                        help="Column name for kit A's label (e.g. Nextera_label).")
    parser.add_argument("--tag_a", required=True,
                        help="Short tag used in the output filename (e.g. nextera).")
    parser.add_argument("--platform_a", type=int, default=1,
                        help="Platform integer ID for kit A (default: 1).")
    parser.add_argument("--label_b_col", required=True,
                        help="Column name for kit B's label (e.g. TruSeq_label).")
    parser.add_argument("--tag_b", required=True,
                        help="Short tag used in the output filename (e.g. truseq).")
    parser.add_argument("--platform_b", type=int, default=2,
                        help="Platform integer ID for kit B (default: 2).")
    parser.add_argument("--out_dir", required=True,
                        help="Directory for matched_bed_<tag>_master.h5 outputs.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input not found: {args.input}")
        raise SystemExit(1)

    logger.info(f"Loading dual-label master: {args.input}")
    df = pd.read_parquet(args.input)
    logger.info(f"  shape={df.shape}, columns={list(df.columns)}")

    # Schema validation. Abort early if any expected column is missing.
    missing = [c for c in (['Sequence'] + PRIOR_COLS_12 + [args.label_a_col, args.label_b_col])
               if c not in df.columns]
    if missing:
        logger.error(f"Missing columns in input: {missing}")
        raise SystemExit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    out_a = os.path.join(args.out_dir, f"matched_bed_{args.tag_a}_master.h5")
    out_b = os.path.join(args.out_dir, f"matched_bed_{args.tag_b}_master.h5")

    write_h5(out_a, df, args.label_a_col, args.platform_a)
    write_h5(out_b, df, args.label_b_col, args.platform_b)

    logger.info("=" * 60)
    logger.info("Both matched-position H5s baked. Same sequences + priors, different labels.")
    logger.info(f"  source candidate (kit A): {out_a}")
    logger.info(f"  target candidate (kit B): {out_b}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
