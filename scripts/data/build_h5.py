#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: 12D HDF5 Factory (CSV -> H5 Converter)
===================================================
Converts the master CSV (from data_fusion.py) into HDF5 format
for efficient training data loading.

Contrastive learning uses Rank-N-Contrast (Zha et al., NeurIPS 2023)
which determines positive/negative pairs dynamically within each
mini-batch based on label distance. No pre-computed indices needed.
"""

import os
import h5py
import pandas as pd
import numpy as np
import argparse
import logging
import gc
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logger = logging.getLogger("H5-Factory-12D")
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
handler = TqdmLoggingHandler()
handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(handler)


PRIOR_COLS_12 = [
    'Tm', 'GC_pct', 'dG', 'Hairpin_Tm', 'Dimer_Tm', 'Yield', 'Norm_Len',
    'GC_Skew', 'Entropy', 'dG_5p', 'dG_3p', 'Collision_Penalty'
]

PRIOR_COLS_18 = PRIOR_COLS_12 + [
    'Longest_Homopolymer', 'Trinuc_Entropy', 'Dinuc_Repeat_Frac',
    'Off_Target_Count', 'Max_Off_Target_Score', 'Mean_Off_Target_Identity'
]


def convert_to_h5(input_path, output_path, chunk_size=50000, prior_dim=12):
    prior_cols = PRIOR_COLS_18 if prior_dim == 18 else PRIOR_COLS_12
    num_priors = len(prior_cols)
    logger.info(f"Starting {num_priors}D H5 Factory: {os.path.basename(input_path)}")

    with open(input_path, 'r') as _f:
        total_expected = sum(1 for _ in _f) - 1

    logger.info(f"Total rows: {total_expected} | Priors: {num_priors}D")

    # auto-create the parent directory tree if missing.
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset('sequences', (0,), maxshape=(None,), dtype='S120', compression="lzf")
        h5f.create_dataset('priors', (0, num_priors), maxshape=(None, num_priors), dtype='float32', compression="lzf")
        h5f.create_dataset('efficiency', (0,), maxshape=(None,), dtype='float32', compression="lzf")
        h5f.create_dataset('modalities', (0,), maxshape=(None,), dtype='int8', compression="lzf")
        h5f.create_dataset('platforms', (0,), maxshape=(None,), dtype='int8', compression="lzf")

        reader = pd.read_csv(input_path, chunksize=chunk_size)
        total_rows = 0
        pbar = tqdm(total=total_expected, desc="12D H5 Conversion", colour='cyan')

        for chunk in reader:
            batch_size = len(chunk)
            new_size = total_rows + batch_size

            h5f['sequences'].resize((new_size,))
            h5f['priors'].resize((new_size, num_priors))
            h5f['efficiency'].resize((new_size,))
            h5f['modalities'].resize((new_size,))
            h5f['platforms'].resize((new_size,))

            h5f['sequences'][total_rows:new_size] = chunk['Sequence'].values.astype('S120')
            h5f['priors'][total_rows:new_size] = chunk[prior_cols].values.astype(np.float32)
            h5f['efficiency'][total_rows:new_size] = chunk['Capture_Efficiency'].values.astype(np.float32)
            h5f['modalities'][total_rows:new_size] = chunk['Modality'].values.astype(np.int8)
            h5f['platforms'][total_rows:new_size] = chunk['Platform'].values.astype(np.int8)

            total_rows = new_size
            pbar.update(batch_size)
            gc.collect()

    logger.info("=" * 60)
    logger.info("12D Master H5 Baked Successfully.")
    logger.info(f"Total probes: {total_rows} | Dimensions: {num_priors}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEPROBE-DNA: 12D H5 Factory")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ref", required=False, help="Kept for CLI compatibility (unused)")
    parser.add_argument("--neg_pool", type=int, default=5, help="Kept for CLI compatibility (unused)")
    parser.add_argument("--chunk", type=int, default=50000)
    parser.add_argument("--prior_dim", type=int, default=12, choices=[12, 18],
                        help="Prior dimension: 12 (original) or 18 (extended)")
    args = parser.parse_args()
    convert_to_h5(args.input, args.output, args.chunk, prior_dim=args.prior_dim)
