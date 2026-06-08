#!/usr/bin/env python3
"""
DEPROBE-DNA: capture-efficiency mapper.

Counts BAM read depth over center-padded probe windows (mirrors
extract_sequences.py windowing for exact Probe_ID alignment downstream),
filters reads at MAPQ >= 20, and applies quantile normalization to produce
standard-normal capture-efficiency labels.

Notes:
    - Parameterized windowing via min_len / max_len / step.
    - Parallel BAM coverage counting across cores.
"""

import argparse
import pysam
import pandas as pd
import numpy as np
import sys
import os
import logging
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Eff-Mapper")


# ====================================================================
# Parallel Workers
# ====================================================================

def get_actual_chrom(target: str, available_refs: set) -> str:
    """Resolves chromosome naming inconsistencies between BED and BAM."""
    if target in available_refs: return target
    if target.startswith("chr") and target[3:] in available_refs: return target[3:]
    if not target.startswith("chr") and f"chr{target}" in available_refs: return f"chr{target}"
    return None


def process_bam_chunk(args):
    """
    Worker: Calculates mean depth for exact sliding/flanked windows.
    Matches extract_sequences.py coordinate logic 1:1.
    """
    bed_lines, bam_path, min_l, max_l, step = args

    # Each process must open its own BAM handle for thread-safety
    bam_file = pysam.AlignmentFile(bam_path, "rb")
    available_refs = set(bam_file.references)
    local_results = []

    for line in bed_lines:
        cols = line.strip().split('\t')
        if len(cols) < 3: continue

        raw_chrom, start, end = cols[0], int(cols[1]), int(cols[2])
        chrom = get_actual_chrom(raw_chrom, available_refs)
        if not chrom: continue

        # Always embed chrom coords in base_id for consistent Probe_ID format
        base_id = f"region_{raw_chrom}_{start}_{end}"
        region_len = end - start

        # Generate probes across all requested lengths
        lengths = range(min_l, max_l + 1, step)

        for L in lengths:
            # -------------------------------------------------------------
            # LOGIC 1: Standard Internal Sliding (For targets > probe length)
            # -------------------------------------------------------------
            if L <= region_len:
                for w_start in range(start, end - L + 1, step):
                    w_end = w_start + L
                    probe_id = f"{base_id}_len{L}_pos{w_start}"
                    try:
                        # Strict mapQ >= 20 Filtering implemented here
                        cov_arrays = bam_file.count_coverage(
                            chrom, w_start, w_end,
                            quality_threshold=15,
                            read_callback=lambda x: x.mapping_quality >= 20
                        )
                        total_bases = np.sum(cov_arrays)
                        mean_depth = total_bases / L if L > 0 else 0
                        local_results.append({'Probe_ID': probe_id, 'Raw_Depth': mean_depth})
                    except Exception:
                        continue

            # -------------------------------------------------------------
            # LOGIC 2: Center-Out Flanking Expansion (For targets < probe length)
            # -------------------------------------------------------------
            else:
                center = start + region_len // 2
                ideal_start = center - L // 2

                # Wobble logic matches sequence extractor 1:1
                wobbles = [0]
                if (L - region_len) // 2 >= step:
                    wobbles = [-step, 0, step]

                for w_offset in wobbles:
                    w_start = ideal_start + w_offset
                    w_end = w_start + L
                    probe_id = f"{base_id}_len{L}_pos{w_start}"
                    try:
                        # Strict mapQ >= 20 Filtering implemented here
                        cov_arrays = bam_file.count_coverage(
                            chrom, w_start, w_end,
                            quality_threshold=15,
                            read_callback=lambda x: x.mapping_quality >= 20
                        )
                        total_bases = np.sum(cov_arrays)
                        mean_depth = total_bases / L if L > 0 else 0
                        local_results.append({'Probe_ID': probe_id, 'Raw_Depth': mean_depth})
                    except Exception:
                        continue

    bam_file.close()
    return local_results


# ====================================================================
# Orchestrator
# ====================================================================

def calculate_capture_efficiency(bam_path, bed_path, output_file, min_len, max_len, step):
    # 1. Indexing Guard
    if not os.path.exists(bam_path + ".bai") and not os.path.exists(bam_path[:-4] + ".bai"):
        logger.info(f"Indexing BAM: {bam_path}...")
        pysam.index(bam_path)

    # 2. Parallel Distribution
    with open(bed_path, 'r') as f:
        all_lines = [l for l in f if not l.startswith(('#', 'track')) and l.strip()]

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    # Split BED into chunks
    chunk_size = max(1, len(all_lines) // (num_workers * 2))
    line_chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]

    logger.info(f"Engaging {num_workers} workers for micro-window BAM coverage mapping...")
    logger.info(f"Window Profile: {min_len}-{max_len}bp, Stride: {step}bp")
    logger.info(f"Physics Rules: Center-out flanking applied | mapQ >= 20 enforced")

    master_results = []
    worker_args = [(chunk, bam_path, min_len, max_len, step) for chunk in line_chunks]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_bam_chunk, arg) for arg in worker_args]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Mapping Depth"):
            master_results.extend(future.result())

    if not master_results:
        logger.error("No valid data extracted. Check BAM/BED overlap.")
        sys.exit(1)

    # 3. Mathematical Normalization (Unaltered Math Model)
    df = pd.DataFrame(master_results)
    logger.info(f"Applying Quantile Normalization to {len(df)} micro-regions...")

    qt = QuantileTransformer(
        n_quantiles=min(len(df), 1000),
        output_distribution='normal',
        random_state=42
    )
    raw_depths = df['Raw_Depth'].values.astype(np.float64).reshape(-1, 1)
    # Add tiny jitter to prevent transform singularity
    raw_depths += np.random.normal(0, 1e-6, raw_depths.shape)

    df['Capture_Efficiency'] = qt.fit_transform(raw_depths)
    df['Capture_Efficiency'] = df['Capture_Efficiency'].fillna(0.0)

    # 4. Binary Output
    # Defensive: auto-create the parent directory tree if missing.
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    if output_file.endswith('.parquet'):
        df.to_parquet(output_file, compression='snappy', index=False)
    else:
        df.to_csv(output_file, index=False)

    logger.info(f"[SUCCESS] Efficiency Master synced to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel BAM depth mapper with sliding-window awareness.")
    parser.add_argument("-b", "--bam", required=True)
    parser.add_argument("-d", "--bed", required=True)
    parser.add_argument("-o", "--out", required=True)

    # Adding necessary sliding window parameters
    parser.add_argument("--min_len", type=int, default=40)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--step", type=int, default=10)

    args = parser.parse_args()

    calculate_capture_efficiency(
        args.bam,
        args.bed,
        args.out,
        min_len=args.min_len,
        max_len=args.max_len,
        step=args.step
    )