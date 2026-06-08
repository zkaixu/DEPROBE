#!/usr/bin/env python3
"""
DEPROBE-DNA: sequence extractor (multi-core, Parquet).

Extracts probe-candidate sequences from a reference FASTA over BED-defined
target regions. Short targets are center-padded outward to reach the required
probe length, and wobble sliding adds micro-shifts to flanked probes to enrich
the candidate pool.

Notes:
    - Parallelized region processing across cores.
    - Parquet binary output for downstream stages.
"""

import argparse
import pysam
import pandas as pd
import os
import sys
import concurrent.futures
import multiprocessing
from tqdm import tqdm


# --- Core Physics & Normalization  ---

def get_actual_chrom(target, available_refs):
    """Resolves chromosome naming inconsistencies between BED and FASTA."""
    if target in available_refs: return target
    if target.startswith("chr") and target[3:] in available_refs: return target[3:]
    if not target.startswith("chr") and f"chr{target}" in available_refs: return f"chr{target}"
    return None


# --- Parallel Worker ---

def process_region_batch(args):
    """
    Worker function: Extracts standard sliding windows AND flanked probes.
    """
    bed_lines, fasta_path, min_l, max_l, step = args

    # Each process must open its own FastaFile handle for thread-safety
    fa = pysam.FastaFile(fasta_path)
    available_refs = set(fa.references)
    local_data = []

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
                        seq = fa.fetch(chrom, w_start, w_end).upper()
                        if 'N' in seq: continue
                        local_data.append({'Probe_ID': probe_id, 'Sequence': seq})
                    except Exception:
                        continue

            # -------------------------------------------------------------
            # LOGIC 2: Center-Out Flanking Expansion (For targets < probe length)
            # -------------------------------------------------------------
            else:
                center = start + region_len // 2
                ideal_start = center - L // 2

                # Wobble logic: create small micro-shifts (-step, 0, +step) to enrich pool,
                # ensuring the entire short target remains fully covered by the probe.
                wobbles = [0]
                if (L - region_len) // 2 >= step:
                    wobbles = [-step, 0, step]

                for w_offset in wobbles:
                    w_start = ideal_start + w_offset
                    w_end = w_start + L
                    probe_id = f"{base_id}_len{L}_pos{w_start}"
                    try:
                        seq = fa.fetch(chrom, w_start, w_end).upper()
                        if 'N' in seq: continue
                        local_data.append({'Probe_ID': probe_id, 'Sequence': seq})
                    except Exception:
                        continue

    fa.close()
    return local_data


# --- Orchestrator ---

def extract_fasta_sequences(fasta_path, bed_path, output_file,
                            min_len=40, max_len=120, step=10):
    # Indexing Check
    if not os.path.exists(fasta_path + ".fai"):
        print(f"[INFO] Indexing FASTA...")
        pysam.faidx(fasta_path)

    # Read BED and split into chunks for parallel extraction
    with open(bed_path, 'r') as f:
        all_lines = [l for l in f if not l.startswith(('#', 'track')) and l.strip()]

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    chunk_size = max(1, len(all_lines) // (num_workers * 2))
    line_chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]

    print(f"=======================================================")
    print(f" DEPROBE-DNA: Center-Out Flanking Extraction Engine")
    print(f" Workers: {num_workers} | Regions: {len(all_lines)}")
    print(f" Window: {min_len}-{max_len}bp | Step: {step}bp")
    print(f"=======================================================")

    worker_args = [(chunk, fasta_path, min_len, max_len, step)
                   for chunk in line_chunks]

    master_probe_data = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_region_batch, arg) for arg in worker_args]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting"):
            master_probe_data.extend(future.result())

    if not master_probe_data:
        print("[ERROR] No sequences extracted. Check FASTA/BED overlap.")
        sys.exit(1)

    df = pd.DataFrame(master_probe_data)
    before_dedup = len(df)
    print(f"[INFO] Deduplicating {before_dedup} candidates...")
    df = df.drop_duplicates(subset=['Sequence'])
    after_dedup = len(df)
    if before_dedup != after_dedup:
        print(f"[INFO] Deduplication removed {before_dedup - after_dedup} duplicates "
              f"({(before_dedup - after_dedup) / before_dedup * 100:.1f}% reduction)")

    # Parquet/CSV Support
    # Defensive: auto-create the parent directory tree if missing.
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    if output_file.endswith('.parquet'):
        print(f"[INFO] Exporting to Binary Parquet...")
        df.to_parquet(output_file, compression='snappy', index=False)
    else:
        df.to_csv(output_file, index=False)

    print(f"[SUCCESS] Tiled {len(df)} unique sequences to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel sequence extractor with center-padding and wobble sliding.")
    parser.add_argument("-f", "--fasta", required=True)
    parser.add_argument("-b", "--bed", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--min_len", type=int, default=40)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--step", type=int, default=10)

    args = parser.parse_args()
    extract_fasta_sequences(
        args.fasta, args.bed, args.output,
        min_len=args.min_len, max_len=args.max_len, step=args.step
    )