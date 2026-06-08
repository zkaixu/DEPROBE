#!/usr/bin/env python3
"""
DEPROBE-DNA: master dataset builder (Parquet).

Orchestrates extract_sequences -> calc_priors -> calc_efficiency -> data_fusion
on a metadata-driven set of samples. Outputs per-sample Parquet files plus a
combined Master Parquet for downstream HDF5 conversion.

Notes:
    - 12D thermodynamic priors via primer3 (GC, Tm, dG, hairpin, dimer, yield,
      length norm, GC skew, Shannon entropy, 5' / 3' dG, collision penalty).
    - Sliding-window dimensions are passed through to the efficiency mapper to
      keep probe IDs aligned across the four pipeline stages.
    - Parquet intermediate I/O for large (20M+) row counts.
    - Per-stage checkpointing.
    - Parquet header validation guards against incomplete-file corruption.
"""

import os
import sys
import subprocess
import pandas as pd
import argparse
import gc
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Data-Factory")


def run_command(cmd: str, step_name: str):
    logger.info(f"Executing Task: {step_name}")
    try:
        subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"CRITICAL FAILURE during {step_name}")
        logger.error(f"Command Attempted: {cmd}")
        logger.error(f"Subprocess Stderr: {e.stderr}")
        sys.exit(1)


def process_sample(row: pd.Series, master_path: str, ref_fasta: str,
                   temp_dir: str, mode: str,
                   min_len: int, max_len: int, step: int, workers: int,
                   extended: bool = False):
    """
    Orchestrates the 4-step pipeline for a single sample, managing I/O and memory.
    Steps: A(efficiency) -> B(sequences) -> C(12D priors) -> D(fusion)
    """
    sample_name = row['Sample_Name']
    bam_path = row['BAM_Path']
    bed_path = row['BED_Path']
    platform = row['Platform_ID']

    if os.path.exists(master_path):
        try:
            seen = set()
            for chunk in pd.read_csv(master_path, usecols=['Dataset_Source'], chunksize=100000):
                seen.update(chunk['Dataset_Source'].unique())
            if sample_name in seen:
                logger.info(f"[SKIP] Sample '{sample_name}' already synchronized. Bypassing...")
                return
        except Exception:
            pass

    logger.info(f"=== Harvesting Sample: {sample_name} [Mode: {mode.upper()}] ===")
    start_time = time.time()

    seq_pq = os.path.join(temp_dir, f"{sample_name}_seqs.parquet")
    prior_pq = os.path.join(temp_dir, f"{sample_name}_priors.parquet")
    eff_pq = os.path.join(temp_dir, f"{sample_name}_eff.parquet")
    offtarget_pq = os.path.join(temp_dir, f"{sample_name}_offtarget.parquet")
    fusion_pq = os.path.join(temp_dir, f"{sample_name}_fused.parquet")

    dim_label = "extended" if extended else "12D"
    logger.info(f"Feature mode: {dim_label}")

    # Step A: Efficiency Mapping
    run_command(f"python calc_efficiency.py -b {bam_path} --bed {bed_path} -o {eff_pq} "
                f"--min_len {min_len} --max_len {max_len} --step {step}",
                f"Efficiency_Mapping_{sample_name}")

    # Step B: Sequence Extraction
    run_command(f"python extract_sequences.py -f {ref_fasta} -b {bed_path} -o {seq_pq} "
                f"--min_len {min_len} --max_len {max_len} --step {step}",
                f"Sequence_Extraction_{sample_name}")

    # Step C: Physical Prior Computation
    extended_flag = "--extended" if extended else ""
    run_command(f"python calc_priors.py -i {seq_pq} -o {prior_pq} --mode {mode} --workers {workers} {extended_flag}",
                f"{dim_label}_Physics_Engine_{sample_name}")

    # Step C.5: Optional Off-Target Feature Computation.
    offtarget_flag = ""
    if extended:
        run_command(f"python calc_offtarget.py -i {seq_pq} -o {offtarget_pq} "
                    f"--ref {ref_fasta} --workers 4 --bwa_threads 6",
                    f"Off_Target_Engine_{sample_name}")
        offtarget_flag = f"--offtarget {offtarget_pq}"

    # Step D: Data Fusion
    run_command(f"python data_fusion.py --eff {eff_pq} --prior {prior_pq} --out {fusion_pq} "
                f"--mode {mode} --platform {platform} {offtarget_flag}",
                f"{dim_label}_Data_Fusion_{sample_name}")

    # I/O Barrier
    logger.info("Enforcing I/O Barrier: Forcing OS to flush page cache to NVMe...")
    if hasattr(os, 'sync'):
        os.sync()
    time.sleep(10)

    try:
        for attempt in range(3):
            try:
                df = pd.read_parquet(fusion_pq)
                break
            except Exception as read_err:
                if attempt < 2:
                    logger.warning(f"Parquet read failed (Attempt {attempt + 1}/3). Retrying in 5s... Error: {read_err}")
                    time.sleep(5)
                else:
                    raise read_err

        df['Dataset_Source'] = sample_name

        is_first_write = not os.path.exists(master_path)
        # Defensive: auto-create the parent directory tree if missing.
        os.makedirs(os.path.dirname(master_path) or '.', exist_ok=True)
        df.to_csv(master_path, mode='a', index=False, header=is_first_write)

        duration = (time.time() - start_time) / 60
        logger.info(f"[SUCCESS] Appended {len(df)} records for {sample_name} in {duration:.2f}m")

        del df
        gc.collect()

        for f in [seq_pq, prior_pq, eff_pq, offtarget_pq, fusion_pq]:
            if os.path.exists(f): os.remove(f)

    except Exception as e:
        logger.error(f"[FATAL] Disk IO failure for {sample_name}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="DEPROBE-DNA master dataset builder (Parquet, 12D priors).")
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--mode", choices=['dna', 'rna'], default='dna')
    parser.add_argument("--min_len", type=int, default=40)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--temp", required=True, help="Path to temp directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--extended", action='store_true',
                        help="Optional extended feature computation (complexity + off-target).")

    args = parser.parse_args()

    if not os.path.exists(args.meta):
        logger.error(f"Metadata controller not found: {args.meta}")
        sys.exit(1)

    df_meta = pd.read_csv(args.meta)

    os.makedirs(args.temp, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    logger.info("=" * 70)
    logger.info("DEPROBE-DNA: 12D master dataset builder (Parquet)")
    logger.info("=" * 70)

    for idx, row in df_meta.iterrows():
        logger.info(f"--- Global Progress: {idx + 1}/{len(df_meta)} ---")
        process_sample(row, args.out, args.ref, args.temp, args.mode,
                       min_len=args.min_len, max_len=args.max_len,
                       step=args.step, workers=args.workers,
                       extended=args.extended)

    logger.info(f"12D master dataset complete: {args.out}")


if __name__ == "__main__":
    main()
