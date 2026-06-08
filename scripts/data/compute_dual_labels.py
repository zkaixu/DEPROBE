#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: Dual-Label Master Builder for Matched-position Cross-Platform Diagnostic
================================================================================
Given an intersection BED (Nextera ∩ TruSeq, ~33.7 Mbp) and two BAMs
sequenced from the same DNA on two different capture kits, produces ONE
master Parquet keyed on Probe_ID with columns:

    Probe_ID, Sequence, [12 priors], <label_a_name>, <label_b_name>

Each label column is the quantile-normalised capture-efficiency at that
sliding-window position in the corresponding BAM. Labels are quantile-
normalised independently per BAM so they are directly comparable as
z-scores.

This Parquet is the input to:
  - label_correlation_diagnostic.py  (smoking-gun: Pearson/Spearman between
                                       the two labels at IDENTICAL genomic
                                       positions)
  - build_matched_bed_h5.py          (splits into two training-ready H5s,
                                       one per kit, sharing sequence+priors)

Pipeline (each step is an existing script invoked via subprocess):
    1. extract_sequences.py    : intersection BED + FASTA -> seqs.parquet
    2. calc_priors.py          : seqs.parquet            -> priors.parquet
    3. calc_efficiency.py × 2  : intersection BED + each BAM -> eff_<tag>.parquet
    4. inline merge            : join priors + 2 eff parquets on Probe_ID

Usage:
    python compute_dual_labels.py \\
        --bed   data/beds/nextera_truseq_intersection.bed \\
        --ref   data/hg19/human_g1k_v37.fasta \\
        --bam_a data/bams/NIST7086.bam --label_a_name Nextera_label \\
        --bam_b data/bams/NIST-hg001-7001-ready.bam --label_b_name TruSeq_label \\
        --out_dir data/data_factory/staging/matched_bed_dual \\
        --workers 24
"""

import os
import sys
import argparse
import subprocess
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("dual-labels")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))


def run(cmd: list[str]):
    """Defensive subprocess wrapper: log + check return code."""
    logger.info("+ " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        logger.error(f"Subprocess failed (exit {proc.returncode}): {' '.join(cmd)}")
        sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Build dual-label master Parquet for Matched-position cross-platform diagnostic.")
    parser.add_argument("--bed", required=True,
                        help="Intersection BED (e.g. Nextera ∩ TruSeq), already produced by bedtools intersect.")
    parser.add_argument("--ref", required=True,
                        help="Reference FASTA (e.g. human_g1k_v37.fasta).")

    parser.add_argument("--bam_a", required=True,
                        help="First BAM (e.g. NIST7086 Nextera).")
    parser.add_argument("--label_a_name", default="Nextera_label",
                        help="Column name for the first kit's label (default: Nextera_label).")

    parser.add_argument("--bam_b", required=True,
                        help="Second BAM (e.g. NIST-hg001-7001 TruSeq).")
    parser.add_argument("--label_b_name", default="TruSeq_label",
                        help="Column name for the second kit's label (default: TruSeq_label).")

    parser.add_argument("--out_dir", required=True,
                        help="Directory for intermediate parquets + final dual_labels.parquet.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Workers for calc_priors.py.")

    parser.add_argument("--min_len", type=int, default=120)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()

    # Defensive: auto-create the output directory tree.
    os.makedirs(args.out_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Path layout for intermediate artefacts
    # ----------------------------------------------------------------
    seqs_pq    = os.path.join(args.out_dir, "seqs.parquet")
    priors_pq  = os.path.join(args.out_dir, "priors.parquet")
    eff_a_pq   = os.path.join(args.out_dir, f"eff_{args.label_a_name}.parquet")
    eff_b_pq   = os.path.join(args.out_dir, f"eff_{args.label_b_name}.parquet")
    final_pq   = os.path.join(args.out_dir, "dual_labels.parquet")

    data_scripts = SCRIPT_DIR  # extract_sequences / calc_priors / calc_efficiency live here

    # ----------------------------------------------------------------
    # Step 1: sliding-window sequence extraction on the intersection BED
    # ----------------------------------------------------------------
    logger.info("Step 1/4 — extract_sequences.py on intersection BED")
    run([
        sys.executable, os.path.join(data_scripts, "extract_sequences.py"),
        "-f", args.ref, "-b", args.bed, "-o", seqs_pq,
        "--min_len", str(args.min_len), "--max_len", str(args.max_len), "--step", str(args.step),
    ])

    # ----------------------------------------------------------------
    # Step 2: 12D thermodynamic priors on the extracted sequences
    # ----------------------------------------------------------------
    logger.info("Step 2/4 — calc_priors.py (12D mode, no --extended flag)")
    run([
        sys.executable, os.path.join(data_scripts, "calc_priors.py"),
        "-i", seqs_pq, "-o", priors_pq,
        "--mode", "dna", "--workers", str(args.workers),
    ])

    # ----------------------------------------------------------------
    # Step 3: capture-efficiency labels, once per BAM, identical BED
    # ----------------------------------------------------------------
    logger.info(f"Step 3a/4 — calc_efficiency.py with BAM A ({os.path.basename(args.bam_a)})")
    run([
        sys.executable, os.path.join(data_scripts, "calc_efficiency.py"),
        "-b", args.bam_a, "-d", args.bed, "-o", eff_a_pq,
        "--min_len", str(args.min_len), "--max_len", str(args.max_len), "--step", str(args.step),
    ])

    logger.info(f"Step 3b/4 — calc_efficiency.py with BAM B ({os.path.basename(args.bam_b)})")
    run([
        sys.executable, os.path.join(data_scripts, "calc_efficiency.py"),
        "-b", args.bam_b, "-d", args.bed, "-o", eff_b_pq,
        "--min_len", str(args.min_len), "--max_len", str(args.max_len), "--step", str(args.step),
    ])

    # ----------------------------------------------------------------
    # Step 4: inline merge of priors + eff_a + eff_b on Probe_ID
    # ----------------------------------------------------------------
    logger.info("Step 4/4 — merging priors + dual labels on Probe_ID")
    df_priors = pd.read_parquet(priors_pq)
    df_a = pd.read_parquet(eff_a_pq)[['Probe_ID', 'Capture_Efficiency']].rename(
        columns={'Capture_Efficiency': args.label_a_name})
    df_b = pd.read_parquet(eff_b_pq)[['Probe_ID', 'Capture_Efficiency']].rename(
        columns={'Capture_Efficiency': args.label_b_name})

    logger.info(f"  priors: {len(df_priors):,} rows | "
                f"eff_a: {len(df_a):,} | eff_b: {len(df_b):,}")

    merged = df_priors.merge(df_a, on='Probe_ID', how='inner') \
                      .merge(df_b, on='Probe_ID', how='inner')

    # Drop any row whose either label is NaN (rare edge case, zero coverage in one BAM)
    before = len(merged)
    merged = merged.dropna(subset=[args.label_a_name, args.label_b_name]).reset_index(drop=True)
    logger.info(f"  merged: {before:,} rows -> after dropna: {len(merged):,} rows")

    if len(merged) == 0:
        logger.error("Merged dataframe is empty — no Probe_IDs in common across the three parquets.")
        sys.exit(1)

    # Defensive directory creation right before write.
    os.makedirs(os.path.dirname(final_pq) or '.', exist_ok=True)
    merged.to_parquet(final_pq, compression='snappy', index=False)

    logger.info("=" * 60)
    logger.info(f"Dual-label master written: {final_pq}")
    logger.info(f"  shape: {merged.shape}")
    logger.info(f"  columns: {list(merged.columns)}")
    logger.info(f"  label_a stats ({args.label_a_name}): "
                f"mean={merged[args.label_a_name].mean():.4f}, std={merged[args.label_a_name].std():.4f}")
    logger.info(f"  label_b stats ({args.label_b_name}): "
                f"mean={merged[args.label_b_name].mean():.4f}, std={merged[args.label_b_name].std():.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
