#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: 12D thermodynamic prior calculator.

Computes 12 sequence-derived priors per probe candidate using primer3:
GC content, melting temperature, free energy (dG), hairpin Tm, dimer Tm,
predicted yield, normalized length, GC skew, Shannon entropy, 5' / 3' dG,
and collision penalty.

Notes:
    - Chunk-based processing in 2M-row blocks to bound memory.
    - Localized LSH for per-chunk collision penalty.
    - primer3 C backend for all thermodynamic computations.
"""

import argparse
import os
import sys
import math
import re
import itertools
import pandas as pd
import numpy as np
from collections import Counter
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
import primer3
from datasketch import MinHash, MinHashLSH
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import gc


# ====================================================================
# Parallel Workers
# ====================================================================

def _minhash_worker(seq_str):
    m = MinHash(num_perm=64)
    s = str(seq_str).upper()
    for j in range(len(s) - 5):
        m.update(s[j:j + 5].encode('utf8'))
    return m


def _physics_worker(args):
    seq, mode, temp, salt = args
    return compute_advanced_priors(seq, mode, temp, salt)


# ====================================================================
# Thermodynamic priors via primer3
# ====================================================================

def get_env_params(mode='dna', custom_temp=None, custom_salt=None):
    params = {'temp': 65.0, 'na_salt': 0.75}
    if custom_temp is not None: params['temp'] = custom_temp
    if custom_salt is not None: params['na_salt'] = custom_salt
    return params


def _is_valid_tm(tm_value):
    """
    Filter invalid Tm values returned by primer3's NN model.

    primer3's calc_hairpin/calc_homodimer can return sentinel values
    (e.g., 120000+) when the nearest-neighbor model fails to converge
    on edge-case sequences. These are not real temperatures.

    Physical upper bound: a 60bp stem with 100% GC at 750mM Na+ gives
    Tm ≈ 102°C (theoretical maximum). Anything above this is an artifact.
    """
    return 0 < tm_value < 120


def _sliding_thermo(seq_str, na_mM, window=60, step=30):
    """
    Compute thermodynamic properties on overlapping sub-windows for
    sequences exceeding primer3's 60bp NN-model limit.

    For dG: returns the MOST NEGATIVE value (worst-case binding stability).
    For hairpin/dimer Tm: returns the HIGHEST valid Tm (worst-case secondary
    structure). Invalid primer3 returns are discarded, not clamped.

    This is biologically conservative. Any local region forming strong
    secondary structure or stable self-dimer will impair the entire probe.
    """
    seq_len = len(seq_str)
    if seq_len <= window:
        comp_str = str(Seq(seq_str).complement())
        dg = primer3.calc_heterodimer(seq_str, comp_str, mv_conc=na_mM).dg / 1000.0
        hp = primer3.calc_hairpin(seq_str, mv_conc=na_mM).tm
        dm = primer3.calc_homodimer(seq_str, mv_conc=na_mM).tm
        return dg, hp if _is_valid_tm(hp) else 0.0, dm if _is_valid_tm(dm) else 0.0

    best_dg = 0.0
    best_hp_tm = 0.0
    best_dim_tm = 0.0

    for start in range(0, seq_len - window + 1, step):
        sub = seq_str[start:start + window]
        comp = str(Seq(sub).complement())
        try:
            dg = primer3.calc_heterodimer(sub, comp, mv_conc=na_mM).dg / 1000.0
            if dg < best_dg:
                best_dg = dg
        except Exception:
            pass
        try:
            hp = primer3.calc_hairpin(sub, mv_conc=na_mM).tm
            if _is_valid_tm(hp) and hp > best_hp_tm:
                best_hp_tm = hp
        except Exception:
            pass
        try:
            dm = primer3.calc_homodimer(sub, mv_conc=na_mM).tm
            if _is_valid_tm(dm) and dm > best_dim_tm:
                best_dim_tm = dm
        except Exception:
            pass

    return best_dg, best_hp_tm, best_dim_tm


def compute_advanced_priors(sequence_str, mode='dna', temp=None, salt=None):
    raw_seq = str(sequence_str).upper().replace('N', '')
    calc_seq = raw_seq[:120] if len(raw_seq) > 120 else raw_seq

    # Sane defaults (11 items; 12th collision added later)
    tm, gc_pct, dG, hp_tm, dim_tm, norm_yield, norm_len = 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0
    gc_skew, entropy, dg_5p, dg_3p = 0.0, 0.0, 0.0, 0.0

    if len(calc_seq) < 15:
        return [tm, gc_pct, dG, hp_tm, dim_tm, norm_yield, norm_len, gc_skew, entropy, dg_5p, dg_3p]

    env = get_env_params(mode, temp, salt)
    seq_obj = Seq(calc_seq)
    seq_str = str(seq_obj)

    norm_len = len(raw_seq) / 120.0
    gc_pct = gc_fraction(seq_obj) * 100.0

    na_mM = env['na_salt'] * 1000

    # --- 1. Tm (no length limit in primer3) ---
    try:
        tm = primer3.calc_tm(seq_str, mv_conc=na_mM)
    except Exception:
        pass

    # --- 2. dG, Hairpin_Tm, Dimer_Tm (sliding window for >60bp) ---
    try:
        dG, hp_tm, dim_tm = _sliding_thermo(seq_str, na_mM, window=60, step=30)
    except Exception:
        pass

    # --- 3. Yield (derived from dG) ---
    try:
        if dG < 0:
            norm_yield = 1.0 / (1.0 + (dG / -20.0))
        else:
            norm_yield = 0.5
    except Exception:
        pass

    # --- 4. GC Skew, Entropy, Positional dG ---
    try:
        counts = Counter(calc_seq)
        gs, cs = counts.get('G', 0), counts.get('C', 0)
        gc_skew = (gs - cs) / (gs + cs + 1e-6)

        kmers = [calc_seq[i:i+2] for i in range(len(calc_seq)-1)]
        if len(kmers) > 0:
            k_counts = Counter(kmers)
            total_kmers = len(kmers)
            for count in k_counts.values():
                p = count / total_kmers
                entropy -= p * math.log2(p)

        if len(calc_seq) >= 20:
            dg_5p = primer3.calc_hairpin(calc_seq[:20], mv_conc=na_mM).dg / 1000.0
            dg_3p = primer3.calc_hairpin(calc_seq[-20:], mv_conc=na_mM).dg / 1000.0
        else:
            dg_5p = primer3.calc_hairpin(calc_seq, mv_conc=na_mM).dg / 1000.0
            dg_3p = dg_5p
    except Exception:
        pass

    return [tm, gc_pct, dG, hp_tm, dim_tm, norm_yield, norm_len, gc_skew, entropy, dg_5p, dg_3p]


# ====================================================================
# Extended complexity feature computation (optional).
# ====================================================================

def compute_complexity_features(sequence_str):
    """
    Compute 3 sequence complexity features (optional extended feature variant).
    These capture non-thermodynamic sequence properties that complement the
    thermodynamic prior.

    Returns: [Longest_Homopolymer, Trinuc_Entropy, Dinuc_Repeat_Frac]
    """
    raw_seq = str(sequence_str).upper().replace('N', '')
    if len(raw_seq) < 10:
        return [0.0, 0.0, 0.0]

    # 1. Longest Homopolymer Run
    #    e.g., ATGCAAAAAT → 5 (five consecutive A's)
    max_run = max(len(list(g)) for _, g in itertools.groupby(raw_seq))

    # 2. Trinucleotide Entropy (k=3 Shannon entropy)
    #    Distinct from existing k=2 Entropy. Captures higher-order patterns
    trimers = [raw_seq[i:i+3] for i in range(len(raw_seq) - 2)]
    if len(trimers) > 0:
        tri_counts = Counter(trimers)
        total_tri = len(trimers)
        trinuc_entropy = -sum((c / total_tri) * math.log2(c / total_tri)
                              for c in tri_counts.values())
    else:
        trinuc_entropy = 0.0

    # 3. Dinucleotide Repeat Fraction
    #    Fraction of sequence occupied by ≥3 consecutive dinucleotide repeats
    #    e.g., ATATATATAT = (AT)x5 → 10 bases in repeats
    repeat_matches = re.findall(r'(([ACGT]{2})\2{2,})', raw_seq)
    repeat_bases = sum(len(m[0]) for m in repeat_matches)
    dinuc_frac = repeat_bases / len(raw_seq)

    return [float(max_run), trinuc_entropy, dinuc_frac]


def _extended_physics_worker(args):
    """Worker for 14D (11 physics + 3 complexity) computation."""
    seq, mode, temp, salt = args
    base_11 = compute_advanced_priors(seq, mode, temp, salt)
    complexity_3 = compute_complexity_features(seq)
    return base_11 + complexity_3


# ====================================================================
# Orchestration (Divide and Conquer)
# ====================================================================

def process_priors(input_file, output_file, mode='dna', temp=None, salt=None,
                    max_workers=4, extended=False):
    print(f"\n[INFO] Loading Master Stream: {input_file}")
    df = pd.read_parquet(input_file) if input_file.endswith('.parquet') else pd.read_csv(input_file)

    base_dim = 14 if extended else 11  # 11 physics + 3 complexity when extended
    dim_label = "extended (14 base + collision)" if extended else "12D (11 base + collision)"

    total_seqs = len(df)
    chunk_size = 2000000
    num_chunks = (total_seqs + chunk_size - 1) // chunk_size

    print(f"[INFO] Mode: {dim_label}")
    print(f"[INFO] Total Records: {total_seqs} | Target Chunks: {num_chunks} | Workers: {max_workers}")
    print(f"[INFO] Allocating secure memory blocks (Zero-List-Bloat)...")

    arr_11d = np.zeros((total_seqs, base_dim), dtype=np.float32)
    arr_collision = np.zeros(total_seqs, dtype=np.float32)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_seqs)
        chunk_len = end_idx - start_idx

        print(f"\n{'=' * 50}")
        print(f"[CHUNK {i + 1}/{num_chunks}] Processing rows {start_idx} to {end_idx}")
        print(f"{'=' * 50}")

        chunk_seqs = df['Sequence'].iloc[start_idx:end_idx]

        # --- Phase 1: Physics for Current Chunk ---
        args_gen = ((str(s), mode, temp, salt) for s in chunk_seqs)
        worker_fn = _extended_physics_worker if extended else _physics_worker
        phase_label = f"Physics {dim_label} C{i + 1}"
        chunk_res_11d = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for res in tqdm(executor.map(worker_fn, args_gen, chunksize=1000),
                            total=chunk_len, desc=phase_label, unit="seq"):
                chunk_res_11d.append(res)

        arr_11d[start_idx:end_idx] = chunk_res_11d
        del chunk_res_11d, args_gen
        gc.collect()

        # --- Phase 2: MinHash Signatures for Current Chunk ---
        chunk_mhs = []
        seq_gen = (str(s) for s in chunk_seqs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for mh in tqdm(executor.map(_minhash_worker, seq_gen, chunksize=2000),
                           total=chunk_len, desc=f"Hashing C{i + 1}", unit="hash"):
                chunk_mhs.append(mh)
        del seq_gen
        gc.collect()

        # --- Phase 3: LSH Indexing & Collision Query for Current Chunk ---
        lsh = MinHashLSH(threshold=0.7, num_perm=64)
        for j, m in enumerate(tqdm(chunk_mhs, desc=f"Indexing C{i + 1}", leave=False)):
            lsh.insert(f"idx_{j}", m)

        chunk_collisions = []
        for m in tqdm(chunk_mhs, desc=f"Querying C{i + 1}", leave=False):
            chunk_collisions.append(float(len(lsh.query(m))))

        arr_collision[start_idx:end_idx] = chunk_collisions

        del chunk_mhs, chunk_collisions, lsh
        gc.collect()

    # --- Data Fusion & Storage ---
    print(f"\n[INFO] Stitching final matrices into Dataframe...")
    cols_base = [
        'Tm', 'GC_pct', 'dG', 'Hairpin_Tm', 'Dimer_Tm', 'Yield', 'Norm_Len',
        'GC_Skew', 'Entropy', 'dG_5p', 'dG_3p'
    ]
    if extended:
        cols_base += ['Longest_Homopolymer', 'Trinuc_Entropy', 'Dinuc_Repeat_Frac']
    df[cols_base] = arr_11d
    df['Collision_Penalty'] = arr_collision

    del arr_11d, arr_collision
    gc.collect()

    print(f"[INFO] Flashing Master Dataset to Disk...")
    # Defensive: auto-create the parent directory tree if missing.
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    if output_file.endswith('.parquet'):
        df.to_parquet(output_file, compression='snappy', index=False)
    else:
        df.to_csv(output_file, index=False)

    total_dim = len(cols_base) + 1  # +1 for Collision_Penalty
    print(f"\n{'=' * 60}\n[SUCCESS] {total_dim}D Harvest Synchronized: {len(df)} records.\n{'=' * 60}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="DEPROBE-DNA 12D thermodynamic prior calculator.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--mode", choices=['dna', 'rna'], default='dna')
    parser.add_argument("--temp", type=float)
    parser.add_argument("--salt", type=float)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of CPU cores to use")
    parser.add_argument("--extended", action='store_true',
                        help="Compute 3 additional sequence complexity features (Longest_Homopolymer, Trinuc_Entropy, Dinuc_Repeat_Frac).")

    args = parser.parse_args()

    process_priors(
        input_file=args.input,
        output_file=args.output,
        mode=args.mode,
        temp=args.temp,
        salt=args.salt,
        max_workers=args.workers,
        extended=args.extended
    )