#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: probe design inference
===================================
DNA probe discovery using the trained 12D physics-informed model. Takes
target regions as input and outputs ranked probe candidates with
risk-adjusted scores.

Components:
    - 12D physics priors (GC skew, Shannon entropy, 5' / 3' dG, etc.) via
      primer3.
    - sys.path resolution for cross-directory module imports.
    - Risk-adjusted scoring balances predicted efficiency against
      thermodynamic safety constraints.
    - Batch inference for large candidate pools.
    - Attention heatmap generation for per-probe explanation.
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
from typing import List, Optional, Tuple
from datasketch import MinHash, MinHashLSH

# ====================================================================
# [SYSTEM] Dynamic Dependency Resolution
# ====================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(SCRIPTS_ROOT, 'model')
DATA_DIR = os.path.join(SCRIPTS_ROOT, 'data')

sys.path.append(SCRIPTS_ROOT)
sys.path.append(MODEL_DIR)
sys.path.append(DATA_DIR)

try:
    from model import DEPROBE
    from calc_priors import compute_advanced_priors
except ImportError as e:
    print(f"[CRITICAL] Missing DEPROBE core modules. Check directory structure: {e}")
    sys.exit(1)

# ====================================================================
# Logging and physics constraints
# ====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AI-Oracle-DNA")

BASE_TO_IDX = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0, 'PAD': 0}

QC_CONFIG = {
    'hairpin_limit': 40.0,
    'dimer_limit': 35.0,
    'yield_target': 0.85,
    'penalty_scale': 0.1,
}


# ====================================================================
# [ENGINE] AI Oracle Master Class
# ====================================================================
class AIOracleEngine:
    def __init__(self, weights_path: str, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Initializing 12D AI Oracle on device: {self.device}")

        self.model = DEPROBE(
            num_platforms=10,
            prior_dim=12,
            num_modalities=5,
            d_model=256
        ).to(self.device)

        self._load_weights(weights_path)
        self.model.eval()

    def _load_weights(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights asset not found: {path}")

        logger.info(f"Loading neural backbone from: {path}")
        ckpt = torch.load(path, map_location=self.device)

        state_dict = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(state_dict, strict=False)

        # Load training-time normalization statistics (critical for correct inference)
        if 'prior_mean' in ckpt and 'prior_std' in ckpt:
            self.prior_mu = ckpt['prior_mean'].float().to(self.device)
            self.prior_sigma = ckpt['prior_std'].float().to(self.device)
            self.prior_sigma[self.prior_sigma == 0] = 1.0
            logger.info(f"Loaded prior normalization stats from checkpoint.")
        else:
            logger.warning("Checkpoint missing prior_mean/prior_std. Raw priors will be used (UNRELIABLE).")
            self.prior_mu = None
            self.prior_sigma = None

        logger.info("Neural backbone successfully instantiated and verified.")

    def _prepare_batch_tensors(self, seq_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_indices = []
        batch_masks = []

        for seq in seq_list:
            indices = [BASE_TO_IDX.get(b.upper(), 0) for b in seq]
            actual_len = len(indices)

            if actual_len < 120:
                indices += [BASE_TO_IDX['PAD']] * (120 - actual_len)
            else:
                indices = indices[:120]

            mask = torch.zeros(120, dtype=torch.bool)
            if actual_len < 120:
                mask[actual_len:] = True

            batch_indices.append(indices)
            batch_masks.append(mask)

        return (
            torch.tensor(batch_indices, dtype=torch.long).to(self.device),
            torch.stack(batch_masks).to(self.device)
        )

    def _calculate_soft_physics_penalty(self, row: pd.Series) -> float:
        penalty = 0.0
        if row['Hairpin_Tm'] > QC_CONFIG['hairpin_limit']:
            diff = row['Hairpin_Tm'] - QC_CONFIG['hairpin_limit']
            penalty += QC_CONFIG['penalty_scale'] * (diff ** 1.5)

        if row['Dimer_Tm'] > QC_CONFIG['dimer_limit']:
            diff = row['Dimer_Tm'] - QC_CONFIG['dimer_limit']
            penalty += QC_CONFIG['penalty_scale'] * (diff ** 1.5)

        if row['Yield'] < QC_CONFIG['yield_target']:
            penalty += 2.0 * (QC_CONFIG['yield_target'] - row['Yield'])

        return penalty

    @torch.no_grad()
    def execute_design(self, target_raw_seq: str, min_len: int = 60, max_len: int = 120, step: int = 2,
                       batch_size: int = 2048) -> pd.DataFrame:
        candidate_pool = []
        logger.info(f"Scanning target region for DNA candidates...")

        for l_idx in range(min_len, max_len + 1, 10):
            for s_idx in range(0, len(target_raw_seq) - l_idx + 1, step):
                candidate_pool.append({
                    'Start': s_idx,
                    'End': s_idx + l_idx,
                    'Length': l_idx,
                    'Sequence': target_raw_seq[s_idx: s_idx + l_idx]
                })

        if not candidate_pool:
            return pd.DataFrame()

        df = pd.DataFrame(candidate_pool)

        logger.info(f"Computing 12D Physics Engine for {len(df)} candidates (parallel)...")
        num_workers = max(1, os.cpu_count() - 2)
        seqs = df['Sequence'].tolist()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            physics_res = list(executor.map(compute_advanced_priors, seqs, chunksize=500))
        priors_11d = np.array(physics_res)

        logger.info("Computing 12th Dimension: Spatial Autocorrelation (LSH)...")
        lsh = MinHashLSH(threshold=0.7, num_perm=64)
        mhs = []
        for seq in df['Sequence']:
            m = MinHash(num_perm=64)
            for j in range(len(seq) - 5):
                m.update(seq[j:j + 5].encode('utf8'))
            mhs.append(m)

        for i, m in enumerate(mhs):
            lsh.insert(f"c_{i}", m)

        collisions = np.array([float(len(lsh.query(m))) for m in mhs]).reshape(-1, 1)
        priors_12d = np.concatenate([priors_11d, collisions], axis=1)

        logger.info(f"Executing Batch Neural Inference (Batch Size: {batch_size})...")
        all_raw_scores = []

        for i in range(0, len(df), batch_size):
            batch_chunk = df.iloc[i: i + batch_size]
            b_priors = torch.tensor(priors_12d[i: i + batch_size], dtype=torch.float32).to(self.device)

            # Z-score normalization using training statistics (matches training pipeline)
            if self.prior_mu is not None and self.prior_sigma is not None:
                b_priors = (b_priors - self.prior_mu) / self.prior_sigma
                b_priors = torch.clamp(b_priors, -10.0, 10.0)

            indices, masks = self._prepare_batch_tensors(batch_chunk['Sequence'].tolist())

            mods = torch.full((len(batch_chunk),), 0, dtype=torch.long).to(self.device)
            _, pred_eff, _ = self.model(indices, b_priors, mods, pad_mask=masks, alpha=0.0)

            all_raw_scores.extend(pred_eff.squeeze(-1).cpu().numpy().tolist())

        df['AI_Raw_Score'] = all_raw_scores

        cols_11d = ['Tm', 'GC_pct', 'dG', 'Hairpin_Tm', 'Dimer_Tm', 'Yield',
                    'Norm_Len', 'GC_Skew', 'Entropy', 'dG_5p', 'dG_3p']
        for i, col in enumerate(cols_11d):
            df[col] = priors_11d[:, i]

        df['Collision_Score'] = collisions

        logger.info("Fusing AI neural scores with physical safety constraints...")
        df['Physics_Penalty'] = df.apply(self._calculate_soft_physics_penalty, axis=1)
        df['Final_Score'] = df['AI_Raw_Score'] - df['Physics_Penalty']

        return df.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)

    def apply_diversity_filter(self, df: pd.DataFrame, top_n: int = 10, min_dist: int = 40) -> pd.DataFrame:
        if df.empty: return df
        logger.info(f"Applying spatial diversity filter (min_dist={min_dist}bp)...")
        selected = []
        used_starts = []
        lsh = MinHashLSH(threshold=0.7, num_perm=64)

        for _, row in df.iterrows():
            if len(selected) >= top_n:
                break

            if any(abs(row['Start'] - s) < min_dist for s in used_starts):
                continue

            m = MinHash(num_perm=64)
            for j in range(len(row['Sequence']) - 5):
                m.update(row['Sequence'][j:j + 5].encode('utf8'))

            if len(lsh.query(m)) > 0:
                continue

            selected.append(row)
            used_starts.append(row['Start'])
            lsh.insert(f"final_{len(selected)}", m)

        return pd.DataFrame(selected).reset_index(drop=True)

    # ====================================================================
    # Attention heatmap generator
    # ====================================================================
    def generate_attention_heatmap(self, row: pd.Series, output_path: str):
        """Runs a single probe through the model with hook enabled to extract and plot attention."""
        if 'matplotlib' not in sys.modules or 'seaborn' not in sys.modules:
            logger.warning("Skipping visualization: matplotlib or seaborn not installed.")
            return

        logger.info(f"Generating DEPROBE-Lens Attention Map -> {output_path}")
        seq = row['Sequence']

        # 1. Reconstruct 12D Priors from the row dataframe
        priors_12d = [
            row['Tm'], row['GC_pct'], row['dG'], row['Hairpin_Tm'], row['Dimer_Tm'],
            row['Yield'], row['Norm_Len'], row['GC_Skew'], row['Entropy'],
            row['dG_5p'], row['dG_3p'], row['Collision_Score']
        ]
        priors_tensor = torch.tensor([priors_12d], dtype=torch.float32).to(self.device)
        indices, masks = self._prepare_batch_tensors([seq])
        mods = torch.tensor([0], dtype=torch.long).to(self.device)

        # 2. Attach Hook and Forward Pass
        self.model.register_attention_hook(layer_idx=0)
        with torch.no_grad():
            self.model(indices, priors_tensor, mods, pad_mask=masks, alpha=0.0)

        attn_matrix = self.model.attention_weights
        self.model.remove_attention_hook()

        if attn_matrix is None:
            logger.error("Failed to capture attention weights.")
            return

        # Extract the (L, L) matrix for the single batch item
        # PyTorch returns shape (Batch, Target, Source)
        attn_matrix = attn_matrix.squeeze(0).numpy()

        # 3. Dynamic Axis Labels (Map pooling logic)
        # Because of MaxPool1d(kernel=2, stride=2), 120bp becomes 60 tokens.
        valid_tokens = len(seq) // 2

        # Clip the matrix to remove padding noise (Valid length + 1 Physics Token)
        valid_matrix = attn_matrix[:valid_tokens + 1, :valid_tokens + 1]

        # Generate sequence labels (e.g. 'AT', 'GC') to match the 1D-CNN receptive field
        labels = ["[PHYSICS]"] + [seq[i:i + 2] for i in range(0, len(seq) - 1, 2)]

        # Failsafe alignment (in case of odd lengths)
        labels = labels[:valid_tokens + 1]

        # 4. Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(valid_matrix, xticklabels=labels, yticklabels=labels,
                    cmap="mako", annot=False, cbar_kws={'label': 'Attention Weight'})

        plt.title(
            f"DEPROBE-Lens: Spatial & Thermodynamic Cross-Attention\nScore: {row['Final_Score']:.3f} | Len: {len(seq)}bp")
        plt.xlabel("Key (Source Sequence & Physics Token)")
        plt.ylabel("Query (Target Sequence & Physics Token)")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        # Defensive: re-ensure parent directory exists at write time.
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()


# ====================================================================
# [CLI] Terminal Interface
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="DEPROBE-DNA: 12D AI Oracle Engine")
    parser.add_argument("-s", "--seq", required=True, help="Target DNA raw sequence string")
    parser.add_argument("-w", "--weights", required=True, help="Absolute path to .pth model weights")
    parser.add_argument("--top_n", type=int, default=10, help="Number of final probes to output")
    parser.add_argument("--out", default="output/probe_design_manifest.csv", help="Output CSV path")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate DEPROBE-Lens attention heatmaps for the top probe")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    oracle = AIOracleEngine(args.weights)
    raw_ranked = oracle.execute_design(args.seq)

    if raw_ranked.empty:
        logger.error("No candidates met the minimum discovery criteria.")
        return

    final_probes = oracle.apply_diversity_filter(raw_ranked, top_n=args.top_n)

    print(f"\n{'=' * 75}")
    print(f" DEPROBE-DNA 2.0 AI ORACLE DESIGN REPORT (12D ARCHITECTURE)")
    print(f"{'=' * 75}")
    print(f" Target Length:      {len(args.seq)} bp")
    print(f" Candidates Scanned: {len(raw_ranked)}")
    print(f" Panel Size:         {len(final_probes)} probes")
    print(f" Top Panel Score:    {final_probes.iloc[0]['Final_Score']:.4f}")
    print(f" Top Probe Safety:   Hairpin Tm {final_probes.iloc[0]['Hairpin_Tm']:.1f}°C | Yield {final_probes.iloc[0]['Yield']:.2%}")
    print(f"{'=' * 75}\n")

    # Defensive: re-ensure parent directory exists at write time.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    final_probes.to_csv(args.out, index=False)
    logger.info(f"Design manifest written to: {args.out}")

    # --- Trigger DEPROBE-viz ---
    if args.visualize and not final_probes.empty:
        vis_out_path = args.out.replace(".csv", "_top1_attention.png")
        oracle.generate_attention_heatmap(final_probes.iloc[0], vis_out_path)


if __name__ == "__main__":
    main()