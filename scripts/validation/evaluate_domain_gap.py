#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: domain gap evaluator
=================================
Purpose: Quantifies the domain gap (MSE discrepancy) between
         a labelled source dataset (e.g., TruSeq) and an unlabelled
         target dataset (e.g., Nextera), using a Phase 1 pre-trained
         model. Reports Spearman rank correlation and Top 20% AUROC
         alongside MSE.

         Source-to-target statistic inheritance is enforced during
         dynamic scaling to prevent covariate-shift artifacts in
         the metric.
Location: scripts/validation/evaluate_domain_gap.py
"""

import os
import sys
import json
import argparse
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

# ==============================================================================
# Path Resolution & Dependency Injection
# ==============================================================================
# Dynamically append the model directory to sys.path to access core modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../model'))
sys.path.append(MODEL_DIR)

# Canonical Paper-1 output destinations.
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables')
JSON_DIR = os.path.join(PROJECT_ROOT, 'results', 'json')

try:
    from dataset import PanMolecularProbeDataset
    from model import DEPROBE
except ImportError as e:
    print(f"[CRITICAL] Failed to import core DEPROBE modules. Check sys.path: {e}")
    sys.exit(1)

# ==============================================================================
# Logging configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Domain-Gap-Eval")


class DomainGapEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized Evaluator on Hardware: {self.device}")

        # Reference to store Source Domain dataset for statistic inheritance
        self.source_dataset_ref = None

        # 1. Initialize Neural Architecture
        self.model = DEPROBE(num_platforms=10, prior_dim=12, num_modalities=5, d_model=256)
        self._load_weights(args.checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # 2. DataLoader Configuration (High-Throughput)
        self.loader_cfg = {
            'batch_size': args.batch_size,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4
        }

    def _load_weights(self, ckpt_path):
        """Safely loads weights handling potential DDP or Optimizer state dicts."""
        logger.info(f"Restoring Engine State from: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(state_dict, strict=False)
        logger.info("Weights restored successfully.")

    def _evaluate_domain(self, h5_path: str, domain_name: str, source_dataset=None) -> dict:
        """
        Executes high-speed inference and calculates MSE, Spearman, and AUROC.

        Args:
            h5_path (str): Path to the HDF5 dataset.
            domain_name (str): Label for logging (e.g., 'SOURCE' or 'TARGET').
            source_dataset (PanMolecularProbeDataset, optional): The loaded Source Dataset.
                If provided, Target will strictly inherit its Mean and Std for scaling.
        """
        logger.info(f"Staging {domain_name} Domain Dataset: {h5_path}")

        # ==============================================================================
        # UDA Covariate Shift Protection
        # ==============================================================================
        if source_dataset is None:
            # Mode A: Staging Source Domain. Calculate intrinsic statistics and store them.
            dataset = PanMolecularProbeDataset(h5_path=h5_path)
            self.source_dataset_ref = dataset  # Cache the dataset to extract .mu and .sigma later
        else:
            # Mode B: Staging Target Domain. STRICTLY inherit Source statistics.
            logger.info(f"Enforcing Cross-Domain Consistency: Inheriting statistics from Source...")
            dataset = PanMolecularProbeDataset(
                h5_path=h5_path,
                prior_mean=source_dataset.mu,  # Inherited Source Mean
                prior_std=source_dataset.sigma  # Inherited Source Std
            )

        loader = DataLoader(dataset, shuffle=False, **self.loader_cfg)

        all_preds = []
        all_targets = []

        # Automatic Mixed Precision for VRAM/Speed optimization
        amp_ctx = torch.amp.autocast('cuda') if self.device.type == 'cuda' else torch.amp.autocast('cpu', enabled=False)
        with torch.no_grad(), amp_ctx:
            for batch in tqdm(loader, desc=f"Scanning {domain_name}", unit="batch"):
                x_anc = batch['anchor'].to(self.device, non_blocking=True)
                priors = batch['priors'].to(self.device, non_blocking=True)
                modality = batch['modality'].to(self.device, non_blocking=True)
                pad_mask = batch['anchor_mask'].to(self.device, non_blocking=True)
                targets = batch['efficiency'].to(self.device, non_blocking=True)

                # Forward pass (alpha=0 disables GRL as we only want physics predictions)
                _, eff_pred, _ = self.model(x_anc, priors, modality, pad_mask=pad_mask, alpha=0.0)

                all_preds.append(eff_pred.view(-1).cpu().numpy())
                all_targets.append(targets.view(-1).cpu().numpy())

        # Concatenate all batches
        preds_np = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)

        # 1. MSE & MAE
        final_mse = np.mean((preds_np - targets_np) ** 2)
        final_mae = np.mean(np.abs(preds_np - targets_np))

        # 2. Correlation (Spearman for ranking, Pearson for linearity)
        spearman_corr, _ = spearmanr(preds_np, targets_np)
        pearson_corr, _ = pearsonr(preds_np, targets_np)

        # 3. Top 20% AUROC (classification power for "good probe" selection)
        threshold_80 = np.percentile(targets_np, 80)
        binary_labels = (targets_np >= threshold_80).astype(int)

        if len(np.unique(binary_labels)) > 1:
            auroc_top20 = roc_auc_score(binary_labels, preds_np)
        else:
            auroc_top20 = float('nan')

        logger.info(
            f"[{domain_name}] MSE: {final_mse:.4f} | MAE: {final_mae:.4f} | "
            f"Pearson: {pearson_corr:.4f} | Spearman: {spearman_corr:.4f} | AUROC(Top20%): {auroc_top20:.4f}")

        return {
            "MSE": final_mse,
            "MAE": final_mae,
            "Pearson": pearson_corr,
            "Spearman": spearman_corr,
            "AUROC": auroc_top20
        }

    def execute(self):
        print(f"\n{'=' * 75}")
        print(f" DEPROBE-DNA: DOMAIN GAP & RANKING POWER EVALUATION")
        print(f"{'=' * 75}\n")

        # 1. Evaluate Source Domain (This automatically stores Source Mean/Std internally)
        source_metrics = self._evaluate_domain(self.args.source_h5, "SOURCE (In-Domain)")
        print("-" * 75)

        # 2. Evaluate Target Domain (Pass the cached Source dataset to enforce scaling consistency)
        target_metrics = self._evaluate_domain(
            self.args.target_h5,
            "TARGET (Cross-Domain)",
            source_dataset=self.source_dataset_ref
        )

        print(f"\n{'=' * 75}")
        print(f" FINAL ABLATION VERDICT:")
        print(f"{'=' * 75}")
        print(f" Metric                  | SOURCE (In-Domain) | TARGET (Cross-Domain)")
        print(f"-------------------------|--------------------|----------------------")
        print(f" MSE (Global Loss)       | {source_metrics['MSE']:.4f}             | {target_metrics['MSE']:.4f}")
        print(f" MAE (Absolute Error)    | {source_metrics['MAE']:.4f}             | {target_metrics['MAE']:.4f}")
        print(f" Pearson Correlation     | {source_metrics['Pearson']:.4f}             | {target_metrics['Pearson']:.4f}")
        print(f" Spearman Rank Corr.     | {source_metrics['Spearman']:.4f}             | {target_metrics['Spearman']:.4f}")
        print(f" AUROC (Top 20% Probes)  | {source_metrics['AUROC']:.4f}             | {target_metrics['AUROC']:.4f}")
        print(f"-------------------------|--------------------|----------------------")
        print(f" Domain Gap Delta (MSE)  : {abs(target_metrics['MSE'] - source_metrics['MSE']):.4f}")
        print(f"{'=' * 75}\n")

        if target_metrics['MSE'] > source_metrics['MSE'] * 1.5:
            logger.info("Significant Domain Gap detected. Phase 2 (DANN) implementation is highly justified.")

        # ============================================================
        # Paper cross-platform table, one CSV per (source, target) pair.
        # CSV : results/tables/cross_platform_<source>_vs_<target>.csv
        # JSON: results/json/cross_platform_<source>_vs_<target>.json
        # ============================================================
        os.makedirs(TABLES_DIR, exist_ok=True)
        os.makedirs(JSON_DIR, exist_ok=True)

        src_tag = os.path.splitext(os.path.basename(self.args.source_h5))[0]
        tgt_tag = os.path.splitext(os.path.basename(self.args.target_h5))[0]
        # Strip the noisy "deprobe_*_master" prefix to keep filenames concise.
        for prefix in ('deprobe_', 'master', '_master'):
            src_tag = src_tag.replace(prefix, '')
            tgt_tag = tgt_tag.replace(prefix, '')
        src_tag = src_tag.strip('_') or 'source'
        tgt_tag = tgt_tag.strip('_') or 'target'
        pair_tag = f"{src_tag}_vs_{tgt_tag}"

        rows = []
        for label, m in (('source', source_metrics), ('target', target_metrics)):
            rows.append({
                'Domain': label,
                'H5': self.args.source_h5 if label == 'source' else self.args.target_h5,
                'MSE': round(float(m['MSE']), 4),
                'MAE': round(float(m['MAE']), 4),
                'Pearson_r': round(float(m['Pearson']), 4),
                'Spearman_rho': round(float(m['Spearman']), 4),
                'AUROC_top20pct': round(float(m['AUROC']), 4),
            })
        rows.append({
            'Domain': 'gap',
            'H5': '',
            'MSE': round(float(abs(target_metrics['MSE'] - source_metrics['MSE'])), 4),
            'MAE': round(float(abs(target_metrics['MAE'] - source_metrics['MAE'])), 4),
            'Pearson_r': round(float(target_metrics['Pearson'] - source_metrics['Pearson']), 4),
            'Spearman_rho': round(float(target_metrics['Spearman'] - source_metrics['Spearman']), 4),
            'AUROC_top20pct': round(float(target_metrics['AUROC'] - source_metrics['AUROC']), 4),
        })
        csv_path = os.path.join(TABLES_DIR, f"cross_platform_{pair_tag}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        json_path = os.path.join(JSON_DIR, f"cross_platform_{pair_tag}.json")
        with open(json_path, 'w') as fh:
            json.dump({
                'checkpoint': os.path.abspath(self.args.checkpoint),
                'source_h5': os.path.abspath(self.args.source_h5),
                'target_h5': os.path.abspath(self.args.target_h5),
                'source_metrics': {k: float(v) for k, v in source_metrics.items()},
                'target_metrics': {k: float(v) for k, v in target_metrics.items()},
            }, fh, indent=2)

        print(f"\n  Cross-platform CSV : {csv_path}")
        print(f"  Cross-platform JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Domain Gap between Source and Target datasets.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained .pth model")
    parser.add_argument("--source_h5", required=True, help="Path to the Source Domain H5 (e.g., Nextera/TruSeq)")
    parser.add_argument("--target_h5", required=True, help="Path to the Target Domain H5 (e.g., Target Platform)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Inference batch size")

    args = parser.parse_args()
    evaluator = DomainGapEvaluator(args)
    evaluator.execute()