#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiGRU Baseline: Reproducing Zhang et al. (2021) architecture on DEPROBE data.
=============================================================================
Zhang et al. used a bidirectional GRU with base-unpairing probabilities.
This baseline uses a BiGRU on the same 120bp sequences + 12D physics priors
to establish the contribution of DEPROBE's CNN-Transformer architecture
over the prior recurrent approach.

Reference: Zhang et al., Nature Communications 12:4387, 2021.

Usage:
    python baseline_bigru.py
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from scipy import stats
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PanMolecularProbeDataset

# Canonical Paper-1 output destinations.
SCRIPT_DIR_BOOTSTRAP = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR_BOOTSTRAP, '..', '..'))
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables')
JSON_DIR = os.path.join(PROJECT_ROOT, 'results', 'json')

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("BiGRU-Baseline")


class BiGRUModel(nn.Module):
    """
    Bidirectional GRU baseline inspired by Zhang et al. (2021).
    Input: nucleotide sequence + 12D physics priors.
    """
    def __init__(self, d_model=256, prior_dim=12, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=6, embedding_dim=d_model, padding_idx=0)
        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # Late fusion with physics priors
        combined_dim = d_model + prior_dim
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, priors, pad_mask=None):
        x_emb = self.embedding(x)
        output, _ = self.bigru(x_emb)

        # Masked mean pooling
        if pad_mask is not None:
            valid = (~pad_mask).float().unsqueeze(-1)
            z = (output * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)
        else:
            z = output.mean(dim=1)

        z = self.layer_norm(z)
        z_fused = torch.cat([z, priors], dim=1)
        return self.regressor(z_fused).squeeze(-1)


def top_k_precision(y_true, y_pred, k_pct):
    n = len(y_true)
    k = max(1, int(n * k_pct / 100))
    pred_top = set(np.argsort(y_pred)[-k:])
    true_top = set(np.argsort(y_true)[-k:])
    return len(pred_top & true_top) / k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val_data", default=None)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=19)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.model_dir, exist_ok=True)

    # Data
    dataset = PanMolecularProbeDataset(h5_path=args.data)
    val_size = max(1, int(len(dataset) * 0.1))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size],
                                       generator=torch.Generator().manual_seed(66))

    loader_cfg = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 4}
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, batch_size=args.batch, **loader_cfg)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=args.batch, **loader_cfg)

    ext_val_loader = None
    if args.val_data and os.path.exists(args.val_data):
        ext_set = PanMolecularProbeDataset(args.val_data, prior_mean=dataset.mu, prior_std=dataset.sigma)
        ext_val_loader = DataLoader(ext_set, shuffle=False, batch_size=args.batch, **loader_cfg)

    # Model
    model = BiGRUModel(d_model=256, prior_dim=12, num_layers=2).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"BiGRU parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # threshold=5e-4 abs prevents tracking sub-noise improvements; min_lr=1e-6 caps the floor.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
        threshold=5e-4, threshold_mode='abs', min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler('cuda')

    best_mse = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_mse = 0.0
        start = time.time()

        for i, raw in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(batch['anchor'], batch['priors'], batch['anchor_mask'])
                loss = F.huber_loss(pred, batch['efficiency'], delta=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                epoch_mse += F.mse_loss(pred, batch['efficiency']).item()

            if i % 500 == 0:
                logger.info(f"E{epoch+1} S{i} | MSE: {F.mse_loss(pred, batch['efficiency']).item():.4f}")

        # Validation
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for raw in val_loader:
                b = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
                with torch.amp.autocast('cuda'):
                    p = model(b['anchor'], b['priors'], b['anchor_mask'])
                    val_mse += F.mse_loss(p, b['efficiency']).item()
        val_mse /= len(val_loader)

        ext_mse = 0.0
        if ext_val_loader:
            with torch.no_grad():
                for raw in ext_val_loader:
                    b = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
                    with torch.amp.autocast('cuda'):
                        p = model(b['anchor'], b['priors'], b['anchor_mask'])
                        ext_mse += F.mse_loss(p, b['efficiency']).item()
            ext_mse /= len(ext_val_loader)

        scheduler.step(val_mse)
        duration = time.time() - start

        logger.info(f"Epoch [{epoch+1}/{args.epochs}] | Train MSE: {epoch_mse/len(train_loader):.4f} | "
                     f"Int Val: {val_mse:.4f} | Ext Val: {ext_mse:.4f} | Time: {duration:.1f}s")

        # Plateau-matched termination (parity with main_12d.py protocol):
        # require Int Val improvement above 5e-4 absolute threshold to count as
        # progress; otherwise patience accumulates. This matches the threshold
        # used by the LR scheduler and prevents the early-stop counter from
        # being reset by sub-noise improvements.
        if val_mse < best_mse - 5e-4:
            best_mse = val_mse
            patience_counter = 0
            # Defensive: re-ensure model_dir exists right before write.
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'best_mse': best_mse,
                         'prior_mean': dataset.mu, 'prior_std': dataset.sigma},
                        os.path.join(args.model_dir, 'bigru_best.pth'))
            logger.info(f"  >> Best: {best_mse:.4f}")
        else:
            patience_counter += 1
            current_lr_post_step = optimizer.param_groups[0]['lr']
            logger.info(f"  >> [Patience] {patience_counter}/7 epochs without "
                        f">5e-4 improvement (LR={current_lr_post_step:.2e}).")
            if patience_counter >= 7:
                logger.info(f"  Plateau-matched early stop at epoch {epoch+1}")
                break

    # Final evaluation on ext val
    if ext_val_loader:
        ckpt = torch.load(os.path.join(args.model_dir, 'bigru_best.pth'), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for raw in ext_val_loader:
                b = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
                with torch.amp.autocast('cuda'):
                    p = model(b['anchor'], b['priors'], b['anchor_mask'])
                all_preds.append(p.cpu().numpy())
                all_labels.append(b['efficiency'].cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        mse = np.mean((y_pred - y_true) ** 2)
        spearman, _ = stats.spearmanr(y_pred, y_true)
        pearson, _ = stats.pearsonr(y_pred, y_true)
        top10 = top_k_precision(y_true, y_pred, 10)

        logger.info(f"\n{'='*60}")
        logger.info(f"BiGRU FINAL EVALUATION (Ext Val)")
        logger.info(f"{'='*60}")
        logger.info(f"MSE:          {mse:.4f}")
        logger.info(f"Spearman:     {spearman:.4f}")
        logger.info(f"Pearson:      {pearson:.4f}")
        logger.info(f"Top-10%:      {top10:.1%}")
        logger.info(f"Parameters:   {param_count:,}")

        # Coerce numpy/torch floats to native Python types so json.dump
        # doesn't fail on float32 (a known historical crash for this script).
        summary = {
            'model': 'BiGRU',
            'best_int_val_mse': float(best_mse),
            'ext_val_mse': float(mse),
            'spearman': float(spearman),
            'pearson': float(pearson),
            'top10': float(top10),
            'parameters': int(param_count),
        }
        # Defensive: re-ensure model_dir exists right before write.
        os.makedirs(args.model_dir, exist_ok=True)
        with open(os.path.join(args.model_dir, 'bigru_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        # ============================================================
        # Paper Table 3. BiGRU baseline row (paired with the traditional
        # ML CSV and DEPROBE main metrics CSV).
        # CSV : results/tables/baselines_bigru.csv
        # JSON: results/json/baselines_bigru.json
        # ============================================================
        os.makedirs(TABLES_DIR, exist_ok=True)
        os.makedirs(JSON_DIR, exist_ok=True)

        bigru_row = {
            'Model': 'BiGRU',
            'Best_Internal_MSE': round(float(best_mse), 4),
            'External_MSE': round(float(mse), 4),
            'Pearson_r': round(float(pearson), 4),
            'Spearman_rho': round(float(spearman), 4),
            'Top10_pct_precision': round(float(top10) * 100, 2),
            'Parameters': int(param_count),
        }
        bigru_csv = os.path.join(TABLES_DIR, 'baselines_bigru.csv')
        pd.DataFrame([bigru_row]).to_csv(bigru_csv, index=False)
        bigru_json = os.path.join(JSON_DIR, 'baselines_bigru.json')
        with open(bigru_json, 'w') as f:
            json.dump(bigru_row, f, indent=2)
        logger.info(f"Paper Table 3 row CSV: {bigru_csv}")


if __name__ == "__main__":
    main()
