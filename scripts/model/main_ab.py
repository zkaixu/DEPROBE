#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: Ablation Study Trainer
===================================
Standalone training script for ablation experiments.
Does NOT modify main_12d.py or model.py.

Supports four ablation configurations via command-line flags:
  --no_physics       : Zero out 12D priors (test sequence-only performance)
  --late_fusion_only : Remove early fusion physics token, keep late concat
  --no_rnc           : Disable Rank-N-Contrast loss (Huber only)

Combinations:
  A. --no_physics                → No physics priors at all
  B. --late_fusion_only          → Late fusion only (no physics token)
  C. --no_rnc                    → No contrastive loss
  D. --no_physics --no_rnc       → Pure sequence encoder + Huber
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import os
import json
import logging

from dataset import PanMolecularProbeDataset
from model import DEPROBE
from loss import rank_n_contrast_loss

# Canonical Paper-1 output destinations (project root = two levels up from this file).
SCRIPT_DIR_BOOTSTRAP = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR_BOOTSTRAP, '..', '..'))
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables')
JSON_DIR = os.path.join(PROJECT_ROOT, 'results', 'json')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DEPROBE-Ablation")


# ====================================================================
# SAM Optimizer (copied from main_12d.py for protocol parity)
# ====================================================================
class SAM(torch.optim.Optimizer):
    """SAM wrapper sharing a base optimizer INSTANCE for LR-scheduler propagation."""
    def __init__(self, params, base_optimizer_instance, rho=0.05):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer_instance
        for group in self.base_optimizer.param_groups:
            group.setdefault('rho', rho)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([p.grad.norm(p=2).to(shared_device) for group in self.param_groups
                         for p in group["params"] if p.grad is not None]),
            p=2
        )
        return norm


class AblationTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler('cuda')

        # Ablation flags
        self.use_physics = not args.no_physics
        self.use_early_fusion = not args.late_fusion_only and self.use_physics
        self.use_rnc = not args.no_rnc

        # Loss weights
        self.w_mse = 1.0
        self.w_rnc = 0.1 if self.use_rnc else 0.0

        # Describe configuration
        # Filename convention: all-lowercase, underscore-only separators (no '+').
        config_parts = []
        if not self.use_physics:
            config_parts.append("no_physics")
        if not self.use_early_fusion and self.use_physics:
            config_parts.append("late_fusion_only")
        if not self.use_rnc:
            config_parts.append("no_rnc")
        if not config_parts:
            config_parts.append("full_model")
        self.config_name = "_".join(config_parts)

        logger.info(f"Ablation config: {self.config_name}")
        logger.info(f"  use_physics={self.use_physics}, use_early_fusion={self.use_early_fusion}, use_rnc={self.use_rnc}")
        logger.info(f"  loss weights: mse={self.w_mse}, rnc={self.w_rnc}")

        # Dataset
        self.dataset = PanMolecularProbeDataset(h5_path=args.data)
        val_size = max(1, int(len(self.dataset) * 0.1))
        self.train_set, self.val_set = random_split(
            self.dataset, [len(self.dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(66)
        )

        loader_cfg = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 4}
        self.train_loader = DataLoader(self.train_set, shuffle=True, drop_last=True,
                                       batch_size=args.batch, **loader_cfg)
        self.val_loader = DataLoader(self.val_set, shuffle=False,
                                     batch_size=args.batch, **loader_cfg)

        # External validation
        self.ext_val_loader = None
        if args.val_data and os.path.exists(args.val_data):
            ext_set = PanMolecularProbeDataset(
                h5_path=args.val_data,
                prior_mean=self.dataset.mu,
                prior_std=self.dataset.sigma
            )
            self.ext_val_loader = DataLoader(ext_set, shuffle=False,
                                             batch_size=args.batch, **loader_cfg)

        logger.info(f"Train: {len(self.train_set):,} | Int Val: {len(self.val_set):,}")

        # Model
        prior_dim = 12 if self.use_physics else 12  # Keep dim=12 but zero out values
        self.model = DEPROBE(num_platforms=10, prior_dim=prior_dim,
                             num_modalities=5, d_model=256).to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer_adam = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=1e-4
        )
        # SAM wraps optimizer_adam INSTANCE so the scheduler attached below propagates.
        self.optimizer_sam = SAM(trainable_params, self.optimizer_adam, rho=0.01)
        self.optimizer = self.optimizer_adam  # legacy alias for parts that read self.optimizer

        # SAM activation matches main_12d.py: enables after first LR reduction (current_lr < initial_lr * 0.99).
        self.sam_activated = False
        self.initial_lr = args.lr

        # threshold=5e-4 abs prevents tracking sub-noise improvements;
        # min_lr=1e-6 caps LR at a numerically meaningful floor.
        # Same scheduler config as main_12d.py for protocol parity.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_adam, mode='min', factor=0.5, patience=3,
            threshold=5e-4, threshold_mode='abs', min_lr=1e-6,
        )

        # Resume support. Best.pth stores: epoch, model_state_dict, best_mse,
        # config, sam_activated, prior_mean, prior_std. Optimizer / scheduler
        # state are not stored, so Adam moments and LR position will reset.
        # For early-epoch resume (LR still at initial), this is essentially
        # lossless. For late-epoch resume, LR rolls back to initial.
        self.start_epoch = 0
        self.best_mse_init = float('inf')
        if args.resume_from:
            if not os.path.exists(args.resume_from):
                raise FileNotFoundError(f"--resume_from path not found: {args.resume_from}")
            logger.info(f"[Resume] Loading checkpoint from {args.resume_from}")
            ckpt = torch.load(args.resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.start_epoch = ckpt.get('epoch', -1) + 1
            self.best_mse_init = ckpt.get('best_mse', float('inf'))
            self.sam_activated = ckpt.get('sam_activated', False)
            logger.info(f"[Resume] Restored at epoch {self.start_epoch}, "
                        f"best_mse={self.best_mse_init:.4f}, sam_activated={self.sam_activated}")

    def _move(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _compute_loss(self, batch):
        x = batch['anchor']
        mask = batch['anchor_mask']
        priors = batch['priors']
        targets = batch['efficiency']
        mod = batch['modality']

        # Zero out physics if ablation requires it
        if not self.use_physics:
            priors = torch.zeros_like(priors)

        # Forward pass, with early fusion controlled by inject_physics
        z_latent, z_fused = self.model.encode(
            x, priors, mod, pad_mask=mask,
            inject_physics=self.use_early_fusion
        )
        eff_pred = self.model.efficiency_regressor(z_fused).squeeze(-1)

        # Huber regression loss
        l_mse = F.huber_loss(eff_pred, targets, delta=0.1)
        with torch.no_grad():
            pure_mse = F.mse_loss(eff_pred, targets)

        # RNC loss (on pure sequence embedding, no physics)
        if self.use_rnc:
            if not self.use_physics:
                z_for_rnc = z_latent  # Already no physics
            else:
                z_for_rnc, _ = self.model.encode(
                    x, priors, mod, pad_mask=mask, inject_physics=False
                )
            l_rnc = rank_n_contrast_loss(z_for_rnc, targets,
                                         tau=self.args.tau, margin=self.args.rnc_margin)
        else:
            l_rnc = torch.tensor(0.0, device=self.device)

        total = self.w_mse * l_mse + self.w_rnc * l_rnc
        return total, pure_mse, l_rnc

    def _validate(self, loader):
        self.model.eval()
        total_mse = 0.0
        with torch.no_grad():
            for raw in loader:
                batch = self._move(raw)
                priors = batch['priors']
                if not self.use_physics:
                    priors = torch.zeros_like(priors)
                with torch.amp.autocast('cuda'):
                    _, z_fused = self.model.encode(
                        batch['anchor'], priors, batch['modality'],
                        pad_mask=batch['anchor_mask'],
                        inject_physics=self.use_early_fusion
                    )
                    pred = self.model.efficiency_regressor(z_fused).squeeze(-1)
                    total_mse += F.mse_loss(pred, batch['efficiency']).item()
        return total_mse / len(loader)

    def run(self):
        os.makedirs(self.args.model_dir, exist_ok=True)
        best_mse = self.best_mse_init
        # Plateau-matched termination (parity with main_12d.py protocol):
        # Terminate when LR has been at min_lr (=1e-6) for `min_lr_patience`
        # consecutive epochs without improvement. This corresponds to ~6
        # LR halvings from initial 1e-4 (1e-4 -> 5e-5 -> 2.5e-5 -> ... -> 1.5625e-6
        # which is clamped to 1e-6) plus a final stabilisation window.
        min_lr_floor = 1e-6
        min_lr_patience = 5  # epochs at min_lr without improvement to terminate
        epochs_at_min_lr_no_improve = 0
        # Fail-safe early stop: many consecutive non-improvement epochs before LR reaches min.
        # Set high enough that plateau scheduler controls termination in normal cases.
        failsafe_patience = 40
        failsafe_counter = 0

        logger.info(f"Starting ablation: {self.config_name} (max {self.args.epochs} epochs, "
                    f"plateau-matched termination)")

        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            epoch_mse = 0.0
            epoch_start = time.time()

            # Match main_12d.py: SAM activates AFTER the first LR reduction.
            current_lr = self.optimizer_adam.param_groups[0]['lr']
            if not self.sam_activated and current_lr < self.initial_lr * 0.99:
                self.sam_activated = True
                logger.info(f"[{self.config_name}] [SAM Activation] LR reduced from "
                            f"{self.initial_lr:.2e} to {current_lr:.2e} at epoch {epoch+1}. "
                            f"Enabling Sharpness-Aware Minimization.")
            use_sam = self.sam_activated
            optimizer = self.optimizer_sam if use_sam else self.optimizer_adam

            for i, raw in enumerate(self.train_loader):
                batch = self._move(raw)

                if use_sam:
                    # SAM in pure float32. AMP autocast disabled for stability.
                    loss, mse, rnc = self._compute_loss(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.first_step(zero_grad=True)

                    loss_2, _, _ = self._compute_loss(batch)
                    loss_2.backward()

                    with torch.no_grad():
                        for group in optimizer.param_groups:
                            for p in group["params"]:
                                if p.grad is None: continue
                                p.sub_(optimizer.state[p]["e_w"])

                    optimizer.base_optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        loss, mse, rnc = self._compute_loss(batch)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()

                epoch_mse += mse.item()

                if i % 500 == 0:
                    logger.info(f"[{self.config_name}] E{epoch+1} S{i} | "
                                f"MSE: {mse.item():.4f} | RNC: {rnc.item():.4f} | "
                                f"SAM: {use_sam}")

            # Validation
            int_val = self._validate(self.val_loader)
            ext_val = self._validate(self.ext_val_loader) if self.ext_val_loader else 0.0
            train_mse = epoch_mse / len(self.train_loader)
            self.scheduler.step(int_val)

            duration = time.time() - epoch_start
            logger.info(
                f"[{self.config_name}] Epoch [{epoch+1}/{self.args.epochs}] | "
                f"Train MSE: {train_mse:.4f} | Int Val: {int_val:.4f} | "
                f"Ext Val: {ext_val:.4f} | Time: {duration:.1f}s"
            )

            if int_val < best_mse - 5e-4:  # absolute improvement threshold matching scheduler
                best_mse = int_val
                failsafe_counter = 0
                epochs_at_min_lr_no_improve = 0
                save_path = os.path.join(self.args.model_dir, f"ablation_{self.config_name}_best.pth")
                # Defensive: ensure parent dir exists right before write.
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'best_mse': best_mse,
                    'config': self.config_name,
                    'sam_activated': self.sam_activated,
                    'prior_mean': self.dataset.mu,
                    'prior_std': self.dataset.sigma,
                }, save_path)
                logger.info(f"  >> Best checkpoint: {save_path} (MSE: {best_mse:.4f})")
            else:
                failsafe_counter += 1
                # If LR has reached the floor, count epochs at min_lr without improvement.
                current_lr_post_step = self.optimizer_adam.param_groups[0]['lr']
                if current_lr_post_step <= min_lr_floor * 1.001:
                    epochs_at_min_lr_no_improve += 1
                    logger.info(f"  >> [Plateau] LR at min ({current_lr_post_step:.2e}); "
                                f"{epochs_at_min_lr_no_improve}/{min_lr_patience} epochs "
                                f"at min_lr without improvement.")
                    if epochs_at_min_lr_no_improve >= min_lr_patience:
                        logger.info(f"  Plateau-matched termination at epoch {epoch+1} "
                                    f"(LR at min for {min_lr_patience} epochs).")
                        break
                else:
                    logger.info(f"  >> [Patience] {failsafe_counter}/{failsafe_patience} "
                                f"non-improvement epochs (LR={current_lr_post_step:.2e}).")
                if failsafe_counter >= failsafe_patience:
                    logger.info(f"  Failsafe termination at epoch {epoch+1} "
                                f"({failsafe_patience} non-improvement epochs).")
                    break

            # Resumability: write a per-epoch progress JSON (status='in_progress')
            # so that if the run is killed mid-training, the latest best is captured.
            # The final summary JSON (status='completed') is written after the loop ends
            # and is what run_ablation.sh checks to decide whether to skip a config.
            progress_path = os.path.join(self.args.model_dir, f"ablation_{self.config_name}_progress.json")
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            with open(progress_path, 'w') as f:
                json.dump({
                    'config': self.config_name,
                    'status': 'in_progress',
                    'epochs_completed': epoch + 1,
                    'epochs_planned_max': self.args.epochs,
                    'best_int_val_mse_so_far': float(best_mse),
                    'latest_int_val_mse': float(int_val),
                    'latest_ext_val_mse': float(ext_val),
                    'sam_activated': bool(self.sam_activated),
                    'use_physics': bool(self.use_physics),
                    'use_early_fusion': bool(self.use_early_fusion),
                    'use_rnc': bool(self.use_rnc),
                }, f, indent=2)

        # Save final summary
        summary = {
            'config': self.config_name,
            'best_int_val_mse': best_mse,
            'epochs_trained': epoch + 1,
            'use_physics': self.use_physics,
            'use_early_fusion': self.use_early_fusion,
            'use_rnc': self.use_rnc,
        }
        summary_path = os.path.join(self.args.model_dir, f"ablation_{self.config_name}_summary.json")
        # Defensive: ensure parent dir exists right before write.
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # ============================================================
        # Paper Table 6 (one CSV row per ablation config). Concatenate
        # the four files later to assemble the full ablation table.
        # CSV: results/tables/ablation_<config>.csv
        # ============================================================
        os.makedirs(TABLES_DIR, exist_ok=True)
        row = {
            'Config': self.config_name,
            'Use_Physics': bool(self.use_physics),
            'Use_Early_Fusion': bool(self.use_early_fusion),
            'Use_RNC': bool(self.use_rnc),
            'Epochs_Trained': int(epoch + 1),
            'Best_Internal_MSE': round(float(best_mse), 4),
        }
        ablation_csv = os.path.join(TABLES_DIR, f"ablation_{self.config_name}.csv")
        pd.DataFrame([row]).to_csv(ablation_csv, index=False)
        logger.info(f"[{self.config_name}] Paper-row CSV: {ablation_csv}")

        logger.info(f"[{self.config_name}] DONE. Best Int Val MSE: {best_mse:.4f}")
        return best_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEPROBE-DNA Ablation Study")
    parser.add_argument("--data", required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=19)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--rnc_margin", type=float, default=0.3)

    # Ablation flags
    parser.add_argument("--no_physics", action="store_true",
                        help="Zero out 12D priors to test sequence-only performance")
    parser.add_argument("--late_fusion_only", action="store_true",
                        help="Remove early fusion physics token, keep late concat only")
    parser.add_argument("--no_rnc", action="store_true",
                        help="Disable Rank-N-Contrast loss, use Huber only")

    # Resume support (model weights, epoch, best_mse, sam_activated only;
    # optimizer/scheduler state are NOT in checkpoint, so they reinitialize).
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint .pth to resume training from")

    AblationTrainer(parser.parse_args()).run()
