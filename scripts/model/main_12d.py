#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: 12D master trainer
===============================
Feature Set:   12D thermodynamic priors (Tm, GC, dG, hairpin/dimer Tm,
               yield, normalized length, GC skew, dinucleotide entropy,
               dG_5p, dG_3p, MinHash collision penalty)
Optimization:  Pure DNA Domain Adaptation (DANN)
Validation:    Dual-Track (Internal 10% Holdout + External Target Domain)
Data Flow:     Dual-Stream Batch Merging for Phase 2 DANN Finetuning
Contrastive:   Rank-N-Contrast (Zha et al., NeurIPS 2023)
Reliability:   Checkpoint resumption, plateau-triggered SAM, AMP
Hardware:      tested on NVIDIA RTX 5090
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import time
import os
import json
import subprocess
import logging
from typing import Dict, Tuple

from dataset import PanMolecularProbeDataset
from model import DEPROBE
from loss import rank_n_contrast_loss

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DEPROBE-Trainer")


# ====================================================================
# Sharpness-Aware Minimization (SAM) Optimizer
# ====================================================================
class SAM(torch.optim.Optimizer):
    """SAM wrapper that shares a base optimizer INSTANCE (not class).

    Critical: base_optimizer_instance must already exist, and its
    param_groups are shared with SAM so any LR scheduler attached
    to the base optimizer propagates directly to SAM.

    Previous implementation took a class and instantiated a SECOND
    AdamW internally. That broke LR propagation because the scheduler
    was attached to the outer AdamW while training used the inner one.
    """
    def __init__(self, params, base_optimizer_instance, rho=0.05):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer_instance
        # Inject 'rho' into each of base_optimizer's param_groups so first_step/
        # second_step can read group["rho"]. Without this injection, base_optimizer
        # (AdamW) was created without rho, and the share-param_groups below would
        # wipe out SAM's original rho-containing groups.
        for group in self.base_optimizer.param_groups:
            group.setdefault('rho', rho)
        # Share param_groups with base_optimizer (LR scheduling propagates automatically)
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


def get_gpu_hardware_stats() -> Tuple[float, float]:
    try:
        cmd = "nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL).decode('utf-8').strip()
        vram, pwr = output.split(',')
        return float(vram), float(pwr)
    except Exception:
        return 0.0, 0.0


# ====================================================================
# DEPROBE-DNA Master Trainer
# ====================================================================
class DEPROBETrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.scaler = torch.amp.GradScaler('cuda')

        self.loss_weights = {"mse": 1.0, "rnc": 0.1, "dom": 0.1}

        # =================================================================
        # 1. Dataset & Split
        # =================================================================
        self.dataset = PanMolecularProbeDataset(h5_path=args.data)

        total_batch_size = args.batch

        # Dynamic Batch Splitting for DANN
        if args.target_data and os.path.exists(args.target_data) and not getattr(args, 'master', False):
            bs_source = total_batch_size // 2
            bs_target = total_batch_size - bs_source
            self.is_dann_mode = True
            logger.info(f"DANN Mode Activated: Batch split -> Source:{bs_source} | Target:{bs_target}")
        else:
            bs_source = total_batch_size
            bs_target = 0
            self.is_dann_mode = False
            if getattr(args, 'master', False):
                logger.info(f"MASTER SUB-MODEL MODE: DANN Disabled.")
            else:
                logger.info(f"Phase 1 Mode: Pure Source Training (Batch: {bs_source})")

        self.bs_source = bs_source

        # Track A: Internal Source Validation (10% Holdout)
        val_size = max(1, int(len(self.dataset) * 0.1))
        self.train_set, self.internal_val_set = random_split(
            self.dataset, [len(self.dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(66)
        )

        base_loader_cfg = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 4}

        self.train_loader = DataLoader(self.train_set, shuffle=True, drop_last=True, batch_size=bs_source,
                                       **base_loader_cfg)
        self.internal_val_loader = DataLoader(self.internal_val_set, shuffle=False, batch_size=bs_source,
                                              **base_loader_cfg)
        self.source_name = os.path.basename(args.data)
        logger.info(f"Loaded Source Domain ({self.source_name}): {len(self.train_set)} Train | {len(self.internal_val_set)} Internal Val")

        # Track B: Unlabeled Target Domain for DANN Training
        self.target_loader = None
        if self.is_dann_mode:
            self.target_set = PanMolecularProbeDataset(
                h5_path=args.target_data,
                prior_mean=self.dataset.mu,
                prior_std=self.dataset.sigma
            )
            self.target_loader = DataLoader(self.target_set, shuffle=True, drop_last=True, batch_size=bs_target,
                                            **base_loader_cfg)
            self.target_name = os.path.basename(args.target_data)
            logger.info(f"Loaded Target Domain for DANN ({self.target_name}): {len(self.target_set)} Samples")

        # Track C: External Target Validation
        self.external_val_loader = None
        self.val_name = "N/A"
        if args.val_data and os.path.exists(args.val_data):
            self.external_val_set = PanMolecularProbeDataset(
                h5_path=args.val_data,
                prior_mean=self.dataset.mu,
                prior_std=self.dataset.sigma
            )
            self.external_val_loader = DataLoader(self.external_val_set, shuffle=False, batch_size=total_batch_size,
                                                  **base_loader_cfg)
            self.val_name = os.path.basename(args.val_data)
            logger.info(f"Loaded Target Domain Validation ({self.val_name}): {len(self.external_val_set)} External Val")

        # =================================================================
        # 2. Model & Optimization
        # =================================================================
        # Hardcoded 12D prior dimension; this trainer is dimension-locked.
        self.model = DEPROBE(num_platforms=10, prior_dim=12, num_modalities=5, d_model=256).to(self.device)
        self.current_lr = args.lr

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer_adam = torch.optim.AdamW(trainable_params, lr=self.current_lr, weight_decay=1e-4)
        # SAM wraps optimizer_adam INSTANCE, shares param_groups so scheduler LR
        # reductions propagate to SAM automatically (fix for silent LR-propagation bug)
        self.optimizer_sam = SAM(trainable_params, self.optimizer_adam, rho=0.01)
        # threshold=5e-4 abs : ignore sub-noise "improvements" so the scheduler
        # doesn't keep cutting LR while we're already at the metric noise floor.
        # min_lr=1e-6        : LR floor; below this each cut is numerically meaningless.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_adam, mode='min', factor=0.5, patience=3,
            threshold=5e-4, threshold_mode='abs', min_lr=1e-6,
        )
        self.initial_lr = self.current_lr
        self.sam_activated = False

        # 3. State Management
        self.start_epoch = 0
        self.start_step = 0
        self.best_internal_mse = float('inf')
        self.best_external_mse = float('inf')

        if args.resume_from and os.path.exists(args.resume_from):
            logger.info(f"[Resume] Restoring full training state from: {args.resume_from}")
            ckpt = torch.load(args.resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
            if 'optimizer_adam_state' in ckpt: self.optimizer_adam.load_state_dict(ckpt['optimizer_adam_state'])
            # Defensive: load_state_dict may strip our 'rho' injection if the checkpoint's
            # param_groups didn't contain it. Re-inject to keep SAM.first_step happy.
            for group in self.optimizer_adam.param_groups:
                group.setdefault('rho', 0.01)
            # Re-share param_groups: load_state_dict replaces optimizer_adam's param_groups
            # with a freshly-built list, breaking the sharing set up in SAM.__init__.
            # Without this re-link, optimizer_sam keeps pointing to the stale original list,
            # which causes scheduler LR reductions to silently not appear in SAM's view
            # (and also causes the epoch-end log to display the pre-reduce LR).
            self.optimizer_sam.param_groups = self.optimizer_adam.param_groups
            # Older checkpoints with 'optimizer_sam_state' are ignored to avoid inner AdamW state inconsistency.
            if 'scheduler_state' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state'])
                # Defensive: re-apply current threshold / min_lr in case the
                # checkpoint was saved before these settings were tightened.
                # ReduceLROnPlateau.load_state_dict() restores ALL knobs (including
                # threshold and min_lrs), which silently re-introduces the loose
                # pre-fix configuration. Force the new values back.
                self.scheduler.threshold = 5e-4
                self.scheduler.threshold_mode = 'abs'
                self.scheduler.min_lrs = [1e-6] * len(self.scheduler.min_lrs)
            if 'scaler_state' in ckpt: self.scaler.load_state_dict(ckpt['scaler_state'])
            self.start_epoch = ckpt.get('epoch', 0)
            self.start_step = ckpt.get('step', 0)
            self.best_internal_mse = ckpt.get('best_internal_mse', float('inf'))
            self.best_external_mse = ckpt.get('best_external_mse', float('inf'))
            # Restore sam_activated flag (defaults to False if absent).
            self.sam_activated = ckpt.get('sam_activated', False)
            restored_lr = self.optimizer_adam.param_groups[0]['lr']
            logger.info(
                f"[Resume] Restored at Epoch {self.start_epoch + 1}, Step {self.start_step} | "
                f"LR={restored_lr:.2e} | sam_activated={self.sam_activated} | "
                f"scheduler.num_bad_epochs={self.scheduler.num_bad_epochs}"
            )

        elif args.finetune_from and os.path.exists(args.finetune_from):
            logger.info(f"[Finetune] Loading physical weights from: {args.finetune_from}")
            ckpt = torch.load(args.finetune_from, map_location=self.device)
            weights = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(weights, strict=False)
            if args.freeze_base:
                for name, param in self.model.named_parameters():
                    if any(x in name for x in ['embedding', 'conv', 'transformer']):
                        param.requires_grad = False
                logger.info("[Finetune] Base physical feature extractors frozen for DANN.")

    def _move_batch_to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _compute_all_losses(self, batch, lambda_adv):
        x_anc = batch['anchor']
        anc_mask = batch['anchor_mask']
        priors, targets = batch['priors'], batch['efficiency']
        mod = batch['modality']
        plat = batch.get('platform', torch.zeros_like(mod))

        # Forward pass
        z_anc, eff_pred, dom_pred = self.model(x_anc, priors, mod, pad_mask=anc_mask, alpha=lambda_adv)

        # For RNC: get pure sequence embedding (without physics injection)
        z_pure, _ = self.model.encode(x_anc, priors, mod, pad_mask=anc_mask, inject_physics=False)

        eff_pred_flat = eff_pred.view(-1)
        targets_flat = targets.view(-1)

        # MSE Loss (source domain only in DANN mode)
        if self.is_dann_mode:
            bs_source = self.bs_source
            l_mse = F.huber_loss(eff_pred_flat[:bs_source], targets_flat[:bs_source], delta=0.1)
            with torch.no_grad():
                pure_mse = F.mse_loss(eff_pred_flat[:bs_source], targets_flat[:bs_source])
            # RNC on source domain only (target has no reliable labels)
            l_rnc = rank_n_contrast_loss(z_pure[:bs_source], targets_flat[:bs_source],
                                         tau=self.args.tau, margin=self.args.rnc_margin)
        else:
            l_mse = F.huber_loss(eff_pred_flat, targets_flat, delta=0.1)
            with torch.no_grad():
                pure_mse = F.mse_loss(eff_pred_flat, targets_flat)
            l_rnc = rank_n_contrast_loss(z_pure, targets_flat,
                                         tau=self.args.tau, margin=self.args.rnc_margin)

        # Domain Loss
        if getattr(self.args, 'master', False):
            l_dom = torch.tensor(0.0, device=self.device)
        else:
            unique_plats = torch.unique(plat).size(0)
            l_dom = F.cross_entropy(dom_pred, plat) if unique_plats > 1 else torch.tensor(0.0, device=self.device)

        total_loss = self.loss_weights["mse"] * l_mse + self.loss_weights["rnc"] * l_rnc + self.loss_weights["dom"] * l_dom

        return total_loss, pure_mse, l_rnc

    def _validate_epoch(self, loader) -> float:
        self.model.eval()
        val_mse_accum = 0.0
        with torch.no_grad():
            for v_raw in loader:
                v_batch = self._move_batch_to_device(v_raw)
                with torch.amp.autocast('cuda'):
                    _, pred, _ = self.model(v_batch['anchor'], v_batch['priors'], v_batch['modality'],
                                            pad_mask=v_batch['anchor_mask'], alpha=0.0)
                    val_mse_accum += F.mse_loss(pred.squeeze(), v_batch['efficiency']).item()
        return val_mse_accum / len(loader)

    def run_training(self):
        os.makedirs(self.args.model_dir, exist_ok=True)
        peak_vram, power_readings = 0.0, []
        history = {"train_mse": [], "internal_val_mse": [], "external_val_mse": [], "rnc": []}
        early_stop_patience = 7
        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            epoch_mse, epoch_rnc, epoch_total = 0.0, 0.0, 0.0
            epoch_start = time.time()

            lambda_adv = 2. / (1. + np.exp(-10 * (float(epoch) / self.args.epochs))) - 1
            if not self.is_dann_mode:
                lambda_adv = 0.0

            # SAM activation triggered by LR scheduler plateau detection.
            # Applies to both source-only and DANN modes; in DANN finetune
            # it remains off unless the scheduler reduces LR below
            # initial_lr × 0.99 (typically does not trigger because DANN
            # starts at a lower fixed LR than Phase 1 by design).
            current_lr = self.optimizer_adam.param_groups[0]['lr']
            if not self.sam_activated and current_lr < self.initial_lr * 0.99:
                self.sam_activated = True
                logger.info(
                    f"[SAM Activation] LR reduced from {self.initial_lr:.2e} to {current_lr:.2e} at epoch {epoch + 1}. Enabling Sharpness-Aware Minimization.")
            use_sam = self.sam_activated
            optimizer = self.optimizer_sam if use_sam else self.optimizer_adam

            target_iter = iter(self.target_loader) if self.target_loader else None

            for i, raw_source_batch in enumerate(self.train_loader):
                if epoch == self.start_epoch and i < self.start_step:
                    if i % 500 == 0: logger.info(f"[Fast-Forward] Skipping Step {i} / {self.start_step}...")
                    continue

                # Dual-Stream Batch Merging for DANN
                if self.is_dann_mode and target_iter:
                    try:
                        raw_target_batch = next(target_iter)
                    except StopIteration:
                        target_iter = iter(self.target_loader)
                        raw_target_batch = next(target_iter)

                    raw_batch = {k: torch.cat([raw_source_batch[k], raw_target_batch[k]], dim=0)
                                 for k in raw_source_batch.keys()}
                else:
                    raw_batch = raw_source_batch

                batch = self._move_batch_to_device(raw_batch)

                if use_sam:
                    # SAM in pure float32. AMP autocast disabled for stability.
                    # SAM's two forward-backward passes are incompatible with
                    # float16 scaling: the perturbation step can push activations
                    # into float16 overflow territory, causing NaN.
                    loss, mse, rnc = self._compute_all_losses(batch, lambda_adv)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.first_step(zero_grad=True)

                    loss_2, _, _ = self._compute_all_losses(batch, lambda_adv)
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
                        loss, mse, rnc = self._compute_all_losses(batch, lambda_adv)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()

                if i % 500 == 0:
                    v_now, p_now = get_gpu_hardware_stats()
                    logger.info(
                        f"E{epoch + 1} S{i} | Loss: {loss.item():.4f} | "
                        f"MSE(Src): {mse.item():.4f} | RNC: {rnc.item():.4f} | "
                        f"VRAM: {v_now / 1024:.1f}G | Pwr: {p_now:.0f}W"
                    )

                if i > 0 and i % 2000 == 0:
                    temp_save_path = os.path.join(self.args.model_dir, f"dann_temp_E{epoch + 1}_S{i}.pth")
                    # Defensive: re-ensure model_dir exists right before write.
                    os.makedirs(self.args.model_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch, 'step': i,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_adam_state': self.optimizer_adam.state_dict(),
                        'scheduler_state': self.scheduler.state_dict(),
                        'scaler_state': self.scaler.state_dict(),
                        'sam_activated': self.sam_activated,
                        'prior_mean': self.dataset.mu,
                        'prior_std': self.dataset.sigma
                    }, temp_save_path)
                    logger.info(f"[Auto-Save] Temporary checkpoint saved: {temp_save_path}")

                epoch_total += loss.item()
                epoch_mse += mse.item()
                epoch_rnc += rnc.item()
                v_now, p_now = get_gpu_hardware_stats()
                peak_vram = max(peak_vram, v_now)
                power_readings.append(p_now)

            self.start_step = 0

            # Dual-Track Validation
            avg_train_mse = epoch_mse / len(self.train_loader)
            internal_val_mse = self._validate_epoch(self.internal_val_loader)

            external_val_mse = 0.0
            ext_log_str = "N/A"
            if self.external_val_loader:
                external_val_mse = self._validate_epoch(self.external_val_loader)
                ext_log_str = f"{external_val_mse:.4f}"

            epoch_duration = time.time() - epoch_start
            current_lr_now = optimizer.param_groups[0]['lr']
            self.scheduler.step(internal_val_mse)

            history["train_mse"].append(avg_train_mse)
            history["internal_val_mse"].append(internal_val_mse)
            if self.external_val_loader: history["external_val_mse"].append(external_val_mse)
            history["rnc"].append(epoch_rnc / len(self.train_loader))

            logger.info(
                f"Epoch [{epoch + 1}/{self.args.epochs}] | "
                f"Alpha: {lambda_adv:.3f} | LR: {current_lr_now:.2e} | "
                f"Train MSE: {avg_train_mse:.4f} | "
                f"Int Val ({self.source_name}): {internal_val_mse:.4f} | "
                f"Ext Val ({self.val_name}): {ext_log_str} | "
                f"Time: {epoch_duration:.1f}s"
            )

            # Checkpoint Logic
            checkpoint_base = {
                'epoch': epoch, 'step': 0,
                'model_state_dict': self.model.state_dict(),
                'optimizer_adam_state': self.optimizer_adam.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'scaler_state': self.scaler.state_dict(),
                'sam_activated': self.sam_activated,
                'best_internal_mse': self.best_internal_mse,
                'best_external_mse': self.best_external_mse,
                'prior_mean': self.dataset.mu,
                'prior_std': self.dataset.sigma
            }

            if internal_val_mse < self.best_internal_mse:
                self.best_internal_mse = internal_val_mse
                patience_counter = 0
                checkpoint_base['best_internal_mse'] = self.best_internal_mse
                int_save_path = os.path.join(self.args.model_dir, f"deprobe_best_internal.pth")
                # Defensive: re-ensure model_dir exists right before write.
                os.makedirs(self.args.model_dir, exist_ok=True)
                torch.save(checkpoint_base, int_save_path)
                logger.info(f" >> [Best Internal] Updated: {int_save_path} (MSE: {self.best_internal_mse:.4f})")
            else:
                patience_counter += 1
                logger.info(f" >> [Patience] {patience_counter}/{early_stop_patience} epochs without improvement.")
                if patience_counter >= early_stop_patience:
                    logger.info(f"[Early Stop] No improvement for {early_stop_patience} epochs. Stopping training.")
                    break

            if self.external_val_loader and external_val_mse < self.best_external_mse:
                self.best_external_mse = external_val_mse
                checkpoint_base['best_external_mse'] = self.best_external_mse
                ext_save_path = os.path.join(self.args.model_dir, f"deprobe_best_external_holdout.pth")
                # Defensive: re-ensure model_dir exists right before write.
                os.makedirs(self.args.model_dir, exist_ok=True)
                torch.save(checkpoint_base, ext_save_path)
                logger.info(f" >> [Best External] Updated: {ext_save_path} (MSE: {self.best_external_mse:.4f})")

        self._save_final_report(self.best_internal_mse, self.best_external_mse, peak_vram, power_readings, history,
                                start_time, self.args.epochs)

    def _save_final_report(self, best_int, best_ext, vram, power, history, start_time, total_epochs):
        report = {
            "hardware": {"gpu": self.gpu_name, "peak_vram_mb": vram, "avg_power_draw": np.mean(power) if power else 0},
            "training": {"best_internal_mse": best_int, "best_external_mse": best_ext,
                         "duration_sec": time.time() - start_time},
            "history": history, "config": vars(self.args)
        }
        summary_path = os.path.join(self.args.model_dir, f"production_summary_E{total_epochs}.json")
        # Defensive: re-ensure model_dir exists right before write.
        os.makedirs(self.args.model_dir, exist_ok=True)
        with open(summary_path, 'w') as f: json.dump(report, f, indent=4)
        logger.info(f"[COMPLETE] Training Finished. Best Internal: {best_int:.4f} | Best External: {best_ext:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEPROBE-DNA 12D Master Trainer")
    parser.add_argument("--data", required=True, help="Path to main Pretrain Master H5")
    parser.add_argument("--target_data", type=str, default=None, help="Path to Unlabeled Target Domain H5 for DANN")
    parser.add_argument("--val_data", type=str, default=None, help="Path to Holdout Val H5")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.1, help="RNC temperature")
    parser.add_argument("--rnc_margin", type=float, default=0.3,
                        help="RNC label distance margin for positive pairs "
                             "(in efficiency label units; 0.3 for z-score normalized labels)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--finetune_from", type=str, help="Load weights only, start new training.")
    parser.add_argument("--resume_from", type=str, help="Load full checkpoint to resume training.")
    # Locked to 12D for this trainer; do not expose --prior_dim as a CLI knob.
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--master", action="store_true",
                        help="Enable Master Sub-model Finetuning (Forces DANN/Domain Loss to 0)")
    DEPROBETrainer(parser.parse_args()).run_training()
