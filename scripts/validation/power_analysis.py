#!/usr/bin/env python3
"""
DEPROBE-DNA: Statistical Power Analysis (Cohen's d for capture-efficiency differences)
======================================================================================
Computes expected effect size and statistical power for the proposed
Top-25 vs Bottom-25 probe comparison experiment.

Uses in-silico data (model predictions + NIST7035 actual efficiency)
to estimate downstream measurement sensitivity.
"""

import os
import sys
import numpy as np
import torch
import h5py
from scipy import stats
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'model')
sys.path.insert(0, MODEL_DIR)

from dataset import PanMolecularProbeDataset
from model import DEPROBE


def load_model(checkpoint_path, device):
    model = DEPROBE(num_platforms=10, prior_dim=12, num_modalities=5, d_model=256).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    weights = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(weights, strict=False)
    model.eval()
    prior_mean = ckpt.get('prior_mean', None)
    prior_std = ckpt.get('prior_std', None)
    if prior_mean is not None: prior_mean = prior_mean.cpu()
    if prior_std is not None: prior_std = prior_std.cpu()
    return model, prior_mean, prior_std


def predict_all(model, dataloader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['anchor'].to(device)
            priors = batch['priors'].to(device)
            mod = batch['modality'].to(device)
            mask = batch['anchor_mask'].to(device)
            with torch.amp.autocast('cuda'):
                _, pred, _ = model(x, priors, mod, pad_mask=mask, alpha=0.0)
            all_preds.append(pred.squeeze().cpu().numpy())
            all_labels.append(batch['efficiency'].numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_power_nct(d, n_per_group, alpha=0.05):
    """
    Compute power using non-central t-distribution (exact method).
    d: Cohen's d effect size
    n_per_group: samples per group
    alpha: significance level (two-tailed)
    """
    df = 2 * n_per_group - 2
    ncp = d * np.sqrt(n_per_group / 2)  # non-centrality parameter
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # Power = P(|T| > t_crit | H1) using non-central t
    # For large ncp, stats.nct.cdf can return NaN due to numerical overflow.
    # Fall back to normal approximation when this happens.
    try:
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        if np.isnan(power):
            raise ValueError
    except (ValueError, RuntimeWarning):
        # Normal approximation: Z = (T - ncp) / 1, power ≈ Phi(ncp - z_crit) + Phi(-ncp - z_crit)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        power = stats.norm.cdf(ncp - z_crit) + stats.norm.cdf(-ncp - z_crit)
    return power


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--n_per_group", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = args.n_per_group

    print("=" * 70)
    print("  DEPROBE-DNA: STATISTICAL POWER ANALYSIS")
    print("=" * 70)

    # Load and predict
    model, prior_mean, prior_std = load_model(args.checkpoint, device)
    dataset = PanMolecularProbeDataset(args.h5, prior_mean=prior_mean, prior_std=prior_std)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False,
                        num_workers=8, pin_memory=True, persistent_workers=True)
    print(f"  Total probes: {len(dataset):,}")
    print(f"  Running inference...")
    y_pred, y_true = predict_all(model, loader, device)

    # Select Top-N and Bottom-N by prediction
    top_idx = np.argsort(y_pred)[-n:]
    bot_idx = np.argsort(y_pred)[:n]

    top_true = y_true[top_idx]
    bot_true = y_true[bot_idx]

    # ================================================================
    # Effect size analysis
    # ================================================================
    mean_top = np.mean(top_true)
    mean_bot = np.mean(bot_true)
    std_top = np.std(top_true, ddof=1)
    std_bot = np.std(bot_true, ddof=1)
    pooled_std = np.sqrt((std_top**2 + std_bot**2) / 2)
    cohens_d = (mean_top - mean_bot) / pooled_std

    # Direct t-test on in-silico data
    t_stat, p_value = stats.ttest_ind(top_true, bot_true)

    print(f"\n{'='*70}")
    print(f"  IN-SILICO EFFECT SIZE (Top-{n} vs Bottom-{n})")
    print(f"{'='*70}")
    print(f"  Top-{n} predicted → actual efficiency:")
    print(f"    Mean: {mean_top:.4f}  SD: {std_top:.4f}")
    print(f"  Bottom-{n} predicted → actual efficiency:")
    print(f"    Mean: {mean_bot:.4f}  SD: {std_bot:.4f}")
    print(f"  Difference: {mean_top - mean_bot:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  In-silico t-test: t={t_stat:.2f}, p={p_value:.2e}")

    # Interpretation
    if cohens_d > 0.8:
        effect_label = "LARGE (d > 0.8)"
    elif cohens_d > 0.5:
        effect_label = "MEDIUM (0.5 < d < 0.8)"
    elif cohens_d > 0.2:
        effect_label = "SMALL (0.2 < d < 0.5)"
    else:
        effect_label = "NEGLIGIBLE (d < 0.2)"
    print(f"  Effect size category: {effect_label}")

    # ================================================================
    # Power analysis at different noise levels
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  POWER ANALYSIS (two-sample t-test, alpha={args.alpha}, two-tailed)")
    print(f"{'='*70}")
    print(f"  Method: Non-central t-distribution (exact)")
    print(f"  Assumption: downstream effect size = k × in-silico effect size")
    print(f"  (k < 1 accounts for experimental noise reducing the signal)\n")

    print(f"  {'Noise scenario':<30s} {'Effective d':>12s} {'Power':>8s} {'Verdict':>15s}")
    print(f"  {'-'*65}")

    scenarios = [
        ("In-silico (no added noise)", 1.0),
        ("Mild downstream noise (k=0.8)", 0.8),
        ("Moderate noise (k=0.6)", 0.6),
        ("Heavy noise (k=0.4)", 0.4),
        ("Severe noise (k=0.2)", 0.2),
    ]

    for label, k in scenarios:
        d_eff = cohens_d * k
        power = compute_power_nct(d_eff, n, args.alpha)
        if power > 0.9:
            verdict = "ROBUST"
        elif power > 0.8:
            verdict = "ADEQUATE"
        elif power > 0.5:
            verdict = "MARGINAL"
        else:
            verdict = "UNDERPOWERED"
        print(f"  {label:<30s} {d_eff:>12.3f} {power:>7.1%} {verdict:>15s}")

    # ================================================================
    # Minimum sample size for 80% power
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  MINIMUM SAMPLE SIZE FOR 80% POWER")
    print(f"{'='*70}")

    for label, k in scenarios:
        d_eff = cohens_d * k
        if d_eff <= 0:
            continue
        for n_try in range(5, 200):
            if compute_power_nct(d_eff, n_try, args.alpha) >= 0.80:
                print(f"  {label:<30s}  d={d_eff:.3f}  → n={n_try} per group")
                break

    # ================================================================
    # Bootstrap: what fraction of random Top/Bottom draws are significant?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  BOOTSTRAP VALIDATION (10,000 resamples)")
    print(f"{'='*70}")
    print(f"  Simulating: randomly draw {n} from predicted-top pool and {n} from")
    print(f"  predicted-bottom pool, test if actual efficiency differs.\n")

    # Use top 5% and bottom 5% as pools (more realistic than exact top/bottom 25)
    n_pool = max(n * 5, int(len(y_pred) * 0.05))
    top_pool_idx = np.argsort(y_pred)[-n_pool:]
    bot_pool_idx = np.argsort(y_pred)[:n_pool]
    top_pool_true = y_true[top_pool_idx]
    bot_pool_true = y_true[bot_pool_idx]

    n_boot = 10000
    sig_count = 0
    d_values = []
    for _ in range(n_boot):
        sample_top = np.random.choice(top_pool_true, n, replace=True)
        sample_bot = np.random.choice(bot_pool_true, n, replace=True)
        _, p = stats.ttest_ind(sample_top, sample_bot)
        d = (sample_top.mean() - sample_bot.mean()) / np.sqrt((sample_top.var() + sample_bot.var()) / 2)
        d_values.append(d)
        if p < args.alpha:
            sig_count += 1

    boot_power = sig_count / n_boot
    d_values = np.array(d_values)

    print(f"  Top pool: top {n_pool} probes by prediction")
    print(f"  Bottom pool: bottom {n_pool} probes by prediction")
    print(f"  Significant results: {sig_count}/{n_boot} ({boot_power:.1%})")
    print(f"  Bootstrap Cohen's d: mean={d_values.mean():.3f}, "
          f"median={np.median(d_values):.3f}, 5th pct={np.percentile(d_values, 5):.3f}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  CONCLUSION")
    print(f"{'='*70}")
    print(f"  In-silico Cohen's d = {cohens_d:.3f} ({effect_label})")
    print(f"  Power at n={n}/group (no added noise): {compute_power_nct(cohens_d, n, args.alpha):.1%}")
    print(f"  Power at n={n}/group (moderate noise k=0.6): {compute_power_nct(cohens_d*0.6, n, args.alpha):.1%}")
    print(f"  Bootstrap success rate (from top/bottom 5% pools): {boot_power:.1%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
