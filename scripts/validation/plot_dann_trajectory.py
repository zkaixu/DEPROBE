#!/usr/bin/env python3
"""
Plot DANN training trajectory: source vs target MSE + alpha schedule.

Parses the matched-position DANN training log (per-epoch summary line) and
produces a two-panel figure used as Fig. 5 in the manuscript.

Usage:
    python plot_dann_trajectory.py \
        --log logs/matched_bed_dann_20260426_181026.log \
        --out results/plots/dann_trajectory.png
"""
import argparse
import os
import re

import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette, consistent across all paper figures.
# Blue = source-domain training (DEPROBE model fitting improving).
# Vermillion = target-domain failure (the canonical DANN miss).
# Amber = alpha schedule. Grey = random-prediction baseline.
C_SOURCE = '#0072B2'   # blue
C_TARGET = '#D55E00'   # vermillion
C_ALPHA = '#E69F00'    # amber
C_REF = '#444444'      # dark grey (random-prediction baseline)


def parse_log(log_path):
    """Extract per-epoch (epoch, alpha, source_mse, target_mse)."""
    pat = re.compile(
        r'Epoch \[(\d+)/\d+\] \| Alpha: ([\d.]+) \|'
        r'.*?Int Val[^:]*:\s*([\d.]+).*?Ext Val[^:]*:\s*([\d.]+)'
    )
    epochs, alphas, src_mse, tgt_mse = [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                alphas.append(float(m.group(2)))
                src_mse.append(float(m.group(3)))
                tgt_mse.append(float(m.group(4)))
    return epochs, alphas, src_mse, tgt_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='matched_bed_dann log path')
    ap.add_argument('--out', required=True, help='output PNG path')
    args = ap.parse_args()

    epochs, alphas, src, tgt = parse_log(args.log)
    if not epochs:
        raise SystemExit(f"No epoch summary lines parsed from: {args.log}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )

    # Top: source vs target MSE
    ax1.plot(epochs, src, color=C_SOURCE, linewidth=2.0, marker='o', markersize=3,
             label=f'Source held-out (Nextera): {src[0]:.3f} → {min(src):.3f}')
    ax1.plot(epochs, tgt, color=C_TARGET, linewidth=2.0, marker='s', markersize=3,
             label=f'Target (TruSeq): {min(tgt):.2f}–{max(tgt):.2f} (flat)')
    ax1.axhline(1.0, color=C_REF, linestyle=':', linewidth=1.2, alpha=0.8,
                label='Random baseline (MSE = 1.000)')
    ax1.set_ylabel('MSE', fontsize=11)
    ax1.legend(fontsize=10, loc='center right')
    ax1.grid(alpha=0.3)

    # Bottom: alpha schedule
    ax2.plot(epochs, alphas, color=C_ALPHA, linewidth=2.0)
    ax2.fill_between(epochs, 0, alphas, color=C_ALPHA, alpha=0.3)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel(r'$\alpha$', fontsize=11)
    ax2.set_ylim(0, max(alphas) * 1.05 if max(alphas) > 0 else 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.out}  ({len(epochs)} epochs parsed)")


if __name__ == "__main__":
    main()
