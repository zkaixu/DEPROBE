#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: model-comparison figure (bootstrap CI forest + ablation waterfall)
===============================================================================
Panel A  Paired bootstrap 95% CI of the DEPROBE-DNA minus BiGRU Top-K gap.
         DEPROBE-DNA leads at strict thresholds and BiGRU leads at the broad
         threshold, all three intervals excluding zero.
Panel B  Waterfall decomposition of the 9.1 pp Top-1% advantage over BiGRU
         into encoder, Rank-N-Contrast, physics priors, and their interaction.

Colourblind-safe Okabe-Ito palette. Editable text in SVG/PDF for production.
Reads results/json/bootstrap_ci.json; the ablation values are the manuscript
Table 3 / Table 2 point estimates, defined inline.

Usage:
    python3 make_model_comparison_figure.py
"""
import os
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PLOTS = os.path.join(PROJECT_ROOT, 'results', 'plots')
os.makedirs(PLOTS, exist_ok=True)

# Okabe-Ito colourblind-safe palette
C_MODEL = '#0072B2'   # DEPROBE-DNA / blue
C_BASE  = '#D55E00'   # BiGRU / drop / vermillion
C_GAIN  = '#009E73'   # gains / bluish green
C_REF   = '#444444'   # reference lines / dark grey

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
})


def save_pub(fig, stem, dpi=600):
    for ext in ('svg', 'pdf'):
        fig.savefig(os.path.join(PLOTS, f'{stem}.{ext}'), bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS, f'{stem}.tiff'), dpi=dpi, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS, f'{stem}.png'), dpi=dpi, bbox_inches='tight')


def load_gaps():
    p = os.path.join(PROJECT_ROOT, 'results', 'json', 'bootstrap_ci.json')
    d = json.load(open(p))['results']
    rows = []
    for key, lab in [('top1pct', 'Top-1%'), ('top5pct', 'Top-5%'), ('top10pct', 'Top-10%')]:
        r = d[key]
        rows.append((lab, float(r['gap_point']), float(r['gap_ci95'][0]), float(r['gap_ci95'][1])))
    return rows


def panel_forest(ax, rows):
    ys = np.arange(len(rows))[::-1]
    for y, (lab, gap, lo, hi) in zip(ys, rows):
        col = C_MODEL if gap > 0 else C_BASE
        ax.errorbar(gap, y, xerr=[[gap - lo], [hi - gap]], fmt='o', color=col,
                    ecolor=col, elinewidth=1.3, capsize=2.5, markersize=4.5, zorder=3)
    ax.axvline(0, color=C_REF, linestyle='--', linewidth=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel('Gap')
    ax.set_xlim(-6, 13)
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.text(-3.0, len(rows) - 0.45, 'BiGRU', ha='center', va='bottom',
            fontsize=5.5, color=C_BASE, style='italic')
    ax.text(8.5, len(rows) - 0.45, 'DEPROBE-DNA', ha='center', va='bottom',
            fontsize=5.5, color=C_MODEL, style='italic')


def panel_waterfall(ax):
    labels = ['BiGRU', '+Encoder', '+RNC', '+Physics', '+Interaction', 'DEPROBE-DNA']
    base = 23.4
    steps = [2.5, 4.0, -0.4, 3.1]
    cum = [base]
    for s in steps:
        cum.append(cum[-1] + s)
    full = cum[-1]   # 32.6

    ax.bar(0, base, color=C_MODEL, width=0.5, zorder=3)
    ax.bar(5, full, color=C_MODEL, width=0.5, zorder=3)
    for i, s in enumerate(steps):
        bottom = min(cum[i], cum[i + 1])
        col = C_GAIN if s > 0 else C_BASE
        ax.bar(i + 1, abs(s), bottom=bottom, color=col, width=0.5, zorder=3)
        ax.text(i + 1, max(cum[i], cum[i + 1]) + 0.35, f'{s:+.1f}',
                ha='center', va='bottom', fontsize=6, color=C_REF)
    ax.text(0, base + 0.35, f'{base:.1f}', ha='center', va='bottom', fontsize=6, color=C_REF)
    ax.text(5, full + 0.35, f'{full:.1f}', ha='center', va='bottom', fontsize=6, color=C_REF)

    conn_y = [base, cum[1], cum[2], cum[3], cum[4]]
    for i in range(5):
        ax.plot([i + 0.25, i + 1 - 0.25], [conn_y[i], conn_y[i]],
                color=C_REF, linewidth=0.6, zorder=2)

    ax.set_xticks(range(6))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=6)
    ax.set_ylabel('Top-1% selection precision (%)')
    ax.set_ylim(0, 36)
    handles = [Patch(facecolor=C_MODEL, label='Model'),
               Patch(facecolor=C_GAIN, label='Gain'),
               Patch(facecolor=C_BASE, label='Loss')]
    ax.legend(handles=handles, loc='upper left', fontsize=5.6, handlelength=1.1,
              handleheight=1.0, borderpad=0.3, labelspacing=0.3)


def main():
    rows = load_gaps()
    fig = plt.figure(figsize=(7.0, 3.0), constrained_layout=True)
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.25], figure=fig)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_forest(ax_a, rows)
    panel_waterfall(ax_b)
    fig.text(0.005, 0.96, 'a', fontsize=10, fontweight='bold')
    fig.text(0.44, 0.96, 'b', fontsize=10, fontweight='bold')
    save_pub(fig, 'model_comparison')
    plt.close(fig)
    print('Saved model_comparison.{svg,pdf,tiff,png} to', PLOTS)


if __name__ == '__main__':
    main()
