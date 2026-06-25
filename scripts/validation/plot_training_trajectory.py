#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE: Phase 1 Training Trajectory Visualiser
===============================================
Parses one or more Phase 1 12D training logs and produces a single
two-panel plot of the per-epoch MSE trajectories (Train / Int Val /
Ext Val) plus the learning-rate schedule.

Resume runs are handled correctly: when several log files cover
overlapping epoch ranges, the most-recent record for each epoch wins.
This makes it trivial to glue together the four-segment Phase 1
training history (initial run → rho-fix resume → post-restart resume
→ current run) into a single coherent trajectory plot.

QC: resume artifact epochs are excluded
---------------------------------------
When training resumes from a mid-epoch temp checkpoint
(`dann_temp_E{N}_S{step}.pth` with step > 0), the *first* epoch of
that resume log only averages the Train MSE over the **remaining**
batches in epoch N, not a full epoch. This produces a single-epoch
V-shaped dip (Train MSE drops sharply, then snaps back at epoch N+1)
that is purely a measurement artifact, not a real training improvement.

For Phase 1 12D, the affected epochs are:
    E84   ← Log 2 (20260422_203637) resumed from dann_temp_E84_S2000.pth
    E126  ← Log 3 (20260423_171012) resumed from a mid-E126 temp checkpoint

These epochs are excluded by `--qc_exclude_epochs` (default: 84 126).
The CSV dump and the plot both use the filtered set. Source logs in
`logs/` retain the raw unfiltered values for reproducibility.

Outputs (defensive `os.makedirs` everywhere):
    results/plots/training_trajectory.png      # 2-panel figure
    results/tables/training_trajectory.csv     # per-epoch data dump

Usage:
    # Default: autodiscovers all train_phase1_*.log under logs/,
    # excludes E84 and E126 as resume artifacts.
    python plot_training_trajectory.py

    # Explicit log list, in any order, script merges by epoch number
    python plot_training_trajectory.py \\
        --logs logs/train_phase1_20260421_141954.log \\
               logs/train_phase1_20260422_203637.log \\
               logs/train_phase1_20260423_171012.log \\
               logs/train_phase1_20260424_235716.log

    # Disable QC filtering (e.g. to inspect raw artifacts)
    python plot_training_trajectory.py --qc_exclude_epochs
"""

import os
import re
import sys
import glob
import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("trajectory")

# ----------------------------------------------------------------------
# Canonical output paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'plots')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables')

# ----------------------------------------------------------------------
# Regex for the epoch-end log line.
#
# Sample line we are matching:
#   [2026-04-25 14:28:54] [INFO] Epoch [173/1000] | Alpha: 0.000 |
#       LR: 6.25e-06 | Train MSE: 0.1568 | Int Val (deprobe_train_master.h5):
#       0.1528 | Ext Val (deprobe_val_master.h5): 0.1701 | Time: 2472.5s
#
# The pattern is forgiving: anything between the named fields, possibly
# with H5 filenames in parentheses, is allowed to wander.
# ----------------------------------------------------------------------
EPOCH_END_PATTERN = re.compile(
    r'\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?'
    r'Epoch \[(?P<epoch>\d+)/\d+\].*?'
    r'LR:\s*(?P<lr>[\d.eE+-]+).*?'
    r'Train MSE:\s*(?P<train>[\d.]+).*?'
    r'Int Val[^:]*:\s*(?P<int_val>[\d.]+).*?'
    r'Ext Val[^:]*:\s*(?P<ext_val>[\d.]+|N/A)'
)

RESUME_PATTERN = re.compile(
    r'\[Resume\] Restored at Epoch (?P<ep>\d+),\s*Step (?P<st>\d+)'
)


def parse_log(path: str) -> Tuple[Dict[int, dict], List[Tuple[str, int]]]:
    """Parse one log file. Returns
        (epoch_records, resume_events)

    epoch_records   : {epoch_int: {timestamp, lr, train_mse, int_val, ext_val}}
    resume_events   : list of (timestamp, epoch) tuples where a Resume line fired
    """
    epoch_records: Dict[int, dict] = {}
    resume_events: List[Tuple[str, int]] = []
    if not os.path.exists(path):
        logger.warning(f"missing log file, skipping: {path}")
        return epoch_records, resume_events

    with open(path, 'r', errors='replace') as fh:
        for raw in fh:
            line = raw.rstrip('\n')

            ep_match = EPOCH_END_PATTERN.search(line)
            if ep_match:
                ev = ep_match.group('ext_val')
                if ev == 'N/A':
                    # External val not configured for this epoch. Skip silently.
                    ext_val = float('nan')
                else:
                    ext_val = float(ev)
                epoch_records[int(ep_match.group('epoch'))] = {
                    'timestamp': ep_match.group('ts'),
                    'lr': float(ep_match.group('lr')),
                    'train_mse': float(ep_match.group('train')),
                    'int_val': float(ep_match.group('int_val')),
                    'ext_val': ext_val,
                    'source_log': os.path.basename(path),
                }
                continue

            res_match = RESUME_PATTERN.search(line)
            if res_match:
                # Try to grab the timestamp at the start of the line, if any.
                ts_match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
                ts = ts_match.group(1) if ts_match else ''
                resume_events.append((ts, int(res_match.group('ep'))))

    return epoch_records, resume_events


def merge_logs(log_paths: List[str]) -> Tuple[pd.DataFrame, List[int], List[Tuple[str, int]]]:
    """Parse all logs in chronological order (later logs win on conflict).
    Returns:
        df              : per-epoch DataFrame, sorted by epoch
        lr_drop_epochs  : epochs where LR dropped versus the previous epoch
        resume_events   : combined list of resume events
    """
    # Sort by filename. train_phase1_<YYYYMMDD>_<HHMMSS>.log is naturally chronological.
    log_paths = sorted(log_paths)

    merged: Dict[int, dict] = {}
    all_resumes: List[Tuple[str, int]] = []
    for p in log_paths:
        epochs, resumes = parse_log(p)
        logger.info(f"  {os.path.basename(p):<40s} | epochs parsed: {len(epochs):>4d} "
                    f"| resumes: {len(resumes)}")
        # Later logs win on overlap.
        merged.update(epochs)
        all_resumes.extend(resumes)

    if not merged:
        logger.error("no epoch records parsed — aborting.")
        sys.exit(1)

    eps = sorted(merged.keys())
    df = pd.DataFrame([
        {
            'Epoch': e,
            'Timestamp': merged[e]['timestamp'],
            'LR': merged[e]['lr'],
            'Train_MSE': merged[e]['train_mse'],
            'Int_Val_MSE': merged[e]['int_val'],
            'Ext_Val_MSE': merged[e]['ext_val'],
            'Source_Log': merged[e]['source_log'],
        }
        for e in eps
    ])

    # Detect scheduler-induced LR reductions (where LR dropped relative to previous epoch).
    lr_drop_epochs: List[int] = []
    for i in range(1, len(df)):
        if df.iloc[i]['LR'] < df.iloc[i - 1]['LR']:
            lr_drop_epochs.append(int(df.iloc[i]['Epoch']))

    return df, lr_drop_epochs, all_resumes


def render_plot(df: pd.DataFrame, lr_drops: List[int], resumes: List[Tuple[str, int]],
                out_path: str, noise_floor: float = None):
    """Two-panel figure: MSE trajectories (top) + LR schedule (bottom)."""
    eps = df['Epoch'].values
    train_mse = df['Train_MSE'].values
    int_val = df['Int_Val_MSE'].values
    ext_val = df['Ext_Val_MSE'].values
    lrs = df['LR'].values

    # Best-internal / best-external markers
    best_int_idx = int(np.nanargmin(int_val))
    best_ext_idx = int(np.nanargmin(ext_val))
    best_int_ep, best_int_val = int(eps[best_int_idx]), float(int_val[best_int_idx])
    best_ext_ep, best_ext_val = int(eps[best_ext_idx]), float(ext_val[best_ext_idx])

    # Okabe-Ito colourblind-safe palette, consistent across all paper figures.
    C_TRAIN = '#0072B2'   # blue (training MSE)
    C_INT   = '#009E73'   # bluish green (internal validation)
    C_EXT   = '#D55E00'   # vermillion (external validation)
    C_FLOOR = '#444444'   # dark grey (optional reference line)
    C_LR    = '#444444'   # dark grey (learning-rate schedule)

    fig, (ax_mse, ax_lr) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )

    # ----- MSE panel -----
    ax_mse.plot(eps, train_mse, color=C_TRAIN, linewidth=1.5, alpha=0.85, label='Train MSE')
    ax_mse.plot(eps, int_val,   color=C_INT,   linewidth=1.8, label='Int Val MSE')
    ax_mse.plot(eps, ext_val,   color=C_EXT,   linewidth=1.8, label='Ext Val MSE')

    # Optional noise-floor reference line (off by default; the noise-floor
    # concept is not used). Pass an explicit --noise_floor X to overlay a
    # horizontal reference for diagnostic plotting only.
    if noise_floor is not None:
        ax_mse.axhline(noise_floor, color=C_FLOOR, linestyle='--', linewidth=1.2,
                       alpha=0.85,
                       label=f'Reference line = {noise_floor:.3f}')

    # Best markers
    ax_mse.scatter([best_int_ep], [best_int_val], s=80, color=C_INT,
                   edgecolors='black', linewidths=1.2, zorder=5,
                   label=f'Best Int: {best_int_val:.4f} @ E{best_int_ep}')
    ax_mse.scatter([best_ext_ep], [best_ext_val], s=80, color=C_EXT, marker='^',
                   edgecolors='black', linewidths=1.2, zorder=5,
                   label=f'Best Ext: {best_ext_val:.4f} @ E{best_ext_ep}')

    # LR-drop vertical lines (= scheduler reductions = SAM activations downstream)
    for i, drop in enumerate(lr_drops):
        ax_mse.axvline(drop, color='#999999', linestyle=':', linewidth=1.0, alpha=0.55,
                       label='LR reduce / SAM activate' if i == 0 else None)

    ax_mse.set_ylabel('MSE', fontsize=12)
    ax_mse.grid(alpha=0.3)
    ax_mse.legend(loc='upper right', fontsize=9, ncol=1)

    # ----- LR panel -----
    ax_lr.plot(eps, lrs, color=C_LR, linewidth=1.5, drawstyle='steps-post')
    for i, drop in enumerate(lr_drops):
        ax_lr.axvline(drop, color='#999999', linestyle=':', linewidth=1.0, alpha=0.55)
    ax_lr.set_xlabel('Epoch', fontsize=12)
    ax_lr.set_ylabel('Learning Rate', fontsize=11)
    ax_lr.set_yscale('log')
    ax_lr.grid(alpha=0.3, which='both')

    # Mark resume events (lighter, on the LR axis to avoid crowding the MSE panel)
    for _, ep in resumes:
        if ep in eps:
            ax_lr.axvline(ep, color='#888888', linestyle='-', linewidth=0.6, alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Phase-1 12D training trajectory across one or more log files.")
    parser.add_argument('--logs', nargs='+', default=None,
                        help="Log files. If omitted, autodiscovers logs/train_phase1_*.log.")
    parser.add_argument('--out', default=os.path.join(PLOTS_DIR, 'training_trajectory.png'),
                        help=f"Output PNG (default: {os.path.join(PLOTS_DIR, 'training_trajectory.png')}).")
    parser.add_argument('--csv', default=os.path.join(TABLES_DIR, 'training_trajectory.csv'),
                        help=f"Output CSV (default: {os.path.join(TABLES_DIR, 'training_trajectory.csv')}).")
    parser.add_argument('--noise_floor', type=float, default=None,
                        help="Optional horizontal reference line (off by default; "
                             "the noise-floor concept is not used).")
    parser.add_argument('--qc_exclude_epochs', type=int, nargs='*', default=[84, 126],
                        help="Epochs to drop from the trajectory as resume / partial-epoch "
                             "artifacts. Pass an empty list to disable QC filtering. "
                             "Defaults to [84, 126] for Phase 1 12D.")
    args = parser.parse_args()

    if args.logs is None:
        autopattern = os.path.join(DEFAULT_LOG_DIR, 'train_phase1_*.log')
        args.logs = sorted(glob.glob(autopattern))
        if not args.logs:
            logger.error(f"no logs matched {autopattern} — pass --logs explicitly.")
            sys.exit(1)
        logger.info(f"autodiscovered {len(args.logs)} logs from {DEFAULT_LOG_DIR}/")

    logger.info("Parsing logs in chronological order:")
    df, lr_drops, resumes = merge_logs(args.logs)

    # ------------------------------------------------------------------
    # QC: drop resume-induced partial-epoch artifacts.
    # The first epoch of any resume log that started from a mid-epoch
    # temp checkpoint reports a Train MSE averaged over only the
    # remaining batches → a single-point V-shaped dip. We drop those
    # epochs before plotting / writing the CSV so the trajectory
    # reflects genuine optimisation behaviour only.
    # ------------------------------------------------------------------
    qc_excluded: List[int] = sorted(set(int(e) for e in (args.qc_exclude_epochs or [])))
    if qc_excluded:
        before = len(df)
        keep_mask = ~df['Epoch'].isin(qc_excluded)
        dropped_rows = df.loc[~keep_mask, ['Epoch', 'Train_MSE', 'Int_Val_MSE',
                                           'Ext_Val_MSE', 'Source_Log']]
        df = df.loc[keep_mask].reset_index(drop=True)
        # Re-detect LR-drops on the QC-cleaned set so the vertical
        # markers don't get distorted by the dropped rows.
        lr_drops = [int(df.iloc[i]['Epoch'])
                    for i in range(1, len(df))
                    if df.iloc[i]['LR'] < df.iloc[i - 1]['LR']]
        logger.info(f"QC: dropped {before - len(df)} epoch(s) as resume artifacts: "
                    f"{qc_excluded}")
        for _, row in dropped_rows.iterrows():
            logger.info(f"     E{int(row['Epoch']):>3d} | Train {row['Train_MSE']:.4f} | "
                        f"Int {row['Int_Val_MSE']:.4f} | Ext {row['Ext_Val_MSE']:.4f} | "
                        f"{row['Source_Log']}")
    else:
        logger.info("QC: --qc_exclude_epochs disabled (raw trajectory).")

    logger.info("=" * 60)
    logger.info(f"Total unique epochs : {len(df)}  (after QC)")
    logger.info(f"Epoch range         : E{int(df['Epoch'].min())} → E{int(df['Epoch'].max())}")
    logger.info(f"LR-drop epochs      : {lr_drops}")
    logger.info(f"Resume events       : {len(resumes)}")
    logger.info(f"Best Int Val MSE    : {df['Int_Val_MSE'].min():.4f}  "
                f"@ E{int(df.loc[df['Int_Val_MSE'].idxmin(), 'Epoch'])}")
    if df['Ext_Val_MSE'].notna().any():
        logger.info(f"Best Ext Val MSE    : {df['Ext_Val_MSE'].min():.4f}  "
                    f"@ E{int(df.loc[df['Ext_Val_MSE'].idxmin(), 'Epoch'])}")
    logger.info("=" * 60)

    # Persist the merged per-epoch table.
    os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
    df.to_csv(args.csv, index=False)
    logger.info(f"CSV: {args.csv}")

    render_plot(df, lr_drops, resumes, args.out, noise_floor=args.noise_floor)


if __name__ == "__main__":
    main()
