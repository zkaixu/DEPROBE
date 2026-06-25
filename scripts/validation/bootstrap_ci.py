#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: Bootstrap confidence intervals for Top-K selection precision
=========================================================================
Quantifies the uncertainty of the L3 real-probe Top-K precision point
estimates and of the DEPROBE-DNA minus BiGRU gap at each threshold.

Checkpoints are loaded as-is (no retraining); inference runs once over the
L3 set to recover per-probe (prediction, label) pairs, then a paired
bootstrap resamples the 344,090 probes B times. Because the two models
share the same probes (DataLoader shuffle=False), each bootstrap replicate
uses one resample index for both, so the per-replicate difference is a
paired statistic and its percentile interval is the CI of the gap.

Top-K precision uses a threshold definition (fraction of the predicted
top-K whose label reaches the label's own top-K threshold), which is
resample-safe and matches the set-based suite definition without ties.

Usage (defaults point at the canonical paths, so no args are needed):
    python3 bootstrap_ci.py
    python3 bootstrap_ci.py --n_boot 2000 --seed 42
"""
import os
import sys
import json
import argparse
import contextlib
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'model'))

import probe_validation_suite as pvs            # noqa: E402  (set-based reference metric)
from dataset import PanMolecularProbeDataset     # noqa: E402
from model import DEPROBE                         # noqa: E402
from baseline_bigru import BiGRUModel             # noqa: E402


def amp_ctx(device):
    return torch.amp.autocast('cuda') if device.type == 'cuda' else contextlib.nullcontext()


def load_deprobe(ckpt_path, device):
    model = DEPROBE(num_platforms=10, prior_dim=12, num_modalities=5, d_model=256).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    pm, ps = ckpt.get('prior_mean'), ckpt.get('prior_std')
    return model, (pm.cpu() if pm is not None else None), (ps.cpu() if ps is not None else None)


def load_bigru(ckpt_path, device):
    model = BiGRUModel(d_model=256, prior_dim=12, num_layers=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    pm, ps = ckpt.get('prior_mean'), ckpt.get('prior_std')
    return model, (pm.cpu() if pm is not None else None), (ps.cpu() if ps is not None else None)


def infer(model, loader, device, kind):
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['anchor'].to(device)
            priors = batch['priors'].to(device)
            mask = batch['anchor_mask'].to(device)
            with amp_ctx(device):
                if kind == 'deprobe':
                    mod = batch['modality'].to(device)
                    _, pred, _ = model(x, priors, mod, pad_mask=mask, alpha=0.0)
                else:
                    pred = model(x, priors, mask)
            preds.append(pred.squeeze().float().cpu().numpy())
            labels.append(batch['efficiency'].numpy())
    return np.concatenate(preds), np.concatenate(labels)


def topk_prec(y_true, y_pred, k_percent):
    n = len(y_true)
    k = max(1, int(n * k_percent / 100))
    pred_top = np.argpartition(y_pred, -k)[-k:]
    thresh = np.partition(y_true, -k)[-k]
    return float(np.mean(y_true[pred_top] >= thresh))


def run_inference(h5, ckpt, loader_kwargs, device, kind, loadfn):
    model, pm, ps = loadfn(ckpt, device)
    ds = PanMolecularProbeDataset(h5, prior_mean=pm, prior_std=ps)
    loader = DataLoader(ds, shuffle=False, **loader_kwargs)
    yp, yt = infer(model, loader, device, kind)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return yp, yt


def main():
    ap = argparse.ArgumentParser(description="Bootstrap CI for Top-K precision and the DEPROBE-BiGRU gap.")
    ap.add_argument('--h5', default=os.path.join(PROJECT_ROOT, 'data/data_factory/final/probe_validation/deprobe_probe_val_master.h5'))
    ap.add_argument('--deprobe_ckpt', default=os.path.join(PROJECT_ROOT, 'models/phase1_pure_physics/deprobe_best_internal.pth'))
    ap.add_argument('--bigru_ckpt', default=os.path.join(PROJECT_ROOT, 'models/baseline_bigru/bigru_best.pth'))
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=4096)
    ap.add_argument('--num_workers', type=int, default=4, help="Set 0 if h5py multiprocessing errors occur.")
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lk = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    print("Device:", device)

    print("Running DEPROBE-DNA inference over L3 ...")
    yp_d, yt = run_inference(args.h5, args.deprobe_ckpt, lk, device, 'deprobe', load_deprobe)
    print("  n =", len(yt))

    print("Running BiGRU inference over L3 ...")
    yp_b, yt_b = run_inference(args.h5, args.bigru_ckpt, lk, device, 'bigru', load_bigru)
    print("  n =", len(yt_b))

    assert len(yt) == len(yt_b), "row count mismatch between the two runs"
    assert np.allclose(yt, yt_b, atol=1e-4), "label vectors differ between runs; ordering not aligned"

    n = len(yt)

    print("\nPoint estimates (threshold metric | set-based reference):")
    for kp in args.ks:
        d_thr = topk_prec(yt, yp_d, kp) * 100
        d_set = pvs.top_k_accuracy(yt, yp_d, kp) * 100
        g_thr = topk_prec(yt, yp_b, kp) * 100
        g_set = pvs.top_k_accuracy(yt, yp_b, kp) * 100
        print(f"  Top-{kp:>2d}%  DEPROBE {d_thr:6.2f} | {d_set:6.2f}    BiGRU {g_thr:6.2f} | {g_set:6.2f}")

    rng = np.random.default_rng(args.seed)
    acc = {kp: {'d': [], 'g': [], 'diff': []} for kp in args.ks}
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        yt_r, yd_r, yb_r = yt[idx], yp_d[idx], yp_b[idx]
        for kp in args.ks:
            d = topk_prec(yt_r, yd_r, kp)
            g = topk_prec(yt_r, yb_r, kp)
            acc[kp]['d'].append(d)
            acc[kp]['g'].append(g)
            acc[kp]['diff'].append(d - g)
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot}")

    def pct(a):
        a = np.asarray(a) * 100
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    out = {'n_probes': int(n), 'n_boot': int(args.n_boot), 'seed': int(args.seed),
           'ci_method': 'percentile (2.5/97.5)', 'metric': 'threshold Top-K precision (%)',
           'results': {}}

    print("\n" + "=" * 64)
    print("  BOOTSTRAP 95%% CI  (B=%d, seed=%d)" % (args.n_boot, args.seed))
    print("=" * 64)
    for kp in args.ks:
        d_pt = topk_prec(yt, yp_d, kp) * 100
        g_pt = topk_prec(yt, yp_b, kp) * 100
        d_lo, d_hi = pct(acc[kp]['d'])
        g_lo, g_hi = pct(acc[kp]['g'])
        df = np.asarray(acc[kp]['diff']) * 100
        df_lo, df_hi = float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))
        excl0 = bool(df_lo > 0 or df_hi < 0)
        print(f"Top-{kp}%")
        print(f"  DEPROBE-DNA   {d_pt:6.2f}   95%CI [{d_lo:6.2f}, {d_hi:6.2f}]")
        print(f"  BiGRU         {g_pt:6.2f}   95%CI [{g_lo:6.2f}, {g_hi:6.2f}]")
        print(f"  gap (D-B)     {d_pt - g_pt:+6.2f}   95%CI [{df_lo:+6.2f}, {df_hi:+6.2f}]   CI excludes 0: {excl0}")
        out['results'][f'top{kp}pct'] = {
            'deprobe_point': d_pt, 'deprobe_ci95': [d_lo, d_hi],
            'bigru_point': g_pt, 'bigru_ci95': [g_lo, g_hi],
            'gap_point': d_pt - g_pt, 'gap_ci95': [df_lo, df_hi],
            'gap_ci_excludes_zero': excl0,
        }

    json_path = os.path.join(PROJECT_ROOT, 'results', 'json', 'bootstrap_ci.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", json_path)


if __name__ == "__main__":
    main()
