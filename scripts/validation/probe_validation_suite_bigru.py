#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiGRU Probe Validation Suite (adapter wrapper).
===============================================
Reuses the DEPROBE probe_validation_suite.py modules (1-5) but loads BiGRU
instead of DEPROBE for inference. Same CLI args as the original suite.

Output filenames are prefixed with --output_suffix (recommend "_bigru") to
prevent overwriting DEPROBE's CSV/JSON outputs.

Usage:
    python3 probe_validation_suite_bigru.py \
        --checkpoint ../../models/baseline_bigru/bigru_best.pth \
        --h5 ../../data/data_factory/final/probe_validation/deprobe_probe_val_master.h5 \
        --staging_csv ../../data/data_factory/staging/probe_validation/deprobe_probe_val_master.csv \
        --probe_mapping ../../data/beds/nextera_probe_mapping.csv \
        --target_bed ../../data/beds/nexterarapidcapture_expandedexome_targetedregions.bed \
        --batch_size 4096 \
        --output_suffix _bigru
"""
import os
import sys
import torch
import numpy as np

# Add model and validation dirs to path so we can import BiGRUModel and the suite.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)  # for probe_validation_suite
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'model'))  # for BiGRUModel

import probe_validation_suite as pvs  # noqa: E402
from baseline_bigru import BiGRUModel  # noqa: E402


# ============================================================
# Override load_model: load BiGRU instead of DEPROBE
# ============================================================
def load_bigru_model(checkpoint_path, device, prior_dim=12):
    model = BiGRUModel(d_model=256, prior_dim=prior_dim, num_layers=2).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    weights = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(weights, strict=False)
    model.eval()
    prior_mean = ckpt.get('prior_mean', None)
    prior_std = ckpt.get('prior_std', None)
    if prior_mean is not None:
        prior_mean = prior_mean.cpu()
    if prior_std is not None:
        prior_std = prior_std.cpu()
    return model, prior_mean, prior_std


# ============================================================
# Override predict_all: BiGRU's forward takes (x, priors, pad_mask), no modality, no alpha
# ============================================================
def predict_all_bigru(model, dataloader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['anchor'].to(device)
            priors = batch['priors'].to(device)
            mask = batch['anchor_mask'].to(device)
            with torch.amp.autocast('cuda'):
                pred = model(x, priors, mask)
            all_preds.append(pred.squeeze().cpu().numpy())
            all_labels.append(batch['efficiency'].numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ============================================================
# Monkey-patch the suite's two model-specific functions, then run main()
# ============================================================
pvs.load_model = load_bigru_model
pvs.predict_all = predict_all_bigru

if __name__ == "__main__":
    pvs.main()
