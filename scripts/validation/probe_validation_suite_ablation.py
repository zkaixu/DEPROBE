#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Probe Validation Suite (adapter wrapper).
==================================================
Reuses the DEPROBE probe_validation_suite.py modules (1-5) but loads an
ablation checkpoint with the appropriate inference-time flags
(use_physics, use_early_fusion) parsed from the checkpoint's config name.
The ablation training script (main_ab.py) saves the config name in the
checkpoint metadata; we recover the inference behavior from that name.

Output filenames are prefixed with --output_suffix (e.g. "_abl_no_physics")
to prevent overwriting the main DEPROBE / BiGRU CSVs.

Usage:
    python3 probe_validation_suite_ablation.py \
        --checkpoint ../../models/ablation/ablation_no_physics_best.pth \
        --h5 ../../data/data_factory/final/probe_validation/deprobe_probe_val_master.h5 \
        --staging_csv ../../data/data_factory/staging/probe_validation/deprobe_probe_val_master.csv \
        --probe_mapping ../../data/beds/nextera_probe_mapping.csv \
        --target_bed ../../data/beds/nexterarapidcapture_expandedexome_targetedregions.bed \
        --batch_size 4096 \
        --output_suffix _abl_no_physics
"""
import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)  # for probe_validation_suite
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'model'))  # for DEPROBE

import probe_validation_suite as pvs  # noqa: E402
from model import DEPROBE  # noqa: E402


# ============================================================
# Parse ablation config name to inference flags
# (mirrors main_ab.py's __init__ logic)
# ============================================================
def _parse_ablation_flags(config_name):
    """
    Map config_name string -> (use_physics, use_early_fusion).
    Mirrors main_ab.py:
        use_physics       = not args.no_physics
        use_early_fusion  = not args.late_fusion_only and use_physics
    """
    has_no_physics = 'no_physics' in config_name
    has_late_fusion_only = 'late_fusion_only' in config_name
    use_physics = not has_no_physics
    use_early_fusion = (not has_late_fusion_only) and use_physics
    return use_physics, use_early_fusion


# ============================================================
# Override load_model: load DEPROBE + recover ablation flags
# ============================================================
def load_ablation_model(checkpoint_path, device, prior_dim=12):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config_name = ckpt.get('config', 'unknown')

    use_physics, use_early_fusion = _parse_ablation_flags(config_name)

    print(f"  Ablation config: {config_name}")
    print(f"    use_physics       = {use_physics}")
    print(f"    use_early_fusion  = {use_early_fusion}")
    print(f"    sam_activated     = {ckpt.get('sam_activated', 'unknown')}")
    print(f"    best_int_val_mse  = {ckpt.get('best_mse', 'unknown')}")
    print(f"    epoch             = {ckpt.get('epoch', 'unknown')}")

    model = DEPROBE(
        num_platforms=10, prior_dim=prior_dim,
        num_modalities=5, d_model=256
    ).to(device)
    weights = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(weights, strict=False)
    model.eval()

    # Attach ablation flags to the model so predict_all can use them
    model._abl_use_physics = use_physics
    model._abl_use_early_fusion = use_early_fusion

    prior_mean = ckpt.get('prior_mean', None)
    prior_std = ckpt.get('prior_std', None)
    if prior_mean is not None:
        prior_mean = prior_mean.cpu()
    if prior_std is not None:
        prior_std = prior_std.cpu()

    return model, prior_mean, prior_std


# ============================================================
# Override predict_all: mimic main_ab.py's _validate routine
# (zero priors when use_physics=False, control inject_physics)
# ============================================================
def predict_all_ablation(model, dataloader, device):
    use_physics = getattr(model, '_abl_use_physics', True)
    use_early_fusion = getattr(model, '_abl_use_early_fusion', True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['anchor'].to(device)
            priors = batch['priors'].to(device)
            mod = batch['modality'].to(device)
            mask = batch['anchor_mask'].to(device)

            if not use_physics:
                priors = torch.zeros_like(priors)

            with torch.amp.autocast('cuda'):
                _, z_fused = model.encode(
                    x, priors, mod,
                    pad_mask=mask,
                    inject_physics=use_early_fusion
                )
                pred = model.efficiency_regressor(z_fused).squeeze(-1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch['efficiency'].numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ============================================================
# Monkey-patch the suite, then run main()
# ============================================================
pvs.load_model = load_ablation_model
pvs.predict_all = predict_all_ablation


if __name__ == "__main__":
    pvs.main()
