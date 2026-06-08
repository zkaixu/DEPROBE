#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPROBE-DNA: molecular probe dataset (12D)
==========================================
SWMR-enabled HDF5 with 12D standardized input.

Note: neg_indices and hard negative retrieval have been removed.
Contrastive learning now uses Rank-N-Contrast (Zha et al., NeurIPS 2023),
which operates on efficiency labels within each mini-batch. No
pre-computed indices needed.
"""

import os
import gc
import h5py
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional

logger = logging.getLogger("DEPROBE-Dataset")


class PanMolecularProbeDataset(Dataset):
    def __init__(self, h5_path: str, max_seq_len: int = 120,
                 prior_mean: Optional[torch.Tensor] = None,
                 prior_std: Optional[torch.Tensor] = None):
        """
        Args:
            h5_path: Absolute path to the preprocessed HDF5 data file.
            max_seq_len: Maximum sequence length for DNA padding/truncation.
            prior_mean: Pre-calculated mean for 12D priors (for UDA cross-domain consistency).
            prior_std: Pre-calculated std for 12D priors (for UDA cross-domain consistency).
        """
        self.h5_path = h5_path
        self.max_seq_len = max_seq_len
        self.base_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}

        self.db: Optional[h5py.File] = None

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"[CRITICAL] Dataset path invalid: {h5_path}")

        with h5py.File(self.h5_path, 'r') as f:
            self.num_samples = f['sequences'].shape[0]
            self.keys = list(f.keys())

            # --- 12D Metadata Detection & Scaling ---
            prior_dim = f['priors'].shape[1]
            logger.info(f"Synchronizing {prior_dim}D Priors for {self.num_samples} samples...")

            # UDA Cross-Domain Consistency: inherit Source Domain statistics
            if prior_mean is not None and prior_std is not None:
                self.mu = prior_mean.clone().float()
                self.sigma = prior_std.clone().float()
                self.sigma[self.sigma == 0] = 1.0
                logger.info(f"[UDA SYNC] Inherited EXTERNAL Source Domain Statistics.")
            else:
                raw_priors = f['priors'][:]
                self.mu = torch.from_numpy(np.mean(raw_priors, axis=0)).float()
                self.sigma = torch.from_numpy(np.std(raw_priors, axis=0)).float()
                self.sigma[self.sigma == 0] = 1.0
                logger.info(f"[AUTO-SCALE] Engaged Internal Dynamic Scaling ({prior_dim}D)")
                del raw_priors
                gc.collect()

        self.db: Optional[h5py.File] = None

    def _init_db(self):
        if self.db is None:
            self.db = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')

    def _encode_sequence(self, seq_data) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(seq_data, bytes):
            seq_str = seq_data.decode('ascii').upper()
        else:
            seq_str = str(seq_data).upper()

        indices = [self.base_map.get(b, 0) for b in seq_str if b in self.base_map]
        actual_len = len(indices)
        mask = torch.ones(self.max_seq_len, dtype=torch.bool)

        if actual_len >= self.max_seq_len:
            indices = indices[:self.max_seq_len]
            mask[:] = False
        else:
            indices = indices + [0] * (self.max_seq_len - actual_len)
            mask[:actual_len] = False

        return torch.tensor(indices, dtype=torch.long), mask

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._init_db()

        # 1. Sequence Encoding
        raw_seq = self.db['sequences'][idx]
        seq_idx, seq_mask = self._encode_sequence(raw_seq)

        # 2. Structural Priors (12D Scaling)
        priors_raw = torch.from_numpy(self.db['priors'][idx]).float()
        priors_scaled = (priors_raw - self.mu) / self.sigma
        priors_final = torch.clamp(priors_scaled, -10.0, 10.0)

        # 3. Efficiency Label
        efficiency = torch.tensor(self.db['efficiency'][idx], dtype=torch.float32)

        # 4. Metadata
        modality = torch.tensor(self.db['modalities'][idx],
                                dtype=torch.long) if 'modalities' in self.keys else torch.tensor(0, dtype=torch.long)
        platform = torch.tensor(self.db['platforms'][idx],
                                dtype=torch.long) if 'platforms' in self.keys else torch.tensor(0, dtype=torch.long)

        return {
            'anchor': seq_idx,
            'anchor_mask': seq_mask,
            'priors': priors_final,
            'efficiency': efficiency,
            'modality': modality,
            'platform': platform
        }
