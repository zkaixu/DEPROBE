# DEPROBE-DNA

**Deep learning for hybridization-capture probe selection and the limits of cross-kit transfer**

Zhikai Xu, Department of Advanced and Regenerative Medicine, Institute of Health and Medicine, Hefei Comprehensive National Science Center, Anhui, China

## Overview

Hybridization-capture sequencing enriches selected genomic targets before sequencing and underlies most clinical exome panels in variant discovery, pharmacogenomic profiling, and hereditary disease screening. A working panel contains tens to hundreds of thousands of biotinylated oligonucleotide probes selected from a much larger candidate pool to maximise per-target sequencing depth at fixed panel size. Existing deep-learning approaches optimise within-panel regression accuracy on fixed manufactured panels and have not been purpose-built for the top-K candidate-selection regime that drives commercial panel curation.

Designed for that regime, DEPROBE-DNA is a physics-informed convolutional Transformer that integrates a twelve-dimensional thermodynamic prior vector with learned sequence representations through dual-pathway early and late fusion, and trains on genome-wide sliding-window candidates rather than on a pre-existing manufactured panel. A Huber regression objective is paired with a Rank-N-Contrast contrastive auxiliary loss to preserve label rank ordering at the prediction-distribution extremes, with plateau-triggered Sharpness-Aware Minimization for late-stage refinement.

This repository reproduces two findings from the accompanying manuscript. On 344,090 designed Nextera Expanded Exome probes evaluated against an independent Genome in a Bottle technical replicate of NA12878, DEPROBE-DNA achieves 32.6-fold enrichment over random at Top-1% selection, a 9.1-percentage-point Top-1% lead over a methodologically matched BiGRU baseline. On a matched-position dual-kit dataset built from 1.68 million Nextera ∩ TruSeq intersection probes in NA12878, capture-efficiency labels at identical genomic positions diverge between kits (Pearson r = 0.21 cross-kit vs r = 0.60 within-kit), and domain-adversarial training does not close the gap. The divergence is attributable to kit-specific contextual contributions to depth-derived labels rather than feature-space mismatch, ruling out unsupervised domain adaptation as a remedy and indicating that cross-kit deployment requires direct measurement of probe-target hybridization in place of depth-derived labels.

## Repository Structure

```
DEPROBE_DNA/
├── scripts/
│   ├── data/                        # Data pipeline (12D)
│   │   ├── extract_sequences.py     # Sliding window probe candidate generation
│   │   ├── calc_efficiency.py       # BAM depth-based efficiency labeling
│   │   ├── calc_priors.py           # 12D thermodynamic prior computation
│   │   ├── data_fusion.py           # Feature and label merging
│   │   ├── build_h5.py              # CSV to HDF5 conversion
│   │   ├── build_pretrain_dataset.py # End-to-end pipeline orchestrator
│   │   ├── parse_probe_manifest.py  # Manifest → centered BED conversion
│   │   ├── compute_dual_labels.py   # Matched-position dual-kit label computation
│   │   ├── build_matched_bed_h5.py  # Matched-position dual-kit H5 assembly
│   │   ├── setup_train_data.sh      # Download and process training data
│   │   ├── setup_val_data.sh        # Download and process validation data
│   │   ├── setup_truseq_data.sh     # Download and process TruSeq DANN target
│   │   └── nextera_probe.sh         # Download Illumina Nextera Expanded Exome probe manifest
│   │
│   ├── model/                       # Model architecture and training
│   │   ├── model.py                 # DEPROBE-DNA architecture definition
│   │   ├── dataset.py               # HDF5 dataset loader
│   │   ├── loss.py                  # Rank-N-Contrast loss implementation
│   │   ├── main_12d.py              # 12D Phase 1 / Phase 2 master trainer
│   │   ├── main_ab.py               # Ablation study trainer
│   │   ├── baseline_bigru.py        # BiGRU baseline (Zhang et al. 2021)
│   │   ├── go_phase1_12d.sh         # Phase 1 training launcher
│   │   └── go_matched_bed_dann.sh   # Matched-position cross-kit DANN launcher
│   │
│   ├── validation/                  # Evaluation and analysis
│   │   ├── evaluate_model.py        # Spearman, Top-K, NDCG evaluation
│   │   ├── evaluate_domain_gap.py   # Cross-platform domain gap analysis
│   │   ├── evaluate_overlap_regions.py # Position-matched overlap evaluation
│   │   ├── estimate_noise_floor.py  # Technical replicate noise floor estimation
│   │   ├── probe_validation_suite.py # 6-module real-probe validation suite
│   │   ├── probe_replicate_check.py # Inter-replicate consistency check
│   │   ├── verify_h5.py             # HDF5 dataset verification suite
│   │   ├── baseline_traditional_ml.py # Traditional ML baselines
│   │   ├── power_analysis.py        # Power / Cohen's d analysis
│   │   ├── plot_priors.py           # Prior distribution visualization
│   │   ├── plot_dann_trajectory.py  # DANN training trajectory visualization
│   │   ├── plot_training_trajectory.py # Phase 1 training trajectory
│   │   ├── label_correlation_diagnostic.py # Matched-position cross-kit scatter
│   │   ├── roc_analysis.py          # ROC and PR curves under top-K binarisation
│   │   ├── probe_validation_suite_ablation.py # Ablation evaluation wrapper
│   │   ├── probe_validation_suite_bigru.py # BiGRU baseline evaluation wrapper
│   │   ├── go_probe_eval.sh         # Probe evaluation launcher
│   │   ├── go_probe_suite_12d.sh    # Probe validation suite launcher (12D)
│   │   ├── run_ablation.sh          # Ablation study runner
│   │   └── run_ablation_l3_eval.sh  # L3 real-probe evaluation runner for ablation
│   │
│   └── design/
│       └── design.py                # Probe panel design inference engine
│
├── results/                         # Output destination (auto-created)
│   ├── plots/                       # PNG / PDF figures
│   ├── tables/                      # CSV / TSV result tables
│   └── json/                        # Metric summaries
│
├── .gitignore
└── README.md
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA-compatible GPU (tested on NVIDIA RTX 5090)

### Python packages

```
torch
h5py
pandas
numpy
scipy
scikit-learn
pysam
primer3-py
biopython
datasketch
tqdm
matplotlib
seaborn
```

### System tools

```
samtools
aria2c (for data download)
```

## Quick Start

### 1. Data Preparation

Download GIAB reference data and generate HDF5 training files:

```bash
cd scripts/data
bash setup_train_data.sh    # Nextera source domain (training)
bash setup_val_data.sh      # Nextera source domain (validation)
bash setup_truseq_data.sh   # TruSeq DANN target
```

Each script automatically downloads BAM files from the GIAB FTP, extracts probe candidates via sliding windows, computes 12D thermodynamic priors, and generates HDF5 files. Output directories are created on demand if missing.

### 2. Verify Data Integrity

```bash
cd scripts/validation
python verify_h5.py
```

### 3. Phase 1 Training (Source Domain)

```bash
cd scripts/model
bash go_phase1_12d.sh
```

Training runs to convergence under early stopping (patience 7) with a `ReduceLROnPlateau` scheduler (patience 3, factor 0.5). When the scheduler reduces the learning rate below the initial value, plateau-triggered Sharpness-Aware Minimization activates automatically for the late-stage refinement. Monitor with:

```bash
tail -f ../../logs/train_phase1_*.log
```

### 4. Evaluate Phase 1

```bash
cd scripts/validation
python evaluate_model.py \
    --checkpoint ../../models/phase1_pure_physics/deprobe_best_internal.pth \
    --data ../../data/data_factory/final/val/deprobe_val_master.h5
```

Plots are written to `results/plots/`. JSON metrics summary is written to `results/json/` when configured.

### 5. Probe Validation Suite

```bash
cd scripts/validation
bash go_probe_suite_12d.sh
```

The suite runs six analysis modules: global metrics, per-region best-probe identification, simulated panel redesign, per-chromosome consistency, error analysis by sequence features, and a paper-ready summary. It produces a multi-panel figure at `results/plots/probe_validation_suite.png`.

### 6. Phase 2 Matched-position DANN Training (Cross-Kit Domain Adaptation)

This experiment runs adversarial domain adaptation with Nextera and TruSeq probes drawn from the same genomic positions (kit-intersection BED).

```bash
# Step 1: build matched-position dataset (Nextera + TruSeq intersection)
cd scripts/data
bash nextera_probe.sh                # download Illumina probe manifest
python parse_probe_manifest.py       # parse to centered BED
python compute_dual_labels.py        # compute matched-position dual labels
python build_matched_bed_h5.py \
    --input  ../../data/data_factory/staging/matched_bed_dual/dual_labels.parquet \
    --label_a_col Nextera_label --tag_a nextera --platform_a 1 \
    --label_b_col TruSeq_label  --tag_b truseq  --platform_b 2 \
    --out_dir ../../data/data_factory/final/matched_bed

# Step 2: run matched-position DANN training (warm-started from Phase 1)
cd ../model
bash go_matched_bed_dann.sh
```

The runner reads `SOURCE_H5` / `TARGET_H5` / `VAL_H5` / `PHASE1_CKPT` environment variables for path overrides. Output: `models/matched_bed_dann/deprobe_best_internal.pth`.

### 7. Cross-Platform Evaluation

```bash
cd scripts/validation

# Before DANN: Phase 1 base model on matched-position target
python evaluate_domain_gap.py \
    --checkpoint ../../models/phase1_pure_physics/deprobe_best_internal.pth \
    --source_h5 ../../data/data_factory/final/matched_bed/matched_bed_nextera_master.h5 \
    --target_h5 ../../data/data_factory/final/matched_bed/matched_bed_truseq_master.h5

# After DANN: matched-position DANN model on same target
python evaluate_domain_gap.py \
    --checkpoint ../../models/matched_bed_dann/deprobe_best_internal.pth \
    --source_h5 ../../data/data_factory/final/matched_bed/matched_bed_nextera_master.h5 \
    --target_h5 ../../data/data_factory/final/matched_bed/matched_bed_truseq_master.h5
```

### 8. Baselines and Ablation

```bash
# Traditional ML baselines
cd scripts/validation
python baseline_traditional_ml.py

# BiGRU baseline
cd scripts/model
python baseline_bigru.py \
    --data ../../data/data_factory/final/train/deprobe_train_master.h5 \
    --val_data ../../data/data_factory/final/val/deprobe_val_master.h5 \
    --model_dir ../../models/baseline_bigru

# Ablation study (4 configurations)
cd scripts/validation
bash run_ablation.sh
```

## Training Data

All training data are derived from publicly available Genome in a Bottle (GIAB) reference samples:

| Dataset | Sample | Capture Kit | Platform | Role |
|---------|--------|-------------|----------|------|
| Train | NA12878 (NIST7086) | Nextera Rapid Capture | HiSeq 2500 | Source domain |
| Val | NA12878 (NIST7035) | Nextera Rapid Capture | HiSeq 2500 | Held-out validation |
| DANN Target | NA12878 (NIST-hg001-7001) | TruSeq DNA Exome | HiSeq 2500 | Domain adaptation |

Data are automatically downloaded from the GIAB FTP by the setup scripts.

## Output Conventions

All result artifacts are written to `results/` with three subdirectories:

| Directory | Contents |
|-----------|----------|
| `results/plots/` | PNG / PDF figures (e.g. `model_evaluation.png`, `noise_floor_scatter.png`, `probe_validation_suite.png`) |
| `results/tables/` | CSV / TSV result tables |
| `results/json/` | Metric summaries (e.g. `paper1_main_metrics.json`) |

File names follow a strict convention: lowercase ASCII, underscore as the only separator, no spaces or special characters.

Every output-writing script defensively creates its parent directory tree if missing (`os.makedirs(..., exist_ok=True)`), so a fresh checkout can run end-to-end without manual directory setup.

## Citation

To be submitted.

## License

This project is licensed under the MIT License.

## Contact

Zhikai Xu - zk.xu@ihm.ac.cn
