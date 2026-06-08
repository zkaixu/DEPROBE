#!/bin/bash
# ==============================================================================
# DEPROBE-DNA: Probe Validation Suite (12D)
# ==============================================================================
# 6 analysis modules covering metrics, per-region selection, panel redesign, per-chromosome consistency, sequence-feature error stratification, and summary reporting.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

TRUE_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LOG_DIR="${TRUE_PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/probe_suite_${TIMESTAMP}.log"

CHECKPOINT="${TRUE_PROJECT_ROOT}/models/phase1_pure_physics/deprobe_best_internal.pth"
PROBE_H5="${TRUE_PROJECT_ROOT}/data/data_factory/final/probe_validation/deprobe_probe_val_master.h5"
STAGING_CSV="${TRUE_PROJECT_ROOT}/data/data_factory/staging/probe_validation/deprobe_probe_val_master.csv"
PROBE_MAPPING="${TRUE_PROJECT_ROOT}/data/beds/nextera_probe_mapping.csv"
TARGET_BED="${TRUE_PROJECT_ROOT}/data/beds/nexterarapidcapture_expandedexome_targetedregions.bed"
# Extra H5s used by roc_analysis.py for the Int/Ext discrimination panels.
INT_H5="${TRUE_PROJECT_ROOT}/data/data_factory/final/train/deprobe_train_master.h5"
EXT_H5="${TRUE_PROJECT_ROOT}/data/data_factory/final/val/deprobe_val_master.h5"
OUTPUT_DIR="${LOG_DIR}"
PRIOR_DIM=12  # 12-dimensional thermodynamic prior

echo -e "\033[1;36m======================================================================="
echo -e " DEPROBE-DNA: PROBE VALIDATION SUITE"
echo -e "=======================================================================\033[0m"
echo -e " Checkpoint    : $(basename "$CHECKPOINT")"
echo -e " Probe H5      : $(basename "$PROBE_H5") ($(python3 -c "import h5py; print(h5py.File('$PROBE_H5','r')['efficiency'].shape[0])" 2>/dev/null) probes)"
echo -e " Log File      : \033[1;33m$LOG_FILE\033[0m"
echo -e "\033[1;36m=======================================================================\033[0m"

for file in "$CHECKPOINT" "$PROBE_H5" "$STAGING_CSV" "$PROBE_MAPPING" "$TARGET_BED" "$INT_H5" "$EXT_H5"; do
    if [ ! -f "$file" ]; then
        echo -e "\033[0;31m[FATAL] Missing: $file\033[0m"
        exit 1
    fi
done

# ----------------------------------------------------------------------
# Step 1: real-probe validation suite
#   → 6 CSVs + probe_validation_full.json + probe_validation_suite.png
# ----------------------------------------------------------------------
echo -e "\033[1;36m======================================================================="
echo -e " STEP 1/2: PROBE VALIDATION SUITE  (probe_validation_suite.py)"
echo -e "=======================================================================\033[0m"

python3 probe_validation_suite.py \
    --checkpoint "$CHECKPOINT" \
    --h5 "$PROBE_H5" \
    --staging_csv "$STAGING_CSV" \
    --probe_mapping "$PROBE_MAPPING" \
    --target_bed "$TARGET_BED" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4096 \
    2>&1 | tee "$LOG_FILE"

# ----------------------------------------------------------------------
# Step 2: ROC / PR curve analysis across Int / Ext / Real-Probe splits
#   → results/plots/roc_curves.png + roc_metrics.csv + roc_metrics.json
# ----------------------------------------------------------------------
echo -e "\033[1;36m======================================================================="
echo -e " STEP 2/2: ROC / PR DISCRIMINATION ANALYSIS  (roc_analysis.py)"
echo -e "=======================================================================\033[0m"

python3 roc_analysis.py \
    --checkpoint "$CHECKPOINT" \
    --int_data "$INT_H5" \
    --ext_data "$EXT_H5" \
    --probe_data "$PROBE_H5" \
    --topk 0.10 0.20 0.50 \
    --prior_dim "$PRIOR_DIM" \
    --batch_size 2048 \
    2>&1 | tee -a "$LOG_FILE"

echo -e "\033[1;36m=======================================================================\033[0m"
echo -e "\033[1;32m SUITE COMPLETE. Results in: $LOG_FILE\033[0m"
echo -e " Figures:"
echo -e "   ${OUTPUT_DIR}/probe_validation_suite.png"
echo -e "   ${TRUE_PROJECT_ROOT}/results/plots/roc_curves.png"
echo -e " Tables / JSON:"
echo -e "   ${TRUE_PROJECT_ROOT}/results/tables/roc_metrics.csv"
echo -e "   ${TRUE_PROJECT_ROOT}/results/json/roc_metrics.json"
echo -e "\033[1;36m=======================================================================\033[0m"
