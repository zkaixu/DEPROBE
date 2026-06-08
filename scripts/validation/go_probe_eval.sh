#!/bin/bash
# ==============================================================================
# DEPROBE-DNA: Probe Validation Evaluator
# ==============================================================================
# Runs the Phase 1 model on REAL Nextera Expanded Exome probes (344K)
# and evaluates prediction quality against NIST7035 BAM ground truth.
#
# Runs the probe-validation evaluator on a checkpoint:
#   "Model trained on sliding windows generalizes to real commercial probes"
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

TRUE_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LOG_DIR="${TRUE_PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/probe_eval_${TIMESTAMP}.log"

# --- Assets ---
CHECKPOINT="${TRUE_PROJECT_ROOT}/models/phase1_pure_physics/deprobe_best_internal.pth"
PROBE_H5="${TRUE_PROJECT_ROOT}/data/data_factory/final/probe_validation/deprobe_probe_val_master.h5"
OUTPUT_DIR="${TRUE_PROJECT_ROOT}/logs"

# --- Pre-flight ---
echo -e "\033[1;36m======================================================================="
echo -e " DEPROBE-DNA: PROBE VALIDATION EVALUATION"
echo -e "=======================================================================\033[0m"
echo -e " Checkpoint    : $CHECKPOINT"
echo -e " Probe H5      : $PROBE_H5"
echo -e " Output Dir    : $OUTPUT_DIR"
echo -e " Log File      : \033[1;33m$LOG_FILE\033[0m"
echo -e "\033[1;36m=======================================================================\033[0m"

for file in "$CHECKPOINT" "$PROBE_H5"; do
    if [ ! -f "$file" ]; then
        echo -e "\033[0;31m[FATAL] Required file missing: $file\033[0m"
        exit 1
    fi
done

# --- Run evaluation (foreground, takes ~30 seconds) ---
echo -e "\033[1;34m[INFO]\033[0m Running inference on 344K real probes..."

python3 evaluate_model.py \
    --checkpoint "$CHECKPOINT" \
    --data "$PROBE_H5" \
    --batch_size 4096 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo -e "\033[1;36m=======================================================================\033[0m"
echo -e "\033[1;32m EVALUATION COMPLETE\033[0m"
echo -e " Full log: $LOG_FILE"
echo -e "\033[1;36m=======================================================================\033[0m"
