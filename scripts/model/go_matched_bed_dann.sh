#!/bin/bash
# ==============================================================================
# DEPROBE-DNA: MATCHED-POSITION DANN RUNNER
# ==============================================================================
# Strategy: Adversarial training with source AND target probes drawn from the
#           SAME genomic positions (Nextera ∩ TruSeq intersection BED).
#
#           If DANN successfully bridges the gap on full-BED data but FAILS on
#           matched-position data, the gap is not covariate shift but a
#           label-function incompatibility between the two kits. This is the
#           mechanistic claim of this work.
#
# Inputs : matched_bed_<kit_a>_master.h5  (source, labelled)
#          matched_bed_<kit_b>_master.h5  (target, labels ignored by DANN
#                                          training, used at evaluation time)
# Output : models/matched_bed_dann/deprobe_best_internal.pth
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

TRUE_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ------------------------------------------------------------------------------
# Path Definitions (override via environment variables if needed)
# ------------------------------------------------------------------------------
LOG_DIR="${TRUE_PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/matched_bed_dann_${TIMESTAMP}.log"

# Default H5 layout: matched_bed_<kit>_master.h5 sit under data_factory/final/matched_bed/.
# Override SOURCE_H5 / TARGET_H5 in the env to point at different kits if needed.
MATCHED_DIR="${TRUE_PROJECT_ROOT}/data/data_factory/final/matched_bed"
SOURCE_H5="${SOURCE_H5:-${MATCHED_DIR}/matched_bed_nextera_master.h5}"
TARGET_H5="${TARGET_H5:-${MATCHED_DIR}/matched_bed_truseq_master.h5}"
# Use the source-side H5 for held-out validation by default; user can override.
VAL_H5="${VAL_H5:-${SOURCE_H5}}"

# Phase-1 12D weights warm-start the DANN training.
PHASE1_CKPT="${PHASE1_CKPT:-${TRUE_PROJECT_ROOT}/models/phase1_pure_physics/deprobe_best_internal.pth}"
MODEL_DIR="${MODEL_DIR:-${TRUE_PROJECT_ROOT}/models/matched_bed_dann}"

# ------------------------------------------------------------------------------
# Pre-flight
# ------------------------------------------------------------------------------
echo -e "\033[1;36m=======================================================================\033[0m"
echo -e "\033[1;35m INITIATING DEPROBE-DNA - MATCHED-POSITION DANN (cross-platform analysis)\033[0m"
echo -e "\033[1;36m=======================================================================\033[0m"
echo -e " Source H5 (labelled)  : $SOURCE_H5"
echo -e " Target H5 (DANN)      : $TARGET_H5"
echo -e " Validation H5         : $VAL_H5"
echo -e " Phase-1 base weights  : $PHASE1_CKPT"
echo -e " Output model dir      : $MODEL_DIR"
echo -e " Diagnostic log file   : \033[1;33m$LOG_FILE\033[0m"
echo -e "\033[1;36m=======================================================================\033[0m"

for file in "$SOURCE_H5" "$TARGET_H5" "$VAL_H5" "$PHASE1_CKPT"; do
    if [ ! -f "$file" ]; then
        echo -e "\033[0;31m[FATAL] Required file missing: $file\033[0m"
        exit 1
    fi
done

mkdir -p "$MODEL_DIR"

TRAINER="main_12d.py"
echo -e " Trainer entry-point   : $TRAINER"

# ------------------------------------------------------------------------------
# Detached Background Execution
# ------------------------------------------------------------------------------
echo -e "\033[1;34m[INFO]\033[0m Starting matched-position DANN training in detached mode..."

nohup python3 "$TRAINER" \
    --data "$SOURCE_H5" \
    --target_data "$TARGET_H5" \
    --val_data "$VAL_H5" \
    --model_dir "$MODEL_DIR" \
    --finetune_from "$PHASE1_CKPT" \
    --epochs 200 \
    --batch 1024 \
    --lr 5e-5 \
    --tau 0.1 \
    --rnc_margin 0.3 > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo -e "\033[1;36m=======================================================================\033[0m"
echo -e "\033[1;32m ✅ Matched-position DANN dispatched.\033[0m"
echo -e " Training process ID (PID) : \033[1m$TRAIN_PID\033[0m"
echo -e ""
echo -e " Monitor:"
echo -e " \033[1;33mtail -f $LOG_FILE\033[0m"
echo -e ""
echo -e " Terminate (graceful):"
echo -e " \033[1;31mkill $TRAIN_PID\033[0m"
echo -e "\033[1;36m=======================================================================\033[0m"
