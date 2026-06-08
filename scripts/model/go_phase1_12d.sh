#!/bin/bash
# ==============================================================================
# DEPROBE-DNA: Phase 1 pre-training runner
# ==============================================================================
# Strategy: pure physics learning, DANN disabled.
# Execution: detached background process via nohup with persistent logging.
# Hardware: tested on RTX 5090.
# Location: scripts/model/
# ==============================================================================

set -euo pipefail

# 1. Lock execution context to the script's directory (scripts/model/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# 2. Dynamically resolve the True Project Root (2 levels up from scripts/model/)
TRUE_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ------------------------------------------------------------------------------
# 3. Path Definitions & Dynamic Log Naming (Routed to True Root)
# ------------------------------------------------------------------------------
LOG_DIR="${TRUE_PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Dynamically name the log file to identify the phase, state, and timestamp
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/train_phase1_${TIMESTAMP}.log"

# Dataset Paths
TRAIN_H5="${TRUE_PROJECT_ROOT}/data/data_factory/final/train/deprobe_train_master.h5"
VAL_H5="${TRUE_PROJECT_ROOT}/data/data_factory/final/val/deprobe_val_master.h5"

# Phase 1 checkpoint directory
MODEL_DIR="${TRUE_PROJECT_ROOT}/models/phase1_pure_physics"

# ------------------------------------------------------------------------------
# 4. Pre-flight Diagnostics
# ------------------------------------------------------------------------------
echo -e "\033[1;36m=======================================================================\033[0m"
echo -e "\033[1;32m INITIATING DEPROBE-DNA - PHASE 1 12D TRAINING (AUTO-CONVERGE)\033[0m"
echo -e "\033[1;36m=======================================================================\033[0m"
echo -e " Task                : Pure Physics Pre-training (DANN Disabled)"
echo -e " Script Location     : $SCRIPT_DIR"
echo -e " Project Root        : $TRUE_PROJECT_ROOT"
echo -e " Source Domain       : Nextera (NIST7086)"
echo -e " Holdout Val         : Nextera (NIST7035)"
echo -e " Target Model Dir    : $MODEL_DIR"
echo -e " Diagnostic Log File : \033[1;33m$LOG_FILE\033[0m"
echo -e "\033[1;36m=======================================================================\033[0m"

# Validate H5 assets
for file in "$TRAIN_H5" "$VAL_H5"; do
    if [ ! -f "$file" ]; then
        echo -e "\033[0;31m[FATAL] Required dataset missing: $file\033[0m"
        echo "Please ensure the BWA Miner has successfully packed the H5 files."
        exit 1
    fi
done

# Ensure model output directory exists
mkdir -p "$MODEL_DIR"

# ------------------------------------------------------------------------------
# 5. Detached Background Execution
# ------------------------------------------------------------------------------
echo -e "\033[1;34m[INFO]\033[0m Starting Phase 1 training in detached mode..."
echo -e "\033[1;34m[INFO]\033[0m You can safely close this terminal once the PID is displayed."

# Execute main_12d.py via nohup (main_12d.py is in the current SCRIPT_DIR)
nohup python3 main_12d.py \
    --data "$TRAIN_H5" \
    --val_data "$VAL_H5" \
    --model_dir "$MODEL_DIR" \
    --epochs 1000 \
    --batch 1024 \
    --lr 1e-4 \
    --tau 0.1 \
    --rnc_margin 0.3 \
    --resume_from "$MODEL_DIR/deprobe_best_internal.pth" > "$LOG_FILE" 2>&1 &

# Capture the Process ID (PID) of the background job
TRAIN_PID=$!

echo -e "\033[1;36m=======================================================================\033[0m"
echo -e "\033[1;32m ✅ Background Process Successfully Dispatched.\033[0m"
echo -e " Training process ID (PID) : \033[1m$TRAIN_PID\033[0m"
echo -e ""
echo -e " To monitor the training stream in real-time, execute:"
echo -e " \033[1;33mtail -f $LOG_FILE\033[0m"
echo -e ""
echo -e " To gracefully terminate the training early, execute:"
echo -e " \033[1;31mkill -9 $TRAIN_PID\033[0m"
echo -e "\033[1;36m=======================================================================\033[0m"