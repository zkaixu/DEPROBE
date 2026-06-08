#!/bin/bash
# ==============================================================================
# DEPROBE-DNA: ABLATION STUDY RUNNER (Plateau-Matched Protocol)
# ==============================================================================
# Runs 4 ablation experiments sequentially on a single GPU.
# Each experiment trains under the SAME protocol as the main model:
#   - AdamW + SAM (SAM activates after first LR reduction, like main_12d.py)
#   - ReduceLROnPlateau (factor=0.5, patience=3, threshold=5e-4 abs, min_lr=1e-6)
#   - Plateau-matched termination: 5 epochs at min_lr without improvement
#   - Failsafe: 40 non-improvement epochs total, OR 200-epoch hard ceiling
#
# Per-config epoch count varies (each terminates at its own true plateau).
# Estimated total: 30-100h depending on per-epoch wall-clock with SAM enabled.
# Run pilot (D, 5 epochs) first to measure per-epoch time.
#
# Experiments:
#   A. No physics priors (sequence-only with RNC)
#   B. Late fusion only (no early physics token, with RNC)
#   C. No RNC (full physics, Huber only)
#   D. No physics + No RNC (pure sequence + Huber)
#
# Usage:
#   bash run_ablation.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TRUE_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_SCRIPT="${TRUE_PROJECT_ROOT}/scripts/model/main_ab.py"

# Datasets
TRAIN_H5="${TRUE_PROJECT_ROOT}/data/data_factory/final/train/deprobe_train_master.h5"
VAL_H5="${TRUE_PROJECT_ROOT}/data/data_factory/final/val/deprobe_val_master.h5"

# Output
ABLATION_DIR="${TRUE_PROJECT_ROOT}/models/ablation"
LOG_DIR="${TRUE_PROJECT_ROOT}/logs"
mkdir -p "$ABLATION_DIR" "$LOG_DIR"

# Common args
# epochs=200 is a hard ceiling; plateau-matched termination usually fires earlier (~30-100 epochs per config).
COMMON="--data $TRAIN_H5 --val_data $VAL_H5 --model_dir $ABLATION_DIR --epochs 200 --batch 1024 --lr 1e-4 --tau 0.1 --rnc_margin 0.3"

echo "======================================================================"
echo "  DEPROBE-DNA: ABLATION STUDY (plateau-matched, 4 experiments)"
echo "======================================================================"
echo "  Train:        $TRAIN_H5"
echo "  Val:          $VAL_H5"
echo "  Output:       $ABLATION_DIR"
echo "  Epoch ceiling: 200 (plateau-matched termination usually fires earlier)"
echo "======================================================================"

# Must cd to model dir so imports work
cd "${TRUE_PROJECT_ROOT}/scripts/model"

# ------------------------------------------------------------------
# Resumability: skip a config whose summary JSON already exists.
# Each config writes ablation_<name>_summary.json on completion (in main_ab.py).
# Killing the script mid-run preserves all previously-completed configs;
# re-running this script will resume from the first incomplete config.
# ------------------------------------------------------------------
run_or_skip() {
    local label="$1"          # e.g. "A: NO_PHYSICS"
    local config_name="$2"    # e.g. "no_physics"
    local logfile="$3"        # e.g. "ablation_A_no_physics.log"
    shift 3
    local summary_path="${ABLATION_DIR}/ablation_${config_name}_summary.json"

    if [ -f "$summary_path" ]; then
        echo ""
        echo "[$(date)] >>> SKIP Experiment ${label}: summary already exists at:"
        echo "                ${summary_path}"
        return
    fi

    echo ""
    echo "[$(date)] >>> Starting Experiment ${label}"
    python3 main_ab.py $COMMON "$@" > "${LOG_DIR}/${logfile}" 2>&1
    echo "[$(date)] >>> Experiment ${label} complete."
}

# ------------------------------------------------------------------
# Experiment A: No physics priors (sequence + RNC)
# ------------------------------------------------------------------
run_or_skip "A: NO_PHYSICS" "no_physics" "ablation_A_no_physics.log" --no_physics

# ------------------------------------------------------------------
# Experiment B: Late fusion only (no early fusion physics token, with RNC)
# ------------------------------------------------------------------
run_or_skip "B: LATE_FUSION_ONLY" "late_fusion_only" "ablation_B_late_fusion_only.log" --late_fusion_only

# ------------------------------------------------------------------
# Experiment C: No RNC (full physics, Huber only)
# ------------------------------------------------------------------
run_or_skip "C: NO_RNC" "no_rnc" "ablation_C_no_rnc.log" --no_rnc

# ------------------------------------------------------------------
# Experiment D: No physics + No RNC (pure sequence + Huber)
# ------------------------------------------------------------------
run_or_skip "D: NO_PHYSICS + NO_RNC" "no_physics_no_rnc" "ablation_D_no_physics_no_rnc.log" --no_physics --no_rnc

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "  ALL ABLATION EXPERIMENTS COMPLETE"
echo "======================================================================"
echo "  Results saved in: $ABLATION_DIR"
echo "  Logs:"
echo "    A: ${LOG_DIR}/ablation_A_no_physics.log"
echo "    B: ${LOG_DIR}/ablation_B_late_fusion_only.log"
echo "    C: ${LOG_DIR}/ablation_C_no_rnc.log"
echo "    D: ${LOG_DIR}/ablation_D_no_physics_no_rnc.log"
echo ""
echo "  Summary JSONs:"
ls -la "${ABLATION_DIR}"/ablation_*_summary.json 2>/dev/null || echo "    (none found)"
echo "======================================================================"
