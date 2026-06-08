#!/bin/bash
# ==============================================================================
# DEPROBE-DNA: Ablation L3 (real-probe) evaluation runner
# ==============================================================================
# After all 4 ablation configurations finish training (run_ablation.sh produces
# the *_summary.json files), this script loads each best.pth checkpoint and
# evaluates it on the 344,090-probe real-probe set using the same
# probe_validation_suite pipeline used for the main DEPROBE-DNA model and BiGRU
# baseline. Output CSVs are suffixed with _abl_<config> to avoid collisions.
#
# Each evaluation takes ~5 min (inference only, no retraining).
# 4 configs total -> ~20 min wall-clock.
#
# Usage:
#   bash run_ablation_l3_eval.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TRUE_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROBE_H5="${TRUE_PROJECT_ROOT}/data/data_factory/final/probe_validation/deprobe_probe_val_master.h5"
STAGING_CSV="${TRUE_PROJECT_ROOT}/data/data_factory/staging/probe_validation/deprobe_probe_val_master.csv"
PROBE_MAPPING="${TRUE_PROJECT_ROOT}/data/beds/nextera_probe_mapping.csv"
TARGET_BED="${TRUE_PROJECT_ROOT}/data/beds/nexterarapidcapture_expandedexome_targetedregions.bed"
ABLATION_DIR="${TRUE_PROJECT_ROOT}/models/ablation"
LOG_DIR="${TRUE_PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

cd "${TRUE_PROJECT_ROOT}/scripts/validation"

echo "======================================================================"
echo "  DEPROBE-DNA: Ablation L3 (real-probe) evaluation"
echo "======================================================================"
echo "  Probe H5      : $(basename "$PROBE_H5")"
echo "  Ablation dir  : $ABLATION_DIR"
echo "  Output CSVs   : ${TRUE_PROJECT_ROOT}/results/tables/probe_global_metrics_abl_*.csv"
echo "======================================================================"

CONFIGS=(no_physics late_fusion_only no_rnc no_physics_no_rnc)

for config in "${CONFIGS[@]}"; do
    CKPT="${ABLATION_DIR}/ablation_${config}_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo ""
        echo "[SKIP] Configuration ${config}: checkpoint not found at:"
        echo "       $CKPT"
        continue
    fi

    echo ""
    echo "------------------------------------------------------"
    echo "[$(date)] Evaluating ${config} on real-probe set"
    echo "------------------------------------------------------"

    python3 probe_validation_suite_ablation.py \
        --checkpoint "$CKPT" \
        --h5 "$PROBE_H5" \
        --staging_csv "$STAGING_CSV" \
        --probe_mapping "$PROBE_MAPPING" \
        --target_bed "$TARGET_BED" \
        --output_dir "$LOG_DIR" \
        --batch_size 4096 \
        --output_suffix "_abl_${config}" \
        2>&1 | tee "${LOG_DIR}/ablation_l3_eval_${config}.log"

    echo "[$(date)] ${config} L3 evaluation complete."
done

echo ""
echo "======================================================================"
echo "  All ablation L3 evaluations complete"
echo "======================================================================"
echo "  Summary CSVs:"
ls "${TRUE_PROJECT_ROOT}/results/tables/probe_global_metrics_abl_"*.csv 2>/dev/null || \
    echo "    (no abl CSVs found - check evaluation logs)"
echo ""
echo "  To assemble Table 5 numbers, read each CSV's row:"
for config in "${CONFIGS[@]}"; do
    csv="${TRUE_PROJECT_ROOT}/results/tables/probe_global_metrics_abl_${config}.csv"
    if [ -f "$csv" ]; then
        echo "    abl_${config}:  $csv"
    fi
done
echo "======================================================================"
