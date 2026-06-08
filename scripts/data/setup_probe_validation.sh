#!/bin/bash
# ==============================================================================
# DEPROBE-DNA: Probe Validation Data Pipeline
# ==============================================================================
# Uses REAL Nextera Expanded Exome probe coordinates (347K probes, 94bp)
# to create an H5 dataset for model inference and validation.
#
# Pipeline:
#   1. parse_probe_manifest.py  → centered BED (120bp per probe)
#   2. build_pretrain_dataset.py → staging CSV (reuses calc_efficiency,
#      extract_sequences, calc_priors, data_fusion with --min/max_len 120)
#   3. build_h5.py              → final H5 for model inference
#
# Validation BAM: NIST7035 (independent replicate, never seen during training)
# ==============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
LOG_DIR="${PROJECT_ROOT}/logs"

REF_DIR="${DATA_DIR}/hg19"
BAM_DIR="${DATA_DIR}/bams"
BED_DIR="${DATA_DIR}/beds"

STAGE_DIR="${DATA_DIR}/data_factory/staging/probe_validation"
FINAL_DIR="${DATA_DIR}/data_factory/final/probe_validation"
TEMP_DIR="${DATA_DIR}/temp_processing/probe_validation"
META_DIR="${DATA_DIR}/metadata"

# --- Input Assets ---
PROBE_MANIFEST="${BED_DIR}/nexterarapidcapture_expandedexome_probes.txt"
REF_FASTA="${REF_DIR}/human_g1k_v37.fasta"
BAM_7035="${BAM_DIR}/project.NIST_NIST7035_H7AP8ADXX_TAAGGCGA_2_NA12878.bwa.markDuplicates.bam"

# --- Generated Assets ---
CENTERED_BED="${BED_DIR}/nextera_probes_centered_120bp.bed"
PROBE_MAPPING="${BED_DIR}/nextera_probe_mapping.csv"
STAGE_CSV="${STAGE_DIR}/deprobe_probe_val_master.csv"
FINAL_H5="${FINAL_DIR}/deprobe_probe_val_master.h5"

# --- Pipeline Params (exactly 1 window per 120bp centered region) ---
PROBE_MIN_LEN=120
PROBE_MAX_LEN=120
PROBE_STEP=120
NEG_POOL_SIZE=5
WORKERS=24

# --- Logging ---
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/probe_validation_$(date +'%Y%m%d_%H%M%S').log"

log_info()    { local ts=$(date +'%T'); echo -e "${BLUE}[${ts}]${NC} ${CYAN}${BOLD}[INFO]${NC} $1"; echo "[${ts}] [INFO] $1" >> "$LOG_FILE"; }
log_success() { local ts=$(date +'%T'); echo -e "${BLUE}[${ts}]${NC} ${GREEN}${BOLD}[SUCCESS]${NC} $1"; echo "[${ts}] [SUCCESS] $1" >> "$LOG_FILE"; }
log_warning() { local ts=$(date +'%T'); echo -e "${BLUE}[${ts}]${NC} ${YELLOW}${BOLD}[WARNING]${NC} $1"; echo "[${ts}] [WARNING] $1" >> "$LOG_FILE"; }
log_fatal()   { echo -e "${RED}${BOLD}[FATAL]${NC} $1"; echo "[FATAL] $1" >> "$LOG_FILE"; exit 1; }

# ==============================================================================
clear
echo -e "${CYAN}${BOLD}"
echo "============================================================"
echo "  DEPROBE-DNA: PROBE VALIDATION DATA PIPELINE"
echo "============================================================"
echo -e "${NC}"
log_info "Log file: $LOG_FILE"

# --- Pre-flight Checks ---
[ -f "$PROBE_MANIFEST" ] || log_fatal "Probe manifest not found: $PROBE_MANIFEST (run nextera_probe.sh first)"
[ -f "$REF_FASTA" ]      || log_fatal "Reference FASTA not found: $REF_FASTA"
[ -f "${REF_FASTA}.fai" ] || log_fatal "FASTA index not found: ${REF_FASTA}.fai"
[ -f "$BAM_7035" ]        || log_fatal "Validation BAM not found: $BAM_7035"

mkdir -p "$STAGE_DIR" "$FINAL_DIR" "$TEMP_DIR" "$META_DIR"

# ==============================================================================
# Phase 1: Convert Probe Manifest → Centered BED (120bp)
# ==============================================================================
log_info "PHASE 1: Probe manifest → 120bp centered BED..."

if [ -f "$CENTERED_BED" ] && [ -f "$PROBE_MAPPING" ]; then
    log_info "Centered BED already exists: $(wc -l < "$CENTERED_BED") probes"
else
    python3 parse_probe_manifest.py \
        --manifest "$PROBE_MANIFEST" \
        --fai "${REF_FASTA}.fai" \
        --bed_out "$CENTERED_BED" \
        --map_out "$PROBE_MAPPING" \
        --window 120 >> "$LOG_FILE" 2>&1

    [ -s "$CENTERED_BED" ] || log_fatal "Centered BED is empty."
    log_success "Centered BED: $(wc -l < "$CENTERED_BED") probes → $CENTERED_BED"
fi

# ==============================================================================
# Phase 2: Metadata
# ==============================================================================
log_info "Phase 2: creating metadata..."
cat <<EOF > "${META_DIR}/metadata_probe_validation.csv"
Sample_Name,BAM_Path,BED_Path,Platform_ID
NA12878_ProbeVal_7035,${BAM_7035},${CENTERED_BED},1
EOF

# ==============================================================================
# Phase 3: Feature Extraction (reuse full pipeline)
# ==============================================================================
log_info "PHASE 3: Feature extraction (12D pipeline on real probe positions)..."
log_info "  BAM: NIST7035 (validation replicate)"
log_info "  Probes: $(wc -l < "$CENTERED_BED") centered to 120bp"
log_info "  Workers: ${WORKERS}"

rm -rf "${TEMP_DIR:?}"/*
sync && sleep 2

start_time=$(date +%s)
python3 build_pretrain_dataset.py \
    --meta "${META_DIR}/metadata_probe_validation.csv" \
    --ref "$REF_FASTA" \
    --mode dna \
    --out "$STAGE_CSV" \
    --temp "$TEMP_DIR" \
    --min_len $PROBE_MIN_LEN --max_len $PROBE_MAX_LEN --step $PROBE_STEP \
    --workers $WORKERS >> "$LOG_FILE" 2>&1

end_time=$(date +%s)
log_success "Feature extraction completed in $(( (end_time - start_time) / 60 )) minutes."

# ==============================================================================
# Phase 4: HDF5 Generation
# ==============================================================================
log_info "PHASE 4: HDF5 generation..."

sync && sleep 5

start_h5=$(date +%s)
python3 build_h5.py \
    --input "$STAGE_CSV" \
    --output "$FINAL_H5" \
    --ref "$REF_FASTA" \
    --neg_pool $NEG_POOL_SIZE >> "$LOG_FILE" 2>&1

end_h5=$(date +%s)
log_success "H5 generation completed in $(( (end_h5 - start_h5) / 60 )) minutes."

# ==============================================================================
# Phase 5: Quick Sanity Check
# ==============================================================================
log_info "PHASE 5: Sanity check..."
python3 -c "
import h5py
h5 = h5py.File('${FINAL_H5}', 'r')
print(f'  sequences: {h5[\"sequences\"].shape}')
print(f'  priors:    {h5[\"priors\"].shape}')
print(f'  labels:    {h5[\"efficiencies\"].shape}')
print(f'  probe_ids: {h5[\"probe_ids\"].shape}')
n = h5['sequences'].shape[0]
print(f'  Total probes in H5: {n}')
h5.close()
" >> "$LOG_FILE" 2>&1

python3 -c "
import h5py
h5 = h5py.File('${FINAL_H5}', 'r')
print(f'sequences: {h5[\"sequences\"].shape}')
print(f'priors:    {h5[\"priors\"].shape}')
print(f'labels:    {h5[\"efficiencies\"].shape}')
print(f'probe_ids: {h5[\"probe_ids\"].shape}')
h5.close()
"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo -e "${CYAN}${BOLD}============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================${NC}"
log_success "Probe validation H5: ${FINAL_H5}"
log_info "Probe mapping CSV: ${PROBE_MAPPING}"
log_info "Next step: run model inference with Phase 1 checkpoint on this H5"
