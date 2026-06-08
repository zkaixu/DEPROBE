#!/bin/bash
# ==============================================================================
# DEPROBE-DNA: NIST7086 Nextera Rapid Capture training data setup
# ==============================================================================
# Downloads BAM, extracts candidate probes, computes 12D thermodynamic priors,
# builds HDF5 dataset for Phase 1 training.
# ==============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# ------------------------------------------------------------------------------
# Paths and assets
# ------------------------------------------------------------------------------
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
LOG_DIR="${PROJECT_ROOT}/logs"

REF_DIR="${DATA_DIR}/hg19"
BAM_DIR="${DATA_DIR}/bams"
BED_DIR="${DATA_DIR}/beds"

STAGE_DIR="${DATA_DIR}/data_factory/staging/train"
FINAL_DIR="${DATA_DIR}/data_factory/final/train"
TEMP_DIR="${DATA_DIR}/temp_processing/train"
META_DIR="${DATA_DIR}/metadata"

DNA_STAGE_CSV="${STAGE_DIR}/deprobe_train_master.csv"
DNA_FINAL_H5="${FINAL_DIR}/deprobe_train_master.h5"

# Reference Genome
REF_FASTA="${REF_DIR}/human_g1k_v37.fasta"
REF_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/human_g1k_v37.fasta.gz"

# Experimental data (NIST7086, training)
BAM_NEXTERA="${BAM_DIR}/project.NIST_NIST7086_H7AP8ADXX_CGTACTAG_2_NA12878.bwa.markDuplicates.bam"
# AWS S3 download route
BAM_S3_URI="s3://giab/data/NA12878/Garvan_NA12878_HG001_HiSeq_Exome/project.NIST_NIST7086_H7AP8ADXX_CGTACTAG_2_NA12878.bwa.markDuplicates.bam"
BAI_S3_URI="s3://giab/data/NA12878/Garvan_NA12878_HG001_HiSeq_Exome/project.NIST_NIST7086_H7AP8ADXX_CGTACTAG_2_NA12878.bwa.markDuplicates.bai"
# Fallback Route: HTTPS (NCBI Mirror)
BAM_HTTPS_URL="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/Garvan_NA12878_HG001_HiSeq_Exome/project.NIST_NIST7086_H7AP8ADXX_CGTACTAG_2_NA12878.bwa.markDuplicates.bam"
BAI_HTTPS_URL="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/Garvan_NA12878_HG001_HiSeq_Exome/project.NIST_NIST7086_H7AP8ADXX_CGTACTAG_2_NA12878.bwa.markDuplicates.bai"

# Target Regions (Nextera Expanded Exome)
BED_NEXTERA="${BED_DIR}/nexterarapidcapture_expandedexome_targetedregions.bed"
# Upgraded to HTTPS
BED_HTTPS_URL="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/Garvan_NA12878_HG001_HiSeq_Exome/nexterarapidcapture_expandedexome_targetedregions.bed.gz"

PROBE_MIN_LEN=120
PROBE_MAX_LEN=120
PROBE_STEP=10
NEG_POOL_SIZE=5
WORKERS=24

# ------------------------------------------------------------------------------
# [SETUP] LOGGING & DIAGNOSTICS
# ------------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_factory_$(date +'%Y%m%d_%H%M%S').log"

echo "===================================================" > "$LOG_FILE"
echo "DEPROBE-DNA TRAIN PIPELINE LOG INITIALIZED AT $(date)" >> "$LOG_FILE"
echo "===================================================" >> "$LOG_FILE"

log_info() { echo -e "${BLUE}[$(date +'%T')]${NC} ${CYAN}${BOLD}[INFO]${NC} $1" | tee -a "$LOG_FILE"; }
log_success() { echo -e "${BLUE}[$(date +'%T')]${NC} ${GREEN}${BOLD}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"; }
log_warning() { echo -e "${BLUE}[$(date +'%T')]${NC} ${YELLOW}${BOLD}[WARNING]${NC} $1" | tee -a "$LOG_FILE"; }
log_fatal() { echo -e "${RED}${BOLD}[FATAL ERROR]${NC} $1" | tee -a "$LOG_FILE"; exit 1; }

# --- DOWNLOAD WRAPPERS ---
download_aria2() {
    local out_path=$1
    local url=$2
    log_info "Fetching via Aria2c (Parallel HTTPS): $(basename "$out_path")"
    aria2c -x 8 -s 8 --retry-wait=2 --max-tries=5 --continue=true \
           -d "$(dirname "$out_path")" -o "$(basename "$out_path")" "$url" >> "$LOG_FILE" 2>&1
}

download_giab() {
    local out_path=$1
    local s3_uri=$2
    local fallback_url=$3
    
    if command -v aws >/dev/null 2>&1; then
        log_info "AWS CLI detected. Engaging high-speed S3 multi-part transfer..."
        if ! aws s3 cp --no-sign-request "$s3_uri" "$out_path" >> "$LOG_FILE" 2>&1; then
            log_warning "S3 Transfer failed. Falling back to Aria2c HTTPS..."
            download_aria2 "$out_path" "$fallback_url"
        fi
    else
        log_warning "AWS CLI not found. Engaging Aria2c HTTPS direct download..."
        download_aria2 "$out_path" "$fallback_url"
    fi
}

check_dependencies() {
    log_info "Auditing system dependencies..."
    command -v python3 >/dev/null 2>&1 || log_fatal "python3 is not installed or not in PATH."
    command -v aria2c >/dev/null 2>&1 || log_fatal "aria2 is required for multi-part downloading."
    command -v samtools >/dev/null 2>&1 || log_fatal "samtools is required for BAM indexing."
    command -v gunzip >/dev/null 2>&1 || log_fatal "gunzip is required for FASTA decompression."
    if ! command -v aws >/dev/null 2>&1; then
        log_warning "AWS CLI not found. Global installation recommended for maximum GIAB BAM speed."
    fi
    log_success "Core system dependencies verified."
}

# ------------------------------------------------------------------------------
# [EXECUTION]
# ------------------------------------------------------------------------------
clear
echo -e "${CYAN}${BOLD}============================================================"
echo "      DEPROBE-DNA: TRAIN DATA FACTORY (NIST7086)"
echo "============================================================${NC}"

mkdir -p "$REF_DIR" "$BAM_DIR" "$BED_DIR" "$STAGE_DIR" "$FINAL_DIR" "$TEMP_DIR" "$META_DIR"
check_dependencies

# --- Reference Genome (Auto-download if missing) ---
if [ ! -f "$REF_FASTA" ]; then
    log_warning "Reference FASTA not found. Downloading (~800MB)..."
    download_aria2 "${REF_FASTA}.gz" "$REF_URL"
    log_info "Decompressing FASTA... This may take a minute."
    gunzip "${REF_FASTA}.gz" || log_warning "gunzip reported trailing garbage, but decompression is usually fine."
    log_info "Building FASTA index (samtools faidx)..."
    samtools faidx "$REF_FASTA"
    log_success "Reference Genome prepared in ${REF_DIR}."
else
    log_info "Reference Genome already exists. Skipping download."
    if [ ! -f "${REF_FASTA}.fai" ]; then
        log_warning "FASTA index missing. Rebuilding..."
        samtools faidx "$REF_FASTA"
    fi
fi

# --- Disk Space Pre-check ---
REQUIRED_GB=20
AVAIL_KB=$(df --output=avail "$DATA_DIR" | tail -1)
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
    log_fatal "Insufficient disk space: ${AVAIL_GB}GB available, ${REQUIRED_GB}GB required."
fi
log_info "Disk space check passed: ${AVAIL_GB}GB available."


log_info "Preparing NIST7086 alignment assets..."

if [ ! -f "$BAM_NEXTERA" ]; then
    log_warning "Downloading NIST7086 BAM..."
    download_giab "$BAM_NEXTERA" "$BAM_S3_URI" "$BAM_HTTPS_URL"
    
    log_info "Running samtools quickcheck to verify BAM integrity..."
    if ! samtools quickcheck -v "$BAM_NEXTERA" >> "$LOG_FILE" 2>&1; then
        rm -f "$BAM_NEXTERA" "${BAM_NEXTERA}.bai"
        log_fatal "BAM corruption detected (Missing EOF)! File purged. Please re-run the pipeline."
    fi
    log_success "BAM structural integrity validated."

    log_info "Downloading BAI index..."
    download_giab "${BAM_NEXTERA}.bai" "$BAI_S3_URI" "$BAI_HTTPS_URL"
    log_success "NIST7086 alignment assets prepared."
else
    log_info "NIST7086 BAM already exists. Verifying integrity..."
    if ! samtools quickcheck -v "$BAM_NEXTERA" >> "$LOG_FILE" 2>&1; then
        log_fatal "Existing BAM is corrupted! Please manually delete ${BAM_NEXTERA} and re-run."
    fi
    if [ ! -f "${BAM_NEXTERA}.bai" ]; then
        log_warning "BAI index missing. Building index via samtools..."
        samtools index "$BAM_NEXTERA"
    fi
fi

if [ ! -f "$BED_NEXTERA" ]; then
    log_warning "Downloading Nextera BED file..."
    download_aria2 "${BED_NEXTERA}.gz" "$BED_HTTPS_URL"
    gunzip -c "${BED_NEXTERA}.gz" > "$BED_NEXTERA"
    rm -f "${BED_NEXTERA}.gz"
    log_success "Nextera BED securely extracted."
else
    log_info "Nextera BED already exists."
fi

log_info "Synchronizing Metadata (Platform_ID = 1)..."
cat <<EOF > "${META_DIR}/metadata_train.csv"
Sample_Name,BAM_Path,BED_Path,Platform_ID
NA12878_Nextera_7086,${BAM_NEXTERA},${BED_NEXTERA},1
EOF

log_warning "Cleaning previous staging and final outputs..."
rm -f "$DNA_STAGE_CSV" "$DNA_FINAL_H5"
rm -rf "${TEMP_DIR:?}"/*
sync && sleep 2

log_info "Phase 4: feature extraction (12D priors)..."
python3 build_pretrain_dataset.py \
    --meta "${META_DIR}/metadata_train.csv" \
    --ref "$REF_FASTA" \
    --mode dna \
    --out "$DNA_STAGE_CSV" \
    --temp "$TEMP_DIR" \
    --min_len $PROBE_MIN_LEN --max_len $PROBE_MAX_LEN --step $PROBE_STEP \
    --workers $WORKERS >> "$LOG_FILE" 2>&1

log_success "TRAIN Staging completed."

sync && sleep 5

log_info "Phase 5: HDF5 Digitization..."
python3 build_h5.py \
    --input "$DNA_STAGE_CSV" \
    --output "$DNA_FINAL_H5" \
    --ref "$REF_FASTA" \
    --neg_pool $NEG_POOL_SIZE >> "$LOG_FILE" 2>&1

if [ -f "$DNA_FINAL_H5" ]; then
    log_success "Training HDF5 location: ${DNA_FINAL_H5}"
else
    log_fatal "HDF5 output missing."
fi