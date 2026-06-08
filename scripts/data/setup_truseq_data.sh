#!/bin/bash
# ==============================================================================
# DEPROBE-DNA: NIST-hg001-7001 TruSeq Exome DANN target data setup
# ==============================================================================
# Pipeline: download BAM, extract candidate probes, compute 12D thermodynamic
# priors via Primer3, build HDF5 dataset for DANN target adaptation.
# ==============================================================================

# ------------------------------------------------------------------------------
# [SYSTEM] STRICT EXECUTION & WORKING DIRECTORY LOCK
# ------------------------------------------------------------------------------
# Exit immediately on error, undefined variables, or pipe failures
set -euo pipefail

# Lock the execution context to the directory where this script resides.
# This ensures Python scripts are found even if this bash script is executed from elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# --- Terminal colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# ------------------------------------------------------------------------------
# [CONFIG] GLOBAL PATH ORCHESTRATION (ABSOLUTE PATHS)
# ------------------------------------------------------------------------------
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
LOG_DIR="${PROJECT_ROOT}/logs"

# Subdirectory paths for data files
REF_DIR="${DATA_DIR}/hg19"
BAM_DIR="${DATA_DIR}/bams"
BED_DIR="${DATA_DIR}/beds"

# Pipeline Staging Directories
STAGE_DIR="${DATA_DIR}/data_factory/staging/dann/truseq"
FINAL_DIR="${DATA_DIR}/data_factory/final/dann/truseq"
TEMP_DIR="${DATA_DIR}/temp_processing/dann/truseq"
META_DIR="${DATA_DIR}/metadata"

# 12D Output Manifests
DNA_STAGE_CSV="${STAGE_DIR}/deprobe_truseq_master.csv"
DNA_FINAL_H5="${FINAL_DIR}/deprobe_truseq_master.h5"

# ------------------------------------------------------------------------------
# Biological data assets
# ------------------------------------------------------------------------------
# 1. Reference Genome (hg19 / GRCh37) -> Stored in data/hg19/
REF_FASTA="${REF_DIR}/human_g1k_v37.fasta"
# Upgraded to HTTPS to prevent FTP timeout drops
REF_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/human_g1k_v37.fasta.gz"

# 2. Experimental Data (GIAB NA12878 TruSeq Exome) -> Stored in data/bams/
BAM_NA12878="${BAM_DIR}/NIST-hg001-7001-ready.bam"
# AWS S3 download route
BAM_S3_URI="s3://giab/data/NA12878/Nebraska_NA12878_HG001_TruSeq_Exome/NIST-hg001-7001-ready.bam"
BAI_S3_URI="s3://giab/data/NA12878/Nebraska_NA12878_HG001_TruSeq_Exome/NIST-hg001-7001-ready.bam.bai"
# Fallback Route: HTTPS (NCBI Mirror)
BAM_HTTPS_URL="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/Nebraska_NA12878_HG001_TruSeq_Exome/NIST-hg001-7001-ready.bam"
BAI_HTTPS_URL="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/Nebraska_NA12878_HG001_TruSeq_Exome/NIST-hg001-7001-ready.bam.bai"

# 3. Target Regions (Illumina TruSeq Exome v1.2) -> Stored in data/beds/
BED_TRUSEQ="${BED_DIR}/TruSeq_DNA_Exome_TargetedRegions_v1.2.bed"
BED_URL="https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/truseq/truseq-dna-exome/truseq-dna-exome-targeted-regions-manifest-v1-2-bed.zip"

# ------------------------------------------------------------------------------
# [CONFIG] HYPERPARAMETERS & HARDWARE ALLOCATION
# ------------------------------------------------------------------------------
PROBE_MIN_LEN=120
PROBE_MAX_LEN=120
PROBE_STEP=10
NEG_POOL_SIZE=5
WORKERS=24  # Maximize Ultra 9 multi-core performance

# ==============================================================================
# HELPER FUNCTIONS & ISOLATED LOGGING SETUP
# ==============================================================================
# Initialize Log Directory and File securely
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/truseq_factory_$(date +'%Y%m%d_%H%M%S').log"

echo "===================================================" > "$LOG_FILE"
echo "DEPROBE-DNA TRUSEQ PIPELINE LOG INITIALIZED AT $(date)" >> "$LOG_FILE"
echo "Working Directory Locked to: $SCRIPT_DIR" >> "$LOG_FILE"
echo "===================================================" >> "$LOG_FILE"

# Logging functions: Output color to terminal, append plain text to log file
log_info() {
    local ts=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[${ts}]${NC} ${CYAN}${BOLD}[INFO]${NC} $1"
    echo "[${ts}] [INFO] $1" >> "$LOG_FILE"
}

log_success() {
    local ts=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[${ts}]${NC} ${GREEN}${BOLD}[SUCCESS]${NC} $1"
    echo "[${ts}] [SUCCESS] $1" >> "$LOG_FILE"
}

log_warning() {
    local ts=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[${ts}]${NC} ${YELLOW}${BOLD}[WARNING]${NC} $1"
    echo "[${ts}] [WARNING] $1" >> "$LOG_FILE"
}

log_fatal() {
    local ts=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${RED}${BOLD}[FATAL ERROR]${NC} $1"
    echo "[${ts}] [FATAL ERROR] $1" >> "$LOG_FILE"
    exit 1
}

# --- Download wrappers ---
download_aria2() {
    local out_path=$1
    local url=$2
    log_info "Fetching via Aria2c (Parallel HTTPS): $(basename "$out_path")"
    # -x 8: 8 connections per server | -s 8: Split into 8 parts | --continue: True resume
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
    command -v aria2c >/dev/null 2>&1 || log_fatal "aria2 is required for multi-part downloading. (Install via: sudo apt install aria2)"
    command -v samtools >/dev/null 2>&1 || log_fatal "samtools is required for BAM/FASTA indexing."
    command -v gunzip >/dev/null 2>&1 || log_fatal "gunzip is required for FASTA decompression."
    command -v unzip >/dev/null 2>&1 || log_fatal "unzip is required to extract BED files."
    if ! command -v aws >/dev/null 2>&1; then
        log_warning "AWS CLI not found. Installing it globally via pixi is recommended for maximum GIAB BAM speed."
    fi
    log_success "Core system dependencies verified."
}

# ==============================================================================
# EXECUTION PIPELINE
# ==============================================================================

clear
echo -e "${CYAN}${BOLD}"
echo "============================================================"
echo "      DEPROBE-DNA: TRUSEQ FACTORY (NIST-hg001-7001)"
echo "============================================================"
echo -e "${NC}"
log_info "Pipeline diagnostics securely routed to: $LOG_FILE"
log_info "Tip: Run 'tail -f $LOG_FILE' in a split terminal to monitor real-time I/O."

# --- Phase 1: Infrastructure & Topology ---
log_info "Initializing workspace topologies at ${PROJECT_ROOT}..."
mkdir -p "$DATA_DIR" "$REF_DIR" "$BAM_DIR" "$BED_DIR" "$STAGE_DIR" "$FINAL_DIR" "$TEMP_DIR" "$META_DIR"

check_dependencies

# --- Phase 2: BAM and reference download ---
# --- Disk Space Pre-check ---
REQUIRED_GB=30
AVAIL_KB=$(df --output=avail "$DATA_DIR" | tail -1)
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
    log_fatal "Insufficient disk space: ${AVAIL_GB}GB available, ${REQUIRED_GB}GB required."
fi
log_info "Disk space check passed: ${AVAIL_GB}GB available."

log_info "Phase 2: BAM and reference download..."

# 2.1 Reference Genome
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


# 2.2 BAM & BAI (GIAB NA12878)
if [ ! -f "$BAM_NA12878" ]; then
    log_warning "BAM file not found. Downloading (10GB)..."
    download_giab "$BAM_NA12878" "$BAM_S3_URI" "$BAM_HTTPS_URL"
    
    # Structural sanity check to prevent "missing EOF block" crashes
    log_info "Running samtools quickcheck to verify BAM integrity..."
    if ! samtools quickcheck -v "$BAM_NA12878" >> "$LOG_FILE" 2>&1; then
        rm -f "$BAM_NA12878" "${BAM_NA12878}.bai"
        log_fatal "BAM corruption detected (Missing EOF)! File purged. Please re-run the pipeline."
    fi
    log_success "BAM structural integrity validated."

    log_info "Downloading corresponding BAI index..."
    download_giab "${BAM_NA12878}.bai" "$BAI_S3_URI" "$BAI_HTTPS_URL"
    log_success "NA12878 BAM and Index prepared in ${BAM_DIR}."
else
    log_info "NA12878 BAM already exists. Verifying integrity..."
    if ! samtools quickcheck -v "$BAM_NA12878" >> "$LOG_FILE" 2>&1; then
        log_fatal "Existing BAM is corrupted! Please manually delete ${BAM_NA12878} and re-run."
    fi
    if [ ! -f "${BAM_NA12878}.bai" ]; then
        log_warning "BAI index missing. Building index via samtools..."
        samtools index "$BAM_NA12878"
    fi
fi

# 2.3 BED File (Target Regions - Safe ZIP Extraction)
if [ ! -f "$BED_TRUSEQ" ]; then
    log_warning "TruSeq BED file not found at ${BED_TRUSEQ}."
    log_info "Downloading ZIP archive from Illumina servers..."
    download_aria2 "${BED_TRUSEQ}.zip" "$BED_URL"
    
    log_info "Extracting payload from ZIP archive..."
    # 'unzip -p' streams the file contents safely to standard output, redirecting to .bed
    TEMP_UNZIP="${BED_DIR}/_unzip_tmp"
    mkdir -p "$TEMP_UNZIP"
    unzip -o "${BED_TRUSEQ}.zip" -d "$TEMP_UNZIP" >> "$LOG_FILE" 2>&1
    BED_EXTRACTED=$(find "$TEMP_UNZIP" -name "*.bed" -type f | head -1)
    if [ -z "$BED_EXTRACTED" ]; then
        rm -rf "$TEMP_UNZIP"
        log_fatal "No .bed file found inside ZIP archive!"
    fi
    mv "$BED_EXTRACTED" "$BED_TRUSEQ"
    rm -rf "$TEMP_UNZIP" "${BED_TRUSEQ}.zip"

    log_success "TruSeq BED file safely extracted into ${BED_DIR}."
else
    log_info "TruSeq BED file already exists."
fi

# --- Phase 3: Metadata Synchronization ---
log_info "Phase 3: metadata generation..."
cat <<EOF > "${META_DIR}/metadata_truseq.csv"
Sample_Name,BAM_Path,BED_Path,Platform_ID
NA12878_TruSeq,${BAM_NA12878},${BED_TRUSEQ},0
EOF

# --- Phase 4: Feature Extraction (Staging) ---
log_info "Phase 4: feature extraction (12D priors)..."
log_info "Workers allocated: ${WORKERS}"

# Clean temporary storage and enforce I/O barrier for NTFS/WSL2 stability
rm -rf "${TEMP_DIR:?}"/*
sync && sleep 2

start_staging=$(date +%s)
# Python outputs are piped entirely to the log file.
python3 build_pretrain_dataset.py \
    --meta "${META_DIR}/metadata_truseq.csv" \
    --ref "$REF_FASTA" \
    --mode dna \
    --out "$DNA_STAGE_CSV" \
    --temp "$TEMP_DIR" \
    --min_len $PROBE_MIN_LEN --max_len $PROBE_MAX_LEN --step $PROBE_STEP \
    --workers $WORKERS >> "$LOG_FILE" 2>&1

end_staging=$(date +%s)
log_success "12D Staging completed in $(( (end_staging - start_staging) / 60 )) minutes."

# --- Phase 5: HDF5 Digitization & Hard Negative Mining ---
log_info "Phase 5: HDF5 build..."

# Ensure the OS flushes the large intermediate CSV to the NVMe disk safely
sync && sleep 5

start_final=$(date +%s)
python3 build_h5.py \
    --input "$DNA_STAGE_CSV" \
    --output "$DNA_FINAL_H5" \
    --ref "$REF_FASTA" \
    --neg_pool $NEG_POOL_SIZE >> "$LOG_FILE" 2>&1

end_final=$(date +%s)
log_success "12D HDF5 generation completed in $(( (end_final - start_final) / 60 )) minutes."

if [ -f "$DNA_FINAL_H5" ]; then
    log_success "TruSeq HDF5 location: ${DNA_FINAL_H5}"
else
    log_fatal "HDF5 output missing after pipeline completion."
fi