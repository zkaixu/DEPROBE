#!/usr/bin/env bash
# ==============================================================================
# DEPROBE-DNA Environment Setup
# ==============================================================================
# Dual-layer architecture: global CLI tools + local Python/AI environment
# Tested on Ubuntu 22.04 (WSL2) with NVIDIA RTX 5090
# ==============================================================================

set -euo pipefail

PROJECT_NAME="DEPROBE"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO $(date +'%H:%M:%S')]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK   $(date +'%H:%M:%S')]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN $(date +'%H:%M:%S')]${NC} $*"; }

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  DEPROBE-DNA Environment Setup${NC}"
echo -e "${GREEN}============================================================${NC}"

# ------------------------------------------------------------------------------
# 1. Pixi package manager
# ------------------------------------------------------------------------------
if ! command -v pixi &> /dev/null; then
    if [ -x "$HOME/.pixi/bin/pixi" ]; then
        export PATH="$HOME/.pixi/bin:$PATH"
    else
        log_info "Installing Pixi..."
        curl -fsSL https://pixi.sh/install.sh | bash
        export PATH="$HOME/.pixi/bin:$PATH"
    fi
else
    log_info "Pixi already installed."
fi

# ------------------------------------------------------------------------------
# 2. Global CLI tools (isolated from Python env to avoid conflicts)
# ------------------------------------------------------------------------------
log_info "Provisioning global CLI tools..."

install_global() {
    local tool=$1
    if ! pixi global list 2>/dev/null | grep -q "$tool"; then
        log_info "Installing globally: $tool"
        pixi global install -c conda-forge -c bioconda "$tool"
    else
        log_success "Already installed: $tool"
    fi
}

install_global "aria2"       # Multi-threaded data download
install_global "samtools"    # BAM/FASTA indexing and manipulation

# ------------------------------------------------------------------------------
# 3. Project workspace
# ------------------------------------------------------------------------------
if [ ! -d "$PROJECT_NAME" ] || [ ! -f "$PROJECT_NAME/pixi.toml" ]; then
    log_info "Creating workspace: $PROJECT_NAME"
    pixi init "$PROJECT_NAME"
else
    log_info "Workspace '$PROJECT_NAME' exists."
fi

cd "$PROJECT_NAME"

if ! grep -q "bioconda" pixi.toml; then
    pixi project channel add bioconda conda-forge
fi

# ------------------------------------------------------------------------------
# 4. Python + AI + Bioinformatics dependencies
# ------------------------------------------------------------------------------
if grep -q "pytorch" pixi.toml && grep -q "primer3-py" pixi.toml; then
    log_info "Manifest found. Running fast sync..."
    pixi install
else
    log_warn "Building full dependency manifest..."

    # Core Python + data science
    log_info "[1/4] Python + data science stack..."
    pixi add "python=3.11" numpy pandas scipy scikit-learn

    # Bioinformatics (conda packages for C-binding stability)
    log_info "[2/4] Bioinformatics tools..."
    pixi add biopython primer3-py pysam

    # DEPROBE-specific libraries
    log_info "[3/4] DEPROBE-DNA dependencies..."
    pixi add datasketch h5py pyarrow matplotlib tqdm

    # PyTorch (PyPI for CUDA support)
    log_info "[4/4] PyTorch stack..."
    pixi add --pypi torch torchvision torchaudio
fi

log_success "============================================================"
log_success "DEPROBE-DNA environment ready."
log_success "Activate with: pixi shell"
log_success "============================================================"
