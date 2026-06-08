#!/bin/bash
# ==============================================================================
# DEPROBE-DNA: Nextera Expanded Exome PROBE MANIFEST Downloader & Validator
# ==============================================================================
# Downloads the probe-level coordinate file (347K probes, 94bp each) from
# Illumina's support page and validates correspondence with existing
# NIST7035/7086 BAM and target region BED files.
#
# Probe manifest for downstream validation pipeline:
#   real probe coordinates → model prediction → compare with BAM depth
# ==============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BED_DIR="${PROJECT_ROOT}/data/beds"
BAM_DIR="${PROJECT_ROOT}/data/bams"

PROBE_FILE="${BED_DIR}/nexterarapidcapture_expandedexome_probes.txt"
TARGET_BED="${BED_DIR}/nexterarapidcapture_expandedexome_targetedregions.bed"
BAM_7086="${BAM_DIR}/project.NIST_NIST7086_H7AP8ADXX_CGTACTAG_2_NA12878.bwa.markDuplicates.bam"
BAM_7035="${BAM_DIR}/project.NIST_NIST7035_H7AP8ADXX_TAAGGCGA_2_NA12878.bwa.markDuplicates.bam"
PROBE_URL="https://support.illumina.com/content/dam/illumina-support/documents/documentation/chemistry_documentation/samplepreps_nextera/nexterarapidcapture/nexterarapidcapture_expandedexome_probes.txt"

# Temp file: parse 20MB manifest once, reuse everywhere (avoids SIGPIPE with head)
TEMP_PROBES=$(mktemp /tmp/deprobe_probes_XXXXXX.tsv)
trap 'rm -f "$TEMP_PROBES"' EXIT

log_info()    { echo -e "${BLUE}[$(date +'%T')]${NC} ${CYAN}${BOLD}[INFO]${NC} $1"; }
log_success() { echo -e "${BLUE}[$(date +'%T')]${NC} ${GREEN}${BOLD}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${BLUE}[$(date +'%T')]${NC} ${YELLOW}${BOLD}[WARNING]${NC} $1"; }
log_fatal()   { echo -e "${RED}${BOLD}[FATAL]${NC} $1"; exit 1; }

echo -e "${CYAN}${BOLD}"
echo "============================================================"
echo "  DEPROBE-DNA: Nextera Expanded Exome Probe Manifest Tool"
echo "============================================================${NC}"

# ==== Phase 1: Download ====
mkdir -p "$BED_DIR"
if [ -f "$PROBE_FILE" ]; then
    log_info "Probe manifest exists: $(du -h "$PROBE_FILE" | cut -f1)"
else
    log_info "Downloading probe manifest (~20 MB)..."
    if command -v aria2c &>/dev/null; then
        aria2c -x 8 -s 8 --retry-wait=2 --max-tries=5 --continue=true \
            -d "$BED_DIR" -o "$(basename "$PROBE_FILE")" "$PROBE_URL"
    else
        curl -L -o "$PROBE_FILE" "$PROBE_URL"
    fi
    [ -s "$PROBE_FILE" ] || log_fatal "Download failed."
    log_success "Downloaded: $(du -h "$PROBE_FILE" | cut -f1)"
fi

# ==== Phase 2: Parse once → temp file ====
log_info "Parsing probe manifest..."
awk '/^\[Regions\]/{found=1; next} found && /^Name\t/{next} found && NF>=4 && !/^\[/ && !/^#/' "$PROBE_FILE" > "$TEMP_PROBES"
PROBE_COUNT=$(wc -l < "$TEMP_PROBES")
log_info "Extracted ${PROBE_COUNT} probe records"

# ==== Phase 3: Inspection ====
echo ""
log_info "========== INSPECTION =========="

log_info "Header:"
head -6 "$PROBE_FILE" | sed 's/^/  /'

echo ""
log_info "Columns:"
awk '/^\[Regions\]/{found=1; next} found && /^Name\t/{print "  "$0; exit}' "$PROBE_FILE"

echo ""
log_info "First 10 probes:"
head -10 "$TEMP_PROBES" | awk -F'\t' '{printf "  %-40s %s:%s-%s (len=%d)\n", $1, $2, $3, $4, $4-$3}'

echo ""
log_info "Probe length distribution:"
awk -F'\t' '{len=$4-$3; lengths[len]++} END {
    min=999999; max=0; total=0; n=0
    for (l in lengths) {
        n += lengths[l]; total += l * lengths[l]
        if (l+0 < min+0) min = l
        if (l+0 > max+0) max = l
    }
    printf "  Min: %d bp | Max: %d bp | Mean: %.1f bp\n", min, max, total/n
    printf "  Unique lengths: "
    first=1
    for (l in lengths) {
        if (!first) printf ", "
        printf "%dbp(%d)", l, lengths[l]
        first=0
    }
    printf "\n"
}' "$TEMP_PROBES"

echo ""
log_info "Chromosome distribution:"
awk -F'\t' '{chr[$2]++} END {
    n = asorti(chr, sorted); total = 0
    for (i=1; i<=n; i++) { printf "  %-8s %6d\n", sorted[i], chr[sorted[i]]; total += chr[sorted[i]] }
    printf "  %-8s %6d\n", "TOTAL", total
}' "$TEMP_PROBES"

# ==== Phase 4: Cross-Validation ====
echo ""
log_info "========== CROSS-VALIDATION =========="
PASS=0; FAIL=0

[ -f "$TARGET_BED" ] || log_fatal "Target BED not found: $TARGET_BED"
log_info "Target regions: $(wc -l < "$TARGET_BED")"

# CHECK 1: Chromosome naming
echo ""
log_info "[CHECK 1] Chromosome naming..."
PROBE_CHR=$(awk -F'\t' '{print $2}' "$TEMP_PROBES" | sort -u | grep -E '^chr([0-9]+|[XY])$' | sort -V)
BED_CHR=$(awk -F'\t' '{print $1}' "$TARGET_BED" | sort -u | grep -E '^chr([0-9]+|[XY])$' | sort -V)

if [ "$PROBE_CHR" = "$BED_CHR" ]; then
    log_success "[PASS] Probe manifest ↔ target BED chromosomes match"
    PASS=$((PASS+1))
else
    log_warning "[FAIL] Chromosome mismatch"; FAIL=$((FAIL+1))
fi

if [ -f "$BAM_7086" ]; then
    BAM_CHR=$(samtools view -H "$BAM_7086" 2>/dev/null | awk '/^@SQ/{for(i=1;i<=NF;i++) if($i~/^SN:/) print substr($i,4)}' | grep -E '^chr([0-9]+|[XY])$' | sort -V)
    if [ "$PROBE_CHR" = "$BAM_CHR" ]; then
        log_success "[PASS] Probe manifest ↔ NIST7086 BAM chromosomes match"
        PASS=$((PASS+1))
    else
        log_warning "[FAIL] BAM chromosome mismatch"; FAIL=$((FAIL+1))
    fi
fi

# CHECK 2: Probe ⊂ target regions (±100bp)
echo ""
log_info "[CHECK 2] Probes within target regions (±100bp slop)..."

TEMP_PROBE_BED=$(mktemp /tmp/probes_bed_XXXXXX.tsv)
awk -F'\t' '{OFS="\t"; print $2, $3, $4}' "$TEMP_PROBES" > "$TEMP_PROBE_BED"

OVERLAP_RESULT=$(python3 -c "
import bisect
targets = {}
with open('${TARGET_BED}') as f:
    for line in f:
        c = line.strip().split('\t')
        targets.setdefault(c[0], []).append((max(0,int(c[1])-100), int(c[2])+100))
for ch in targets:
    targets[ch].sort()
    m = [targets[ch][0]]
    for s,e in targets[ch][1:]:
        if s <= m[-1][1]: m[-1] = (m[-1][0], max(m[-1][1], e))
        else: m.append((s,e))
    targets[ch] = m
starts = {c:[iv[0] for iv in ivs] for c,ivs in targets.items()}
total=hit=0
with open('${TEMP_PROBE_BED}') as f:
    for line in f:
        c = line.strip().split('\t')
        ch,mid = c[0], (int(c[1])+int(c[2]))//2
        total += 1
        if ch in starts:
            i = bisect.bisect_right(starts[ch], mid)-1
            if i>=0 and targets[ch][i][0]<=mid<=targets[ch][i][1]: hit+=1
print(hit, total)
")
rm -f "$TEMP_PROBE_BED"

OH=$(echo "$OVERLAP_RESULT" | awk '{print $1}')
OT=$(echo "$OVERLAP_RESULT" | awk '{print $2}')
if [ -n "$OH" ] && [ "$OT" -gt 0 ] 2>/dev/null; then
    OP=$(awk "BEGIN{printf \"%.1f\", 100.0*$OH/$OT}")
    log_info "Overlap: ${OH} / ${OT} (${OP}%)"
    if (( $(echo "$OP > 95" | bc -l) )); then
        log_success "[PASS] ${OP}% probes within target regions"; PASS=$((PASS+1))
    else
        log_warning "[WARN] Only ${OP}% overlap"; FAIL=$((FAIL+1))
    fi
else
    log_warning "[WARN] Overlap check error: $OVERLAP_RESULT"; FAIL=$((FAIL+1))
fi

# CHECK 3: BAM depth spot-check
echo ""
log_info "[CHECK 3] BAM depth spot-check..."

for BAM_LABEL in "NIST7086:${BAM_7086}" "NIST7035:${BAM_7035}"; do
    LABEL="${BAM_LABEL%%:*}"
    BAM="${BAM_LABEL#*:}"
    if [ ! -f "$BAM" ]; then log_warning "[SKIP] ${LABEL} not found"; continue; fi
    log_info "Sampling 5 probes from ${LABEL}..."
    shuf -n 5 "$TEMP_PROBES" | while IFS=$'\t' read -r name chr start end rest; do
        mid=$(( (start + end) / 2 ))
        depth=$(samtools depth -r "${chr}:${mid}-${mid}" "$BAM" 2>/dev/null | awk '{print $3}')
        depth=${depth:-0}
        if [ "$depth" -gt 0 ]; then
            log_info "  ${chr}:${start}-${end} → depth=${depth}"
        else
            log_warning "  ${chr}:${start}-${end} → depth=0"
        fi
    done
done

if [ -f "$BAM_7086" ] && [ -f "$BAM_7035" ]; then
    log_success "[PASS] Both BAMs accessible"; PASS=$((PASS+1))
fi

# ==== Summary ====
echo ""
echo -e "${CYAN}${BOLD}============================================================"
echo "  SUMMARY"
echo "============================================================${NC}"
log_info "Passed: ${PASS} | Failed: ${FAIL}"
if [ "$FAIL" -eq 0 ]; then
    log_success "Probe manifest ready."
fi
log_info "File: ${PROBE_FILE}"
log_info "Probes: ${PROBE_COUNT}"
