#!/usr/bin/env python3
"""
DEPROBE-DNA: Illumina Probe Manifest → Centered BED Converter
=============================================================
Reads an Illumina Nextera Rapid Capture probe manifest (.txt) and outputs:
  1. A BED file with 120bp windows centered on each probe (for pipeline input)
  2. A CSV mapping from original probe names to centered coordinates

The 120bp centering allows direct reuse of the existing DEPROBE pipeline
(extract_sequences.py, calc_efficiency.py, calc_priors.py) with
--min_len 120 --max_len 120 to produce exactly one window per probe.

Usage:
    python parse_probe_manifest.py \
        --manifest /path/to/nexterarapidcapture_expandedexome_probes.txt \
        --fai /path/to/human_g1k_v37.fasta.fai \
        --bed_out /path/to/probes_centered_120bp.bed \
        --map_out /path/to/probe_mapping.csv
"""

import argparse
import csv
import sys


def load_chrom_sizes(fai_path):
    """Load chromosome sizes from FASTA index."""
    sizes = {}
    with open(fai_path) as f:
        for line in f:
            cols = line.strip().split('\t')
            sizes[cols[0]] = int(cols[1])
    return sizes


def parse_manifest(manifest_path):
    """Yield (name, chrom, start, end) from Illumina probe manifest."""
    in_regions = False
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line == '[Regions]':
                in_regions = True
                continue
            if not in_regions:
                continue
            if line.startswith('Name\t'):
                continue  # column header
            if not line or line.startswith('[') or line.startswith('#'):
                continue
            cols = line.split('\t')
            if len(cols) < 4:
                continue
            yield cols[0], cols[1], int(cols[2]), int(cols[3])


def main():
    parser = argparse.ArgumentParser(description="Convert Illumina probe manifest to 120bp centered BED")
    parser.add_argument("--manifest", required=True, help="Illumina probe manifest TXT file")
    parser.add_argument("--fai", required=True, help="Reference FASTA index (.fai) for chromosome sizes")
    parser.add_argument("--bed_out", required=True, help="Output BED file (120bp centered windows)")
    parser.add_argument("--map_out", required=True, help="Output CSV mapping (probe name → coordinates)")
    parser.add_argument("--window", type=int, default=120, help="Target window size (default: 120)")
    args = parser.parse_args()

    chrom_sizes = load_chrom_sizes(args.fai)
    half_window = args.window // 2  # 60 for 120bp window

    total = 0
    skipped = 0
    written = 0

    with open(args.bed_out, 'w') as bed_f, open(args.map_out, 'w', newline='') as map_f:
        map_writer = csv.writer(map_f)
        map_writer.writerow([
            'Probe_Name', 'Chromosome', 'Probe_Start', 'Probe_End', 'Probe_Length',
            'Centered_Start', 'Centered_End', 'Centered_Length'
        ])

        for name, chrom, start, end in parse_manifest(args.manifest):
            total += 1
            probe_len = end - start
            center = (start + end) // 2
            c_start = center - half_window
            c_end = center + half_window

            # Boundary check (handle chr prefix mismatch: manifest=chr1, fai=1)
            chrom_len = chrom_sizes.get(chrom) or chrom_sizes.get(chrom.replace('chr', ''))
            if chrom_len is None:
                skipped += 1
                continue
            if c_start < 0:
                c_start = 0
                c_end = args.window
            if c_end > chrom_len:
                c_end = chrom_len
                c_start = chrom_len - args.window

            # Write BED (4-column: chr, start, end, name)
            bed_f.write(f"{chrom}\t{c_start}\t{c_end}\t{name}\n")

            # Write mapping
            map_writer.writerow([name, chrom, start, end, probe_len,
                                 c_start, c_end, c_end - c_start])
            written += 1

    print(f"[DONE] Processed {total} probes: {written} written, {skipped} skipped")
    print(f"[INFO] Centered BED: {args.bed_out}")
    print(f"[INFO] Mapping CSV:  {args.map_out}")


if __name__ == "__main__":
    main()
