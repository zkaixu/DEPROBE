import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse

# Resolve project root and canonical figure home (results/plots).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, 'results', 'plots', 'priors_12d_distribution.png')

parser = argparse.ArgumentParser(description="Visualize 12D prior distributions from H5 dataset.")
parser.add_argument("--h5", required=True, help="Path to H5 dataset")
parser.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output image path (default: {DEFAULT_OUT})")
args = parser.parse_args()

H5_PATH = args.h5
SAVE_PATH = args.out
# Defensive: auto-create the parent directory if missing.
os.makedirs(os.path.dirname(os.path.abspath(SAVE_PATH)) or '.', exist_ok=True)

FEATURE_NAMES = [
    "Tm", "GC%", "dG (kcal/mol)", "Hairpin Tm", "Dimer Tm",
    "Yield", "Norm Len", "GC Skew", "Entropy (2-mer)",
    "dG 5'", "dG 3'", "Collision Penalty"
]

print(f"Loading HDF5 data from: {H5_PATH}")
start_time = time.time()

with h5py.File(H5_PATH, 'r') as f:
    total_samples = f['priors'].shape[0]
    step = 100 
    print(f"Total samples: {total_samples:,}. Sampling every {step} rows...")
    sampled_priors = f['priors'][::step, :]

sampled_count = sampled_priors.shape[0]
print(f"Loaded {sampled_count:,} samples in {time.time() - start_time:.2f} seconds.")

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 10})

fig, axes = plt.subplots(4, 3, figsize=(16, 16))
axes = axes.flatten()

print("Generating distribution plots...")

for i in range(12):
    ax = axes[i]
    feature_data = sampled_priors[:, i]
    
    q_low = np.percentile(feature_data, 0.1)
    q_high = np.percentile(feature_data, 99.9)
    filtered_data = feature_data[(feature_data >= q_low) & (feature_data <= q_high)]
    
    sns.histplot(filtered_data, bins=50, kde=True, color="steelblue", ax=ax, 
                 stat="density", line_kws={"linewidth": 2})
    
    mean_val = np.mean(filtered_data)
    std_val = np.std(filtered_data)
    ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_title(f"Dim {i+1}: {FEATURE_NAMES[i]}\n(μ={mean_val:.2f}, σ={std_val:.2f})", fontweight='bold')
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")

plt.tight_layout()
# Defensive: re-ensure parent directory exists at write time.
os.makedirs(os.path.dirname(os.path.abspath(SAVE_PATH)) or '.', exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"\n[SUCCESS] Distribution plot saved securely to: {SAVE_PATH}")
