import numpy as np
from pathlib import Path
import json

# Path to the processed dataset
dataset_path = Path("/home/devj/local-debian/datasets/HKH/bibek_w512_o64_f1_velocity")
normalize_file = dataset_path / "normalize_train.npy"
band_meta_file = dataset_path / "band_metadata.json"

# Check if files exist
if not normalize_file.exists():
    print(f"Error: Normalization file not found at {normalize_file}")
    exit()
if not band_meta_file.exists():
    print(f"Error: Band metadata file not found at {band_meta_file}")
    exit()

# Load the normalization array (means, stds, mins, maxs)
stats = np.load(normalize_file)
means = stats[0]
stds = stats[1]

# Load band metadata to identify velocity channels
with open(band_meta_file, "r") as f:
    band_meta = json.load(f)
band_names = band_meta["band_names"]

# Correctly identify velocity channel names
velocity_channel_names = ["velocity", "velocity_x", "velocity_y", "velocity_mask"]
velocity_indices = [
    i for i, name in enumerate(band_names) if name in velocity_channel_names
]

print(f"Dataset Path: {dataset_path}")
print(f"Found {len(band_names)} total channels.")
print("-" * 30)

if not velocity_indices:
    print("Error: No velocity channels found in band_metadata.json")
else:
    print("Velocity Channel Statistics:")
    for i in velocity_indices:
        channel_name = band_names[i]
        mean_val = means[i]
        std_val = stds[i]

        print(f"  - Channel '{channel_name}' (index {i}):")
        print(f"    - Mean: {mean_val:.4f}")
        print(f"    - Std Dev: {std_val:.4f}")

        # Sanity checks
        if np.isnan(mean_val) or np.isinf(mean_val):
            print("    - WARNING: Mean is NaN or Inf. Data is likely corrupt.")
        if np.isnan(std_val) or np.isinf(std_val) or std_val <= 0:
            print(
                "    - WARNING: Std Dev is NaN, Inf, or zero/negative. Data may be uniform or corrupt."
            )
        print("-" * 30)
