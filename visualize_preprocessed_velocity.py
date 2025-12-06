import numpy as np
from pathlib import Path
import json
import random
import matplotlib.pyplot as plt


def visualize_velocity_slice(
    dataset_path_str: str,
    num_samples: int = 3,
    output_filename: str = "velocity_slice_check.png",
):
    """
    Loads random slices from a preprocessed dataset (with separate tiff/mask files),
    reconstructs the full data array, and saves a visual comparison of a visible
    band and the velocity magnitude channel.
    """
    dataset_path = Path(dataset_path_str)
    # Corrected: Look in the 'test' directory as 'train' is missing
    data_path = dataset_path / "test"
    band_meta_file = dataset_path / "band_metadata.json"

    # --- Validation ---
    if not data_path.exists():
        print(f"Error: Data directory not found at {data_path}")
        return
    # band_meta_file is not strictly needed for this visualization script,
    # as we are hardcoding the indices. We can proceed without it.
    # if not band_meta_file.exists():
    #     print(f"Error: Band metadata file not found at {band_meta_file}")
    #     return

    # --- Load Metadata ---
    # The band_metadata.json might be missing, so we hardcode the indices
    # Based on configs/preprocess.yaml and the order in slice.py:
    # B1, B2, B3, B4, B5, B6_1, B6_2, B7, elev, slope, vel, vel_x, vel_y, vel_mask ...
    try:
        vis_band_index = 3  # B4 is the 4th band, so index 3
        velocity_mag_index = 10  # velocity is the 11th band, so index 10
    except IndexError:
        print(f"Error: Could not find hardcoded band index.")
        return

    # --- Find corresponding tiff and mask files ---
    # Corrected: Glob for .npy files
    tiff_files = sorted(list(data_path.glob("tiff_*.npy")))
    if not tiff_files:
        print(f"Error: No tiff files (.npy) found in {data_path}")
        return

    random_tiff_files = random.sample(tiff_files, min(num_samples, len(tiff_files)))

    # --- Plotting ---
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])  # Ensure axes is always 2D array
    fig.suptitle(f"Visual Velocity Check (Corrected): {dataset_path.name}", fontsize=16)

    for i, tiff_file in enumerate(random_tiff_files):
        mask_file = data_path / tiff_file.name.replace("tiff_", "mask_")

        if not mask_file.exists():
            print(
                f"Warning: Corresponding mask file not found for {tiff_file.name}. Skipping."
            )
            continue

        # Load and stack the data to reconstruct the full array
        # The tiff data is (H, W, C), so we need to transpose it to (C, H, W)
        tiff_data = np.load(tiff_file).transpose(2, 0, 1)
        mask_data = np.load(mask_file)

        # Ensure mask is 3D for stacking: (1, H, W)
        if mask_data.ndim == 2:
            mask_data = mask_data[np.newaxis, :, :]

        full_data = np.vstack((tiff_data, mask_data))

        # Extract the specific channels for visualization
        vis_band_img = full_data[vis_band_index, :, :]
        velocity_img = full_data[velocity_mag_index, :, :]

        ax_vis = axes[i, 0]
        ax_vel = axes[i, 1]

        ax_vis.imshow(vis_band_img, cmap="gray")
        ax_vis.set_title(
            f"Slice: {tiff_file.name.replace('tiff_', '')}\\nBand 'B4' (Visible)"
        )
        ax_vis.set_xticks([])
        ax_vis.set_yticks([])

        im = ax_vel.imshow(velocity_img, cmap="viridis")
        ax_vel.set_title("Velocity Magnitude")
        ax_vel.set_xticks([])
        ax_vel.set_yticks([])
        fig.colorbar(im, ax=ax_vel)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_filename)
    print(f"\\nVerification image saved to: {output_filename}")


if __name__ == "__main__":
    dataset_path = "/home/devj/local-debian/datasets/HKH/bibek_w512_o64_f1_velocity"
    visualize_velocity_slice(dataset_path)
