#!/usr/bin/env python3
"""
Create proper velocity overlay images for thesis
"""

import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from glacier_mapping.utils.visualize import make_rgb_preview
from glacier_mapping.utils.config import load_server_config

# Configuration
OUTPUT_DIR = Path(__file__).parent / "velocity_mosaic"
VECTOR_DPI = 1200
V_MAG_RANGE = (0, 500)  # m/yr
V_COMPONENT_RANGE = (-250, 250)  # m/yr


def create_velocity_overlay(rgb_base, velocity_data, value_range, cmap_name, mask):
    """Create RGB + velocity overlay with 50% transparency."""
    # Create velocity visualization (no colorbar)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=VECTOR_DPI)

    # Apply mask
    data_masked = np.ma.masked_where(~mask, velocity_data)

    # Plot with colormap (no colorbar)
    im = ax.imshow(
        data_masked, cmap=cmap_name, vmin=value_range[0], vmax=value_range[1]
    )
    ax.axis("off")

    # Save and reload as array
    temp_path = OUTPUT_DIR / "temp_velocity_overlay.png"
    plt.savefig(temp_path, dpi=VECTOR_DPI, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Load as array
    velocity_viz = cv2.imread(str(temp_path))
    velocity_viz = cv2.cvtColor(velocity_viz, cv2.COLOR_BGR2RGB)
    temp_path.unlink()  # Clean up

    # Ensure same size
    if rgb_base.shape[:2] != velocity_viz.shape[:2]:
        velocity_viz = cv2.resize(velocity_viz, (rgb_base.shape[1], rgb_base.shape[0]))

    # Create alpha mask for velocity data (50% transparency)
    alpha_mask = np.zeros((*mask.shape, 1), dtype=np.float32)
    alpha_mask[mask] = 0.5

    # Blend RGB with velocity overlay
    result = (
        rgb_base.astype(np.float32) * (1 - alpha_mask)
        + velocity_viz.astype(np.float32) * alpha_mask
    ).astype(np.uint8)

    return result


def main():
    """Generate proper overlay images."""
    # Load data
    server_config = load_server_config("desktop")
    landsat_path = Path(server_config["image_dir"]) / "image1.tif"
    velocity_path = Path(server_config["velocity_dir"]) / "image1.tif"

    with rasterio.open(landsat_path) as src:
        landsat_data = src.read()

    with rasterio.open(velocity_path) as src:
        velocity_data = src.read()

    # Create RGB base
    rgb_standard = make_rgb_preview(landsat_data[[2, 1, 0], :, :].transpose(1, 2, 0))

    # Extract velocity components and mask
    v_magnitude = velocity_data[0]
    v_x = velocity_data[1]
    v_y = velocity_data[2]
    v_mask = velocity_data[3]
    mask = v_mask > 0

    # Create overlays
    mag_overlay = create_velocity_overlay(
        rgb_standard, v_magnitude, V_MAG_RANGE, "viridis", mask
    )
    x_overlay = create_velocity_overlay(
        rgb_standard, v_x, V_COMPONENT_RANGE, "RdBu_r", mask
    )
    y_overlay = create_velocity_overlay(
        rgb_standard, v_y, V_COMPONENT_RANGE, "RdBu_r", mask
    )

    # Save overlays
    cv2.imwrite(
        str(OUTPUT_DIR / "rgb_velocity_magnitude_overlay.png"),
        cv2.cvtColor(mag_overlay, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "rgb_velocity_x_overlay.png"),
        cv2.cvtColor(x_overlay, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "rgb_velocity_y_overlay.png"),
        cv2.cvtColor(y_overlay, cv2.COLOR_RGB2BGR),
    )

    print("Generated proper overlay images!")


if __name__ == "__main__":
    main()
