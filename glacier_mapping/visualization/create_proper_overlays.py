#!/usr/bin/env python3
"""
Create proper velocity overlay images with colorbars, titles, and units for thesis
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
OUTPUT_DIR = Path(__file__).parent / "output" / "velocity_mosaic"
VECTOR_DPI = 200
V_MAG_RANGE = (0, 500)  # m/yr
V_COMPONENT_RANGE = (-250, 250)  # m/yr


def create_velocity_triptych(
    rgb_base, velocity_data, value_range, cmap_name, mask, title, units
):
    """Create triptych: RGB | Overlay | Channel with colorbar."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=VECTOR_DPI)

    # Original RGB
    axes[0].imshow(rgb_base)
    axes[0].axis("off")
    axes[0].set_title("RGB", fontsize=12)

    # Apply mask to velocity data
    data_masked = np.ma.masked_where(~mask, velocity_data)

    # Overlay
    axes[1].imshow(rgb_base)
    im_overlay = axes[1].imshow(
        data_masked, cmap=cmap_name, vmin=value_range[0], vmax=value_range[1], alpha=0.5
    )
    axes[1].axis("off")
    axes[1].set_title("RGB + Velocity Overlay", fontsize=12)
    cbar_overlay = fig.colorbar(im_overlay, ax=axes[1], fraction=0.046, pad=0.02)
    cbar_overlay.set_label(f"Velocity ({units})", fontsize=10)

    # Channel only
    im = axes[2].imshow(
        data_masked, cmap=cmap_name, vmin=value_range[0], vmax=value_range[1]
    )
    axes[2].axis("off")
    axes[2].set_title("Velocity Channel", fontsize=12)
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
    cbar.set_label(f"Velocity ({units})", fontsize=10)

    # Main title
    fig.suptitle(title, fontsize=14)

    # Save and reload as array
    temp_path = (
        OUTPUT_DIR
        / f"temp_triptych_{title.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('%', 'percent')}.png"
    )
    plt.savefig(temp_path, dpi=VECTOR_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    # Load as array
    triptych_with_cbar = cv2.imread(str(temp_path))
    triptych_with_cbar = cv2.cvtColor(triptych_with_cbar, cv2.COLOR_BGR2RGB)
    temp_path.unlink()  # Clean up

    return triptych_with_cbar


def main():
    """Generate proper overlay images with colorbars and titles."""
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

    # Create triptychs with proper colorbars and titles
    # RGB + Velocity Magnitude triptych
    mag_triptych = create_velocity_triptych(
        rgb_standard,
        v_magnitude,
        V_MAG_RANGE,
        "viridis",
        mask,
        "Velocity Magnitude Triptych - image1",
        "m/yr",
    )

    # RGB + X component triptych
    x_triptych = create_velocity_triptych(
        rgb_standard,
        v_x,
        V_COMPONENT_RANGE,
        "RdBu_r",
        mask,
        "Velocity X-Component Triptych - image1",
        "m/yr",
    )

    # RGB + Y component triptych
    y_triptych = create_velocity_triptych(
        rgb_standard,
        v_y,
        V_COMPONENT_RANGE,
        "RdBu_r",
        mask,
        "Velocity Y-Component Triptych - image1",
        "m/yr",
    )

    # Save triptychs
    cv2.imwrite(
        str(OUTPUT_DIR / "rgb_velocity_magnitude_triptych.png"),
        cv2.cvtColor(mag_triptych, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "rgb_velocity_x_triptych.png"),
        cv2.cvtColor(x_triptych, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "rgb_velocity_y_triptych.png"),
        cv2.cvtColor(y_triptych, cv2.COLOR_RGB2BGR),
    )

    print("Generated proper overlay images with colorbars, titles, and units!")


if __name__ == "__main__":
    main()
