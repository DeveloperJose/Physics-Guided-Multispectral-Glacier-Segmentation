#!/usr/bin/env python3
"""
Updated Velocity Mosaic Pipeline Visualization for Thesis

This script creates individual high-resolution visualizations for ITS_LIVE velocity
mosaic preprocessing pipeline with overlays and proper titles.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from glacier_mapping.utils.visualize import (
    make_rgb_preview,
    COLOR_BG,
    COLOR_CI,
    COLOR_DEB,
    COLOR_IGNORE,
    label_to_color,
    make_overlay,
    add_title,
)
from glacier_mapping.utils.config import load_server_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "velocity_mosaic"
OUTPUT_DIR.mkdir(exist_ok=True)

# High resolution settings
THESIS_DPI = 600
VECTOR_DPI = 1200

# Color schemes
VELOCITY_MAGNITUDE_CMAP = "viridis"
VELOCITY_COMPONENT_CMAP = "RdBu_r"

# Velocity ranges for consistent scaling
V_MAG_RANGE = (0, 500)  # m/yr
V_COMPONENT_RANGE = (-250, 250)  # m/yr


class VelocityMosaicVisualizer:
    """Main class for generating velocity mosaic visualizations."""

    def __init__(self, server: str = "desktop", image_name: str = "image1"):
        """Initialize visualizer with server configuration and sample data."""
        self.server = server
        self.image_name = image_name

        # Load server configuration
        self.server_config = load_server_config(server)

        # Set up paths
        self.landsat_path = Path(self.server_config["image_dir"]) / f"{image_name}.tif"
        self.velocity_path = (
            Path(self.server_config["velocity_dir"]) / f"{image_name}.tif"
        )
        self.stats_path = (
            Path(self.server_config["velocity_dir"]) / f"{image_name}_stats.json"
        )
        self.labels_path = Path(self.server_config["labels_dir"]) / f"{image_name}.tif"

        # Data storage
        self.landsat_data = None
        self.velocity_data = None
        self.labels_data = None
        self.stats = None
        self.landsat_meta = None
        self.velocity_meta = None

        logger.info(f"Initialized visualizer for {image_name} on {server}")

    def load_data(self) -> None:
        """Load all required data files."""
        logger.info("Loading data files...")

        # Load Landsat data
        with rasterio.open(self.landsat_path) as src:
            self.landsat_data = src.read()
            self.landsat_meta = src.meta.copy()
            logger.info(f"Loaded Landsat: {self.landsat_data.shape}, CRS: {src.crs}")

        # Load velocity data
        with rasterio.open(self.velocity_path) as src:
            self.velocity_data = src.read()
            self.velocity_meta = src.meta.copy()
            logger.info(f"Loaded Velocity: {self.velocity_data.shape}, CRS: {src.crs}")

        # Load labels data
        if self.labels_path.exists():
            with rasterio.open(self.labels_path) as src:
                self.labels_data = src.read(1)  # Read first band
                logger.info(f"Loaded Labels: {self.labels_data.shape}")
        else:
            logger.warning(f"Labels file not found: {self.labels_path}")

        # Load statistics
        if self.stats_path.exists():
            with open(self.stats_path) as f:
                self.stats = json.load(f)
            logger.info(f"Loaded stats: {self.stats['coverage_percent']:.1f}% coverage")

    def create_rgb_previews(self) -> Dict[str, np.ndarray]:
        """Create RGB preview images from Landsat data."""
        logger.info("Creating RGB previews...")

        rgb_images = {}

        # Standard RGB (bands 3,2,1 = R,G,B)
        rgb_standard = make_rgb_preview(
            self.landsat_data[[2, 1, 0], :, :].transpose(1, 2, 0)
        )
        rgb_images["standard"] = rgb_standard

        # False color infrared (bands 4,3,2)
        rgb_infrared = make_rgb_preview(
            self.landsat_data[[3, 2, 1], :, :].transpose(1, 2, 0)
        )
        rgb_images["infrared"] = rgb_infrared

        return rgb_images

    def create_velocity_visualizations(self) -> Dict[str, np.ndarray]:
        """Create visualizations of velocity components."""
        logger.info("Creating velocity component visualizations...")

        v_images = {}

        # Extract components
        v_magnitude = self.velocity_data[0]  # Band 1
        v_x = self.velocity_data[1]  # Band 2
        v_y = self.velocity_data[2]  # Band 3
        v_mask = self.velocity_data[3]  # Band 4

        # Create masked versions for visualization
        mask = v_mask > 0

        # Velocity magnitude
        v_images["magnitude"] = self._create_colormap_image(
            v_magnitude, V_MAG_RANGE, VELOCITY_MAGNITUDE_CMAP, mask
        )

        # X component
        v_images["x_component"] = self._create_colormap_image(
            v_x, V_COMPONENT_RANGE, VELOCITY_COMPONENT_CMAP, mask
        )

        # Y component
        v_images["y_component"] = self._create_colormap_image(
            v_y, V_COMPONENT_RANGE, VELOCITY_COMPONENT_CMAP, mask
        )

        return v_images

    def _create_colormap_image(
        self,
        data: np.ndarray,
        value_range: Tuple[float, float],
        cmap_name: str,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Create RGB image from 2D data using specified colormap."""
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(10, 8), dpi=VECTOR_DPI)

        # Apply mask
        data_masked = np.ma.masked_where(~mask, data)

        # Plot with colormap
        im = ax.imshow(
            data_masked, cmap=cmap_name, vmin=value_range[0], vmax=value_range[1]
        )
        ax.axis("off")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Velocity (m/yr)")

        # Save and reload as array
        temp_path = OUTPUT_DIR / "temp_colormap.png"
        plt.savefig(temp_path, dpi=VECTOR_DPI, bbox_inches="tight", pad_inches=0)
        plt.close()

        # Load as array
        rgb_array = cv2.imread(str(temp_path))
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
        temp_path.unlink()  # Clean up

        return rgb_array

    def create_classification_overlay(self) -> Optional[np.ndarray]:
        """Create RGB + classification overlay."""
        logger.info("Creating classification overlay...")

        if self.labels_data is None:
            logger.warning("No labels data available for classification overlay")
            return None

        # Get RGB standard
        rgb_images = self.create_rgb_previews()
        rgb_standard = rgb_images["standard"]

        # Create colormap for classification
        cmap = {
            0: COLOR_BG,  # Background
            1: COLOR_CI,  # Clean Ice
            2: COLOR_DEB,  # Debris
            255: COLOR_IGNORE,  # Mask
        }

        # Create overlay with 0.5 transparency
        overlay = make_overlay(rgb_standard, self.labels_data, cmap, alpha=0.5)

        return overlay

    def create_velocity_overlays(self) -> Dict[str, np.ndarray]:
        """Create RGB + velocity component overlays."""
        logger.info("Creating velocity overlays...")

        # Get RGB standard
        rgb_images = self.create_rgb_previews()
        rgb_standard = rgb_images["standard"]

        # Extract velocity components
        v_magnitude = self.velocity_data[0]
        v_x = self.velocity_data[1]
        v_y = self.velocity_data[2]
        v_mask = self.velocity_data[3]

        # Create mask for valid data
        mask = v_mask > 0

        overlays = {}

        # RGB + Velocity Magnitude overlay
        overlays["magnitude"] = self._create_velocity_overlay(
            rgb_standard, v_magnitude, V_MAG_RANGE, VELOCITY_MAGNITUDE_CMAP, mask
        )

        # RGB + X component overlay
        overlays["x_component"] = self._create_velocity_overlay(
            rgb_standard, v_x, V_COMPONENT_RANGE, VELOCITY_COMPONENT_CMAP, mask
        )

        # RGB + Y component overlay
        overlays["y_component"] = self._create_velocity_overlay(
            rgb_standard, v_y, V_COMPONENT_RANGE, VELOCITY_COMPONENT_CMAP, mask
        )

        return overlays

    def _create_velocity_overlay(
        self,
        rgb_base: np.ndarray,
        velocity_data: np.ndarray,
        value_range: Tuple[float, float],
        cmap_name: str,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Create RGB + velocity overlay with transparency."""
        # Create velocity visualization (without colorbar for overlay)
        velocity_viz = self._create_colormap_image_overlay(
            velocity_data, value_range, cmap_name, mask
        )

        # Ensure same size
        if rgb_base.shape[:2] != velocity_viz.shape[:2]:
            velocity_viz = cv2.resize(
                velocity_viz, (rgb_base.shape[1], rgb_base.shape[0])
            )

        # Create alpha mask for velocity data (0.5 transparency)
        alpha_mask = np.zeros((*mask.shape, 1), dtype=np.float32)
        alpha_mask[mask] = 0.5

        # Blend RGB with velocity overlay
        result = (
            rgb_base.astype(np.float32) * (1 - alpha_mask)
            + velocity_viz.astype(np.float32) * alpha_mask
        ).astype(np.uint8)

        return result

    def _create_colormap_image_overlay(
        self,
        data: np.ndarray,
        value_range: Tuple[float, float],
        cmap_name: str,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Create RGB image from 2D data using specified colormap (no colorbar for overlay)."""
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(10, 8), dpi=VECTOR_DPI)

        # Apply mask
        data_masked = np.ma.masked_where(~mask, data)

        # Plot with colormap (no colorbar for overlay)
        im = ax.imshow(
            data_masked, cmap=cmap_name, vmin=value_range[0], vmax=value_range[1]
        )
        ax.axis("off")

        # Save and reload as array
        temp_path = OUTPUT_DIR / "temp_colormap_overlay.png"
        plt.savefig(temp_path, dpi=VECTOR_DPI, bbox_inches="tight", pad_inches=0)
        plt.close()

        # Load as array
        rgb_array = cv2.imread(str(temp_path))
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
        temp_path.unlink()  # Clean up

        return rgb_array

    def create_statistical_plots(self) -> Dict[str, str]:
        """Create statistical analysis plots."""
        logger.info("Creating statistical analysis plots...")

        plots = {}

        # Extract velocity data
        v_magnitude = self.velocity_data[0]
        v_x = self.velocity_data[1]
        v_y = self.velocity_data[2]
        v_mask = self.velocity_data[3]

        # Apply mask
        mask = v_mask > 0
        v_mag_valid = v_magnitude[mask]
        v_x_valid = v_x[mask]
        v_y_valid = v_y[mask]

        # 1. Velocity magnitude histogram
        plots["histogram"] = self._create_velocity_histogram(v_mag_valid)

        # 2. Scatter plot (vx vs vy)
        plots["scatter"] = self._create_velocity_scatter(v_x_valid, v_y_valid)

        # 3. Coverage map
        plots["coverage"] = self._create_coverage_map(mask)

        return plots

    def _create_velocity_histogram(self, v_mag_valid: np.ndarray) -> str:
        """Create velocity magnitude histogram."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=VECTOR_DPI)

        # Linear scale histogram
        ax1.hist(v_mag_valid, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax1.set_xlabel("Velocity Magnitude (m/yr)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Velocity Distribution (Linear Scale)")
        ax1.grid(True, alpha=0.3)

        # Log scale histogram
        ax2.hist(v_mag_valid, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax2.set_xlabel("Velocity Magnitude (m/yr)")
        ax2.set_ylabel("Frequency (log scale)")
        ax2.set_title("Velocity Distribution (Log Scale)")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(v_mag_valid)
        median_val = np.median(v_mag_valid)
        std_val = np.std(v_mag_valid)

        stats_text = f"Mean: {mean_val:.2f} m/yr\nMedian: {median_val:.2f} m/yr\nStd: {std_val:.2f} m/yr"
        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        output_path = OUTPUT_DIR / "velocity_histogram.svg"
        plt.savefig(output_path, format="svg", dpi=VECTOR_DPI, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def _create_velocity_scatter(
        self, v_x_valid: np.ndarray, v_y_valid: np.ndarray
    ) -> str:
        """Create vx vs vy scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=VECTOR_DPI)

        # Sample points for performance (if too many)
        if len(v_x_valid) > 100000:
            indices = np.random.choice(len(v_x_valid), 100000, replace=False)
            v_x_sample = v_x_valid[indices]
            v_y_sample = v_y_valid[indices]
        else:
            v_x_sample = v_x_valid
            v_y_sample = v_y_valid

        # Create scatter plot
        scatter = ax.scatter(
            v_x_sample, v_y_sample, c="steelblue", s=1, alpha=0.6, edgecolors="none"
        )

        # Add zero lines
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, linewidth=1)

        # Labels and title
        ax.set_xlabel("X-Component Velocity (m/yr)")
        ax.set_ylabel("Y-Component Velocity (m/yr)")
        ax.set_title("Velocity Components Scatter Plot")
        ax.grid(True, alpha=0.3)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        plt.tight_layout()

        output_path = OUTPUT_DIR / "velocity_scatter.svg"
        plt.savefig(output_path, format="svg", dpi=VECTOR_DPI, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def _create_coverage_map(self, mask: np.ndarray) -> str:
        """Create data coverage visualization."""
        fig, ax = plt.subplots(figsize=(12, 8), dpi=VECTOR_DPI)

        # Calculate coverage percentage
        coverage_pct = np.sum(mask) / mask.size * 100

        # Display coverage map
        im = ax.imshow(mask, cmap="Blues", vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Data Availability")

        # Add coverage statistics
        ax.set_title(f"Velocity Data Coverage: {coverage_pct:.1f}%")
        ax.set_xlabel("Pixel Column")
        ax.set_ylabel("Pixel Row")

        # Add text annotation
        valid_pixels = np.sum(mask)
        total_pixels = mask.size
        stats_text = f"Valid: {valid_pixels:,} pixels\nTotal: {total_pixels:,} pixels\nCoverage: {coverage_pct:.2f}%"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        output_path = OUTPUT_DIR / "coverage_map.svg"
        plt.savefig(output_path, format="svg", dpi=VECTOR_DPI, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def save_individual_figures(self) -> None:
        """Save all individual figures with proper titles."""
        logger.info("Saving individual figures with titles...")

        # RGB previews
        rgb_images = self.create_rgb_previews()

        # Standard RGB with title
        rgb_standard_titled = add_title(
            rgb_images["standard"], f"Landsat RGB (Bands 3,2,1) - {self.image_name}"
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "landsat_rgb_standard.png"),
            cv2.cvtColor(rgb_standard_titled, cv2.COLOR_RGB2BGR),
        )

        # False color infrared with title (note bands)
        rgb_infrared_titled = add_title(
            rgb_images["infrared"],
            f"False Color Infrared (Bands 4,3,2) - {self.image_name}",
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "landsat_rgb_infrared.png"),
            cv2.cvtColor(rgb_infrared_titled, cv2.COLOR_RGB2BGR),
        )

        # Velocity visualizations
        v_images = self.create_velocity_visualizations()

        # Velocity magnitude with title
        v_mag_titled = add_title(
            v_images["magnitude"],
            f"Velocity Magnitude (v) - Range: {V_MAG_RANGE[0]}-{V_MAG_RANGE[1]} m/yr",
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "velocity_magnitude.png"),
            cv2.cvtColor(v_mag_titled, cv2.COLOR_RGB2BGR),
        )

        # X component with title
        v_x_titled = add_title(
            v_images["x_component"],
            f"Velocity X-Component (vx) - Range: {V_COMPONENT_RANGE[0]}-{V_COMPONENT_RANGE[1]} m/yr",
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "velocity_x_component.png"),
            cv2.cvtColor(v_x_titled, cv2.COLOR_RGB2BGR),
        )

        # Y component with title
        v_y_titled = add_title(
            v_images["y_component"],
            f"Velocity Y-Component (vy) - Range: {V_COMPONENT_RANGE[0]}-{V_COMPONENT_RANGE[1]} m/yr",
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "velocity_y_component.png"),
            cv2.cvtColor(v_y_titled, cv2.COLOR_RGB2BGR),
        )

        # Classification overlay
        classification_overlay = self.create_classification_overlay()
        if classification_overlay is not None:
            class_titled = add_title(
                classification_overlay,
                f"RGB + Classification Overlay (CI=Blue, DCI=Orange) - {self.image_name}",
            )
            cv2.imwrite(
                str(OUTPUT_DIR / "rgb_classification_overlay.png"),
                cv2.cvtColor(class_titled, cv2.COLOR_RGB2BGR),
            )

        # Velocity overlays
        velocity_overlays = self.create_velocity_overlays()

        # RGB + Velocity Magnitude overlay
        mag_overlay_titled = add_title(
            velocity_overlays["magnitude"],
            f"RGB + Velocity Magnitude Overlay (50% transparency) - {self.image_name}",
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "rgb_velocity_magnitude_overlay.png"),
            cv2.cvtColor(mag_overlay_titled, cv2.COLOR_RGB2BGR),
        )

        # RGB + X component overlay
        x_overlay_titled = add_title(
            velocity_overlays["x_component"],
            f"RGB + Velocity X-Component Overlay (50% transparency) - {self.image_name}",
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "rgb_velocity_x_overlay.png"),
            cv2.cvtColor(x_overlay_titled, cv2.COLOR_RGB2BGR),
        )

        # RGB + Y component overlay
        y_overlay_titled = add_title(
            velocity_overlays["y_component"],
            f"RGB + Velocity Y-Component Overlay (50% transparency) - {self.image_name}",
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "rgb_velocity_y_overlay.png"),
            cv2.cvtColor(y_overlay_titled, cv2.COLOR_RGB2BGR),
        )

        logger.info("Saved all individual figures with titles")

    def generate_metadata_files(self) -> None:
        """Generate metadata files."""
        logger.info("Generating metadata files...")

        # Processing log
        processing_log = {
            "image_name": self.image_name,
            "server": self.server,
            "landsat_path": str(self.landsat_path),
            "velocity_path": str(self.velocity_path),
            "labels_path": str(self.labels_path),
            "processing_date": str(pd.Timestamp.now()),
            "data_shape": {
                "landsat": self.landsat_data.shape
                if self.landsat_data is not None
                else None,
                "velocity": self.velocity_data.shape
                if self.velocity_data is not None
                else None,
                "labels": self.labels_data.shape
                if self.labels_data is not None
                else None,
            },
            "coordinate_systems": {
                "landsat_epsg": int(str(self.landsat_meta["crs"]).split(":")[-1])
                if self.landsat_meta
                else None,
                "datacube_epsg": self.stats.get("datacube_epsg")
                if self.stats
                else None,
                "cross_zone_reprojection": self.stats.get("cross_zone_reproj", False)
                if self.stats
                else None,
            },
            "velocity_statistics": self.stats.get("velocity_stats", {})
            if self.stats
            else {},
            "coverage_percent": self.stats.get("coverage_percent", 0)
            if self.stats
            else 0,
        }

        with open(OUTPUT_DIR / "processing_log.json", "w") as f:
            json.dump(processing_log, f, indent=2)

        logger.info("Generated metadata files")

    def generate_all_visualizations(self) -> None:
        """Generate complete set of individual visualizations."""
        logger.info("Starting individual visualization generation...")

        # Load data
        self.load_data()

        # Generate all visualizations
        self.create_statistical_plots()
        self.save_individual_figures()
        self.generate_metadata_files()

        logger.info(
            f"Complete visualization generation finished. Outputs in: {OUTPUT_DIR}"
        )


def main():
    """Main function to run velocity mosaic visualization generation."""
    parser = argparse.ArgumentParser(
        description="Generate individual velocity mosaic visualizations for thesis"
    )
    parser.add_argument(
        "--server",
        default="desktop",
        choices=["desktop", "bilbo", "frodo"],
        help="Server configuration to use",
    )
    parser.add_argument(
        "--image", default="image1", help="Image name to process (default: image1)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=THESIS_DPI,
        help=f"Output DPI for raster images (default: {THESIS_DPI})",
    )

    args = parser.parse_args()

    # Create visualizer and generate all outputs
    visualizer = VelocityMosaicVisualizer(server=args.server, image_name=args.image)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
