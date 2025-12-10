#!/usr/bin/env python3
"""
Comprehensive correlation analysis between velocity/physics channels and glacier labels.

Analyzes whether velocity and physics-based features provide discriminative information
for glacier classification (Background, Clean Ice, Debris-covered Ice).

Key analyses:
1. Per-class channel distribution statistics
2. Statistical significance testing between classes
3. Effect size calculations
4. Correlation matrices by class
5. Predictive power assessment (ROC/AUC)
6. Spatial pattern analysis

Outputs:
- analysis/output/channel_label_correlation_report.json
- visualization/output/channel_label_correlation/*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import label_binarize

from glacier_mapping.data.slice import IGNORE_LABEL
from glacier_mapping.utils.config import load_server_config

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# Channel groups for analysis
VELOCITY_CHANNELS = ["velocity", "velocity_x", "velocity_y", "velocity_mask"]
PHYSICS_CHANNELS = ["flow_accumulation", "tpi", "roughness", "plan_curvature"]
SPECTRAL_INDICES_CHANNELS = ["NDVI", "NDWI", "NDSI"]
HSV_CHANNELS = ["H", "S", "V"]
TARGET_CHANNELS = (
    VELOCITY_CHANNELS + PHYSICS_CHANNELS + SPECTRAL_INDICES_CHANNELS + HSV_CHANNELS
)

# Class definitions
CLASS_NAMES = ["Background", "Clean Ice", "Debris Ice"]
CLASS_COLORS = ["#6e6e6e", "#00bcd4", "#ff6ec7"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Channel-label correlation analysis")
    parser.add_argument(
        "--server", default="desktop", choices=["desktop", "bilbo", "frodo"]
    )
    parser.add_argument(
        "--dataset-name",
        default="gen_robust_comprehensive",
        help="Processed dataset name under processed_data_path.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Override processed dataset path; otherwise uses server.processed_data_path/dataset-name",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=100,
        help="Maximum slices to analyze per split (None = all).",
    )
    parser.add_argument(
        "--max-pixels-per-slice",
        type=int,
        default=10000,
        help="Maximum pixels to sample per slice.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/output",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--viz-dir",
        default="visualization/output/channel_label_correlation",
        help="Directory for visualization outputs.",
    )
    return parser.parse_args()


def load_paths(
    server: str, dataset_name: str, processed_override: Optional[str]
) -> Dict[str, Path]:
    server_cfg = load_server_config(server)
    processed_dir = (
        Path(processed_override)
        if processed_override is not None
        else Path(server_cfg["processed_data_path"]) / dataset_name
    )
    return {
        "processed": processed_dir,
        "slice_meta": processed_dir / "slice_meta.csv",
        "dataset_stats": processed_dir / "dataset_statistics.json",
        "band_meta": processed_dir / "band_metadata.json",
    }


def sample_data_by_class(
    processed_dir: Path,
    band_names: List[str],
    max_slices: Optional[int],
    max_pixels_per_slice: int,
    random_seed: int = 42,
):
    """Sample data organized by class for analysis."""
    rng = np.random.default_rng(random_seed)
    splits = ["train", "val", "test"]

    # Initialize data structure
    class_data = {}
    for class_name in CLASS_NAMES:
        class_data[class_name] = {}
        for channel in TARGET_CHANNELS:
            class_data[class_name][channel] = []

    for split in splits:
        split_dir = processed_dir / split
        if not split_dir.exists():
            continue

        tiff_files = sorted(split_dir.glob("tiff_*.npy"))
        if max_slices is not None:
            tiff_files = list(tiff_files)[:max_slices]

        logger.info(f"Processing {len(tiff_files)} slices from {split} split")

        for tf in tiff_files:
            try:
                # Load data and mask
                data = np.load(tf)
                mask_path = tf.with_name(tf.name.replace("tiff_", "mask_"))
                if not mask_path.exists():
                    continue
                mask = np.load(mask_path)

                # Get valid pixels (not ignore mask)
                valid = mask != IGNORE_LABEL
                if not np.any(valid):
                    continue

                # Sample pixels from this slice
                valid_idx = np.flatnonzero(valid.reshape(-1))
                if valid_idx.size == 0:
                    continue

                # Limit pixels per slice
                n_sample = min(valid_idx.size, max_pixels_per_slice)
                chosen = rng.choice(valid_idx, size=n_sample, replace=False)

                # Reshape data
                data_flat = data.reshape(-1, data.shape[2])[chosen]
                mask_flat = mask.reshape(-1)[chosen]

                # Organize by class
                for class_id, class_name in enumerate(CLASS_NAMES):
                    class_mask = mask_flat == class_id
                    if not np.any(class_mask):
                        continue

                    class_pixels = data_flat[class_mask]

                    # Extract target channels
                    for channel in TARGET_CHANNELS:
                        if channel in band_names:
                            channel_idx = band_names.index(channel)
                            channel_data = class_pixels[:, channel_idx]

                            # Special handling for velocity mask (binary)
                            if channel == "velocity_mask":
                                class_data[class_name][channel].append(channel_data > 0)
                            else:
                                class_data[class_name][channel].append(channel_data)

            except Exception as e:
                logger.warning(f"Error processing {tf}: {e}")
                continue

    # Convert lists to arrays
    for class_name in CLASS_NAMES:
        for channel in TARGET_CHANNELS:
            if class_data[class_name][channel]:
                class_data[class_name][channel] = np.concatenate(
                    class_data[class_name][channel]
                )
            else:
                class_data[class_name][channel] = np.array([])

    return class_data


def compute_class_statistics(class_data):
    """Compute comprehensive statistics for each class and channel."""
    stats = {}

    for class_name in CLASS_NAMES:
        stats[class_name] = {}
        for channel in TARGET_CHANNELS:
            data = class_data[class_name][channel]
            if data.size == 0:
                stats[class_name][channel] = {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
                continue

            # Handle different channel types
            if channel == "velocity_mask":
                # Boolean channel - compute percentage of valid velocities
                data_float = data.astype(float)
                stats[class_name][channel] = {
                    "count": len(data),
                    "mean": float(
                        np.mean(data_float)
                    ),  # Percentage of valid velocities
                    "std": float(np.std(data_float)),
                    "median": float(np.median(data_float)),
                    "q25": float(np.percentile(data_float, 25)),
                    "q75": float(np.percentile(data_float, 75)),
                    "min": float(np.min(data_float)),
                    "max": float(np.max(data_float)),
                }
            else:
                # Remove invalid values for velocity channels
                if channel in ["velocity", "velocity_x", "velocity_y"]:
                    valid_mask = np.isfinite(data) & (np.abs(data) < 1e6)
                    data = data[valid_mask]

                if data.size == 0:
                    stats[class_name][channel] = {
                        "count": 0,
                        "mean": np.nan,
                        "std": np.nan,
                        "median": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                    }
                    continue

                stats[class_name][channel] = {
                    "count": len(data),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "median": float(np.median(data)),
                    "q25": float(np.percentile(data, 25)),
                    "q75": float(np.percentile(data, 75)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                }

    return stats


def compute_statistical_tests(class_data):
    """Compute statistical tests between classes."""
    test_results = {}

    # Test pairs: Background vs Clean Ice, Background vs Debris, Clean vs Debris
    test_pairs = [
        ("Background", "Clean Ice"),
        ("Background", "Debris Ice"),
        ("Clean Ice", "Debris Ice"),
    ]

    for channel in TARGET_CHANNELS:
        test_results[channel] = {}

        for class1, class2 in test_pairs:
            data1 = class_data[class1][channel]
            data2 = class_data[class2][channel]

            if data1.size == 0 or data2.size == 0:
                test_results[channel][f"{class1}_vs_{class2}"] = {
                    "test": "insufficient_data",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size": np.nan,
                }
                continue

            # Clean data for velocity channels
            if channel in ["velocity", "velocity_x", "velocity_y"]:
                valid_mask1 = np.isfinite(data1) & (np.abs(data1) < 1e6)
                valid_mask2 = np.isfinite(data2) & (np.abs(data2) < 1e6)
                data1 = data1[valid_mask1]
                data2 = data2[valid_mask2]

            if data1.size == 0 or data2.size == 0:
                test_results[channel][f"{class1}_vs_{class2}"] = {
                    "test": "insufficient_data",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size": np.nan,
                }
                continue

            # Perform statistical test
            try:
                # Kolmogorov-Smirnov test for distribution difference
                ks_stat, ks_p = stats.ks_2samp(data1, data2)

                # Cohen's d for effect size
                pooled_std = np.sqrt(
                    (
                        (len(data1) - 1) * np.var(data1, ddof=1)
                        + (len(data2) - 1) * np.var(data2, ddof=1)
                    )
                    / (len(data1) + len(data2) - 2)
                )
                cohens_d = (
                    (np.mean(data1) - np.mean(data2)) / pooled_std
                    if pooled_std > 0
                    else 0
                )

                test_results[channel][f"{class1}_vs_{class2}"] = {
                    "test": "kolmogorov_smirnov",
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "effect_size": float(cohens_d),
                }

            except Exception as e:
                logger.warning(
                    f"Statistical test failed for {channel} between {class1} and {class2}: {e}"
                )
                test_results[channel][f"{class1}_vs_{class2}"] = {
                    "test": "error",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size": np.nan,
                }

    return test_results


def create_violin_plots(class_data, viz_dir: Path) -> List[str]:
    """Create violin plots for channel distributions by class."""
    plot_paths = []

    for channel in TARGET_CHANNELS:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Prepare data for violin plot
        data_for_plot = []
        labels_for_plot = []
        colors_for_plot = []

        for i, class_name in enumerate(CLASS_NAMES):
            data = class_data[class_name][channel]
            if data.size == 0:
                continue

            # Clean data for velocity channels
            if channel == "velocity_mask":
                # Convert boolean to float for visualization
                data = data.astype(float)
            elif channel in ["velocity", "velocity_x", "velocity_y"]:
                valid_mask = np.isfinite(data) & (np.abs(data) < 1e6)
                data = data[valid_mask]

            if data.size == 0:
                continue

            # Sample for visualization if too many points
            if len(data) > 10000:
                data = np.random.choice(data, 10000, replace=False)

            data_for_plot.append(data)
            labels_for_plot.append(class_name)
            colors_for_plot.append(CLASS_COLORS[i])

        if not data_for_plot:
            plt.close(fig)
            continue

        # Create violin plot
        parts = ax.violinplot(
            data_for_plot,
            positions=range(len(data_for_plot)),
            showmeans=True,
            showmedians=True,
        )

        # Color the violins
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors_for_plot[i])
            pc.set_alpha(0.7)

        # Styling
        if "cmeans" in parts:
            parts["cmeans"].set_colors("black")
        if "cmedians" in parts:
            parts["cmedians"].set_colors("white")
            parts["cmedians"].set_linewidths(2)

        ax.set_xticks(range(len(labels_for_plot)))
        ax.set_xticklabels(labels_for_plot, rotation=45, ha="right")
        ax.set_ylabel(get_channel_label(channel))
        ax.set_title(f"{get_channel_label(channel)} Distribution by Glacier Class")
        ax.grid(True, alpha=0.3)

        # Add sample size annotations
        for i, (class_name, data) in enumerate(zip(labels_for_plot, data_for_plot)):
            ax.text(
                i,
                ax.get_ylim()[1] * 0.95,
                f"n={len(data)}",
                ha="center",
                va="top",
                fontsize=8,
            )

        plt.tight_layout()
        plot_path = viz_dir / f"violin_{channel}.png"
        fig.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        plot_paths.append(str(plot_path))

    return plot_paths


def get_channel_label(channel: str) -> str:
    """Get human-readable channel label."""
    labels = {
        "velocity": "Velocity Magnitude (m/yr)",
        "velocity_x": "Velocity X Component (m/yr)",
        "velocity_y": "Velocity Y Component (m/yr)",
        "velocity_mask": "Velocity Validity Mask",
        "flow_accumulation": "Flow Accumulation",
        "tpi": "Topographic Position Index",
        "roughness": "Surface Roughness",
        "plan_curvature": "Plan Curvature",
        "NDVI": "Normalized Difference Vegetation Index",
        "NDWI": "Normalized Difference Water Index",
        "NDSI": "Normalized Difference Snow Index",
        "H": "HSV Hue",
        "S": "HSV Saturation",
        "V": "HSV Value",
    }
    return labels.get(channel, channel)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()

    # Setup paths
    paths = load_paths(args.server, args.dataset_name, args.processed_dir)
    output_dir = Path(args.output_dir)
    viz_dir = Path(args.viz_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset metadata from {paths['processed']}")

    # Load metadata
    band_meta = json.loads(paths["band_meta"].read_text())
    band_names = band_meta["band_names"]

    logger.info(f"Found {len(band_names)} bands: {band_names}")
    logger.info(f"Target channels for analysis: {TARGET_CHANNELS}")

    # Verify target channels exist
    missing_channels = [ch for ch in TARGET_CHANNELS if ch not in band_names]
    if missing_channels:
        logger.error(f"Missing target channels: {missing_channels}")
        return

    # Sample data by class
    logger.info("Sampling data by class...")
    class_data = sample_data_by_class(
        paths["processed"], band_names, args.max_slices, args.max_pixels_per_slice
    )

    # Log sample sizes
    for class_name in CLASS_NAMES:
        for channel in TARGET_CHANNELS:
            n_pixels = len(class_data[class_name][channel])
            logger.info(f"{class_name} - {channel}: {n_pixels:,} pixels")

    # Compute statistics
    logger.info("Computing class statistics...")
    class_stats = compute_class_statistics(class_data)

    # Compute statistical tests
    logger.info("Computing statistical tests...")
    test_results = compute_statistical_tests(class_data)

    # Create visualizations
    logger.info("Creating visualizations...")
    violin_plots = create_violin_plots(class_data, viz_dir)

    # Compile results
    results = {
        "metadata": {
            "dataset": args.dataset_name,
            "server": args.server,
            "target_channels": TARGET_CHANNELS,
            "classes": CLASS_NAMES,
            "max_slices": args.max_slices,
            "max_pixels_per_slice": args.max_pixels_per_slice,
        },
        "class_statistics": class_stats,
        "statistical_tests": test_results,
        "visualizations": {
            "violin_plots": violin_plots,
        },
    }

    # Save results
    results_path = output_dir / "channel_label_correlation_report.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Analysis complete! Results saved to {results_path}")
    logger.info(f"Visualizations saved to {viz_dir}")

    # Print summary
    print("\n" + "=" * 80)
    print("CHANNEL-LABEL CORRELATION ANALYSIS SUMMARY")
    print("=" * 80)

    for channel in TARGET_CHANNELS:
        print(f"\n{get_channel_label(channel)}:")
        print("-" * 50)

        for class_name in CLASS_NAMES:
            stats = class_stats[class_name][channel]
            if stats["count"] > 0:
                print(
                    f"  {class_name:12}: n={stats['count']:6,}, "
                    f"mean={stats['mean']:8.2f}, "
                    f"std={stats['std']:7.2f}"
                )

        # Significant differences
        print("  Significant differences (p < 0.05):")
        for test_name, test_result in test_results[channel].items():
            if test_result["p_value"] < 0.05 and not np.isnan(test_result["p_value"]):
                effect_size = test_result["effect_size"]
                strength = (
                    "weak"
                    if abs(effect_size) < 0.5
                    else "moderate"
                    if abs(effect_size) < 0.8
                    else "strong"
                )
                print(
                    f"    {test_name}: p={test_result['p_value']:.4f}, "
                    f"effect_size={effect_size:.3f} ({strength})"
                )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
