#!/usr/bin/env python3
"""
Generate analysis plots for thesis figures:
 - Class balance per split (stacked bars)
 - Ignore/valid fractions per split
 - Per-image debris/clean/background fractions histogram
 - Velocity magnitude histogram (sampled slices)
 - Elevation & slope distributions (sampled slices)
 - Slope vs elevation hexbin
 - Feature correlation heatmap (elevation, slope, NDVI, NDSI, velocity, physics flow)

Defaults target the processed dataset: bibek_w512_o64_f1_v2_phys64_s1_velocity.
Outputs are saved under visualization/output/dataset_analytics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glacier_mapping.data.slice import IGNORE_LABEL
from glacier_mapping.utils.config import load_server_config

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# Defaults (desktop server paths + dataset name)
DEFAULT_DATASET_NAME = "bibek_w512_o64_f1_v2"
DEFAULT_OUTPUT_DIR = Path("visualization/output/dataset_analytics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset analysis plots.")
    parser.add_argument("--server", default="desktop", choices=["desktop", "bilbo", "frodo"])
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Processed dataset name under processed_data_path.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Override processed dataset path; otherwise uses server.processed_data_path/dataset-name",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for plots.",
    )
    parser.add_argument(
        "--max-slices-per-split",
        type=int,
        default=None,
        help="Number of slices to sample per split for channel distributions (None = all slices).",
    )
    parser.add_argument(
        "--max-pixels-per-split",
        type=int,
        default=None,
        help="Maximum random pixels to sample per split (after masking). None = use all valid pixels.",
    )
    return parser.parse_args()


def load_paths(server: str, dataset_name: str, processed_override: Optional[str]) -> Dict[str, Path]:
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


def stacked_bar(values: Dict[str, Dict[str, float]], title: str, out_path: Path):
    splits = list(values.keys())
    bg = [values[s]["background"] for s in splits]
    ci = [values[s]["clean_ice"] for s in splits]
    deb = [values[s]["debris_ice"] for s in splits]
    mask = [values[s]["masked_invalid"] for s in splits]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.bar(splits, bg, label="Background", color="#6e6e6e")
    ax.bar(splits, ci, bottom=bg, label="Clean ice", color="#00bcd4")
    ax.bar(splits, deb, bottom=np.array(bg) + np.array(ci), label="Debris ice", color="#ff6ec7")
    ax.bar(
        splits,
        mask,
        bottom=np.array(bg) + np.array(ci) + np.array(deb),
        label="Ignore/Masked",
        color="#b0b0b0",
    )
    ax.set_ylabel("Percentage of all pixels")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def debris_histogram(slice_meta: pd.DataFrame, out_path: Path):
    agg = slice_meta.groupby("Landsat ID")[["Background Percentage", "Clean Ice Percentage", "Debris Percentage"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.hist(agg["Debris Percentage"] * 100, bins=30, color="#ff6ec7", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Debris fraction per image (%)")
    ax.set_ylabel("Count of scenes")
    ax.set_title("Per-image debris coverage")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def sample_slices(
    processed_dir: Path,
    band_names: List[str],
    max_slices_per_split: Optional[int],
    max_pixels_per_split: Optional[int],
) -> Dict[str, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(42)
    splits = ["train", "val", "test"]
    samples: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in splits}

    for split in splits:
        split_dir = processed_dir / split
        if not split_dir.exists():
            continue
        tiff_files = sorted(split_dir.glob("tiff_*.npy"))
        if not tiff_files:
            continue
        if max_slices_per_split is not None:
            tiff_files = rng.choice(
                tiff_files,
                size=min(len(tiff_files), max_slices_per_split),
                replace=False,
            )
        feat_accum: List[np.ndarray] = []
        slope_accum: List[np.ndarray] = []
        elev_accum: List[np.ndarray] = []
        ndvi_accum: List[np.ndarray] = []
        ndsi_accum: List[np.ndarray] = []
        vel_accum: List[np.ndarray] = []
        flow_accum: List[np.ndarray] = []

        for tf in tiff_files:
            data = np.load(tf)
            mask_path = tf.with_name(tf.name.replace("tiff_", "mask_"))
            if not mask_path.exists():
                continue
            mask = np.load(mask_path)
            valid = mask != IGNORE_LABEL
            if not np.any(valid):
                continue

            valid_idx = np.flatnonzero(valid.reshape(-1))
            if valid_idx.size == 0:
                continue
            if max_pixels_per_split is None:
                chosen = valid_idx
            else:
                take = min(valid_idx.size, max_pixels_per_split // max(1, len(tiff_files)))
                if take == 0:
                    continue
                chosen = rng.choice(valid_idx, size=take, replace=False)
            data_flat = data.reshape(-1, data.shape[2])[chosen]

            def maybe_append(name: str, target: List[np.ndarray]):
                if name in band_names:
                    target.append(data_flat[:, band_names.index(name)])

            maybe_append("slope_deg", slope_accum)
            maybe_append("elevation", elev_accum)
            maybe_append("NDVI", ndvi_accum)
            maybe_append("NDSI", ndsi_accum)
            if {"velocity_x", "velocity_y", "velocity_mask"}.issubset(set(band_names)):
                vx = data_flat[:, band_names.index("velocity_x")]
                vy = data_flat[:, band_names.index("velocity_y")]
                vmask = data_flat[:, band_names.index("velocity_mask")] > 0
                vel_mag = np.hypot(vx, vy)
                vel_accum.append(vel_mag[vmask])
            if "flow_accumulation" in band_names:
                flow_accum.append(data_flat[:, band_names.index("flow_accumulation")])

        def cat(arrs: List[np.ndarray]) -> np.ndarray:
            return np.concatenate(arrs) if arrs else np.array([])

        samples[split] = {
            "slope": cat(slope_accum),
            "elevation": cat(elev_accum),
            "ndvi": cat(ndvi_accum),
            "ndsi": cat(ndsi_accum),
            "velocity": cat(vel_accum),
            "flow": cat(flow_accum),
        }
    return samples


def hist_plot(data: np.ndarray, title: str, xlabel: str, color: str, out_path: Path, bins: int = 60):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def hexbin_plot(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    hb = ax.hexbin(x, y, gridsize=100, cmap="viridis", mincnt=5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Pixel count")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def correlation_plot(data: pd.DataFrame, out_path: Path, title: str):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    paths = load_paths(args.server, args.dataset_name, args.processed_dir)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata files
    slice_meta = pd.read_csv(paths["slice_meta"])
    stats = json.loads(paths["dataset_stats"].read_text())
    band_meta = json.loads(paths["band_meta"].read_text())
    band_names = band_meta["band_names"]

    # 1) Class balance (all pixels) per split
    split_pct_all = {
        split: stats[split]["percentages_all_pixels"]
        for split in ("train", "val", "test")
        if split in stats
    }
    stacked_bar(split_pct_all, "Class distribution (all pixels)", out_dir / "class_distribution_all_pixels.png")

    # 2) Valid vs ignore per split
    split_ignore = {}
    for split in ("train", "val", "test"):
        if split not in stats:
            continue
        pct = stats[split]["percentages_all_pixels"]
        split_ignore[split] = {
            "valid": 100.0 - pct["masked_invalid"],
            "masked_invalid": pct["masked_invalid"],
        }
    if split_ignore:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        splits = list(split_ignore.keys())
        valid = [split_ignore[s]["valid"] for s in splits]
        masked = [split_ignore[s]["masked_invalid"] for s in splits]
        ax.bar(splits, valid, label="Valid", color="#4caf50")
        ax.bar(splits, masked, bottom=valid, label="Ignore", color="#b0b0b0")
        ax.set_ylabel("Percentage of all pixels")
        ax.set_title("Valid vs ignore mask")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "valid_vs_ignore.png", bbox_inches="tight")
        plt.close(fig)

    # 3) Debris fraction histogram per image
    debris_histogram(slice_meta, out_dir / "debris_fraction_per_image.png")

    # 4) Sample slices for channel distributions
    samples = sample_slices(
        paths["processed"],
        band_names,
        args.max_slices_per_split,
        args.max_pixels_per_split,
    )
    feature_keys = set()
    for split_dict in samples.values():
        feature_keys.update(split_dict.keys())
    merged = {}
    for k in feature_keys:
        arrays = [samples[s][k] for s in samples if k in samples[s] and samples[s][k].size > 0]
        merged[k] = np.concatenate(arrays) if arrays else np.array([])

    # Elevation & slope
    if merged["elevation"].size:
        hist_plot(
            merged["elevation"],
            "Elevation distribution (all splits sampled)",
            "Elevation (m)",
            "#6e6e6e",
            out_dir / "elevation_hist.png",
        )
    if merged["slope"].size:
        hist_plot(
            merged["slope"],
            "Slope distribution (all splits sampled)",
            "Slope (deg)",
            "#8a2be2",
            out_dir / "slope_hist.png",
        )
    if merged["slope"].size and merged["elevation"].size:
        hexbin_plot(
            merged["elevation"],
            merged["slope"],
            "Slope vs Elevation (sampled pixels)",
            "Elevation (m)",
            "Slope (deg)",
            out_dir / "slope_vs_elevation_hexbin.png",
        )

    # Velocity magnitude
    if merged["velocity"].size:
        hist_plot(
            merged["velocity"],
            "Velocity magnitude (valid mask only)",
            "Speed (m/yr)",
            "#ff9800",
            out_dir / "velocity_magnitude_hist.png",
            bins=80,
        )

    # Spectral/physics
    if merged["ndvi"].size:
        hist_plot(
            merged["ndvi"],
            "NDVI distribution",
            "NDVI",
            "#00bcd4",
            out_dir / "ndvi_hist.png",
        )
    if merged["ndsi"].size:
        hist_plot(
            merged["ndsi"],
            "NDSI distribution",
            "NDSI",
            "#4caf50",
            out_dir / "ndsi_hist.png",
        )
    # Correlation heatmap
    corr_features = []
    corr_names = []
    for name, arr in (
        ("Elevation (m)", merged["elevation"]),
        ("Slope (deg)", merged["slope"]),
        ("NDVI", merged["ndvi"]),
        ("NDSI", merged["ndsi"]),
        ("Velocity (m/yr)", merged["velocity"]),
        ("Flow accumulation", merged["flow"]),
    ):
        if arr.size:
            corr_features.append(arr)
            corr_names.append(name)
    if corr_features and len(corr_features) >= 2:
        min_len = min(len(a) for a in corr_features)
        aligned = [a[:min_len] for a in corr_features]
        df_corr = pd.DataFrame(np.stack(aligned, axis=1), columns=corr_names)
        correlation_plot(df_corr, out_dir / "feature_correlation.png", "Feature correlation (sampled)")

    logger.info("Saved plots to %s", out_dir)


if __name__ == "__main__":
    main()
