#!/usr/bin/env python3
"""
Thesis-ready full-scene visualizations.

Primary target: image125.tif (best balance with moderate nodata).
Other options to try later (not rendered by default):
  - image192.tif (low nodata, debris scarcer)
  - image173.tif (very balanced but ~45% nodata stripes)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from glacier_mapping.data.slice import IGNORE_LABEL, get_mask, get_tiff_np
from glacier_mapping.utils.config import load_server_config

# Non-interactive backend for multiprocessing safety
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def percentile_stretch(img: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Stretch image to 0-1 using percentiles to improve contrast."""
    lo_v, hi_v = np.nanpercentile(img, [lo, hi])
    if not np.isfinite(lo_v) or not np.isfinite(hi_v):
        return np.zeros_like(img, dtype=float)
    if hi_v - lo_v < 1e-6:
        return np.clip(img / max(1e-6, hi_v), 0, 1)
    stretched = (img - lo_v) / (hi_v - lo_v)
    return np.clip(stretched, 0, 1)


def save_single_channel(
    channel: np.ndarray,
    out_path: Path,
    cmap: str = "viridis",
    with_colorbar: bool = True,
    normalize: bool = True,
    dpi: int = 500,
    title: str | None = None,
    label: str | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    data = percentile_stretch(channel) if normalize else channel
    im = ax.imshow(data, cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14)
    if with_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
        if label:
            cbar.set_label(label, fontsize=12)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_rgb(rgb: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_overlay(rgb: np.ndarray, mask: np.ndarray, out_path: Path, title: str):
    """Overlay ground-truth labels on RGB."""
    cmap = mcolors.ListedColormap(["#4a4a4a", "#00e5ff", "#ff5cc7", "#000000"])
    bounds = [0, 1, 2, 255, 256]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)
    ax.imshow(rgb)
    ax.imshow(mask, cmap=cmap, norm=norm, alpha=0.35)
    ax.axis("off")
    ax.set_title(title, fontsize=14)

    handles = [
        plt.Line2D([0], [0], color="#4a4a4a", lw=4, label="Background"),
        plt.Line2D([0], [0], color="#00e5ff", lw=4, label="Clean ice"),
        plt.Line2D([0], [0], color="#ff5cc7", lw=4, label="Debris ice"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def compute_velocity_magnitude(vx: np.ndarray, vy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mag = np.hypot(vx, vy)
    return np.where(mask > 0, mag, np.nan)


def render_channel_overlay(
    rgb: np.ndarray,
    channel: np.ndarray,
    out_path: Path,
    cmap: str,
    title: str,
    alpha: float = 0.45,
    normalize: bool = True,
    label: str | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)
    ax.imshow(rgb)
    data = percentile_stretch(channel) if normalize else channel
    im = ax.imshow(data, cmap=cmap, alpha=alpha)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=12)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_triptych(
    rgb: np.ndarray,
    channel: np.ndarray,
    out_path: Path,
    cmap: str,
    title: str,
    label: str | None = None,
    normalize: bool = True,
):
    """Side-by-side: RGB | channel | RGB+channel overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=400)

    axes[0].imshow(rgb)
    axes[0].axis("off")
    axes[0].set_title("RGB", fontsize=12)

    chan_vis = percentile_stretch(channel) if normalize else channel
    im = axes[1].imshow(chan_vis, cmap=cmap)
    axes[1].axis("off")
    axes[1].set_title("Channel", fontsize=12)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=10)

    axes[2].imshow(rgb)
    im_overlay = axes[2].imshow(chan_vis, cmap=cmap, alpha=0.45)
    axes[2].axis("off")
    axes[2].set_title("Overlay", fontsize=12)
    cbar_overlay = fig.colorbar(im_overlay, ax=axes[2], fraction=0.046, pad=0.02)
    if label:
        cbar_overlay.set_label(label, fontsize=10)

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def to_title(name: str) -> str:
    return name.replace("_", " ").title()


def load_scene(
    server: str,
    scene_name: str,
    add_velocity: bool = True,
    add_ndvi: bool = True,
    add_ndwi: bool = True,
    add_ndsi: bool = True,
    add_hsv: bool = True,
    physics_res: int | float | str | None = 64,
    physics_scale: float | int | None = 1.0,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load full scene with all derived channels and ground-truth mask."""
    server_cfg = load_server_config(server)
    scene_path = Path(server_cfg["image_dir"]) / scene_name
    dem_path = Path(server_cfg["dem_dir"]) / scene_name
    velocity_path = Path(server_cfg["velocity_dir"]) / scene_name

    band_stack, band_names = get_tiff_np(
        scene_path,
        dem_path,
        velocity_path if add_velocity else None,
        physics_res,
        physics_scale,
        add_ndvi,
        add_ndwi,
        add_ndsi,
        add_hsv,
        return_band_names=True,
        verbose=False,
    )

    labels = gpd.read_file(Path(server_cfg["labels_dir"]) / "HKH_CIDC_5basins_all.shp")
    mask_multi = get_mask(scene_path, labels)
    mask = np.zeros(mask_multi.shape[:2], dtype=np.uint8)
    for i in range(mask_multi.shape[2]):
        mask[mask_multi[:, :, i] == 1] = i + 1

    invalid = np.sum(band_stack, axis=2) == 0
    mask[invalid] = IGNORE_LABEL

    return band_stack, band_names, mask


def build_jobs(
    band_stack: np.ndarray,
    band_names: List[str],
    mask: np.ndarray,
    out_dir: Path,
    stem: str,
) -> List[Tuple[str, callable]]:
    jobs: List[Tuple[str, callable]] = []

    # RGB from first three BGR channels -> reorder to true RGB (BGR -> RGB)
    rgb_raw = band_stack[:, :, :3][:, :, [2, 1, 0]]
    rgb = percentile_stretch(rgb_raw)
    jobs.append(
        (
            f"{stem}_rgb",
            lambda: render_rgb(rgb, out_dir / f"{stem}_rgb.png", title=f"{stem} RGB"),
        )
    )
    jobs.append(
        (
            f"{stem}_overlay",
            lambda: render_overlay(
                rgb, mask, out_dir / f"{stem}_overlay.png", title=f"{stem} Ground Truth Overlay"
            ),
        )
    )

    # DEM
    if "elevation" in band_names:
        chan = band_stack[:, :, band_names.index("elevation")]
        jobs.append(
            (
                f"{stem}_elevation_m",
                lambda c=chan: save_single_channel(
                    c,
                    out_dir / f"{stem}_elevation_m.png",
                    cmap="terrain",
                    title=f"{stem} Elevation (m)",
                    label="Elevation (m)",
                ),
            )
        )
        jobs.append(
            (
                f"{stem}_overlay_elevation",
                lambda c=chan: render_channel_overlay(
                    rgb,
                    c,
                    out_dir / f"{stem}_overlay_elevation.png",
                    cmap="terrain",
                    title=f"{stem} Elevation Overlay",
                    label="Elevation (m)",
                ),
            )
        )
        jobs.append(
            (
                f"{stem}_triptych_elevation",
                lambda c=chan: render_triptych(
                    rgb,
                    c,
                    out_dir / f"{stem}_triptych_elevation.png",
                    cmap="terrain",
                    title=f"{stem} Elevation (m)",
                    label="Elevation (m)",
                ),
            )
        )
    if "slope_deg" in band_names:
        chan = band_stack[:, :, band_names.index("slope_deg")]
        jobs.append(
            (
                f"{stem}_slope_deg",
                lambda c=chan: save_single_channel(
                    c,
                    out_dir / f"{stem}_slope_deg.png",
                    cmap="magma",
                    title=f"{stem} Slope (deg)",
                    label="Slope (deg)",
                ),
            )
        )
        jobs.append(
            (
                f"{stem}_overlay_slope",
                lambda c=chan: render_channel_overlay(
                    rgb,
                    c,
                    out_dir / f"{stem}_overlay_slope.png",
                    cmap="magma",
                    title=f"{stem} Slope Overlay",
                    label="Slope (deg)",
                ),
            )
        )
        jobs.append(
            (
                f"{stem}_triptych_slope",
                lambda c=chan: render_triptych(
                    rgb,
                    c,
                    out_dir / f"{stem}_triptych_slope.png",
                    cmap="magma",
                    title=f"{stem} Slope (deg)",
                    label="Slope (deg)",
                ),
            )
        )

    # Spectral indices
    for idx_name in ("NDVI", "NDWI", "NDSI"):
        if idx_name in band_names:
            chan = band_stack[:, :, band_names.index(idx_name)]
            jobs.append(
                (
                    f"{stem}_{idx_name.lower()}",
                    lambda c=chan, name=idx_name: save_single_channel(
                        c,
                        out_dir / f"{stem}_{name.lower()}.png",
                        cmap="coolwarm",
                        title=f"{stem} {name}",
                        label=f"{name} (unitless)",
                    ),
                )
            )
            jobs.append(
                (
                    f"{stem}_overlay_{idx_name.lower()}",
                    lambda c=chan, name=idx_name: render_channel_overlay(
                        rgb,
                        c,
                        out_dir / f"{stem}_overlay_{name.lower()}.png",
                        cmap="coolwarm",
                        title=f"{stem} {name} Overlay",
                        label=f"{name} (unitless)",
                    ),
                )
            )
            jobs.append(
                (
                    f"{stem}_triptych_{idx_name.lower()}",
                    lambda c=chan, name=idx_name: render_triptych(
                        rgb,
                        c,
                        out_dir / f"{stem}_triptych_{name.lower()}.png",
                        cmap="coolwarm",
                        title=f"{stem} {name}",
                        label=f"{name} (unitless)",
                    ),
                )
            )

    # HSV (illumination cues)
    for idx_name in ("H", "S", "V"):
        if idx_name in band_names:
            chan = band_stack[:, :, band_names.index(idx_name)]
            jobs.append(
                (
                    f"{stem}_hsv_{idx_name.lower()}",
                    lambda c=chan, n=idx_name: save_single_channel(
                        c,
                        out_dir / f"{stem}_hsv_{n.lower()}.png",
                        cmap="twilight",
                        title=f"{stem} HSV {n}",
                        label=f"HSV {n} (unitless)",
                    ),
                )
            )
            jobs.append(
                (
                    f"{stem}_overlay_hsv_{idx_name.lower()}",
                    lambda c=chan, n=idx_name: render_channel_overlay(
                        rgb,
                        c,
                        out_dir / f"{stem}_overlay_hsv_{n.lower()}.png",
                        cmap="twilight",
                        title=f"{stem} HSV {n} Overlay",
                        label=f"HSV {n} (unitless)",
                    ),
                )
            )
            jobs.append(
                (
                    f"{stem}_triptych_hsv_{idx_name.lower()}",
                    lambda c=chan, n=idx_name: render_triptych(
                        rgb,
                        c,
                        out_dir / f"{stem}_triptych_hsv_{n.lower()}.png",
                        cmap="twilight",
                        title=f"{stem} HSV {n}",
                        label=f"HSV {n} (unitless)",
                    ),
                )
            )

    # Physics-informed channels
    physics_channels: Iterable[Tuple[str, str, str]] = (
        ("flow_accumulation", "plasma", "Flow accumulation (unitless)"),
        ("tpi", "PiYG", "Topographic position index (m)"),
        ("roughness", "cividis", "Roughness (m)"),
        ("plan_curvature", "Spectral", "Plan curvature (1/m)"),
    )
    for ch_name, cmap, label in physics_channels:
        if ch_name in band_names:
            chan = band_stack[:, :, band_names.index(ch_name)]
            jobs.append(
                (
                    f"{stem}_physics_{ch_name}",
                    lambda c=chan, n=ch_name, cm=cmap, lab=label: save_single_channel(
                        c,
                        out_dir / f"{stem}_physics_{n}.png",
                        cmap=cm,
                        title=f"{stem} Physics {to_title(n)}",
                        label=lab,
                    ),
                )
            )
            jobs.append(
                (
                    f"{stem}_overlay_physics_{ch_name}",
                    lambda c=chan, n=ch_name, cm=cmap, lab=label: render_channel_overlay(
                        rgb,
                        c,
                        out_dir / f"{stem}_overlay_physics_{n}.png",
                        cmap=cm,
                        title=f"{stem} Physics {to_title(n)} Overlay",
                        label=lab,
                    ),
                )
            )
            jobs.append(
                (
                    f"{stem}_triptych_physics_{ch_name}",
                    lambda c=chan, n=ch_name, cm=cmap, lab=label: render_triptych(
                        rgb,
                        c,
                        out_dir / f"{stem}_triptych_physics_{n}.png",
                        cmap=cm,
                        title=f"{stem} Physics {to_title(n)}",
                        label=lab,
                    ),
                )
            )

    # Velocity
    if "velocity" in band_names:
        v = band_stack[:, :, band_names.index("velocity")]
        vx = band_stack[:, :, band_names.index("velocity_x")]
        vy = band_stack[:, :, band_names.index("velocity_y")]
        vmask = band_stack[:, :, band_names.index("velocity_mask")]
        mag = compute_velocity_magnitude(vx, vy, vmask)
        jobs.extend(
            [
                (
                    f"{stem}_velocity_speed",
                    lambda c=v: save_single_channel(
                        c,
                        out_dir / f"{stem}_velocity_speed.png",
                        cmap="inferno",
                        title=f"{stem} Velocity Speed",
                        label="Speed (m/yr)",
                    ),
                ),
                (
                    f"{stem}_velocity_vx",
                    lambda c=vx: save_single_channel(
                        c,
                        out_dir / f"{stem}_velocity_vx.png",
                        cmap="coolwarm",
                        title=f"{stem} Velocity Vx",
                        label="Vx (m/yr)",
                    ),
                ),
                (
                    f"{stem}_velocity_vy",
                    lambda c=vy: save_single_channel(
                        c,
                        out_dir / f"{stem}_velocity_vy.png",
                        cmap="coolwarm",
                        title=f"{stem} Velocity Vy",
                        label="Vy (m/yr)",
                    ),
                ),
                (
                    f"{stem}_velocity_magnitude",
                    lambda c=mag: save_single_channel(
                        c,
                        out_dir / f"{stem}_velocity_magnitude.png",
                        cmap="inferno",
                        title=f"{stem} Velocity Magnitude",
                        label="Speed (m/yr)",
                    ),
                ),
                (
                    f"{stem}_overlay_velocity_speed",
                    lambda c=v: render_channel_overlay(
                        rgb,
                        c,
                        out_dir / f"{stem}_overlay_velocity_speed.png",
                        cmap="inferno",
                        title=f"{stem} Velocity Speed Overlay",
                        label="Speed (m/yr)",
                    ),
                ),
                (
                    f"{stem}_overlay_velocity_vx",
                    lambda c=vx: render_channel_overlay(
                        rgb,
                        c,
                        out_dir / f"{stem}_overlay_velocity_vx.png",
                        cmap="coolwarm",
                        title=f"{stem} Velocity Vx Overlay",
                        label="Vx (m/yr)",
                    ),
                ),
                (
                    f"{stem}_overlay_velocity_vy",
                    lambda c=vy: render_channel_overlay(
                        rgb,
                        c,
                        out_dir / f"{stem}_overlay_velocity_vy.png",
                        cmap="coolwarm",
                        title=f"{stem} Velocity Vy Overlay",
                        label="Vy (m/yr)",
                    ),
                ),
                (
                    f"{stem}_overlay_velocity_magnitude",
                    lambda c=mag: render_channel_overlay(
                        rgb,
                        c,
                        out_dir / f"{stem}_overlay_velocity_magnitude.png",
                        cmap="inferno",
                        title=f"{stem} Velocity Magnitude Overlay",
                        label="Speed (m/yr)",
                    ),
                ),
                (
                    f"{stem}_triptych_velocity_speed",
                    lambda c=v: render_triptych(
                        rgb,
                        c,
                        out_dir / f"{stem}_triptych_velocity_speed.png",
                        cmap="inferno",
                        title=f"{stem} Velocity Speed",
                        label="Speed (m/yr)",
                    ),
                ),
                (
                    f"{stem}_triptych_velocity_vx",
                    lambda c=vx: render_triptych(
                        rgb,
                        c,
                        out_dir / f"{stem}_triptych_velocity_vx.png",
                        cmap="coolwarm",
                        title=f"{stem} Velocity Vx",
                        label="Vx (m/yr)",
                    ),
                ),
                (
                    f"{stem}_triptych_velocity_vy",
                    lambda c=vy: render_triptych(
                        rgb,
                        c,
                        out_dir / f"{stem}_triptych_velocity_vy.png",
                        cmap="coolwarm",
                        title=f"{stem} Velocity Vy",
                        label="Vy (m/yr)",
                    ),
                ),
                (
                    f"{stem}_triptych_velocity_magnitude",
                    lambda c=mag: render_triptych(
                        rgb,
                        c,
                        out_dir / f"{stem}_triptych_velocity_magnitude.png",
                        cmap="inferno",
                        title=f"{stem} Velocity Magnitude",
                        label="Speed (m/yr)",
                    ),
                ),
            ]
        )

    return jobs


def run_jobs(jobs: List[Tuple[str, callable]], workers: int):
    if workers and workers > 1:
        logger.info(f"Rendering with {workers} workers (threaded).")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda job: job[1](), jobs))
    else:
        logger.info("Rendering sequentially.")
        for _, fn in jobs:
            fn()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full-scene visualizations.")
    parser.add_argument("--server", default="desktop", choices=["desktop", "bilbo", "frodo"])
    parser.add_argument("--scene", default="image125.tif", help="Scene filename (e.g., image125.tif)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for PNGs. Defaults to visualization/output/image125_full",
    )
    parser.add_argument("--workers", type=int, default=0, help="Thread workers (0/1 = sequential).")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    default_out = Path("visualization/output/image125_full")
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading scene {args.scene} from server={args.server}")
    band_stack, band_names, mask = load_scene(
        args.server,
        args.scene,
        add_velocity=True,
        add_ndvi=True,
        add_ndwi=True,
        add_ndsi=True,
        add_hsv=True,
        physics_res=64,
        physics_scale=1.0,
    )

    stem = Path(args.scene).stem
    jobs = build_jobs(band_stack, band_names, mask, out_dir, stem)
    logger.info(f"Prepared {len(jobs)} plots -> {out_dir}")

    run_jobs(jobs, args.workers)
    logger.info("Done.")


if __name__ == "__main__":
    main()
