#!/usr/bin/env python3
"""
Quick visualization of velocity channels for dataset slices.

Generates a figure per slice with:
  - RGB (Landsat)
  - Overlay: RGB with velocity magnitude heatmap
  - Velocity magnitude (v)
  - Velocity vx (symlog)
  - Velocity vy (symlog)
  - Velocity mask

Defaults:
  data root: /home/devj/local-debian/datasets/HKH/gen_robust_comprehensive
  output dir: visualization/output/velocity_slice_viz
  max samples: 2 per split
"""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


# Band indices for the 24-channel comprehensive dataset
RGB_IDX = (2, 1, 0)  # B3, B2, B1
V_IDX = 10
VX_IDX = 11
VY_IDX = 12
VMASK_IDX = 13


def to_rgb(arr: np.ndarray) -> np.ndarray:
    """Extract RGB and scale to [0,1]."""
    rgb = arr[:, :, RGB_IDX].astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def log1p_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log scale to [0,1] using max of the slice."""
    x_clip = np.maximum(x, 0.0)
    xmax = np.max(x_clip) + eps
    return np.log1p(x_clip) / np.log1p(xmax)


def symlog_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Symmetric log to [-1,1] using max abs of the slice."""
    max_abs = np.max(np.abs(x)) + eps
    return np.sign(x) * np.log1p(np.abs(x)) / np.log1p(max_abs)


def make_fig(tiff: np.ndarray, mask: np.ndarray, title: str) -> plt.Figure:
    rgb = to_rgb(tiff)
    v = tiff[:, :, V_IDX]
    vx = tiff[:, :, VX_IDX]
    vy = tiff[:, :, VY_IDX]
    vmask = tiff[:, :, VMASK_IDX]
    label_int = mask.squeeze()

    v_log = log1p_norm(v)
    vx_symlog = symlog_norm(vx)
    vy_symlog = symlog_norm(vy)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.ravel()

    axs[0].imshow(rgb)
    axs[0].set_title("RGB")

    # Overlay with labels (Debris=orange, Clean=blue)
    overlay = rgb.copy()
    # colors in [0,1]
    overlay[label_int == 2] = [200 / 255.0, 80 / 255.0, 0.0]
    overlay[label_int == 1] = [0.0, 120 / 255.0, 255 / 255.0]
    axs[1].imshow(overlay)
    axs[1].set_title("RGB + Labels")

    im2 = axs[2].imshow(v_log, cmap="inferno")
    axs[2].set_title("Velocity v (log)")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    im3 = axs[3].imshow(vx_symlog, cmap="coolwarm", vmin=-1, vmax=1)
    axs[3].set_title("Velocity vx (symlog)")
    fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    im4 = axs[4].imshow(vy_symlog, cmap="coolwarm", vmin=-1, vmax=1)
    axs[4].set_title("Velocity vy (symlog)")
    fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)

    axs[5].imshow(vmask, cmap="gray")
    axs[5].set_title("Velocity mask")

    for ax in axs:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def process_split(split_dir: Path, out_dir: Path, max_samples: int) -> int:
    tiff_files = sorted(split_dir.glob("tiff_*.npy"))
    if not tiff_files:
        return 0

    count = 0
    for tiff_path in tiff_files:
        if count >= max_samples:
            break
        mask_path = tiff_path.parent / tiff_path.name.replace("tiff", "mask")
        if not mask_path.exists():
            continue
        tiff = np.load(tiff_path)
        mask = np.load(mask_path)
        fig = make_fig(tiff, mask, title=f"{split_dir.name}/{tiff_path.name}")
        out_path = out_dir / f"{split_dir.name}_{tiff_path.stem}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Visualize velocity channels for dataset slices.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/devj/local-debian/datasets/HKH/gen_robust_comprehensive"),
        help="Path to processed dataset root (contains train/val/test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization/output/velocity_slice_viz"),
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2,
        help="Max slices per split to visualize.",
    )
    args = parser.parse_args()

    splits = ["train", "val", "test"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in splits:
        split_dir = args.data_root / split
        if not split_dir.exists():
            continue
        count = process_split(split_dir, args.output_dir, args.max_samples)
        total += count
        print(f"{split}: saved {count} figures")

    print(f"Done. Saved {total} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
