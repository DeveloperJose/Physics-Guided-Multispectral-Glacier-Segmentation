#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from glacier_mapping.data.data import get_no_normalize_channel_names
from glacier_mapping.data.slice import IGNORE_LABEL

LILA_BAND_NAMES = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6_VCID1",
    "B6_VCID2",
    "B7",
    "B8",
    "BQA",
    "NDVI",
    "NDSI",
    "NDWI",
    "elevation",
    "slope_deg",
]

SPLIT_MAP = {"train": "train", "dev": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert released LILA HKH splits into repo processed dataset format."
    )
    parser.add_argument(
        "--lila-root",
        type=Path,
        default=Path("/home/devj/local-arch/data/HKH_raw/LILA/glacier_data"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/devj/local-arch/data/HKH/lila_released_v1"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output dir first if it exists.",
    )
    return parser.parse_args()


def load_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    img_files = sorted(split_dir.glob("*_img_*.npy"))
    pairs: list[tuple[Path, Path]] = []
    for img in img_files:
        mask = Path(str(img).replace("_img_", "_mask_"))
        if not mask.exists():
            raise FileNotFoundError(f"Missing mask for {img}: {mask}")
        pairs.append((img, mask))
    return pairs


def build_y(mask_3ch: np.ndarray) -> np.ndarray:
    if mask_3ch.ndim != 3 or mask_3ch.shape[2] != 3:
        raise ValueError(f"Expected mask [H,W,3], got {mask_3ch.shape}")

    clean = mask_3ch[:, :, 0] > 0.5
    debris = mask_3ch[:, :, 1] > 0.5
    other = mask_3ch[:, :, 2] > 0.5

    y = np.zeros(mask_3ch.shape[:2], dtype=np.uint8)
    y[clean] = 1
    y[debris] = 2

    # Released LILA split masks are one-hot over [clean, debris, other].
    # In practice "other" covers the remaining valid pixels in the benchmark
    # patches, so keep them as background 0 rather than IGNORE_LABEL.
    uncovered = ~(clean | debris | other)
    y[uncovered] = IGNORE_LABEL
    return y


def compute_stats(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    channel_count = x.shape[1]
    means = np.zeros(channel_count, dtype=np.float64)
    stds = np.ones(channel_count, dtype=np.float64)
    mins = np.zeros(channel_count, dtype=np.float64)
    maxs = np.ones(channel_count, dtype=np.float64)

    valid = y != IGNORE_LABEL
    no_norm = get_no_normalize_channel_names()

    for idx, band_name in enumerate(LILA_BAND_NAMES):
        channel = x[:, idx, :, :]
        vals = channel[valid]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        means[idx] = vals.mean()
        stds[idx] = max(vals.std(), 1e-6)
        mins[idx] = vals.min()
        maxs[idx] = vals.max()
        if band_name in no_norm:
            means[idx] = 0.0
            stds[idx] = 1.0
    return np.asarray([means, stds, mins, maxs], dtype=np.float32)


def write_split(out_split: Path, pairs: list[tuple[Path, Path]]) -> dict:
    if not pairs:
        raise ValueError(f"No image/mask pairs for {out_split}")

    first_img = np.load(pairs[0][0], mmap_mode="r")
    if first_img.shape != (512, 512, len(LILA_BAND_NAMES)):
        raise ValueError(f"Unexpected LILA image shape: {first_img.shape}")

    n = len(pairs)
    h, w, c = first_img.shape
    x = np.lib.format.open_memmap(
        out_split / "X.npy", mode="w+", dtype=np.float32, shape=(n, c, h, w)
    )
    y = np.lib.format.open_memmap(
        out_split / "y.npy", mode="w+", dtype=np.uint8, shape=(n, h, w)
    )

    records = []
    for idx, (img_path, mask_path) in enumerate(pairs):
        img = np.load(img_path).astype(np.float32, copy=False)
        mask = np.load(mask_path)
        x[idx] = np.transpose(img, (2, 0, 1))
        y[idx] = build_y(mask)
        records.append(
            {
                "index": idx,
                "img_file": img_path.name,
                "mask_file": mask_path.name,
            }
        )
    x.flush()
    y.flush()

    manifest = {
        "format": "lila_released_v1",
        "layout": "NCHW",
        "normalized": False,
        "x": "X.npy",
        "y": "y.npy",
        "num_samples": n,
        "shape": [n, c, h, w],
        "label_shape": [n, h, w],
        "dtype": {"x": "float32", "y": "uint8"},
        "records": records,
    }
    with open(out_split / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    args = parse_args()
    lila_root = args.lila_root
    out_dir = args.out_dir

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    train_x = train_y = None
    for lila_split, out_split_name in SPLIT_MAP.items():
        src_split = lila_root / "splits" / lila_split
        out_split = out_dir / out_split_name
        out_split.mkdir(parents=True, exist_ok=True)
        pairs = load_pairs(src_split)
        manifests[out_split_name] = write_split(out_split, pairs)
        if out_split_name == "train":
            train_x = np.load(out_split / "X.npy", mmap_mode="r")
            train_y = np.load(out_split / "y.npy", mmap_mode="r")

    if train_x is None or train_y is None:
        raise RuntimeError("Train split missing after conversion")

    norm = compute_stats(train_x, train_y)
    np.save(out_dir / "normalize_train.npy", norm)
    np.save(out_dir / "normalize_val.npy", norm)
    np.save(out_dir / "normalize_test.npy", norm)

    band_metadata = {
        "band_names": LILA_BAND_NAMES,
        "num_bands": len(LILA_BAND_NAMES),
        "source_dataset": "LILA released HKH glacier mapping",
        "source_root": str(lila_root),
        "notes": {
            "labels": "Derived from released LILA 3-channel masks interpreted as [clean, debris, other/background].",
            "ignore_label": IGNORE_LABEL,
            "normalization": "Train stats computed over valid pixels only.",
        },
    }
    with open(out_dir / "band_metadata.json", "w") as f:
        json.dump(band_metadata, f, indent=2)

    dataset_statistics = {
        "splits": manifests,
        "band_names": LILA_BAND_NAMES,
    }
    with open(out_dir / "dataset_statistics.json", "w") as f:
        json.dump(dataset_statistics, f, indent=2)

    print(f"Converted LILA dataset to {out_dir}")
    for split, manifest in manifests.items():
        print(f"  {split}: {manifest['num_samples']} samples")


if __name__ == "__main__":
    main()
