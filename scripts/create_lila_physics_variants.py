#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from glacier_mapping.data.data import get_no_normalize_channel_names
from glacier_mapping.data.physics import compute_phys_v4
from glacier_mapping.data.slice import IGNORE_LABEL

VARIANT_BANDS = {
    "flowacc": ["flow_accumulation"],
    "full_physics": ["flow_accumulation", "tpi", "roughness", "plan_curvature"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LILA dataset variants with terrain physics channels derived from LILA elevation."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/home/devj/local-arch/data/HKH/lila_released_v1"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/devj/local-arch/data/HKH"),
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_BANDS),
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output dataset if it already exists.",
    )
    parser.add_argument(
        "--flow-res",
        default="64",
        help="Flow source stride passed to compute_phys_v4. Use 'full' for every pixel.",
    )
    parser.add_argument(
        "--flow-scale",
        type=float,
        default=0.3,
        help="Downscale factor for flow accumulation computation.",
    )
    return parser.parse_args()


def load_band_names(dataset_dir: Path) -> list[str]:
    path = dataset_dir / "band_metadata.json"
    with path.open() as f:
        meta = json.load(f)
    return list(meta["band_names"])


def copy_non_arrays(source_dir: Path, out_dir: Path) -> None:
    for name in ["dataset_statistics.json", "band_metadata.json"]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def compute_stats(x: np.ndarray, y: np.ndarray, band_names: list[str]) -> np.ndarray:
    channel_count = x.shape[1]
    means = np.zeros(channel_count, dtype=np.float64)
    stds = np.ones(channel_count, dtype=np.float64)
    mins = np.zeros(channel_count, dtype=np.float64)
    maxs = np.ones(channel_count, dtype=np.float64)
    valid = y != IGNORE_LABEL
    no_norm = get_no_normalize_channel_names()

    for idx, band_name in enumerate(band_names):
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


def parse_flow_res(text: str) -> int | str:
    return "full" if text == "full" else int(text)


def write_split(
    source_split: Path,
    out_split: Path,
    source_band_names: list[str],
    added_band_names: list[str],
    flow_res: int | str,
    flow_scale: float,
) -> dict[str, Any]:
    source_x = np.load(source_split / "X.npy", mmap_mode="r")
    source_y = np.load(source_split / "y.npy", mmap_mode="r")
    with (source_split / "manifest.json").open() as f:
        source_manifest = json.load(f)

    elevation_idx = source_band_names.index("elevation")
    n, c, h, w = source_x.shape
    out_c = c + len(added_band_names)

    out_split.mkdir(parents=True, exist_ok=True)
    out_x = np.lib.format.open_memmap(
        out_split / "X.npy", mode="w+", dtype=np.float32, shape=(n, out_c, h, w)
    )
    out_y = np.lib.format.open_memmap(
        out_split / "y.npy", mode="w+", dtype=np.uint8, shape=source_y.shape
    )

    phys_indices = {"flow_accumulation": 0, "tpi": 1, "roughness": 2, "plan_curvature": 3}
    select_phys = [phys_indices[name] for name in added_band_names]

    for idx in range(n):
        out_x[idx, :c] = source_x[idx]
        out_y[idx] = source_y[idx]
        elevation = np.asarray(source_x[idx, elevation_idx], dtype=np.float32)
        elevation = np.nan_to_num(elevation, nan=0.0, posinf=0.0, neginf=0.0)
        phys = compute_phys_v4(elevation, res=flow_res, scale=flow_scale)
        for out_offset, phys_idx in enumerate(select_phys):
            out_x[idx, c + out_offset] = phys[:, :, phys_idx]

        if (idx + 1) % 25 == 0 or idx + 1 == n:
            print(f"  {source_split.name}: {idx + 1}/{n}")

    out_x.flush()
    out_y.flush()

    manifest = dict(source_manifest)
    manifest.update(
        {
            "format": f"{source_manifest.get('format', 'lila_released_v1')}+terrain_physics",
            "shape": [n, out_c, h, w],
            "source_dataset": str(source_split.parent),
            "added_band_names": added_band_names,
            "x": "X.npy",
            "y": "y.npy",
            "dtype": {"x": "float32", "y": "uint8"},
        }
    )
    with (out_split / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    added_band_names = VARIANT_BANDS[args.variant]
    out_dir = args.out_root / f"{source_dir.name}_{args.variant}"

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output exists: {out_dir}. Pass --overwrite to replace.")
    out_dir.mkdir(parents=True)

    source_band_names = load_band_names(source_dir)
    out_band_names = source_band_names + added_band_names
    flow_res = parse_flow_res(args.flow_res)

    manifests: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        manifests[split] = write_split(
            source_dir / split,
            out_dir / split,
            source_band_names,
            added_band_names,
            flow_res,
            args.flow_scale,
        )

    train_x = np.load(out_dir / "train" / "X.npy", mmap_mode="r")
    train_y = np.load(out_dir / "train" / "y.npy", mmap_mode="r")
    norm = compute_stats(train_x, train_y, out_band_names)
    for split in ["train", "val", "test"]:
        np.save(out_dir / f"normalize_{split}.npy", norm)

    band_metadata = {
        "band_names": out_band_names,
        "num_bands": len(out_band_names),
        "source_dataset": str(source_dir),
        "added_band_names": added_band_names,
        "notes": {
            "terrain_physics": "Derived from released LILA elevation channel per 512x512 patch using glacier_mapping.data.physics.compute_phys_v4.",
            "flow_res": args.flow_res,
            "flow_scale": args.flow_scale,
        },
    }
    with (out_dir / "band_metadata.json").open("w") as f:
        json.dump(band_metadata, f, indent=2)

    dataset_statistics = {"splits": manifests, "band_names": out_band_names}
    with (out_dir / "dataset_statistics.json").open("w") as f:
        json.dump(dataset_statistics, f, indent=2)

    print(f"Created {args.variant} dataset: {out_dir}")


if __name__ == "__main__":
    main()
