from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio.features import rasterize
from shapely.geometry import box

matplotlib.use("Agg")

BASE = Path("/home/devj/local-arch/data/HKH_raw")
OLD_DIR = BASE / "Landsat7_2005"
REBUILD_DIR = BASE / "rebuild"
LABELS_PATH = BASE / "labels_fixed/HKH_CIDC_5basins_all.shp"
OUT_DIR = Path("/tmp/sweep_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "old": OLD_DIR,
    "audit8": REBUILD_DIR / "Landsat7_C02_T1",
    "A": REBUILD_DIR / "sweep_A",
    "B": REBUILD_DIR / "sweep_B",
    "C": REBUILD_DIR / "sweep_C",
}

TILES = [24, 31, 67, 96, 131]
REFLECTIVE_BANDS = [0, 1, 2, 3, 4, 7]


def finite_valid_mask(data: np.ndarray) -> np.ndarray:
    return np.isfinite(data).all(axis=0) & (data > 0).all(axis=0)


def per_band_stats(data: np.ndarray, variant: str) -> list[dict[str, float]]:
    out = []
    for b in range(data.shape[0]):
        band = data[b]
        valid = band[np.isfinite(band) & (band > 0)]
        if valid.size == 0:
            out.append({"valid_pct": 0.0})
            continue
        row = {
            "valid_pct": round(valid.size / band.size * 100, 3),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "p1": float(np.percentile(valid, 1)),
            "p5": float(np.percentile(valid, 5)),
            "p50": float(np.percentile(valid, 50)),
            "p95": float(np.percentile(valid, 95)),
            "p99": float(np.percentile(valid, 99)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
        }
        if variant in {"A", "B", "C"}:
            row.update(
                {
                    "frac_lt_0": float(np.mean(valid < 0)),
                    "frac_gt_1": float(np.mean(valid > 1)),
                    "frac_near_0": float(np.mean(valid < 0.02)),
                    "frac_near_1": float(np.mean(valid > 0.95)),
                }
            )
        out.append(row)
    return out


def stretch(rgb: np.ndarray) -> np.ndarray:
    vals = rgb[np.isfinite(rgb) & (rgb > 0)]
    if vals.size == 0:
        return np.zeros_like(rgb)
    p2, p98 = np.percentile(vals, [2, 98])
    if p98 <= p2:
        p98 = p2 + 1e-6
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    rgb[~np.isfinite(rgb)] = 0
    return rgb


def mask_outline(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:-1, :] |= mask[1:, :] != mask[:-1, :]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return edge & mask.astype(bool)


def render_tile(tile: int, variant: str, path: Path, ci_mask: np.ndarray, dci_mask: np.ndarray) -> None:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)

    if variant in {"A", "B", "C"}:
        rgb = np.stack([data[2], data[1], data[0]], axis=-1)
        false = np.stack([data[4], data[3], data[2]], axis=-1)
    else:
        rgb = np.stack([data[2], data[1], data[0]], axis=-1)
        false = np.stack([data[4], data[3], data[2]], axis=-1)

    rgb = stretch(rgb)
    false = stretch(false)

    overlay = rgb.copy()
    ci_edge = mask_outline(ci_mask > 0)
    dci_edge = mask_outline(dci_mask > 0)
    overlay[ci_edge] = [0, 1, 1]
    overlay[dci_edge] = [1, 0, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{variant} RGB")
    axes[1].imshow(false)
    axes[1].set_title(f"{variant} False color")
    axes[2].imshow(overlay)
    axes[2].set_title(f"{variant} RGB + labels")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"tile{tile}_{variant}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    labels = gpd.read_file(LABELS_PATH)
    summary: dict[str, dict] = {}

    for tile in TILES:
        ref_path = None
        for variant, directory in VARIANTS.items():
            candidate = directory / f"image{tile}.tif"
            if candidate.exists():
                ref_path = candidate
                break
        if ref_path is None:
            continue

        with rasterio.open(ref_path) as ref:
            ref_shape = ref.shape
            ref_transform = ref.transform
            ref_bounds = ref.bounds
            local_labels = labels.to_crs(ref.crs) if labels.crs != ref.crs else labels
            inter = local_labels[local_labels.intersects(box(*ref_bounds))]
            ci_polys = inter[inter["Glaciers"] == "Clean Ice"]
            dci_polys = inter[inter["Glaciers"] == "Debris covered"]
            ci_mask = (
                rasterize(ci_polys.geometry, out_shape=ref_shape, transform=ref_transform, fill=0, default_value=1, dtype=np.uint8)
                if len(ci_polys)
                else np.zeros(ref_shape, dtype=np.uint8)
            )
            dci_mask = (
                rasterize(dci_polys.geometry, out_shape=ref_shape, transform=ref_transform, fill=0, default_value=1, dtype=np.uint8)
                if len(dci_polys)
                else np.zeros(ref_shape, dtype=np.uint8)
            )

        tile_rows: dict[str, dict] = {}
        for variant, directory in VARIANTS.items():
            path = directory / f"image{tile}.tif"
            if not path.exists():
                tile_rows[variant] = {"missing": True}
                continue
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
                valid = finite_valid_mask(data)
                row = {
                    "dtype": src.dtypes[0],
                    "shape": list(src.shape),
                    "crs": str(src.crs),
                    "all_band_valid_pct": float(valid.mean() * 100),
                    "finite_all_pct": float(np.isfinite(data).all(axis=0).mean() * 100),
                    "ci_valid_pct": float(((valid & (ci_mask > 0)).sum() / ci_mask.sum() * 100) if ci_mask.sum() else 0),
                    "dci_valid_pct": float(((valid & (dci_mask > 0)).sum() / dci_mask.sum() * 100) if dci_mask.sum() else 0),
                    "per_band": per_band_stats(data, variant),
                }
                tile_rows[variant] = row
            render_tile(tile, variant, path, ci_mask, dci_mask)
        summary[f"tile{tile}"] = tile_rows

    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {OUT_DIR / 'summary.json'}")
    print(f"PNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()
