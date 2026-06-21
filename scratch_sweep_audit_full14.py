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
OUT_DIR = Path("/tmp/sweep_audit_full14")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "old": OLD_DIR,
    "audit8": REBUILD_DIR / "Landsat7_C02_T1",
    "A": REBUILD_DIR / "sweep_A",
    "B": REBUILD_DIR / "sweep_B",
    "C": REBUILD_DIR / "sweep_C",
}

TILES = [0, 1, 3, 24, 31, 46, 67, 68, 70, 90, 96, 131, 132, 135]


def finite_valid_mask(data: np.ndarray) -> np.ndarray:
    return np.isfinite(data).all(axis=0) & (data > 0).all(axis=0)


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


def render_comparison(tile: int, variant_images: dict[str, np.ndarray], ci_mask: np.ndarray, dci_mask: np.ndarray) -> None:
    variants = ["old", "audit8", "A", "B", "C"]
    fig, axes = plt.subplots(len(variants), 3, figsize=(12, 3 * len(variants)))
    for i, variant in enumerate(variants):
        img = variant_images.get(variant)
        if img is None:
            for j in range(3):
                axes[i, j].axis("off")
            continue
        rgb = stretch(np.stack([img[2], img[1], img[0]], axis=-1))
        false = stretch(np.stack([img[4], img[3], img[2]], axis=-1))
        overlay = rgb.copy()
        ci_edge = mask_outline(ci_mask > 0)
        dci_edge = mask_outline(dci_mask > 0)
        overlay[ci_edge] = [0, 1, 1]
        overlay[dci_edge] = [1, 0, 1]
        for ax, arr, title in [
            (axes[i, 0], rgb, f"{variant} RGB"),
            (axes[i, 1], false, f"{variant} False"),
            (axes[i, 2], overlay, f"{variant} Overlay"),
        ]:
            ax.imshow(arr)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"tile{tile}_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    labels = gpd.read_file(LABELS_PATH)
    summary: dict[str, dict] = {}

    for tile in TILES:
        ref_path = None
        for directory in VARIANTS.values():
            candidate = directory / f"image{tile}.tif"
            if candidate.exists():
                ref_path = candidate
                break
        if ref_path is None:
            continue

        with rasterio.open(ref_path) as ref:
            local_labels = labels.to_crs(ref.crs) if labels.crs != ref.crs else labels
            inter = local_labels[local_labels.intersects(box(*ref.bounds))]
            ci_polys = inter[inter["Glaciers"] == "Clean Ice"]
            dci_polys = inter[inter["Glaciers"] == "Debris covered"]
            ci_mask = rasterize(ci_polys.geometry, out_shape=ref.shape, transform=ref.transform, fill=0, default_value=1, dtype=np.uint8) if len(ci_polys) else np.zeros(ref.shape, np.uint8)
            dci_mask = rasterize(dci_polys.geometry, out_shape=ref.shape, transform=ref.transform, fill=0, default_value=1, dtype=np.uint8) if len(dci_polys) else np.zeros(ref.shape, np.uint8)
            ci_pixels = int(ci_mask.sum())
            dci_pixels = int(dci_mask.sum())

        tile_rows: dict[str, dict] = {}
        variant_images: dict[str, np.ndarray] = {}
        for variant, directory in VARIANTS.items():
            path = directory / f"image{tile}.tif"
            if not path.exists():
                tile_rows[variant] = {"missing": True}
                continue
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
                variant_images[variant] = data
                valid = finite_valid_mask(data)
                row = {
                    "dtype": src.dtypes[0],
                    "shape": list(src.shape),
                    "crs": str(src.crs),
                    "all_band_valid_pct": float(valid.mean() * 100),
                    "ci_valid_pct": float(((valid & (ci_mask > 0)).sum() / ci_pixels * 100) if ci_pixels else 0),
                    "dci_valid_pct": float(((valid & (dci_mask > 0)).sum() / dci_pixels * 100) if dci_pixels else 0),
                    "ci_pixels": ci_pixels,
                    "dci_pixels": dci_pixels,
                }
                if variant in {"A", "B", "C"}:
                    refl = data[[0, 1, 2, 3, 4, 7]]
                    refl_valid = refl[np.isfinite(refl) & (refl > 0)]
                    row.update(
                        {
                            "refl_frac_gt_1_pct": float(np.mean(refl_valid > 1) * 100) if refl_valid.size else 0.0,
                            "refl_frac_lt_0_pct": float(np.mean(refl_valid < 0) * 100) if refl_valid.size else 0.0,
                            "refl_frac_near_1_pct": float(np.mean(refl_valid > 0.95) * 100) if refl_valid.size else 0.0,
                        }
                    )
                tile_rows[variant] = row
        summary[f"tile{tile}"] = tile_rows
        render_comparison(tile, variant_images, ci_mask, dci_mask)

    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # aggregate csv-like json
    agg = {v: {"tiles": 0, "mean_valid": 0.0, "mean_ci": 0.0, "mean_dci": 0.0} for v in VARIANTS}
    for rows in summary.values():
        for variant in VARIANTS:
            row = rows.get(variant)
            if not row or row.get("missing"):
                continue
            agg[variant]["tiles"] += 1
            agg[variant]["mean_valid"] += row["all_band_valid_pct"]
            agg[variant]["mean_ci"] += row["ci_valid_pct"]
            agg[variant]["mean_dci"] += row["dci_valid_pct"]
    for variant in VARIANTS:
        n = agg[variant]["tiles"]
        if n:
            agg[variant]["mean_valid"] /= n
            agg[variant]["mean_ci"] /= n
            agg[variant]["mean_dci"] /= n
    with (OUT_DIR / "aggregate.json").open("w") as f:
        json.dump(agg, f, indent=2)

    print(f"Wrote {OUT_DIR / 'summary.json'}")
    print(f"Wrote {OUT_DIR / 'aggregate.json'}")
    print(f"Comparison sheets in {OUT_DIR}")


if __name__ == "__main__":
    main()
