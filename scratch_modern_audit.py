from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Agg")

# ── paths ───────────────────────────────────────────────────────────────────
OLD_TILES = Path("/home/devj/local-arch/data/HKH_raw/Landsat7_2005")
LABEL_SHP = Path("/home/devj/local-arch/data/HKH_raw/labels/HKH_CIDC_5basins_all.shp")
FISHNET_GEOJSON = Path("google_earth_scripts/hkh_fishnet.geojson")
TILE_CSV = Path("google_earth_scripts/tile_inventory.csv")
CANDIDATES_BASE = Path("/home/devj/local-arch/data/HKH_raw/rebuild/modern_audit")
OLD_RGB_PATH = Path("/tmp/modern_audit/old_rgb")  # pre-rendered if needed

TILE_IDS = [24, 31, 67, 96, 131, 132]
CANDIDATES = {
    "dynamic_median24": CANDIDATES_BASE / "HKH_modern_audit_dynamic_median24_Landsat7",
    "dynamic_median16": CANDIDATES_BASE / "HKH_modern_dynamic_median16_Landsat7",
    "dynamic_median12": CANDIDATES_BASE / "HKH_modern_dynamic_median12_Landsat7",
    "dynamic_median8": CANDIDATES_BASE / "HKH_modern_dynamic_median8_Landsat7",
    "dynamic_median6": CANDIDATES_BASE / "HKH_modern_median_6_Landsat7",
    "dynamic_median4": CANDIDATES_BASE / "HKH_modern_median_4_Landsat7",
    "dynamic_medoid16": CANDIDATES_BASE / "HKH_modern_medoid_16_Landsat7",
    "dynamic_medoid12": CANDIDATES_BASE / "HKH_modern_medoid_12_Landsat7",
    "dynamic_medoid8": CANDIDATES_BASE / "HKH_modern_medoid_8_Landsat7",
    "dynamic_best": CANDIDATES_BASE / "HKH_modern_audit_dynamic_best_Landsat7",
    "bibek41_median": CANDIDATES_BASE / "HKH_modern_audit_bibek41_median_Landsat7",
    "bibek41_best": CANDIDATES_BASE / "HKH_modern_audit_bibek41_best_Landsat7",
}
OUT = Path("/tmp/modern_audit")

LANDSAT_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID1", "B6_VCID2", "B7"]


# ── helpers ─────────────────────────────────────────────────────────────────
def stretch(arr: np.ndarray) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    p2, p98 = np.percentile(vals, [2, 98])
    if p98 <= p2:
        p98 = p2 + 1e-6
    out = np.clip((arr - p2) / (p98 - p2), 0, 1).astype(np.float32)
    out[~np.isfinite(out)] = 0
    return out


def band(arr: np.ndarray, idx: int) -> np.ndarray:
    """Extract band from (bands, H, W) or (H, W, bands) layout."""
    if arr.ndim == 3 and arr.shape[0] <= 24:
        return arr[idx]
    return arr[:, :, idx]


def valid_pct(arr: np.ndarray) -> float:
    """Percent of pixels where B1 (index 0) is finite and > 0.
    Modern exports may have 10 bands (8 Landsat + audit), old has 8."""
    b1 = band(arr, 0)
    ok = np.isfinite(b1) & (b1 > 0)
    return float(ok.mean() * 100)


def mask_outline(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:-1, :] |= mask[1:, :] != mask[:-1, :]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return edge & mask.astype(bool)


def nir_cv(arr: np.ndarray, kernel: int = 5) -> float:
    """Local coefficient of variation of NIR (B4, index 3) over valid pixels."""
    from scipy.ndimage import uniform_filter

    nir = band(arr, 3).astype(np.float32)
    valid = np.isfinite(nir) & (nir > 0)
    if valid.sum() < 100:
        return np.nan

    x = np.where(valid, nir, 0.0)
    w = valid.astype(np.float32)

    sum_x = uniform_filter(x, size=kernel, mode="nearest") * (kernel * kernel)
    sum_x2 = uniform_filter(x * x, size=kernel, mode="nearest") * (kernel * kernel)
    count = uniform_filter(w, size=kernel, mode="nearest") * (kernel * kernel)

    mean = np.divide(sum_x, count, out=np.zeros_like(sum_x), where=count > 0)
    mean_x2 = np.divide(sum_x2, count, out=np.zeros_like(sum_x2), where=count > 0)
    var = np.maximum(mean_x2 - mean * mean, 0.0)
    std = np.sqrt(var)
    cv = np.divide(std, mean, out=np.zeros_like(std), where=mean > 0)

    # Use centers with enough support.
    support = count >= max(4, kernel * kernel * 0.5)
    usable = valid & support & np.isfinite(cv)
    if usable.sum() < 100:
        return np.nan
    return float(np.mean(cv[usable]))


# ── tile geometry / CRS ────────────────────────────────────────────────────
def load_crs(tile_idx: int) -> str:
    import csv
    with open(TILE_CSV) as f:
        for row in csv.DictReader(f):
            if int(row["tile_index"]) == tile_idx:
                return row["crs"]
    return "EPSG:4326"


def load_tile_bounds(tile_idx: int) -> tuple[float, float, float, float]:
    import csv
    with open(TILE_CSV) as f:
        for row in csv.DictReader(f):
            if int(row["tile_index"]) == tile_idx:
                return (
                    float(row["bounds_left"]),
                    float(row["bounds_bottom"]),
                    float(row["bounds_right"]),
                    float(row["bounds_top"]),
                )
    raise ValueError(f"Tile {tile_idx} not in inventory")


def rasterize_labels(tile_idx: int, crs_str: str, bounds: tuple[float, float, float, float],
                     res: float = 30.0) -> np.ndarray | None:
    """Rasterize HKH_CIDC labels clipped to tile bounds. Returns (H, W) int array:
    0=bg, 1=clean, 2=debris, 255=outside."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import box

    try:
        labels = gpd.read_file(str(LABEL_SHP), on_invalid='ignore')
    except Exception as e:
        print(f"  Warning: cannot load labels from {LABEL_SHP}: {e}")
        return None

    og_crs = labels.crs
    if og_crs is None or og_crs.to_string() != crs_str:
        labels = labels.to_crs(crs_str)

    xmin, ymin, xmax, ymax = bounds
    tile_poly = box(xmin, ymin, xmax, ymax)
    clipped = labels[labels.intersects(tile_poly)].copy()
    if clipped.empty:
        return np.zeros((int((ymax - ymin) / res), int((xmax - xmin) / res)), dtype=np.uint8)

    # Determine class from Clean_Ice / Debris columns
    shapes = []
    for _, row in clipped.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        cls_val = 1 if row.get("Clean_Ice", 0) > 0 else (2 if row.get("Debris", 0) > 0 else 0)
        if cls_val > 0:
            shapes.append((geom, cls_val))

    if not shapes:
        return np.zeros((int((ymax - ymin) / res), int((xmax - xmin) / res)), dtype=np.uint8)

    width = int((xmax - xmin) / res)
    height = int((ymax - ymin) / res)
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
    out = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return out


# ── load tile data ──────────────────────────────────────────────────────────
def load_old(tile_idx: int) -> np.ndarray | None:
    p = OLD_TILES / f"image{tile_idx}.tif"
    if not p.exists():
        return None
    import rasterio
    with rasterio.open(p) as src:
        d = src.read().astype(np.float32)
    return d  # (bands, H, W), 8 bands


def load_candidate(name: str, tile_idx: int) -> np.ndarray | None:
    cand_dir = CANDIDATES.get(name)
    if cand_dir is None or not cand_dir.exists():
        return None
    p = cand_dir / f"image{tile_idx}.tif"
    if not p.exists():
        return None
    import rasterio
    with rasterio.open(p) as src:
        d = src.read().astype(np.float32)
    return d  # (bands, H, W)


# ── render ──────────────────────────────────────────────────────────────────
def render_tile(tile_idx: int) -> None:
    print(f"\n=== Tile {tile_idx} ===")
    out_dir = OUT / f"tile{tile_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    old = load_old(tile_idx)
    candidates = {}
    for name in CANDIDATES:
        candidates[name] = load_candidate(name, tile_idx)

    labels = rasterize_labels(tile_idx, load_crs(tile_idx), load_tile_bounds(tile_idx))
    label_valid = labels is not None

    # Stats table
    def audit_stats(arr):
        s = {
            "valid_pct": valid_pct(arr),
            "nir_cv": nir_cv(arr),
        }
        if arr.shape[0] >= 9:
            vc = band(arr, 8)
            fv = np.isfinite(vc)
            if fv.sum():
                s["valid_obs_mean"] = float(vc[fv].mean())
                s["valid_obs_max"] = float(vc[fv].max())
        if arr.shape[0] >= 10:
            ds = band(arr, 9)
            fds = np.isfinite(ds) & (band(arr, 8) > 0)
            if fds.sum():
                s["date_spread_mean"] = float(ds[fds].mean())
                s["date_spread_max"] = float(ds[fds].max())
        return s

    stats = {}
    if old is not None:
        stats["old"] = audit_stats(old)
    for name, arr in candidates.items():
        if arr is not None:
            stats[name] = audit_stats(arr)

    with (out_dir / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    # Determine source names and arrays for grid.
    sources = [("old", old)] + [(n, candidates[n]) for n in CANDIDATES if candidates[n] is not None]
    n = len(sources)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    for r, (label, arr) in enumerate(sources):
        rgb = stretch(np.stack([band(arr, 2), band(arr, 1), band(arr, 0)], axis=-1))
        false = stretch(np.stack([band(arr, 4), band(arr, 3), band(arr, 2)], axis=-1))

        overlay = rgb.copy()
        if label_valid:
            overlay[mask_outline(labels == 1)] = [0, 1, 1]
            overlay[mask_outline(labels == 2)] = [1, 0, 1]

        panels = [
            (rgb, "RGB (2/1/0)"),
            (false, "False (4/3/2)"),
            (overlay, "Labels cyan=CI magenta=DCI" if label_valid else "No labels"),
        ]
        s = stats.get(label, {})
        aux = []
        if "valid_obs_mean" in s:
            aux.append(f"obs={s['valid_obs_mean']:.1f}")
        if "date_spread_mean" in s:
            aux.append(f"spread={s['date_spread_mean']:.0f}d")
        aux_str = " | ".join(aux)
        info = f"{label}\nvalid={s.get('valid_pct', 0):.1f}% cv={s.get('nir_cv', 0):.4f}"
        if aux_str:
            info += f"\n{aux_str}"

        for c, (panel_arr, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel_arr)
            ax.set_title(title if r == 0 else "", fontsize=9)
            if c == 0:
                ax.set_ylabel(info, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(pad=1.0)
    plt.savefig(out_dir / "comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'comparison.png'}")


def render_aggregate() -> None:
    """Single big sheet: one row per tile, columns for old and each candidate."""
    sys_sources = [("old", None)] + [(n, None) for n in CANDIDATES]
    n_src = len(sys_sources)
    n_tiles = len(TILE_IDS)

    fig, axes = plt.subplots(n_tiles, n_src * 2, figsize=(6 * n_src, 3.2 * n_tiles))

    for t, tile_idx in enumerate(TILE_IDS):
        old = load_old(tile_idx)
        candidates = {n: load_candidate(n, tile_idx) for n in CANDIDATES}
        tiles_arr = [("old", old)] + [(n, candidates[n]) for n in CANDIDATES]

        for c, (label, arr) in enumerate(tiles_arr):
            if arr is None:
                continue
            rgb = stretch(np.stack([band(arr, 2), band(arr, 1), band(arr, 0)], axis=-1))
            false = stretch(np.stack([band(arr, 4), band(arr, 3), band(arr, 2)], axis=-1))
            axes[t, c * 2].imshow(rgb)
            axes[t, c * 2].set_xticks([])
            axes[t, c * 2].set_yticks([])
            axes[t, c * 2 + 1].imshow(false)
            axes[t, c * 2 + 1].set_xticks([])
            axes[t, c * 2 + 1].set_yticks([])
            if t == 0:
                axes[t, c * 2].set_title(f"{label} RGB", fontsize=8)
                axes[t, c * 2 + 1].set_title(f"{label} False", fontsize=8)
        axes[t, 0].set_ylabel(f"Tile {tile_idx}", fontsize=9)

    plt.tight_layout(pad=0.5)
    plt.savefig(OUT / "aggregate_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / 'aggregate_comparison.png'}")


if __name__ == "__main__":
    import sys

    OUT.mkdir(parents=True, exist_ok=True)

    if "--aggregate" in sys.argv:
        render_aggregate()
    else:
        for tid in TILE_IDS:
            render_tile(tid)

    # Global summary
    rows = []
    for tid in TILE_IDS:
        sp = OUT / f"tile{tid}" / "stats.json"
        if sp.exists():
            rows.append(json.load(open(sp)))
    with (OUT / "summary.json").open("w") as f:
        json.dump({"tiles": TILE_IDS, "per_tile": rows}, f, indent=2)
    print(f"\nSummary: {OUT / 'summary.json'}")
