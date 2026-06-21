from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Agg")

DATASET = Path("/home/devj/local-arch/data/HKH/lila_released_v1")
OUT = Path("/tmp/lila_visual_audit")
OUT.mkdir(parents=True, exist_ok=True)

BANDS = [
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


def stretch(arr: np.ndarray) -> np.ndarray:
    # Released LILA split arrays are already normalized/z-scored. Negative values
    # are valid reflectance values after normalization, not missing data.
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    p2, p98 = np.percentile(vals, [2, 98])
    if p98 <= p2:
        p98 = p2 + 1e-6
    out = np.clip((arr - p2) / (p98 - p2), 0, 1).astype(np.float32)
    out[~np.isfinite(out)] = 0
    return out


def mask_outline(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:-1, :] |= mask[1:, :] != mask[:-1, :]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return edge & mask.astype(bool)


def sample_stats(split: str, idx: int, x: np.ndarray, y: np.ndarray, record: dict) -> dict:
    sl = np.asarray(x[idx], dtype=np.float32)
    yy = np.asarray(y[idx])
    # All-band valid is not useful for released LILA because thermal/BQA-like bands
    # are often zero; track reflective-band validity separately.
    all_band_valid = np.isfinite(sl).all(axis=0)
    reflective = sl[[0, 1, 2, 3, 4, 7]]
    reflective_valid = np.isfinite(reflective).all(axis=0)
    b1_gap = ~np.isfinite(sl[0])
    return {
        "split": split,
        "index": idx,
        "img_file": record["img_file"],
        "all_band_valid_pct": float(all_band_valid.mean() * 100),
        "reflective_valid_pct": float(reflective_valid.mean() * 100),
        "b1_gap_pct": float(b1_gap.mean() * 100),
        "ci_pct": float((yy == 1).mean() * 100),
        "dci_pct": float((yy == 2).mean() * 100),
        "bg_pct": float((yy == 0).mean() * 100),
        "ignore_pct": float((yy == 255).mean() * 100),
    }


def render_sheet(rows: list[dict], title: str, out_path: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = axes[None, :]
    for row_i, row in enumerate(rows):
        split = row["split"]
        idx = row["index"]
        x = np.load(DATASET / split / "X.npy", mmap_mode="r")
        y = np.load(DATASET / split / "y.npy", mmap_mode="r")
        sl = np.asarray(x[idx], dtype=np.float32)
        yy = np.asarray(y[idx])

        rgb_210 = stretch(np.stack([sl[2], sl[1], sl[0]], axis=-1))
        rgb_012 = stretch(np.stack([sl[0], sl[1], sl[2]], axis=-1))
        false = stretch(np.stack([sl[4], sl[3], sl[2]], axis=-1))
        overlay = rgb_210.copy()
        ci_edge = mask_outline(yy == 1)
        dci_edge = mask_outline(yy == 2)
        overlay[ci_edge] = [0, 1, 1]
        overlay[dci_edge] = [1, 0, 1]
        gap = (~np.isfinite(sl[0]) | (sl[0] <= 0)).astype(np.float32)

        panels = [
            (rgb_210, "RGB? 2/1/0 (B3/B2/B1 if Landsat order)"),
            (rgb_012, "RGB? 0/1/2 (if LILA stored RGB order)"),
            (false, "False 4/3/2"),
            (overlay, "Labels on 2/1/0; cyan=CI magenta=DCI"),
        ]
        label = (
            f"{split}[{idx}] {row['img_file']}\n"
            f"CI={row['ci_pct']:.2f}% DCI={row['dci_pct']:.2f}% "
            f"finite_refl={row['reflective_valid_pct']:.1f}% finite_B1_missing={row['b1_gap_pct']:.1f}%"
        )
        for col, (arr, panel_title) in enumerate(panels):
            ax = axes[row_i, col]
            ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
            ax.set_title(panel_title if row_i == 0 else "", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    # Avoid suptitle overlap with per-panel titles and long row labels.
    plt.tight_layout(pad=1.2)
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    all_rows: list[dict] = []
    for split in ["train", "val", "test"]:
        x = np.load(DATASET / split / "X.npy", mmap_mode="r")
        y = np.load(DATASET / split / "y.npy", mmap_mode="r")
        manifest = json.load(open(DATASET / split / "manifest.json"))
        for idx, record in enumerate(manifest["records"]):
            all_rows.append(sample_stats(split, idx, x, y, record))

    # Audit sets: debris-rich, mixed CI/DCI, high gap, and random-ish representative.
    debris = sorted(all_rows, key=lambda r: r["dci_pct"], reverse=True)[:8]
    mixed = sorted(
        [r for r in all_rows if r["ci_pct"] > 1 and r["dci_pct"] > 0.2],
        key=lambda r: min(r["ci_pct"], r["dci_pct"]),
        reverse=True,
    )[:8]
    high_gap = sorted(all_rows, key=lambda r: r["b1_gap_pct"], reverse=True)[:8]
    high_ci = sorted(all_rows, key=lambda r: r["ci_pct"], reverse=True)[:8]

    render_sheet(debris, "Released LILA audit: debris-rich patches", OUT / "debris_rich.png")
    render_sheet(mixed, "Released LILA audit: mixed CI/DCI patches", OUT / "mixed_ci_dci.png")
    render_sheet(high_gap, "Released LILA audit: highest B1 gap patches", OUT / "high_gap.png")
    render_sheet(high_ci, "Released LILA audit: clean-ice-rich patches", OUT / "clean_ice_rich.png")

    summary = {
        "dataset": str(DATASET),
        "num_samples": len(all_rows),
        "mean_all_band_valid_pct": float(np.mean([r["all_band_valid_pct"] for r in all_rows])),
        "mean_reflective_valid_pct": float(np.mean([r["reflective_valid_pct"] for r in all_rows])),
        "mean_b1_gap_pct": float(np.mean([r["b1_gap_pct"] for r in all_rows])),
        "mean_ci_pct": float(np.mean([r["ci_pct"] for r in all_rows])),
        "mean_dci_pct": float(np.mean([r["dci_pct"] for r in all_rows])),
        "top_debris": debris,
        "top_mixed": mixed,
        "top_gap": high_gap,
        "top_clean_ice": high_ci,
    }
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote LILA visual audit to {OUT}")
    print(json.dumps({k: summary[k] for k in ["num_samples", "mean_all_band_valid_pct", "mean_reflective_valid_pct", "mean_b1_gap_pct", "mean_ci_pct", "mean_dci_pct"]}, indent=2))


if __name__ == "__main__":
    main()
