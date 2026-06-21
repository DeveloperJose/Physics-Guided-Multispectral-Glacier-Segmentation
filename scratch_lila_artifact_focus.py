from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Agg")

PROC = Path("/home/devj/local-arch/data/HKH/lila_released_v1")
RAW_SLICES = Path("/home/devj/local-arch/data/HKH_raw/LILA/glacier_data/slices")
SLICES_GEOJSON = Path("/home/devj/local-arch/data/HKH_raw/LILA/glacier_data/slices/slices.geojson")
OUT = Path("/tmp/lila_visual_audit/artifact_focus.png")

SAMPLES = [
    ("test", 8),
    ("test", 9),
    ("train", 56),
    ("train", 7),
    ("train", 0),
]


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


def mask_outline(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:-1, :] |= mask[1:, :] != mask[:-1, :]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return edge & mask.astype(bool)


def load_sources() -> dict[str, dict]:
    data = json.load(open(SLICES_GEOJSON))
    out = {}
    for feat in data["features"]:
        props = feat["properties"]
        out[os.path.basename(props["img_slice"])] = props
    return out


def render() -> None:
    sources = load_sources()
    rows = []
    for split, idx in SAMPLES:
        man = json.load(open(PROC / split / "manifest.json"))
        rec = man["records"][idx]
        rows.append((split, idx, rec, sources.get(rec["img_file"], {})))

    fig, axes = plt.subplots(len(rows), 5, figsize=(18, 3.2 * len(rows)))
    for r, (split, idx, rec, src_props) in enumerate(rows):
        proc_x = np.load(PROC / split / "X.npy", mmap_mode="r")
        proc_y = np.load(PROC / split / "y.npy", mmap_mode="r")
        px = np.asarray(proc_x[idx], dtype=np.float32)
        yy = np.asarray(proc_y[idx])

        raw_path = RAW_SLICES / rec["img_file"]
        raw = np.load(raw_path).astype(np.float32)
        # raw is HWC, processed is CHW
        if raw.ndim == 3 and raw.shape[-1] == px.shape[0]:
            raw_chw = np.transpose(raw, (2, 0, 1))
        else:
            raw_chw = px

        panels = [
            (stretch(np.stack([px[2], px[1], px[0]], axis=-1)), "proc 2/1/0"),
            (stretch(np.stack([px[0], px[1], px[2]], axis=-1)), "proc 0/1/2"),
            (stretch(np.stack([px[4], px[3], px[2]], axis=-1)), "proc false 4/3/2"),
            (stretch(np.stack([raw_chw[2], raw_chw[1], raw_chw[0]], axis=-1)), "raw 2/1/0"),
            (None, "overlay"),
        ]
        overlay = panels[0][0].copy()
        overlay[mask_outline(yy == 1)] = [0, 1, 1]
        overlay[mask_outline(yy == 2)] = [1, 0, 1]
        panels[-1] = (overlay, "labels on proc 2/1/0")

        label = (
            f"{split}[{idx}] {rec['img_file']}\n"
            f"src={os.path.basename(src_props.get('img_source', 'unknown'))}\n"
            f"CI={(yy==1).mean()*100:.1f}% DCI={(yy==2).mean()*100:.1f}%"
        )
        for c, (arr, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(arr)
            ax.set_title(title if r == 0 else "", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout(pad=1.1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary = []
    for split, idx, rec, src_props in rows:
        summary.append(
            {
                "split": split,
                "index": idx,
                "img_file": rec["img_file"],
                "mask_file": rec["mask_file"],
                "source_scene": os.path.basename(src_props.get("img_source", "")),
                "img_mean_raw_metadata": src_props.get("img_mean"),
                "mask_means": {k: v for k, v in src_props.items() if k.startswith("mask_mean")},
            }
        )
    with open(OUT.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    render()
