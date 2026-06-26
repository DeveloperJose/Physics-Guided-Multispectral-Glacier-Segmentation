#!/usr/bin/env python3
"""Metric-based simulated-gap evaluation for local NSPI experiments.

Takes a raw target tile, creates an artificial gap mask by shifting the real
SLC-off mask into target-valid territory, then compares reconstruction methods
against the known true target values.

Initial supported methods:
- donor_copy
- nspi_single
- nspi_multi_paper

Focus: metric comparison, not visual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from run_hkh_nspi_local import _nspi_single, compute_similarity_threshold, load_stack

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B7"]


def shifted_mask(base_missing: np.ndarray, valid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = base_missing.shape
    out = np.zeros_like(base_missing)
    y1_src = max(0, -dy)
    y2_src = min(h, h - dy)
    x1_src = max(0, -dx)
    x2_src = min(w, w - dx)
    y1_dst = max(0, dy)
    y2_dst = min(h, h + dy)
    x1_dst = max(0, dx)
    x2_dst = min(w, w + dx)
    out[y1_dst:y2_dst, x1_dst:x2_dst] = base_missing[y1_src:y2_src, x1_src:x2_src]
    return out & valid


def donor_copy(target: np.ndarray, donor: np.ndarray, sim_missing: np.ndarray, donor_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = target.copy()
    q = np.zeros(target.shape[1:], dtype=np.uint8)
    fill = sim_missing & donor_valid
    out[:, fill] = donor[:, fill]
    q[fill] = 1
    q[sim_missing & ~donor_valid] = 5
    return out, q


def nspi_single_method(
    target: np.ndarray,
    target_valid_sim: np.ndarray,
    donor: np.ndarray,
    donor_valid: np.ndarray,
    num_class: int,
    min_similar: int,
    max_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    th = compute_similarity_threshold(donor, donor_valid, num_class)
    return _nspi_single(target, donor, target_valid_sim, donor_valid, th, min_similar, max_window, -1.0, 2.0)


def nspi_multi_paper(
    target: np.ndarray,
    target_valid_sim: np.ndarray,
    donors: list[np.ndarray],
    donor_valids: list[np.ndarray],
    num_class: int,
    min_similar: int,
    max_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    filled = target.copy()
    quality = np.zeros(target.shape[1:], dtype=np.uint8)
    remaining = ~target_valid_sim
    for idx, (donor, dvalid) in enumerate(zip(donors, donor_valids), start=1):
        if not np.any(remaining & dvalid):
            continue
        th = compute_similarity_threshold(donor, dvalid, num_class)
        donor_out, donor_q = _nspi_single(target, donor, target_valid_sim, dvalid, th, min_similar, max_window, -1.0, 2.0)
        usable = remaining & np.isin(donor_q, np.array([1, 2, 3], dtype=np.uint8))
        filled[:, usable] = donor_out[:, usable]
        quality[usable] = donor_q[usable] + 10 * idx
        remaining[usable] = False
    quality[remaining] = 5
    filled[:, remaining] = np.nan
    return filled, quality


def metrics(pred: np.ndarray, true: np.ndarray, eval_mask: np.ndarray) -> dict:
    out = {}
    dif = pred[:, eval_mask] - true[:, eval_mask]
    out["n_eval"] = int(eval_mask.sum())
    out["mae_mean"] = float(np.nanmean(np.abs(dif)))
    out["rmse_mean"] = float(np.sqrt(np.nanmean(dif ** 2)))
    out["bias_mean"] = float(np.nanmean(dif))
    out["per_band"] = {}
    for i, b in enumerate(BAND_NAMES):
        bd = dif[i]
        out["per_band"][b] = {
            "mae": float(np.nanmean(np.abs(bd))),
            "rmse": float(np.sqrt(np.nanmean(bd ** 2))),
            "bias": float(np.nanmean(bd)),
        }
    # derived metrics
    t_ndsi = (true[1] - true[4]) / (true[1] + true[4] + 1e-6)
    p_ndsi = (pred[1] - pred[4]) / (pred[1] + pred[4] + 1e-6)
    t_brt = true.mean(axis=0)
    p_brt = pred.mean(axis=0)
    out["ndsi_mae"] = float(np.nanmean(np.abs(p_ndsi[eval_mask] - t_ndsi[eval_mask])))
    out["brightness_mae"] = float(np.nanmean(np.abs(p_brt[eval_mask] - t_brt[eval_mask])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--donors", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--shift-dx", type=int, default=220)
    parser.add_argument("--shift-dy", type=int, default=-120)
    parser.add_argument("--min-similar", type=int, default=20)
    parser.add_argument("--single-max-window", type=int, default=8)
    parser.add_argument("--multi-max-window", type=int, default=15)
    parser.add_argument("--num-class", type=int, default=4)
    args = parser.parse_args()

    target, target_valid, _ = load_stack(args.target)
    donors, donor_valids = [], []
    for p in args.donors:
        d, dv, _ = load_stack(p)
        donors.append(d)
        donor_valids.append(dv)

    real_missing = ~target_valid
    sim_gap = shifted_mask(real_missing, target_valid, args.shift_dx, args.shift_dy)
    target_valid_sim = target_valid & ~sim_gap
    target_sim = target.copy()
    target_sim[:, sim_gap] = np.nan

    results = {
        "target": str(args.target),
        "donors": [str(p) for p in args.donors],
        "sim_gap_pixels": int(sim_gap.sum()),
        "shift": {"dx": args.shift_dx, "dy": args.shift_dy},
        "methods": {},
    }

    # donor_copy baseline using first donor
    pred, q = donor_copy(target_sim, donors[0], sim_gap, donor_valids[0])
    results["methods"]["donor_copy_first"] = metrics(pred, target, sim_gap)
    results["methods"]["donor_copy_first"]["quality_counts"] = {
        int(k): int(v) for k, v in zip(*np.unique(q, return_counts=True))
    }

    # single-donor NSPI for each donor
    for p, d, dv in zip(args.donors, donors, donor_valids):
        name = f"nspi_single_{p.stem.split('_')[-1]}"
        pred, q = nspi_single_method(
            target_sim,
            target_valid_sim,
            d,
            dv,
            args.num_class,
            args.min_similar,
            args.single_max_window,
        )
        results["methods"][name] = metrics(pred, target, sim_gap)
        results["methods"][name]["quality_counts"] = {
            int(k): int(v) for k, v in zip(*np.unique(q, return_counts=True))
        }

    # multi paper
    pred, q = nspi_multi_paper(
        target_sim,
        target_valid_sim,
        donors,
        donor_valids,
        args.num_class,
        args.min_similar,
        args.multi_max_window,
    )
    results["methods"]["nspi_multi_paper"] = metrics(pred, target, sim_gap)
    results["methods"]["nspi_multi_paper"]["quality_counts"] = {
        int(k): int(v) for k, v in zip(*np.unique(q, return_counts=True))
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
