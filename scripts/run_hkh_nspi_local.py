#!/usr/bin/env python3
"""Local single-donor NSPI core aligned to original paper + IDL code.

Primary references used for logic auditing:
- Paper: Chen et al. 2011, Remote Sensing of Environment 115, 1053-1064
  (`analysis/nspi_reference/nspi_paper.txt`)
- IDL code: `analysis/nspi_reference/nspi_update_20100824/NSPI update 20100824/FILLGAP_SINGLE_V2.pro`
- IDL readme: `analysis/nspi_reference/nspi_update_20100824/NSPI update 20100824/readme_FILLGAP_SINGLE_V2.txt`

Implemented core logic:
- use donor/reference spectrum D(p) for missing target pixel p
- select nearby common-valid pixels q by RMSE(D(q), D(p)) threshold
- spatial prediction: weighted mean T(q)
- temporal prediction: D(p) + weighted mean(T(q) - D(q))
- blend predictions by original R1/R2 weights

Input raw stacks are exported by scripts/export_hkh_nspi.py:
B1,B2,B3,B4,B5,B7,QA_PIXEL,QA_RADSAT,data_present,clear_valid,slc_gap.
Older 7-band debug stacks with valid_optical as band 7 are still supported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from numba import njit, prange

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B7"]


@njit(parallel=True, cache=True)
def _nspi_single(
    target: np.ndarray,
    donor: np.ndarray,
    target_valid: np.ndarray,
    donor_valid: np.ndarray,
    similar_th: float,
    min_similar: int,
    max_window: int,
    dn_min: float,
    dn_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    bands, height, width = target.shape
    out = target.copy()
    quality = np.zeros((height, width), dtype=np.uint8)
    # 0 = original target, 1 = >=M similar, 2 = <M similar at max window,
    # 3 = no similar; mean temporal-delta fallback, 5 = unfilled.
    max_candidates = (2 * max_window + 1) * (2 * max_window + 1)
    init_extent = int(np.ceil(0.5 * (np.sqrt(min_similar) - 1.0)))
    if init_extent < 1:
        init_extent = 1

    for y in prange(height):
        rmse = np.empty(max_candidates, dtype=np.float32)
        rmse12 = np.empty(max_candidates, dtype=np.float32)
        dist = np.empty(max_candidates, dtype=np.float32)
        cand_y = np.empty(max_candidates, dtype=np.int32)
        cand_x = np.empty(max_candidates, dtype=np.int32)
        for x in range(width):
            if target_valid[y, x]:
                quality[y, x] = 0
                continue
            if not donor_valid[y, x]:
                quality[y, x] = 5
                for b in range(bands):
                    out[b, y, x] = np.nan
                continue

            filled = False
            extent = init_extent
            while extent <= max_window and not filled:
                y1 = max(0, y - extent)
                y2 = min(height - 1, y + extent)
                x1 = max(0, x - extent)
                x2 = min(width - 1, x + extent)

                c_common = 0
                for yy in range(y1, y2 + 1):
                    for xx in range(x1, x2 + 1):
                        if target_valid[yy, xx] and donor_valid[yy, xx]:
                            # common-valid candidates; target pixel itself cannot be valid here
                            diff_sq = 0.0
                            diff_sq2 = 0.0
                            good = True
                            for b in range(bands):
                                dv = donor[b, yy, xx]
                                dp = donor[b, y, x]
                                tv = target[b, yy, xx]
                                if not np.isfinite(dv) or not np.isfinite(dp) or not np.isfinite(tv):
                                    good = False
                                    break
                                d = dv - dp
                                diff_sq += d * d
                                d12 = dv - tv
                                diff_sq2 += d12 * d12
                            if good:
                                rmse[c_common] = np.sqrt(diff_sq / bands) + 0.0001
                                rmse12[c_common] = np.sqrt(diff_sq2 / bands) + 0.0001
                                dy = float(y - yy)
                                dx = float(x - xx)
                                dd = np.sqrt(dy * dy + dx * dx)
                                if dd < 1e-6:
                                    dd = 1e-6
                                dist[c_common] = dd
                                cand_y[c_common] = yy
                                cand_x[c_common] = xx
                                c_common += 1

                if c_common > min_similar:
                    c_similar = 0
                    # mark similar by moving selected candidates to front of arrays
                    for i in range(c_common):
                        if rmse[i] <= similar_th:
                            if c_similar != i:
                                rmse[c_similar] = rmse[i]
                                rmse12[c_similar] = rmse12[i]
                                dist[c_similar] = dist[i]
                                cand_y[c_similar] = cand_y[i]
                                cand_x[c_similar] = cand_x[i]
                            c_similar += 1

                    if c_similar < min_similar and extent < max_window:
                        extent += 1
                        continue

                    if c_similar > 0:
                        use_n = c_similar
                        qcode = 1 if c_similar >= min_similar else 2
                        weight_sum = 0.0
                        r1 = 0.0
                        r2 = 0.0
                        for i in range(use_n):
                            cd = rmse[i] * dist[i]
                            if cd < 1e-6:
                                cd = 1e-6
                            w = 1.0 / cd
                            dist[i] = w  # reuse as weight
                            weight_sum += w
                            r1 += rmse[i]
                            r2 += rmse12[i]
                        if weight_sum <= 0:
                            quality[y, x] = 5
                            break
                        for i in range(use_n):
                            dist[i] /= weight_sum
                        r1 /= use_n
                        r2 /= use_n
                        wt1 = r2 / (r1 + r2)
                        wt2 = r1 / (r1 + r2)

                        for b in range(bands):
                            predict1 = 0.0
                            delta = 0.0
                            for i in range(use_n):
                                yy = cand_y[i]
                                xx = cand_x[i]
                                w = dist[i]
                                predict1 += target[b, yy, xx] * w
                                delta += (target[b, yy, xx] - donor[b, yy, xx]) * w
                            predict2 = donor[b, y, x] + delta
                            if predict2 > dn_min and predict2 < dn_max:
                                out[b, y, x] = wt1 * predict1 + wt2 * predict2
                            else:
                                out[b, y, x] = predict1
                        quality[y, x] = qcode
                        filled = True
                        break

                    # no similar at max window -> temporal-delta fallback from all common pixels
                    if c_similar == 0 and extent >= max_window:
                        if c_common > 0:
                            for b in range(bands):
                                delta_sum = 0.0
                                for i in range(c_common):
                                    yy = cand_y[i]
                                    xx = cand_x[i]
                                    delta_sum += target[b, yy, xx] - donor[b, yy, xx]
                                pred = donor[b, y, x] + delta_sum / c_common
                                if pred < dn_min:
                                    pred = dn_min
                                if pred > dn_max:
                                    pred = dn_max
                                out[b, y, x] = pred
                            quality[y, x] = 3
                            filled = True
                            break
                        quality[y, x] = 5
                        break
                else:
                    if extent < max_window:
                        extent += 1
                        continue
                    if c_common > 0:
                        for b in range(bands):
                            delta_sum = 0.0
                            for i in range(c_common):
                                yy = cand_y[i]
                                xx = cand_x[i]
                                delta_sum += target[b, yy, xx] - donor[b, yy, xx]
                            pred = donor[b, y, x] + delta_sum / c_common
                            if pred < dn_min:
                                pred = dn_min
                            if pred > dn_max:
                                pred = dn_max
                            out[b, y, x] = pred
                        quality[y, x] = 3
                        filled = True
                        break
                    quality[y, x] = 5
                    break

            if not filled and quality[y, x] == 0 and not target_valid[y, x]:
                quality[y, x] = 5
                for b in range(bands):
                    out[b, y, x] = np.nan

    return out, quality


def load_stack(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
    optical = arr[:6]
    if arr.shape[0] >= 10:
        valid = arr[9] > 0.5  # clear_valid
    else:
        valid = arr[6] > 0.5  # legacy valid_optical
    valid &= np.isfinite(optical).all(axis=0)
    return optical, valid, profile


def load_stack_with_fill_mask(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
    optical = arr[:6]
    if arr.shape[0] >= 11:
        valid = arr[9] > 0.5  # clear_valid for common-pixel search
        fill_mask = arr[10] > 0.5  # target slc_gap only
    else:
        valid = arr[6] > 0.5
        fill_mask = ~valid
    valid &= np.isfinite(optical).all(axis=0)
    fill_mask &= np.isfinite(optical).all(axis=0) | fill_mask
    return optical, valid, fill_mask, profile


def compute_similarity_threshold(donor: np.ndarray, donor_valid: np.ndarray, num_class: int) -> float:
    vals = donor[:, donor_valid]
    per_band = np.nanstd(vals, axis=1) * 2.0 / float(num_class)
    return float(np.nanmean(per_band))


def write_outputs(
    out_path: Path,
    quality_path: Path,
    filled: np.ndarray,
    quality: np.ndarray,
    profile: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile_out = profile.copy()
    profile_out.update(count=6, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(out_path, "w", **profile_out) as dst:
        dst.write(filled.astype(np.float32))
        for idx, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(idx, name)

    q_profile = profile.copy()
    q_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(quality_path, "w", **q_profile) as dst:
        dst.write(quality[None, :, :])
        dst.set_band_description(1, "nspi_quality")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-output", type=Path, default=None)
    parser.add_argument("--min-similar", type=int, default=20)
    parser.add_argument("--max-window", type=int, default=8)
    parser.add_argument("--num-class", type=int, default=5)
    parser.add_argument("--dn-min", type=float, default=0.0)
    parser.add_argument("--dn-max", type=float, default=1.0)
    args = parser.parse_args()

    quality_output = args.quality_output or args.output.with_name(args.output.stem + "_quality.tif")

    target, target_valid, profile = load_stack(args.target)
    donor, donor_valid, _ = load_stack(args.donor)
    similar_th = compute_similarity_threshold(donor, donor_valid, args.num_class)

    print(f"target valid fraction: {target_valid.mean():.4f}")
    print(f"donor valid fraction:  {donor_valid.mean():.4f}")
    print(f"fill candidate fraction: {((~target_valid) & donor_valid).mean():.4f}")
    print(f"similarity threshold: {similar_th:.6f}")

    filled, quality = _nspi_single(
        target,
        donor,
        target_valid,
        donor_valid,
        similar_th,
        args.min_similar,
        args.max_window,
        args.dn_min,
        args.dn_max,
    )

    unique, counts = np.unique(quality, return_counts=True)
    summary = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"quality counts: {summary}")

    write_outputs(Path(args.output), Path(quality_output), filled, quality, profile)
    metrics_path = Path(args.output).with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "target": str(args.target),
                "donor": str(args.donor),
                "output": str(args.output),
                "quality_output": str(quality_output),
                "min_similar": args.min_similar,
                "max_window": args.max_window,
                "num_class": args.num_class,
                "similarity_threshold": similar_th,
                "quality_counts": summary,
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")
    print(f"wrote {quality_output}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
