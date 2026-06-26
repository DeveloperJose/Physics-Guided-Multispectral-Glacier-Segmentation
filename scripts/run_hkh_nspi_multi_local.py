#!/usr/bin/env python3
"""Local multi-input NSPI aligned to original Chen/Zhu paper + IDL code.

Primary references used for logic auditing:
- Paper: Chen et al. 2011, Remote Sensing of Environment 115, 1053-1064
  (`analysis/nspi_reference/nspi_paper.txt`)
- IDL code: `analysis/nspi_reference/nspi_update_20100824/NSPI update 20100824/FILLGAP_MULTIPLE_V2.pro`
- IDL readme: `analysis/nspi_reference/nspi_update_20100824/NSPI update 20100824/readme_FILLGAP_MULTIPLE_V2.txt`

Key semantics copied from IDL/code when possible:
- donor priority is external; nearest-date first for paper-faithful multi-input runs
- no donor blending; first donor that fills a pixel wins
- each donor runs single-input NSPI against original target valid pixels only
- `max_window=15` means radius 15 => full 31x31 window (paper multi-input default)
- quality codes follow Table 2 / IDL donor-index scheme
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio

REFERENCE_INFO = {
    "paper": "Chen et al. 2011, Remote Sensing of Environment 115:1053-1064",
    "paper_text": "analysis/nspi_reference/nspi_paper.txt",
    "idl_code": "analysis/nspi_reference/nspi_update_20100824/NSPI update 20100824/FILLGAP_MULTIPLE_V2.pro",
    "idl_readme": "analysis/nspi_reference/nspi_update_20100824/NSPI update 20100824/readme_FILLGAP_MULTIPLE_V2.txt",
}


from run_hkh_nspi_local import (
    BAND_NAMES,
    _nspi_single,
    compute_similarity_threshold,
    load_stack,
    write_outputs,
)


def _write_uint8_band(path: Path, arr: np.ndarray, profile: dict, desc: str) -> None:
    p = profile.copy()
    p.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr[None, :, :].astype(np.uint8))
        dst.set_band_description(1, desc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--donors", type=Path, nargs="+", required=True,
                        help="Donor files in priority order")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quality-output", type=Path, default=None)
    parser.add_argument("--min-similar", type=int, default=20)
    parser.add_argument("--max-window", type=int, default=15)
    parser.add_argument("--num-class", type=int, default=5)
    parser.add_argument("--dn-min", type=float, default=0.0)
    parser.add_argument("--dn-max", type=float, default=1.0)
    parser.add_argument("--fallback-copy", action="store_true",
                        help="After paper NSPI cascade, copy first valid donor for remaining pixels")
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="3-digit prefix for numbered output (e.g. 012)")
    parser.add_argument("--experiment-name", type=str, default="",
                        help="Short experiment description")
    parser.add_argument("--write-debug", action="store_true",
                        help="Write extra debug rasters to debug/ subfolder")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Root dir for output. If set, --output stem is constructed from id+name")
    args = parser.parse_args()

    # Resolve output path
    if args.output_root is not None:
        if args.experiment_id is None:
            raise ValueError("--experiment-id required when --output-root is set")
        stem = f"{args.experiment_id}_{args.experiment_name}" if args.experiment_name else args.experiment_id
        args.output_root = Path(args.output_root)
        args.output = args.output_root / f"{stem}.tif"
        debug_dir = args.output_root / "debug" / stem
        args.quality_output = debug_dir / "quality.tif"
    else:
        if args.output is None:
            parser.error("--output required when --output-root not set")
        debug_dir = None
    quality_output = args.quality_output or args.output.with_name(args.output.stem + "_quality.tif")
    output_path = Path(args.output)
    quality_path = Path(quality_output)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    target, target_valid, profile = load_stack(args.target)
    filled = target.copy()
    quality = np.zeros(target.shape[1:], dtype=np.uint8)
    remaining = ~target_valid

    donor_summaries: list[dict] = []
    print(f"target valid fraction: {target_valid.mean():.4f}")

    for donor_idx, donor_path in enumerate(args.donors, start=1):
        donor, donor_valid, _ = load_stack(donor_path)
        similar_th = compute_similarity_threshold(donor, donor_valid, args.num_class)
        fill_candidate = remaining & donor_valid
        print(
            f"donor{donor_idx}: {donor_path.name} valid={donor_valid.mean():.4f} "
            f"remaining-fill-candidate={fill_candidate.mean():.4f} th={similar_th:.6f}"
        )
        if not fill_candidate.any():
            donor_summaries.append(
                {
                    "donor_index": donor_idx,
                    "path": str(donor_path),
                    "similarity_threshold": similar_th,
                    "filled_pixels": 0,
                }
            )
            continue

        donor_out, donor_quality = _nspi_single(
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

        usable = remaining & np.isin(donor_quality, np.array([1, 2, 3], dtype=np.uint8))
        for b in range(target.shape[0]):
            filled[b, usable] = donor_out[b, usable]
        # Paper quality encodes method + 10 * donor index.
        quality[usable] = donor_quality[usable] + 10 * donor_idx
        remaining[usable] = False

        unique, counts = np.unique(donor_quality, return_counts=True)
        q_summary = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"  donor{donor_idx} single-quality={q_summary}; used={int(usable.sum())}; remaining={int(remaining.sum())}")
        donor_summaries.append(
            {
                "donor_index": donor_idx,
                "path": str(donor_path),
                "similarity_threshold": similar_th,
                "single_quality_counts": q_summary,
                "filled_pixels": int(usable.sum()),
                "remaining_pixels_after": int(remaining.sum()),
            }
        )

    if args.fallback_copy and remaining.any():
        for donor_idx, donor_path in enumerate(args.donors, start=1):
            donor, donor_valid, _ = load_stack(donor_path)
            copy_mask = remaining & donor_valid
            if copy_mask.any():
                for b in range(target.shape[0]):
                    filled[b, copy_mask] = donor[b, copy_mask]
                quality[copy_mask] = 200 + donor_idx
                remaining[copy_mask] = False
                print(f"fallback copied {int(copy_mask.sum())} pixels from donor{donor_idx}")
            if not remaining.any():
                break

    quality[remaining] = 5
    for b in range(target.shape[0]):
        band = filled[b]
        band[remaining] = np.nan
        filled[b] = band

    unique, counts = np.unique(quality, return_counts=True)
    summary = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"multi quality counts: {summary}")

    write_outputs(output_path, quality_path, filled, quality, profile)

    if debug_dir is not None:
        # donor scores CSV always written under debug when using output-root
        csv_path = debug_dir / "donor_scores.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["donor_index", "path", "similarity_threshold", "filled_pixels", "remaining_pixels_after"])
            for d in donor_summaries:
                w.writerow([
                    d["donor_index"],
                    d["path"],
                    d.get("similarity_threshold"),
                    d.get("filled_pixels", 0),
                    d.get("remaining_pixels_after", 0),
                ])
        print(f"wrote {csv_path}")
        if args.write_debug:
            donor_id_map = np.zeros_like(quality, dtype=np.uint8)
            for val in np.unique(quality):
                if val == 0 or val == 5:
                    continue
                if val >= 200:
                    donor_id_map[quality == val] = val - 200
                elif val >= 10:
                    donor_id_map[quality == val] = val // 10
            _write_uint8_band(debug_dir / "donor_used.tif", donor_id_map, profile, "donor_id")

    metrics_path = (debug_dir / "metadata.json") if debug_dir is not None else output_path.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "target": str(args.target),
                "donors": [str(d) for d in args.donors],
                "output": str(args.output),
                "quality_output": str(quality_output),
                "min_similar": args.min_similar,
                "max_window": args.max_window,
                "num_class": args.num_class,
                "references": REFERENCE_INFO,
                "quality_counts": summary,
                "donor_summaries": donor_summaries,
                "quality_codes": {
                    "0": "original target pixel",
                    "11/21/...": "donor index high confidence >= M similar",
                    "12/22/...": "donor index low confidence < M similar at max window",
                    "13/23/...": "donor index temporal-delta fallback",
                    "5": "unfilled",
                },
            },
            indent=2,
        )
    )
    print(f"wrote {output_path}")
    print(f"wrote {quality_path}")
    print(f"wrote {metrics_path}")
    if debug_dir is not None:
        print(f"debug: {debug_dir}")


if __name__ == "__main__":
    main()
