#!/usr/bin/env python3
"""Diagnose LE07 gapfill donor candidates using overlap metrics.

Prints evidence-first donor table without hand-tuned weights.
Default target matches current audit case.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ee

PROJECT = "hkh-glacier-mapping"
COLLECTION = "LANDSAT/LE07/C02/T1_TOA"
BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]


def add_ndsi_brightness(img: ee.Image) -> ee.Image:
    ndsi = img.normalizedDifference(["B2", "B5"]).rename("NDSI")
    brightness = img.select(["B1", "B2", "B3", "B4", "B5", "B7"]).reduce(
        ee.Reducer.mean()
    ).rename("BRIGHTNESS")
    return img.addBands([ndsi, brightness])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=int, default=144)
    parser.add_argument("--row", type=int, default=39)
    parser.add_argument("--date", default="2005-12-11")
    parser.add_argument("--max-cloud", type=float, default=10)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    ee.Authenticate(auth_mode="localhost")
    ee.Initialize(project=PROJECT)

    date_compact = args.date.replace("-", "")
    target_id = f"LE07_{args.path:03d}{args.row:03d}_{date_compact}"
    target = add_ndsi_brightness(ee.Image(f"{COLLECTION}/{target_id}"))
    target_date = ee.Date(target.get("system:time_start"))
    target_doy = ee.Number(target_date.getRelative("day", "year"))
    target_year = ee.Number(target_date.get("year"))

    candidates = (
        ee.ImageCollection(COLLECTION)
        .filter(ee.Filter.eq("WRS_PATH", args.path))
        .filter(ee.Filter.eq("WRS_ROW", args.row))
        .filterDate("1999-04-15", "2003-05-31")
        .filter(ee.Filter.lt("CLOUD_COVER", args.max_cloud))
        .map(add_ndsi_brightness)
    )

    rows: list[dict[str, object]] = []

    cand_list = candidates.toList(candidates.size())
    n = candidates.size().getInfo()
    for i in range(n):
        donor = ee.Image(cand_list.get(i))
        donor_date = ee.Date(donor.get("system:time_start"))
        donor_doy = ee.Number(donor_date.getRelative("day", "year"))
        donor_year = ee.Number(donor_date.get("year"))
        doy_diff = donor_doy.subtract(target_doy).abs()
        doy_diff = doy_diff.min(ee.Number(365).subtract(doy_diff))
        year_diff = target_year.subtract(donor_year).abs()

        metric_bands = BANDS + ["NDSI", "BRIGHTNESS"]
        pair = target.select(metric_bands).addBands(donor.select(metric_bands), overwrite=False)
        # common valid overlap
        common = target.select(BANDS).mask().reduce(ee.Reducer.min()).And(
            donor.select(BANDS).mask().reduce(ee.Reducer.min())
        )
        pair = pair.updateMask(common)

        reducers = ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True)
        stats = pair.reduceRegion(
            reducer=reducers,
            geometry=target.geometry(),
            scale=30,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo()

        # overlap pixel count from first band
        overlap_count = target.select("B1").updateMask(common).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=target.geometry(),
            scale=30,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo().get("B1")

        # abs mean diffs
        diffs = target.select(metric_bands).subtract(donor.select(metric_bands)).abs().updateMask(common)
        diff_stats = diffs.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=target.geometry(),
            scale=30,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo()

        row = {
            "donor_id": donor.get("LANDSAT_PRODUCT_ID").getInfo(),
            "date": donor.get("DATE_ACQUIRED").getInfo(),
            "cloud": donor.get("CLOUD_COVER").getInfo(),
            "year_diff": year_diff.getInfo(),
            "doy_diff": doy_diff.getInfo(),
            "overlap_px": overlap_count,
            "mae_B1": diff_stats.get("B1"),
            "mae_B2": diff_stats.get("B2"),
            "mae_B3": diff_stats.get("B3"),
            "mae_B4": diff_stats.get("B4"),
            "mae_B5": diff_stats.get("B5"),
            "mae_B7": diff_stats.get("B7"),
            "ndsi_abs_diff": diff_stats.get("NDSI"),
            "brightness_abs_diff": diff_stats.get("BRIGHTNESS"),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["ndsi_abs_diff"], r["brightness_abs_diff"], r["mae_B5"]))

    header = [
        "donor_id", "date", "cloud", "year_diff", "doy_diff", "overlap_px",
        "mae_B1", "mae_B2", "mae_B3", "mae_B4", "mae_B5", "mae_B7",
        "ndsi_abs_diff", "brightness_abs_diff",
    ]
    print(",".join(header))
    for row in rows:
        print(",".join(str(row[h]) for h in header))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
