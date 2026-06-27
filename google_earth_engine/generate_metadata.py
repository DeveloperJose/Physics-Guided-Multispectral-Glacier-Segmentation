#!/usr/bin/env python3
"""
Generate metadata sidecar JSON for median and gapfill exports.

Runs the same GEE queries that the export script uses (deterministic),
saves metadata locally without re-exporting imagery.

Output:
  output/hkh_metadata/
    median/
      le07_147038_20040908_c15.json
      ...
    gapfill_mixed/
      le07_147038_20040908.json
      ...
    gapfill_le07/
      ...
    manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import ee

ee.Authenticate(auth_mode="localhost")
ee.Initialize(project="hkh-glacier-mapping")

SLC_FAILURE = datetime(2003, 5, 31)

REPORT_SCENES = [
    (153, 36, 2006, 9, 24, "LE07"), (152, 35, 2006, 9, 17, "LE07"),
    (152, 34, 2006, 7, 31, "LE07"), (152, 33, 2006, 7, 31, "LE07"),
    (151, 35, 2006, 7, 8, "LE07"), (151, 34, 2005, 8, 22, "LE07"),
    (150, 36, 2005, 9, 16, "LE07"), (150, 35, 2007, 11, 9, "LE07"),
    (150, 34, 2005, 9, 16, "LE07"), (149, 37, 2004, 10, 24, "LE07"),
    (149, 36, 2007, 11, 2, "LE07"), (149, 35, 2007, 9, 15, "LE07"),
    (149, 34, 2006, 7, 26, "LE07"), (148, 37, 2007, 11, 27, "LE07"),
    (148, 36, 2005, 9, 2, "LE07"), (148, 35, 2006, 11, 8, "LE07"),
    (147, 38, 2004, 9, 8, "LE07"), (147, 37, 2006, 9, 30, "LE07"),
    (147, 36, 2006, 9, 30, "LE07"), (147, 35, 2005, 8, 26, "LE07"),
    (146, 39, 2005, 11, 23, "LE07"), (146, 38, 2006, 9, 23, "LE07"),
    (146, 37, 2007, 12, 31, "LE07"), (146, 36, 2009, 8, 14, "LE07"),
    (145, 39, 2001, 10, 20, "LE07"), (144, 39, 2005, 12, 11, "LE07"),
    (143, 39, 2008, 12, 12, "LE07"), (143, 40, 2005, 10, 17, "LE07"),
    (142, 40, 2008, 11, 3, "LE07"), (141, 40, 2005, 11, 12, "LT05"),
    (141, 41, 2005, 11, 12, "LT05"), (140, 41, 2007, 12, 21, "LE07"),
    (139, 41, 2007, 12, 14, "LE07"), (138, 41, 2007, 12, 23, "LE07"),
    (137, 41, 2006, 1, 27, "LE07"), (136, 41, 2006, 7, 31, "LE07"),
    (136, 40, 2008, 11, 9, "LE07"), (135, 40, 2008, 12, 4, "LE07"),
    (134, 40, 2009, 9, 27, "LE07"), (133, 40, 2004, 12, 11, "LE07"),
    (133, 41, 2009, 11, 7, "LE07"),
]

LE07_PROXIES = {
    (141, 40): (2005, 11, 4),
    (141, 41): (2005, 11, 4),
}

SENSOR_BANDS = {
    "LE07": ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"],
    "LT05": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
}


def scenes_le07_only():
    out = []
    for p, r, y, m, d, sensor in REPORT_SCENES:
        if (p, r) in LE07_PROXIES:
            y, m, d = LE07_PROXIES[(p, r)]
        out.append((p, r, y, m, d, "LE07"))
    return out


def pr_str(p, r):
    return f"{p:03d}-{r:03d}"


def export_desc(sensor, p, r, y, m, d):
    return f"{sensor.lower()}_{p:03d}{r:03d}_{y:04d}{m:02d}{d:02d}"


def get_median_metadata(p, r, y, m, d, window_days=180, max_cloud=15, max_scenes=5):
    """Return metadata dict for a median composite without exporting."""
    anchor_date = f"{y:04d}-{m:02d}-{d:02d}"
    anchor_dt = datetime(y, m, d)
    start_dt = anchor_dt - timedelta(days=window_days)
    end_dt = anchor_dt + timedelta(days=window_days)

    coll = (
        ee.ImageCollection("LANDSAT/LE07/C02/T1")
        .filter(ee.Filter.eq("WRS_PATH", p))
        .filter(ee.Filter.eq("WRS_ROW", r))
        .filterDate(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
        .map(lambda img: img.set(
            "_tdiff",
            ee.Date(img.get("system:time_start"))
            .difference(ee.Date(anchor_date), "day").abs()
        ))
        .sort("_tdiff")
    )

    total = coll.size().getInfo()
    use_n = min(total, max_scenes)
    limited = coll.limit(use_n)

    scene_ids = limited.aggregate_array("LANDSAT_PRODUCT_ID").getInfo()
    scene_dates = limited.aggregate_array("DATE_ACQUIRED").getInfo()
    scene_clouds = limited.aggregate_array("CLOUD_COVER").getInfo()
    scene_tdiffs = limited.aggregate_array("_tdiff").getInfo()

    # Round tdiffs to ints
    scene_tdiffs = [round(t) for t in scene_tdiffs]

    return {
        "dataset": "median",
        "pr": pr_str(p, r),
        "anchor_date": anchor_date,
        "window_days": window_days,
        "cloud_threshold": max_cloud,
        "n_scenes_total": total,
        "n_scenes_used": use_n,
        "scene_ids": scene_ids,
        "scene_dates": scene_dates,
        "scene_clouds": scene_clouds,
        "scene_tdiff_days": scene_tdiffs,
    }


def get_gapfill_metadata(p, r, y, m, d, sensor):
    """Return metadata dict for a gapfill export."""
    anchor_date = f"{y:04d}-{m:02d}-{d:02d}"
    ename = export_desc(sensor, p, r, y, m, d)
    needs_gapfill = sensor == "LE07" and datetime(y, m, d) > SLC_FAILURE

    meta = {
        "dataset": "gapfill",
        "pr": pr_str(p, r),
        "target_scene": ename,
        "target_date": anchor_date,
        "target_sensor": sensor,
        "needs_gapfill": needs_gapfill,
    }

    if needs_gapfill:
        fill = (
            ee.ImageCollection("LANDSAT/LE07/C02/T1")
            .filterDate("1999-04-15", "2003-05-31")
            .filter(ee.Filter.eq("WRS_ROW", r))
            .filter(ee.Filter.eq("WRS_PATH", p))
            .filter(ee.Filter.lt("CLOUD_COVER", 10))
            .sort("CLOUD_COVER")
            .sort("DATE_ACQUIRED", False)
            .first()
        )

        fill_id = fill.get("LANDSAT_PRODUCT_ID").getInfo()
        fill_date = fill.get("DATE_ACQUIRED").getInfo()
        fill_cloud = fill.get("CLOUD_COVER").getInfo()
        fill_datetime = datetime.strptime(fill_date, "%Y-%m-%d") if fill_date else None

        meta["fill_scene_id"] = fill_id
        meta["fill_date"] = fill_date
        meta["fill_cloud"] = fill_cloud
        if fill_datetime:
            target_dt = datetime(y, m, d)
            meta["fill_tdiff_days"] = abs((fill_datetime - target_dt).days)
    else:
        meta["fill_scene_id"] = None
        meta["fill_date"] = None
        meta["fill_cloud"] = None
        meta["fill_tdiff_days"] = None

    return meta


def save_meta(out_dir, name, meta):
    path = Path(out_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="output/hkh_metadata")
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--max-cloud", type=int, default=15)
    parser.add_argument("--max-scenes", type=int, default=5)
    parser.add_argument("--subset", type=str, help="Comma-separated path/rows")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = set(s.strip() for s in args.subset.split(",")) if args.subset else None

    scenes_le07 = scenes_le07_only()
    manifest_rows = []

    # === GAPFILL_MIXED ===
    print("=== GAPFILL_MIXED ===")
    for p, r, y, m, d, sensor in REPORT_SCENES:
        pr = pr_str(p, r)
        if subset and pr not in subset:
            continue
        ename = export_desc(sensor, p, r, y, m, d)
        meta = get_gapfill_metadata(p, r, y, m, d, sensor)
        path = save_meta(out_dir / "gapfill_mixed", ename, meta)
        print(f"  {ename}  → {path}")

        manifest_rows.append({
            "pr": pr, "dataset": "gapfill_mixed", "anchor_date": meta["target_date"],
            "n_scenes": 1, "cloud_thresh": "N/A", "method": "gapfill",
            "fill_scene": meta.get("fill_scene_id", ""),
        })

    # === GAPFILL_LE07 ===
    print("=== GAPFILL_LE07 ===")
    for p, r, y, m, d, sensor in scenes_le07:
        pr = pr_str(p, r)
        if subset and pr not in subset:
            continue
        ename = export_desc("LE07", p, r, y, m, d)
        meta = get_gapfill_metadata(p, r, y, m, d, "LE07")
        path = save_meta(out_dir / "gapfill_le07", ename, meta)
        print(f"  {ename}  → {path}")

        manifest_rows.append({
            "pr": pr, "dataset": "gapfill_le07", "anchor_date": meta["target_date"],
            "n_scenes": 1, "cloud_thresh": "N/A", "method": "gapfill",
            "fill_scene": meta.get("fill_scene_id", ""),
        })

    # === MEDIAN ===
    print(f"=== MEDIAN (window=±{args.window_days}d, cloud<{args.max_cloud}%, N≤{args.max_scenes}) ===")
    for p, r, y, m, d, sensor in scenes_le07:
        pr = pr_str(p, r)
        if subset and pr not in subset:
            continue
        ename = f"le07_{p:03d}{r:03d}_{y:04d}{m:02d}{d:02d}_c{args.max_cloud}"
        meta = get_median_metadata(p, r, y, m, d,
                                   window_days=args.window_days,
                                   max_cloud=args.max_cloud,
                                   max_scenes=args.max_scenes)
        path = save_meta(out_dir / "median", ename, meta)
        print(f"  {ename}  → {path}")

        manifest_rows.append({
            "pr": pr, "dataset": "median", "anchor_date": meta["anchor_date"],
            "n_scenes": meta["n_scenes_used"], "cloud_thresh": args.max_cloud,
            "method": "median", "fill_scene": "",
        })

    # === MANIFEST CSV ===
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pr", "dataset", "anchor_date", "n_scenes", "cloud_thresh",
            "method", "fill_scene",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest: {manifest_path}")
    print(f"Total entries: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
