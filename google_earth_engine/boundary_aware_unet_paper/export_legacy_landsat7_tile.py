#!/usr/bin/env python3
"""Recreate legacy Landsat7_2005.js export in Python for one fishnet tile.

Goal: audit old JS workflow exactly, without mixing with current rebuild script.
Defaults match old JS as closely as possible:
- LE07 Collection 2 Tier 1 (modern replacement for deprecated C01/T1_RT)
- legacy ids.js scene list
- legacy gapfill.js donor logic (year 2000, cloud<10, local linear fit)
- mosaic all scenes intersecting target fishnet tile
- export only one fishnet tile for fast debugging

Example:
  uv run python google_earth_scripts/export_legacy_landsat7_tile.py --tile-index 161
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import ee

PROJECT = "hkh-glacier-mapping"
FISHNET_PATH = Path("google_earth_scripts/hkh_fishnet.geojson")
FOLDER = "Landsat7_2005"
COLLECTION = "LANDSAT/LE07/C02/T1"
SLC_FAILURE = datetime(2003, 5, 31)
IMAGE_IDS = [
    "LE07_153036_20060924",
    "LE07_152035_20060917",
    "LE07_152034_20060731",
    "LE07_152033_20060731",
    "LE07_151035_20060708",
    "LE07_151034_20050822",
    "LE07_150036_20050916",
    "LE07_150035_20050916",
    "LE07_150034_20050916",
    "LE07_149037_20041024",
    "LE07_149036_20071102",
    "LE07_149035_20070915",
    "LE07_149034_20060726",
    "LE07_148037_20071127",
    "LE07_148036_20050902",
    "LE07_148035_20061108",
    "LE07_147038_20040908",
    "LE07_147037_20060930",
    "LE07_147036_20060930",
    "LE07_147035_20050826",
    "LE07_146039_20051123",
    "LE07_146038_20060923",
    "LE07_146037_20071231",
    "LE07_146036_20090814",
    "LE07_145039_20011020",
    "LE07_144039_20011013",
    "LE07_143039_20081212",
    "LE07_143040_20051017",
    "LE07_142040_20041226",
    "LE07_141040_20041219",
    "LE07_141041_20041203",
    "LE07_140041_20071221",
    "LE07_139041_20071214",
    "LE07_138041_20071223",
    "LE07_137041_20060127",
    "LE07_136041_20060731",
    "LE07_136040_20081109",
    "LE07_135040_20081204",
    "LE07_134040_20090927",
    "LE07_133040_20041211",
    "LE07_133041_20051112",
]
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"]


def load_fishnet_tile(tile_index: int) -> ee.Feature:
    with FISHNET_PATH.open() as f:
        data = json.load(f)
    for feat in data["features"]:
        props = feat.get("properties", {})
        if int(props.get("_export_index", -1)) == tile_index:
            return ee.Feature(feat)
    raise ValueError(f"Fishnet tile not found: {tile_index}")


def gapfill_legacy(src: ee.Image) -> ee.Image:
    """Port of legacy google_earth_scripts/gapfill.js."""
    min_scale = 1 / 3
    max_scale = 3
    min_neighbors = 64
    kernel_size = 5
    ratio = 5

    fill = (
        ee.ImageCollection(COLLECTION)
        .filterDate("2000-01-01", "2000-12-31")
        .filter(ee.Filter.eq("WRS_ROW", src.get("WRS_ROW")))
        .filter(ee.Filter.eq("WRS_PATH", src.get("WRS_PATH")))
        .filter(ee.Filter.lt("CLOUD_COVER", 10))
        .sort("CLOUD_COVER")
        .first()
    )

    common = src.mask().And(fill.mask())
    fc = fill.updateMask(common)
    sc = src.updateMask(common)
    regress = fc.addBands(sc)
    regress = regress.select(ee.List(regress.bandNames()).sort())

    kernel = ee.Kernel.square(kernel_size * 30, "meters", False)
    fit = (
        regress.reduceResolution(ee.Reducer.median(), False, 500)
        .reproject(regress.select(0).projection().scale(ratio, ratio))
        .reduceNeighborhood(ee.Reducer.linearFit().forEach(src.bandNames()), kernel, None, False)
        .unmask()
        .reproject(regress.select(0).projection().scale(ratio, ratio))
    )

    offset = fit.select(".*_offset")
    scale = fit.select(".*_scale")

    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True)
    src_stats = (
        src.reduceResolution(ee.Reducer.median(), False, 500)
        .reproject(src.select(0).projection().scale(ratio, ratio))
        .reduceNeighborhood(reducer, kernel, None, False)
        .reproject(src.select(0).projection().scale(ratio, ratio))
    )
    fill_stats = (
        fill.reduceResolution(ee.Reducer.median(), False, 500)
        .reproject(fill.select(0).projection().scale(ratio, ratio))
        .reduceNeighborhood(reducer, kernel, None, False)
        .reproject(fill.select(0).projection().scale(ratio, ratio))
    )

    scale2 = src_stats.select(".*stdDev").divide(fill_stats.select(".*stdDev"))
    offset2 = src_stats.select(".*mean").subtract(fill_stats.select(".*mean").multiply(scale2))

    invalid = scale.lt(min_scale).Or(scale.gt(max_scale))
    scale = scale.where(invalid, scale2)
    offset = offset.where(invalid, offset2)

    invalid2 = scale.lt(min_scale).Or(scale.gt(max_scale))
    scale = scale.where(invalid2, 1)
    offset = offset.where(
        invalid2, src_stats.select(".*mean").subtract(fill_stats.select(".*mean"))
    )

    count = common.reduceNeighborhood(ee.Reducer.count(), kernel, None, True, "boxcar")
    scaled = fill.multiply(scale).add(offset).updateMask(count.gte(min_neighbors))
    return src.unmask(scaled, True).select(BANDS).uint8()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-index", type=int, default=161)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--folder", default=FOLDER)
    parser.add_argument("--description", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ee.Authenticate(auth_mode="localhost")
    ee.Initialize(project=args.project)

    tile = load_fishnet_tile(args.tile_index)
    geom = tile.geometry()
    desc = args.description or f"legacy_image{args.tile_index}"

    images = []
    for image_id in IMAGE_IDS:
        img = ee.Image(f"{COLLECTION}/{image_id}")
        ts = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        if datetime.strptime(ts, "%Y-%m-%d") > SLC_FAILURE:
            img = gapfill_legacy(img)
        else:
            img = img.select(BANDS).uint8()
        images.append(img.clip(img.geometry()))

    all_images = ee.ImageCollection(images)
    image = all_images.filterBounds(geom).mosaic().clip(geom)
    crs = image.select("B1").projection().crs().getInfo()

    print(f"tile={args.tile_index} desc={desc} crs={crs}")
    if args.dry_run:
        return

    task = ee.batch.Export.image.toDrive(
        image=image,
        folder=args.folder,
        crs=crs,
        description=desc,
        maxPixels=318080701,
        region=geom,
        scale=30,
    )
    task.start()
    print("submitted")


if __name__ == "__main__":
    main()
