#!/usr/bin/env python3
"""Experimental multi-donor pre-SLC gapfill for one HKH tile.

Builds per-donor Phase-I-like gapfill candidates, then chooses the best donor
locally using a compatibility score surface derived from overlapping valid pixels.

Exports variants using top N ranked pre-SLC donors: top2, top3, top5, max.
Debug use only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee

PROJECT = "hkh-glacier-mapping"
COLLECTION = "LANDSAT/LE07/C02/T1_TOA"
FISHNET_PATH = Path("google_earth_scripts/hkh_fishnet.geojson")
FOLDER = "HKH_rebuild_gapfill_le07"
BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
SENSOR_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"]


TOP_DONORS_144039 = [
    "1999-11-09",
    "2000-12-29",
    "2001-10-13",
    "2002-09-30",
    "2000-10-10",
]


def load_fishnet_tile(tile_index: int) -> ee.Geometry:
    with FISHNET_PATH.open() as f:
        data = json.load(f)
    for feat in data["features"]:
        props = feat.get("properties", {})
        if int(props.get("_export_index", -1)) == tile_index:
            return ee.Geometry(feat["geometry"])
    raise ValueError(f"Fishnet tile not found: {tile_index}")


def apply_common_mask(img: ee.Image, mask_bands: list[str] | None = None) -> ee.Image:
    mask_src = img.select(mask_bands) if mask_bands is not None else img
    return img.updateMask(mask_src.mask().reduce(ee.Reducer.min()))


def prepare_export_image(img: ee.Image) -> ee.Image:
    return img.toFloat()


def add_ndsi_brightness(img: ee.Image) -> ee.Image:
    ndsi = img.normalizedDifference(["B2", "B5"]).rename("NDSI")
    brightness = img.select(BANDS).reduce(ee.Reducer.mean()).rename("BRIGHTNESS")
    return img.addBands([ndsi, brightness])


def single_donor_gapfill(
    src: ee.Image,
    fill: ee.Image,
    min_neighbors: int = 64,
    kernel_size: int = 5,
    upscale: bool = True,
) -> ee.Image:
    min_scale = 1 / 3
    max_scale = 3

    common = src.mask().And(fill.mask())
    fc = fill.updateMask(common)
    sc = src.updateMask(common)
    regress = fc.addBands(sc)
    regress = regress.select(ee.List(regress.bandNames()).sort())
    kernel = ee.Kernel.square(kernel_size * 30, "meters", False)
    ratio = 5

    if upscale:
        fit = (
            regress.reduceResolution(ee.Reducer.median(), False, 500)
            .reproject(regress.select(0).projection().scale(ratio, ratio))
            .reduceNeighborhood(ee.Reducer.linearFit().forEach(src.bandNames()), kernel, None, False)
            .unmask()
            .reproject(regress.select(0).projection().scale(ratio, ratio))
        )
    else:
        fit = regress.reduceNeighborhood(
            ee.Reducer.linearFit().forEach(src.bandNames()), kernel, None, False
        )

    offset = fit.select(".*_offset")
    scale = fit.select(".*_scale")
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True)

    if upscale:
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
    else:
        src_stats = src.reduceNeighborhood(reducer, kernel, None, False)
        fill_stats = fill.reduceNeighborhood(reducer, kernel, None, False)

    scale2 = src_stats.select(".*stdDev").divide(fill_stats.select(".*stdDev"))
    offset2 = src_stats.select(".*mean").subtract(fill_stats.select(".*mean").multiply(scale2))
    invalid = scale.lt(min_scale).Or(scale.gt(max_scale))
    scale = scale.where(invalid, scale2)
    offset = offset.where(invalid, offset2)
    invalid2 = scale.lt(min_scale).Or(scale.gt(max_scale))
    scale = scale.where(invalid2, 1)
    offset = offset.where(invalid2, src_stats.select(".*mean").subtract(fill_stats.select(".*mean")))
    count = common.reduceNeighborhood(ee.Reducer.count(), kernel, None, True, "boxcar")
    scaled = fill.multiply(scale).add(offset).updateMask(count.gte(min_neighbors))
    return src.unmask(scaled, True)


def donor_score_surface(src: ee.Image, donor: ee.Image, kernel_size: int = 5) -> ee.Image:
    """Low score = better match. Normalised to 0-1 within valid overlap."""
    srcx = add_ndsi_brightness(src.select(BANDS))
    donx = add_ndsi_brightness(donor.select(BANDS))
    common = srcx.select(BANDS).mask().reduce(ee.Reducer.min()).And(
        donx.select(BANDS).mask().reduce(ee.Reducer.min())
    )
    diffs = srcx.select(BANDS + ["NDSI", "BRIGHTNESS"]).subtract(
        donx.select(BANDS + ["NDSI", "BRIGHTNESS"])
    ).abs().updateMask(common)
    kernel = ee.Kernel.square(kernel_size * 30, "meters", False)
    local = diffs.reduceNeighborhood(ee.Reducer.mean(), kernel, None, False)
    score = local.reduce(ee.Reducer.mean()).rename("score")
    # Normalise to 0-1 so quality is comparable across donors.
    min_s = score.reduceNeighborhood(ee.Reducer.min(), kernel, None, False)
    max_s = score.reduceNeighborhood(ee.Reducer.max(), kernel, None, False)
    norm = score.subtract(min_s).divide(max_s.subtract(min_s).add(1e-12))
    return norm.unmask(1)  # unmasked -> high score (worse) so avoid


def build_candidate(src: ee.Image, donor: ee.Image, donor_date: str, kernel_size: int, min_neighbors: int) -> ee.Image:
    filled = single_donor_gapfill(src, donor, kernel_size=kernel_size, min_neighbors=min_neighbors)
    filled = apply_common_mask(filled.select(SENSOR_BANDS), BANDS)

    src_valid = src.select(SENSOR_BANDS).mask().reduce(ee.Reducer.min())
    fill_valid = filled.select(SENSOR_BANDS).mask().reduce(ee.Reducer.min())
    missing = src_valid.Not()

    # qualityMosaic masks output wherever quality is masked. Give source-valid
    # pixels quality=1 for every donor so full image survives; use donor
    # compatibility only inside SLC-off gaps.
    score = donor_score_surface(src, donor, kernel_size=kernel_size)
    gap_quality = ee.Image.constant(1).subtract(score).updateMask(missing)
    source_quality = ee.Image.constant(1).updateMask(src_valid)
    quality = source_quality.unmask(gap_quality).updateMask(fill_valid).rename("quality")

    return filled.addBands(quality).set("donor_date", donor_date)


def export_blend_variant(
    src: ee.Image,
    tile_geom: ee.Geometry,
    donor_dates: list[str],
    desc: str,
    temperature: float,
    pr_str: str,
) -> None:
    src_clip = src.clip(tile_geom)
    crs = src.select("B1").projection().crs().getInfo()
    src_valid = src_clip.select(SENSOR_BANDS).mask().reduce(ee.Reducer.min())
    missing = src_valid.Not()

    num = ee.Image.constant([0] * len(SENSOR_BANDS)).rename(SENSOR_BANDS).toFloat()
    den = ee.Image.constant(0).rename("weight").toFloat()

    for donor_date in donor_dates:
        donor_id = f"LE07_{pr_str}_{donor_date.replace('-', '')}"
        donor = ee.Image(f"{COLLECTION}/{donor_id}").clip(tile_geom)
        filled = single_donor_gapfill(src_clip, donor, kernel_size=5, min_neighbors=64)
        filled = apply_common_mask(filled.select(SENSOR_BANDS), BANDS).toFloat()
        fill_valid = filled.mask().reduce(ee.Reducer.min())
        score = donor_score_surface(src_clip, donor, kernel_size=5)
        weight = score.divide(temperature).multiply(-1).exp()
        weight = weight.updateMask(missing).updateMask(fill_valid).rename("weight")
        num = num.add(filled.multiply(weight))
        den = den.add(weight)

    blended_gap = num.divide(den).updateMask(den.gt(0))
    out = src_clip.select(SENSOR_BANDS).unmask(blended_gap, True)
    out = prepare_export_image(apply_common_mask(out, BANDS))

    task = ee.batch.Export.image.toDrive(
        image=out,
        description=desc,
        folder=FOLDER,
        crs=crs,
        region=tile_geom,
        scale=30,
        maxPixels=1e9,
    )
    task.start()
    print(f"submitted {desc} donors={donor_dates} temp={temperature} crs={crs}")


def export_variant(src: ee.Image, tile_geom: ee.Geometry, donor_dates: list[str], desc: str, pr_str: str) -> None:
    # Clip everything to tile early to avoid full-scene processing explosion.
    src_clip = src.clip(tile_geom)
    crs = src.select("B1").projection().crs().getInfo()  # e.g. EPSG:32644

    candidates = []
    for donor_date in donor_dates:
        donor_id = f"LE07_{pr_str}_{donor_date.replace('-', '')}"
        donor = ee.Image(f"{COLLECTION}/{donor_id}").clip(tile_geom)
        candidates.append(build_candidate(src_clip, donor, donor_date, kernel_size=5, min_neighbors=64))

    coll = ee.ImageCollection(candidates)
    # Positive quality (higher = better): score is low=good, so quality = 1 - score normed.
    # qualityMosaic picks highest; should now select best donor per pixel.
    winner = coll.qualityMosaic("quality").select(SENSOR_BANDS)
    winner = prepare_export_image(winner)

    task = ee.batch.Export.image.toDrive(
        image=winner,
        description=desc,
        folder=FOLDER,
        crs=crs,
        region=tile_geom,
        scale=30,
        maxPixels=1e9,
    )
    task.start()
    print(f"submitted {desc} donors={donor_dates} crs={crs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-index", type=int, default=161)
    parser.add_argument("--path", type=int, default=144, help="WRS-2 path")
    parser.add_argument("--row", type=int, default=39, help="WRS-2 row")
    parser.add_argument("--target-date", default="2005-12-11")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--blend-top3", action="store_true", help="Export top3 weighted blend")
    parser.add_argument("--blend-temp", type=float, default=0.05, help="Blend softmax temperature")
    args = parser.parse_args()

    ee.Authenticate(auth_mode="localhost")
    ee.Initialize(project=PROJECT)

    tile_geom = load_fishnet_tile(args.tile_index)
    pr_str = f"{args.path:03d}{args.row:03d}"
    target_id = f"LE07_{pr_str}_{args.target_date.replace('-', '')}"
    src = ee.Image(f"{COLLECTION}/{target_id}")

    from datetime import datetime as dt

    target_date = dt.strptime(args.target_date, "%Y-%m-%d")
    target_doy = target_date.timetuple().tm_yday
    full_preslc = (
        ee.ImageCollection(COLLECTION)
        .filter(ee.Filter.eq("WRS_PATH", args.path))
        .filter(ee.Filter.eq("WRS_ROW", args.row))
        .filterDate("1999-04-15", "2003-05-31")
        .filter(ee.Filter.lt("CLOUD_COVER", 15))
        .aggregate_array("DATE_ACQUIRED")
        .getInfo()
    )
    ranked = sorted(
        sorted(set(full_preslc)),
        key=lambda d: abs(dt.strptime(d, "%Y-%m-%d").timetuple().tm_yday - target_doy),
    )
    print(f"{pr_str} donors ranked by DOY proximity: {ranked[:8]}")

    if args.blend_top3:
        donor_dates = ranked[:3]
        temp_label = f"t{int(round(args.blend_temp * 1000)):03d}"
        desc = f"gapfill_le07_multidonor_{pr_str}_tile{args.tile_index}_top3_blend_{temp_label}_optmask"
        if args.dry_run:
            print(desc, donor_dates, args.blend_temp)
        else:
            export_blend_variant(src, tile_geom, donor_dates, desc, args.blend_temp, pr_str)
        return

    variants = {
        "top2": ranked[:2],
        "top3": ranked[:3],
        "top5": ranked[:5],
        "maxpreslc": ranked,
    }

    for name, donor_dates in variants.items():
        desc = f"gapfill_le07_multidonor_{pr_str}_tile{args.tile_index}_{name}_v3"
        if args.dry_run:
            print(desc, donor_dates)
        else:
            export_variant(src, tile_geom, donor_dates, desc, pr_str)


if __name__ == "__main__":
    main()
