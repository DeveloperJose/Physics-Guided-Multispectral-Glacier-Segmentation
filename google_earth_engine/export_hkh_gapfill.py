#!/usr/bin/env python3
"""
Export full WRS scenes for three HKH glacier-mapping datasets.

All three use the same 41 IDs (report-corrected), differing only in
how each scene is processed:
  gapfill_mixed  — single image, LE07+LT05, gapfill for SLC-off
  gapfill_le07   — single image, LE07-only proxies, gapfill for SLC-off
  median         — median of N nearest scenes, LE07-only, no gapfill

Exports full WRS extent. No fishnet tiling in GEE.

Usage:
  # Dry-run
  uv run python scripts/export_hkh_scenes.py --dataset gapfill_le07 --dry-run
  uv run python scripts/export_hkh_scenes.py --dataset median --dry-run

  # Subset test
  uv run python scripts/export_hkh_scenes.py --dataset gapfill_mixed --subset 133-040,133-041
  uv run python scripts/export_hkh_scenes.py --dataset median --subset 133-040,133-041

  # Full export
  uv run python scripts/export_hkh_scenes.py --dataset gapfill_le07
  uv run python scripts/export_hkh_scenes.py --dataset median
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import ee

ee.Authenticate(auth_mode="localhost")
ee.Initialize(project="hkh-glacier-mapping")

COLLECTIONS = {
    "LE07": "LANDSAT/LE07/C02/T1_TOA",
    "LT05": "LANDSAT/LT05/C02/T1_TOA",
}
# Export all spectral bands (thermal too). Preprocessing selects the final recipe.
# LE07 C02 has dual thermal: B6_VCID_1 (low gain), B6_VCID_2 (high gain).
# LT05 C02 has single thermal: B6.
SENSOR_BANDS: dict[str, list[str]] = {
    "LE07": ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"],
    "LT05": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
}
SLC_FAILURE = datetime(2003, 5, 31)
FISHNET_PATH = Path("google_earth_scripts/hkh_fishnet.geojson")

# ---------------------------------------------------------------------------
# Scene definitions: (path, row, year, month, day, sensor)
# Base = Bibek's ids.js. 6 corrected to match ICIMOD report.
# ---------------------------------------------------------------------------
REPORT_SCENES = [
    (153, 36, 2006, 9, 24, "LE07"),
    (152, 35, 2006, 9, 17, "LE07"),
    (152, 34, 2006, 7, 31, "LE07"),
    (152, 33, 2006, 7, 31, "LE07"),
    (151, 35, 2006, 7, 8, "LE07"),
    (151, 34, 2005, 8, 22, "LE07"),
    (150, 36, 2005, 9, 16, "LE07"),
    (150, 35, 2007, 11, 9, "LE07"),  # was 2005-09-16
    (150, 34, 2005, 9, 16, "LE07"),
    (149, 37, 2004, 10, 24, "LE07"),
    (149, 36, 2007, 11, 2, "LE07"),
    (149, 35, 2007, 9, 15, "LE07"),
    (149, 34, 2006, 7, 26, "LE07"),
    (148, 37, 2007, 11, 27, "LE07"),
    (148, 36, 2005, 9, 2, "LE07"),
    (148, 35, 2006, 11, 8, "LE07"),
    (147, 38, 2004, 9, 8, "LE07"),
    (147, 37, 2006, 9, 30, "LE07"),
    (147, 36, 2006, 9, 30, "LE07"),
    (147, 35, 2005, 8, 26, "LE07"),
    (146, 39, 2005, 11, 23, "LE07"),
    (146, 38, 2006, 9, 23, "LE07"),
    (146, 37, 2007, 12, 31, "LE07"),
    (146, 36, 2009, 8, 14, "LE07"),
    (145, 39, 2001, 10, 20, "LE07"),
    (144, 39, 2005, 12, 11, "LE07"),  # was 2001-10-13
    (143, 39, 2008, 12, 12, "LE07"),
    (143, 40, 2005, 10, 17, "LE07"),
    (142, 40, 2008, 11, 3, "LE07"),  # was 2004-12-26
    (141, 40, 2005, 11, 12, "LT05"),  # report: LT05
    (141, 41, 2005, 11, 12, "LT05"),  # report: LT05
    (140, 41, 2007, 12, 21, "LE07"),
    (139, 41, 2007, 12, 14, "LE07"),
    (138, 41, 2007, 12, 23, "LE07"),
    (137, 41, 2006, 1, 27, "LE07"),
    (136, 41, 2006, 7, 31, "LE07"),
    (136, 40, 2008, 11, 9, "LE07"),
    (135, 40, 2008, 12, 4, "LE07"),
    (134, 40, 2009, 9, 27, "LE07"),
    (133, 40, 2004, 12, 11, "LE07"),
    (133, 41, 2009, 11, 7, "LE07"),  # was 2005-11-12
]

# LE07 proxies for LT05 report dates
LE07_PROXIES = {
    (141, 40): (2005, 11, 4),  # 8d before report LT05, 4% cloud
    (141, 41): (2005, 11, 4),  # 8d before report LT05, 3% cloud
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def needs_gapfill(y: int, m: int, d: int, sensor: str) -> bool:
    return sensor == "LE07" and datetime(y, m, d) > SLC_FAILURE


def ee_image_id(sensor: str, p: int, r: int, y: int, m: int, d: int) -> str:
    sid = f"{sensor}_{p:03d}{r:03d}_{y:04d}{m:02d}{d:02d}"
    return f"{COLLECTIONS[sensor]}/{sid}"


def export_desc(sensor: str, p: int, r: int, y: int, m: int, d: int) -> str:
    return f"{sensor.lower()}_{p:03d}{r:03d}_{y:04d}{m:02d}{d:02d}"


def scenes_LE07_only():
    """Report-corrected scenes with LE07 proxies for LT05 dates."""
    out = []
    for p, r, y, m, d, sensor in REPORT_SCENES:
        if (p, r) in LE07_PROXIES:
            y, m, d = LE07_PROXIES[(p, r)]
        out.append((p, r, y, m, d, "LE07"))
    return out


# ---------------------------------------------------------------------------
# Gapfill (USGS SLC-off algorithm, Gorelick/Donchyts)
# ---------------------------------------------------------------------------
def gapfill(
    src: ee.Image,
    forced_fill_date: str | None = None,
    min_neighbors: int = 64,
    kernel_size: int = 5,
    upscale: bool = True,
    fallback_to_donor: bool = False,
) -> ee.Image:
    """USGS SLC-off gapfill: regress a pre-SLC-failure fill scene onto src.

    The fill must be pre-SLC (before May 31, 2003) so it has no stripes.
    A post-SLC fill has stripes at the same positions as src and cannot fill them.

    Fill donor ranking:
      1. year distance to target (closest year first)
      2. day-of-year distance to target (seasonal match)
      3. cloud cover
      4. most recent among ties
    """
    MIN_SCALE = 1 / 3
    MAX_SCALE = 3

    src_date = ee.Date(src.get("system:time_start"))
    src_doy = ee.Number(src_date.getRelative("day", "year"))
    src_year = ee.Number(src_date.get("year"))

    def add_fill_rank(img: ee.Image) -> ee.Image:
        img_date = ee.Date(img.get("system:time_start"))
        img_doy = ee.Number(img_date.getRelative("day", "year"))
        img_year = ee.Number(img_date.get("year"))
        doy_diff = img_doy.subtract(src_doy).abs()
        doy_wrap = ee.Number(365).subtract(doy_diff)
        return img.set("_year_diff", src_year.subtract(img_year).abs()).set(
            "_doy_diff", doy_diff.min(doy_wrap)
        )

    fill_pool = (
        ee.ImageCollection(COLLECTIONS["LE07"])
        .filterDate("1999-04-15", "2003-05-31")
        .filter(ee.Filter.eq("WRS_ROW", src.get("WRS_ROW")))
        .filter(ee.Filter.eq("WRS_PATH", src.get("WRS_PATH")))
        .filter(ee.Filter.lt("CLOUD_COVER", 10))
        .map(add_fill_rank)
    )

    if forced_fill_date is not None:
        fill = fill_pool.filterDate(forced_fill_date, ee.Date(forced_fill_date).advance(1, "day")).first()
    else:
        fill = (
            fill_pool.sort("DATE_ACQUIRED", False)
            .sort("CLOUD_COVER")
            .sort("_doy_diff")
            .sort("_year_diff")
            .first()
        )

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

    invalid = scale.lt(MIN_SCALE).Or(scale.gt(MAX_SCALE))
    scale = scale.where(invalid, scale2)
    offset = offset.where(invalid, offset2)

    invalid2 = scale.lt(MIN_SCALE).Or(scale.gt(MAX_SCALE))
    scale = scale.where(invalid2, 1)
    offset = offset.where(invalid2, src_stats.select(".*mean").subtract(fill_stats.select(".*mean")))

    count = common.reduceNeighborhood(ee.Reducer.count(), kernel, None, True, "boxcar")
    scaled = fill.multiply(scale).add(offset).updateMask(count.gte(min_neighbors))

    out = src.unmask(scaled, True)
    if fallback_to_donor:
        residual = src.mask().reduce(ee.Reducer.min()).Not().And(
            out.mask().reduce(ee.Reducer.min()).Not()
        )
        out = out.unmask(fill.updateMask(residual), True)
    return out


# ---------------------------------------------------------------------------
# Export handler
# ---------------------------------------------------------------------------
def load_fishnet_tile(tile_index: int) -> ee.Geometry:
    with FISHNET_PATH.open() as f:
        data = json.load(f)
    for feat in data["features"]:
        props = feat.get("properties", {})
        if int(props.get("_export_index", -1)) == tile_index:
            return ee.Geometry(feat["geometry"])
    raise ValueError(f"Fishnet tile not found: {tile_index}")


def apply_common_mask(img: ee.Image, mask_bands: list[str] | None = None) -> ee.Image:
    """Keep only pixels valid in all mask bands.

    Prevents false-color edge artifacts where some bands have data at scene
    boundaries and others are masked. Use optical bands as mask source for LE07
    exports so thermal-edge masks do not erase valid optical pixels.
    """
    mask_src = img.select(mask_bands) if mask_bands is not None else img
    return img.updateMask(mask_src.mask().reduce(ee.Reducer.min()))


def common_mask_bands(sensor: str) -> list[str]:
    if sensor == "LE07":
        return ["B1", "B2", "B3", "B4", "B5", "B7"]
    if sensor == "LT05":
        return ["B1", "B2", "B3", "B4", "B5", "B7"]
    return SENSOR_BANDS[sensor]


def prepare_export_image(img: ee.Image, sensor: str) -> ee.Image:
    """Preserve TOA dynamic range.

    Raw DN exports used uint8 historically, but TOA reflectance is float-scale and
    should not be cast to uint8 or it collapses to mostly 0/1 values.
    """
    if COLLECTIONS[sensor].endswith("_TOA"):
        return img.toFloat()
    return img.uint8()


def export(img: ee.Image, desc: str, folder: str, region: ee.Geometry | None = None):
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=desc,
        folder=folder,
        region=region,
        scale=30,
        maxPixels=1e9,
    )
    task.start()
    return task


# ---------------------------------------------------------------------------
# Dataset: gapfill (mixed or LE07-only)
# ---------------------------------------------------------------------------
def export_gapfill(
    scenes: list,
    folder: str,
    dataset: str,
    dry: bool,
    subset: set | None,
    tile_geom: ee.Geometry | None,
    tile_suffix: str,
    forced_fill_date: str | None,
    gapfill_kernel_size: int,
    gapfill_min_neighbors: int,
    gapfill_upscale: bool,
    gapfill_fallback_to_donor: bool,
    disable_common_mask: bool,
    disable_gapfill: bool,
):
    for p, r, y, m, d, sensor in scenes:
        pr = f"{p:03d}-{r:03d}"
        if subset and pr not in subset:
            continue

        ename = export_desc(sensor, p, r, y, m, d)
        full_path = ee_image_id(sensor, p, r, y, m, d)
        print(f"  {ename}{tile_suffix}  ({pr}, {sensor})")
        if dry:
            continue

        img = ee.Image(full_path)
        if needs_gapfill(y, m, d, sensor) and not disable_gapfill:
            # Check pre-SLC fill availability before attempting gapfill
            pr_filter = ee.Filter.eq("WRS_PATH", p).And(ee.Filter.eq("WRS_ROW", r))
            fill_pool = (
                ee.ImageCollection(COLLECTIONS["LE07"])
                .filter(pr_filter)
                .filterDate("1999-04-15", "2003-05-31")
                .filter(ee.Filter.lt("CLOUD_COVER", 10))
            )
            if fill_pool.size().getInfo() > 0:
                img = gapfill(
                    img,
                    forced_fill_date=forced_fill_date,
                    min_neighbors=gapfill_min_neighbors,
                    kernel_size=gapfill_kernel_size,
                    upscale=gapfill_upscale,
                    fallback_to_donor=gapfill_fallback_to_donor,
                )
            else:
                print(f"    no pre-SLC fill — exporting with stripes")
        img = img.select(SENSOR_BANDS[sensor])
        if not disable_common_mask:
            img = apply_common_mask(img, common_mask_bands(sensor))
        if tile_geom is not None:
            img = img.clip(tile_geom)
        img = prepare_export_image(img, sensor)
        export(img, f"{dataset}_{ename}{tile_suffix}", folder, region=tile_geom)


# ---------------------------------------------------------------------------
# Dataset: date-anchored median (LE07-only)
# ---------------------------------------------------------------------------
def export_median(
    scenes: list,
    folder: str,
    dataset: str,
    window_days: int,
    max_cloud: int,
    max_scenes: int,
    month_window: int | None,
    dry: bool,
    subset: set | None,
    tile_geom: ee.Geometry | None,
    tile_suffix: str,
):
    for p, r, y, m, d, sensor in scenes:
        pr = f"{p:03d}-{r:03d}"
        if subset and pr not in subset:
            continue

        anchor_str = f"{y:04d}{m:02d}{d:02d}"
        ename = f"le07_{p:03d}{r:03d}_{anchor_str}_c{max_cloud}"
        anchor_date = f"{y:04d}-{m:02d}-{d:02d}"

        from datetime import timedelta
        anchor_dt = datetime(y, m, d)
        start_dt = anchor_dt - timedelta(days=window_days)
        end_dt = anchor_dt + timedelta(days=window_days)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        def month_distance(img: ee.Image) -> ee.Number:
            img_month = ee.Number(ee.Date(img.get("system:time_start")).get("month"))
            anchor_month = ee.Number(m)
            diff = img_month.subtract(anchor_month).abs()
            return diff.min(ee.Number(12).subtract(diff))

        # Collect scenes within window, annotate with temporal distance
        coll = (
            ee.ImageCollection(COLLECTIONS["LE07"])
            .filter(ee.Filter.eq("WRS_PATH", p))
            .filter(ee.Filter.eq("WRS_ROW", r))
            .filterDate(start_str, end_str)
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
            .map(lambda img: img.set(
                "_tdiff",
                ee.Date(img.get("system:time_start"))
                .difference(ee.Date(anchor_date), "day").abs()
            ).set("_mdiff", month_distance(img)))
        )

        if month_window is not None:
            coll = coll.filter(ee.Filter.lte("_mdiff", month_window)).sort("_tdiff")
        else:
            coll = coll.sort("_tdiff")

        total = coll.size()
        use_n = ee.Number(max_scenes).min(total)
        limited = coll.limit(use_n)

        use_n_val = use_n.getInfo()
        month_note = f", month±{month_window}" if month_window is not None else ""
        print(f"  {ename}{tile_suffix}  ({pr}, ±{window_days}d{month_note})  {use_n_val} closest scenes")
        if dry:
            continue

        geom = limited.first().geometry()
        limited_masked = limited.map(
            lambda img: apply_common_mask(img.select(SENSOR_BANDS["LE07"]), common_mask_bands("LE07"))
        )
        composite = limited_masked.median().clip(geom)
        composite = apply_common_mask(composite, common_mask_bands("LE07"))
        if tile_geom is not None:
            composite = composite.clip(tile_geom)
        composite = prepare_export_image(composite, "LE07")
        export(composite, f"{dataset}_{ename}{tile_suffix}", folder, region=tile_geom)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_subset(s: str) -> set[str]:
    return set(x.strip() for x in s.split(",")) if s else set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["gapfill_mixed", "gapfill_le07", "median"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--subset", type=str,
                        help="Comma-separated path/rows, e.g. '133-040,133-041'")
    parser.add_argument("--window-days", type=int, default=180,
                        help="±days for median window (default: 180)")
    parser.add_argument("--max-cloud", type=int, default=10,
                        help="Max cloud cover % (median, default: 10)")
    parser.add_argument("--max-scenes", type=int, default=5,
                        help="Max closest scenes for median (default: 5)")
    parser.add_argument("--month-window", type=int, default=None,
                        help="Restrict median to scenes within ±N calendar months of anchor month")
    parser.add_argument("--tile-index", type=int, default=None,
                        help="Export only a specific fishnet tile index for faster debugging")
    parser.add_argument("--name-suffix", type=str, default="",
                        help="Extra suffix appended to export description, e.g. '_yrdoy'")
    parser.add_argument("--forced-fill-date", type=str, default=None,
                        help="Force gapfill donor date YYYY-MM-DD for LE07 gapfill debugging")
    parser.add_argument("--gapfill-kernel-size", type=int, default=5,
                        help="Gapfill kernel size in pixels (default: 5)")
    parser.add_argument("--disable-gapfill", action="store_true",
                        help="Export target scene without gapfilling for debugging")
    parser.add_argument("--gapfill-min-neighbors", type=int, default=64,
                        help="Gapfill minimum common neighbors (default: 64)")
    parser.add_argument("--gapfill-upscale", dest="gapfill_upscale", action="store_true",
                        help="Use reduced-resolution upscale path for gapfill (default)")
    parser.add_argument("--no-gapfill-upscale", dest="gapfill_upscale", action="store_false",
                        help="Use full-resolution local fit for gapfill")
    parser.add_argument("--gapfill-fallback-to-donor", action="store_true",
                        help="Fill residual holes with raw donor pixels after LLHM gapfill")
    parser.add_argument("--disable-common-mask", action="store_true",
                        help="Skip common-mask reduction for debug exports")

    parser.set_defaults(gapfill_upscale=True)
    args = parser.parse_args()

    subset = parse_subset(args.subset)
    dry = args.dry_run
    tile_geom = load_fishnet_tile(args.tile_index) if args.tile_index is not None else None
    tile_suffix = f"_tile{args.tile_index}" if args.tile_index is not None else ""
    tile_suffix += args.name_suffix

    if args.dataset == "gapfill_mixed":
        folder = "HKH_rebuild_gapfill_mixed"
        print(f"=== GAPFILL_MIXED → {folder} ===")
        export_gapfill(REPORT_SCENES, folder, "gapfill_mixed", dry, subset, tile_geom, tile_suffix, args.forced_fill_date, args.gapfill_kernel_size, args.gapfill_min_neighbors, args.gapfill_upscale, args.gapfill_fallback_to_donor, args.disable_common_mask, args.disable_gapfill)

    elif args.dataset == "gapfill_le07":
        folder = "HKH_rebuild_gapfill_le07"
        scenes = scenes_LE07_only()
        print(f"=== GAPFILL_LE07 → {folder} ===")
        export_gapfill(scenes, folder, "gapfill_le07", dry, subset, tile_geom, tile_suffix, args.forced_fill_date, args.gapfill_kernel_size, args.gapfill_min_neighbors, args.gapfill_upscale, args.gapfill_fallback_to_donor, args.disable_common_mask, args.disable_gapfill)

    elif args.dataset == "median":
        folder = "HKH_rebuild_median"
        dataset_name = (
            f"median_w{args.window_days}_c{args.max_cloud}_n{args.max_scenes}"
        )
        if args.month_window is not None:
            dataset_name += f"_m{args.month_window}"
        scenes = scenes_LE07_only()
        print(f"=== {dataset_name.upper()} → {folder} ===")
        month_msg = (
            f", month window ±{args.month_window}" if args.month_window is not None else ""
        )
        print(
            f"Window: ±{args.window_days}d, cloud <{args.max_cloud}%, max {args.max_scenes} closest scenes{month_msg}"
        )
        export_median(scenes, folder, dataset_name,
                      window_days=args.window_days,
                      max_cloud=args.max_cloud,
                      max_scenes=args.max_scenes,
                      month_window=args.month_window,
                      dry=dry, subset=subset,
                      tile_geom=tile_geom,
                      tile_suffix=tile_suffix)


if __name__ == "__main__":
    main()
