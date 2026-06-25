#!/usr/bin/env python3
"""Export benchmark-faithful HKH rebuild tiles from Earth Engine.

Track-A policy:
- keep legacy benchmark labels unchanged (HKH_CIDC_5basins_all.shp)
- rebuild imagery deterministically with current EE datasets
- Landsat: LANDSAT/LE07/C02/T1 raw DN, 8 bands (B1-B7, B6_VCID1, B6_VCID2), uint8
- DEM: NASA/NASADEM_HGT/001 elevation+slope+aspect+curvature, float32
- no velocity in this pass
- target window 2002-2008, target date 2005-07-01
- cloud filter CLOUD_COVER <= 40, QA_PIXEL mask (fill/dilated cloud/cloud/shadow)
- scene ranking: abs(date-target days)+10*CLOUD_COVER+1e-12*system:time_start
- SLC-off: multi-scene qualityMosaic (no gapfill)
- per-tile CRS from tile_inventory.csv when available
- full metadata logging per tile and global

Default mode is dry-run. Pass --start to submit EE export tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import ee

PROJECT = "hkh-glacier-mapping"
FISHNET_PATH = Path("google_earth_scripts/hkh_fishnet.geojson")
DEFAULT_OUT_DIR = Path("google_earth_scripts/export_manifests/hkh_rebuild_sample")
TILE_INVENTORY_PATH = Path("google_earth_scripts/tile_inventory.csv")
LANDSAT_COLLECTION = "LANDSAT/LE07/C02/T1_TOA"
DEM_COLLECTION = "NASA/NASADEM_HGT/001"
LANDSAT_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6_VCID_1",
    "B6_VCID_2",
    "B7",
]
LANDSAT_EXPORT_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6_VCID1",
    "B6_VCID2",
    "B7",
]
DEM_EXPORT_BANDS = ["elevation", "slope", "aspect", "curvature"]
SLC_OFF_DATE = "2003-05-31"

# Bibek's 41 handselected Landsat 7 scene IDs (C01 origin, matched to C02 by path/row/date).
BIBEK_IDS = [
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

AUDIT_BAND_NAMES = ["valid_obs_count", "date_spread_days"]


@dataclass(frozen=True)
class ExportPolicy:
    scene_pool: str = "dynamic"
    landsat_collection: str = LANDSAT_COLLECTION
    dem_collection: str = DEM_COLLECTION
    start_date: str = "2002-01-01"
    end_date: str = "2008-12-31"
    target_date: str = "2005-07-01"
    max_cloud_cover: float = 40.0
    max_scenes_per_tile: int = 24
    start_month: int = 1
    end_month: int = 12
    scale_m: int = 30
    landsat_bands: tuple[str, ...] = tuple(LANDSAT_BANDS)
    landsat_export_bands: tuple[str, ...] = tuple(LANDSAT_EXPORT_BANDS)
    dem_export_bands: tuple[str, ...] = tuple(DEM_EXPORT_BANDS)
    audit_bands: tuple[str, ...] = tuple(AUDIT_BAND_NAMES)
    cloud_mask: str = "mask QA_PIXEL fill, dilated cloud, cloud, cloud shadow; keep snow/ice"
    slc_policy: str = (
        "use all 2002-2008 Landsat 7 C02 T1 scenes; post-2003-05-31 SLC-off "
        "gaps are handled by deterministic multi-scene quality mosaic, not C01 gapfill"
    )
    ranking: str = (
        "per tile: filter by bounds/date/cloud; add rank = "
        "abs(date-target days) + 10*CLOUD_COVER + 1e-12*system:time_start; "
        "qualityMosaic(-rank) chooses lowest-rank valid pixel"
    )


@dataclass(frozen=True)
class TileRecord:
    tile_index: int
    image_name: str
    object_id: Any
    fishnet_id: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or start EE exports for HKH rebuild sample tiles."
    )
    parser.add_argument("--project", default=PROJECT, help="Earth Engine project ID")
    parser.add_argument("--fishnet", type=Path, default=FISHNET_PATH)
    parser.add_argument(
        "--tile-inventory",
        type=Path,
        default=TILE_INVENTORY_PATH,
        help="Legacy tile inventory used to preserve per-tile export CRS when present.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--tiles",
        default="0,1,2,3,4,54,101,135,142,201",
        help="Comma-separated _export_index tile IDs, or 'all'.",
    )
    parser.add_argument("--drive-folder", default="HKH_rebuild")
    parser.add_argument(
        "--composite-mode",
        choices=("quality_mosaic", "best_scene", "median", "medoid"),
        default="quality_mosaic",
        help="Compositing strategy. quality_mosaic=per-pixel winner, best_scene=single best scene, median=per-pixel median, medoid=closest real observation to median spectral vector.",
    )
    parser.add_argument(
        "--scene-pool",
        choices=("dynamic", "bibek_ids", "bibek_anchored"),
        default="dynamic",
        help="Scene pool strategy. dynamic=per-tile C02 query (default), bibek_ids=from Bibek's 41 handselected IDs matched to C02, bibek_anchored=exact Bibek scenes with nearby same-path/row fill.",
    )
    parser.add_argument(
        "--products",
        choices=("landsat", "dem", "both"),
        default="both",
        help="Products to export. Useful for retrying failed Landsat tasks without duplicating DEM.",
    )
    parser.add_argument("--max-cloud-cover", type=float, default=40.0)
    parser.add_argument("--max-scenes-per-tile", type=int, default=24)
    parser.add_argument("--start-date", default="2002-01-01")
    parser.add_argument("--end-date", default="2008-12-31")
    parser.add_argument("--start-month", type=int, default=1, help="Start month for window sweep (1-12)")
    parser.add_argument("--end-month", type=int, default=12, help="End month for window sweep (1-12)")
    parser.add_argument("--target-date", default="2005-07-01")
    parser.add_argument("--scale", type=int, default=30)
    parser.add_argument("--prefix", default="", help="Prefix for exported file names (useful for sweeps)")
    parser.add_argument(
        "--start",
        action="store_true",
        help="Submit EE export tasks. Without this, only manifests are written.",
    )
    return parser.parse_args()


def load_fishnet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing fishnet: {path}")
    with path.open() as f:
        data = json.load(f)
    features = data.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"Fishnet has no features: {path}")
    return sorted(
        features,
        key=lambda feat: int(feat.get("properties", {}).get("_export_index", -1)),
    )


def load_tile_crs(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return {
            int(row["tile_index"]): row["crs"]
            for row in rows
            if row.get("tile_index") and row.get("crs")
        }


def parse_tiles(tile_text: str, features: list[dict[str, Any]]) -> list[int]:
    available = {
        int(feat["properties"]["_export_index"])
        for feat in features
        if "_export_index" in feat.get("properties", {})
    }
    if tile_text.lower() == "all":
        return sorted(available)
    requested = [int(part.strip()) for part in tile_text.split(",") if part.strip()]
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"Requested tiles not in fishnet: {missing}")
    return requested


def initialize(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run "
            "`uv run python google_earth_scripts/auth.py` first if needed."
        ) from exc


def feature_to_ee(feat: dict[str, Any]) -> ee.Feature:
    return ee.Feature(feat)


def tile_record(feat: dict[str, Any]) -> TileRecord:
    props = feat["properties"]
    tile_index = int(props["_export_index"])
    return TileRecord(
        tile_index=tile_index,
        image_name=f"image{tile_index}.tif",
        object_id=props.get("OBJECTID"),
        fishnet_id=props.get("Id"),
    )


def landsat_cloud_mask(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    fill = qa.bitwiseAnd(1 << 0).eq(0)
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    cloud_shadow = qa.bitwiseAnd(1 << 4).eq(0)
    mask = fill.And(dilated_cloud).And(cloud).And(cloud_shadow)
    return image.updateMask(mask)


def add_rank(image: ee.Image, target: ee.Date) -> ee.Image:
    date_delta = image.date().difference(target, "day").abs()
    cloud = ee.Number(image.get("CLOUD_COVER"))
    time_tiebreak = ee.Number(image.get("system:time_start")).multiply(1e-12)
    rank = date_delta.add(cloud.multiply(10)).add(time_tiebreak)
    quality = ee.Image.constant(rank.multiply(-1)).toFloat().rename("quality")
    return image.addBands(quality).set("rebuild_rank", rank)


def landsat_collection_for_tile(geometry: ee.Geometry, policy: ExportPolicy) -> ee.ImageCollection:
    """Build per-tile dynamic scene collection from C02."""
    target = ee.Date(policy.target_date)
    return (
        ee.ImageCollection(policy.landsat_collection)
        .filterBounds(geometry)
        .filterDate(policy.start_date, policy.end_date)
        .filter(ee.Filter.calendarRange(policy.start_month, policy.end_month, "month"))
        .filter(ee.Filter.lte("CLOUD_COVER", policy.max_cloud_cover))
        .map(landsat_cloud_mask)
        .map(lambda image: add_rank(image, target))
        .sort("rebuild_rank")
        .limit(policy.max_scenes_per_tile)
    )


def bibek_id_collection(policy: ExportPolicy, tag_bibek: bool = False) -> ee.ImageCollection:
    """Build ImageCollection from Bibek's 41 handselected IDs, matched to C02 by WRS path/row/date.

    If tag_bibek=True, scenes are tagged with is_bibek_scene=1 and get a rank bonus
    so they are preferred in qualityMosaic compositing.
    """
    target = ee.Date(policy.target_date)
    all_images = []
    for scene_id in BIBEK_IDS:
        parts = scene_id.split("_")
        wrs_path = int(parts[1][:3])
        wrs_row = int(parts[1][3:])
        ds = parts[2]  # YYYYMMDD
        start = f"{ds[:4]}-{ds[4:6]}-{ds[6:]}"
        coll = (
            ee.ImageCollection(LANDSAT_COLLECTION)
            .filter(ee.Filter.eq("WRS_PATH", wrs_path))
            .filter(ee.Filter.eq("WRS_ROW", wrs_row))
            .filterDate(start, ee.Date(start).advance(1, "day"))
        )
        image = coll.first()
        if tag_bibek and image:
            image = image.set("is_bibek_scene", 1).set("bibek_source_id", scene_id)
        all_images.append(image)
    col = ee.ImageCollection(all_images)
    target = ee.Date(policy.target_date)
    result = (
        col.map(landsat_cloud_mask)
        .map(lambda image: add_rank(image, target))
        .sort("rebuild_rank")
    )
    if policy.start_date and policy.end_date:
        result = result.filterDate(policy.start_date, policy.end_date)
    return result


def bibek_anchored_collection(policy: ExportPolicy) -> ee.ImageCollection:
    """Build collection anchored to Bibek's 41 IDs, with auxiliary same-path/row fill scenes.

    Exact Bibek scenes get a rank bonus so they are always preferred where valid.
    Auxiliary scenes fill in only where Bibek pixels are masked (cloud, shadow, SLC-off).
    """
    # Build exact Bibek scenes tagged with is_bibek_scene=1
    bibek = bibek_id_collection(policy, tag_bibek=True)
    return bibek


def landsat_mosaic(
    collection: ee.ImageCollection,
    geometry: ee.Geometry,
    mode: str = "quality_mosaic",
    target_date: str | None = None,
) -> ee.Image:
    if mode == "quality_mosaic":
        main = collection.qualityMosaic("quality")
    elif mode == "best_scene":
        main = collection.first()
    elif mode == "median":
        main = collection.median()
    elif mode == "medoid":
        # Medoid: compute median spectral vector, then pick the real observation
        # closest (minimum Euclidean distance) to that vector.
        median_ref = collection.select(LANDSAT_BANDS, LANDSAT_EXPORT_BANDS).median()

        def add_medoid_quality(img):
            bands = img.select(LANDSAT_BANDS, LANDSAT_EXPORT_BANDS)
            diff = bands.subtract(median_ref).pow(2)
            dist = diff.reduce(ee.Reducer.sum()).sqrt()
            quality = dist.multiply(-1).rename("quality")
            return img.addBands(quality, overwrite=True)

        with_mq = collection.map(add_medoid_quality)
        main = with_mq.qualityMosaic("quality")
    else:
        raise ValueError(f"Unknown composite mode: {mode}")

    main = main.select(LANDSAT_BANDS, LANDSAT_EXPORT_BANDS)

    # Temporal audit bands: valid observation count and date spread.
    if target_date is not None:
        target = ee.Date(target_date)

        def add_audit(img):
            doff = (
                ee.Image.constant(img.date().difference(target, "day"))
                .float()
                .rename("audit_date_offset")
            )
            # has_data = 1 where B1 is unmasked, 0 where masked.
            has_data = img.select("B1").mask().rename("audit_has_data")
            return img.addBands([doff, has_data])

        audit_col = collection.map(add_audit)

        valid_count = audit_col.select("audit_has_data").sum().rename("valid_obs_count")

        # Only include valid observations in min/max.
        masked_dates = audit_col.map(
            lambda img: img.select("audit_date_offset").updateMask(
                img.select("audit_has_data")
            )
        )
        min_dt = masked_dates.min().rename("min_date_offset")
        max_dt = masked_dates.max().rename("max_date_offset")
        date_spread = max_dt.subtract(min_dt).rename("date_spread_days")

        main = main.addBands([valid_count, date_spread])

    return main.clip(geometry).float()


def dem_image(geometry: ee.Geometry) -> ee.Image:
    elevation = ee.Image(DEM_COLLECTION).select("elevation")
    slope = ee.Terrain.slope(elevation).rename("slope")
    aspect = ee.Terrain.aspect(elevation).rename("aspect")
    curvature = ee.Terrain.slope(slope).rename("curvature")
    return elevation.addBands([slope, aspect, curvature]).clip(geometry).float()


def scene_metadata(collection: ee.ImageCollection) -> list[dict[str, Any]]:
    rows = collection.map(
        lambda image: ee.Feature(
            None,
            {
                "image_id": image.id(),
                "date": image.date().format("YYYY-MM-dd"),
                "cloud_cover": image.get("CLOUD_COVER"),
                "wrs_path": image.get("WRS_PATH"),
                "wrs_row": image.get("WRS_ROW"),
                "spacecraft_id": image.get("SPACECRAFT_ID"),
                "sensor_id": image.get("SENSOR_ID"),
                "system_time_start": image.get("system:time_start"),
                "rebuild_rank": image.get("rebuild_rank"),
            },
        )
    )
    return ee.FeatureCollection(rows).getInfo()["features"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_scene_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "tile_index",
        "image_name",
        "scene_pool",
        "image_id",
        "date",
        "cloud_cover",
        "wrs_path",
        "wrs_row",
        "spacecraft_id",
        "sensor_id",
        "system_time_start",
        "rebuild_rank",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def export_image(
    image: ee.Image,
    geometry: ee.Geometry,
    description: str,
    folder: str,
    scale: int,
    crs: str | None = None,
    file_name_prefix: str | None = None,
) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=file_name_prefix or description,
        region=geometry,
        scale=scale,
        crs=crs,
        maxPixels=318080701,
    )
    task.start()
    return task


def collection_for_tile(
    geometry: ee.Geometry,
    policy: ExportPolicy,
    bibek_master: ee.ImageCollection | None = None,
) -> tuple[ee.ImageCollection, str]:
    """Return (collection, pool_name) for the given tile."""
    if policy.scene_pool == "bibek_ids":
        if bibek_master is None:
            raise ValueError("bibek_ids mode requires a pre-built bibek_master collection")
        return bibek_master.filterBounds(geometry), "bibek"
    elif policy.scene_pool == "bibek_anchored":
        if bibek_master is None:
            raise ValueError("bibek_anchored mode requires a pre-built bibek_master collection")
        bibek = bibek_master.filterBounds(geometry)
        fill = landsat_collection_for_tile(geometry, policy)

        def bibek_rank_bonus(img):
            bonus = ee.Image.constant(
                ee.Number(img.get("is_bibek_scene", 0)).multiply(10000)
            ).toFloat().rename("rank_bonus")
            old_q = img.select("quality")
            new_q = old_q.add(bonus).rename("quality")
            return img.addBands(new_q, overwrite=True)

        bibek = bibek.map(bibek_rank_bonus)
        merged = bibek.merge(fill).sort("rebuild_rank")
        return merged, "bibek_anchored"
    else:
        return landsat_collection_for_tile(geometry, policy), "dynamic"


def main() -> None:
    args = parse_args()
    features = load_fishnet(args.fishnet)
    tile_crs = load_tile_crs(args.tile_inventory)
    tile_ids = parse_tiles(args.tiles, features)
    selected = [
        feat for feat in features if int(feat["properties"].get("_export_index")) in tile_ids
    ]
    selected.sort(key=lambda feat: tile_ids.index(int(feat["properties"]["_export_index"])))

    policy = ExportPolicy(
        scene_pool=args.scene_pool,
        start_date=args.start_date,
        end_date=args.end_date,
        start_month=args.start_month,
        end_month=args.end_month,
        target_date=args.target_date,
        max_cloud_cover=args.max_cloud_cover,
        max_scenes_per_tile=args.max_scenes_per_tile,
        scale_m=args.scale,
    )
    composite_mode = args.composite_mode
    scene_pool = args.scene_pool
    print(f"Composite mode: {composite_mode}   Scene pool: {scene_pool}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "export_policy.json",
        {
            **asdict(policy),
            "composite_mode": composite_mode,
            "created_utc_date": date.today().isoformat(),
            "fishnet": str(args.fishnet),
            "tile_inventory": str(args.tile_inventory),
            "tiles": tile_ids,
            "dry_run": not args.start,
        },
    )

    initialize(args.project)

    # Pre-build bibek master collection once if needed.
    bibek_master: ee.ImageCollection | None = None
    if scene_pool in ("bibek_ids", "bibek_anchored"):
        print("Building Bibek ID collection...")
        bibek_master = bibek_id_collection(policy, tag_bibek=(scene_pool == "bibek_anchored"))
        print(f"Bibek pool built: {len(BIBEK_IDS)} IDs")

    global_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    # Derive Drive folder names from args, not hardcoded.
    landsat_folder = f"{args.drive_folder}_Landsat7"
    dem_folder = f"{args.drive_folder}_DEM"

    for feat in selected:
        rec = tile_record(feat)
        ee_feature = feature_to_ee(feat)
        geometry = ee_feature.geometry()
        collection, pool_label = collection_for_tile(geometry, policy, bibek_master)
        scenes = scene_metadata(collection)
        export_crs = tile_crs.get(rec.tile_index)
        rows = []
        for scene in scenes:
            props = scene["properties"]
            row = {"tile_index": rec.tile_index, "image_name": rec.image_name, "scene_pool": pool_label, **props}
            rows.append(row)
            global_rows.append(row)

        tile_dir = args.out_dir / f"image{rec.tile_index}"
        write_json(
            tile_dir / "tile.geojson",
            {
                "type": "FeatureCollection",
                "features": [feat],
                "export_crs": export_crs,
            },
        )
        write_json(tile_dir / "scene_metadata.json", rows)
        write_scene_csv(tile_dir / "scene_metadata.csv", rows)

        if not rows:
            task_rows.append(
                {
                    "tile_index": rec.tile_index,
                    "image_name": rec.image_name,
                    "export_crs": export_crs,
                    "status": "skipped_no_landsat_scenes",
                }
            )
            continue

        products = ["landsat", "dem"] if args.products == "both" else [args.products]

        if args.start:
            if "landsat" in products:
                landsat = landsat_mosaic(collection, geometry, composite_mode, policy.target_date)
                desc = f"{args.prefix}landsat_image{rec.tile_index}"
                landsat_task = export_image(
                    landsat,
                    geometry,
                    desc,
                    landsat_folder,
                    policy.scale_m,
                    export_crs,
                    f"{args.prefix}image{rec.tile_index}",
                )
                task_rows.append(
                    {
                        "tile_index": rec.tile_index,
                        "image_name": rec.image_name,
                        "product": "landsat",
                        "task_id": landsat_task.id,
                        "export_crs": export_crs,
                        "status": "started",
                    }
                )
            if "dem" in products:
                dem = dem_image(geometry)
                desc = f"{args.prefix}dem_image{rec.tile_index}"
                dem_task = export_image(
                    dem,
                    geometry,
                    desc,
                    dem_folder,
                    policy.scale_m,
                    export_crs,
                    f"{args.prefix}image{rec.tile_index}",
                )
                task_rows.append(
                    {
                        "tile_index": rec.tile_index,
                        "image_name": rec.image_name,
                        "product": "dem",
                        "task_id": dem_task.id,
                        "export_crs": export_crs,
                        "status": "started",
                    }
                )
        else:
            for product in products:
                task_rows.append(
                    {
                        "tile_index": rec.tile_index,
                        "image_name": rec.image_name,
                        "product": product,
                        "export_crs": export_crs,
                        "status": "dry_run_not_started",
                    }
                )

    write_scene_csv(args.out_dir / "scene_metadata_all_tiles.csv", global_rows)
    write_json(args.out_dir / "task_manifest.json", task_rows)
    print(
        f"Prepared {len(selected)} tiles; {len(global_rows)} scene rows. "
        f"Pool: {scene_pool}. Mode: {composite_mode}. "
        f"Manifests: {args.out_dir}. start={args.start}"
    )


if __name__ == "__main__":
    main()
