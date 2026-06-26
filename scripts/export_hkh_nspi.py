#!/usr/bin/env python3
"""Plan and submit LE07 raw scene exports for local NSPI.

This script is intentionally a planner/exporter only. It does not run NSPI.

Core contract for downstream local NSPI:
- fill target pixels from `target_slc_gap`, not from generic invalid pixels;
- choose donors by target-gap coverage and local radiometric similarity;
- export explicit QA/mask bands so mask logic is auditable.

Inputs:
- report-corrected targets are read from `scripts/export_hkh_scenes.py` by AST
  (avoids importing that module, which authenticates at import time);
- LE07-only targets are used; report LT05 rows use existing LE07 proxies.

Outputs under `google_earth_scripts/export_manifests/nspi_le07/`:
- target_summary.csv
- candidate_donors.csv
- selected_donors.csv
- rejected_donors.csv
- export_manifest.json

Commands:
  uv run python scripts/export_hkh_nspi.py inspect --target-id 148035
  uv run python scripts/export_hkh_nspi.py plan --subset 148035
  uv run python scripts/export_hkh_nspi.py submit --target-id 148035
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import ee

PROJECT = "hkh-glacier-mapping"
COLLECTION = "LANDSAT/LE07/C02/T1_TOA"
SLC_FAILURE = date(2003, 5, 31)
OPTICAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
STACK_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B7",
    "QA_PIXEL",
    "QA_RADSAT",
    "data_present",
    "clear_valid",
    "slc_gap",
]
DEFAULT_MANIFEST_DIR = Path("google_earth_scripts/export_manifests/nspi_le07")
DEFAULT_LOCAL_RAW_DIR = Path("/home/devj/local-arch/data/HKH_raw/HKH_nspi_raw_scenes")
DEFAULT_DRIVE_FOLDER = "HKH_nspi_raw_scenes"
EXPORT_SCENES_SCRIPT = Path("scripts/export_hkh_scenes.py")


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def date_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def compact_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def target_id(path: int, row: int) -> str:
    return f"{path:03d}{row:03d}"


def doy_delta(a: date, b: date) -> int:
    diff = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(diff, 365 - diff)


def ee_image_id(path: int, row: int, d: date) -> str:
    return f"{COLLECTION}/LE07_{path:03d}{row:03d}_{compact_date(d)}"


def get_literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"Could not find literal assignment {name} in {path}")


@dataclass(frozen=True)
class TargetScene:
    path: int
    row: int
    target_date: date
    report_sensor: str
    target_id: str
    ee_id: str


@dataclass
class Candidate:
    target: TargetScene
    donor_date: date
    ee_id: str
    image: ee.Image
    groups: list[str]
    cloud_cover: float | None
    is_pre_slc: bool
    is_slc_off: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    cover_img: ee.Image | None = None


def load_targets() -> list[TargetScene]:
    report_scenes = get_literal_assignment(EXPORT_SCENES_SCRIPT, "REPORT_SCENES")
    proxies = get_literal_assignment(EXPORT_SCENES_SCRIPT, "LE07_PROXIES")
    out: list[TargetScene] = []
    for p, r, y, m, d, sensor in report_scenes:
        if (p, r) in proxies:
            y, m, d = proxies[(p, r)]
            sensor = "LE07_PROXY_FOR_LT05"
        dt = date(y, m, d)
        out.append(
            TargetScene(
                path=p,
                row=r,
                target_date=dt,
                report_sensor=sensor,
                target_id=target_id(p, r),
                ee_id=ee_image_id(p, r, dt),
            )
        )
    return out


def bit_is_set(img: ee.Image, band: str, bit: int) -> ee.Image:
    return img.select(band).bitwiseAnd(1 << bit).neq(0)


def make_masks(img: ee.Image, mask_snow: bool) -> dict[str, ee.Image]:
    qa = img.select("QA_PIXEL")
    radsat = img.select("QA_RADSAT")
    optical = img.select(OPTICAL_BANDS)

    qa_fill = qa.bitwiseAnd(1 << 0).neq(0)
    dilated_cloud = qa.bitwiseAnd(1 << 1).neq(0)
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    cloud_shadow = qa.bitwiseAnd(1 << 4).neq(0)
    snow = qa.bitwiseAnd(1 << 5).neq(0)
    saturated = radsat.neq(0)

    optical_mask = optical.mask().reduce(ee.Reducer.min()).unmask(0).neq(0)
    data_present = optical_mask.And(qa_fill.Not()).rename("data_present")
    data_present0 = data_present.unmask(0)

    clear = data_present0.And(dilated_cloud.Not()).And(cloud.Not()).And(cloud_shadow.Not()).And(saturated.Not())
    if mask_snow:
        clear = clear.And(snow.Not())
    clear = clear.rename("clear_valid")

    footprint = ee.Image.constant(1).clip(img.geometry()).unmask(0).rename("footprint")
    slc_gap = footprint.And(data_present0.eq(0)).rename("slc_gap")

    return {
        "qa_fill": qa_fill.rename("qa_fill"),
        "dilated_cloud": dilated_cloud.rename("dilated_cloud"),
        "cloud": cloud.rename("cloud"),
        "cloud_shadow": cloud_shadow.rename("cloud_shadow"),
        "snow": snow.rename("snow"),
        "saturated": saturated.rename("saturated"),
        "data_present": data_present,
        "clear_valid": clear,
        "slc_gap": slc_gap,
    }


def nspi_stack(img: ee.Image, mask_snow: bool) -> ee.Image:
    masks = make_masks(img, mask_snow)
    optical = img.select(OPTICAL_BANDS).toFloat()
    qa = img.select(["QA_PIXEL", "QA_RADSAT"]).toFloat()
    return optical.addBands(qa).addBands(masks["data_present"].toFloat()).addBands(
        masks["clear_valid"].toFloat()
    ).addBands(masks["slc_gap"].toFloat()).select(STACK_BANDS)


def image_for_target(t: TargetScene) -> ee.Image:
    return ee.Image(t.ee_id)


def collection_for_target(t: TargetScene, max_cloud: float) -> ee.ImageCollection:
    return (
        ee.ImageCollection(COLLECTION)
        .filter(ee.Filter.eq("WRS_PATH", t.path))
        .filter(ee.Filter.eq("WRS_ROW", t.row))
        .filter(ee.Filter.lte("CLOUD_COVER", max_cloud))
    )


def candidate_groups(target_dt: date, donor_dt: date, close_days: int, seasonal_days: int) -> list[str]:
    groups: list[str] = []
    if abs((donor_dt - target_dt).days) <= close_days:
        groups.append("close_date")
    if doy_delta(target_dt, donor_dt) <= seasonal_days:
        groups.append("seasonal")
    if donor_dt <= SLC_FAILURE:
        groups.append("pre_slc")
    return groups


def get_candidate_metadata(
    t: TargetScene,
    max_cloud: float,
    close_days: int,
    seasonal_days: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    coll = collection_for_target(t, max_cloud)
    props = coll.select(["B1"]).aggregate_array("system:index").getInfo()
    rows: list[dict[str, Any]] = []
    for idx in props:
        # Expected LE07_148035_20061108.
        try:
            donor_dt = datetime.strptime(idx.split("_")[-1], "%Y%m%d").date()
        except ValueError:
            continue
        if donor_dt == t.target_date:
            continue
        groups = candidate_groups(t.target_date, donor_dt, close_days, seasonal_days)
        if not groups:
            continue
        image = ee.Image(f"{COLLECTION}/{idx}")
        info = image.toDictionary(["CLOUD_COVER", "DATE_ACQUIRED"]).getInfo()
        cloud = info.get("CLOUD_COVER")
        rows.append(
            {
                "system_index": idx,
                "donor_date": donor_dt,
                "ee_id": f"{COLLECTION}/{idx}",
                "groups": groups,
                "cloud_cover": cloud,
                "date_delta_days": abs((donor_dt - t.target_date).days),
                "doy_delta": doy_delta(t.target_date, donor_dt),
                "is_pre_slc": donor_dt <= SLC_FAILURE,
                "is_slc_off": donor_dt > SLC_FAILURE,
            }
        )
    # Keep broad but bounded: group richness, cloud, season/date.
    rows.sort(
        key=lambda r: (
            -len(r["groups"]),
            float(r["cloud_cover"] if r["cloud_cover"] is not None else 999),
            r["doy_delta"],
            r["date_delta_days"],
        )
    )
    return rows[:max_candidates]


def safe_number(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = d.get(key)
    if v is None:
        return default
    return float(v)


def compute_candidate_metrics(
    t: TargetScene,
    cand_meta: dict[str, Any],
    target_img: ee.Image,
    target_masks: dict[str, ee.Image],
    region: ee.Geometry,
    scale: int,
    mask_snow: bool,
    context_radius_m: int,
) -> Candidate:
    donor = ee.Image(cand_meta["ee_id"])
    donor_masks = make_masks(donor, mask_snow)

    target_gap = target_masks["slc_gap"]
    target_clear = target_masks["clear_valid"]
    donor_clear = donor_masks["clear_valid"]
    donor_data = donor_masks["data_present"]
    donor_snow = donor_masks["snow"]
    donor_cloud_shadow = donor_masks["cloud_shadow"]
    donor_saturated = donor_masks["saturated"]

    gap_cover_clear = target_gap.And(donor_clear).rename("donor_valid_gap_pixels")
    gap_cover_data = target_gap.And(donor_data).rename("donor_data_gap_pixels")
    context = target_gap.focal_max(radius=context_radius_m, units="meters")
    common = target_clear.And(donor_clear).And(target_gap.Not()).And(context).rename("common")

    tgt_opt = target_img.select(OPTICAL_BANDS)
    don_opt = donor.select(OPTICAL_BANDS)
    absdiff = tgt_opt.subtract(don_opt).abs().reduce(ee.Reducer.mean()).rename("spectral_mae")
    tgt_ndsi = tgt_opt.normalizedDifference(["B2", "B5"])
    don_ndsi = don_opt.normalizedDifference(["B2", "B5"])
    ndsi_mae = tgt_ndsi.subtract(don_ndsi).abs().rename("ndsi_mae")
    tgt_bright = tgt_opt.reduce(ee.Reducer.mean())
    don_bright = don_opt.reduce(ee.Reducer.mean())
    brightness_mae = tgt_bright.subtract(don_bright).abs().rename("brightness_mae")

    metrics_img = (
        target_gap.rename("target_gap_pixels")
        .addBands(gap_cover_clear)
        .addBands(gap_cover_data)
        .addBands(common.rename("common_valid_pixels"))
        .addBands(absdiff.updateMask(common))
        .addBands(ndsi_mae.updateMask(common))
        .addBands(brightness_mae.updateMask(common))
        .addBands(target_gap.And(donor_saturated).rename("saturated_gap_pixels"))
        .addBands(target_gap.And(donor_cloud_shadow).rename("cloud_shadow_gap_pixels"))
        .addBands(target_gap.And(donor_snow).rename("snow_gap_pixels"))
    )
    sum_bands = [
        "target_gap_pixels",
        "donor_valid_gap_pixels",
        "donor_data_gap_pixels",
        "common_valid_pixels",
        "saturated_gap_pixels",
        "cloud_shadow_gap_pixels",
        "snow_gap_pixels",
    ]
    mean_bands = ["spectral_mae", "ndsi_mae", "brightness_mae"]
    raw_sum = metrics_img.select(sum_bands).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e10,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    raw_mean = metrics_img.select(mean_bands).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        maxPixels=1e10,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    raw = {**raw_sum, **raw_mean}

    target_gap_pixels = safe_number(raw, "target_gap_pixels")
    donor_valid_gap_pixels = safe_number(raw, "donor_valid_gap_pixels")
    donor_data_gap_pixels = safe_number(raw, "donor_data_gap_pixels")
    common_valid_pixels = safe_number(raw, "common_valid_pixels")
    saturated_gap_pixels = safe_number(raw, "saturated_gap_pixels")
    cloud_shadow_gap_pixels = safe_number(raw, "cloud_shadow_gap_pixels")
    snow_gap_pixels = safe_number(raw, "snow_gap_pixels")
    denom = max(target_gap_pixels, 1.0)
    metrics = {
        "target_id": t.target_id,
        "target_date": date_str(t.target_date),
        "path": t.path,
        "row": t.row,
        "target_ee_id": t.ee_id,
        "donor_date": date_str(cand_meta["donor_date"]),
        "donor_ee_id": cand_meta["ee_id"],
        "groups": ";".join(cand_meta["groups"]),
        "cloud_cover": cand_meta["cloud_cover"],
        "is_pre_slc": cand_meta["is_pre_slc"],
        "is_slc_off": cand_meta["is_slc_off"],
        "target_gap_pixels": target_gap_pixels,
        "donor_valid_gap_pixels": donor_valid_gap_pixels,
        "donor_data_gap_pixels": donor_data_gap_pixels,
        "clear_valid_gap_coverage": donor_valid_gap_pixels / denom,
        "data_present_gap_coverage": donor_data_gap_pixels / denom,
        "common_valid_pixels": common_valid_pixels,
        "common_valid_fraction": common_valid_pixels / max(float(scale), 1.0),  # raw count retained; fraction approximate not used.
        "spectral_mae": safe_number(raw, "spectral_mae", default=999.0),
        "ndsi_mae": safe_number(raw, "ndsi_mae", default=999.0),
        "brightness_mae": safe_number(raw, "brightness_mae", default=999.0),
        "saturated_gap_fraction": saturated_gap_pixels / denom,
        "cloud_shadow_gap_fraction": cloud_shadow_gap_pixels / denom,
        "snow_gap_fraction": snow_gap_pixels / denom,
        "doy_delta": cand_meta["doy_delta"],
        "date_delta_days": cand_meta["date_delta_days"],
    }
    return Candidate(
        target=t,
        donor_date=cand_meta["donor_date"],
        ee_id=cand_meta["ee_id"],
        image=donor,
        groups=list(cand_meta["groups"]),
        cloud_cover=cand_meta["cloud_cover"],
        is_pre_slc=bool(cand_meta["is_pre_slc"]),
        is_slc_off=bool(cand_meta["is_slc_off"]),
        metrics=metrics,
        cover_img=gap_cover_clear,
    )


def reject_reason(c: Candidate, min_gap_coverage: float, min_common_pixels: int) -> str | None:
    m = c.metrics
    if m["target_gap_pixels"] <= 0:
        return "target_has_no_slc_gap_pixels"
    if m["clear_valid_gap_coverage"] < min_gap_coverage:
        return "low_target_gap_coverage"
    if m["common_valid_pixels"] < min_common_pixels:
        return "too_few_common_valid_pixels"
    if m["spectral_mae"] >= 999:
        return "no_similarity_metric"
    return None


def donor_sort_key(m: dict[str, Any], marginal: float) -> tuple[float, float, float, float, int, int]:
    # Python sorts ascending, so negate beneficial values.
    return (
        -marginal,
        -float(m["clear_valid_gap_coverage"]),
        float(m["spectral_mae"]),
        float(m["ndsi_mae"]),
        int(m["doy_delta"]),
        int(m["date_delta_days"]),
    )


def marginal_coverage(
    candidate_cover: ee.Image,
    covered: ee.Image,
    target_gap: ee.Image,
    target_gap_pixels: float,
    region: ee.Geometry,
    scale: int,
) -> tuple[float, float]:
    marginal_img = candidate_cover.And(covered.Not()).rename("marginal")
    raw = marginal_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e10,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    pix = safe_number(raw, "marginal")
    return pix, pix / max(target_gap_pixels, 1.0)


def greedy_select(
    candidates: list[Candidate],
    target_masks: dict[str, ee.Image],
    region: ee.Geometry,
    scale: int,
    max_donors: int,
    cumulative_goal: float,
    min_marginal: float,
    min_gap_coverage: float,
    min_common_pixels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not candidates:
        return [], []
    target_gap_pixels = max(float(candidates[0].metrics["target_gap_pixels"]), 1.0)
    covered = ee.Image.constant(0).rename("covered").toByte()
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    remaining = candidates.copy()
    cumulative = 0.0

    # First reject hard failures.
    survivors: list[Candidate] = []
    for c in remaining:
        rr = reject_reason(c, min_gap_coverage, min_common_pixels)
        if rr is not None:
            row = dict(c.metrics)
            row["reject_reason"] = rr
            rejected.append(row)
        else:
            survivors.append(c)
    remaining = survivors

    while cumulative < cumulative_goal and len(selected) < max_donors and remaining:
        scored: list[tuple[tuple[float, float, float, float, int, int], Candidate, float, float]] = []
        for c in remaining:
            pix, marg = marginal_coverage(c.cover_img, covered, target_masks["slc_gap"], target_gap_pixels, region, scale)
            scored.append((donor_sort_key(c.metrics, marg), c, pix, marg))
        scored.sort(key=lambda x: x[0])
        _, best, best_pix, best_marg = scored[0]
        if best_marg < min_marginal:
            for _, c, pix, marg in scored:
                row = dict(c.metrics)
                row["marginal_gap_pixels"] = pix
                row["marginal_gap_coverage"] = marg
                row["reject_reason"] = "low_marginal_gap_coverage"
                rejected.append(row)
            break
        rank = len(selected) + 1
        cumulative += best_marg
        covered = covered.Or(best.cover_img).rename("covered")
        row = dict(best.metrics)
        row["donor_rank"] = rank
        row["marginal_gap_pixels"] = best_pix
        row["marginal_gap_coverage"] = best_marg
        row["cumulative_gap_coverage_after_selection"] = min(cumulative, 1.0)
        row["decision_reason"] = "selected_by_greedy_gap_coverage_similarity"
        row["export_name"] = export_description(best.target, "donor", rank, best.donor_date)
        row["local_path"] = str(local_filename(best.target, "donor", rank, best.donor_date))
        selected.append(row)
        remaining = [c for c in remaining if c.ee_id != best.ee_id]

    for c in remaining:
        if any(r.get("donor_ee_id") == c.ee_id for r in rejected):
            continue
        row = dict(c.metrics)
        row["reject_reason"] = "rank_cap_or_cumulative_goal_met"
        rejected.append(row)
    return selected, rejected


def local_filename(t: TargetScene, role: str, rank: int | None = None, d: date | None = None) -> Path:
    pr = t.target_id
    if role == "target":
        name = f"le07_nspi_raw_{pr}_{compact_date(t.target_date)}_target.tif"
    else:
        assert rank is not None and d is not None
        name = f"le07_nspi_raw_{pr}_{compact_date(t.target_date)}_donor{rank:02d}_{compact_date(d)}.tif"
    return DEFAULT_LOCAL_RAW_DIR / name


def export_description(t: TargetScene, role: str, rank: int | None = None, d: date | None = None) -> str:
    return local_filename(t, role, rank, d).stem


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_target_by_id(tid: str) -> TargetScene:
    tid = tid.replace("-", "")
    for t in load_targets():
        if t.target_id == tid:
            return t
    raise ValueError(f"Unknown target id: {tid}")


def resolve_donor_for_target(t: TargetScene, donor_date_str: str) -> ee.Image:
    d = parse_date(donor_date_str)
    return ee.Image(ee_image_id(t.path, t.row, d))


def inspect_target(args: argparse.Namespace) -> None:
    ee.Initialize(project=PROJECT)
    t = resolve_target_by_id(args.target_id)
    img = image_for_target(t)
    masks = make_masks(img, args.mask_snow)
    region = img.geometry()
    counts = (
        ee.Image.constant(1)
        .rename("footprint")
        .clip(region)
        .addBands(masks["data_present"].rename("data_present"))
        .addBands(masks["clear_valid"].rename("clear_valid"))
        .addBands(masks["slc_gap"].rename("slc_gap"))
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=args.scale,
            maxPixels=1e10,
            bestEffort=True,
            tileScale=4,
        )
        .getInfo()
    )
    info = img.toDictionary([
        "system:index",
        "DATE_ACQUIRED",
        "CLOUD_COVER",
        "WRS_PATH",
        "WRS_ROW",
    ]).getInfo()
    summary = {
        "target_id": t.target_id,
        "target_date": date_str(t.target_date),
        "ee_id": t.ee_id,
        "report_sensor": t.report_sensor,
        "image_info": info,
        "scale": args.scale,
        "mask_snow": args.mask_snow,
        "counts": counts,
    }
    print(json.dumps(summary, indent=2))

    if args.export_target:
        out = nspi_stack(img, args.mask_snow).clip(region)
        desc = f"inspect_target_{t.target_id}_{compact_date(t.target_date)}"
        crs = img.select("B1").projection().crs().getInfo()
        task = ee.batch.Export.image.toDrive(
            image=out,
            description=desc,
            folder=args.drive_folder,
            crs=crs,
            region=region,
            scale=30,
            maxPixels=1e13,
        )
        task.start()
        print(json.dumps({"submitted_export": desc, "drive_folder": args.drive_folder, "crs": crs}, indent=2))


def inspect_donor(args: argparse.Namespace) -> None:
    ee.Initialize(project=PROJECT)
    t = resolve_target_by_id(args.target_id)
    d = parse_date(args.donor_date)
    img = resolve_donor_for_target(t, args.donor_date)
    target_img = image_for_target(t)
    target_masks = make_masks(target_img, args.mask_snow)
    masks = make_masks(img, args.mask_snow)
    region = target_img.geometry()
    target_gap = target_masks["slc_gap"]
    target_clear = target_masks["clear_valid"]
    donor_data = masks["data_present"]
    donor_clear = masks["clear_valid"]
    context = target_gap.focal_max(radius=args.context_radius_m, units="meters")
    common = target_clear.And(donor_clear).And(target_gap.Not()).And(context).rename("common")
    tgt_opt = target_img.select(OPTICAL_BANDS)
    don_opt = img.select(OPTICAL_BANDS)
    absdiff = tgt_opt.subtract(don_opt).abs().reduce(ee.Reducer.mean()).rename("spectral_mae")
    tgt_ndsi = tgt_opt.normalizedDifference(["B2", "B5"])
    don_ndsi = don_opt.normalizedDifference(["B2", "B5"])
    ndsi_mae = tgt_ndsi.subtract(don_ndsi).abs().rename("ndsi_mae")
    metrics_img = (
        ee.Image.constant(1).rename("footprint").clip(region)
        .addBands(masks["data_present"].rename("data_present"))
        .addBands(masks["clear_valid"].rename("clear_valid"))
        .addBands(masks["slc_gap"].rename("slc_gap"))
        .addBands(target_gap.rename("target_gap_pixels"))
        .addBands(target_gap.And(donor_data).rename("donor_data_gap_pixels"))
        .addBands(target_gap.And(donor_clear).rename("donor_clear_gap_pixels"))
        .addBands(common.rename("common_valid_pixels"))
        .addBands(absdiff.updateMask(common))
        .addBands(ndsi_mae.updateMask(common))
    )
    raw_sum = metrics_img.select([
        "footprint",
        "data_present",
        "clear_valid",
        "slc_gap",
        "target_gap_pixels",
        "donor_data_gap_pixels",
        "donor_clear_gap_pixels",
        "common_valid_pixels",
    ]).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=args.scale,
        maxPixels=1e10,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    raw_mean = metrics_img.select(["spectral_mae", "ndsi_mae"]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=args.scale,
        maxPixels=1e10,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    counts = {**raw_sum, **raw_mean}
    info = img.toDictionary([
        "system:index",
        "DATE_ACQUIRED",
        "CLOUD_COVER",
        "WRS_PATH",
        "WRS_ROW",
    ]).getInfo()
    target_gap_pixels = safe_number(counts, "target_gap_pixels")
    donor_data_gap_pixels = safe_number(counts, "donor_data_gap_pixels")
    donor_clear_gap_pixels = safe_number(counts, "donor_clear_gap_pixels")
    summary = {
        "target_id": t.target_id,
        "target_date": date_str(t.target_date),
        "donor_date": date_str(d),
        "ee_id": ee_image_id(t.path, t.row, d),
        "scale": args.scale,
        "mask_snow": args.mask_snow,
        "context_radius_m": args.context_radius_m,
        "image_info": info,
        "counts": counts,
        "coverage": {
            "target_gap_pixels": target_gap_pixels,
            "donor_data_gap_pixels": donor_data_gap_pixels,
            "donor_clear_gap_pixels": donor_clear_gap_pixels,
            "data_present_gap_coverage": donor_data_gap_pixels / max(target_gap_pixels, 1.0),
            "clear_valid_gap_coverage": donor_clear_gap_pixels / max(target_gap_pixels, 1.0),
            "common_valid_pixels": safe_number(counts, "common_valid_pixels"),
            "spectral_mae": safe_number(counts, "spectral_mae", 999.0),
            "ndsi_mae": safe_number(counts, "ndsi_mae", 999.0),
        },
    }
    print(json.dumps(summary, indent=2))
    if args.export_donor:
        out = nspi_stack(img, args.mask_snow).clip(region)
        desc = f"inspect_donor_{t.target_id}_{compact_date(t.target_date)}_{compact_date(d)}"
        crs = image_for_target(t).select("B1").projection().crs().getInfo()
        task = ee.batch.Export.image.toDrive(
            image=out,
            description=desc,
            folder=args.drive_folder,
            crs=crs,
            region=region,
            scale=30,
            maxPixels=1e13,
        )
        task.start()
        print(json.dumps({"submitted_export": desc, "drive_folder": args.drive_folder, "crs": crs}, indent=2))


def build_plan(args: argparse.Namespace) -> None:
    ee.Initialize(project=PROJECT)
    targets = load_targets()
    if args.subset:
        wanted = {s.replace("-", "") for s in args.subset.split(",")}
        targets = [t for t in targets if t.target_id in wanted or f"{t.path:03d}{t.row:03d}" in wanted]
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    export_tasks: list[dict[str, Any]] = []

    for idx, t in enumerate(targets, start=1):
        print(f"[{idx}/{len(targets)}] target {t.target_id} {date_str(t.target_date)}")
        target_img = image_for_target(t)
        target_masks = make_masks(target_img, args.mask_snow)
        region = target_img.geometry()
        metas = get_candidate_metadata(
            t,
            args.max_cloud,
            args.close_days,
            args.seasonal_days,
            args.max_candidates,
        )
        candidates: list[Candidate] = []
        for meta in metas:
            try:
                cand = compute_candidate_metrics(
                    t,
                    meta,
                    target_img,
                    target_masks,
                    region,
                    args.metrics_scale,
                    args.mask_snow,
                    args.context_radius_m,
                )
                candidates.append(cand)
                candidate_rows.append(cand.metrics)
            except Exception as exc:  # noqa: BLE001 - keep planning resilient
                row = {
                    "target_id": t.target_id,
                    "target_date": date_str(t.target_date),
                    "path": t.path,
                    "row": t.row,
                    "target_ee_id": t.ee_id,
                    "donor_date": date_str(meta["donor_date"]),
                    "donor_ee_id": meta["ee_id"],
                    "reject_reason": f"metric_error:{type(exc).__name__}:{exc}",
                }
                rejected_rows.append(row)
                print(f"  metric error {meta['ee_id']}: {exc}")
        selected, rejected = greedy_select(
            candidates,
            target_masks,
            region,
            args.metrics_scale,
            args.max_donors,
            args.cumulative_goal,
            args.min_marginal_coverage,
            args.min_gap_coverage,
            args.min_common_pixels,
        )
        selected_rows.extend(selected)
        rejected_rows.extend(rejected)
        target_gap_pixels = candidates[0].metrics["target_gap_pixels"] if candidates else 0
        best_single = max((c.metrics["clear_valid_gap_coverage"] for c in candidates), default=0.0)
        final_cov = selected[-1]["cumulative_gap_coverage_after_selection"] if selected else 0.0
        summary_rows.append(
            {
                "target_id": t.target_id,
                "target_date": date_str(t.target_date),
                "path": t.path,
                "row": t.row,
                "target_ee_id": t.ee_id,
                "target_gap_pixels": target_gap_pixels,
                "candidate_count": len(candidates),
                "selected_donor_count": len(selected),
                "final_cumulative_gap_coverage": final_cov,
                "best_single_donor_coverage": best_single,
                "has_coverage_warning": final_cov < args.cumulative_goal,
                "manifest_status": "planned",
            }
        )
        # Always export target once if any donors selected.
        if selected:
            export_tasks.append(
                {
                    "role": "target",
                    "target_id": t.target_id,
                    "target_date": date_str(t.target_date),
                    "path": t.path,
                    "row": t.row,
                    "image_ee_id": t.ee_id,
                    "target_ee_id": t.ee_id,
                    "export_name": export_description(t, "target"),
                    "drive_folder": args.drive_folder,
                    "local_path": str(local_filename(t, "target")),
                }
            )
        for row in selected:
            export_tasks.append(
                {
                    "role": "donor",
                    "target_id": t.target_id,
                    "target_date": date_str(t.target_date),
                    "path": t.path,
                    "row": t.row,
                    "donor_rank": row["donor_rank"],
                    "image_ee_id": row["donor_ee_id"],
                    "target_ee_id": t.ee_id,
                    "export_name": row["export_name"],
                    "drive_folder": args.drive_folder,
                    "local_path": row["local_path"],
                }
            )
        print(f"  candidates={len(candidates)} selected={len(selected)} final_cov={final_cov:.3f}")

    common_fields = [
        "target_id",
        "target_date",
        "path",
        "row",
        "target_ee_id",
        "donor_date",
        "donor_ee_id",
        "groups",
        "cloud_cover",
        "is_pre_slc",
        "is_slc_off",
        "target_gap_pixels",
        "donor_valid_gap_pixels",
        "donor_data_gap_pixels",
        "clear_valid_gap_coverage",
        "data_present_gap_coverage",
        "marginal_gap_pixels",
        "marginal_gap_coverage",
        "cumulative_gap_coverage_after_selection",
        "common_valid_pixels",
        "common_valid_fraction",
        "spectral_mae",
        "ndsi_mae",
        "brightness_mae",
        "saturated_gap_fraction",
        "cloud_shadow_gap_fraction",
        "snow_gap_fraction",
        "doy_delta",
        "date_delta_days",
        "donor_rank",
        "decision_reason",
        "reject_reason",
        "export_name",
        "local_path",
    ]
    write_csv(manifest_dir / "candidate_donors.csv", candidate_rows, common_fields)
    write_csv(manifest_dir / "selected_donors.csv", selected_rows, common_fields)
    write_csv(manifest_dir / "rejected_donors.csv", rejected_rows, common_fields)
    write_csv(
        manifest_dir / "target_summary.csv",
        summary_rows,
        [
            "target_id",
            "target_date",
            "path",
            "row",
            "target_ee_id",
            "target_gap_pixels",
            "candidate_count",
            "selected_donor_count",
            "final_cumulative_gap_coverage",
            "best_single_donor_coverage",
            "has_coverage_warning",
            "manifest_status",
        ],
    )
    manifest = {
        "collection": COLLECTION,
        "stack_bands": STACK_BANDS,
        "mask_snow": args.mask_snow,
        "metrics_scale": args.metrics_scale,
        "context_radius_m": args.context_radius_m,
        "tasks": export_tasks,
    }
    (manifest_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_dir}")
    print(f"targets={len(summary_rows)} selected_donor_rows={len(selected_rows)} export_tasks={len(export_tasks)}")


def submit_targets(args: argparse.Namespace) -> None:
    ee.Initialize(project=PROJECT)
    targets = load_targets()
    if args.subset:
        wanted = {s.replace("-", "") for s in args.subset.split(",")}
        targets = [t for t in targets if t.target_id in wanted]
    print(f"submitting {len(targets)} target exports")
    for t in targets:
        img = image_for_target(t)
        out = nspi_stack(img, args.mask_snow).clip(img.geometry())
        desc = export_description(t, "target")
        crs = img.select("B1").projection().crs().getInfo()
        task = ee.batch.Export.image.toDrive(
            image=out,
            description=desc,
            folder=args.drive_folder,
            crs=crs,
            region=img.geometry(),
            scale=30,
            maxPixels=1e13,
        )
        task.start()
        print(f"submitted {desc} -> {args.drive_folder} crs={crs}")


def submit_exports(args: argparse.Namespace) -> None:
    ee.Initialize(project=PROJECT)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]
    if args.target_id:
        wanted = args.target_id.replace("-", "")
        tasks = [t for t in tasks if t["target_id"] == wanted]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"submitting {len(tasks)} exports")
    for row in tasks:
        img = ee.Image(row["image_ee_id"])
        target = ee.Image(row["target_ee_id"])
        out = nspi_stack(img, bool(manifest.get("mask_snow", False))).clip(target.geometry())
        crs = target.select("B1").projection().crs().getInfo()
        task = ee.batch.Export.image.toDrive(
            image=out,
            description=row["export_name"],
            folder=row.get("drive_folder", DEFAULT_DRIVE_FOLDER),
            crs=crs,
            region=target.geometry(),
            scale=30,
            maxPixels=1e13,
        )
        task.start()
        print(f"submitted {row['export_name']} -> {row.get('drive_folder', DEFAULT_DRIVE_FOLDER)} crs={crs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect one target scene masks only")
    inspect.add_argument("--target-id", type=str, required=True)
    inspect.add_argument("--scale", type=int, default=120)
    inspect.add_argument("--mask-snow", action="store_true")
    inspect.add_argument("--export-target", action="store_true")
    inspect.add_argument("--drive-folder", type=str, default=DEFAULT_DRIVE_FOLDER)
    inspect.set_defaults(func=inspect_target)

    inspect_d = sub.add_parser("inspect-donor", help="Inspect one donor scene for a target")
    inspect_d.add_argument("--target-id", type=str, required=True)
    inspect_d.add_argument("--donor-date", type=str, required=True)
    inspect_d.add_argument("--scale", type=int, default=120)
    inspect_d.add_argument("--context-radius-m", type=int, default=450)
    inspect_d.add_argument("--mask-snow", action="store_true")
    inspect_d.add_argument("--export-donor", action="store_true")
    inspect_d.add_argument("--drive-folder", type=str, default=DEFAULT_DRIVE_FOLDER)
    inspect_d.set_defaults(func=inspect_donor)

    plan = sub.add_parser("plan", help="Compute donor metrics and write manifests; no export tasks")
    plan.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    plan.add_argument("--subset", type=str, default=None, help="Comma-separated target ids, e.g. 148035,144039")
    plan.add_argument("--max-cloud", type=float, default=80.0)
    plan.add_argument("--close-days", type=int, default=180)
    plan.add_argument("--seasonal-days", type=int, default=45)
    plan.add_argument("--max-candidates", type=int, default=30)
    plan.add_argument("--max-donors", type=int, default=5)
    plan.add_argument("--cumulative-goal", type=float, default=0.98)
    plan.add_argument("--min-gap-coverage", type=float, default=0.05)
    plan.add_argument("--min-marginal-coverage", type=float, default=0.01)
    plan.add_argument("--min-common-pixels", type=int, default=500)
    plan.add_argument("--metrics-scale", type=int, default=120)
    plan.add_argument("--context-radius-m", type=int, default=450)
    plan.add_argument("--mask-snow", action="store_true", help="Mask QA snow/ice from clear_valid; default keeps snow")
    plan.add_argument("--drive-folder", type=str, default=DEFAULT_DRIVE_FOLDER)
    plan.set_defaults(func=build_plan)

    submit_t = sub.add_parser("submit-targets", help="Submit exports for all target scenes only")
    submit_t.add_argument("--subset", type=str, default=None, help="Comma-separated target ids")
    submit_t.add_argument("--mask-snow", action="store_true")
    submit_t.add_argument("--drive-folder", type=str, default=DEFAULT_DRIVE_FOLDER)
    submit_t.set_defaults(func=submit_targets)

    submit = sub.add_parser("submit", help="Submit exports from existing export_manifest.json")
    submit.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_DIR / "export_manifest.json")
    submit.add_argument("--target-id", type=str, default=None)
    submit.add_argument("--limit", type=int, default=None)
    submit.set_defaults(func=submit_exports)

    args = parser.parse_args()
    ee.Authenticate(auth_mode="localhost")
    args.func(args)


if __name__ == "__main__":
    main()
