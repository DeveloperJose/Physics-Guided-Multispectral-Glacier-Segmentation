#!/usr/bin/env python3
"""Check C02 availability for report-provenance IDs and fishnet coverage.

Usage:
    uv run python google_earth_scripts/check_c02_availability.py --mismatches
    uv run python google_earth_scripts/check_c02_availability.py --coverage
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import ee

PROJECT = "hkh-glacier-mapping"
FISHNET_PATH = Path("google_earth_scripts/hkh_fishnet.geojson")
EXPORTER_PATH = Path("google_earth_scripts/export_hkh_rebuild.py")

# (path, row, report_date_YMD, report_sensor, bibek_date_YMD)
MISMATCHES = [
    (133, 41, "20091107", "LE07", "20051112"),
    (141, 40, "20051112", "LT05", "20041219"),
    (141, 41, "20051112", "LT05", "20041203"),
    (142, 40, "20081212", "LE07", "20041226"),
    (144, 39, "20051211", "LE07", "20011013"),
    (150, 35, "20070913", "LE07", "20050916"),
]


def yyymmdd_to_date(s: str) -> datetime:
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))


def query_window(path: int, row: int, center_ymd: str, days: int,
                 collection: str = "LANDSAT/LE07/C02/T1") -> list[dict]:
    dt = yyymmdd_to_date(center_ymd)
    start = (dt - timedelta(days=days)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        coll = (
            ee.ImageCollection(collection)
            .filterMetadata("WRS_PATH", "equals", path)
            .filterMetadata("WRS_ROW", "equals", row)
            .filterDate(start, end)
        )
        infos = coll.toList(100).getInfo()
        results = []
        for img in infos:
            results.append({
                "id": img.get("id", ""),
                "date": img.get("properties", {}).get("DATE_ACQUIRED", ""),
                "cloud_cover": img.get("properties", {}).get("CLOUD_COVER", -1),
            })
        results.sort(key=lambda x: x["cloud_cover"])
        return results
    except Exception as e:
        print(f"    ERROR: {e}")
        return []


def check_mismatches():
    print("=" * 70)
    print("C02 AVAILABILITY FOR 6 MISMATCHED REPORT IDS")
    print("=" * 70)

    for path, row, report_ymd, report_sensor, bibek_ymd in MISMATCHES:
        pr = f"{path:03d}-{row:03d}"
        print(f"\n--- {pr} ---")
        print(f"  Bibek:   LE07_{path:03d}{row:03d}_{bibek_ymd}")
        print(f"  Report:  {report_sensor}_{path:03d}{row:03d}_{report_ymd}")

        if report_sensor == "LE07":
            # Exact date
            exact = query_window(path, row, report_ymd, 0)
            if exact:
                for r in exact:
                    print(f"  ✓ Exact: {r['id']}  cloud={r['cloud_cover']}%")
            else:
                print(f"  ✗ No T1 scene for report date {report_ymd}")
                exprec = query_window(path, row, report_ymd, 0,
                                      collection="LANDSAT/LE07/C02/T2")
                for r in exprec:
                    print(f"  ✓ T2: {r['id']}  cloud={r['cloud_cover']}%")

            # ±90d
            print(f"  Search ±90d C02 T1:")
            alt = query_window(path, row, report_ymd, 90)
            if alt:
                rc = yyymmdd_to_date(report_ymd)
                for a in alt[:5]:
                    ddist = abs(
                        datetime.strptime(a["date"], "%Y-%m-%d") - rc
                    ).days
                    print(f"    {a['id']}  date={a['date']}  cloud={a['cloud_cover']}%  (diff={ddist}d)")
            else:
                print(f"    No T1 in ±90d — checking T2:")
                alt2 = query_window(path, row, report_ymd, 90,
                                    collection="LANDSAT/LE07/C02/T2")
                for a in alt2[:5]:
                    print(f"    T2: {a['id']}  date={a['date']}  cloud={a['cloud_cover']}%")
        else:
            # LT05 report: check LE07 and LT05 availability
            print(f"  Report uses LT05 — check LE07 C02 T1 ±180d:")
            alt = query_window(path, row, report_ymd, 180)
            if alt:
                rc = yyymmdd_to_date(report_ymd)
                for a in alt[:5]:
                    ddist = abs(
                        datetime.strptime(a["date"], "%Y-%m-%d") - rc
                    ).days
                    print(f"    {a['id']}  date={a['date']}  cloud={a['cloud_cover']}%  dist={ddist}d")
            else:
                print(f"    No LE07 in ±180d")

            # Check LT05 C02 availability for report date
            print(f"  Check LT05 C02 T1 for report date:")
            lt5 = query_window(path, row, report_ymd, 0,
                              collection="LANDSAT/LT05/C02/T1")
            if lt5:
                for r in lt5:
                    print(f"    ✓ LT05: {r['id']}  cloud={r['cloud_cover']}%")
            else:
                # Search ±30d
                lt5_w = query_window(path, row, report_ymd, 30,
                                    collection="LANDSAT/LT05/C02/T1")
                if lt5_w:
                    rc = yyymmdd_to_date(report_ymd)
                    print(f"    No exact. ±30d:")
                    for a in lt5_w[:3]:
                        ddist = abs(datetime.strptime(a["date"], "%Y-%m-%d") - rc).days
                        print(f"      {a['id']}  date={a['date']}  cloud={a['cloud_cover']}%  dist={ddist}d")
                else:
                    print(f"    No LT05 found")


def check_coverage():
    """Check which fishnet tiles are at WRS path/rows not in Bibek's 41."""
    import re
    exp_text = Path(EXPORTER_PATH).read_text()
    bibek_ids = set(re.findall(r'"LE07_\d{6}_\d{8}"', exp_text))
    bibek_prs = set(b.strip('"').split("_")[1] for b in bibek_ids)
    print(f"Bibek path/rows: {len(bibek_prs)} — {sorted(bibek_prs)[:5]}...")

    with open(FISHNET_PATH) as f:
        fishnet = json.load(f)
    n = len(fishnet["features"])
    print(f"Fishnet tiles: {n}")

    # Check first 50 tiles for WRS path/row
    uncovered = []
    for i, feat in enumerate(fishnet["features"][:50]):
        props = feat.get("properties", {})
        tile_idx = props.get("tile_index", i)
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            cent_lon = sum(c[0] for c in coords) / len(coords)
            cent_lat = sum(c[1] for c in coords) / len(coords)
        else:
            continue

        try:
            pt = ee.Geometry.Point(cent_lon, cent_lat)
            img = (ee.ImageCollection("LANDSAT/LE07/C02/T1")
                   .filterBounds(pt)
                   .filterDate("2005-01-01", "2005-12-31")
                   .sort("CLOUD_COVER")
                   .first())
            info = img.getInfo()
            if info and info.get("properties"):
                p = info["properties"]
                wrs_path = str(p.get("WRS_PATH", "")).zfill(3)
                wrs_row = str(p.get("WRS_ROW", "")).zfill(3)
                wrs_pr = wrs_path + wrs_row
                if wrs_pr and wrs_pr not in bibek_prs:
                    uncovered.append((tile_idx, wrs_pr, cent_lat, cent_lon))
            else:
                pass  # No image at this location
        except Exception as e:
            print(f"  Tile {tile_idx}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  Checked {i+1}/{n}...")

    print(f"\nTiles with WRS path/row NOT in Bibek 41: {len(uncovered)}")
    for t in uncovered:
        print(f"  Tile {t[0]}: WRS {t[1]}  at ({t[2]:.3f}, {t[3]:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mismatches", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()

    ee.Authenticate(auth_mode="localhost")
    ee.Initialize(project=PROJECT)

    if args.mismatches:
        check_mismatches()
    if args.coverage:
        check_coverage()
    if not args.mismatches and not args.coverage:
        parser.print_help()


if __name__ == "__main__":
    main()
