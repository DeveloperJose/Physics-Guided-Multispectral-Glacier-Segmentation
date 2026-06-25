#!/usr/bin/env python3
"""Check WRS path/row coverage for fishnet tiles against Bibek's 41 IDs.

Usage:
    uv run python google_earth_scripts/check_fishnet_coverage.py --tile 0
    uv run python google_earth_scripts/check_fishnet_coverage.py --batch 0-10

Tests whether each fishnet tile falls within a WRS path/row that Bibek's
41 IDs cover. If not, that tile needs additional imagery sourcing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import ee

PROJECT = "hkh-glacier-mapping"
FISHNET_PATH = Path("google_earth_scripts/hkh_fishnet.geojson")
EXPORTER_PATH = Path("google_earth_scripts/export_hkh_rebuild.py")


def get_bibek_prs() -> set[str]:
    """Parse Bibek 41 path/rows from exporter."""
    text = Path(EXPORTER_PATH).read_text()
    ids = set(re.findall(r'"LE07_\d{6}_\d{8}"', text))
    return set(b.strip('"').split("_")[1] for b in ids)


def tile_centroid(feat: dict) -> tuple[float, float]:
    """Get centroid of a fishnet tile polygon."""
    geom = feat.get("geometry", {})
    if geom.get("type") == "Polygon":
        coords = geom["coordinates"][0]
        lon = sum(c[0] for c in coords) / len(coords)
        lat = sum(c[1] for c in coords) / len(coords)
        return lon, lat
    raise ValueError("Not a polygon")


def check_tile(tile_idx: int, feat: dict) -> dict:
    """Query GEE for WRS path/row at tile centroid."""
    try:
        lon, lat = tile_centroid(feat)
    except ValueError:
        return {"tile": tile_idx, "error": "no polygon"}

    try:
        pt = ee.Geometry.Point(lon, lat)
        img = (ee.ImageCollection("LANDSAT/LE07/C02/T1")
               .filterBounds(pt)
               .filterDate("2005-01-01", "2005-12-31")
               .sort("CLOUD_COVER")
               .first())
        info = img.getInfo()
        if info and info.get("properties"):
            p = info["properties"]
            wrs_pr = f"{p.get('WRS_PATH', 0):03d}{p.get('WRS_ROW', 0):03d}"
            cloud = p.get("CLOUD_COVER", -1)
            scene_id = info.get("id", "")
            return {
                "tile": tile_idx,
                "lon": round(lon, 4),
                "lat": round(lat, 4),
                "wrs_pr": wrs_pr,
                "cloud": cloud,
                "scene_id": scene_id,
            }
        else:
            return {"tile": tile_idx, "lon": round(lon, 4), "lat": round(lat, 4),
                    "wrs_pr": None, "cloud": None, "scene_id": None,
                    "note": "no image found at this location"}
    except Exception as e:
        return {"tile": tile_idx, "lon": round(lon, 4), "lat": round(lat, 4),
                "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Check fishnet WRS coverage")
    parser.add_argument("--tile", type=int, help="Single tile index to check")
    parser.add_argument("--batch", type=str, help="Tile range e.g. 0-10")
    args = parser.parse_args()

    if args.tile is None and args.batch is None:
        parser.print_help()
        return

    ee.Initialize(project=PROJECT)

    bibek_prs = get_bibek_prs()
    print(f"Bibek {len(bibek_prs)} path/rows: {sorted(bibek_prs)[:5]}...{sorted(bibek_prs)[-5:]}")

    with open(FISHNET_PATH) as f:
        fishnet = json.load(f)
    n = len(fishnet["features"])
    print(f"Fishnet tiles: {n}")

    if args.tile is not None:
        idx = args.tile
        if idx < 0 or idx >= n:
            print(f"Tile {idx} out of range (0-{n-1})")
            return
        feat = fishnet["features"][idx]
        result = check_tile(idx, feat)
        _print_result(result, bibek_prs)

    elif args.batch:
        parts = args.batch.split("-")
        start, end = int(parts[0]), int(parts[1])
        results = []
        for i in range(start, min(end + 1, n)):
            feat = fishnet["features"][i]
            result = check_tile(i, feat)
            _print_result(result, bibek_prs)
            results.append(result)
            if (i - start + 1) % 10 == 0:
                print(f"  --- {i-start+1}/{end-start+1} checked ---")

        # Summary
        covered = [r for r in results if r.get("wrs_pr") in bibek_prs]
        uncovered = [r for r in results if r.get("wrs_pr") and r["wrs_pr"] not in bibek_prs]
        no_data = [r for r in results if r.get("wrs_pr") is None]
        print(f"\n=== Batch {start}-{end} Summary ===")
        print(f"Covered by Bibek 41: {len(covered)}/{len(results)}")
        print(f"NOT covered: {len(uncovered)}/{len(results)}")
        if uncovered:
            for r in uncovered:
                print(f"  Tile {r['tile']}: WRS {r['wrs_pr']} at ({r['lat']}, {r['lon']}) best={r.get('scene_id','')}")
        if no_data:
            for r in no_data:
                lat = r.get("lat", "?")
                lon = r.get("lon", "?")
                print(f"  Tile {r['tile']}: no image at ({lat}, {lon})")


def _print_result(result: dict, bibek_prs: set[str]):
    tile = result["tile"]
    wrs = result.get("wrs_pr")
    lat = result.get("lat", "?")
    lon = result.get("lon", "?")
    note = result.get("note", "")
    if wrs is None:
        print(f"Tile {tile}: {note} at ({lat}, {lon})")
    elif wrs in bibek_prs:
        print(f"Tile {tile}: WRS {wrs} ✓ (covered by Bibek 41)")
    else:
        print(f"Tile {tile}: WRS {wrs} ✗ NOT in Bibek 41. Scene: {result.get('scene_id')} cloud={result.get('cloud')}%")


if __name__ == "__main__":
    main()
