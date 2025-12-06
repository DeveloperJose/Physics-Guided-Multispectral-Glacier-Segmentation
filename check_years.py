#!/usr/bin/env python3
"""
Quick script to check available years in ITS_LIVE datacubes.
"""

import json
import s3fs
import xarray as xr
from pathlib import Path

def load_catalog(catalog_path: Path):
    with open(catalog_path) as f:
        catalog = json.load(f)
    
    datacubes = []
    for feat in catalog.get("features", []):
        props = feat.get("properties", {})
        if props.get("composite_zarr_url"):
            datacubes.append({
                "url": props.get("composite_zarr_url"),
                "epsg": props.get("epsg")
            })
    return datacubes

def check_datacube_years(datacube):
    try:
        # Convert to S3 URL
        s3_url = datacube["url"].replace(
            "http://its-live-data.s3.amazonaws.com/", "s3://its-live-data/"
        )
        
        s3 = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=s3_url, s3=s3, check=False)
        ds = xr.open_zarr(store, consolidated=True)
        
        # Get available years
        years = sorted(ds.time.dt.year.unique().values)
        return years
    except Exception as e:
        return None

# Load catalog
catalog_path = Path("catalog_v02.json")
datacubes = load_catalog(catalog_path)

print(f"Checking years in first 5 datacubes...")
for i, dc in enumerate(datacubes[:5]):
    years = check_datacube_years(dc)
    if years:
        print(f"Datacube {i+1}: {dc['epsg']} - Years: {years}")
    else:
        print(f"Datacube {i+1}: {dc['epsg']} - Failed to load")