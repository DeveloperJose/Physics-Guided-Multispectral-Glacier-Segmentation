import json
import logging
from pathlib import Path
import rasterio
from rasterio.features import rasterize
import numpy as np
import geopandas as gpd
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(message)s")

OLD_LANDSAT_DIR = Path("/home/devj/local-arch/data/HKH_raw/Landsat7_2005")
NEW_LANDSAT_DIR = Path("/home/devj/local-arch/data/HKH_raw/rebuild/Landsat7_C02_T1")
LABELS_PATH = Path("/home/devj/local-arch/data/HKH_raw/labels_fixed/HKH_CIDC_5basins_all.shp")

def main():
    logging.info("--- DEEP AUDIT: COVERAGE & LABEL RASTERIZATION ---")
    try:
        labels_gdf = gpd.read_file(LABELS_PATH)
    except Exception as e:
        logging.error(f"Failed to load labels: {e}")
        return

    tiles = [0, 1, 3, 24, 31, 46, 67, 68, 70, 90, 96, 131, 132, 135]
    
    for idx in tiles:
        img_name = f"image{idx}.tif"
        old_l = OLD_LANDSAT_DIR / img_name
        new_l = NEW_LANDSAT_DIR / img_name
        
        if not new_l.exists() or not old_l.exists():
            continue
            
        logging.info(f"\nTile {idx}:")
        
        with rasterio.open(old_l) as src_old, rasterio.open(new_l) as src_new:
            old_data = src_old.read()
            new_data = src_new.read()
            
            # 1. Per-band valid coverage
            # Assuming 0 is nodata for legacy and new
            new_band_valid = (new_data > 0).sum(axis=(1, 2)) / (new_data.shape[1] * new_data.shape[2]) * 100
            logging.info(f"  New Per-Band Valid %: {np.round(new_band_valid, 1)}")
            
            # All-band valid mask (pixels valid in ALL bands)
            new_all_valid_mask = (new_data > 0).all(axis=0)
            old_all_valid_mask = (old_data > 0).all(axis=0)
            
            new_all_cov = new_all_valid_mask.sum() / new_all_valid_mask.size * 100
            old_all_cov = old_all_valid_mask.sum() / old_all_valid_mask.size * 100
            logging.info(f"  All-Band Valid % -> Old: {old_all_cov:.1f}%, New: {new_all_cov:.1f}%")
            
            # 2. Rasterized Label Check
            tile_box = box(*src_new.bounds)
            if labels_gdf.crs != src_new.crs:
                # To accurately rasterize, project labels to the tile's CRS
                local_labels = labels_gdf.to_crs(src_new.crs)
            else:
                local_labels = labels_gdf
                
            intersecting = local_labels[local_labels.intersects(tile_box)]
            
            if len(intersecting) == 0:
                logging.info("  Labels: No intersecting glacier polygons.")
                continue
                
            # Rasterize CI
            ci_polys = intersecting[intersecting['Glaciers'] == 'Clean Ice']
            if len(ci_polys) > 0:
                ci_mask = rasterize(shapes=ci_polys.geometry, out_shape=src_new.shape, transform=src_new.transform, fill=0, default_value=1, dtype=np.uint8)
            else:
                ci_mask = np.zeros(src_new.shape, dtype=np.uint8)
                
            # Rasterize DCI
            dci_polys = intersecting[intersecting['Glaciers'] == 'Debris covered']
            if len(dci_polys) > 0:
                dci_mask = rasterize(shapes=dci_polys.geometry, out_shape=src_new.shape, transform=src_new.transform, fill=0, default_value=1, dtype=np.uint8)
            else:
                dci_mask = np.zeros(src_new.shape, dtype=np.uint8)
            
            ci_pixels = ci_mask.sum()
            dci_pixels = dci_mask.sum()
            logging.info(f"  Rasterized Label Pixels -> CI: {ci_pixels}, DCI: {dci_pixels}")
            
            if ci_pixels > 0:
                ci_valid_old = (old_all_valid_mask & ci_mask).sum() / ci_pixels * 100
                ci_valid_new = (new_all_valid_mask & ci_mask).sum() / ci_pixels * 100
                logging.info(f"  CI Pixel Valid Coverage -> Old: {ci_valid_old:.1f}%, New: {ci_valid_new:.1f}%")
                
            if dci_pixels > 0:
                dci_valid_old = (old_all_valid_mask & dci_mask).sum() / dci_pixels * 100
                dci_valid_new = (new_all_valid_mask & dci_mask).sum() / dci_pixels * 100
                logging.info(f"  DCI Pixel Valid Coverage -> Old: {dci_valid_old:.1f}%, New: {dci_valid_new:.1f}%")
                
            # 3. uint8 safety proxy check (look at min/max of new valid pixels)
            valid_pixels = new_data[:, new_all_valid_mask]
            if valid_pixels.size > 0:
                band_max = valid_pixels.max(axis=1)
                band_min = valid_pixels.min(axis=1)
                # Count how many are exactly 255, which implies clipping
                clipped = (valid_pixels == 255).sum(axis=1) / valid_pixels.shape[1] * 100
                logging.info(f"  New Data uint8 check -> Max per band: {band_max}")
                logging.info(f"  New Data uint8 check -> % Pixels at 255: {np.round(clipped, 2)}%")

if __name__ == '__main__':
    main()
