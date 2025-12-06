#!/usr/bin/env python3
"""
Visualize original high-quality ITS_LIVE velocity data before resampling.

This script loads the original ITS_LIVE velocity mosaics at their native resolution
(120m) and creates visualizations of v, vx, vy components before they get resampled
to match Landsat-7's 30m resolution.

Usage:
    python visualize_original_velocity.py --catalog catalog_v02.json --output-dir /tmp/velocity_viz
    python visualize_original_velocity.py --catalog catalog_v02.json --region "himalaya" --max-images 5
"""

import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import s3fs
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ITS_LIVE configuration
VELOCITY_YEARS = list(range(2002, 2009))  # 2002-2008 inclusive
VELOCITY_RESOLUTION = 120  # meters (native ITS_LIVE resolution)

def load_catalog(catalog_path: Path) -> List[Dict]:
    """Load the official ITS_LIVE datacube catalog."""
    with open(catalog_path) as f:
        catalog = json.load(f)

    datacubes = []
    for feat in catalog.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry")

        if not geom or not props.get("composite_zarr_url"):
            continue

        # Convert HTTP URL to S3 URL
        composite_url = props.get("composite_zarr_url", "")
        if composite_url.startswith("http://its-live-data.s3.amazonaws.com/"):
            s3_url = composite_url.replace(
                "http://its-live-data.s3.amazonaws.com/", "s3://its-live-data/"
            )
        elif composite_url.startswith("https://"):
            s3_url = composite_url.replace(
                "https://its-live-data.s3.amazonaws.com/", "s3://its-live-data/"
            )
        else:
            s3_url = composite_url

        datacubes.append(
            {
                "geometry": geom,
                "epsg": props.get("epsg"),
                "composite_url": s3_url,
                "zarr_url": props.get("zarr_url"),
                "coverage": props.get("roi_percent_coverage", 0),
            }
        )

    return datacubes


def load_itslive_mosaic(zarr_url: str) -> xr.Dataset:
    """Load ITS_LIVE velocity mosaic from S3."""
    s3 = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=zarr_url, s3=s3, check=False)
    ds = xr.open_zarr(store, consolidated=True)
    return ds


def extract_temporal_median(
    ds: xr.Dataset, years: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract velocity data for specified years and compute temporal median.
    
    Returns:
        Tuple of (v_median, vx_median, vy_median)
    """
    # Filter by years
    year_mask = ds.time.dt.year.isin(years)
    ds_years = ds.isel(time=year_mask)

    n_years = len(ds_years.time)
    if n_years == 0:
        raise ValueError(f"No data found for years {years}")

    # Compute temporal median (robust to outliers)
    v_median = ds_years["v"].median(dim="time").values
    vx_median = ds_years["vx"].median(dim="time").values
    vy_median = ds_years["vy"].median(dim="time").values

    return v_median, vx_median, vy_median


def create_velocity_colormaps():
    """Create custom colormaps for velocity visualization."""
    
    # Velocity magnitude (v) - blue to red through white
    colors_v = ['#0000FF', '#4040FF', '#8080FF', '#C0C0FF', '#FFFFFF', 
                '#FFC0C0', '#FF8080', '#FF4040', '#FF0000']
    n_bins = 256
    cmap_v = LinearSegmentedColormap.from_list('velocity', colors_v, N=n_bins)
    
    # Velocity components (vx, vy) - diverging colormap
    colors_comp = ['#0000FF', '#4040FF', '#8080FF', '#C0C0FF', '#FFFFFF',
                   '#FFC0C0', '#FF8080', '#FF4040', '#FF0000']
    cmap_comp = LinearSegmentedColormap.from_list('velocity_comp', colors_comp, N=n_bins)
    
    return cmap_v, cmap_comp


def visualize_velocity_components(
    v_data: np.ndarray,
    vx_data: np.ndarray, 
    vy_data: np.ndarray,
    title: str,
    output_path: Path,
    max_velocity: float = 1000.0  # m/year
):
    """
    Create visualization of velocity components.
    
    Args:
        v_data: Velocity magnitude data
        vx_data: X-component velocity data  
        vy_data: Y-component velocity data
        title: Title for the visualization
        output_path: Path to save the figure
        max_velocity: Maximum velocity for color scaling (m/year)
    """
    cmap_v, cmap_comp = create_velocity_colormaps()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Helper function for plotting
    def plot_velocity_axis(ax, data, cmap, title_str, vmin, vmax, cbar_label):
        # Handle NaN values
        data_plot = np.ma.masked_invalid(data)
        
        im = ax.imshow(data_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, rotation=270, labelpad=20)
        
        # Add statistics text
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            stats_text = (f'Mean: {np.mean(valid_data):.1f} m/yr\n'
                         f'Median: {np.median(valid_data):.1f} m/yr\n'
                         f'Max: {np.max(valid_data):.1f} m/yr\n'
                         f'Std: {np.std(valid_data):.1f} m/yr')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8), fontsize=9)
    
    # Plot velocity magnitude
    plot_velocity_axis(axes[0, 0], v_data, cmap_v, 
                      'Velocity Magnitude (v)', 0, max_velocity, 
                      'Velocity (m/year)')
    
    # Plot x-component
    plot_velocity_axis(axes[0, 1], vx_data, cmap_comp,
                      'X-Component Velocity (vx)', -max_velocity, max_velocity,
                      'Vx (m/year)')
    
    # Plot y-component  
    plot_velocity_axis(axes[1, 0], vy_data, cmap_comp,
                      'Y-Component Velocity (vy)', -max_velocity, max_velocity,
                      'Vy (m/year)')
    
    # Create velocity vector field (quiver plot)
    ax_quiver = axes[1, 1]
    
    # Downsample for quiver plot (every Nth pixel)
    step = max(1, min(v_data.shape) // 50)  # Aim for ~50x50 grid
    y_coords, x_coords = np.mgrid[0:v_data.shape[0]:step, 0:v_data.shape[1]:step]
    
    # Downsample velocity components
    vx_down = vx_data[::step, ::step]
    vy_down = vy_data[::step, ::step]
    v_mag_down = np.sqrt(vx_down**2 + vy_down**2)
    
    # Create magnitude background
    im_bg = ax_quiver.imshow(v_data, cmap='gray', alpha=0.3, vmin=0, vmax=max_velocity)
    
    # Add quiver plot
    quiver = ax_quiver.quiver(x_coords, y_coords, vx_down, vy_down, 
                             v_mag_down, cmap='viridis', 
                             scale=max_velocity*20, scale_units='xy',
                             width=0.003, headwidth=3, headlength=4)
    
    ax_quiver.set_title('Velocity Vector Field', fontsize=12, fontweight='bold')
    ax_quiver.axis('off')
    
    # Add colorbar for vector magnitude
    cbar_quiver = plt.colorbar(quiver, ax=ax_quiver, fraction=0.046, pad=0.04)
    cbar_quiver.set_label('Vector Magnitude (m/year)', rotation=270, labelpad=20)
    
    # Add resolution info
    resolution_text = f'Native Resolution: {VELOCITY_RESOLUTION}m\nData Shape: {v_data.shape}'
    fig.text(0.02, 0.02, resolution_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved velocity visualization: {output_path}")


def visualize_single_datacube(datacube: Dict, years: List[int], output_dir: Path):
    """Visualize velocity data from a single ITS_LIVE datacube."""
    
    logger.info(f"Processing datacube: {datacube['zarr_url']}")
    
    try:
        # Load the datacube
        ds = load_itslive_mosaic(datacube["composite_url"])
        
        # Extract temporal median
        v_median, vx_median, vy_median = extract_temporal_median(ds, years)
        
        # Create output filename
        datacube_name = datacube["zarr_url"].split('/')[-2]  # Extract folder name
        output_path = output_dir / f"{datacube_name}_velocity_components.png"
        
        # Create visualization
        title = (f"ITS_LIVE Velocity Data\n"
                f"Datacube: {datacube_name}\n"
                f"EPSG: {datacube['epsg']} | Years: {years[0]}-{years[-1]}")
        
        visualize_velocity_components(
            v_median, vx_median, vy_median, title, output_path
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to visualize datacube {datacube['zarr_url']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Visualize original high-quality ITS_LIVE velocity data"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default="catalog_v02.json",
        help="Path to ITS_LIVE catalog JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/tmp/velocity_visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--max-datacubes",
        type=int,
        default=5,
        help="Maximum number of datacubes to visualize"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2002,
        help="Start year for velocity median"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2008,
        help="End year for velocity median (inclusive)"
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=1000.0,
        help="Maximum velocity for color scaling (m/year)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Filter by region (partial match on datacube URL)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load catalog
    logger.info(f"Loading ITS_LIVE catalog from {args.catalog}...")
    catalog = load_catalog(args.catalog)
    logger.info(f"Loaded {len(catalog)} datacubes from catalog")
    
    # Filter by region if specified
    if args.region:
        region_lower = args.region.lower()
        catalog = [dc for dc in catalog if region_lower in dc['zarr_url'].lower()]
        logger.info(f"Filtered to {len(catalog)} datacubes matching region '{args.region}'")
    
    # Limit number of datacubes
    if len(catalog) > args.max_datacubes:
        catalog = catalog[:args.max_datacubes]
        logger.info(f"Processing first {args.max_datacubes} datacubes")
    
    # Prepare years
    years = list(range(args.start_year, args.end_year + 1))
    logger.info(f"Extracting velocity median for years {years}")
    
    # Process datacubes
    successful = 0
    failed = 0
    
    logger.info("=" * 60)
    logger.info("Starting velocity visualization...")
    logger.info("=" * 60)
    
    for i, datacube in enumerate(catalog, 1):
        logger.info(f"Processing datacube {i}/{len(catalog)}")
        
        if visualize_single_datacube(datacube, years, args.output_dir):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("VISUALIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total datacubes processed: {len(catalog)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()