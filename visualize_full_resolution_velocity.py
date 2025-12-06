#!/usr/bin/env python3
"""
Visualize full-resolution velocity data from glacier dataset.

This script loads original velocity TIFF images and creates high-quality
visualizations showing v, vx, vy components at their native resolution.

Usage:
    python visualize_full_resolution_velocity.py --velocity-dir /path/to/velocity --output-dir /tmp/velocity_viz
    python visualize_full_resolution_velocity.py --velocity-dir /path/to/velocity --max-images 5
"""

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pathlib import Path
from typing import List, Tuple, Optional
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_velocity_tiff(tiff_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load velocity TIFF image and return data with metadata.
    
    Args:
        tiff_path: Path to velocity TIFF file
        
    Returns:
        Tuple of (velocity_data, metadata)
    """
    with rasterio.open(tiff_path) as src:
        # Read all bands (should be 4: v, vx, vy, mask)
        data = src.read()
        # Transpose to (H, W, C) format
        data = np.transpose(data, (1, 2, 0))
        
        # Get metadata
        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "descriptions": src.descriptions,
        }
        
        return data.astype(np.float32), metadata

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

def visualize_single_velocity(
    velocity_path: Path, 
    output_dir: Path, 
    max_velocity: float = 1000.0,
    max_size: Optional[int] = None
) -> bool:
    """
    Create comprehensive visualization of a single velocity TIFF image.
    
    Args:
        velocity_path: Path to velocity TIFF file
        output_dir: Output directory for visualizations
        max_velocity: Maximum velocity for color scaling (m/year)
        max_size: Maximum size for display (None for full resolution)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {velocity_path.name}")
        
        # Load velocity data
        velocity_data, metadata = load_velocity_tiff(velocity_path)
        
        # Check band count
        if velocity_data.shape[2] < 3:
            logger.warning(f"  Expected at least 3 bands, got {velocity_data.shape[2]}")
            return False
        
        # Extract components
        v_data = velocity_data[:, :, 0]  # Velocity magnitude
        vx_data = velocity_data[:, :, 1]  # X-component
        vy_data = velocity_data[:, :, 2]  # Y-component
        mask_data = velocity_data[:, :, 3] if velocity_data.shape[2] > 3 else None  # Mask
        
        # Resize for display if needed
        if max_size is not None:
            h, w = v_data.shape
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                # Simple nearest neighbor resize for speed
                import cv2
                v_data = cv2.resize(v_data, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                vx_data = cv2.resize(vx_data, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                vy_data = cv2.resize(vy_data, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if mask_data is not None:
                    mask_data = cv2.resize(mask_data, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Create colormaps
        cmap_v, cmap_comp = create_velocity_colormaps()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f"Full Resolution Velocity Analysis: {velocity_path.name}", fontsize=16, fontweight='bold')
        
        # Helper function for plotting
        def plot_velocity_axis(ax, data, cmap, title_str, vmin, vmax, cbar_label):
            # Handle NaN and no-data values
            data_plot = np.ma.masked_invalid(data)
            if mask_data is not None:
                data_plot = np.ma.masked_where(mask_data == 0, data_plot)
            
            im = ax.imshow(data_plot, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title_str, fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, rotation=270, labelpad=20)
            
            # Add statistics text
            valid_data = data[~np.isnan(data)]
            if mask_data is not None:
                valid_data = data[mask_data > 0]
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
        if mask_data is not None:
            mask_down = mask_data[::step, ::step]
            vx_down = np.where(mask_down > 0, vx_down, np.nan)
            vy_down = np.where(mask_down > 0, vy_down, np.nan)
            v_mag_down = np.where(mask_down > 0, v_mag_down, np.nan)
        
        quiver = ax_quiver.quiver(x_coords, y_coords, vx_down, vy_down, 
                                 v_mag_down, cmap='viridis', 
                                 scale=max_velocity*20, scale_units='xy',
                                 width=0.003, headwidth=3, headlength=4)
        
        ax_quiver.set_title('Velocity Vector Field', fontsize=12, fontweight='bold')
        ax_quiver.axis('off')
        
        # Add colorbar for vector magnitude
        cbar_quiver = plt.colorbar(quiver, ax=ax_quiver, fraction=0.046, pad=0.04)
        cbar_quiver.set_label('Vector Magnitude (m/year)', rotation=270, labelpad=20)
        
        # Add metadata text
        metadata_text = (
            f"Dimensions: {metadata['width']} × {metadata['height']} pixels\n"
            f"Bands: {metadata['count']}\n"
            f"Data Type: {metadata['dtype']}\n"
            f"CRS: {metadata['crs']}\n"
            f"Native Resolution: ~120m (ITS_LIVE)"
        )
        if metadata['descriptions']:
            metadata_text += f"\nBand Descriptions:\n"
            for i, desc in enumerate(metadata['descriptions'][:4]):  # First 4 bands
                metadata_text += f"  Band {i+1}: {desc}\n"
        
        fig.text(0.02, 0.02, metadata_text, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = output_dir / f"{velocity_path.stem}_velocity_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save individual component images
        v_path = output_dir / f"{velocity_path.stem}_v_magnitude.png"
        vx_path = output_dir / f"{velocity_path.stem}_vx_component.png"
        vy_path = output_dir / f"{velocity_path.stem}_vy_component.png"
        
        # Save individual components
        fig_v, ax_v = plt.subplots(1, 1, figsize=(10, 8))
        plot_velocity_axis(ax_v, v_data, cmap_v, 'Velocity Magnitude (v)', 0, max_velocity, 'Velocity (m/year)')
        plt.tight_layout()
        plt.savefig(v_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        fig_vx, ax_vx = plt.subplots(1, 1, figsize=(10, 8))
        plot_velocity_axis(ax_vx, vx_data, cmap_comp, 'X-Component Velocity (vx)', -max_velocity, max_velocity, 'Vx (m/year)')
        plt.tight_layout()
        plt.savefig(vx_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        fig_vy, ax_vy = plt.subplots(1, 1, figsize=(10, 8))
        plot_velocity_axis(ax_vy, vy_data, cmap_comp, 'Y-Component Velocity (vy)', -max_velocity, max_velocity, 'Vy (m/year)')
        plt.tight_layout()
        plt.savefig(vy_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {output_path.name}")
        logger.info(f"   Components: {v_path.name}, {vx_path.name}, {vy_path.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process {velocity_path.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Visualize full-resolution velocity data from glacier dataset"
    )
    parser.add_argument(
        "--velocity-dir",
        type=Path,
        required=True,
        help="Directory containing velocity TIFF images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/tmp/velocity_visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)"
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=1000.0,
        help="Maximum velocity for color scaling (m/year)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum display size (None for full resolution)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.tif",
        help="File pattern to match (default: *.tif)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find velocity TIFF files
    velocity_files = sorted(args.velocity_dir.glob(args.pattern))
    
    if args.max_images is not None:
        velocity_files = velocity_files[:args.max_images]
    
    logger.info(f"Found {len(velocity_files)} velocity TIFF files to process")
    
    if len(velocity_files) == 0:
        logger.error(f"No velocity TIFF files found in {args.velocity_dir} with pattern {args.pattern}")
        return
    
    # Process images
    successful = 0
    failed = 0
    
    logger.info("=" * 60)
    logger.info("STARTING FULL RESOLUTION VELOCITY VISUALIZATION")
    logger.info("=" * 60)
    
    for i, velocity_path in enumerate(velocity_files, 1):
        logger.info(f"Processing {i}/{len(velocity_files)}: {velocity_path.name}")
        
        if visualize_single_velocity(velocity_path, args.output_dir, args.max_velocity, args.max_size):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("VELOCITY VISUALIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images processed: {len(velocity_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)
    
    if successful > 0:
        logger.info(f"\n🎯 Full resolution velocity visualizations created successfully!")
        logger.info(f"📁 View images in: {args.output_dir}")
        logger.info(f"📸 Each image has 4 files:")
        logger.info(f"   - *_velocity_analysis.png (4-panel analysis)")
        logger.info(f"   - *_v_magnitude.png (velocity magnitude)")
        logger.info(f"   - *_vx_component.png (x-component)")
        logger.info(f"   - *_vy_component.png (y-component)")

if __name__ == "__main__":
    main()