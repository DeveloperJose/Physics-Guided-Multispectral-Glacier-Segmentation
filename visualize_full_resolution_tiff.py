#!/usr/bin/env python3
"""
Visualize full-resolution TIFF images from your glacier dataset.

This script loads the original Landsat TIFF images and creates high-quality
visualizations showing the raw data before any processing or scaling.

Usage:
    python visualize_full_resolution_tiff.py --image-dir /path/to/Landsat7_2005 --output-dir /tmp/full_res_viz
    python visualize_full_resolution_tiff.py --image-dir /path/to/Landsat7_2005 --max-images 10
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

# Landsat band information
LANDSAT_BANDS = {
    1: {"name": "B1", "wavelength": "0.45-0.52 μm", "description": "Blue"},
    2: {"name": "B2", "wavelength": "0.52-0.60 μm", "description": "Green"},
    3: {"name": "B3", "wavelength": "0.63-0.69 μm", "description": "Red"},
    4: {"name": "B4", "wavelength": "0.76-0.90 μm", "description": "NIR"},
    5: {"name": "B5", "wavelength": "1.55-1.75 μm", "description": "SWIR 1"},
    6: {"name": "B6_VCID1", "wavelength": "10.4-12.5 μm", "description": "Thermal IR 1"},
    7: {"name": "B6_VCID2", "wavelength": "10.4-12.5 μm", "description": "Thermal IR 2"},
    8: {"name": "B7", "wavelength": "2.08-2.35 μm", "description": "SWIR 2"},
}

def load_tiff_image(tiff_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load TIFF image and return data with metadata.
    
    Args:
        tiff_path: Path to TIFF file
        
    Returns:
        Tuple of (image_data, metadata)
    """
    with rasterio.open(tiff_path) as src:
        # Read all bands
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
        }
        
        return data.astype(np.float32), metadata

def create_rgb_composite(image_data: np.ndarray, rgb_bands: List[int] = [3, 2, 1]) -> np.ndarray:
    """
    Create RGB composite from multispectral image.
    
    Args:
        image_data: Image data with shape (H, W, C)
        rgb_bands: List of band indices for RGB [R, G, B]
        
    Returns:
        RGB composite with shape (H, W, 3)
    """
    rgb = np.stack([image_data[:, :, i-1] for i in rgb_bands], axis=2)
    
    # Handle no-data values
    nodata_mask = np.any(rgb == 0, axis=2)
    
    # Apply percentile stretch for better visualization
    valid_pixels = rgb[~nodata_mask]
    if len(valid_pixels) > 0:
        p2, p98 = np.percentile(valid_pixels, [2, 98])
        rgb_stretched = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    else:
        rgb_stretched = rgb / np.max(rgb) if np.max(rgb) > 0 else rgb
    
    # Set no-data to black
    rgb_stretched[nodata_mask] = 0
    
    return rgb_stretched

def create_false_color_composite(image_data: np.ndarray) -> np.ndarray:
    """
    Create false color composite (NIR, Red, Green) for better glacier visualization.
    
    Args:
        image_data: Image data with shape (H, W, C)
        
    Returns:
        False color composite with shape (H, W, 3)
    """
    # NIR, Red, Green = Bands 4, 3, 2
    return create_rgb_composite(image_data, rgb_bands=[4, 3, 2])

def calculate_ndvi(image_data: np.ndarray) -> np.ndarray:
    """
    Calculate NDVI from image data.
    
    Args:
        image_data: Image data with shape (H, W, C)
        
    Returns:
        NDVI array with shape (H, W)
    """
    nir = image_data[:, :, 3]  # Band 4
    red = image_data[:, :, 2]  # Band 3
    
    ndvi = (nir - red) / (nir + red + 1e-8)  # Add small value to avoid division by zero
    return np.clip(ndvi, -1, 1)

def calculate_ndsi(image_data: np.ndarray) -> np.ndarray:
    """
    Calculate NDSI from image data (good for snow/ice detection).
    
    Args:
        image_data: Image data with shape (H, W, C)
        
    Returns:
        NDSI array with shape (H, W)
    """
    green = image_data[:, :, 1]  # Band 2
    swir = image_data[:, :, 6]  # Band 7
    
    ndsi = (green - swir) / (green + swir + 1e-8)
    return np.clip(ndsi, -1, 1)

def visualize_single_tiff(
    tiff_path: Path, 
    output_dir: Path, 
    max_size: Optional[int] = None
) -> bool:
    """
    Create comprehensive visualization of a single TIFF image.
    
    Args:
        tiff_path: Path to TIFF file
        output_dir: Output directory for visualizations
        max_size: Maximum size for display (None for full resolution)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {tiff_path.name}")
        
        # Load image
        image_data, metadata = load_tiff_image(tiff_path)
        
        # Resize for display if needed
        if max_size is not None:
            h, w = image_data.shape[:2]
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                # Simple nearest neighbor resize for speed
                import cv2
                image_data = cv2.resize(image_data, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Create visualizations
        rgb_true = create_rgb_composite(image_data)
        rgb_false = create_false_color_composite(image_data)
        ndvi = calculate_ndvi(image_data)
        ndsi = calculate_ndsi(image_data)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Full Resolution Landsat Analysis: {tiff_path.name}", fontsize=16, fontweight='bold')
        
        # True color RGB
        axes[0, 0].imshow(rgb_true)
        axes[0, 0].set_title('True Color (RGB)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # False color RGB
        axes[0, 1].imshow(rgb_false)
        axes[0, 1].set_title('False Color (NIR-R-G)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # NDVI
        ndvi_im = axes[0, 2].imshow(ndvi, cmap='RdYlGn', vmin=-0.5, vmax=0.8)
        axes[0, 2].set_title('NDVI', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(ndvi_im, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # NDSI
        ndsi_im = axes[1, 0].imshow(ndsi, cmap='Blues_r', vmin=-0.5, vmax=0.8)
        axes[1, 0].set_title('NDSI (Snow/Ice Index)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(ndsi_im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Individual bands (show NIR and SWIR)
        nir_band = image_data[:, :, 3]  # Band 4
        swir_band = image_data[:, :, 6]  # Band 7
        
        # NIR
        nir_im = axes[1, 1].imshow(nir_band, cmap='gray')
        axes[1, 1].set_title('NIR Band (B4)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(nir_im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # SWIR
        swir_im = axes[1, 2].imshow(swir_band, cmap='gray')
        axes[1, 2].set_title('SWIR Band (B7)', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(swir_im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # Add metadata text
        metadata_text = (
            f"Dimensions: {metadata['width']} × {metadata['height']} pixels\n"
            f"Bands: {metadata['count']}\n"
            f"Data Type: {metadata['dtype']}\n"
            f"CRS: {metadata['crs']}\n"
            f"Bounds: {metadata['bounds']}"
        )
        fig.text(0.02, 0.02, metadata_text, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = output_dir / f"{tiff_path.stem}_full_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save high-resolution RGB composites
        rgb_true_path = output_dir / f"{tiff_path.stem}_rgb_true.png"
        rgb_false_path = output_dir / f"{tiff_path.stem}_rgb_false.png"
        
        plt.imsave(rgb_true_path, rgb_true)
        plt.imsave(rgb_false_path, rgb_false)
        
        logger.info(f"✅ Saved: {output_path.name}")
        logger.info(f"   RGB true: {rgb_true_path.name}")
        logger.info(f"   RGB false: {rgb_false_path.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process {tiff_path.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Visualize full-resolution TIFF images from glacier dataset"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing TIFF images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/tmp/full_resolution_visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)"
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
    
    # Find TIFF files
    tiff_files = sorted(args.image_dir.glob(args.pattern))
    
    if args.max_images is not None:
        tiff_files = tiff_files[:args.max_images]
    
    logger.info(f"Found {len(tiff_files)} TIFF files to process")
    
    if len(tiff_files) == 0:
        logger.error(f"No TIFF files found in {args.image_dir} with pattern {args.pattern}")
        return
    
    # Process images
    successful = 0
    failed = 0
    
    logger.info("=" * 60)
    logger.info("STARTING FULL RESOLUTION VISUALIZATION")
    logger.info("=" * 60)
    
    for i, tiff_path in enumerate(tiff_files, 1):
        logger.info(f"Processing {i}/{len(tiff_files)}: {tiff_path.name}")
        
        if visualize_single_tiff(tiff_path, args.output_dir, args.max_size):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("VISUALIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images processed: {len(tiff_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)
    
    if successful > 0:
        logger.info(f"\n🎯 Full resolution visualizations created successfully!")
        logger.info(f"📁 View images in: {args.output_dir}")
        logger.info(f"📸 Each image has 3 files:")
        logger.info(f"   - *_full_analysis.png (6-panel analysis)")
        logger.info(f"   - *_rgb_true.png (true color RGB)")
        logger.info(f"   - *_rgb_false.png (false color NIR-R-G)")

if __name__ == "__main__":
    main()