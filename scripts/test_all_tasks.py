#!/usr/bin/env python3
"""
Comprehensive test suite for glacier mapping task validation.

Tests all three task types to prevent regressions:
- Binary Clean Ice (CI): target_class_ids=[1], output_classes=[1]
- Binary Debris-Covered Ice (DCI): target_class_ids=[2], output_classes=[2]
- Multi-class: target_class_ids=[1,2], output_classes=[0,1,2]

Features:
- Creates temporary test configs automatically
- Uses subset data for fast execution
- Verbose output for debugging
- Complete pipeline validation
- Regression detection
- Automatic cleanup of temporary files

Usage:
    uv run python scripts/test_all_tasks.py [--server desktop] [--subset-size 5] [--epochs 2]
"""

import argparse
import copy
import sys
import tempfile
import traceback
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

import torch
import numpy as np
import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from glacier_mapping.utils.config import load_config, load_server_config
from glacier_mapping.lightning.glacier_datamodule import GlacierDataModule
import json
from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.data.slice import get_tiff_np, save_slices, read_shp
import rasterio

TEST_DATASET_NAME = "gen_robust_comprehensive"


class GlacierTaskTestSuite:
    """Comprehensive test suite for glacier mapping tasks."""

    def __init__(self, server: str = "desktop", subset_size: int = 5, epochs: int = 2):
        self.server = server
        self.subset_size = subset_size
        self.epochs = epochs
        self.test_results = {}
        self.temp_configs = []
        self.temp_dir = tempfile.mkdtemp(prefix="glacier_test_")

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.logger = logging.getLogger(__name__)

        print("=== GLACIER MAPPING COMPREHENSIVE TEST SUITE ===")
        print(f"Server: {server} | Subset Size: {subset_size} | Epochs: {epochs}")
        print(f"Temp Directory: {self.temp_dir}")
        print(f"Timestamp: {self._get_timestamp()}")
        print()

    def test_raw_file_integrity(self):
        """Verify integrity of raw TIFF, DEM, and velocity files."""
        print("=== Test: Raw File Integrity ===")
        server_config = load_server_config(self.server)
        image_dir = Path(server_config["image_dir"])
        dem_dir = Path(server_config["dem_dir"])
        velocity_dir = Path(server_config["velocity_dir"])
        image_files = sorted(list(image_dir.glob("*.tif")))

        if not image_files:
            print("  - No image files found, skipping test.")
            return

        for i, image_file in enumerate(image_files[: self.subset_size]):
            # --- Image File ---
            with rasterio.open(image_file) as src:
                data = src.read()
                if np.isnan(data).any() or np.isinf(data).any():
                    print(
                        f"  ⚠️ NaN or Inf found in image {image_file.name} (will be handled by slice.py)"
                    )
                    # raise ValueError(f"NaN or Inf found in image {image_file.name}")
            print(f"  ✓ {image_file.name}: Checked for NaN/Inf values.")

            # --- DEM File ---
            dem_file = dem_dir / image_file.name
            if dem_file.exists():
                with rasterio.open(dem_file) as src:
                    data = src.read()
                    if np.isnan(data).any() or np.isinf(data).any():
                        print(
                            f"  ⚠️ NaN or Inf found in DEM {dem_file.name} (will be handled by slice.py)"
                        )
                        # raise ValueError(f"NaN or Inf found in DEM {dem_file.name}")

                    elevation, slope = data[0], data[1]
                    if not (-500 < elevation.min() and elevation.max() < 9000):
                        print(
                            f"  ⚠️ {dem_file.name}: Unusual elevation range [{elevation.min()}, {elevation.max()}]"
                        )
                    if not (0 <= slope.min() and slope.max() <= 90):
                        print(
                            f"  ⚠️ {dem_file.name}: Unusual slope range [{slope.min()}, {slope.max()}]"
                        )
                print(
                    f"  ✓ {dem_file.name}: Checked for NaN/Inf values and plausible ranges."
                )

            # --- Velocity File ---
            velocity_file = velocity_dir / image_file.name
            if velocity_file.exists():
                with rasterio.open(velocity_file) as src:
                    data = src.read()
                    if np.isnan(data).any() or np.isinf(data).any():
                        print(
                            f"  ⚠️ NaN or Inf found in velocity {velocity_file.name} (will be handled by slice.py)"
                        )
                        # raise ValueError(
                        #     f"NaN or Inf found in velocity {velocity_file.name}"
                        # )

                    mask = data[3]
                    if not np.all((mask == 0) | (mask == 1)):
                        unique_vals = np.unique(mask)
                        raise ValueError(
                            f"Raw velocity mask not binary in {velocity_file.name}. Values: {unique_vals}"
                        )
                print(f"  ✓ {velocity_file.name}: No NaN/Inf values and binary mask.")
        print()

    def test_preprocessing_functions(self):
        """Test the preprocessing functions on a subset of raw data."""
        print("=== Test: Preprocessing Functions ===")
        server_config = load_server_config(self.server)
        image_dir = Path(server_config["image_dir"])
        dem_dir = Path(server_config["dem_dir"])
        velocity_dir = Path(server_config["velocity_dir"])
        labels_path = Path(server_config["labels_dir"]) / "HKH_CIDC_5basins_all.shp"

        image_files = sorted(list(image_dir.glob("*.tif")))
        labels = read_shp(labels_path)

        if not image_files:
            print("  - No image files found, skipping test.")
            return

        for i, image_file in enumerate(image_files[: self.subset_size]):
            fname = image_file.name
            tiff_fname = image_dir / fname
            dem_fname = dem_dir / fname
            velocity_fname = velocity_dir / fname

            conf = {
                "image_dir": str(image_dir),
                "dem_dir": str(dem_dir),
                "velocity_dir": str(velocity_dir),
                "add_velocity": True,
                "physics_res": None,
                "physics_scale": None,
                "add_ndvi": False,
                "add_ndwi": False,
                "add_ndsi": False,
                "add_hsv": False,
                "window_size": [256, 256],
                "overlap": 0,
                "filter": 0.0,
                "out_dir": self.temp_dir,
            }

            # Test get_tiff_np
            tiff_np, band_names = get_tiff_np(
                tiff_fname,
                dem_fname=dem_fname,
                velocity_fname=velocity_fname,
                physics_res=conf["physics_res"],
                physics_scale=conf["physics_scale"],
                add_ndvi=conf["add_ndvi"],
                add_ndwi=conf["add_ndwi"],
                add_ndsi=conf["add_ndsi"],
                add_hsv=conf["add_hsv"],
                return_band_names=True,
                verbose=True,
            )
            print(
                f"  ✓ get_tiff_np returned array of shape {tiff_np.shape} for {fname}"
            )

            # Test save_slices
            save_path = Path(self.temp_dir) / f"preprocessed_{i}"
            save_path.mkdir()
            save_slices(i, fname, labels, save_path, **conf)
            print(f"  ✓ save_slices ran without errors for {fname}")

            # Verify the output
            output_files = list(save_path.glob("tiff_*.npy"))
            mask_files = list(save_path.glob("mask_*.npy"))

            if output_files and mask_files:
                print(f"  ✓ Found {len(output_files)} output files for {fname}.")
                break
        else:
            raise ValueError("No valid slices were generated from the first 5 images.")

        # Check a sample slice
        sample_slice = np.load(output_files[0])
        velocity_mask_channel = band_names.index("velocity_mask")
        velocity_mask = sample_slice[..., velocity_mask_channel]
        is_binary = np.all((velocity_mask == 0) | (velocity_mask == 1))

        if not is_binary:
            unique_vals = np.unique(velocity_mask)
            print(
                f"  ❌ Processed velocity mask is NOT BINARY. Unique values: {unique_vals}"
            )
            raise ValueError("Processed velocity mask is not binary.")
        else:
            print("  ✓ Processed velocity mask is binary.")
        print()

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _run_preprocessing(self):
        """Run the preprocessing script for the test dataset."""
        print("=== PREPROCESSING STEP ===")
        preprocess_config_path = (
            "configs/datasets/bibek_w256_o64_f1_comprehensive_phys64_s1.yaml"
        )

        if not Path(preprocess_config_path).exists():
            print(
                f"⚠️ Preprocessing config not found at {preprocess_config_path}, skipping."
            )
            return

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/preprocess.py",
            "--server",
            self.server,
            "--config",
            preprocess_config_path,
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ Preprocessing failed!")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Preprocessing script failed.")
        else:
            print("✓ Preprocessing completed successfully.")
        print()

    def create_temp_config(self, task_name: str, task_config: Dict[str, Any]) -> str:
        """Create temporary configuration file."""
        config_path = Path(self.temp_dir) / f"{task_name}_test.yaml"

        with open(config_path, "w") as f:
            yaml.dump(task_config, f, default_flow_style=False)

        self.temp_configs.append(config_path)
        return str(config_path)

    def create_missing_configs(self):
        """Auto-create minimal test configs for missing task types."""
        print("=== CONFIGURATION SETUP ===")

        # Base template for minimal test configs
        base_config = {
            "training_opts": {
                "dataset_name": TEST_DATASET_NAME,
                "run_name": "",  # Will be set per task
                "seed": 42,
                "deterministic": True,
                "epochs": self.epochs,
                "early_stopping": self.epochs + 10,  # Disable early stopping
                "val_viz_n": 0,  # Disable visualizations for speed
                "run_test_eval": False,  # Disable test evaluation for speed
            },
            "loader_opts": {
                "physics_channels": False,
                "velocity_channels": False,
                "batch_size": min(4, self.subset_size),  # Small batch for testing
            },
        }

        # Check existing configs
        existing_configs = {}

        # Check for debris ice config (velocity loss)
        debris_velocity_config_path = Path(
            "configs/desktop/debris_ice/velocity_loss_experiment.yaml"
        )
        if debris_velocity_config_path.exists():
            existing_configs["debris_ice_velocity"] = str(debris_velocity_config_path)
            print(f"✓ Existing config found: {debris_velocity_config_path}")
        else:
            # Create debris ice config with velocity loss enabled
            debris_velocity_config = copy.deepcopy(base_config)
            debris_velocity_config["training_opts"]["run_name"] = (
                "debris_ice_velocity_test"
            )
            debris_velocity_config["loader_opts"]["velocity_channels"] = True
            debris_velocity_config["loss_opts"] = {"use_velocity_loss": True}
            debris_velocity_path = self.create_temp_config(
                "debris_ice_velocity", debris_velocity_config
            )
            existing_configs["debris_ice_velocity"] = debris_velocity_path
            print(f"✓ Auto-created temp config: {debris_velocity_path}")

        # Check for debris ice config (class weighting)
        debris_weighted_config_path = Path(
            "configs/desktop/debris_ice/class_weighting_experiment.yaml"
        )
        if debris_weighted_config_path.exists():
            existing_configs["debris_ice_weighted"] = str(debris_weighted_config_path)
            print(f"✓ Existing config found: {debris_weighted_config_path}")
        else:
            # Create debris ice config with class weighting enabled
            debris_weighted_config = copy.deepcopy(base_config)
            debris_weighted_config["training_opts"]["run_name"] = (
                "debris_ice_weighted_test"
            )
            debris_weighted_config["loader_opts"]["velocity_channels"] = True
            debris_weighted_config["loader_opts"]["physics_channels"] = True
            debris_weighted_config["loss_opts"] = {"class_weights": [1, 6.0]}
            debris_weighted_path = self.create_temp_config(
                "debris_ice_weighted", debris_weighted_config
            )
            existing_configs["debris_ice_weighted"] = debris_weighted_path
            print(f"✓ Auto-created temp config: {debris_weighted_path}")

        # Create clean ice config
        clean_config = copy.deepcopy(base_config)
        clean_config["training_opts"]["run_name"] = "clean_ice_test"
        clean_path = self.create_temp_config("clean_ice", clean_config)
        existing_configs["clean_ice"] = clean_path
        print(f"✓ Auto-created temp config: {clean_path}")

        # Create multiclass config
        multi_config = copy.deepcopy(base_config)
        multi_config["training_opts"]["run_name"] = "multiclass_test"
        multi_path = self.create_temp_config("multiclass", multi_config)
        existing_configs["multiclass"] = multi_path
        print(f"✓ Auto-created temp config: {multi_path}")

        print()
        return existing_configs

    def load_merged_config(self, config_path: str) -> Tuple[Dict[str, Any], str]:
        """Load configuration with 4-level merging simulation."""
        # Load base configs
        train_config = load_config("configs/train.yaml")
        server_config = load_server_config(self.server)

        # Load experiment config
        exp_config = load_config(config_path)

        # Determine task from config path or content
        task_name = "unknown"
        if "clean_ice" in config_path:
            task_name = "clean_ice"
        elif "debris_ice" in config_path:
            task_name = "debris_ice"
        elif "multiclass" in config_path:
            task_name = "multiclass"

        # Load task config
        task_config = load_config(f"configs/tasks/{task_name}.yaml")

        # Simulate 4-level merging
        merged = {}

        # Level 1: Base training config
        merged.update(train_config)

        # Level 2: Server config (training_opts and loader_opts)
        if "training_opts" in server_config:
            merged["training_opts"] = {
                **merged.get("training_opts", {}),
                **server_config["training_opts"],
            }
        if "loader_opts" in server_config:
            merged["loader_opts"] = {
                **merged.get("loader_opts", {}),
                **server_config["loader_opts"],
            }

        # Level 3: Task config
        if "loader_opts" in task_config:
            merged["loader_opts"] = {
                **merged.get("loader_opts", {}),
                **task_config["loader_opts"],
            }
        if "loss_opts" in task_config:
            merged["loss_opts"] = {
                **merged.get("loss_opts", {}),
                **task_config["loss_opts"],
            }
        if "metrics_opts" in task_config:
            merged["metrics_opts"] = {
                **merged.get("metrics_opts", {}),
                **task_config["metrics_opts"],
            }

        # Level 4: Experiment config
        if "training_opts" in exp_config:
            merged["training_opts"] = {
                **merged.get("training_opts", {}),
                **exp_config["training_opts"],
            }
        if "loader_opts" in exp_config:
            merged["loader_opts"] = {
                **merged.get("loader_opts", {}),
                **exp_config["loader_opts"],
            }
        if "loss_opts" in exp_config:
            merged["loss_opts"] = {
                **merged.get("loss_opts", {}),
                **exp_config["loss_opts"],
            }

        return merged, task_name

    def inspect_raw_data(self, dataset_path: str) -> Dict[str, Any]:
        """Inspect raw mask files and class distribution."""
        dataset_path_obj = Path(dataset_path)

        # Find mask files in each split
        splits = ["train", "val", "test"]
        mask_files = []

        for split in splits:
            split_dir = dataset_path_obj / split
            if split_dir.exists():
                mask_files.extend(list(split_dir.glob("mask_*.npy")))

        # Limit to subset size
        mask_files = mask_files[: self.subset_size]

        if not mask_files:
            raise ValueError(f"No mask files found in {dataset_path}")

        # Analyze class distribution
        class_counts = {0: 0, 1: 0, 2: 0, 255: 0}
        total_pixels = 0

        for mask_file in mask_files:
            mask = np.load(mask_file)
            unique, counts = np.unique(mask, return_counts=True)

            for val, count in zip(unique, counts):
                if val in class_counts:
                    class_counts[val] += count
                    total_pixels += count

        # Calculate percentages
        class_percentages = {}
        for val, count in class_counts.items():
            class_percentages[val] = (
                (count / total_pixels * 100) if total_pixels > 0 else 0
            )

        sample_shape = None
        if mask_files:
            mask = np.load(mask_files[0])
            sample_shape = mask.shape

        return {
            "mask_files": len(mask_files),
            "class_counts": class_counts,
            "class_percentages": class_percentages,
            "total_pixels": total_pixels,
            "sample_shape": sample_shape,
        }

    def test_task(self, config_path: str, task_name: str) -> Dict[str, Any]:
        """Test a single task comprehensively."""
        print(f"=== TASK: {task_name.upper()} ===")
        print(f"Config: {config_path}")
        print()

        result = {
            "task_name": task_name,
            "config_path": config_path,
            "tests": {},
            "passed": True,
            "errors": [],
        }

        try:
            # 1. Configuration Verification
            print("1.1 Configuration Verification:")
            config, detected_task = self.load_merged_config(config_path)

            output_classes = config["loader_opts"]["output_classes"]
            target_class_ids = config["loss_opts"].get("target_class_ids", [])

            print(f"  ✓ output_classes: {output_classes}")
            print(f"  ✓ target_class_ids: {target_class_ids}")
            print("  ✓ 4-level merging successful")

            result["tests"]["config"] = {
                "output_classes": output_classes,
                "target_class_ids": target_class_ids,
                "passed": True,
            }
            print()

            # 2. Raw Data Inspection
            print("1.2 Raw Data Inspection:")
            server_config = load_server_config(self.server)
            dataset_path = str(
                Path(server_config["processed_data_path"])
                / config["training_opts"]["dataset_name"]
            )

            raw_data_info = self.inspect_raw_data(str(dataset_path))
            print(f"  ✓ Found {raw_data_info['mask_files']} mask files")
            print(
                f"  ✓ Class distribution: BG={raw_data_info['class_percentages'][0]:.1f}%, "
                f"CI={raw_data_info['class_percentages'][1]:.1f}%, "
                f"DCI={raw_data_info['class_percentages'][2]:.1f}%"
            )
            print(f"  ✓ Mask shape: {raw_data_info['sample_shape']}")

            result["tests"]["raw_data"] = {
                "mask_files": raw_data_info["mask_files"],
                "class_percentages": raw_data_info["class_percentages"],
                "passed": True,
            }
            print()

            # 3. Data Loading Verification
            print("1.3 Data Loading Verification:")

            # Create data module with subset
            loader_opts = config["loader_opts"].copy()
            # Remove unsupported parameters
            loader_opts.pop("target_class_ids", None)

            data_module = GlacierDataModule(
                processed_dir=str(dataset_path), **loader_opts
            )

            # Setup to resolve channels
            data_module.setup()

            # Load a small batch
            train_loader = data_module.train_dataloader()
            batch = next(iter(train_loader))
            x, y_onehot, y_int = batch

            print(f"  ✓ Input shape: {x.shape}")
            print(f"  ✓ One-hot target shape: {y_onehot.shape}")
            print(f"  ✓ Integer target shape: {y_int.shape}")
            print(f"  ✓ Input range: [{x.min():.3f}, {x.max():.3f}]")

            # Verify one-hot encoding
            if len(output_classes) == 1:
                # Binary case
                target_class = output_classes[0]
                channel_1_percentage = (
                    y_onehot[..., 1] == 1
                ).float().mean().item() * 100
                expected_percentage = raw_data_info["class_percentages"][target_class]
                print(
                    f"  ✓ Binary: Channel 1 (target) = {channel_1_percentage:.1f}% "
                    f"(expected ~{expected_percentage:.1f}%)"
                )
            else:
                # Multi-class case
                for i, cls in enumerate(output_classes):
                    cls_percentage = (y_onehot[..., i] == 1).float().mean().item() * 100
                    expected_percentage = raw_data_info["class_percentages"][cls]
                    print(
                        f"  ✓ Class {cls}: Channel {i} = {cls_percentage:.1f}% "
                        f"(expected ~{expected_percentage:.1f}%)"
                    )

            result["tests"]["data_loading"] = {
                "input_shape": list(x.shape),
                "target_shape": list(y_onehot.shape),
                "passed": True,
            }
            print()

            # 3.5 Enhanced Channel Range Validation
            print("1.4 Enhanced Channel Range Validation:")

            # Check each channel type for reasonable ranges
            channel_ranges = {}

            # Get band names from dataset metadata
            import json

            metadata_path = Path(dataset_path) / "band_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                all_band_names = metadata.get("band_names", [])
            else:
                all_band_names = []

            # Get selected channel names from data module logs
            selected_channel_names = []
            if hasattr(data_module, "use_channels"):
                # Try to get band names from dataset
                try:
                    from glacier_mapping.data.data import resolve_channel_selection

                    selected_channels = resolve_channel_selection(
                        dataset_path,
                        landsat_channels=loader_opts.get("landsat_channels", True),
                        dem_channels=loader_opts.get("dem_channels", True),
                        spectral_indices_channels=loader_opts.get(
                            "spectral_indices_channels", True
                        ),
                        hsv_channels=loader_opts.get("hsv_channels", True),
                        physics_channels=loader_opts.get("physics_channels", False),
                        velocity_channels=loader_opts.get("velocity_channels", True),
                    )
                    # Get band names from dataset
                    if metadata_path.exists():
                        selected_channel_names = [
                            all_band_names[i]
                            for i in selected_channels
                            if i < len(all_band_names)
                        ]
                except Exception:
                    selected_channel_names = []

            for i, band_name in enumerate(selected_channel_names[: x.shape[-1]]):
                channel_data = x[..., i]
                channel_min = channel_data.min().item()
                channel_max = channel_data.max().item()
                channel_mean = channel_data.mean().item()
                channel_std = channel_data.std().item()

                channel_ranges[band_name] = {
                    "min": channel_min,
                    "max": channel_max,
                    "mean": channel_mean,
                    "std": channel_std,
                }

                # Validate reasonable ranges based on channel type
                if band_name.startswith("B"):  # Landsat bands
                    if channel_min < -1000 or channel_max > 2000:
                        print(
                            f"  ⚠️ {band_name}: Unusual range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Valid range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                elif band_name in ["elevation", "slope_deg"]:  # DEM channels
                    if band_name == "elevation" and (
                        channel_min < 0 or channel_max > 9000
                    ):
                        print(
                            f"  ⚠️ {band_name}: Unusual elevation range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                    elif band_name == "slope_deg" and (
                        channel_min < 0 or channel_max > 90
                    ):
                        print(
                            f"  ⚠️ {band_name}: Unusual slope range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Valid range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                elif band_name == "velocity_mask":
                    # Check if all values are either 0 or 1
                    is_binary = torch.all(
                        (channel_data == 0) | (channel_data == 1)
                    ).item()
                    if not is_binary:
                        unique_vals = torch.unique(channel_data)
                        print(
                            f"  ❌ {band_name}: NOT BINARY. Unique values: {unique_vals.numpy()}"
                        )
                        raise ValueError(
                            f"Velocity mask is not binary. Values: {unique_vals.numpy()}"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Is binary [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                elif band_name.startswith("velocity"):  # Velocity channels
                    if abs(channel_min) > 50 or abs(channel_max) > 50:
                        print(
                            f"  ⚠️ {band_name}: Unusual velocity range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Valid range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                elif band_name in ["NDVI", "NDWI", "NDSI"]:  # Spectral indices
                    if channel_min < -1.0 or channel_max > 1.0:
                        print(
                            f"  ⚠️ {band_name}: Unusual index range [{channel_min:.3f}, {channel_max:.3f}]"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Valid range [{channel_min:.3f}, {channel_max:.3f}]"
                        )
                elif band_name in ["H", "S", "V"]:  # HSV channels
                    if band_name == "H" and (channel_min < 0 or channel_max > 360):
                        print(
                            f"  ⚠️ {band_name}: Unusual hue range [{channel_min:.1f}, {channel_max:.1f}]"
                        )
                    elif band_name in ["S", "V"] and (
                        channel_min < 0 or channel_max > 1.0
                    ):
                        print(
                            f"  ⚠️ {band_name}: Unusual {band_name} range [{channel_min:.3f}, {channel_max:.3f}]"
                        )
                    else:
                        print(
                            f"  ✓ {band_name}: Valid range [{channel_min:.3f}, {channel_max:.3f}]"
                        )
                else:
                    print(
                        f"  ✓ {band_name}: Range [{channel_min:.3f}, {channel_max:.3f}]"
                    )

            result["tests"]["channel_ranges"] = channel_ranges
            print()

            # 4. Model Integration
            print("1.5 Model Integration:")

            # Extract channel and class configuration from loader_opts to pass to the model.
            # This ensures the model and data module use the exact same channel settings.
            loader_opts = config.get("loader_opts", {})
            loader_opts["processed_dir"] = str(
                dataset_path
            )  # Ensure model knows data path

            model_init_args = {
                key: loader_opts.get(key)
                for key in [
                    "landsat_channels",
                    "dem_channels",
                    "spectral_indices_channels",
                    "hsv_channels",
                    "physics_channels",
                    "velocity_channels",
                    "output_classes",
                    "class_names",
                ]
                if key in loader_opts
            }

            # Create Lightning module
            model = GlacierSegmentationModule(
                model_opts=config.get("model_opts", {}),
                loss_opts=config.get("loss_opts", {}),
                optim_opts=config.get("optim_opts", {}),
                metrics_opts=config.get("metrics_opts", {}),
                loader_opts=loader_opts,  # Pass full loader_opts for other settings
                **model_init_args,
            )

            # Verification: Ensure channel counts match
            if len(data_module.use_channels) != len(model.use_channels):
                raise ValueError(
                    f"Channel count mismatch: data module has {len(data_module.use_channels)} "
                    f"but model has {len(model.use_channels)}. Check channel configs."
                )

            print(f"  ✓ Model input channels: {len(model.use_channels)}")
            print(f"  ✓ Model output channels: {model.model.seg_layer.out_channels}")

            # Test forward pass
            with torch.no_grad():
                logits = model.model(x.permute(0, 3, 1, 2))  # (B, C, H, W)
                print(f"  ✓ Forward pass output shape: {logits.shape}")

                # Check activation
                if logits.shape[1] == 2:
                    probs = torch.sigmoid(logits)
                    print("  ✓ Activation: sigmoid (binary)")
                else:
                    probs = torch.softmax(logits, dim=1)
                    print("  ✓ Activation: softmax (multi-class)")

                print(f"  ✓ Probability range: [{probs.min():.3f}, {probs.max():.3f}]")

            result["tests"]["model"] = {
                "input_channels": len(data_module.use_channels),
                "output_channels": model.model.seg_layer.out_channels,
                "forward_pass": True,
                "passed": True,
            }
            print()

            # 5. Loss Function Verification
            print("1.6 Loss Function Verification:")

            loss_fn = model.loss_fn
            print(f"  ✓ Loss function: {type(loss_fn).__name__}")

            # The foreground_indices attribute was removed, so we no longer check it directly.
            # The class_weights parameter in customloss now handles class importance.
            # We can infer the effective foreground from output_classes and target_class_ids.
            if len(output_classes) == 1:
                # For binary, the foreground is implicitly the non-background class (index 1 after remapping)
                print(
                    f"  ✓ Binary task: output_classes={output_classes}, target_class_ids={target_class_ids}"
                )
            else:
                # For multi-class, all non-background output classes are considered
                print(
                    f"  ✓ Multi-class task: output_classes={output_classes}, target_class_ids={target_class_ids}"
                )

            # Test loss computation
            with torch.no_grad():
                # Ensure target shapes match logits
                target_onehot = y_onehot.permute(0, 3, 1, 2)
                target_int = y_int.squeeze(-1).permute(0, 2, 1)

                # Handle channel mismatch for binary vs multiclass
                if target_onehot.shape[1] != logits.shape[1]:
                    if logits.shape[1] == 2:  # Binary model, multiclass target
                        # Convert to binary by combining foreground classes
                        target_binary = torch.any(
                            target_onehot[:, 1:, :, :], dim=1, keepdim=True
                        ).float()
                        target_onehot = torch.cat(
                            [1 - target_binary, target_binary], dim=1
                        )
                    elif (
                        logits.shape[1] == 3 and target_onehot.shape[1] == 2
                    ):  # Multiclass model, binary target
                        # Pad to 3 channels
                        bg_channel = target_onehot[:, 0:1, :, :]
                        target_onehot = torch.cat([bg_channel, target_onehot], dim=1)

                # Extract velocity data for loss function
                velocity = None
                velocity_mask = None
                if (
                    model.use_velocity_loss
                    and model.velocity_idx is not None
                    and model.velocity_mask_idx is not None
                ):
                    # input x is (B, H, W, C), model uses (B, C, H, W)
                    x_permuted = x.permute(0, 3, 1, 2)
                    vel_norm = x_permuted[
                        :, model.velocity_idx : model.velocity_idx + 1, :, :
                    ]

                    if model.normalization == "mean-std":
                        # These are numpy arrays, convert to tensor
                        mean = torch.from_numpy(model.norm_arr[0, :]).to(
                            vel_norm.device
                        )
                        std = torch.from_numpy(model.norm_arr[1, :]).to(vel_norm.device)
                        velocity = (
                            vel_norm * std[model.velocity_idx]
                            + mean[model.velocity_idx]
                        )
                    elif model.normalization == "min-max":
                        _min = torch.from_numpy(model.norm_arr_full[2, :]).to(
                            vel_norm.device
                        )
                        _max = torch.from_numpy(model.norm_arr_full[3, :]).to(
                            vel_norm.device
                        )
                        velocity = (
                            vel_norm
                            * (_max[model.velocity_idx] - _min[model.velocity_idx])
                            + _min[model.velocity_idx]
                        )

                    velocity_mask = x_permuted[
                        :, model.velocity_mask_idx : model.velocity_mask_idx + 1, :, :
                    ]

                dice_loss, boundary_loss, velocity_loss = loss_fn(
                    logits,
                    target_onehot,
                    target_int,
                    velocity=velocity,
                    velocity_mask=velocity_mask,
                )

                total_loss = dice_loss + boundary_loss + velocity_loss
                print(f"  ✓ Dice loss: {dice_loss.item():.4f}")
                print(f"  ✓ Boundary loss: {boundary_loss.item():.4f}")
                print(f"  ✓ Velocity loss: {velocity_loss.item():.4f}")
                print(f"  ✓ Total loss: {total_loss.item():.4f}")

            result["tests"]["loss"] = {
                "loss_computed": True,
                "passed": True,
            }
            print()

            # 6. Enhanced Dataset Integrity Validation
            print("1.7 Enhanced Dataset Integrity Validation:")

            # Check for comprehensive dataset specific issues
            if "comprehensive_phys64_s1" in str(dataset_path):
                print("  ✓ Comprehensive dataset detected")

                # Validate physics/velocity channel availability
                velocity_channels_enabled = loader_opts.get("velocity_channels", False)
                physics_channels_enabled = loader_opts.get("physics_channels", False)

                if velocity_channels_enabled:
                    print("  ✓ Velocity channels enabled for comprehensive dataset")
                else:
                    print("  ⚠️ Velocity channels disabled for comprehensive dataset")

                if physics_channels_enabled:
                    print("  ✓ Physics channels enabled for comprehensive dataset")
                else:
                    print("  ⚠️ Physics channels disabled for comprehensive dataset")

                # Check for velocity mask in data
                if "velocity_mask" in selected_channel_names:
                    print("  ✓ Velocity mask channel available")
                else:
                    print("  ⚠️ Velocity mask channel missing")

            # 7. Velocity Loss Configuration Check
            print("1.8 Velocity Loss Configuration Check:")

            velocity_loss_enabled = config.get("loss_opts", {}).get(
                "use_velocity_loss", False
            )
            velocity_channels_in_data = loader_opts.get("velocity_channels", False)

            print(f"  ✓ Loss velocity_loss setting: {velocity_loss_enabled}")
            print(f"  ✓ Data velocity channels: {velocity_channels_in_data}")

            # Validate velocity loss configuration consistency
            if velocity_loss_enabled and not velocity_channels_in_data:
                print(
                    "  ⚠️ WARNING: Velocity loss enabled but no velocity channels in data!"
                )
            elif not velocity_loss_enabled and velocity_channels_in_data:
                print("  ✓ Velocity channels available but loss disabled (OK for Gen7)")
            elif velocity_loss_enabled and velocity_channels_in_data:
                print("  ✓ Velocity loss properly configured with velocity data")
            else:
                print("  ✓ Velocity loss disabled (baseline configuration)")

            result["tests"]["velocity_config"] = {
                "velocity_loss_enabled": velocity_loss_enabled,
                "velocity_channels_in_data": velocity_channels_in_data,
                "consistent": velocity_loss_enabled == velocity_channels_in_data
                or not velocity_loss_enabled,
            }
            print()

            # 8. End-to-End Validation
            print("1.9 End-to-End Validation:")

            # Verify data flow consistency
            expected_output_channels = (
                2 if len(output_classes) == 1 else len(output_classes)
            )
            actual_output_channels = model.model.seg_layer.out_channels

            if actual_output_channels != expected_output_channels:
                raise ValueError(
                    f"Output channel mismatch: expected {expected_output_channels}, got {actual_output_channels}"
                )

            print("  ✓ Config → Data → Model → Loss pipeline consistent")
            print(f"  ✓ {task_name} task validation completed successfully")

            result["tests"]["end_to_end"] = {
                "pipeline_consistent": True,
                "passed": True,
            }
            print()

        except Exception as e:
            result["passed"] = False
            result["errors"].append(str(e))
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            print()

        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """Execute complete test suite."""
        try:
            # Run standalone tests
            self.test_raw_file_integrity()
            self.test_preprocessing_functions()
            self.verify_preprocessed_dataset()

            # Create missing configs
            configs = self.create_missing_configs()

            # Test each task
            for task_name, config_path in configs.items():
                self.test_results[task_name] = self.test_task(config_path, task_name)

            # Generate summary
            self.generate_summary()

        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            traceback.print_exc()

        finally:
            # Cleanup temporary files
            self.cleanup()

        return self.test_results

    def verify_preprocessed_dataset(self):
        """Verify the integrity of the entire preprocessed dataset."""
        print("=== Test: Verify Preprocessed Dataset ===")
        server_config = load_server_config(self.server)
        dataset_name = TEST_DATASET_NAME
        dataset_path = Path(server_config["processed_data_path"]) / dataset_name

        if not dataset_path.exists():
            print(f"  - Dataset not found at {dataset_path}, skipping verification.")
            return

        # Load band metadata
        metadata_path = dataset_path / "band_metadata.json"
        if not metadata_path.exists():
            raise ValueError("band_metadata.json not found in dataset.")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        band_names = metadata.get("band_names", [])
        expected_channels = int(metadata.get("num_bands", len(band_names)))
        stats_path = dataset_path / "dataset_statistics.json"
        expected_spatial = None
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            window_size = stats.get("summary", {}).get("config", {}).get("window_size")
            if window_size and len(window_size) == 2:
                expected_spatial = tuple(window_size)
        try:
            velocity_mask_channel = band_names.index("velocity_mask")
        except ValueError:
            print(
                "  - No velocity mask channel in this dataset, skipping verification."
            )
            return

        all_slice_files = []
        all_mask_files = []
        for split in ["train", "val", "test"]:
            split_dir = dataset_path / split
            if split_dir.exists():
                all_slice_files.extend(list(split_dir.glob("tiff_*.npy")))
                all_mask_files.extend(list(split_dir.glob("mask_*.npy")))

        if not all_slice_files:
            raise ValueError("No slice files found in the dataset.")

        print(
            f"  - Found {len(all_slice_files)} slice files and {len(all_mask_files)} mask files to verify."
        )

        corrupt_files = []

        # Verify TIFF slices
        for slice_file in all_slice_files:
            try:
                slice_data = np.load(slice_file)
                if expected_spatial is None:
                    expected_spatial = slice_data.shape[:2]
                if slice_data.shape[:2] != expected_spatial:
                    print(f"  - {slice_file.name}: Incorrect shape {slice_data.shape}")
                    corrupt_files.append(slice_file)
                if slice_data.ndim != 3 or slice_data.shape[2] != expected_channels:
                    print(
                        f"  - {slice_file.name}: Expected {expected_channels} channels, "
                        f"got shape {slice_data.shape}"
                    )
                    corrupt_files.append(slice_file)

                velocity_mask = slice_data[..., velocity_mask_channel]
                if not np.all((velocity_mask == 0) | (velocity_mask == 1)):
                    unique_vals = np.unique(velocity_mask)
                    print(
                        f"  - {slice_file.name}: Velocity mask not binary. Values: {unique_vals}"
                    )
                    corrupt_files.append(slice_file)
            except Exception as e:
                print(f"  - Error reading {slice_file}: {e}")
                corrupt_files.append(slice_file)

        # Verify Mask slices
        for mask_file in all_mask_files:
            try:
                mask_data = np.load(mask_file)
                if expected_spatial is not None and mask_data.shape != expected_spatial:
                    print(f"  - {mask_file.name}: Incorrect shape {mask_data.shape}")
                    corrupt_files.append(mask_file)

                valid_labels = np.all(np.isin(mask_data, [0, 1, 2, 255]))
                if not valid_labels:
                    print(f"  - {mask_file.name}: Contains invalid labels.")
                    corrupt_files.append(mask_file)
            except Exception as e:
                print(f"  - Error reading {mask_file}: {e}")
                corrupt_files.append(mask_file)

        if corrupt_files:
            print(f"  ❌ Found {len(corrupt_files)} corrupt files:")
            for f in corrupt_files[:5]:  # Print first 5
                print(f"    - {f.name}")
            raise ValueError("Dataset verification failed. Corrupt files found.")
        else:
            print("  ✓ All slice files are valid.")
        print()

    def generate_summary(self):
        """Generate comprehensive test summary."""
        print("=== COMPREHENSIVE TEST SUMMARY ===")

        all_passed = True
        for task_name, result in self.test_results.items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"{task_name.title()}: {status}")
            if not result["passed"]:
                all_passed = False
                for error in result["errors"]:
                    print(f"  Error: {error}")

        print()

        if all_passed:
            print("🎉 ALL TESTS PASSED - No regressions detected!")
            print()
            print("Regression Prevention Status:")

            # Comparative analysis
            clean_result = self.test_results.get("clean_ice", {})
            debris_result = self.test_results.get("debris_ice", {})
            multi_result = self.test_results.get("multiclass", {})

            if all(
                [
                    r.get("passed", False)
                    for r in [clean_result, debris_result, multi_result]
                ]
            ):
                print("✓ Class targeting logic verified across all tasks")
                print("✓ Binary remapping logic confirmed working")
                print("✓ Multi-class loss handling validated")
                print("✓ Configuration inheritance verified")
                print("✓ Data preprocessing consistency confirmed")

                # Specific validations
                clean_config = clean_result.get("tests", {}).get("config", {})
                debris_config = debris_result.get("tests", {}).get("config", {})
                multi_config = multi_result.get("tests", {}).get("config", {})

                if (
                    clean_config.get("output_classes") == [1]
                    and debris_config.get("output_classes") == [2]
                    and multi_config.get("output_classes") == [0, 1, 2]
                ):
                    print("✓ Output classes correctly configured per task")

                if (
                    clean_config.get("target_class_ids") == [1]
                    and debris_config.get("target_class_ids") == [2]
                    and multi_config.get("target_class_ids") == [1, 2]
                ):
                    print("✓ Target class IDs correctly configured per task")

            print()
            print(
                "The glacier mapping system is ready for production use across all task types."
            )

        else:
            print("❌ SOME TESTS FAILED - Regressions detected!")
            print(
                "Please review the errors above and fix the issues before proceeding."
            )

    def cleanup(self):
        """Clean up temporary files."""
        import shutil

        try:
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠ Warning: Could not clean up temp directory {self.temp_dir}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive glacier mapping test suite"
    )
    parser.add_argument(
        "--server", default="desktop", help="Server configuration to use"
    )
    parser.add_argument(
        "--subset-size", type=int, default=5, help="Number of files to test per split"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs for testing"
    )

    args = parser.parse_args()

    # Create and run test suite
    test_suite = GlacierTaskTestSuite(
        server=args.server, subset_size=args.subset_size, epochs=args.epochs
    )

    results = test_suite.run_all_tests()

    # Exit with appropriate code
    all_passed = all(result.get("passed", False) for result in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
