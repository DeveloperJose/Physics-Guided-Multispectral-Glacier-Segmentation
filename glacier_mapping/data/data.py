import glob
import json
import logging
import os
import pathlib

import elasticdeform
import numpy as np
import torch
from torch.utils.data import Dataset

import glacier_mapping.utils.logging as fn

# Channel group definitions for semantic selection
# NOTE: Indices are resolved dynamically from band_metadata.json at runtime.
# The "names" field is the source of truth - indices are looked up by name.
CHANNEL_GROUP_DEFINITIONS = {
    "landsat": {
        "names": ["B1", "B2", "B3", "B4", "B5", "B6_VCID1", "B6_VCID2", "B7"],
        "description": "Landsat-7 spectral bands",
    },
    "dem": {
        "names": ["elevation", "slope_deg"],
        "description": "Digital Elevation Model features",
    },
    "spectral_indices": {
        "names": ["NDVI", "NDWI", "NDSI"],
        "description": "Spectral indices",
    },
    "hsv": {
        "names": ["H", "S", "V"],
        "description": "HSV color space channels",
    },
    "velocity": {
        "names": ["velocity", "velocity_x", "velocity_y", "velocity_mask"],
        "description": "ITS_LIVE glacier velocity data (magnitude, vx, vy, mask)",
        "mandatory": [
            "velocity_mask"
        ],  # velocity_mask must always be included when any velocity channel is selected
        "no_normalize": ["velocity_mask"],  # Binary mask - should NOT be normalized
    },
    "physics": {
        "names": ["flow_accumulation", "tpi", "roughness", "plan_curvature"],
        "description": "Physics-based terrain features",
    },
}

# Channels requiring logarithmic scaling (0 to +inf -> 0 to 1)
LOG_CHANNELS = {"velocity", "flow_accumulation", "roughness"}

# Channels requiring symmetric logarithmic scaling (-inf to +inf -> -1 to 1)
SYMLOG_CHANNELS = {"velocity_x", "velocity_y", "tpi", "plan_curvature"}


def load_band_names(processed_dir):
    """
    Load band names from band_metadata.json if available

    Args:
        processed_dir: Path to processed dataset directory

    Returns:
        np.ndarray: Array of band names
    """
    if isinstance(processed_dir, str):
        processed_dir = pathlib.Path(processed_dir)

    metadata_path = processed_dir / "band_metadata.json"

    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            band_names = np.array(metadata["band_names"])
            fn.log(
                logging.INFO,
                f"Loaded {len(band_names)} band names from {metadata_path}",
            )
            return band_names
        except Exception as e:
            fn.log(
                logging.WARNING,
                f"Failed to load band_metadata.json: {e}. Throwing exception.",
            )
            # return BAND_NAMES_LEGACY.copy()
            raise e
    else:
        fn.log(logging.INFO, "No band_metadata.json found. Throwing exception.")
        raise Exception("Failed to load band_metadata.json")
        # return BAND_NAMES_LEGACY.copy()


def resolve_channel_indices_by_name(band_names, channel_names):
    """
    Look up channel indices by name from band_names array.

    Args:
        band_names: np.ndarray of band names from dataset
        channel_names: List of channel names to resolve

    Returns:
        Dict[str, Optional[int]]: Mapping of channel name to index (None if not found)
    """
    band_names_list = band_names.tolist()
    indices = {}
    for name in channel_names:
        if name in band_names_list:
            indices[name] = band_names_list.index(name)
        else:
            indices[name] = None
    return indices


def get_no_normalize_channel_names():
    """
    Return set of channel names that should NOT be normalized.

    These are typically binary mask channels where normalization
    would destroy semantic meaning (e.g., velocity_mask is 0/1).

    Returns:
        Set[str]: Channel names to exclude from normalization
    """
    no_norm = set()
    for group in CHANNEL_GROUP_DEFINITIONS.values():
        if "no_normalize" in group:
            no_norm.update(group["no_normalize"])
    return no_norm


def resolve_channel_selection(
    processed_dir,
    landsat_channels=None,
    dem_channels=None,
    spectral_indices_channels=None,
    hsv_channels=None,
    physics_channels=None,
    velocity_channels=None,
):
    """
    Resolve semantic channel group specifications to numerical indices.

    Uses name-based lookup from band_metadata.json to dynamically resolve
    channel indices, making the system robust to different band orderings.

    Args:
        processed_dir: Path to processed dataset directory
        landsat_channels: true (all), false/None/[] (skip), or list of indices/names
        dem_channels: true (all), false/None/[] (skip), or list of indices/names
        spectral_indices_channels: true (all), false/None/[] (skip), or list of indices/names
        hsv_channels: true (all), false/None/[] (skip), or list of indices/names
        physics_channels: true (all), false/None/[] (skip), or list of indices/names
        velocity_channels: true (all), false/None/[] (skip), or list of indices/names

    Returns:
        List[int]: Sorted list of channel indices to use

    Raises:
        ValueError: If no channels are selected

    Warnings:
        - Logs warning if requested channel not in dataset (graceful skip)
    """
    # Load band names from metadata - this is the source of truth for channel indices
    band_names = load_band_names(processed_dir)

    fn.log(logging.INFO, f"Available channels in dataset: {len(band_names)}")
    fn.log(logging.DEBUG, f"Band names: {band_names.tolist()}")

    # Build name-to-index lookup for all channels in dataset
    all_channel_indices = resolve_channel_indices_by_name(
        band_names, band_names.tolist()
    )

    selected_channels = []
    selected_channel_names = []  # Track names for mandatory channel logic

    # Process each channel group
    channel_groups = [
        ("landsat", landsat_channels),
        ("dem", dem_channels),
        ("spectral_indices", spectral_indices_channels),
        ("hsv", hsv_channels),
        ("velocity", velocity_channels),
        ("physics", physics_channels),
    ]

    for group_name, group_value in channel_groups:
        if group_value is None or group_value is False:
            fn.log(logging.DEBUG, f"Skipping channel group: {group_name}")
            continue

        if group_value == []:
            fn.log(logging.DEBUG, f"Skipping channel group (empty list): {group_name}")
            continue

        group_def = CHANNEL_GROUP_DEFINITIONS[group_name]
        group_channel_names = group_def["names"]

        # Resolve indices for this group's channels by name
        group_indices = resolve_channel_indices_by_name(band_names, group_channel_names)

        if group_value is True:
            # Use all channels in this group (if available in dataset)
            fn.log(logging.INFO, f"Enabling all {group_name} channels")
            for name in group_channel_names:
                idx = group_indices.get(name)
                if idx is not None:
                    selected_channels.append(idx)
                    selected_channel_names.append(name)
                else:
                    fn.log(
                        logging.WARNING,
                        f"Channel '{name}' from {group_name} not found in dataset. Skipping.",
                    )

        elif isinstance(group_value, list):
            # Parse list of indices (within group) and/or names
            fn.log(
                logging.INFO, f"Enabling selected {group_name} channels: {group_value}"
            )
            for item in group_value:
                if isinstance(item, int):
                    # Treat as index WITHIN the group (0-based)
                    if 0 <= item < len(group_channel_names):
                        name = group_channel_names[item]
                        idx = group_indices.get(name)
                        if idx is not None:
                            selected_channels.append(idx)
                            selected_channel_names.append(name)
                        else:
                            fn.log(
                                logging.WARNING,
                                f"Channel '{name}' (group index {item}) from {group_name} "
                                f"not found in dataset. Skipping.",
                            )
                    else:
                        fn.log(
                            logging.WARNING,
                            f"Index {item} out of range for {group_name} "
                            f"(valid: 0-{len(group_channel_names) - 1})",
                        )

                elif isinstance(item, str):
                    # Channel name - resolve to index
                    if item in group_channel_names:
                        idx = group_indices.get(item)
                        if idx is not None:
                            selected_channels.append(idx)
                            selected_channel_names.append(item)
                        else:
                            fn.log(
                                logging.WARNING,
                                f"Channel '{item}' from {group_name} not found in dataset. Skipping.",
                            )
                    else:
                        fn.log(
                            logging.WARNING,
                            f"Channel name '{item}' not in {group_name} group. "
                            f"Valid names: {group_channel_names}",
                        )
                else:
                    fn.log(
                        logging.WARNING,
                        f"Invalid channel specification in {group_name}: {item}. "
                        f"Must be int (group index) or str (name).",
                    )

    # Add mandatory channels for groups that have them (e.g., velocity_mask for velocity)
    for group_name, group_value in channel_groups:
        if group_value is None or group_value is False or group_value == []:
            continue

        group_def = CHANNEL_GROUP_DEFINITIONS[group_name]
        mandatory = group_def.get("mandatory", [])

        for mandatory_name in mandatory:
            if mandatory_name not in selected_channel_names:
                # Check if any channel from this group was selected
                group_channel_names = group_def["names"]
                if any(name in selected_channel_names for name in group_channel_names):
                    # Try to add the mandatory channel
                    idx = all_channel_indices.get(mandatory_name)
                    if idx is not None:
                        selected_channels.append(idx)
                        selected_channel_names.append(mandatory_name)
                        fn.log(
                            logging.INFO,
                            f"Auto-included mandatory '{mandatory_name}' channel (index {idx})",
                        )
                    else:
                        fn.log(
                            logging.WARNING,
                            f"Mandatory channel '{mandatory_name}' not found in dataset!",
                        )

    # Remove duplicates and sort
    selected_channels = sorted(list(set(selected_channels)))

    if not selected_channels:
        raise ValueError(
            "No channels selected! At least one channel group must be enabled. "
            "Set landsat_channels, dem_channels, spectral_indices_channels, "
            "hsv_channels, physics_channels, or velocity_channels to true or provide a list."
        )

    fn.log(logging.INFO, f"Resolved channel selection: {selected_channels}")
    fn.log(
        logging.INFO, f"Selected band names: {band_names[selected_channels].tolist()}"
    )
    fn.log(logging.INFO, f"Total channels: {len(selected_channels)}")

    return selected_channels


class GlacierDataset(Dataset):
    """
    Custom Dataset for Glacier Data.

    Returns:
        x        : float32 tensor (H, W, C_in) (normalized, except mask channels)
        y_onehot: float32 tensor (H, W, C_out) (one-hot)
        y_int   : int64   tensor (H, W, 1) with values {0,1,2,255}
    """

    def __init__(
        self,
        folder_path,
        use_channels,
        output_classes,
        normalize,
        robust_scaling=True,
        transforms=None,
    ):
        self.folder_path = folder_path
        self.use_channels = use_channels
        self.output_classes = np.array(output_classes, dtype=np.uint8)
        self.normalize = normalize
        self.robust_scaling = robust_scaling
        self.transforms = transforms

        if isinstance(self.folder_path, str):
            self.folder_path = pathlib.Path(self.folder_path)

        assert isinstance(output_classes, list), "output_classes must be a list"
        assert len(set(output_classes)) == len(output_classes), (
            "output_classes cannot have duplicates"
        )
        assert all(self.output_classes >= 0) and all(self.output_classes < 3), (
            "output_classes must be either 0 (BG), 1 (CleanIce), or 2 (Debris)"
        )

        # Find image + mask files
        self.img_files = glob.glob(os.path.join(folder_path, "*tiff*"))
        self.mask_files = [s.replace("tiff", "mask") for s in self.img_files]

        # Normalization stats
        arr = np.load(folder_path.parent / "normalize_train.npy")
        if self.normalize == "min-max":
            self.min, self.max = arr[2][use_channels], arr[3][use_channels]
        elif self.normalize == "mean-std":
            self.mean, self.std = arr[0], arr[1]
            self.mean, self.std = self.mean[use_channels], self.std[use_channels]
            # We still need min/max for robust scaling of special channels
            self.min, self.max = arr[2][use_channels], arr[3][use_channels]
        else:
            raise ValueError("normalize must be 'min-max' or 'mean-std'")

        # Identify channels that should NOT be normalized (e.g., binary masks)
        # These channels retain their original values (typically 0/1)
        band_names = load_band_names(folder_path.parent)
        no_norm_names = get_no_normalize_channel_names()

        # --- Enhanced Channel Handling for Physics/Velocity ---

        # 1. Map use_channels indices to names
        self.channel_names = [band_names[ch] for ch in use_channels]

        # 2. Build masks for special scaling types
        self.log_mask = np.array([name in LOG_CHANNELS for name in self.channel_names])
        self.symlog_mask = np.array(
            [name in SYMLOG_CHANNELS for name in self.channel_names]
        )

        # 3. Build no_normalize mask (include binary masks AND special scaling channels)
        # The special scaling channels are handled separately, so we exclude them from
        # standard min-max/mean-std normalization loops.
        # IF robust_scaling is False, we do NOT exclude special channels (except strict no_norm ones)
        # so they fall through to standard normalization.
        if self.robust_scaling:
            self.no_normalize_mask = np.array(
                [
                    (name in no_norm_names)
                    or (name in LOG_CHANNELS)
                    or (name in SYMLOG_CHANNELS)
                    for name in self.channel_names
                ]
            )
        else:
            # Linear scaling mode: Only exclude strict no-norm channels (masks)
            # Velocity/Physics channels will be treated as standard channels
            self.no_normalize_mask = np.array(
                [name in no_norm_names for name in self.channel_names]
            )

        if self.robust_scaling and (np.any(self.log_mask) or np.any(self.symlog_mask)):
            fn.log(
                logging.INFO,
                f"Applied robust scaling to: {[n for n in self.channel_names if n in LOG_CHANNELS or n in SYMLOG_CHANNELS]}",
            )
        elif not self.robust_scaling:
            fn.log(
                logging.INFO,
                "Robust scaling DISABLED. Using linear scaling for all physics/velocity channels.",
            )

        # 4. Pre-compute scaling factors for special channels
        # LOG: log1p(x) / log1p(max) -> [0, 1]
        self.log_max = np.log1p(np.maximum(self.max, 1e-6))

        # SYMLOG: sign(x)*log1p(abs(x)) / log1p(max_abs) -> [-1, 1]
        max_abs = np.maximum(np.abs(self.min), np.abs(self.max))
        self.symlog_max = np.log1p(np.maximum(max_abs, 1e-6))

        if np.any(self.no_normalize_mask):
            skip_names = [
                band_names[ch]
                for ch, skip in zip(use_channels, self.no_normalize_mask)
                if skip
            ]
            fn.log(
                logging.INFO,
                f"Channels excluded from standard normalization: {skip_names}",
            )

    def __getitem__(self, index):
        file_data = np.load(self.img_files[index])
        data = file_data[:, :, self.use_channels]

        # Store original values for channels that should not be normalized (binary masks etc)
        # AND for special channels (log/symlog) which we will process manually
        data_no_norm = None
        if np.any(self.no_normalize_mask):
            data_no_norm = data[:, :, self.no_normalize_mask].copy()

        # Apply Standard Normalization (Min-Max or Mean-Std)
        # This will affect all channels, but we will overwrite the "no_normalize" ones later.
        # Ideally we'd only touch the relevant ones, but masking in place is tricky.
        # Overwriting is cleaner.
        if self.normalize == "min-max":
            data = np.clip(data, self.min, self.max)
            data = (data - self.min) / (self.max - self.min)
        elif self.normalize == "mean-std":
            data = (data - self.mean) / self.std

        # Restore/Process special channels
        if data_no_norm is not None:
            # First restore everything
            data[:, :, self.no_normalize_mask] = data_no_norm

            # Now apply custom transforms for Log/Symlog channels
            # We iterate because boolean masking 3D array flattens it
            # But we can use boolean indexing on the last dim if we are careful

            # Apply Log Transform: log1p(x) / log_max -> [0, 1]
            if self.robust_scaling and np.any(self.log_mask):
                # We need to act only on log_mask channels.
                # Since data_no_norm contains ALL no_norm channels, we need to map
                # global indices to no_norm indices? No.
                # data is (H, W, C). self.log_mask is (C,).
                # We can update in place using fancy indexing on axis 2?
                # No, data[:, :, mask] returns copy or flattened.

                # Loop is safest for clarity and avoiding shape errors
                for i in range(data.shape[2]):
                    if self.log_mask[i]:
                        # Clip to 0 just in case
                        val = np.maximum(data[:, :, i], 0)
                        data[:, :, i] = np.log1p(val) / self.log_max[i]

            # Apply SymLog Transform: sign(x) * log1p(abs(x)) / symlog_max -> [-1, 1]
            if self.robust_scaling and np.any(self.symlog_mask):
                for i in range(data.shape[2]):
                    if self.symlog_mask[i]:
                        val = data[:, :, i]
                        # symlog
                        data[:, :, i] = (
                            np.sign(val) * np.log1p(np.abs(val)) / self.symlog_max[i]
                        )

            # Binary masks (no_norm but NOT log/symlog) are already restored and left untouched.

        label_int = np.load(self.mask_files[index]).astype(np.uint8)
        label_int = np.expand_dims(label_int, axis=2)

        if len(self.output_classes) == 1:
            binary_class = self.output_classes[0]
            label = np.concatenate(
                (label_int != binary_class, label_int == binary_class), axis=2
            )
        else:
            label = np.concatenate(
                [label_int == x for x in self.output_classes], axis=2
            )

        # Convert boolean mask to uint8 for OpenCV/Albumentations compatibility
        label = label.astype(np.uint8)

        if self.transforms:
            transformed = self.transforms(image=data, mask=label)
            data = transformed["image"]
            label = transformed["mask"]

        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()

        return data, label, torch.from_numpy(label_int).long()

    def __len__(self):
        return len(self.img_files)


class DropoutChannels(object):
    """Random channel dropout augmentation."""

    def __init__(self, p):
        if (p < 0) or (p > 1):
            raise ValueError("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            # When channel count is very small, np.random.randint(size=0) fails.
            if data.shape[2] >= 5:
                rand_channel_index = np.random.randint(
                    low=0, high=data.shape[2], size=int(data.shape[2] / 5)
                )
                data[:, :, rand_channel_index] = 0
        return {"image": data, "mask": label}


class ElasticDeform(object):
    """Elastic deformation augmentation."""

    def __init__(self, p):
        if (p < 0) or (p > 1):
            raise ValueError("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample["image"], sample["mask"]
        label = label.astype(np.float32)
        if torch.rand(1) < self.p:
            [data, label] = elasticdeform.deform_random_grid([data, label], axis=(0, 1))
        label = np.round(label).astype(bool)
        return {"image": data, "mask": label}
