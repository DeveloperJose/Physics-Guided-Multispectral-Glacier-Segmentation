"""Lightning data module for glacier mapping."""

import pathlib
from typing import List, Optional

import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from glacier_mapping.data.data import GlacierDataset


class GlacierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        processed_dir: str,
        batch_size: int = 8,
        landsat_channels=True,
        dem_channels=True,
        spectral_indices_channels=True,
        hsv_channels=True,
        physics_channels=False,
        velocity_channels=True,
        augmentations: Optional[dict] = None,
        output_classes: List[int] = [0, 1, 2],
        class_names: List[str] = ["BG", "CleanIce", "Debris"],
        normalize: str = "mean-std",
        robust_scaling: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.processed_dir = pathlib.Path(processed_dir)
        self.batch_size = batch_size
        self.landsat_channels = landsat_channels
        self.dem_channels = dem_channels
        self.spectral_indices_channels = spectral_indices_channels
        self.hsv_channels = hsv_channels
        self.physics_channels = physics_channels
        self.velocity_channels = velocity_channels
        self.output_classes = output_classes
        self.class_names = class_names
        self.normalize = normalize
        self.robust_scaling = robust_scaling
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if augmentations is not None:
            self.train_transform = self.create_augmentations(augmentations)
        else:
            self.train_transform = None

        self.val_transform = None

    def create_augmentations(self, aug_opts: dict) -> A.Compose:
        transforms = []
        if aug_opts.get("h_flip_prob", 0) > 0:
            transforms.append(A.HorizontalFlip(p=aug_opts["h_flip_prob"]))
        if aug_opts.get("v_flip_prob", 0) > 0:
            transforms.append(A.VerticalFlip(p=aug_opts["v_flip_prob"]))
        if aug_opts.get("rotate90_prob", 0) > 0:
            transforms.append(A.RandomRotate90(p=aug_opts["rotate90_prob"]))
        if aug_opts.get("transpose_prob", 0) > 0:
            transforms.append(A.Transpose(p=aug_opts["transpose_prob"]))
        return A.Compose(transforms)

    def setup(self, stage: Optional[str] = None):
        from glacier_mapping.data.data import resolve_channel_selection

        self.use_channels = resolve_channel_selection(
            self.processed_dir,
            landsat_channels=self.landsat_channels,
            dem_channels=self.dem_channels,
            spectral_indices_channels=self.spectral_indices_channels,
            hsv_channels=self.hsv_channels,
            physics_channels=self.physics_channels,
            velocity_channels=self.velocity_channels,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = GlacierDataset(
                self.processed_dir / "train",
                self.use_channels,
                self.output_classes,
                self.normalize,
                robust_scaling=self.robust_scaling,
                transforms=self.train_transform,
            )

            self.val_dataset = GlacierDataset(
                self.processed_dir / "val",
                self.use_channels,
                self.output_classes,
                self.normalize,
                robust_scaling=self.robust_scaling,
                transforms=self.val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
