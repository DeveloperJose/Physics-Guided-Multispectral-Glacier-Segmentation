"""Lightning module for glacier segmentation."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchmetrics import JaccardIndex as TorchMetricsIoU
from torchmetrics import Precision, Recall

import pytorch_lightning as pl

from glacier_mapping.model.losses import customloss
from glacier_mapping.model.unet import Unet


class GlacierSegmentationModule(pl.LightningModule):
    """Lightning module for glacier segmentation with U-Net."""

    def __init__(
        self,
        model_opts: Dict[str, Any],
        loss_opts: Dict[str, Any],
        optim_opts: Dict[str, Any],
        scheduler_opts: Optional[Dict[str, Any]] = None,
        metrics_opts: Optional[Dict[str, Any]] = None,
        training_opts: Optional[Dict[str, Any]] = None,
        reg_opts: Optional[Dict[str, Any]] = None,
        loader_opts: Optional[Dict[str, Any]] = None,
        class_names: List[str] = ["BG", "CleanIce", "Debris"],
        output_classes: List[int] = [0, 1, 2],
        landsat_channels=True,
        dem_channels=True,
        spectral_indices_channels=True,
        hsv_channels=True,
        physics_channels=False,
        velocity_channels=True,
        **kwargs,
    ):
        """
        Initialize Glacier segmentation module.

        Args:
            model_opts: Model configuration options
            loss_opts: Loss function configuration options
            optim_opts: Optimizer configuration options
            scheduler_opts: Learning rate scheduler options
            metrics_opts: Metrics configuration options
            training_opts: Training configuration options
            reg_opts: Regularization options
            loader_opts: Loader configuration (includes processed_dir)
            class_names: Names for each class
            output_classes: Output class indices (0=BG, 1=CleanIce, 2=Debris)
            landsat_channels: Landsat band selection (true/false/list)
            dem_channels: DEM feature selection (true/false/list)
            spectral_indices_channels: Spectral indices selection (true/false/list)
            hsv_channels: HSV channel selection (true/false/list)
            physics_channels: Physics feature selection (true/false/list)
            velocity_channels: Velocity data selection (true/false/list)
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Model configuration
        self.model_opts = model_opts
        self.loss_opts = loss_opts
        self.use_velocity_loss = self.loss_opts.get("use_velocity_loss", False)
        self.optim_opts = optim_opts
        self.scheduler_opts = scheduler_opts
        self.metrics_opts = metrics_opts or {
            "metrics": ["IoU", "precision", "recall"],
            "threshold": [0.5, 0.5],
        }
        self.training_opts = training_opts or {}
        self.reg_opts = reg_opts
        self.class_names = class_names
        self.output_classes = output_classes

        # Store channel group selections
        self.landsat_channels = landsat_channels
        self.dem_channels = dem_channels
        self.spectral_indices_channels = spectral_indices_channels
        self.hsv_channels = hsv_channels
        self.physics_channels = physics_channels
        self.velocity_channels = velocity_channels

        # Add processed_dir for normalization and prediction
        if loader_opts:
            self.processed_dir = loader_opts.get("processed_dir", "/tmp")
            self.normalization = loader_opts.get("normalize", "mean-std")
        else:
            self.processed_dir = "/tmp"
            self.normalization = "mean-std"

        # Resolve channels from semantic groups
        from glacier_mapping.data.data import resolve_channel_selection

        self.use_channels = resolve_channel_selection(
            self.processed_dir,
            landsat_channels=landsat_channels,
            dem_channels=dem_channels,
            spectral_indices_channels=spectral_indices_channels,
            hsv_channels=hsv_channels,
            physics_channels=physics_channels,
            velocity_channels=velocity_channels,
        )

        # Load normalization parameters
        self._load_normalization_params()

        # Initialize model with proper parameters
        model_args = model_opts.get("args", {})
        # For binary classification, output_classes has 1 element but we need 2 output channels (BG, class)
        out_channels = 2 if len(output_classes) == 1 else len(output_classes)
        self.model = Unet(
            inchannels=len(self.use_channels), outchannels=out_channels, **model_args
        )

        # Initialize loss function - filter only supported args
        supported_loss_args = {
            "act",
            "smooth",
            "label_smoothing",
            "masked",
            "theta0",
            "theta",
            "alpha",
            "class_weights",
            "velocity_confidence_threshold",
            "velocity_low_speed_threshold",
            "velocity_loss_weight",
            "velocity_loss_warmup_epochs",
            "velocity_loss_ramp_epochs",
            "velocity_loss_clip",
        }
        loss_args = {k: v for k, v in loss_opts.items() if k in supported_loss_args}

        print(
            f"DEBUG: Passing verbose to customloss: {self.hparams.get('verbose', False)}"
        )
        self.loss_fn = customloss(
            verbose=self.hparams.get("verbose", False), **loss_args
        )

        # Initialize learnable sigma parameters for loss weighting
        # Fixed 3-sigma system: Dice, Boundary, Velocity
        # Based on original stable design from frame.py
        self.sigma_dice = nn.Parameter(torch.tensor(0.5))
        self.sigma_boundary = nn.Parameter(torch.tensor(0.5))
        # Initialize all sigmas to 0.5 for equal initial weighting
        # The network will learn optimal weights through Kendall's uncertainty formulation
        self.sigma_velocity = (
            nn.Parameter(torch.tensor(0.5)) if self.use_velocity_loss else None
        )

        # Keep sigma_list for backward compatibility
        self.sigma_list = nn.ParameterList([self.sigma_dice, self.sigma_boundary])
        if self.use_velocity_loss:
            self.sigma_list.append(self.sigma_velocity)

        # Identify velocity channel indices for Physics Loss
        self.velocity_idx = None
        self.velocity_mask_idx = None

        # Load band names to find indices
        from glacier_mapping.data.data import load_band_names

        band_names = load_band_names(self.processed_dir)

        # Find indices in the USED channels list
        used_band_names = band_names[self.use_channels]
        if "velocity" in used_band_names:
            # np.where returns tuple of arrays, take first element [0]
            self.velocity_idx = np.where(used_band_names == "velocity")[0][0]

        if "velocity_mask" in used_band_names:
            self.velocity_mask_idx = np.where(used_band_names == "velocity_mask")[0][0]

        if self.velocity_idx is not None and self.velocity_mask_idx is not None:
            print(
                f"Physics Loss Enabled: velocity_idx={self.velocity_idx}, velocity_mask_idx={self.velocity_mask_idx}"
            )

        # Initialize metrics
        self._setup_metrics()

        # For mixed precision training - use automatic optimization
        self.automatic_optimization = True

        # Track best validation loss
        self.best_val_loss = float("inf")

    def _setup_metrics(self):
        """Setup torchmetrics for tracking."""
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()

        # Setup metrics for each class
        for i, class_idx in enumerate(self.output_classes):
            if class_idx == 0:  # Skip background
                continue

            class_name = self.class_names[class_idx]

            # IoU (JaccardIndex)
            self.train_metrics[f"{class_name}_iou"] = TorchMetricsIoU(
                task="binary", average="macro"
            )
            self.val_metrics[f"{class_name}_iou"] = TorchMetricsIoU(
                task="binary", average="macro"
            )

            # Precision and Recall
            self.train_metrics[f"{class_name}_precision"] = Precision(
                task="binary", average="macro"
            )
            self.val_metrics[f"{class_name}_precision"] = Precision(
                task="binary", average="macro"
            )

            self.train_metrics[f"{class_name}_recall"] = Recall(
                task="binary", average="macro"
            )
            self.val_metrics[f"{class_name}_recall"] = Recall(
                task="binary", average="macro"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with automatic optimization."""
        x, y_onehot, y_int = batch

        # Convert from NHWC to NCHW format expected by Conv2D
        x = x.permute(0, 3, 1, 2)
        y_onehot = y_onehot.permute(0, 3, 1, 2)
        y_int = y_int.squeeze(-1)  # Remove last dimension

        # Forward pass
        y_hat = self(x)

        # Extract Physics Data if available
        velocity = None
        velocity_mask = None

        if (
            self.use_velocity_loss
            and self.velocity_idx is not None
            and self.velocity_mask_idx is not None
        ):
            vel_norm = x[:, self.velocity_idx : self.velocity_idx + 1, :, :]
            velocity = self._denormalize_velocity(vel_norm)

            velocity_mask = x[
                :, self.velocity_mask_idx : self.velocity_mask_idx + 1, :, :
            ]

        loss = self.compute_loss(
            y_hat, y_onehot, y_int, velocity=velocity, velocity_mask=velocity_mask
        )

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update and log additional metrics
        self._update_metrics(y_hat, y_int, self.train_metrics, "train")

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y_onehot, y_int = batch

        # Convert from NHWC to NCHW format expected by Conv2D
        x = x.permute(0, 3, 1, 2)
        y_onehot = y_onehot.permute(0, 3, 1, 2)
        y_int = y_int.squeeze(-1)  # Remove last dimension

        with torch.no_grad():
            y_hat = self(x)

            # Extract Physics Data if available
            velocity = None
            velocity_mask = None

            if (
                self.use_velocity_loss
                and self.velocity_idx is not None
                and self.velocity_mask_idx is not None
            ):
                vel_norm = x[:, self.velocity_idx : self.velocity_idx + 1, :, :]
                velocity = self._denormalize_velocity(vel_norm)
                velocity_mask = x[
                    :, self.velocity_mask_idx : self.velocity_mask_idx + 1, :, :
                ]

            else:
                velocity = None
                velocity_mask = None

            loss = self.compute_loss(
                y_hat, y_onehot, y_int, velocity=velocity, velocity_mask=velocity_mask
            )

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update and log additional metrics
        self._update_metrics(y_hat, y_int, self.val_metrics, "val")

        # Track best validation loss (store scalar to avoid device issues)
        loss_value = loss.detach().item()
        if loss_value < self.best_val_loss:
            self.best_val_loss = loss_value
        self.log(
            "best_val_loss",
            torch.tensor(self.best_val_loss, device=loss.device),
            on_step=False,
            on_epoch=True,
        )

        return loss

    def _update_metrics(
        self,
        y_hat: torch.Tensor,
        y_int: torch.Tensor,
        metrics_dict: nn.ModuleDict,
        prefix: str,
    ):
        """Update metrics with predictions and targets."""
        # Convert predictions to probabilities / hard labels
        y_prob = torch.softmax(y_hat, dim=1)
        y_pred_argmax = torch.argmax(y_prob, dim=1)
        thresholds = self.metrics_opts.get(
            "threshold", [0.5 for _ in range(len(self.class_names))]
        )

        valid_mask = y_int != 255
        if valid_mask.sum() == 0:
            return

        # Update metrics for each class
        for i, class_idx in enumerate(self.output_classes):
            if class_idx == 0:  # Skip background
                continue

            class_name = self.class_names[class_idx]

            if len(self.output_classes) == 1:
                # Binary case: channel 1 is always the positive class in the two-logit head
                pos_prob = y_prob[:, 1]
                thr = thresholds[class_idx] if class_idx < len(thresholds) else 0.5
                y_pred_class = (pos_prob >= thr).float()
                y_true_class = (y_int == class_idx).float()
            else:
                # Multiclass: align with argmax inference to avoid empty positives
                y_pred_class = (y_pred_argmax == class_idx).float()
                y_true_class = (y_int == class_idx).float()

            # Apply ignore mask
            y_pred_class = y_pred_class[valid_mask]
            y_true_class = y_true_class[valid_mask]

            if y_true_class.numel() == 0:
                continue

            # Update metrics
            if f"{class_name}_iou" in metrics_dict:
                iou_metric: Any = metrics_dict[f"{class_name}_iou"]
                iou_metric.update(y_pred_class, y_true_class.int())
                iou_value = iou_metric.compute()
                self.log(
                    f"{prefix}_{class_name}_iou",
                    iou_value,
                    on_step=False,
                    on_epoch=True,
                )

            if f"{class_name}_precision" in metrics_dict:
                precision_metric: Any = metrics_dict[f"{class_name}_precision"]
                precision_metric.update(y_pred_class, y_true_class.int())
                precision_value = precision_metric.compute()
                self.log(
                    f"{prefix}_{class_name}_precision",
                    precision_value,
                    on_step=False,
                    on_epoch=True,
                )

            if f"{class_name}_recall" in metrics_dict:
                recall_metric: Any = metrics_dict[f"{class_name}_recall"]
                recall_metric.update(y_pred_class, y_true_class.int())
                recall_value = recall_metric.compute()
                self.log(
                    f"{prefix}_{class_name}_recall",
                    recall_value,
                    on_step=False,
                    on_epoch=True,
                )

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y_onehot: torch.Tensor,
        y_int: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        velocity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute custom loss with sigma weighting."""
        velocity_valid = False
        if (
            self.use_velocity_loss
            and velocity_mask is not None
            and y_int is not None
            and velocity_mask.numel() > 0
        ):
            ignore_mask = (y_int != 255).unsqueeze(1)
            masked_velocity = velocity_mask * ignore_mask
            valid_pixels = masked_velocity.sum()
            total_pixels = ignore_mask.sum()
            if total_pixels > 0:
                coverage = valid_pixels.float() / total_pixels.float()
                self.log(
                    "velocity_valid_fraction",
                    coverage,
                    on_step=True,
                    on_epoch=True,
                )
            velocity_valid = valid_pixels > 0

        # Compute custom loss (returns list of losses)
        if self.use_velocity_loss:
            losses = self.loss_fn(
                y_hat,
                y_onehot,
                y_int,
                velocity=velocity,
                velocity_mask=velocity_mask,
                current_epoch=self.current_epoch,
            )
        else:
            # Pass None for velocity if not used, customloss will return 0.0 for velocity_loss
            losses = self.loss_fn(
                y_hat,
                y_onehot,
                y_int,
                velocity=None,
                velocity_mask=None,
                current_epoch=self.current_epoch,
            )

        # Apply sigma weighting like in original Framework
        total_loss = torch.zeros(1, device=y_hat.device)

        # Telemetry for physics loss scaling
        if (
            self.use_velocity_loss
            and getattr(self.loss_fn, "last_velocity_valid", False)
            and len(losses) >= 3
        ):
            self.log(
                "velocity_loss_raw",
                self.loss_fn.last_velocity_base,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "velocity_loss_weighted",
                self.loss_fn.last_velocity_loss,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "velocity_loss_weight",
                torch.tensor(
                    getattr(self.loss_fn, "last_velocity_weight", 0.0),
                    device=y_hat.device,
                ),
                on_step=True,
                on_epoch=True,
            )

        # Log sigmas for monitoring
        # Log sigma parameters for monitoring
        self.log("sigma_dice", self.sigma_dice, on_step=True, on_epoch=True)
        self.log("sigma_boundary", self.sigma_boundary, on_step=True, on_epoch=True)
        if self.use_velocity_loss and self.sigma_velocity is not None:
            self.log("sigma_velocity", self.sigma_velocity, on_step=True, on_epoch=True)

        # Use fixed sigma parameters with proper constraints (original stable design)
        sigma_params = [self.sigma_dice, self.sigma_boundary]
        loss_components = [losses[0], losses[1]]
        if (
            self.use_velocity_loss
            and self.sigma_velocity is not None
            and velocity_valid
            and len(losses) >= 3
        ):
            sigma_params.append(self.sigma_velocity)
            loss_components.append(losses[2])

        for _loss, sig in zip(loss_components, sigma_params):
            # Allow sigma to be learned freely, enabling the model to down-weight
            # unhelpful loss components or heavily weight informative ones.
            # Numerical stability is maintained by the epsilon term.
            var = sig**2 + 1e-8

            # Kendall et al. (2018) uses 2*sigma^2 denominator
            weighted_loss = _loss / (2.0 * var)
            total_loss += weighted_loss

            # Regularization term: log(sigma) - prevents sigmas from collapsing
            total_loss += 0.5 * torch.log(var)

        return total_loss[0]

    def configure_optimizers(self) -> Any:
        """Configure optimizers and learning rate schedulers."""
        # Setup optimizer
        optimizer_name = self.optim_opts.get("name", "AdamW")
        optimizer_args = self.optim_opts.get("args", {})

        # Convert string values to proper types (fix for YAML parsing issues)
        if "weight_decay" in optimizer_args and isinstance(
            optimizer_args["weight_decay"], str
        ):
            try:
                optimizer_args["weight_decay"] = float(optimizer_args["weight_decay"])
            except ValueError:
                optimizer_args["weight_decay"] = float(
                    str(optimizer_args["weight_decay"]).replace("E", "e")
                )

        if "lr" in optimizer_args and isinstance(optimizer_args["lr"], str):
            try:
                optimizer_args["lr"] = float(optimizer_args["lr"])
            except ValueError:
                optimizer_args["lr"] = float(optimizer_args["lr"].replace("e", "e-"))

        # Create parameter groups for sigma learning (original stable design)
        param_groups = [
            {"params": self.model.parameters(), **optimizer_args},
            {"params": [self.sigma_dice], **optimizer_args},
            {"params": [self.sigma_boundary], **optimizer_args},
        ]

        # Add velocity sigma parameter if velocity loss is enabled
        if self.use_velocity_loss and self.sigma_velocity is not None:
            param_groups.append({"params": [self.sigma_velocity], **optimizer_args})

        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(param_groups, **optimizer_args)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(param_groups, **optimizer_args)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Setup scheduler if specified
        scheduler = None
        if self.scheduler_opts:
            scheduler_name = self.scheduler_opts.get("name", "OneCycleLR")
            scheduler_args = self.scheduler_opts.get("args", {})

            if scheduler_name == "OneCycleLR":
                # Need total_steps for OneCycleLR
                estimated_steps = int(self.trainer.estimated_stepping_batches)
                if estimated_steps <= 0:
                    # Fall back to a single step to avoid crashing; warn loudly so the run gets fixed
                    print(
                        "⚠️ Estimated stepping batches is 0; falling back to total_steps=1. "
                        "Check that the training dataloader has samples and drop_last "
                        "is not eliminating all batches."
                    )
                    total_steps = 1
                else:
                    total_steps = estimated_steps
                scheduler = OneCycleLR(
                    optimizer, total_steps=total_steps, **scheduler_args
                )
            elif scheduler_name == "ReduceLROnPlateau":
                scheduler = ReduceLROnPlateau(optimizer, **scheduler_args)

        if scheduler:
            if isinstance(scheduler, OneCycleLR):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                    },
                }
            else:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "frequency": 1,
                    },
                }
        else:
            return optimizer

    def on_train_epoch_end(self):
        """Reset training metrics at end of epoch."""
        for metric_name, metric_obj in self.train_metrics.items():
            metric_obj: Any = metric_obj
            metric_obj.reset()

    def on_validation_epoch_end(self):
        """Reset validation metrics at end of epoch."""
        for metric_name, metric_obj in self.val_metrics.items():
            metric_obj: Any = metric_obj
            metric_obj.reset()

    def _load_normalization_params(self):
        """Load normalization array from processed data directory."""
        from glacier_mapping.data.data import (
            load_band_names,
            get_no_normalize_channel_names,
        )

        norm_path = Path(self.processed_dir) / "normalize_train.npy"
        if norm_path.exists():
            self.norm_arr_full = np.load(norm_path)
            # Use first 2 rows (mean, std) for mean-std normalization
            self.norm_arr = self.norm_arr_full[:2, self.use_channels]
        else:
            # Fallback if normalization file doesn't exist - create array with proper channel dimensions
            num_channels = len(self.use_channels)
            self.norm_arr_full = np.array(
                [
                    [0] * num_channels,
                    [1] * num_channels,
                    [0] * num_channels,
                    [1] * num_channels,
                ]
            )  # mean, std, min, max
            self.norm_arr = self.norm_arr_full[:2, self.use_channels]

        # Identify channels that should NOT be normalized (e.g., binary masks)
        band_names = load_band_names(self.processed_dir)
        no_norm_names = get_no_normalize_channel_names()

        # Build boolean mask: True = skip normalization for this channel
        self.no_normalize_mask = np.array(
            [band_names[ch] in no_norm_names for ch in self.use_channels]
        )

    def _denormalize_velocity(self, vel_norm):
        # mean and std for velocity channel
        if self.velocity_idx is None:
            # Guard against misuse; should not happen when velocity loss is active
            raise ValueError("velocity_idx is None, cannot denormalize velocity.")

        channel_idx = self.use_channels[int(self.velocity_idx)]
        mean = torch.tensor(self.norm_arr_full[0, channel_idx], device=vel_norm.device)
        std = torch.tensor(self.norm_arr_full[1, channel_idx], device=vel_norm.device)

        return vel_norm * std + mean

    def normalize(self, x):
        """Normalize input data (from Framework).

        Skips normalization for channels marked as no_normalize (e.g., binary masks).
        """
        # Store original values for no-normalize channels
        x_no_norm = None
        if hasattr(self, "no_normalize_mask") and np.any(self.no_normalize_mask):
            x_no_norm = x[:, :, self.no_normalize_mask].copy()

        if self.normalization == "mean-std":
            _mean, _std = self.norm_arr[0], self.norm_arr[1]
            x_normalized = (x - _mean) / _std
        elif self.normalization == "min-max":
            # Use full normalization array for min/max values (following original Framework)
            _min = self.norm_arr_full[2, self.use_channels]  # mins for used channels
            _max = self.norm_arr_full[3, self.use_channels]  # maxs for used channels
            x_normalized = (np.clip(x, _min, _max) - _min) / (_max - _min)
        else:
            raise Exception("Invalid normalization")

        # Restore original values for no-normalize channels
        if x_no_norm is not None:
            x_normalized[:, :, self.no_normalize_mask] = x_no_norm

        return x_normalized

    def predict_slice(self, slice_arr, threshold=None, preprocess=True, use_mask=True):
        """Predict on single slice (from Framework)."""
        if preprocess:
            slice_arr = slice_arr[:, :, self.use_channels]
            slice_arr = self.normalize(slice_arr)

        _mask = np.sum(slice_arr, axis=2) == 0

        _x = torch.from_numpy(np.expand_dims(slice_arr, axis=0)).float().to(self.device)

        # Forward pass with no_grad to avoid building computation graph
        with torch.no_grad():
            _y = self.forward(_x.permute(0, 3, 1, 2))  # NCHW format

        if len(self.output_classes) == 1:  # Binary
            if threshold is None:
                threshold = [0.5]
            elif isinstance(threshold, (int, float)):
                threshold = [threshold]
            elif isinstance(threshold, list):
                pass  # Use provided threshold
            else:
                threshold = [0.5]

            # Use softmax for consistency with training (loss uses softmax)
            _y = torch.nn.functional.softmax(_y, dim=1)
            _y = _y.cpu().numpy()  # (1, 2, H, W) - move to CPU immediately
            if _y.shape[0] == 1:  # Remove batch dimension
                _y = _y[0]  # (2, H, W)
            y_pred = (_y[1] >= threshold[0]).astype(
                np.uint8
            )  # Use positive class probability
        else:  # Multi-class
            _y = torch.nn.functional.softmax(_y, dim=1)
            _y = _y.cpu().numpy()  # (1, C, H, W) - move to CPU immediately
            if _y.shape[0] == 1:  # Remove batch dimension
                _y = _y[0]  # (C, H, W)
            y_pred = np.argmax(_y, axis=0).astype(
                np.uint8
            )  # 0..C-1 (0=BG, 1=CI, 2=Debris)

        # Explicitly delete GPU tensor to free memory
        del _x

        if use_mask:
            # Ensure mask matches y_pred shape
            if y_pred.ndim == 2:  # (H, W)
                y_pred[_mask] = 0
            elif y_pred.ndim == 1:  # (H*W,) flattened
                y_pred[_mask.flatten()] = 0
            return y_pred, _mask
        return y_pred

    def freeze_layers(self, layers=None):
        """Freeze specified layers (from Framework)."""
        for i, param in enumerate(self.model.parameters()):
            if layers is None:
                param.requires_grad = False
            elif i < layers:
                param.requires_grad = False
