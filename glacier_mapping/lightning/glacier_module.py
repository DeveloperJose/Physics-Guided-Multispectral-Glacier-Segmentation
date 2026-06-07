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
        super().__init__()
        self.save_hyperparameters()

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

        self.landsat_channels = landsat_channels
        self.dem_channels = dem_channels
        self.spectral_indices_channels = spectral_indices_channels
        self.hsv_channels = hsv_channels
        self.physics_channels = physics_channels
        self.velocity_channels = velocity_channels

        if loader_opts:
            self.processed_dir = loader_opts.get("processed_dir", "/tmp")
            self.normalization = loader_opts.get("normalize", "mean-std")
            self.robust_scaling = loader_opts.get("robust_scaling", True)
        else:
            self.processed_dir = "/tmp"
            self.normalization = "mean-std"
            self.robust_scaling = True

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

        self._load_normalization_params()

        model_args = model_opts.get("args", {})
        out_channels = 2 if len(output_classes) == 1 else len(output_classes)
        self.model = Unet(
            inchannels=len(self.use_channels), outchannels=out_channels, **model_args
        )

        supported_loss_args = {
            "act",
            "smooth",
            "label_smoothing",
            "theta0",
            "theta",
            "class_weights",
            "velocity_high_speed_threshold",
            "velocity_loss_weight",
            "velocity_loss_warmup_epochs",
            "velocity_loss_ramp_epochs",
        }
        loss_args = {k: v for k, v in loss_opts.items() if k in supported_loss_args}

        self.loss_fn = customloss(**loss_args)

        self.sigma_dice = nn.Parameter(torch.tensor(1.0))
        self.sigma_boundary = nn.Parameter(torch.tensor(1.0))
        self.sigma_velocity = (
            nn.Parameter(torch.tensor(1.0)) if self.use_velocity_loss else None
        )

        self.sigma_list = nn.ParameterList([self.sigma_dice, self.sigma_boundary])
        if self.use_velocity_loss:
            self.sigma_list.append(self.sigma_velocity)

        self.velocity_idx = None
        self.velocity_mask_idx = None

        from glacier_mapping.data.data import load_band_names

        band_names = load_band_names(self.processed_dir)
        used_band_names = band_names[self.use_channels]
        if "velocity" in used_band_names:
            self.velocity_idx = np.where(used_band_names == "velocity")[0][0]

        if "velocity_mask" in used_band_names:
            self.velocity_mask_idx = np.where(used_band_names == "velocity_mask")[0][0]

        self._setup_metrics()
        self.automatic_optimization = True
        self.best_val_loss = float("inf")

    def _setup_metrics(self):
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()

        for i, class_idx in enumerate(self.output_classes):
            if class_idx == 0:
                continue

            class_name = self.class_names[class_idx]

            self.train_metrics[f"{class_name}_iou"] = TorchMetricsIoU(
                task="binary", average="macro"
            )
            self.val_metrics[f"{class_name}_iou"] = TorchMetricsIoU(
                task="binary", average="macro"
            )

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
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y_onehot, y_int = batch

        x = x.permute(0, 3, 1, 2)
        y_onehot = y_onehot.permute(0, 3, 1, 2)
        y_int = y_int.squeeze(-1)

        y_hat = self(x)

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

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self._update_metrics(y_hat, y_int, self.train_metrics, "train")

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y_onehot, y_int = batch

        x = x.permute(0, 3, 1, 2)
        y_onehot = y_onehot.permute(0, 3, 1, 2)
        y_int = y_int.squeeze(-1)

        with torch.no_grad():
            y_hat = self(x)

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

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._update_metrics(y_hat, y_int, self.val_metrics, "val")

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
        y_prob = torch.softmax(y_hat, dim=1)
        y_pred_argmax = torch.argmax(y_prob, dim=1)
        thresholds = self.metrics_opts.get(
            "threshold", [0.5 for _ in range(len(self.class_names))]
        )

        valid_mask = y_int != 255
        if valid_mask.sum() == 0:
            return

        for i, class_idx in enumerate(self.output_classes):
            if class_idx == 0:
                continue

            class_name = self.class_names[class_idx]

            if len(self.output_classes) == 1:
                pos_prob = y_prob[:, 1]
                thr = thresholds[class_idx] if class_idx < len(thresholds) else 0.5
                y_pred_class = (pos_prob >= thr).float()
                y_true_class = (y_int == class_idx).float()
            else:
                y_pred_class = (y_pred_argmax == class_idx).float()
                y_true_class = (y_int == class_idx).float()

            y_pred_class = y_pred_class[valid_mask]
            y_true_class = y_true_class[valid_mask]

            if y_true_class.numel() == 0:
                continue

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
            losses = self.loss_fn(
                y_hat,
                y_onehot,
                y_int,
                velocity=None,
                velocity_mask=None,
                current_epoch=self.current_epoch,
            )

        total_loss = torch.zeros(1, device=y_hat.device)

        if (
            self.use_velocity_loss
            and getattr(self.loss_fn, "last_velocity_valid", False)
            and len(losses) >= 3
        ):
            velocity_base = self.loss_fn.last_velocity_base
            velocity_loss = self.loss_fn.last_velocity_loss
            if velocity_base is None or velocity_loss is None:
                raise RuntimeError("Velocity loss marked valid without logged values.")
            self.log(
                "velocity_loss_raw",
                velocity_base,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "velocity_loss_weighted",
                velocity_loss,
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

        self.log("sigma_dice", self.sigma_dice, on_step=True, on_epoch=True)
        self.log("sigma_boundary", self.sigma_boundary, on_step=True, on_epoch=True)
        if self.use_velocity_loss and self.sigma_velocity is not None:
            self.log("sigma_velocity", self.sigma_velocity, on_step=True, on_epoch=True)

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
            var = sig**2 + 1e-8
            weighted_loss = _loss / (2.0 * var)
            total_loss += weighted_loss
            total_loss += 0.5 * torch.log(var)

        return total_loss[0]

    def configure_optimizers(self) -> Any:
        optimizer_name = self.optim_opts.get("name", "AdamW")
        optimizer_args = self.optim_opts.get("args") or {}

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

        param_groups = [
            {"params": self.model.parameters(), **optimizer_args},
            {"params": [self.sigma_dice], **optimizer_args},
            {"params": [self.sigma_boundary], **optimizer_args},
        ]

        if self.use_velocity_loss and self.sigma_velocity is not None:
            param_groups.append({"params": [self.sigma_velocity], **optimizer_args})

        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(param_groups, **optimizer_args)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(param_groups, **optimizer_args)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        scheduler = None
        if self.scheduler_opts:
            scheduler_name = self.scheduler_opts.get("name", "OneCycleLR")
            scheduler_args = self.scheduler_opts.get("args", {})

            if scheduler_name == "OneCycleLR":
                estimated_steps = int(self.trainer.estimated_stepping_batches)
                if estimated_steps <= 0:
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
        for metric_name, metric_obj in self.train_metrics.items():
            metric_obj: Any = metric_obj
            metric_obj.reset()

    def on_validation_epoch_end(self):
        for metric_name, metric_obj in self.val_metrics.items():
            metric_obj: Any = metric_obj
            metric_obj.reset()

    def _load_normalization_params(self):
        from glacier_mapping.data.data import (
            load_band_names,
            get_no_normalize_channel_names,
        )

        norm_path = Path(self.processed_dir) / "normalize_train.npy"
        if norm_path.exists():
            self.norm_arr_full = np.load(norm_path)
            self.norm_arr = self.norm_arr_full[:2, self.use_channels]
        else:
            num_channels = len(self.use_channels)
            self.norm_arr_full = np.array(
                [
                    [0] * num_channels,
                    [1] * num_channels,
                    [0] * num_channels,
                    [1] * num_channels,
                ]
            )
            self.norm_arr = self.norm_arr_full[:2, self.use_channels]

        band_names = load_band_names(self.processed_dir)
        no_norm_names = get_no_normalize_channel_names()

        self.no_normalize_mask = np.array(
            [band_names[ch] in no_norm_names for ch in self.use_channels]
        )

    def _denormalize_velocity(self, vel_norm):
        if self.velocity_idx is None:
            raise ValueError("velocity_idx is None, cannot denormalize velocity.")

        channel_idx = self.use_channels[int(self.velocity_idx)]

        if self.robust_scaling:
            max_val = torch.tensor(
                self.norm_arr_full[3, channel_idx], device=vel_norm.device
            )
            log_max = torch.log1p(
                torch.maximum(max_val, torch.tensor(1e-6, device=vel_norm.device))
            )
            return torch.expm1(vel_norm * log_max)

        if self.normalization == "mean-std":
            mean = torch.tensor(self.norm_arr_full[0, channel_idx], device=vel_norm.device)
            std = torch.tensor(self.norm_arr_full[1, channel_idx], device=vel_norm.device)
            return vel_norm * std + mean

        if self.normalization == "min-max":
            min_val = torch.tensor(
                self.norm_arr_full[2, channel_idx], device=vel_norm.device
            )
            max_val = torch.tensor(
                self.norm_arr_full[3, channel_idx], device=vel_norm.device
            )
            return vel_norm * (max_val - min_val) + min_val

        raise ValueError(f"Invalid normalization: {self.normalization}")

    def normalize(self, x):
        x_no_norm = None
        if hasattr(self, "no_normalize_mask") and np.any(self.no_normalize_mask):
            x_no_norm = x[:, :, self.no_normalize_mask].copy()

        if self.normalization == "mean-std":
            _mean, _std = self.norm_arr[0], self.norm_arr[1]
            x_normalized = (x - _mean) / _std
        elif self.normalization == "min-max":
            _min = self.norm_arr_full[2, self.use_channels]
            _max = self.norm_arr_full[3, self.use_channels]
            x_normalized = (np.clip(x, _min, _max) - _min) / (_max - _min)
        else:
            raise Exception("Invalid normalization")

        if x_no_norm is not None:
            x_normalized[:, :, self.no_normalize_mask] = x_no_norm

        return x_normalized

    def predict_slice(self, slice_arr, threshold=None, preprocess=True, use_mask=True):
        if preprocess:
            slice_arr = slice_arr[:, :, self.use_channels]
            slice_arr = self.normalize(slice_arr)

        _mask = np.sum(slice_arr, axis=2) == 0

        _x = torch.from_numpy(np.expand_dims(slice_arr, axis=0)).float().to(self.device)

        with torch.no_grad():
            _y = self.forward(_x.permute(0, 3, 1, 2))

        if len(self.output_classes) == 1:
            if threshold is None:
                threshold = [0.5]
            elif isinstance(threshold, (int, float)):
                threshold = [threshold]
            elif isinstance(threshold, list):
                pass
            else:
                threshold = [0.5]

            _y = torch.nn.functional.softmax(_y, dim=1)
            _y = _y.cpu().numpy()
            if _y.shape[0] == 1:
                _y = _y[0]
            y_pred = (_y[1] >= threshold[0]).astype(np.uint8)
        else:
            _y = torch.nn.functional.softmax(_y, dim=1)
            _y = _y.cpu().numpy()
            if _y.shape[0] == 1:
                _y = _y[0]
            y_pred = np.argmax(_y, axis=0).astype(np.uint8)

        del _x

        if use_mask:
            if y_pred.ndim == 2:
                y_pred[_mask] = 0
            elif y_pred.ndim == 1:
                y_pred[_mask.flatten()] = 0
            return y_pred, _mask
        return y_pred

    def freeze_layers(self, layers=None):
        for i, param in enumerate(self.model.parameters()):
            if layers is None:
                param.requires_grad = False
            elif i < layers:
                param.requires_grad = False
