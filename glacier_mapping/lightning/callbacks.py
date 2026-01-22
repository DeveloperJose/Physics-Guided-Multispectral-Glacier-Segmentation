from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from glacier_mapping.utils import cleanup_gpu_memory
from glacier_mapping.utils.callback_utils import (
    load_dataset_metadata,
    generate_single_visualization,
    select_slices_by_iou_thirds,
    parse_slice_path,
)
import glacier_mapping.utils.logging as log


class ValidationVisualizationCallback(Callback):
    def __init__(
        self,
        viz_n: int = 4,
        log_every_n_epochs: int = 10,
        selection: str = "iou",
        save_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        scale_factor: float = 1,
    ):
        super().__init__()
        self.viz_n = viz_n
        self.log_every_n_epochs = log_every_n_epochs
        self.selection = selection
        self.save_dir = Path(save_dir) if save_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.scale_factor = scale_factor
        self.selected_slice_paths: Optional[List[Path]] = None
        self.slice_metadata: Dict[Path, Tuple[int, int]] = {}
        self.tile_rank_map: Dict[Path, int] = {}
        self._metadata_cache = None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if self.log_every_n_epochs == 0:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            self._generate_visualizations(trainer, pl_module)

    def _generate_visualizations(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if self.viz_n < 1:
            return

        processed_dir = getattr(pl_module, "processed_dir", "data/processed")
        val_dir = Path(processed_dir) / "val"

        if not val_dir.exists():
            log.warning(f"Validation directory not found: {val_dir}")
            return

        val_slices_all = sorted(val_dir.glob("tiff*"))

        if not val_slices_all:
            log.warning("No validation slices found")
            return

        if self.selected_slice_paths is None:
            num_samples = 3 * self.viz_n
            if self.selection == "iou":
                self.selected_slice_paths = select_slices_by_iou_thirds(
                    val_slices_all, pl_module, num_samples
                )
                self.tile_rank_map = {}
            else:
                self.selected_slice_paths = val_slices_all[:num_samples]
                self.tile_rank_map = {}

            for path in self.selected_slice_paths:
                tiff_num, slice_num = parse_slice_path(path)
                self.slice_metadata[path] = (tiff_num, slice_num)

            log.info(
                f"Selected {len(self.selected_slice_paths)} validation slices for tracking"
            )

        if self._metadata_cache is None:
            self._metadata_cache = load_dataset_metadata(
                pl_module, "val", self.image_dir
            )

        output_dir = self.save_dir / "val_visualizations" if self.save_dir else None

        for slice_path in self.selected_slice_paths:
            try:
                generate_single_visualization(
                    x_path=slice_path,
                    pl_module=pl_module,
                    output_dir=output_dir,
                    epoch=trainer.current_epoch + 1,
                    title_prefix="VAL",
                    metadata_cache=self._metadata_cache,
                    image_dir=self.image_dir,
                    scale_factor=self.scale_factor,
                    tile_rank_map=self.tile_rank_map,
                )
            except Exception as e:
                import traceback

                log.error(f"Error generating visualization for {slice_path}: {e}")
                log.error(f"Full traceback: {traceback.format_exc()}")

        cleanup_gpu_memory()

        if self.save_dir:
            from glacier_mapping.utils.callback_utils import (
                log_visualizations_to_all_loggers,
            )

            val_output_dir = self.save_dir / "val_visualizations"
            log_visualizations_to_all_loggers(
                trainer, val_output_dir, trainer.current_epoch + 1, "val_visualizations"
            )


class GlacierModelCheckpoint(ModelCheckpoint):
    pass


class GlacierTrainingMonitor(Callback):
    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        if batch_idx % self.log_every_n_steps == 0:
            import torch

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                pl_module.log("gpu_memory_gb", gpu_memory, on_step=True, on_epoch=False)

            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]["lr"]
            pl_module.log("learning_rate", current_lr, on_step=True, on_epoch=False)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if (
            hasattr(trainer, "callback_metrics")
            and "val_loss" in trainer.callback_metrics
        ):
            val_loss = trainer.callback_metrics["val_loss"]
            pl_module.log("epoch", trainer.current_epoch, on_step=False, on_epoch=True)

            if (
                hasattr(pl_module, "best_val_loss")
                and pl_module.best_val_loss is not None
            ):
                best_loss = pl_module.best_val_loss
                if isinstance(best_loss, (int, float)):
                    improvement = best_loss - float(val_loss)
                    pl_module.log(
                        "val_loss_improvement",
                        improvement,
                        on_step=False,
                        on_epoch=True,
                    )
