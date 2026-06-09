from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.model.metrics import IoU, precision, recall, tp_fp_fn
from glacier_mapping.utils.gpu import cleanup_gpu_memory


CLASS_TO_INDEX = {"bg": 0, "ci": 1, "dci": 2}
CLASS_TO_LEGACY_NAME = {"ci": "CleanIce", "dci": "Debris"}
CLASS_TO_ACC_PREFIX = {"ci": "ci", "dci": "db"}


@dataclass
class FullSplitEvaluationResult:
    metrics: dict[str, float]
    legacy_metrics: dict[str, float]
    rows: list[list[Any]]
    columns: list[str]
    output_dir: Path | None = None
    status: str = "SUCCESS"


def metric_name(scope: str, split: str, target: str, metric: str) -> str:
    """Build canonical metric names such as full_val_dci_iou."""
    return f"{scope}_{split}_{target}_{metric}"


def window_metric_name(split: str, target: str, metric: str) -> str:
    return metric_name("window", split, target, metric)


def full_metric_name(split: str, target: str, metric: str) -> str:
    return metric_name("full", split, target, metric)


def compute_window_binary_metrics(
    y_hat: torch.Tensor,
    y_int: torch.Tensor,
    *,
    split: str,
    target: str,
    class_idx: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute canonical window_* metrics from model logits and integer labels."""
    target = _normalize_target(target)
    pred, true = window_binary_prediction(
        y_hat,
        y_int,
        class_idx=class_idx,
        threshold=threshold,
    )
    from glacier_mapping.model.metrics import tp_fp_fn

    tp_, fp_, fn_ = tp_fp_fn(pred, true)
    return {
        window_metric_name(split, target, "precision"): float(
            precision(tp_, fp_, fn_)
        ),
        window_metric_name(split, target, "recall"): float(recall(tp_, fp_, fn_)),
        window_metric_name(split, target, "iou"): float(IoU(tp_, fp_, fn_)),
    }


def window_binary_prediction(
    y_hat: torch.Tensor,
    y_int: torch.Tensor,
    *,
    class_idx: int,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened valid binary prediction/target tensors for window metrics."""
    y_prob = torch.softmax(y_hat, dim=1)
    pos_prob = y_prob[:, 1]
    y_pred = (pos_prob >= threshold).int()
    y_true = (y_int == class_idx).int()
    valid_mask = y_int != 255
    return y_pred[valid_mask], y_true[valid_mask]


def predict_whole(module, whole_arr, window_size, threshold=None):
    """Predict on a whole image using sliding windows."""
    use_channels = module.use_channels
    whole_arr = whole_arr[:, :, use_channels]
    whole_arr = module.normalize(whole_arr)
    mask = np.sum(whole_arr, axis=2) == 0

    y_pred = np.zeros((whole_arr.shape[0], whole_arr.shape[1]), dtype=np.uint8)

    for row in range(0, whole_arr.shape[0], window_size[0]):
        for column in range(0, whole_arr.shape[1], window_size[1]):
            current_slice = whole_arr[
                row : row + window_size[0], column : column + window_size[1], :
            ]

            if current_slice.shape[:2] != window_size:
                temp = np.zeros((window_size[0], window_size[1], whole_arr.shape[2]))
                temp[: current_slice.shape[0], : current_slice.shape[1], :] = (
                    current_slice
                )
                current_slice = temp

            pred = module.predict_slice(
                current_slice, threshold, preprocess=False, use_mask=False
            )

            endrow_dest = min(row + window_size[0], y_pred.shape[0])
            endrow_source = min(window_size[0], y_pred.shape[0] - row)
            endcolumn_dest = min(column + window_size[1], y_pred.shape[1])
            endcolumn_source = min(window_size[1], y_pred.shape[1] - column)

            y_pred[row:endrow_dest, column:endcolumn_dest] = pred[
                0:endrow_source, 0:endcolumn_source
            ]

    y_pred[mask] = 0
    return y_pred, mask


def get_y_true(label_mask, output_classes, is_binary, binary_class_idx=None, mask=None):
    """Convert one-hot mask to true labels."""
    y_true = np.zeros((label_mask.shape[0], label_mask.shape[1]), dtype=np.uint8)

    if is_binary:
        assert binary_class_idx is not None
        y_true[label_mask[:, :, binary_class_idx - 1] != 1] = 0
        y_true[label_mask[:, :, binary_class_idx - 1] == 1] = 1
    else:
        for i in range(label_mask.shape[2]):
            y_true[label_mask[:, :, i] == 1] = i + 1

    if mask is not None:
        y_true[mask] = 0
    return y_true


def get_probabilities(module, x_full):
    """Get a probability cube from a Lightning module using softmax."""
    use_ch = module.use_channels
    x = x_full[:, :, use_ch]
    x_norm = module.normalize(x)

    inp = torch.from_numpy(np.expand_dims(x_norm, 0)).float().to(module.device)
    logits = module.forward(inp.permute(0, 3, 1, 2))
    probs = (
        torch.nn.functional.softmax(logits, dim=1)[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    return probs


def predict_from_probs(probs, module, threshold=None):
    """Convert probabilities to hard predictions using module configuration."""
    if len(module.output_classes) == 1:
        if threshold is None:
            config_threshold = module.metrics_opts.get("threshold", [0.5])
            threshold = (
                config_threshold[0]
                if isinstance(config_threshold, list)
                else config_threshold
            )
        return (probs[:, :, 1] >= threshold).astype(np.uint8)

    return np.argmax(probs, axis=2).astype(np.uint8)


def predict_with_probs(module, x_full, threshold=None):
    """Return `(probabilities, hard_prediction)` for an input window."""
    probs = get_probabilities(module, x_full)
    prediction = predict_from_probs(probs, module, threshold)
    return probs, prediction


def get_pr_iou(pred, true):
    """Calculate precision, recall, IoU, and confusion components."""
    pred_t = torch.from_numpy(pred.astype(np.uint8))
    true_t = torch.from_numpy(true.astype(np.uint8))
    tp, fp, fn = tp_fp_fn(pred_t, true_t)
    return (precision(tp, fp, fn), recall(tp, fp, fn), IoU(tp, fp, fn), tp, fp, fn)


def calculate_binary_metrics(y_pred, y_true, target_class, mask=None):
    """Calculate binary precision/recall/IoU for a target class."""
    if mask is not None:
        valid = ~mask
        y_pred_valid = y_pred[valid]
        y_true_valid = y_true[valid]
    else:
        y_pred_valid = y_pred
        y_true_valid = y_true

    t_bin = (y_true_valid == target_class).astype(np.uint8)
    p_bin = (y_pred_valid == 1).astype(np.uint8)

    tp_, fp_, fn_ = tp_fp_fn(torch.from_numpy(p_bin), torch.from_numpy(t_bin))
    return (
        precision(tp_, fp_, fn_),
        recall(tp_, fp_, fn_),
        IoU(tp_, fp_, fn_),
        tp_,
        fp_,
        fn_,
    )


def create_invalid_mask(x_full, y_true):
    """Create invalid mask where pixels are empty imagery or ignored labels."""
    return (np.sum(x_full, axis=2) == 0) | (y_true == 255)


def softmax_probs(module, x_full):
    """Deprecated compatibility alias for `get_probabilities`."""
    return get_probabilities(module, x_full)


def merge_ci_debris(
    prob_ci: np.ndarray, prob_dci: np.ndarray, thr_ci: float, thr_dci: float
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two binary CI/DCI models into one 3-class map."""
    if prob_ci is None or prob_dci is None:
        raise ValueError("Both prob_ci and prob_dci must be provided")

    ci_mask = binary_fill_holes(prob_ci[:, :, 1] >= thr_ci)
    dci_mask = binary_fill_holes(prob_dci[:, :, 1] >= thr_dci)

    if ci_mask is None:
        ci_mask = prob_ci[:, :, 1] >= thr_ci
    if dci_mask is None:
        dci_mask = prob_dci[:, :, 1] >= thr_dci

    height, width = ci_mask.shape
    merged = np.zeros((height, width), dtype=np.uint8)
    merged[ci_mask] = CLASS_TO_INDEX["ci"]
    merged[dci_mask] = CLASS_TO_INDEX["dci"]

    probs = np.zeros((height, width, 3), dtype=np.float32)
    probs[:, :, CLASS_TO_INDEX["ci"]] = prob_ci[:, :, 1]
    probs[:, :, CLASS_TO_INDEX["dci"]] = prob_dci[:, :, 1]
    probs[:, :, CLASS_TO_INDEX["bg"]] = np.minimum(prob_ci[:, :, 0], prob_dci[:, :, 0])

    return merged, probs


def resolve_prediction_device(gpu_id: int | None = 0) -> torch.device:
    if gpu_id is None or gpu_id < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_id}")


def load_lightning_module(
    checkpoint_path: str | Path,
    device: torch.device,
    processed_data_path: str | Path | None = None,
) -> GlacierSegmentationModule:
    checkpoint_path = Path(checkpoint_path)
    if processed_data_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "loader_opts" in checkpoint["hyper_parameters"]:
            old_processed_dir = checkpoint["hyper_parameters"]["loader_opts"][
                "processed_dir"
            ]
            checkpoint["hyper_parameters"]["loader_opts"]["processed_dir"] = str(
                processed_data_path
            )
            print(
                f"Overriding processed_dir: {old_processed_dir} -> {processed_data_path}"
            )

        temp_ckpt = checkpoint_path.parent / "temp_modified.ckpt"
        torch.save(checkpoint, temp_ckpt)

        try:
            module = GlacierSegmentationModule.load_from_checkpoint(temp_ckpt)
        finally:
            if temp_ckpt.exists():
                temp_ckpt.unlink()
    else:
        module = GlacierSegmentationModule.load_from_checkpoint(checkpoint_path)

    module.eval()
    module.to(device)
    return module


def get_split_tiles(module: GlacierSegmentationModule, split: str) -> list[Path]:
    data_dir = Path(module.processed_dir) / split
    return sorted(data_dir.glob("tiff*"))


def evaluate_full_split_modules(
    *,
    frame_ci: GlacierSegmentationModule | None = None,
    frame_dci: GlacierSegmentationModule | None = None,
    split: str = "test",
    ci_threshold: float = 0.5,
    dci_threshold: float = 0.5,
    output_dir: str | Path | None = None,
    save_probabilities: bool = True,
    progress: bool = True,
) -> FullSplitEvaluationResult:
    """Evaluate one CI/DCI model or a paired CI+DCI model on a full split."""
    has_ci = frame_ci is not None
    has_dci = frame_dci is not None
    if not (has_ci or has_dci):
        raise ValueError("Must provide at least one model")

    module = frame_ci if frame_ci is not None else frame_dci
    if module is None:
        raise RuntimeError("No valid model loaded for full split evaluation")

    tiles = get_split_tiles(module, split)
    if not tiles:
        raise FileNotFoundError(f"No {split} tiles found under {module.processed_dir}")

    out_path = Path(output_dir) if output_dir is not None else None
    preds_dir = None
    if out_path is not None:
        preds_dir = out_path / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running full {split} evaluation on {len(tiles)} tiles...")

    rows, acc = _run_full_split_prediction(
        frame_ci=frame_ci,
        frame_dci=frame_dci,
        ci_threshold=ci_threshold,
        dci_threshold=dci_threshold,
        tiles=tiles,
        preds_dir=preds_dir,
        save_probabilities=save_probabilities,
        progress=progress,
    )

    metrics, legacy_metrics, total_row = _aggregate_metrics(
        acc=acc,
        split=split,
        has_ci=has_ci,
        has_dci=has_dci,
    )
    rows.append(total_row)

    rows_with_checkpoint = [["prediction"] + row for row in rows]
    columns = _legacy_columns(has_ci=has_ci, has_dci=has_dci)

    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_with_checkpoint, columns=columns).to_csv(
            out_path / "metrics.csv", index=False
        )

    return FullSplitEvaluationResult(
        metrics=metrics,
        legacy_metrics=legacy_metrics,
        rows=rows_with_checkpoint,
        columns=columns,
        output_dir=out_path,
    )


def evaluate_single_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str = "val",
    target: str = "dci",
    threshold: float = 0.5,
    processed_data_path: str | Path | None = None,
    gpu_id: int | None = 0,
    output_dir: str | Path | None = None,
    save_probabilities: bool = True,
) -> FullSplitEvaluationResult:
    target = _normalize_target(target)
    device = resolve_prediction_device(gpu_id)
    module = load_lightning_module(checkpoint_path, device, processed_data_path)
    try:
        return evaluate_full_split_modules(
            frame_ci=module if target == "ci" else None,
            frame_dci=module if target == "dci" else None,
            split=split,
            ci_threshold=threshold,
            dci_threshold=threshold,
            output_dir=output_dir,
            save_probabilities=save_probabilities,
        )
    finally:
        del module
        cleanup_gpu_memory(synchronize=False)


def evaluate_paired_checkpoints(
    ci_checkpoint_path: str | Path,
    dci_checkpoint_path: str | Path,
    *,
    split: str = "val",
    ci_threshold: float = 0.5,
    dci_threshold: float = 0.5,
    processed_data_path: str | Path | None = None,
    gpu_id: int | None = 0,
    output_dir: str | Path | None = None,
    save_probabilities: bool = True,
) -> FullSplitEvaluationResult:
    device = resolve_prediction_device(gpu_id)
    frame_ci = load_lightning_module(ci_checkpoint_path, device, processed_data_path)
    frame_dci = load_lightning_module(dci_checkpoint_path, device, processed_data_path)
    try:
        return evaluate_full_split_modules(
            frame_ci=frame_ci,
            frame_dci=frame_dci,
            split=split,
            ci_threshold=ci_threshold,
            dci_threshold=dci_threshold,
            output_dir=output_dir,
            save_probabilities=save_probabilities,
        )
    finally:
        del frame_ci, frame_dci
        cleanup_gpu_memory(synchronize=False)


def _run_full_split_prediction(
    *,
    frame_ci: GlacierSegmentationModule | None,
    frame_dci: GlacierSegmentationModule | None,
    ci_threshold: float,
    dci_threshold: float,
    tiles: list[Path],
    preds_dir: Path | None,
    save_probabilities: bool,
    progress: bool,
) -> tuple[list[list[Any]], dict[str, float]]:
    rows: list[list[Any]] = []
    acc = {
        "ci_tp": 0.0,
        "ci_fp": 0.0,
        "ci_fn": 0.0,
        "db_tp": 0.0,
        "db_fp": 0.0,
        "db_fn": 0.0,
    }

    has_ci = frame_ci is not None
    has_dci = frame_dci is not None
    iterator = tqdm(tiles, desc="Predicting") if progress else tiles

    with torch.no_grad():
        for tile in iterator:
            name = tile.name
            base = tile.stem

            x_full = np.load(tile)
            y_full = np.load(tile.parent / name.replace("tiff", "mask")).astype(
                np.uint8
            )
            invalid = create_invalid_mask(x_full, y_full)
            valid = ~invalid

            if has_ci ^ has_dci:
                frame = frame_ci if frame_ci is not None else frame_dci
                if frame is None:
                    raise RuntimeError("Missing single-model frame")
                target = "ci" if has_ci else "dci"
                model_class = CLASS_TO_INDEX[target]
                threshold = ci_threshold if has_ci else dci_threshold

                probs = get_probabilities(frame, x_full)
                if save_probabilities and preds_dir is not None:
                    np.save(preds_dir / f"{base}_probs.npy", probs)

                pred_bin = _threshold_binary_probs(probs, threshold)
                p_, r_, iou_, tp_, fp_, fn_ = calculate_binary_metrics(
                    pred_bin, y_full, model_class, invalid
                )

                acc_prefix = CLASS_TO_ACC_PREFIX[target]
                acc[f"{acc_prefix}_tp"] += tp_
                acc[f"{acc_prefix}_fp"] += fp_
                acc[f"{acc_prefix}_fn"] += fn_
                rows.append([name, p_, r_, iou_])

            else:
                if frame_ci is None or frame_dci is None:
                    raise RuntimeError("Both CI and DCI frames are required")

                prob_ci = get_probabilities(frame_ci, x_full)
                prob_dci = get_probabilities(frame_dci, x_full)
                merged, probs = merge_ci_debris(
                    prob_ci, prob_dci, ci_threshold, dci_threshold
                )
                if save_probabilities and preds_dir is not None:
                    np.save(preds_dir / f"{base}_probs.npy", probs)

                p_ci, r_ci, i_ci, tp_, fp_, fn_ = get_pr_iou(
                    (merged[valid] == CLASS_TO_INDEX["ci"]).astype(np.uint8),
                    (y_full[valid] == CLASS_TO_INDEX["ci"]).astype(np.uint8),
                )
                acc["ci_tp"] += tp_
                acc["ci_fp"] += fp_
                acc["ci_fn"] += fn_

                p_dci, r_dci, i_dci, tp_, fp_, fn_ = get_pr_iou(
                    (merged[valid] == CLASS_TO_INDEX["dci"]).astype(np.uint8),
                    (y_full[valid] == CLASS_TO_INDEX["dci"]).astype(np.uint8),
                )
                acc["db_tp"] += tp_
                acc["db_fp"] += fp_
                acc["db_fn"] += fn_

                rows.append([name, p_ci, r_ci, i_ci, p_dci, r_dci, i_dci])

    return rows, acc


def _aggregate_metrics(
    *,
    acc: dict[str, float],
    split: str,
    has_ci: bool,
    has_dci: bool,
) -> tuple[dict[str, float], dict[str, float], list[Any]]:
    metrics: dict[str, float] = {}
    legacy_metrics: dict[str, float] = {}

    if has_ci:
        p_ci = precision(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        r_ci = recall(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        i_ci = IoU(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        metrics.update(
            {
                full_metric_name(split, "ci", "precision"): float(p_ci),
                full_metric_name(split, "ci", "recall"): float(r_ci),
                full_metric_name(split, "ci", "iou"): float(i_ci),
            }
        )
        legacy_metrics.update({"CI_P": p_ci, "CI_R": r_ci, "CI_IoU": i_ci})

    if has_dci:
        p_dci = precision(acc["db_tp"], acc["db_fp"], acc["db_fn"])
        r_dci = recall(acc["db_tp"], acc["db_fp"], acc["db_fn"])
        i_dci = IoU(acc["db_tp"], acc["db_fp"], acc["db_fn"])
        metrics.update(
            {
                full_metric_name(split, "dci", "precision"): float(p_dci),
                full_metric_name(split, "dci", "recall"): float(r_dci),
                full_metric_name(split, "dci", "iou"): float(i_dci),
            }
        )
        legacy_metrics.update({"Deb_P": p_dci, "Deb_R": r_dci, "Deb_IoU": i_dci})

    if has_ci and has_dci:
        total_row = [
            "TOTAL",
            legacy_metrics["CI_P"],
            legacy_metrics["CI_R"],
            legacy_metrics["CI_IoU"],
            legacy_metrics["Deb_P"],
            legacy_metrics["Deb_R"],
            legacy_metrics["Deb_IoU"],
        ]
    elif has_ci:
        total_row = [
            "TOTAL",
            legacy_metrics["CI_P"],
            legacy_metrics["CI_R"],
            legacy_metrics["CI_IoU"],
        ]
    else:
        total_row = [
            "TOTAL",
            legacy_metrics["Deb_P"],
            legacy_metrics["Deb_R"],
            legacy_metrics["Deb_IoU"],
        ]

    return metrics, legacy_metrics, total_row


def _legacy_columns(*, has_ci: bool, has_dci: bool) -> list[str]:
    if has_ci and has_dci:
        return [
            "checkpoint",
            "tile",
            "CleanIce_precision",
            "CleanIce_recall",
            "CleanIce_IoU",
            "Debris_precision",
            "Debris_recall",
            "Debris_IoU",
        ]

    target = "ci" if has_ci else "dci"
    cname = CLASS_TO_LEGACY_NAME[target]
    return [
        "checkpoint",
        "tile",
        f"{cname}_precision",
        f"{cname}_recall",
        f"{cname}_IoU",
    ]


def _threshold_binary_probs(probs: np.ndarray, threshold: float) -> np.ndarray:
    pred_mask = probs[:, :, 1] >= threshold
    pred_filled = binary_fill_holes(pred_mask)
    if pred_filled is None:
        pred_filled = pred_mask
    return pred_filled.astype(np.uint8)


def _normalize_target(target: str) -> str:
    normalized = target.lower().replace("debris", "dci").replace("cleanice", "ci")
    if normalized not in {"ci", "dci"}:
        raise ValueError(f"target must be 'ci' or 'dci', got {target!r}")
    return normalized
