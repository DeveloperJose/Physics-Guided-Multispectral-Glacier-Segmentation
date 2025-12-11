#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified predictor for:
 - Single CleanIce model
 - Single Debris model
 - Merged CleanIce + Debris binary models → 3-class output

Outputs for each test tile:
 - *_probs.npy (probability cube)

Also produces:
 - metrics.csv (per-tile metrics)
 - summary printout
"""

import argparse
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

import torch

from glacier_mapping.data.data import load_band_names
from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule
from glacier_mapping.model.metrics import IoU, precision, recall
from glacier_mapping.utils import cleanup_gpu_memory
from glacier_mapping.utils.prediction import (
    calculate_binary_metrics,
    create_invalid_mask,
    get_pr_iou,
    get_probabilities,
    merge_ci_debris,
)


def load_lightning_module(checkpoint_path, device, processed_data_path=None):
    """Load Lightning module from checkpoint."""

    if processed_data_path is not None:
        # Load checkpoint to modify hyperparameters
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Update processed_dir in loader_opts
        if "loader_opts" in checkpoint["hyper_parameters"]:
            old_processed_dir = checkpoint["hyper_parameters"]["loader_opts"][
                "processed_dir"
            ]
            checkpoint["hyper_parameters"]["loader_opts"]["processed_dir"] = (
                processed_data_path
            )
            print(
                f"Overriding processed_dir: {old_processed_dir} -> {processed_data_path}"
            )

        # Save modified checkpoint to temp file
        temp_ckpt = Path(checkpoint_path).parent / "temp_modified.ckpt"
        torch.save(checkpoint, temp_ckpt)

        try:
            # Load from modified checkpoint
            module = GlacierSegmentationModule.load_from_checkpoint(temp_ckpt)
        finally:
            # Clean up temp file
            if temp_ckpt.exists():
                temp_ckpt.unlink()
    else:
        module = GlacierSegmentationModule.load_from_checkpoint(checkpoint_path)

    module.eval()
    module.to(device)

    return module


def clean_run_name(run_name: str) -> str:
    """Clean run name by removing task prefixes."""
    # Remove common prefixes
    prefixes_to_remove = [
        "ablation_ci_",
        "ablation_dci_",
        "ablation_mc_",
        "ci_",
        "dci_",
        "mc_",
    ]
    cleaned = run_name
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned


def find_best_checkpoint_for_run(run_name_pattern: str, server_config: dict) -> Path:
    """Find best checkpoint for given run pattern on specific server."""
    server_output_path = Path(server_config["output_path"])
    best_checkpoint = None
    best_val_loss = float("inf")

    # Search for directories containing the run pattern
    matching_dirs = []
    if server_output_path.exists():
        matching_dirs = [
            d
            for d in server_output_path.iterdir()
            if run_name_pattern in d.name and d.is_dir()
        ]

    for run_dir in matching_dirs:
        checkpoints_dir = run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            continue

        # Look for checkpoints with epoch info
        ckpts = list(checkpoints_dir.glob("*epoch=*.ckpt"))
        for ckpt in ckpts:
            # Extract validation loss from filename
            if "val_loss=" in ckpt.name:
                try:
                    val_loss_str = ckpt.name.split("val_loss=")[1].split(".ckpt")[0]
                    val_loss = float(val_loss_str)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = ckpt
                except (ValueError, IndexError):
                    continue

    if best_checkpoint is None:
        raise ValueError(
            f"Could not find any checkpoints for run pattern '{run_name_pattern}' on server {server_config.get('hostname', 'unknown')}"
        )

    return best_checkpoint


def run_prediction_on_models(
    ci_checkpoint: Path | None,
    deb_checkpoint: Path | None,
    ci_threshold: float,
    deb_threshold: float,
    output_dir: Path,
    gpu_id: int,
    processed_data_path: str,
    feature_importance: bool = False,
    fi_samples: int | None = None,
) -> dict:
    """Run prediction on given models and return metrics summary."""

    # Set device
    gpu = gpu_id if gpu_id is not None else 0

    # Determine which models are available
    has_ci = ci_checkpoint is not None
    has_deb = deb_checkpoint is not None

    if not (has_ci or has_deb):
        raise ValueError("Must provide at least one checkpoint")

    # Load models
    frame_ci = None
    frame_deb = None

    if has_ci and ci_checkpoint is not None:
        frame_ci = load_lightning_module(ci_checkpoint, gpu, processed_data_path)

    if has_deb and deb_checkpoint is not None:
        frame_deb = load_lightning_module(deb_checkpoint, gpu, processed_data_path)

    # Get test tiles (from whichever model was loaded)
    if frame_ci is not None:
        data_dir = pathlib.Path(frame_ci.processed_dir)
    elif frame_deb is not None:
        data_dir = pathlib.Path(frame_deb.processed_dir)
    else:
        raise RuntimeError("No valid model loaded for test tiles")
    test_tiles = sorted(pathlib.Path(data_dir, "test").glob("tiff*"))

    # Create output directory for this checkpoint
    preds_dir = output_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running on {len(test_tiles)} test tiles...")

    # Run predictions (no visualization params)
    df_rows, acc = run_prediction(
        frame_ci,
        frame_deb,
        ci_threshold,
        deb_threshold,
        test_tiles,
        None,  # vis_mode
        None,  # vis_maxw
        preds_dir,
        has_ci,
        has_deb,
    )

    # Compute summary metrics
    metrics = {}

    if has_ci and has_deb:
        Pci = precision(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        Rci = recall(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])
        Ici = IoU(acc["ci_tp"], acc["ci_fp"], acc["ci_fn"])

        Pdb = precision(acc["db_tp"], acc["db_fp"], acc["db_fn"])
        Rdb = recall(acc["db_tp"], acc["db_fp"], acc["db_fn"])
        Idb = IoU(acc["db_tp"], acc["db_fp"], acc["db_fn"])

        metrics = {
            "CI_P": Pci,
            "CI_R": Rci,
            "CI_IoU": Ici,
            "Deb_P": Pdb,
            "Deb_R": Rdb,
            "Deb_IoU": Idb,
        }

        df_rows.append(["TOTAL", Pci, Rci, Ici, Pdb, Rdb, Idb])

    else:
        if has_ci:
            tp = acc["ci_tp"]
            fp = acc["ci_fp"]
            fn = acc["ci_fn"]
        else:
            tp = acc["db_tp"]
            fp = acc["db_fp"]
            fn = acc["db_fn"]

        P = precision(tp, fp, fn)
        R = recall(tp, fp, fn)
        iou = IoU(tp, fp, fn)

        if has_ci:
            metrics = {"CI_P": P, "CI_R": R, "CI_IoU": iou}
        else:
            metrics = {"Deb_P": P, "Deb_R": R, "Deb_IoU": iou}

        df_rows.append(["TOTAL", P, R, iou])

    # Add checkpoint column to all rows
    ckpt_name = "prediction"
    df_rows_with_ckpt = [[ckpt_name] + row for row in df_rows]

    # Determine CSV columns based on model type
    if has_ci and has_deb:
        columns = [
            "checkpoint",
            "tile",
            "CleanIce_precision",
            "CleanIce_recall",
            "CleanIce_IoU",
            "Debris_precision",
            "Debris_recall",
            "Debris_IoU",
        ]
    else:
        cname = "CleanIce" if has_ci else "Debris"
        columns = [
            "checkpoint",
            "tile",
            f"{cname}_precision",
            f"{cname}_recall",
            f"{cname}_IoU",
        ]

    # Save per-tile metrics for this checkpoint
    df = pd.DataFrame(df_rows_with_ckpt, columns=pd.Index(columns))
    df.to_csv(output_dir / "metrics.csv", index=False)

    # Feature importance analysis (if enabled)
    if feature_importance and (frame_ci or frame_deb):
        saliency_frame = frame_ci if frame_ci else frame_deb
        if saliency_frame:
            print("Computing feature importance...")

            # Compute feature importance
            fi_output_dir = output_dir / "feature_importance"
            fi_output_dir.mkdir(parents=True, exist_ok=True)

            importance_scores = compute_feature_importance(
                saliency_frame,
                test_tiles,
                target_class_idx=1,
                num_samples=fi_samples,
            )

            # Get channel names
            use_channels = saliency_frame.use_channels
            BAND_NAMES = load_band_names(saliency_frame.processed_dir)
            channel_names = BAND_NAMES[use_channels]

            # Save results
            save_feature_importance_results(
                importance_scores, channel_names, fi_output_dir
            )

    # Free GPU memory
    del frame_ci, frame_deb
    cleanup_gpu_memory(synchronize=False)

    return {"metrics": metrics, "output_dir": output_dir, "status": "SUCCESS"}


def main_prediction_logic(args) -> dict:
    """Main prediction logic that can be called from other scripts."""
    try:
        # Load server configurations
        servers_config = yaml.safe_load(Path("configs/servers.yaml").read_text())

        # Determine server to use
        if hasattr(args, "server") and args.server:
            current_server = args.server
            if current_server not in servers_config:
                raise ValueError(f"Server '{current_server}' not found in servers.yaml")
        else:
            # Auto-detect current server by checking hostname and paths
            current_server = None
            import socket

            hostname = socket.gethostname()

            for server_name, server_config in servers_config.items():
                if hostname == server_config.get("hostname", ""):
                    current_server = server_name
                    break

            if current_server is None:
                # Fallback: try to match by code path
                import os

                current_dir = os.getcwd()
                for server_name, server_config in servers_config.items():
                    if current_dir == server_config.get("code_path", ""):
                        current_server = server_name
                        break

            if current_server is None:
                raise RuntimeError(
                    "Could not determine current server. Check servers.yaml configuration or use --server parameter."
                )

        print(f"Using server: {current_server}")
        server_config = servers_config[current_server]

        # Resolve run names to best checkpoint paths
        ci_checkpoint_path = None
        deb_checkpoint_path = None

        if args.ci_run_name:
            ci_checkpoint_path = find_best_checkpoint_for_run(
                args.ci_run_name, server_config
            )
            print(f"Found CI checkpoint: {ci_checkpoint_path}")

        if args.deb_run_name:
            deb_checkpoint_path = find_best_checkpoint_for_run(
                args.deb_run_name, server_config
            )
            print(f"Found DCI checkpoint: {deb_checkpoint_path}")

        # Create output directory - use run name without ci/dci prefixes
        if args.ci_run_name and args.deb_run_name:
            # For merged runs, use the CI run name cleaned
            base_run_name = clean_run_name(args.ci_run_name)
        elif args.ci_run_name:
            base_run_name = clean_run_name(args.ci_run_name)
        else:
            base_run_name = clean_run_name(args.deb_run_name)

        # Use server's output path for predictions
        server_output_path = Path(server_config["output_path"])
        out_root = server_output_path.parent / "output_predictions" / base_run_name
        out_root.mkdir(parents=True, exist_ok=True)

        # Use the correct processed data path that has band_metadata.json
        processed_data_path = server_config["processed_data_path"]
        # For ablation models, they expect gen_robust_comprehensive subdirectory
        if "ablation" in (args.ci_run_name or "") or "ablation" in (
            args.deb_run_name or ""
        ):
            processed_data_path = f"{processed_data_path}/gen_robust_comprehensive"

        # Run prediction
        result = run_prediction_on_models(
            ci_checkpoint_path,
            deb_checkpoint_path,
            args.ci_threshold,
            args.deb_threshold,
            out_root,
            args.gpu,
            processed_data_path,
            getattr(args, "feature_importance", False),
            getattr(args, "fi_samples", None),
        )

        return result

    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


# Helpers
# Removed duplicate functions - now using shared utilities from glacier_mapping.utils.prediction


# ========================================================================
# FEATURE IMPORTANCE (GRADIENT-BASED SALIENCY)
# ========================================================================


def compute_feature_importance(
    module,
    test_tiles,
    target_class_idx,
    num_samples=None,
):
    """
    Compute gradient-based feature importance for input channels.

    Uses backpropagation to measure how much each input channel contributes
    to the prediction of a target class. Higher gradient magnitude indicates
    higher importance.

    Args:
        module: Lightning module with loaded model
        test_tiles: List of test tile paths
        target_class_idx: Class index to compute gradients for (0-indexed in model output)
        num_samples: Number of samples to use (None = all)

    Returns:
        channel_gradients: np.array of shape (num_channels,) with importance scores
    """
    module.eval()
    device = module.device
    use_channels = module.use_channels
    num_channels = len(use_channels)

    # Select subset of tiles if requested
    if num_samples is not None and num_samples < len(test_tiles):
        tiles_to_use = test_tiles[:num_samples]
    else:
        tiles_to_use = test_tiles

    print(f"Computing feature importance using {len(tiles_to_use)} test samples...")
    print(f"Target class index: {target_class_idx}")

    # Accumulate gradients across all samples
    channel_gradients = np.zeros(num_channels, dtype=np.float64)

    for tile_path in tqdm(tiles_to_use, desc="Computing saliency"):
        # Load tile
        x_full = np.load(tile_path)
        x = x_full[:, :, use_channels]

        # Normalize
        x_norm = module.normalize(x)

        # Convert to tensor with gradient tracking
        x_tensor = torch.from_numpy(x_norm).float().to(device).unsqueeze(0)
        x_tensor = x_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x_tensor.requires_grad_(True)

        # Forward pass
        logits = module(x_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Get mean activation for target class
        target_prob = probs[0, target_class_idx, :, :].mean()

        # Backward pass to compute gradients
        target_prob.backward()

        # Extract per-channel gradient magnitude
        channel_grad = np.zeros(len(use_channels))
        if x_tensor.grad is not None:
            grads = x_tensor.grad.data.abs()  # (1, C, H, W)
            channel_grad = (
                grads.sum(dim=(2, 3)).cpu().numpy()[0]
            )  # Sum over spatial dims -> (C,)

        # Accumulate
        channel_gradients += channel_grad

        # Clear gradients to free memory
        x_tensor.grad = None
        del x_tensor, logits, probs

    # Average across samples
    channel_gradients /= len(tiles_to_use)

    return channel_gradients


def save_feature_importance_results(importance_scores, channel_names, output_dir):
    """
    Save feature importance scores as CSV.

    Args:
        importance_scores: np.array of shape (num_channels,)
        channel_names: List or array of channel name strings
        output_dir: pathlib.Path to output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize scores to sum to 1
    total = importance_scores.sum()
    normalized = importance_scores / total if total > 0 else importance_scores

    # Create DataFrame
    df = pd.DataFrame(
        {
            "channel_idx": range(len(channel_names)),
            "channel_name": channel_names,
            "importance_score": importance_scores,
            "normalized_score": normalized,
        }
    )

    # Sort by importance (descending)
    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = output_dir / "channel_importance.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved feature importance CSV: {csv_path}")

    # Print ALL channels (not just top 10)
    print(f"\n{'=' * 70}")
    print("CHANNEL IMPORTANCE RANKING (ALL CHANNELS)")
    print(f"{'=' * 70}")
    print(
        f"{'Rank':<6}{'Idx':<6}{'Channel':<20}{'Raw Score':<15}{'Norm. Score':<15}{'Bar'}"
    )
    print("-" * 70)

    for rank, (i, row) in enumerate(df.iterrows()):
        bar_length = int(row["normalized_score"] * 50)
        bar = "█" * bar_length
        print(
            f"{rank + 1:<6}{row['channel_idx']:<6}{row['channel_name']:<20}"
            f"{row['importance_score']:<15.2f}{row['normalized_score']:<15.4f}{bar}"
        )


def get_checkpoint_paths(runs_dir, run_name, model_type):
    """
    Get list of checkpoint paths based on model_type.

    Args:
        model_type: "all" for all checkpoints, or specific name like "best", "final"

    Returns:
        List of tuples: [(checkpoint_path, checkpoint_name), ...]
    """
    # Try Lightning structure first (output/run_name/checkpoints/*.ckpt)
    lightning_dir = runs_dir / run_name / "checkpoints"

    if lightning_dir.exists():
        if model_type == "all":
            # All .ckpt files
            ckpts = sorted(lightning_dir.glob("*.ckpt"))
            ckpt_pairs = []
            for ckpt in ckpts:
                # Extract epoch number from filename like "run_name_epoch=366_val_loss=0.0048.ckpt"
                if "epoch=" in ckpt.name:
                    epoch_part = ckpt.name.split("epoch=")[1].split("_")[0]
                    ckpt_name = f"epoch_{epoch_part}"
                elif "last.ckpt" in ckpt.name:
                    ckpt_name = "last"
                else:
                    ckpt_name = ckpt.stem
                ckpt_pairs.append((ckpt, ckpt_name))
            return ckpt_pairs
        else:
            # Single checkpoint - look for best or last
            if model_type == "best":
                # Look for checkpoint with epoch in name
                ckpts = list(lightning_dir.glob("*epoch=*.ckpt"))
                if ckpts:
                    # Get the one with highest epoch number
                    best_ckpt = max(
                        ckpts,
                        key=lambda x: int(x.name.split("epoch=")[1].split("_")[0]),
                    )
                    epoch_num = best_ckpt.name.split("epoch=")[1].split("_")[0]
                    return [(best_ckpt, f"epoch_{epoch_num}")]
                # Fallback to last.ckpt
                last_ckpt = lightning_dir / "last.ckpt"
                if last_ckpt.exists():
                    return [(last_ckpt, "last")]
            else:
                # Specific checkpoint name
                ckpt = lightning_dir / f"{model_type}.ckpt"
                if ckpt.exists():
                    return [(ckpt, model_type)]
                # Try pattern matching
                matching_ckpts = list(lightning_dir.glob(f"*{model_type}*.ckpt"))
                if matching_ckpts:
                    return [(matching_ckpts[0], model_type)]

            raise FileNotFoundError(
                f"Checkpoint not found: {lightning_dir / model_type}"
            )

    # Fallback to original structure (runs/run_name/models/model_*.pt)
    models_dir = runs_dir / run_name / "models"

    if model_type == "all":
        # All model_*.pt files, but exclude "best" since it's a duplicate
        ckpts = sorted(models_dir.glob("model_*.pt"))
        ckpt_pairs = [(ckpt, ckpt.stem.replace("model_", "")) for ckpt in ckpts]
        # Filter out "best" checkpoint
        return [(path, name) for path, name in ckpt_pairs if name != "best"]
    else:
        # Single checkpoint
        ckpt = models_dir / f"model_{model_type}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return [(ckpt, model_type)]


def load_existing_checkpoint_results(comparison_csv_path):
    """
    Load existing checkpoint results from checkpoints_comparison.csv.

    Returns:
        dict: Mapping from checkpoint name to metrics dict, or empty dict if file doesn't exist
    """
    if not comparison_csv_path.exists():
        return {}

    df = pd.read_csv(comparison_csv_path)
    results = {}

    for _, row in df.iterrows():
        ckpt_name = row["checkpoint"]
        metrics = row.to_dict()
        del metrics["checkpoint"]
        results[ckpt_name] = metrics

    return results


def identify_best_checkpoint(results_df, has_ci, has_deb, metric_strategy="IoU"):
    """
    Identify the best checkpoint based on specified metric strategy.

    Args:
        results_df: DataFrame with checkpoint results
        has_ci: bool, whether CleanIce model is included
        has_deb: bool, whether Debris model is included
        metric_strategy: str, metric to use for determining best checkpoint

    Returns:
        tuple: (best_checkpoint_name, best_metric_value)
    """
    if has_ci and has_deb:
        # Merged CI + Debris case
        if metric_strategy == "average_IoU":
            # Use average of CI_IoU and Deb_IoU
            best_idx = (results_df["CI_IoU"] + results_df["Deb_IoU"]).idxmax()
            best_metric = (
                results_df.loc[best_idx, "CI_IoU"] + results_df.loc[best_idx, "Deb_IoU"]
            ) / 2
        elif metric_strategy == "CI_IoU":
            best_idx = results_df["CI_IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "CI_IoU"]
        elif metric_strategy == "Deb_IoU":
            best_idx = results_df["Deb_IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "Deb_IoU"]
        elif metric_strategy == "CI_Precision":
            best_idx = results_df["CI_P"].idxmax()
            best_metric = results_df.loc[best_idx, "CI_P"]
        elif metric_strategy == "Deb_Precision":
            best_idx = results_df["Deb_P"].idxmax()
            best_metric = results_df.loc[best_idx, "Deb_P"]
        else:
            # Default to average IoU
            best_idx = (results_df["CI_IoU"] + results_df["Deb_IoU"]).idxmax()
            best_metric = (
                results_df.loc[best_idx, "CI_IoU"] + results_df.loc[best_idx, "Deb_IoU"]
            ) / 2
    else:
        # Single model case
        if metric_strategy == "IoU":
            best_idx = results_df["IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "IoU"]
        elif metric_strategy == "Precision":
            best_idx = results_df["P"].idxmax()
            best_metric = results_df.loc[best_idx, "P"]
        elif metric_strategy == "Recall":
            best_idx = results_df["R"].idxmax()
            best_metric = results_df.loc[best_idx, "R"]
        else:
            # Default to IoU
            best_idx = results_df["IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "IoU"]

    best_checkpoint = results_df.loc[best_idx, "checkpoint"]
    return best_checkpoint, best_metric


# Prediction runner for a single checkpoint combination


def run_prediction(
    frame_ci,
    frame_deb,
    thr_ci,
    thr_deb,
    test_tiles,
    vis_mode,
    vis_maxw,
    preds_dir,
    has_ci,
    has_deb,
):
    """
    Run predictions on all test tiles for a single checkpoint combination.

    Returns:
        df_rows: List of per-tile metric rows
        acc: Dict of accumulated TP/FP/FN counts
    """
    df_rows = []
    acc = dict(
        ci_tp=0.0,
        ci_fp=0.0,
        ci_fn=0.0,
        db_tp=0.0,
        db_fp=0.0,
        db_fn=0.0,
    )

    for tile in tqdm(test_tiles, desc="Predicting"):
        name = tile.name
        base = tile.stem

        x_full = np.load(tile)
        y_full = np.load(tile.parent / name.replace("tiff", "mask")).astype(np.uint8)
        invalid = create_invalid_mask(x_full, y_full)
        valid = ~invalid

        prob_path = preds_dir / f"{base}_probs.npy"

        # ===============================================================
        # SINGLE MODEL
        # ===============================================================
        if has_ci ^ has_deb:
            frame = frame_ci if has_ci else frame_deb
            model_class = 1 if has_ci else 2
            model_name = "CleanIce" if has_ci else "Debris"
            thr = thr_ci if has_ci else thr_deb

            probs = get_probabilities(frame, x_full)  # (H,W,2)
            np.save(prob_path, probs)

            pred_bin = (probs[:, :, 1] >= thr).astype(np.uint8)

            # GT comparison using unified metrics
            P, R, iou, tp, fp, fn = calculate_binary_metrics(
                pred_bin, y_full, model_class, invalid
            )

            # accumulate
            if has_ci:
                acc["ci_tp"] += tp
                acc["ci_fp"] += fp
                acc["ci_fn"] += fn
            else:
                acc["db_tp"] += tp
                acc["db_fp"] += fp
                acc["db_fn"] += fn

            df_rows.append([name, P, R, iou])

            # Visualization label maps - use binary labeling for consistency
            y_gt = y_full.copy()
            y_gt[invalid] = 255

            # For binary visualization: convert to 0=NOT~class, 1=class, 255=mask
            y_gt_vis = np.zeros_like(y_full)
            y_gt_vis[valid & (y_full == model_class)] = 1
            y_gt_vis[valid & (y_full != model_class)] = 0
            y_gt_vis[invalid] = 255

            y_pred = np.zeros_like(y_full)
            y_pred[valid & (pred_bin == 1)] = 1  # class
            y_pred[valid & (pred_bin == 0)] = 0  # NOT~class
            y_pred[invalid] = 255

            # TP/FP/FN masks - use binary visualization labels
            gt_pos = (y_gt_vis == 1) & valid  # class pixels in GT
            pred_pos = (y_pred == 1) & valid  # class pixels in prediction
            tp_mask = gt_pos & pred_pos
            fp_mask = (~gt_pos) & pred_pos
            fn_mask = gt_pos & (~pred_pos)

        # ===============================================================
        # MERGED CI + DEBRIS BINARY MODELS
        # ===============================================================
        else:
            prob_ci = get_probabilities(frame_ci, x_full)
            prob_db = get_probabilities(frame_deb, x_full)

            merged, probs = merge_ci_debris(prob_ci, prob_db, thr_ci, thr_deb)
            np.save(prob_path, probs)

            y_gt = y_full.copy()
            y_gt[invalid] = 255

            merged_vis = merged.copy()
            merged_vis[invalid] = 255

            # CleanIce metrics
            Pci, Rci, Ici, tp, fp, fn = get_pr_iou(
                (merged[valid] == 1).astype(np.uint8),
                (y_full[valid] == 1).astype(np.uint8),
            )
            acc["ci_tp"] += tp
            acc["ci_fp"] += fp
            acc["ci_fn"] += fn

            # Debris metrics
            Pdb, Rdb, Idb, tp, fp, fn = get_pr_iou(
                (merged[valid] == 2).astype(np.uint8),
                (y_full[valid] == 2).astype(np.uint8),
            )
            acc["db_tp"] += tp
            acc["db_fp"] += fp
            acc["db_fn"] += fn

            df_rows.append([name, Pci, Rci, Ici, Pdb, Rdb, Idb])

            # TP/FP/FN full 3-class
            tp_mask = (merged == y_full) & (~invalid) & (y_full != 0)
            fp_mask = (merged != y_full) & (~invalid) & (merged != 0)
            fn_mask = (merged != y_full) & (~invalid) & (y_full != 0)

            # Confidence = prob of predicted class
            conf_map = probs[
                np.arange(probs.shape[0])[:, None],
                np.arange(probs.shape[1])[None, :],
                merged,
            ]
            conf_map[invalid] = 0

        # ===============================================================
        # Resize if needed
        # ===============================================================

    return df_rows, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run glacier mapping predictions")

    # Model arguments
    parser.add_argument(
        "--ci-run-name",
        type=str,
        help="CleanIce run name (will look in output/ directory)",
    )
    parser.add_argument(
        "--ci-threshold",
        type=float,
        default=0.5,
        help="CleanIce prediction threshold (default: 0.5)",
    )
    parser.add_argument(
        "--deb-run-name",
        type=str,
        help="Debris run name (will look in output/ directory)",
    )
    parser.add_argument(
        "--deb-threshold",
        type=float,
        default=0.5,
        help="Debris prediction threshold (default: 0.5)",
    )

    # Feature importance
    parser.add_argument(
        "--feature-importance",
        action="store_true",
        help="Compute feature importance (saliency)",
    )
    parser.add_argument(
        "--fi-samples",
        type=int,
        help="Number of samples for feature importance (default: all)",
    )
    parser.add_argument(
        "--fi-target-class",
        type=int,
        default=1,
        help="Target class for saliency (default: 1)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use (default: 0)",
    )
    parser.add_argument(
        "--server",
        type=str,
        help="Server name from servers.yaml (default: auto-detect)",
    )

    args = parser.parse_args()

    result = main_prediction_logic(args)

    if result["status"] == "SUCCESS":
        print("\n✓ Prediction complete.")
    else:
        print(f"\n✗ Prediction failed: {result.get('error', 'Unknown error')}")
        exit(1)
