#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
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
    if processed_data_path is not None:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

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

        temp_ckpt = Path(checkpoint_path).parent / "temp_modified.ckpt"
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


def clean_run_name(run_name: str) -> str:
    prefixes_to_remove = [
        "ablation_ci_",
        "ablation_dci_",
        "ablation_mc_",
        "ablation2_ci_",
        "ablation2_dci_",
        "ablation2_mc_",
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
    server_output_path = Path(server_config["output_path"])
    best_checkpoint = None
    best_val_loss = float("inf")

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

        ckpts = list(checkpoints_dir.glob("*epoch=*.ckpt"))
        for ckpt in ckpts:
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
    gpu = gpu_id if gpu_id is not None else 0

    has_ci = ci_checkpoint is not None
    has_deb = deb_checkpoint is not None

    if not (has_ci or has_deb):
        raise ValueError("Must provide at least one checkpoint")

    frame_ci = None
    frame_deb = None

    if has_ci and ci_checkpoint is not None:
        frame_ci = load_lightning_module(ci_checkpoint, gpu, processed_data_path)

    if has_deb and deb_checkpoint is not None:
        frame_deb = load_lightning_module(deb_checkpoint, gpu, processed_data_path)

    if frame_ci is not None:
        data_dir = pathlib.Path(frame_ci.processed_dir)
    elif frame_deb is not None:
        data_dir = pathlib.Path(frame_deb.processed_dir)
    else:
        raise RuntimeError("No valid model loaded for test tiles")
    test_tiles = sorted(pathlib.Path(data_dir, "test").glob("tiff*"))

    preds_dir = output_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running on {len(test_tiles)} test tiles...")

    df_rows, acc = run_prediction(
        frame_ci,
        frame_deb,
        ci_threshold,
        deb_threshold,
        test_tiles,
        None,
        None,
        preds_dir,
        has_ci,
        has_deb,
    )

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

    ckpt_name = "prediction"
    df_rows_with_ckpt = [[ckpt_name] + row for row in df_rows]

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

    df = pd.DataFrame(df_rows_with_ckpt, columns=pd.Index(columns))
    df.to_csv(output_dir / "metrics.csv", index=False)

    if feature_importance and (frame_ci or frame_deb):
        saliency_frame = frame_ci if frame_ci else frame_deb
        if saliency_frame:
            print("Computing feature importance...")

            fi_output_dir = output_dir / "feature_importance"
            fi_output_dir.mkdir(parents=True, exist_ok=True)

            importance_scores = compute_feature_importance(
                saliency_frame,
                test_tiles,
                target_class_idx=1,
                num_samples=fi_samples,
            )

            use_channels = saliency_frame.use_channels
            BAND_NAMES = load_band_names(saliency_frame.processed_dir)
            channel_names = BAND_NAMES[use_channels]

            save_feature_importance_results(
                importance_scores, channel_names, fi_output_dir
            )

    del frame_ci, frame_deb
    cleanup_gpu_memory(synchronize=False)

    return {"metrics": metrics, "output_dir": output_dir, "status": "SUCCESS"}


def main_prediction_logic(args) -> dict:
    try:
        servers_config = yaml.safe_load(Path("configs/servers.yaml").read_text())

        if hasattr(args, "server") and args.server:
            current_server = args.server
            if current_server not in servers_config:
                raise ValueError(f"Server '{current_server}' not found in servers.yaml")
        else:
            current_server = None
            import socket

            hostname = socket.gethostname()

            for server_name, server_config in servers_config.items():
                if hostname == server_config.get("hostname", ""):
                    current_server = server_name
                    break

            if current_server is None:
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

        if hasattr(args, "output_dir") and args.output_dir:
            out_root = Path(args.output_dir)
        else:
            if args.ci_run_name and args.deb_run_name:
                base_run_name = clean_run_name(args.ci_run_name)
            elif args.ci_run_name:
                base_run_name = clean_run_name(args.ci_run_name)
            else:
                base_run_name = clean_run_name(args.deb_run_name)

            server_output_path = Path(server_config["output_path"])
            out_root = server_output_path.parent / "output_predictions" / base_run_name

        out_root.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {out_root}")

        processed_data_path = server_config["processed_data_path"]
        if "ablation" in (args.ci_run_name or "") or "ablation" in (
            args.deb_run_name or ""
        ):
            processed_data_path = f"{processed_data_path}/gen_robust_comprehensive"

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


def compute_feature_importance(
    module,
    test_tiles,
    target_class_idx,
    num_samples=None,
):
    module.eval()
    device = module.device
    use_channels = module.use_channels
    num_channels = len(use_channels)

    if num_samples is not None and num_samples < len(test_tiles):
        tiles_to_use = test_tiles[:num_samples]
    else:
        tiles_to_use = test_tiles

    print(f"Computing feature importance using {len(tiles_to_use)} test samples...")
    print(f"Target class index: {target_class_idx}")

    channel_gradients = np.zeros(num_channels, dtype=np.float64)

    for tile_path in tqdm(tiles_to_use, desc="Computing saliency"):
        x_full = np.load(tile_path)
        x = x_full[:, :, use_channels]

        x_norm = module.normalize(x)

        x_tensor = torch.from_numpy(x_norm).float().to(device).unsqueeze(0)
        x_tensor = x_tensor.permute(0, 3, 1, 2)
        x_tensor.requires_grad_(True)

        logits = module(x_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)

        target_prob = probs[0, target_class_idx, :, :].mean()

        target_prob.backward()

        channel_grad = np.zeros(len(use_channels))
        if x_tensor.grad is not None:
            grads = x_tensor.grad.data.abs()
            channel_grad = (
                grads.sum(dim=(2, 3)).cpu().numpy()[0]
            )

        channel_gradients += channel_grad

        x_tensor.grad = None
        del x_tensor, logits, probs

    channel_gradients /= len(tiles_to_use)

    return channel_gradients


def save_feature_importance_results(importance_scores, channel_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    total = importance_scores.sum()
    normalized = importance_scores / total if total > 0 else importance_scores

    df = pd.DataFrame(
        {
            "channel_idx": range(len(channel_names)),
            "channel_name": channel_names,
            "importance_score": importance_scores,
            "normalized_score": normalized,
        }
    )

    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)

    csv_path = output_dir / "channel_importance.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved feature importance CSV: {csv_path}")

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
    lightning_dir = runs_dir / run_name / "checkpoints"

    if lightning_dir.exists():
        if model_type == "all":
            ckpts = sorted(lightning_dir.glob("*.ckpt"))
            ckpt_pairs = []
            for ckpt in ckpts:
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
            if model_type == "best":
                ckpts = list(lightning_dir.glob("*epoch=*.ckpt"))
                if ckpts:
                    best_ckpt = max(
                        ckpts,
                        key=lambda x: int(x.name.split("epoch=")[1].split("_")[0]),
                    )
                    epoch_num = best_ckpt.name.split("epoch=")[1].split("_")[0]
                    return [(best_ckpt, f"epoch_{epoch_num}")]
                last_ckpt = lightning_dir / "last.ckpt"
                if last_ckpt.exists():
                    return [(last_ckpt, "last")]
            else:
                ckpt = lightning_dir / f"{model_type}.ckpt"
                if ckpt.exists():
                    return [(ckpt, model_type)]
                matching_ckpts = list(lightning_dir.glob(f"*{model_type}*.ckpt"))
                if matching_ckpts:
                    return [(matching_ckpts[0], model_type)]

            raise FileNotFoundError(
                f"Checkpoint not found: {lightning_dir / model_type}"
            )

    models_dir = runs_dir / run_name / "models"

    if model_type == "all":
        ckpts = sorted(models_dir.glob("model_*.pt"))
        ckpt_pairs = [(ckpt, ckpt.stem.replace("model_", "")) for ckpt in ckpts]
        return [(path, name) for path, name in ckpt_pairs if name != "best"]
    else:
        ckpt = models_dir / f"model_{model_type}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return [(ckpt, model_type)]


def load_existing_checkpoint_results(comparison_csv_path):
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
    if has_ci and has_deb:
        if metric_strategy == "average_IoU":
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
            best_idx = (results_df["CI_IoU"] + results_df["Deb_IoU"]).idxmax()
            best_metric = (
                results_df.loc[best_idx, "CI_IoU"] + results_df.loc[best_idx, "Deb_IoU"]
            ) / 2
    else:
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
            best_idx = results_df["IoU"].idxmax()
            best_metric = results_df.loc[best_idx, "IoU"]

    best_checkpoint = results_df.loc[best_idx, "checkpoint"]
    return best_checkpoint, best_metric


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

        if has_ci ^ has_deb:
            frame = frame_ci if has_ci else frame_deb
            model_class = 1 if has_ci else 2
            thr = thr_ci if has_ci else thr_deb

            probs = get_probabilities(frame, x_full)
            np.save(prob_path, probs)

            pred_mask = probs[:, :, 1] >= thr
            pred_filled = binary_fill_holes(pred_mask)
            pred_bin = pred_filled.astype(np.uint8)

            P, R, iou, tp, fp, fn = calculate_binary_metrics(
                pred_bin, y_full, model_class, invalid
            )

            if has_ci:
                acc["ci_tp"] += tp
                acc["ci_fp"] += fp
                acc["ci_fn"] += fn
            else:
                acc["db_tp"] += tp
                acc["db_fp"] += fp
                acc["db_fn"] += fn

            df_rows.append([name, P, R, iou])

            y_gt = y_full.copy()
            y_gt[invalid] = 255

            y_gt_vis = np.zeros_like(y_full)
            y_gt_vis[valid & (y_full == model_class)] = 1
            y_gt_vis[valid & (y_full != model_class)] = 0
            y_gt_vis[invalid] = 255

            y_pred = np.zeros_like(y_full)
            y_pred[valid & (pred_bin == 1)] = 1
            y_pred[valid & (pred_bin == 0)] = 0
            y_pred[invalid] = 255

        else:
            prob_ci = get_probabilities(frame_ci, x_full)
            prob_db = get_probabilities(frame_deb, x_full)

            merged, probs = merge_ci_debris(prob_ci, prob_db, thr_ci, thr_deb)
            np.save(prob_path, probs)

            y_gt = y_full.copy()
            y_gt[invalid] = 255

            merged_vis = merged.copy()
            merged_vis[invalid] = 255

            Pci, Rci, Ici, tp, fp, fn = get_pr_iou(
                (merged[valid] == 1).astype(np.uint8),
                (y_full[valid] == 1).astype(np.uint8),
            )
            acc["ci_tp"] += tp
            acc["ci_fp"] += fp
            acc["ci_fn"] += fn

            Pdb, Rdb, Idb, tp, fp, fn = get_pr_iou(
                (merged[valid] == 2).astype(np.uint8),
                (y_full[valid] == 2).astype(np.uint8),
            )
            acc["db_tp"] += tp
            acc["db_fp"] += fp
            acc["db_fn"] += fn

            df_rows.append([name, Pci, Rci, Ici, Pdb, Rdb, Idb])

            conf_map = probs[
                np.arange(probs.shape[0])[:, None],
                np.arange(probs.shape[1])[None, :],
                merged,
            ]
            conf_map[invalid] = 0

    return df_rows, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run glacier mapping predictions")

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
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Explicit output directory for predictions (overrides default naming)",
    )

    args = parser.parse_args()

    result = main_prediction_logic(args)

    if result["status"] == "SUCCESS":
        print("\nPrediction complete.")
    else:
        print(f"\nPrediction failed: {result.get('error', 'Unknown error')}")
        exit(1)
