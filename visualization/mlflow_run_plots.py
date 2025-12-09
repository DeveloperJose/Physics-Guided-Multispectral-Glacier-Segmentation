#!/usr/bin/env python3
"""
Plot key training curves for an MLflow run (by run_name).

Produces:
  - Loss curves (train/val/best, velocity loss if present)
  - Learning rate
  - Sigma parameters (dice/boundary/velocity)
  - Per-class IoU/precision/recall (val/test if logged)

Default tracking URI: https://mlflow.developerjose.duckdns.org/
Outputs: visualization/output/mlflow_runs/<run_name>/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MLflow run metrics.")
    parser.add_argument("--run-name", required=True, help="MLflow run_name (tag mlflow.runName).")
    parser.add_argument(
        "--tracking-uri",
        default="https://mlflow.developerjose.duckdns.org/",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir for plots (default: visualization/output/mlflow_runs/<run_name>)",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional experiment name filter (otherwise search all).",
    )
    return parser.parse_args()


def find_run_id(client, run_name: str, experiment_name: str | None) -> str:
    if experiment_name:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise ValueError(f"Experiment not found: {experiment_name}")
        experiment_ids = [exp.experiment_id]
    else:
        all_exps = client.search_experiments()
        experiment_ids = [e.experiment_id for e in all_exps]
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=5,
    )
    if not runs:
        raise ValueError(f"No run found with run_name={run_name}")
    if len(runs) > 1:
        logger.warning("Multiple runs found; using the first one.")
    return runs[0].info.run_id


def fetch_metric_series(client, run_id: str, metric_names: List[str]) -> Dict[str, Tuple[List[int], List[float]]]:
    series: Dict[str, Tuple[List[int], List[float]]] = {}
    for m in metric_names:
        try:
            hist = client.get_metric_history(run_id, m)
        except Exception:
            continue
        if not hist:
            continue
        steps = [h.step for h in hist]
        vals = [h.value for h in hist]
        series[m] = (steps, vals)
    return series


def plot_lines(series: Dict[str, Tuple[List[int], List[float]]], title: str, ylabel: str, out_path: Path):
    if not series:
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for name, (steps, vals) in series.items():
        ax.plot(steps, vals, label=name)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def discover_metric_names(client, run_id: str) -> List[str]:
    """Return all metric keys stored for a run by querying search_runs columns."""
    run = client.get_run(run_id)
    exp_id = run.info.experiment_id
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"run_id = '{run_id}'",
        max_results=1,
    )
    if not runs:
        return []
    info = runs[0].data.metrics
    return list(info.keys())


def downsample_to_epochs(series: Dict[str, Tuple[List[int], List[float]]]) -> Dict[str, Tuple[List[int], List[float]]]:
    """Group by epoch (int step) using last value in that epoch."""
    out: Dict[str, Tuple[List[int], List[float]]] = {}
    for name, (steps, vals) in series.items():
        epoch_map = {}
        for s, v in zip(steps, vals):
            epoch_map[int(s)] = v  # last value wins
        epochs = sorted(epoch_map.keys())
        out[name] = (epochs, [epoch_map[e] for e in epochs])
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as e:
        raise SystemExit(f"mlflow not installed: {e}")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    run_id = find_run_id(client, args.run_name, args.experiment_name)
    logger.info("Found run_id=%s for run_name=%s", run_id, args.run_name)

    out_dir = Path(args.output_dir) if args.output_dir else Path("visualization/output/mlflow_runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    available = discover_metric_names(client, run_id)
    logger.info("Discovered %d metric keys: %s", len(available), ", ".join(sorted(available)))

    def filter_by_substrings(subs: List[str]) -> List[str]:
        res = []
        for m in available:
            low = m.lower()
            if any(s in low for s in subs):
                res.append(m)
        return res

    # Metrics to fetch (auto-filter by substrings to handle casing/variants)
    loss_metrics = filter_by_substrings(["loss"])
    lr_metrics = filter_by_substrings(["learning_rate", "lr-"])
    sigma_metrics = [m for m in available if m.lower().startswith("sigma")]
    iou_metrics = filter_by_substrings(["iou"])
    prec_metrics = filter_by_substrings(["precision"])
    rec_metrics = filter_by_substrings(["recall"])

    loss_series = fetch_metric_series(client, run_id, loss_metrics)
    lr_series = fetch_metric_series(client, run_id, lr_metrics)
    sigma_series = fetch_metric_series(client, run_id, sigma_metrics)
    iou_series = fetch_metric_series(client, run_id, iou_metrics)
    prec_series = fetch_metric_series(client, run_id, prec_metrics)
    rec_series = fetch_metric_series(client, run_id, rec_metrics)

    # Step-based plots
    plot_lines(loss_series, "Loss curves (step)", "Loss", out_dir / "loss_step.png")
    plot_lines(lr_series, "Learning rate (step)", "LR", out_dir / "learning_rate_step.png")
    plot_lines(sigma_series, "Sigma parameters (step)", "Sigma", out_dir / "sigma_step.png")
    plot_lines(iou_series, "Per-class IoU (step)", "IoU", out_dir / "iou_step.png")
    plot_lines(prec_series, "Per-class Precision (step)", "Precision", out_dir / "precision_step.png")
    plot_lines(rec_series, "Per-class Recall (step)", "Recall", out_dir / "recall_step.png")

    # Epoch-based plots
    plot_lines(downsample_to_epochs(loss_series), "Loss curves (epoch)", "Loss", out_dir / "loss_epoch.png")
    plot_lines(downsample_to_epochs(lr_series), "Learning rate (epoch)", "LR", out_dir / "learning_rate_epoch.png")
    plot_lines(downsample_to_epochs(sigma_series), "Sigma parameters (epoch)", "Sigma", out_dir / "sigma_epoch.png")
    plot_lines(downsample_to_epochs(iou_series), "Per-class IoU (epoch)", "IoU", out_dir / "iou_epoch.png")
    plot_lines(downsample_to_epochs(prec_series), "Per-class Precision (epoch)", "Precision", out_dir / "precision_epoch.png")
    plot_lines(downsample_to_epochs(rec_series), "Per-class Recall (epoch)", "Recall", out_dir / "recall_epoch.png")

    logger.info("Saved plots to %s", out_dir)


if __name__ == "__main__":
    main()
