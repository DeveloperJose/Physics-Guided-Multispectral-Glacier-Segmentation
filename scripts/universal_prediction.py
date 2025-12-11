#!/usr/bin/env python3
"""
Universal Prediction Script
Runs predictions on all runs for a given generation, pairing CI and DCI models.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import functions from predict.py
sys.path.append(str(Path(__file__).parent))
from predict import clean_run_name


def get_available_gpus() -> List[int]:
    """Detect available GPUs."""
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        else:
            return []
    except ImportError:
        return []


def extract_generation_runs(generation_id: str) -> List[str]:
    """Extract all run names for a generation from local output directory."""
    print(f"Extracting {generation_id} runs from local output directory...")

    output_dir = Path("output")
    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist")
        return []

    generation_runs = []
    for item in output_dir.iterdir():
        if item.is_dir() and generation_id in item.name:
            generation_runs.append(item.name)

    if not generation_runs:
        print(f"\n*** NO {generation_id.upper()} RUNS FOUND ***")
        return []

    print(f"\n*** TOTAL {generation_id.upper()} RUNS FOUND: {len(generation_runs)} ***")
    return sorted(generation_runs)


def group_runs_by_base_name(run_names: List[str]) -> Dict[str, Dict[str, str]]:
    """Group runs by base name (after removing ci_/dci_ prefixes)."""
    grouped = {}

    for run_name in run_names:
        base_name = clean_run_name(run_name)

        if base_name not in grouped:
            grouped[base_name] = {"ci": None, "dci": None}

        # Determine if this is CI or DCI run
        if "_ci_" in run_name or "ci_" in run_name:
            grouped[base_name]["ci"] = run_name
        elif "_dci_" in run_name or "dci_" in run_name:
            grouped[base_name]["dci"] = run_name

    return grouped


def parse_prediction_metrics(csv_path: Path) -> Dict[str, float]:
    """Parse metrics from prediction CSV output."""
    try:
        if not csv_path.exists():
            return {}

        # Read the metrics CSV
        df = pd.read_csv(csv_path)

        # Look for TOTAL row which contains summary metrics
        total_rows = df[df["tile"] == "TOTAL"]
        if total_rows.empty:
            return {}

        total_row = total_rows.iloc[-1]  # Use last TOTAL row

        metrics = {}
        for col in total_row.index:
            col_str = str(col).lower()
            if "precision" in col_str:
                if "cleanice" in col_str:
                    metrics["CI_P"] = float(total_row[col])
                elif "debris" in col_str:
                    metrics["Deb_P"] = float(total_row[col])
            elif "recall" in col_str:
                if "cleanice" in col_str:
                    metrics["CI_R"] = float(total_row[col])
                elif "debris" in col_str:
                    metrics["Deb_R"] = float(total_row[col])
            elif "iou" in col_str:
                if "cleanice" in col_str:
                    metrics["CI_IoU"] = float(total_row[col])
                elif "debris" in col_str:
                    metrics["Deb_IoU"] = float(total_row[col])

        return metrics

    except Exception as e:
        print(f"Error parsing metrics from {csv_path}: {e}")
        return {}


def run_single_prediction(
    base_name: str,
    ci_run: Optional[str],
    dci_run: Optional[str],
    gpu_id: int,
    output_base: Path,
    server_name: str,
) -> Tuple[str, Dict[str, float | str]]:
    """Run prediction for a single base name and return metrics."""

    # Build command for predict.py
    cmd = ["uv", "run", "python", "scripts/predict.py"]

    if ci_run:
        cmd.extend(["--ci-run-name", ci_run])
    if dci_run:
        cmd.extend(["--deb-run-name", dci_run])

    cmd.extend(["--gpu", str(gpu_id)])
    cmd.extend(["--server", server_name])

    # Set environment variable for GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Run prediction
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,  # 30 minutes timeout
        )

        if result.returncode != 0:
            print(f"FAILED: {base_name} - Prediction script failed")
            print(f"Error: {result.stderr}")
            return base_name, {"status": "FAILED"}

        # Parse metrics from generated CSV
        metrics_csv = output_base / base_name / "preds" / "metrics.csv"
        metrics = parse_prediction_metrics(metrics_csv)
        metrics["status"] = "SUCCESS"  # type: ignore

        return base_name, metrics  # type: ignore

    except subprocess.TimeoutExpired:
        print(f"FAILED: {base_name} - Prediction timed out")
        return base_name, {"status": "FAILED"}  # type: ignore
    except Exception as e:
        print(f"FAILED: {base_name} - {e}")
        return base_name, {"status": "FAILED"}  # type: ignore


def cleanup_intermediate_dirs(output_dirs: List[Path]) -> None:
    """Clean up intermediate prediction directories."""
    for dir_path in output_dirs:
        try:
            if dir_path.exists():
                import shutil

                shutil.rmtree(dir_path)
                print(f"Cleaned up: {dir_path}")
        except Exception as e:
            print(f"Failed to clean up {dir_path}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Universal prediction script for glacier mapping"
    )
    parser.add_argument("generation", help="Generation name (e.g., ablation)")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up intermediate output directories",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--gpu-list",
        type=int,
        nargs="+",
        help="List of GPU IDs to use (default: auto-detect)",
    )

    args = parser.parse_args()

    # Detect current server
    servers_config = yaml.safe_load(Path("configs/servers.yaml").read_text())
    current_server = None
    import socket

    hostname = socket.gethostname()
    for server_name, server_config in servers_config.items():
        if hostname == server_config.get("hostname", ""):
            current_server = server_name
            break

    if current_server is None:
        # Fallback to desktop for local development
        current_server = "desktop"
        print(f"Could not detect server, defaulting to: {current_server}")
    else:
        print(f"Detected server: {current_server}")

    # Extract runs from MLflow
    run_names = extract_generation_runs(args.generation)
    if not run_names:
        print(f"No runs found for generation: {args.generation}")
        return

    # Group runs by base name
    grouped_runs = group_runs_by_base_name(run_names)
    print(f"Found {len(grouped_runs)} unique run groups")

    # Prepare prediction jobs
    jobs = []
    output_dirs = []

    for base_name, runs in grouped_runs.items():
        ci_run = runs["ci"]
        dci_run = runs["dci"]

        # Skip if neither CI nor DCI exists
        if not ci_run and not dci_run:
            print(f"Skipping {base_name}: no CI or DCI runs found")
            continue

        # Create output directory
        output_dir = Path("output_predictions") / base_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.append(output_dir)

        jobs.append((base_name, ci_run, dci_run, output_dir, current_server))

    if not jobs:
        print("No valid prediction jobs found")
        return

    # GPU management
    available_gpus = args.gpu_list if args.gpu_list else get_available_gpus()
    if not available_gpus:
        available_gpus = [0]  # Fallback to CPU

    print(f"Available GPUs: {available_gpus}")
    print(f"Running {len(jobs)} prediction jobs...")

    # Determine number of workers
    max_workers = min(args.workers, len(available_gpus), len(jobs))
    print(f"Using {max_workers} workers")

    # Run predictions in parallel
    results = {}

    if max_workers == 1:
        # Sequential execution
        for base_name, ci_run, dci_run, output_dir, server_name in tqdm(
            jobs, desc="Running predictions"
        ):
            base_name, metrics = run_single_prediction(
                base_name, ci_run, dci_run, 0, output_dir, server_name
            )
            results[base_name] = metrics
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_base = {}

            for i, (base_name, ci_run, dci_run, output_dir, server_name) in enumerate(
                jobs
            ):
                gpu_id = available_gpus[i % len(available_gpus)]
                future = executor.submit(
                    run_single_prediction,
                    base_name,
                    ci_run,
                    dci_run,
                    gpu_id,
                    output_dir,
                    server_name,
                )
                future_to_base[future] = base_name

            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_base),
                total=len(jobs),
                desc="Running predictions",
            ):
                base_name = future_to_base[future]
                try:
                    result_base_name, metrics = future.result()
                    results[result_base_name] = metrics
                except Exception as e:
                    print(f"Error in prediction for {base_name}: {e}")
                    results[base_name] = {"status": "FAILED"}

    # Generate summary table
    print("\n" + "=" * 80)
    print("UNIVERSAL PREDICTION SUMMARY")
    print("=" * 80)

    # Prepare table data
    table_data = []
    for base_name in sorted(results.keys()):
        metrics = results[base_name]

        if metrics.get("status") == "FAILED":
            table_data.append(
                {
                    "Run Name": base_name,
                    "DCI IoU": "-",
                    "DCI Prec": "-",
                    "DCI Rec": "-",
                    "CI IoU": "-",
                    "CI Prec": "-",
                    "CI Rec": "-",
                }
            )
        else:
            table_data.append(
                {
                    "Run Name": base_name,
                    "DCI IoU": f"{metrics.get('Deb_IoU', 0):.4f}"
                    if metrics.get("Deb_IoU") is not None
                    else "-",
                    "DCI Prec": f"{metrics.get('Deb_P', 0):.4f}"
                    if metrics.get("Deb_P") is not None
                    else "-",
                    "DCI Rec": f"{metrics.get('Deb_R', 0):.4f}"
                    if metrics.get("Deb_R") is not None
                    else "-",
                    "CI IoU": f"{metrics.get('CI_IoU', 0):.4f}"
                    if metrics.get("CI_IoU") is not None
                    else "-",
                    "CI Prec": f"{metrics.get('CI_P', 0):.4f}"
                    if metrics.get("CI_P") is not None
                    else "-",
                    "CI Rec": f"{metrics.get('CI_R', 0):.4f}"
                    if metrics.get("CI_R") is not None
                    else "-",
                }
            )

    # Print table
    summary_df = pd.DataFrame(table_data)
    print(summary_df.to_string(index=False))

    # Save to CSV
    output_csv = (
        Path("output_predictions") / f"{args.generation}_prediction_summary.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"\nSummary saved to: {output_csv}")

    # Cleanup intermediate directories
    if not args.no_cleanup:
        print("\nCleaning up intermediate directories...")
        cleanup_intermediate_dirs(output_dirs)

    print(f"\n✓ Universal prediction complete for {args.generation}")


if __name__ == "__main__":
    main()
