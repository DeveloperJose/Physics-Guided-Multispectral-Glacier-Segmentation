#!/usr/bin/env python3
"""
Universal Generation Analysis Script
Works for any generation (gen1-6) with comprehensive data extraction and analysis.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import mlflow


def extract_generation_data(generation_id):
    """Extract all runs for a specific generation from MLflow."""

    print(f"Extracting {generation_id} data from MLflow...")
    mlflow.set_tracking_uri("https://mlflow.developerjose.duckdns.org")

    # Get all experiments
    experiments = mlflow.search_experiments()
    print(f"Found {len(experiments)} experiments")

    generation_runs = []
    total_runs = 0

    for exp in experiments:
        try:
            # Search for runs in this experiment
            runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            total_runs += len(runs_df)

            # Check if run name column exists
            run_name_col = "tags.mlflow.runName"
            if run_name_col not in runs_df.columns:
                continue

            # Filter for generation runs
            mask = runs_df[run_name_col].str.contains(
                generation_id, case=False, na=False
            )
            gen_runs = runs_df[mask].copy()

            if len(gen_runs) > 0:
                # Add experiment info
                gen_runs["experiment_name"] = exp.name
                gen_runs["experiment_id"] = exp.experiment_id
                generation_runs.append(gen_runs)
                print(f"  {exp.name}: {len(gen_runs)} {generation_id} runs")

        except Exception as e:
            print(f"  Error searching {exp.name}: {e}")
            continue

    if not generation_runs:
        print(f"\n*** NO {generation_id.upper()} RUNS FOUND ***")
        return pd.DataFrame()

    # Combine all generation runs
    all_gen = pd.concat(generation_runs, ignore_index=True)
    print(f"\n*** TOTAL {generation_id.upper()} RUNS FOUND: {len(all_gen)} ***")

    return all_gen


def classify_run(run_name):
    """Classify run based on name patterns."""

    name = str(run_name).lower()

    # Task classification
    if "ci" in name or "clean_ice" in name:
        task = "clean_ice"
    elif "dci" in name or "debris_ice" in name:
        task = "debris_ice"
    elif "multi" in name:
        task = "multiclass"
    else:
        task = "unknown"

    # Server classification
    if "frodo" in name:
        server = "frodo"
    elif "bilbo" in name:
        server = "bilbo"
    elif "desktop" in name:
        server = "desktop"
    else:
        server = "unknown"

    # Configuration type classification
    if "physics" in name:
        config_type = "physics"
    elif "velocity" in name:
        config_type = "velocity"
    elif "synthesis" in name:
        config_type = "synthesis"
    elif "baseline" in name or "base" in name:
        config_type = "baseline"
    else:
        config_type = "unknown"

    # Window size classification
    if "w512" in name:
        window_size = 512
    elif "w256" in name:
        window_size = 256
    else:
        window_size = None

    return {
        "task": task,
        "server": server,
        "config_type": config_type,
        "window_size": window_size,
    }


def extract_error_logs(run_id, run_name):
    """Attempt to extract error logs from MLflow artifacts."""

    try:
        # Try to access error logs from artifacts
        # This is a placeholder - actual implementation would depend on MLflow artifact access
        return {"error_logs_found": False, "error_details": None}
    except Exception as e:
        return {
            "error_logs_found": False,
            "error_details": f"Could not access error logs: {e}",
        }


def calculate_timing_analysis(run_data):
    """Calculate comprehensive timing analysis for a run."""

    if pd.isna(run_data["start_time"]) or pd.isna(run_data["end_time"]):
        return {"duration_hours": None, "per_epoch_timing": None}

    duration_ms = run_data["end_time"] - run_data["start_time"]
    duration_hours = float(duration_ms) / (1000 * 60 * 60)

    # Try to get epoch-level timing
    per_epoch_timing = None

    # Check if we have epoch data
    if "metrics.epoch" in run_data and pd.notna(run_data["metrics.epoch"]):
        final_epoch = run_data["metrics.epoch"]
        if final_epoch > 0:
            per_epoch_avg = duration_hours / float(final_epoch)
            per_epoch_timing = {
                "total_epochs": int(final_epoch),
                "average_minutes_per_epoch": per_epoch_avg * 60,
                "training_only_estimate": per_epoch_avg * 0.8,  # Rough estimate
                "with_test_estimate": per_epoch_avg,
            }

    return {"duration_hours": duration_hours, "per_epoch_timing": per_epoch_timing}


def analyze_performance(run_data):
    """Analyze performance metrics for a run."""

    performance = {
        "final_metrics": {},
        "best_metrics": {},
        "improvement_metrics": {},
        "all_metrics": {},
    }

    # Extract all available metrics
    metric_cols = [col for col in run_data.index if col.startswith("metrics.")]

    for col in metric_cols:
        if pd.notna(run_data[col]):
            metric_name = col.replace("metrics.", "")
            performance["all_metrics"][metric_name] = float(run_data[col])

            # Categorize metrics
            if "best_" in metric_name:
                performance["best_metrics"][metric_name] = float(run_data[col])
            elif metric_name in ["epoch", "val_loss", "train_loss_epoch"]:
                performance["final_metrics"][metric_name] = float(run_data[col])

    # Calculate improvement metrics
    if (
        "best_val_loss" in performance["best_metrics"]
        and "val_loss" in performance["final_metrics"]
    ):
        improvement = (
            performance["best_metrics"]["best_val_loss"]
            - performance["final_metrics"]["val_loss"]
        )
        performance["improvement_metrics"]["loss_improvement"] = float(improvement)

    # IoU improvements
    iou_types = ["CleanIce_iou", "DebrisIce_iou", "Multiclass_iou"]
    for iou_type in iou_types:
        best_key = f"best_val_{iou_type}"
        final_key = f"val_{iou_type}"

        if (
            best_key in performance["best_metrics"]
            and final_key in performance["final_metrics"]
        ):
            improvement = (
                performance["final_metrics"][final_key]
                - performance["best_metrics"][best_key]
            )
            performance["improvement_metrics"][f"{iou_type}_improvement"] = float(
                improvement
            )

    return performance


def analyze_training_behavior(run_data, performance):
    """Analyze training behavior patterns."""

    behavior = {
        "overfitting_indicator": False,
        "early_stopped": False,
        "convergence_pattern": "unknown",
    }

    # Overfitting detection
    if (
        "best_val_loss" in performance["best_metrics"]
        and "val_loss" in performance["final_metrics"]
    ):
        final_loss = performance["final_metrics"]["val_loss"]
        best_loss = performance["best_metrics"]["best_val_loss"]

        # If final loss is significantly worse than best, likely overfitting
        if final_loss > best_loss * 1.05:  # 5% worse
            behavior["overfitting_indicator"] = True

    # Early stopping detection
    if "metrics.epoch" in performance["final_metrics"]:
        final_epoch = performance["final_metrics"]["metrics.epoch"]
        # Consider early if less than 50 epochs (adjustable threshold)
        if final_epoch < 50:
            behavior["early_stopped"] = True

    # Convergence pattern (basic)
    if "val_loss" in performance["final_metrics"]:
        # This is simplified - could be enhanced with more epoch data
        behavior["convergence_pattern"] = "stable"  # Default assumption

    return behavior


def analyze_failure(run_data, error_logs):
    """Analyze failure patterns."""

    if run_data["status"] != "FAILED":
        return {
            "is_failure": False,
            "error_type": None,
            "error_details": None,
            "stopped_early": False,
        }

    failure_analysis = {
        "is_failure": True,
        "error_type": "unknown",
        "error_details": None,
        "stopped_early": False,
    }

    # Try to determine error type from run name or available data
    run_name = str(run_data.get("tags.mlflow.runName", "")).lower()

    # Check for error logs
    if error_logs["error_logs_found"]:
        failure_analysis["error_details"] = error_logs["error_details"]
        # Could parse error details to determine error type
        if "memory" in str(error_logs["error_details"]).lower():
            failure_analysis["error_type"] = "hardware"
        elif "cuda" in str(error_logs["error_details"]).lower():
            failure_analysis["error_type"] = "hardware"
        elif "file" in str(error_logs["error_details"]).lower():
            failure_analysis["error_type"] = "data"
        else:
            failure_analysis["error_type"] = "code"
    else:
        # Basic classification based on patterns
        if "desktop" in run_name:
            failure_analysis["error_type"] = "configuration"
        else:
            failure_analysis["error_type"] = "unknown"

    # Check if stopped early
    if "metrics.epoch" in run_data and pd.notna(run_data["metrics.epoch"]):
        if run_data["metrics.epoch"] < 20:  # Very early failure
            failure_analysis["stopped_early"] = True

    return failure_analysis


def extract_all_parameters(run_data):
    """Extract all parameters from run data."""

    parameters = {}

    # Extract all parameter columns
    param_cols = [col for col in run_data.index if col.startswith("params.")]

    for col in param_cols:
        if pd.notna(run_data[col]):
            param_name = col.replace("params.", "")
            parameters[param_name] = str(run_data[col])

    return parameters


def process_generation_data(df, generation_id):
    """Process all runs for a generation."""

    print(f"\nProcessing {len(df)} {generation_id} runs...")
    print(f"DEBUG: df shape = {df.shape}")

    processed_runs = []

    for idx, run in df.iterrows():
        try:
            # Basic run info
            run_info = {
                "run_id": run["run_id"],
                "run_name": run.get("tags.mlflow.runName", "Unknown"),
                "status": run["status"],
                "experiment_name": run.get("experiment_name", "Unknown"),
                "start_time": int(run["start_time"].timestamp())
                if pd.notna(run["start_time"])
                else None,
                "end_time": int(run["end_time"].timestamp())
                if pd.notna(run["end_time"])
                else None,
            }

            # Classification
            classification = classify_run(run_info["run_name"])
            run_info["configuration"] = classification

            # Timing analysis
            timing = calculate_timing_analysis(run)
            run_info["timing"] = timing

            # Performance analysis
            performance = analyze_performance(run)
            run_info["performance"] = performance

            # Training behavior analysis
            behavior = analyze_training_behavior(run, performance)
            run_info["analysis"] = behavior

            # Failure analysis
            error_logs = extract_error_logs(run["run_id"], run_info["run_name"])
            failure = analyze_failure(run, error_logs)
            run_info["failure_analysis"] = failure

            # All parameters
            parameters = extract_all_parameters(run)
            run_info["all_parameters"] = parameters

            processed_runs.append(run_info)

        except Exception as e:
            print(f"Error processing run {run.get('run_id', 'unknown')}: {e}")
            continue

    print(f"DEBUG: processed {len(processed_runs)} runs")
    return processed_runs


def calculate_summary_statistics(processed_runs):
    """Calculate summary statistics for generation."""

    if not processed_runs:
        return {}

    summary = {
        "total_runs": len(processed_runs),
        "finished_runs": len([r for r in processed_runs if r["status"] == "FINISHED"]),
        "failed_runs": len([r for r in processed_runs if r["status"] == "FAILED"]),
        "running_runs": len([r for r in processed_runs if r["status"] == "RUNNING"]),
        "success_rate": 0.0,
    }

    if summary["total_runs"] > 0:
        summary["success_rate"] = (
            summary["finished_runs"] / summary["total_runs"]
        ) * 100

    # Timing statistics
    finished_runs = [
        r
        for r in processed_runs
        if r["status"] == "FINISHED" and r["timing"]["duration_hours"] is not None
    ]
    if finished_runs:
        durations = [r["timing"]["duration_hours"] for r in finished_runs]
        summary["timing"] = {
            "average_hours": float(np.mean(durations)),
            "median_hours": float(np.median(durations)),
            "min_hours": float(np.min(durations)),
            "max_hours": float(np.max(durations)),
            "std_hours": float(np.std(durations)),
        }

    # Configuration statistics
    tasks = [r["configuration"]["task"] for r in processed_runs]
    servers = [r["configuration"]["server"] for r in processed_runs]
    config_types = [r["configuration"]["config_type"] for r in processed_runs]

    summary["configuration_distribution"] = {
        "tasks": dict(pd.Series(tasks).value_counts()),
        "servers": dict(pd.Series(servers).value_counts()),
        "config_types": dict(pd.Series(config_types).value_counts()),
    }

    return summary


def generate_outputs(processed_runs, generation_id):
    """Generate comprehensive outputs."""

    # Create comprehensive data structure
    output_data = {
        "extraction_time": datetime.now().isoformat(),
        "generation": generation_id,
        "runs": processed_runs,
        "summary_statistics": calculate_summary_statistics(processed_runs),
    }

    # Save JSON output
    json_filename = f"{generation_id}_all_data.json"
    with open(json_filename, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nData saved to {json_filename}")

    # Generate markdown summary
    generate_markdown_summary(output_data, generation_id)

    return output_data


def generate_markdown_summary(data, generation_id):
    """Generate markdown summary report."""

    summary = data["summary_statistics"]
    runs = data["runs"]

    md_content = f"""# {generation_id.upper()} Comprehensive Analysis Report

Generated: {data["extraction_time"]}

## Executive Summary

- **Total Runs**: {summary["total_runs"]}
- **Success Rate**: {summary["success_rate"]:.1f}%
- **Finished**: {summary["finished_runs"]}
- **Failed**: {summary["failed_runs"]}
- **Running**: {summary["running_runs"]}

## High-Level Analysis

"""

    # Add timing analysis if available
    if "timing" in summary:
        timing = summary["timing"]
        md_content += f"""### Timing Analysis

- **Average Duration**: {timing["average_hours"]:.2f} hours
- **Range**: {timing["min_hours"]:.2f} - {timing["max_hours"]:.2f} hours
- **Median**: {timing["median_hours"]:.2f} hours
- **Consistency**: {timing["std_hours"]:.2f} hours std

"""

    # Add configuration distribution
    config_dist = summary["configuration_distribution"]
    md_content += f"""### Configuration Distribution

**Tasks**:
"""
    for task, count in config_dist["tasks"].items():
        md_content += f"- {task}: {count}\n"

    md_content += "\n**Servers**:\n"
    for server, count in config_dist["servers"].items():
        md_content += f"- {server}: {count}\n"

    md_content += "\n**Configuration Types**:\n"
    for config_type, count in config_dist["config_types"].items():
        md_content += f"- {config_type}: {count}\n"

    # Add detailed analysis section
    md_content += f"""
## Detailed Analysis

### Performance Overview

"""

    # Performance metrics summary
    finished_runs = [r for r in runs if r["status"] == "FINISHED"]
    if finished_runs:
        # Collect performance metrics
        all_metrics = {}
        for run in finished_runs:
            for metric, value in run["performance"]["all_metrics"].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        # Summarize key metrics
        key_metrics = [
            "val_loss",
            "val_CleanIce_iou",
            "val_DebrisIce_iou",
            "val_Multiclass_iou",
        ]
        for metric in key_metrics:
            if metric in all_metrics and all_metrics[metric]:
                values = all_metrics[metric]
                md_content += f"**{metric}**:\n"
                md_content += f"- Mean: {np.mean(values):.4f}\n"
                md_content += f"- Std: {np.std(values):.4f}\n"
                md_content += (
                    f"- Range: {np.min(values):.4f} - {np.max(values):.4f}\n\n"
                )

    # Training behavior analysis
    overfitting_count = len([r for r in runs if r["analysis"]["overfitting_indicator"]])
    early_stopped_count = len([r for r in runs if r["analysis"]["early_stopped"]])

    md_content += f"""### Training Behavior

- **Overfitting Indicators**: {overfitting_count} runs
- **Early Stopped**: {early_stopped_count} runs

"""

    # Failure analysis
    failed_runs = [r for r in runs if r["status"] == "FAILED"]
    if failed_runs:
        md_content += "### Failure Analysis\n\n"

        error_types = {}
        for run in failed_runs:
            error_type = run["failure_analysis"]["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        md_content += "**Error Types**:\n"
        for error_type, count in error_types.items():
            md_content += f"- {error_type}: {count}\n"

        md_content += "\n**Failed Runs**:\n"
        for run in failed_runs:
            md_content += f"- {run['run_name']} ({run['configuration']['task']}, {run['configuration']['server']})\n"

        md_content += "\n"

    # Per-configuration analysis
    md_content += "### Per-Configuration Analysis\n\n"

    # Group by configuration type
    config_groups = {}
    for run in finished_runs:
        config_type = run["configuration"]["config_type"]
        if config_type not in config_groups:
            config_groups[config_type] = []
        config_groups[config_type].append(run)

    for config_type, config_runs in config_groups.items():
        if config_type == "unknown":
            continue

        md_content += f"**{config_type.title()}** ({len(config_runs)} runs):\n"

        if config_runs:
            # Average timing for this config
            durations = [
                r["timing"]["duration_hours"]
                for r in config_runs
                if r["timing"]["duration_hours"] is not None
            ]
            if durations:
                md_content += f"- Average duration: {np.mean(durations):.2f} hours\n"

            # Performance metrics
            if "val_CleanIce_iou" in config_runs[0]["performance"]["all_metrics"]:
                ious = [
                    r["performance"]["all_metrics"]["val_CleanIce_iou"]
                    for r in config_runs
                ]
                md_content += f"- Average CleanIce IoU: {np.mean(ious):.3f}\n"

        md_content += "\n"

    # Save markdown
    md_filename = f"{generation_id}_summary_report.md"
    with open(md_filename, "w") as f:
        f.write(md_content)

    print(f"Summary report saved to {md_filename}")


def main():
    """Main function - can be used for any generation."""

    import sys

    if len(sys.argv) != 2:
        print("Usage: python generation_analysis.py <generation_id>")
        print("Example: python generation_analysis.py gen6")
        sys.exit(1)

    generation_id = sys.argv[1]

    print(f"Universal Generation Analysis Script")
    print("=" * 80)
    print(f"Analyzing generation: {generation_id}")

    # Extract data
    raw_df = extract_generation_data(generation_id)

    if raw_df.empty:
        print(f"No {generation_id} data found!")
        return

    # Process data
    processed_runs = process_generation_data(raw_df, generation_id)

    # Generate outputs
    output_data = generate_outputs(processed_runs, generation_id)

    print(f"\n" + "=" * 80)
    print(f"{generation_id.upper()} ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Outputs generated:")
    print(f"  - {generation_id}_all_data.json (comprehensive data)")
    print(f"  - {generation_id}_summary_report.md (summary insights)")


if __name__ == "__main__":
    main()
