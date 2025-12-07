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
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

import mlflow
from mlflow.tracking import MlflowClient


def safe_float(value: Any) -> Optional[float]:
    """Safely convert value to float, handling NaN, infinity, and None."""
    if value is None or pd.isna(value) or np.isinf(value):
        return None
    return float(value)


def extract_generation_data(generation_id: str) -> pd.DataFrame:
    """Extract all runs for a specific generation from MLflow."""
    print(f"Extracting {generation_id} data from MLflow...")
    mlflow.set_tracking_uri("https://mlflow.developerjose.duckdns.org")

    experiments = mlflow.search_experiments()
    print(f"Found {len(experiments)} experiments")

    generation_runs = []
    for exp in experiments:
        try:
            runs_df: pd.DataFrame = mlflow.search_runs(
                experiment_ids=[exp.experiment_id], output_format="pandas"
            )
            run_name_col = "tags.mlflow.runName"
            if run_name_col not in runs_df.columns:
                continue

            runs_df[run_name_col] = runs_df[run_name_col].astype(str)
            mask = runs_df[run_name_col].str.contains(
                generation_id, case=False, na=False
            )
            gen_runs = runs_df[mask].copy()

            if not gen_runs.empty:
                gen_runs["experiment_name"] = exp.name
                generation_runs.append(gen_runs)
                print(f"  {exp.name}: {len(gen_runs)} {generation_id} runs")
        except Exception as e:
            print(f"  Error searching {exp.name}: {e}")
            continue

    if not generation_runs:
        print(f"\n*** NO {generation_id.upper()} RUNS FOUND ***")
        return pd.DataFrame()

    all_gen = pd.concat(generation_runs, ignore_index=True)
    print(f"\n*** TOTAL {generation_id.upper()} RUNS FOUND: {len(all_gen)} ***")
    return all_gen


def classify_run(run_name: Any) -> Dict[str, Any]:
    """Classify run based on name patterns."""
    name = str(run_name).lower()
    task = "unknown"
    if "ci" in name or "clean_ice" in name:
        task = "clean_ice"
    elif "dci" in name or "debris_ice" in name:
        task = "debris_ice"
    elif "multi" in name:
        task = "multiclass"

    server = "unknown"
    if "frodo" in name:
        server = "frodo"
    elif "bilbo" in name:
        server = "bilbo"
    elif "desktop" in name:
        server = "desktop"

    config_type = "unknown"
    if "physics" in name:
        config_type = "physics"
    elif "velocity" in name:
        config_type = "velocity"
    elif "synthesis" in name:
        config_type = "synthesis"
    elif "baseline" in name or "base" in name:
        config_type = "baseline"

    window_size = None
    if "w512" in name:
        window_size = 512
    elif "w256" in name:
        window_size = 256

    return {
        "task": task,
        "server": server,
        "config_type": config_type,
        "window_size": window_size,
    }


def extract_error_logs(run_id: str) -> Dict[str, Any]:
    """Attempt to extract error logs from MLflow artifacts."""
    try:
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        log_files = [
            f.path
            for f in artifacts
            if f.path.endswith(".log") or f.path.endswith(".txt")
        ]

        if not log_files:
            for item in artifacts:
                if item.is_dir and item.path in ["logs", "hydra"]:
                    sub_artifacts = client.list_artifacts(run_id, item.path)
                    log_files.extend(
                        f"{item.path}/{f.path}"
                        for f in sub_artifacts
                        if f.path.endswith(".log") or f.path.endswith(".txt")
                    )

        if not log_files:
            return {"error_logs_found": False, "error_details": "No log files found."}

        priority = ["stderr.txt", "hydra.log", "stdout.txt"]
        log_file = next((f for f in priority if f in log_files), log_files[0])

        local_path = client.download_artifacts(run_id, log_file)

        with open(local_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            error_snippet = "".join(lines[-50:])

        return {
            "error_logs_found": True,
            "log_file": log_file,
            "error_details": error_snippet,
        }
    except Exception as e:
        return {
            "error_logs_found": False,
            "error_details": f"Could not access error logs: {e}",
        }


def calculate_timing_analysis(run_data: pd.Series) -> Dict[str, Any]:
    """Calculate comprehensive timing analysis for a run."""
    start_time = run_data.get("start_time")
    end_time = run_data.get("end_time")

    if pd.isna(start_time) or pd.isna(end_time):
        return {"duration_hours": None, "per_epoch_timing": None}

    duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
    duration_hours = safe_float(duration.total_seconds() / 3600)

    per_epoch_timing = None
    final_epoch = run_data.get("metrics.epoch")

    if pd.notna(final_epoch) and final_epoch > 0 and duration_hours is not None:
        per_epoch_avg_hours = duration_hours / float(final_epoch)
        per_epoch_timing = {
            "total_epochs": int(final_epoch),
            "average_minutes_per_epoch": safe_float(per_epoch_avg_hours * 60),
        }

    return {"duration_hours": duration_hours, "per_epoch_timing": per_epoch_timing}


def analyze_performance(run_data: pd.Series) -> Dict[str, Any]:
    """Analyze performance metrics for a run."""
    performance: Dict[str, Any] = {
        "all_metrics": {},
        "best_metrics": {},
        "final_metrics": {},
    }
    for col, value in run_data.items():
        col_str = str(col)
        if col_str.startswith("metrics."):
            metric_name = col_str.replace("metrics.", "")
            s_value = safe_float(value)
            if s_value is None:
                continue
            performance["all_metrics"][metric_name] = s_value
            if "best_" in metric_name:
                performance["best_metrics"][metric_name] = s_value
            else:
                performance["final_metrics"][metric_name] = s_value
    return performance


def analyze_training_behavior(performance: Dict[str, Any]) -> Dict[str, bool]:
    """Analyze training behavior patterns."""
    behavior = {"overfitting_indicator": False, "early_stopped": False}
    best_loss = performance.get("best_metrics", {}).get("best_val_loss")
    final_loss = performance.get("final_metrics", {}).get("val_loss")

    if (
        best_loss is not None
        and final_loss is not None
        and final_loss > best_loss * 1.05
    ):
        behavior["overfitting_indicator"] = True

    final_epoch = performance.get("final_metrics", {}).get("epoch")
    if final_epoch is not None and final_epoch < 50:
        behavior["early_stopped"] = True

    return behavior


def analyze_failure(run_data: pd.Series) -> Dict[str, Any]:
    """Analyze failure patterns."""
    if run_data["status"] != "FAILED":
        return {"is_failure": False}

    error_logs = extract_error_logs(str(run_data["run_id"]))
    error_details = error_logs.get("error_details", "").lower()
    error_type = "unknown"
    if "memory" in error_details or "cuda" in error_details:
        error_type = "hardware"
    elif "file" in error_details:
        error_type = "data"
    elif error_logs["error_logs_found"]:
        error_type = "code"

    stopped_early = False
    epoch = run_data.get("metrics.epoch")
    if pd.notna(epoch) and epoch < 20:
        stopped_early = True

    return {
        "is_failure": True,
        "error_type": error_type,
        "error_details": error_logs.get("error_details"),
        "stopped_early": stopped_early,
    }


def extract_all_parameters(run_data: pd.Series) -> Dict[str, str]:
    """Extract all parameters from run data."""
    params = {}
    for col, value in run_data.items():
        col_str = str(col)
        if col_str.startswith("params."):
            param_name = col_str.replace("params.", "")
            if pd.notna(value):
                params[param_name] = str(value)
    return params


def process_generation_data(
    df: pd.DataFrame, generation_id: str
) -> List[Dict[str, Any]]:
    """Process all runs for a generation."""
    print(f"\nProcessing {len(df)} {generation_id} runs...")
    processed_runs = []
    for _, run in df.iterrows():
        try:
            run_name = run.get("tags.mlflow.runName", "Unknown")
            start_time = run.get("start_time")
            end_time = run.get("end_time")
            performance = analyze_performance(run)

            run_info = {
                "run_id": run["run_id"],
                "run_name": run_name,
                "status": run["status"],
                "experiment_name": run.get("experiment_name", "Unknown"),
                "start_time": pd.to_datetime(start_time).isoformat()
                if pd.notna(start_time)
                else None,
                "end_time": pd.to_datetime(end_time).isoformat()
                if pd.notna(end_time)
                else None,
                "configuration": classify_run(run_name),
                "timing": calculate_timing_analysis(run),
                "performance": performance,
                "analysis": analyze_training_behavior(performance),
                "failure_analysis": analyze_failure(run),
                "all_parameters": extract_all_parameters(run),
            }
            processed_runs.append(run_info)
        except Exception as e:
            print(f"Error processing run {run.get('run_id', 'unknown')}: {e}")
            continue
    print(f"Successfully processed {len(processed_runs)} runs")
    return processed_runs


def calculate_summary_statistics(
    processed_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate summary statistics for generation."""
    if not processed_runs:
        return {}

    df = pd.DataFrame(processed_runs)
    summary: Dict[str, Any] = {
        "total_runs": len(df),
        "finished_runs": int(df[df["status"] == "FINISHED"].shape[0]),
        "failed_runs": int(df[df["status"] == "FAILED"].shape[0]),
        "running_runs": int(df[df["status"] == "RUNNING"].shape[0]),
    }
    summary["success_rate"] = (
        (summary["finished_runs"] / summary["total_runs"]) * 100
        if summary["total_runs"] > 0
        else 0.0
    )

    finished_runs_timing = [
        r["timing"] for r in processed_runs if r["status"] == "FINISHED"
    ]
    durations = [
        t["duration_hours"]
        for t in finished_runs_timing
        if t.get("duration_hours") is not None
    ]
    if durations:
        summary["timing"] = {
            "average_hours": safe_float(np.mean(durations)),
            "median_hours": safe_float(np.median(durations)),
            "min_hours": safe_float(np.min(durations)),
            "max_hours": safe_float(np.max(durations)),
            "std_hours": safe_float(np.std(durations)),
        }

    config_list = [r["configuration"] for r in processed_runs]
    if config_list:
        config_df = pd.DataFrame(config_list)
        summary["configuration_distribution"] = {
            "tasks": config_df["task"].value_counts().to_dict(),
            "servers": config_df["server"].value_counts().to_dict(),
            "config_types": config_df["config_type"].value_counts().to_dict(),
        }
    return summary


def generate_outputs(processed_runs: List[Dict[str, Any]], generation_id: str) -> None:
    """Generate comprehensive outputs."""
    output_dir = Path(f"archive/{generation_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = calculate_summary_statistics(processed_runs)
    output_data = {
        "extraction_time": datetime.now().isoformat(),
        "generation": generation_id,
        "summary_statistics": summary,
        "runs": processed_runs,
    }

    json_filename = output_dir / f"{generation_id}_all_data.json"
    with open(json_filename, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nData saved to {json_filename}")

    generate_markdown_summary(output_data, output_dir)


def get_primary_iou(run: Dict[str, Any]) -> float:
    """Get the primary IoU metric for a given run based on its task."""
    task = run.get("configuration", {}).get("task")
    perf = run.get("performance", {}).get("best_metrics", {})

    if task == "clean_ice":
        return perf.get("best_test_CleanIce_iou", 0)
    elif task == "debris_ice":
        return perf.get("best_test_DebrisIce_iou", 0)
    elif task == "multiclass":
        # For multiclass, we might average or prioritize one. Let's prioritize CleanIce for now.
        return perf.get(
            "best_test_CleanIce_iou", perf.get("best_test_DebrisIce_iou", 0)
        )
    return 0


def generate_markdown_summary(data: Dict[str, Any], output_dir: Path) -> None:
    """Generate a comprehensive markdown summary report."""
    summary = data.get("summary_statistics", {})
    runs = data.get("runs", [])
    generation_id = data.get("generation", "Unknown")

    md = f"# {generation_id.upper()} Comprehensive Analysis Report\n\n"
    md += f"Generated: {data.get('extraction_time', 'N/A')}\n\n"

    # --- Executive Summary ---
    md += "## Executive Summary\n\n"
    md += f"- **Total Runs**: {summary.get('total_runs', 0)}\n"
    md += f"- **Success Rate**: {summary.get('success_rate', 0.0):.1f}%\n"
    md += f"- **Finished**: {summary.get('finished_runs', 0)}\n"
    md += f"- **Failed**: {summary.get('failed_runs', 0)}\n\n"

    if "timing" in summary and summary.get("timing"):
        timing = summary["timing"]
        md += "### Timing Analysis (Finished Runs)\n\n"
        md += f"- **Average Duration**: {timing.get('average_hours', 0):.2f} hours\n"
        md += f"- **Range**: {timing.get('min_hours', 0):.2f} - {timing.get('max_hours', 0):.2f} hours\n\n"

    # --- Top Performers ---
    finished_runs = [r for r in runs if r.get("status") == "FINISHED"]
    if finished_runs:
        md += "## Top Performing Runs\n\n"
        top_runs = sorted(finished_runs, key=get_primary_iou, reverse=True)[:5]
        md += "| Run Name | Task | Config | Best IoU |\n"
        md += "|----------|------|--------|----------|\n"
        for run in top_runs:
            iou = get_primary_iou(run)
            md += f"| {run.get('run_name', 'N/A')} | {run.get('configuration', {}).get('task', 'N/A')} | {run.get('configuration', {}).get('config_type', 'N/A')} | {iou:.4f} |\n"
        md += "\n"

    # --- Server Performance ---
    if "configuration_distribution" in summary:
        md += "## Server Performance Comparison\n\n"
        server_stats: Dict[str, Dict[str, Any]] = {}
        for run in runs:
            server = run.get("configuration", {}).get("server", "unknown")
            if server not in server_stats:
                server_stats[server] = {"total": 0, "finished": 0, "durations": []}
            server_stats[server]["total"] += 1
            if run["status"] == "FINISHED":
                server_stats[server]["finished"] += 1
                if run.get("timing", {}).get("duration_hours"):
                    server_stats[server]["durations"].append(
                        run["timing"]["duration_hours"]
                    )

        md += "| Server | Total Runs | Success Rate | Avg. Duration (h) |\n"
        md += "|--------|------------|--------------|-------------------|\n"
        for server, stats in server_stats.items():
            rate = (
                (stats["finished"] / stats["total"] * 100) if stats["total"] > 0 else 0
            )
            avg_dur = np.mean(stats["durations"]) if stats["durations"] else 0
            md += f"| {server} | {stats['total']} | {rate:.1f}% | {avg_dur:.2f} |\n"
        md += "\n"

    # --- Task-Specific Analysis ---
    tasks = sorted(
        list(set(r.get("configuration", {}).get("task", "unknown") for r in runs))
    )
    for task in tasks:
        if task == "unknown":
            continue
        md += f"## Task-Specific Analysis: {task.replace('_', ' ').title()}\n\n"
        task_runs = [
            r for r in finished_runs if r.get("configuration", {}).get("task") == task
        ]

        if not task_runs:
            md += "No finished runs for this task.\n\n"
            continue

        # Config Type Summary
        config_perf: Dict[str, Dict[str, List[float]]] = {}
        for run in task_runs:
            config = run.get("configuration", {}).get("config_type", "unknown")
            if config not in config_perf:
                config_perf[config] = {"ious": [], "losses": []}

            iou = get_primary_iou(run)
            loss = (
                run.get("performance", {}).get("best_metrics", {}).get("best_val_loss")
            )
            if iou:
                config_perf[config]["ious"].append(iou)
            if loss:
                config_perf[config]["losses"].append(loss)

        md += "### Performance by Configuration Type\n\n"
        md += "| Config Type | Avg. Best IoU | Avg. Best Val Loss |\n"
        md += "|-------------|---------------|--------------------|\n"
        for config, perf in config_perf.items():
            avg_iou = np.mean(perf["ious"]) if perf["ious"] else 0
            avg_loss = np.mean(perf["losses"]) if perf["losses"] else 0
            md += f"| {config} | {avg_iou:.4f} | {avg_loss:.4f} |\n"
        md += "\n"

        # Detailed Run Table
        md += "### Detailed Run Results\n\n"
        md += "| Run Name | Config | Duration (h) | Epochs | Best Val Loss | Best IoU | Overfitting |\n"
        md += "|----------|--------|--------------|--------|---------------|----------|-------------|\n"
        for run in task_runs:
            iou = get_primary_iou(run)
            md += f"| {run.get('run_name')} | {run.get('configuration', {}).get('config_type')} | {run.get('timing', {}).get('duration_hours', 0):.2f} | {run.get('timing', {}).get('per_epoch_timing', {}).get('total_epochs', 0)} | {run.get('performance', {}).get('best_metrics', {}).get('best_val_loss', 0):.4f} | {iou:.4f} | {run.get('analysis', {}).get('overfitting_indicator', False)} |\n"
        md += "\n"

    # --- Failure Analysis ---
    failed_runs = [r for r in runs if r.get("status") == "FAILED"]
    if failed_runs:
        md += "## Failure Analysis\n\n"
        failure_list = [r.get("failure_analysis", {}) for r in failed_runs]
        if failure_list:
            error_types = (
                pd.DataFrame(failure_list)["error_type"].value_counts().to_dict()
            )
            md += "**Failure Types**:\n"
            for etype, count in error_types.items():
                md += f"- {etype}: {count}\n"
        md += "\n**Failed Runs & Error Logs**:\n"
        for run in failed_runs:
            md += f"- **{run.get('run_name', 'Unknown')}**\n"
            error_details = run.get("failure_analysis", {}).get(
                "error_details", "Not available."
            )
            md += f"  ```\n  {error_details}\n  ```\n"

    md_filename = output_dir / f"{generation_id}_summary_report.md"
    with open(md_filename, "w") as f:
        f.write(md)
    print(f"Summary report saved to {md_filename}")


def main() -> None:
    """Main function."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python universal_gen_analysis.py <generation_id>")
        sys.exit(1)
    generation_id = sys.argv[1]

    print(f"Analyzing generation: {generation_id}")
    raw_df = extract_generation_data(generation_id)

    if not raw_df.empty:
        processed_runs = process_generation_data(raw_df, generation_id)
        if processed_runs:
            generate_outputs(processed_runs, generation_id)
            print(f"\n{generation_id.upper()} ANALYSIS COMPLETE")
        else:
            print("\nNo runs were successfully processed.")
    else:
        print(f"No {generation_id} data found!")


if __name__ == "__main__":
    main()
