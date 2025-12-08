#!/usr/bin/env python3
"""
Compare two generations of MLflow runs.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
import sys


def load_data(filepath: Path) -> Dict[str, Any]:
    """Load generation data from a JSON file."""
    if not filepath.exists():
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    with open(filepath, "r") as f:
        return json.load(f)


def get_primary_iou(run: Dict[str, Any]) -> float:
    """Get the primary IoU metric for a given run based on its task."""
    task = run.get("configuration", {}).get("task")
    perf = run.get("performance", {}).get("best_metrics", {})

    if task == "clean_ice":
        return perf.get("best_test_CleanIce_iou", 0)
    elif task == "debris_ice":
        return perf.get("best_test_DebrisIce_iou", 0)
    elif task == "multiclass":
        return perf.get(
            "best_test_CleanIce_iou", perf.get("best_test_DebrisIce_iou", 0)
        )
    return 0


def compare_summary_stats(gen_a: Dict[str, Any], gen_b: Dict[str, Any]) -> str:
    """Compare summary statistics of two generations."""
    md = "## Executive Summary Comparison\n\n"
    md += "| Metric | Generation A | Generation B |\n"
    md += "|--------|--------------|--------------|\n"

    summary_a = gen_a["summary_statistics"]
    summary_b = gen_b["summary_statistics"]

    md += f"| Total Runs | {summary_a.get('total_runs', 0)} | {summary_b.get('total_runs', 0)} |\n"
    md += f"| Success Rate | {summary_a.get('success_rate', 0.0):.1f}% | {summary_b.get('success_rate', 0.0):.1f}% |\n"
    md += f"| Avg. Duration (h) | {summary_a.get('timing', {}).get('average_hours', 0):.2f} | {summary_b.get('timing', {}).get('average_hours', 0):.2f} |\n"
    md += "\n"
    return md


def compare_top_performers(gen_a: Dict[str, Any], gen_b: Dict[str, Any]) -> str:
    """Compare top performing models for each task."""
    md = "## Top Performer Comparison\n\n"
    tasks = sorted(
        list(
            set(
                r.get("configuration", {}).get("task", "unknown")
                for r in gen_a["runs"] + gen_b["runs"]
            )
        )
    )

    for task in tasks:
        if task == "unknown":
            continue
        md += f"### {task.replace('_', ' ').title()}\n\n"
        md += "| Generation | Run Name | Config | Best IoU |\n"
        md += "|------------|----------|--------|----------|\n"

        runs_a = [
            r
            for r in gen_a["runs"]
            if r["status"] == "FINISHED"
            and r.get("configuration", {}).get("task") == task
        ]
        runs_b = [
            r
            for r in gen_b["runs"]
            if r["status"] == "FINISHED"
            and r.get("configuration", {}).get("task") == task
        ]

        top_a = sorted(runs_a, key=get_primary_iou, reverse=True)[:3]
        top_b = sorted(runs_b, key=get_primary_iou, reverse=True)[:3]

        for run in top_a:
            iou = get_primary_iou(run)
            md += f"| Gen A | {run.get('run_name', 'N/A')} | {run.get('configuration', {}).get('config_type', 'N/A')} | {iou:.4f} |\n"
        for run in top_b:
            iou = get_primary_iou(run)
            md += f"| Gen B | {run.get('run_name', 'N/A')} | {run.get('configuration', {}).get('config_type', 'N/A')} | {iou:.4f} |\n"
        md += "\n"
    return md


def compare_config_performance(gen_a: Dict[str, Any], gen_b: Dict[str, Any]) -> str:
    """Compare performance by configuration type."""
    md = "## Configuration Performance Comparison\n\n"
    all_runs = gen_a["runs"] + gen_b["runs"]
    tasks = sorted(
        list(set(r.get("configuration", {}).get("task", "unknown") for r in all_runs))
    )

    for task in tasks:
        if task == "unknown":
            continue
        md += f"### {task.replace('_', ' ').title()}\n\n"
        md += "| Config Type | Gen | Avg. Best IoU | Run Count |\n"
        md += "|-------------|-----|---------------|-----------|\n"

        for gen_name, gen_data in [("A", gen_a), ("B", gen_b)]:
            task_runs = [
                r
                for r in gen_data["runs"]
                if r["status"] == "FINISHED"
                and r.get("configuration", {}).get("task") == task
            ]
            config_perf: Dict[str, List[float]] = {}
            for run in task_runs:
                config = run.get("configuration", {}).get("config_type", "unknown")
                if config not in config_perf:
                    config_perf[config] = []
                iou = get_primary_iou(run)
                if iou:
                    config_perf[config].append(iou)

            for config, ious in config_perf.items():
                avg_iou = np.mean(ious) if ious else 0
                md += f"| {config} | {gen_name} | {avg_iou:.4f} | {len(ious)} |\n"
        md += "\n"
    return md


def compare_failure_analysis(gen_a: Dict[str, Any], gen_b: Dict[str, Any]) -> str:
    """Compare failure analysis."""
    md = "## Failure Analysis Comparison\n\n"
    md += "| Generation | Total Failed | Hardware | Data | Code | Unknown |\n"
    md += "|------------|--------------|----------|------|------|---------|\n"

    for gen_name, gen_data in [("A", gen_a), ("B", gen_b)]:
        failed_runs = [r for r in gen_data["runs"] if r["status"] == "FAILED"]
        error_types = pd.DataFrame([r.get("failure_analysis", {}) for r in failed_runs])
        if not error_types.empty:
            counts = error_types["error_type"].value_counts().to_dict()
        else:
            counts = {}

        md += f"| {gen_name} | {len(failed_runs)} | {counts.get('hardware', 0)} | {counts.get('data', 0)} | {counts.get('code', 0)} | {counts.get('unknown', 0)} |\n"
    md += "\n"
    return md


def main() -> None:
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python compare_generations.py <gen_a_id> <gen_b_id>")
        sys.exit(1)

    gen_a_id = sys.argv[1]
    gen_b_id = sys.argv[2]

    gen_a_data = load_data(Path(f"archive/{gen_a_id}/{gen_a_id}_all_data.json"))
    gen_b_data = load_data(Path(f"archive/{gen_b_id}/{gen_b_id}_all_data.json"))

    md = f"# {gen_a_id.upper()} vs {gen_b_id.upper()} Comparison Report\n\n"
    md += compare_summary_stats(gen_a_data, gen_b_data)
    md += compare_top_performers(gen_a_data, gen_b_data)
    md += compare_config_performance(gen_a_data, gen_b_data)
    md += compare_failure_analysis(gen_a_data, gen_b_data)

    output_filename = Path(f"archive/{gen_a_id}_vs_{gen_b_id}_comparison_report.md")
    with open(output_filename, "w") as f:
        f.write(md)
    print(f"Comparison report saved to {output_filename}")


if __name__ == "__main__":
    main()
