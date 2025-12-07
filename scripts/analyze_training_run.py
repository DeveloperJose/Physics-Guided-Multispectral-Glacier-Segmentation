#!/usr/bin/env python3
"""
Analyze a training run from TensorBoard logs.
Restores functionality mentioned in AGENTS.md.
"""

import os
import sys
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import argparse


def get_latest_run_dir(base_dir="output"):
    runs = glob.glob(os.path.join(base_dir, "*"))
    runs.sort(key=os.path.getmtime, reverse=True)
    if not runs:
        return None
    return runs[0]


def ascii_plot(values, width=60, height=10, title=""):
    """Generate a simple ASCII plot of the values."""
    if not values:
        return

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val > min_val else 1.0

    # Normalize to height
    normalized = [int((v - min_val) / range_val * (height - 1)) for v in values]

    # Downsample to width if needed
    if len(normalized) > width:
        chunk_size = len(normalized) / width
        new_normalized = []
        for i in range(width):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            chunk = normalized[start:end]
            if chunk:
                new_normalized.append(int(sum(chunk) / len(chunk)))
            else:
                new_normalized.append(0)
        normalized = new_normalized

    print(f"\n{title}")
    print(f"Max: {max_val:.4f}")

    grid = [[" " for _ in range(len(normalized))] for _ in range(height)]
    for col, row in enumerate(normalized):
        grid[height - 1 - row][col] = "*"

    for row in grid:
        print("".join(row))

    print(f"Min: {min_val:.4f}")


def analyze_run(run_dir, plots=False):
    print(f"Analyzing run: {run_dir}")

    # Find event file
    potential_paths = [
        os.path.join(run_dir, "version_0", "events.out.tfevents.*"),
        os.path.join(run_dir, "logs", "version_0", "events.out.tfevents.*"),
        os.path.join(run_dir, "events.out.tfevents.*"),
    ]

    event_files = []
    for p in potential_paths:
        found = glob.glob(p)
        if found:
            event_files.extend(found)

    if not event_files:
        print("No event files found.")
        return

    event_file = event_files[0]
    print(f"Reading event file: {event_file}")

    ea = EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags()["scalars"]

    # Loss Analysis
    loss_tags = [t for t in tags if "loss" in t and "epoch" in t]
    print("\n" + "=" * 40)
    print("LOSS ANALYSIS")
    print("=" * 40)

    for tag in ["train_loss_epoch", "val_loss"]:
        if tag in tags:
            events = ea.Scalars(tag)
            values = [e.value for e in events]
            steps = [e.step for e in events]
            print(f"\nMetric: {tag}")
            print(f"  Start: {values[0]:.4f}")
            print(f"  End:   {values[-1]:.4f}")
            print(
                f"  Min:   {min(values):.4f} (Step {steps[values.index(min(values))]})"
            )

            ascii_plot(values, title=f"Trend: {tag}")

    # Sigma Analysis
    sigma_tags = [t for t in tags if "sigma" in t and "epoch" in t]
    if sigma_tags:
        print("\n" + "=" * 40)
        print("SIGMA ANALYSIS (Physics/Uncertainty Weights)")
        print("=" * 40)

        for tag in sigma_tags:
            events = ea.Scalars(tag)
            values = [e.value for e in events]
            print(f"\nMetric: {tag}")
            print(f"  Start: {values[0]:.4f}")
            print(f"  End:   {values[-1]:.4f}")
            print(f"  Delta: {values[-1] - values[0]:.4f}")

            # Interpretation
            if values[-1] < values[0]:
                print("  -> Decreasing: Model is becoming MORE confident in this task.")
            else:
                print(
                    "  -> Increasing: Model is becoming LESS confident (down-weighting this task)."
                )

            ascii_plot(values, title=f"Trend: {tag}")

    # Metrics Analysis (IoU)
    iou_tags = [t for t in tags if "iou" in t]
    if iou_tags:
        print("\n" + "=" * 40)
        print("PERFORMANCE METRICS")
        print("=" * 40)

        for tag in iou_tags:
            events = ea.Scalars(tag)
            values = [e.value for e in events]
            print(f"\nMetric: {tag}")
            print(f"  Best:  {max(values):.4f}")
            print(f"  Final: {values[-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a training run.")
    parser.add_argument("run_dir", nargs="?", help="Path to run directory")
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate PNG plots (not implemented in CLI version)",
    )

    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = get_latest_run_dir()

    if run_dir:
        analyze_run(run_dir, args.plots)
    else:
        print("No run directory found.")
