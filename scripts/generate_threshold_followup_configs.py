#!/usr/bin/env python3
"""Generate DCI threshold follow-up configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


SEEDS = [42, 1337, 2026, 3407]
THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.16, 5.0, 10.0]

BASE_TRAINING = {
    "epochs": 200,
    "early_stopping": 15,
    "early_stopping_min_delta": 0.001,
    "mlflow_experiment_name": "reproducibility",
    "experiment_prefix": "reproducibility",
    "val_viz_n": 4,
    "val_viz_every_n_epochs": 0,
    "run_test_eval": True,
    "test_eval_n": 4,
    "deterministic": True,
}

FULL_LOADER = {
    "landsat_channels": True,
    "dem_channels": True,
    "spectral_indices_channels": True,
    "hsv_channels": True,
    "physics_channels": True,
    "velocity_channels": True,
    "output_classes": [2],
    "robust_scaling": False,
}

BASELINE_VARIANTS = [
    (
        "baseline_landsat_dem",
        {
            "landsat_channels": True,
            "dem_channels": True,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": False,
            "velocity_channels": False,
            "output_classes": [2],
            "robust_scaling": False,
        },
        {"use_velocity_loss": False},
    ),
    (
        "full_no_velocity_loss",
        FULL_LOADER,
        {"use_velocity_loss": False},
    ),
    (
        "physics_velocity_no_spectral_threshold_2p0",
        {
            "landsat_channels": True,
            "dem_channels": True,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": True,
            "velocity_channels": True,
            "output_classes": [2],
            "robust_scaling": False,
        },
        {"use_velocity_loss": True, "velocity_high_speed_threshold": 2.0},
    ),
]


def threshold_label(value: float) -> str:
    return str(value).replace(".", "p")


def build_config(name: str, seed: int, loader_opts: dict, loss_opts: dict) -> dict:
    training_opts = deepcopy(BASE_TRAINING)
    training_opts["seed"] = seed
    training_opts["run_name"] = f"reproducibility_dci_followup_{name}_seed{seed}"

    content = {
        "training_opts": training_opts,
        "loader_opts": deepcopy(loader_opts),
        "scheduler_opts": {"args": {"max_lr": 0.0005}},
        "loss_opts": {"use_velocity_loss": False},
    }
    content["loss_opts"].update(loss_opts)
    return content


def write_config(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(content, f, sort_keys=False)


def main() -> None:
    base_path = Path("configs/desktop/debris_ice")
    generated = []

    for threshold in THRESHOLDS:
        name = f"full_threshold_{threshold_label(threshold)}"
        for seed in SEEDS:
            content = build_config(
                name,
                seed,
                FULL_LOADER,
                {
                    "use_velocity_loss": True,
                    "velocity_high_speed_threshold": threshold,
                },
            )
            path = base_path / f"reproducibility_dci_followup_{name}_seed{seed}_gpu0.yaml"
            write_config(path, content)
            generated.append(path)

    # Add a few non-threshold controls for the new seed 3407 only. The earlier
    # sweep already has seeds 42, 1337, and 2026 for these variants.
    for name, loader_opts, loss_opts in BASELINE_VARIANTS:
        seed = 3407
        content = build_config(name, seed, loader_opts, loss_opts)
        path = base_path / f"reproducibility_dci_followup_{name}_seed{seed}_gpu0.yaml"
        write_config(path, content)
        generated.append(path)

    readme = Path("configs/desktop/PUBLICATION_EXPERIMENTS.md")
    readme.write_text(
        "# Threshold Follow-Up Experiments\n\n"
        "All runs are DCI-only and log to the single MLflow experiment `reproducibility`.\n\n"
        f"Total configs: {len(generated)}\n\n"
        "## Rationale\n\n"
        "The 2026-06-08 sweep showed the best DCI result from the full model with "
        "velocity loss threshold 2.0, but that threshold was only tested for seed 42. "
        "This batch tests whether that threshold advantage holds across seeds and "
        "whether the optimum is closer to 1.5, 2.0, or 2.5.\n\n"
        "## Design\n\n"
        "- Full DCI model with velocity loss thresholds 1.0, 1.5, 2.0, 2.5, 3.16, 5.0, and 10.0.\n"
        "- Four seeds per threshold: 42, 1337, 2026, and 3407.\n"
        "- Three seed-3407 controls: Landsat+DEM baseline, full model without velocity loss, and no-spectral physics+velocity at threshold 2.0.\n"
        "- CI and multiclass are deferred; the clean-ice result was already stable enough and not the paper's main claim.\n\n"
        "## Runtime Estimate\n\n"
        "- 31 configs total.\n"
        "- Expected active training time: about 14-16 hours.\n"
        "- With 30 second pauses: about 14.5-16.5 hours wall-clock.\n",
    )

    print(f"Generated {len(generated)} configs")


if __name__ == "__main__":
    main()
