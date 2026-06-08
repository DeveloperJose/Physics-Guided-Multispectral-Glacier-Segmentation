#!/usr/bin/env python3
"""Generate desktop reproducibility experiment configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


SEEDS = [42, 1337, 2026]

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

BASE_LOADER = {
    "landsat_channels": True,
    "dem_channels": False,
    "spectral_indices_channels": False,
    "hsv_channels": False,
    "physics_channels": False,
    "velocity_channels": False,
    "robust_scaling": False,
}


CORE_VARIANTS = [
    (
        "01_landsat_only",
        {},
        {},
    ),
    (
        "02_landsat_dem",
        {"dem_channels": True},
        {},
    ),
    (
        "03_landsat_spectral_hsv",
        {"spectral_indices_channels": True, "hsv_channels": True},
        {},
    ),
    (
        "04_landsat_dem_spectral_hsv",
        {"dem_channels": True, "spectral_indices_channels": True, "hsv_channels": True},
        {},
    ),
    (
        "05_static_physics",
        {"dem_channels": True, "physics_channels": True},
        {},
    ),
    (
        "06_velocity_channels",
        {"velocity_channels": True},
        {},
    ),
    (
        "07_velocity_loss",
        {"velocity_channels": True},
        {"use_velocity_loss": True},
    ),
    (
        "08_physics_velocity_no_spectral",
        {"dem_channels": True, "physics_channels": True, "velocity_channels": True},
        {"use_velocity_loss": True},
    ),
    (
        "09_dissertation_full_no_loss",
        {
            "dem_channels": True,
            "spectral_indices_channels": True,
            "hsv_channels": True,
            "physics_channels": True,
            "velocity_channels": True,
        },
        {},
    ),
    (
        "10_dissertation_full",
        {
            "dem_channels": True,
            "spectral_indices_channels": True,
            "hsv_channels": True,
            "physics_channels": True,
            "velocity_channels": True,
        },
        {"use_velocity_loss": True},
    ),
]

CI_VARIANTS = [
    "01_landsat_only",
    "02_landsat_dem",
    "03_landsat_spectral_hsv",
    "05_static_physics",
    "09_dissertation_full_no_loss",
    "10_dissertation_full",
]

DCI_EXTRA_VARIANTS = [
    (
        "11_flow_accumulation_only",
        {"physics_channels": ["flow_accumulation"]},
        {},
    ),
    (
        "12_tpi_only",
        {"physics_channels": ["tpi"]},
        {},
    ),
    (
        "13_roughness_only",
        {"physics_channels": ["roughness"]},
        {},
    ),
    (
        "14_curvature_only",
        {"physics_channels": ["plan_curvature"]},
        {},
    ),
    (
        "15_full_threshold_2p0",
        {
            "dem_channels": True,
            "spectral_indices_channels": True,
            "hsv_channels": True,
            "physics_channels": True,
            "velocity_channels": True,
        },
        {"use_velocity_loss": True, "velocity_high_speed_threshold": 2.0},
    ),
    (
        "16_full_threshold_5p0",
        {
            "dem_channels": True,
            "spectral_indices_channels": True,
            "hsv_channels": True,
            "physics_channels": True,
            "velocity_channels": True,
        },
        {"use_velocity_loss": True, "velocity_high_speed_threshold": 5.0},
    ),
    (
        "17_full_threshold_10p0",
        {
            "dem_channels": True,
            "spectral_indices_channels": True,
            "hsv_channels": True,
            "physics_channels": True,
            "velocity_channels": True,
        },
        {"use_velocity_loss": True, "velocity_high_speed_threshold": 10.0},
    ),
    (
        "18_static_physics_robust",
        {"dem_channels": True, "physics_channels": True, "robust_scaling": True},
        {},
    ),
    (
        "19_dissertation_full_robust",
        {
            "dem_channels": True,
            "spectral_indices_channels": True,
            "hsv_channels": True,
            "physics_channels": True,
            "velocity_channels": True,
            "robust_scaling": True,
        },
        {"use_velocity_loss": True},
    ),
]

TASKS = {
    "debris_ice": {"abbr": "dci", "output_classes": [2], "variants": CORE_VARIANTS},
    "clean_ice": {
        "abbr": "ci",
        "output_classes": [1],
        "variants": [variant for variant in CORE_VARIANTS if variant[0] in CI_VARIANTS],
    },
}


def build_config(task_abbr: str, output_classes: list[int], name: str, seed: int, loader_overrides: dict, loss_overrides: dict) -> dict:
    loader_opts = deepcopy(BASE_LOADER)
    loader_opts.update(loader_overrides)
    loader_opts["output_classes"] = output_classes

    training_opts = deepcopy(BASE_TRAINING)
    training_opts["seed"] = seed
    training_opts["run_name"] = f"reproducibility_{task_abbr}_{name}_seed{seed}"

    content = {
        "training_opts": training_opts,
        "loader_opts": loader_opts,
        "scheduler_opts": {"args": {"max_lr": 0.0005}},
        "loss_opts": {"use_velocity_loss": False},
    }
    content["loss_opts"].update(loss_overrides)
    return content


def write_config(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(content, f, sort_keys=False)


def main() -> None:
    base_path = Path("configs/desktop")
    generated = []

    for task_name, task in TASKS.items():
        for name, loader_overrides, loss_overrides in task["variants"]:
            for seed in SEEDS:
                content = build_config(
                    task["abbr"],
                    task["output_classes"],
                    name,
                    seed,
                    loader_overrides,
                    loss_overrides,
                )
                filename = f"reproducibility_{task['abbr']}_{name}_seed{seed}_gpu0.yaml"
                path = base_path / task_name / filename
                write_config(path, content)
                generated.append(path)

    for name, loader_overrides, loss_overrides in DCI_EXTRA_VARIANTS:
        content = build_config("dci", [2], name, 42, loader_overrides, loss_overrides)
        filename = f"reproducibility_dci_{name}_seed42_gpu0.yaml"
        path = base_path / "debris_ice" / filename
        write_config(path, content)
        generated.append(path)

    readme = base_path / "PUBLICATION_EXPERIMENTS.md"
    readme.write_text(
        "# Comprehensive Reproducibility Experiments\n\n"
        "All runs log to the single MLflow experiment `reproducibility`.\n\n"
        f"Total configs: {len(generated)}\n\n"
        "## Design\n\n"
        "- DCI core ablations: 10 variants x 3 seeds = 30 runs.\n"
        "- DCI sensitivity and component checks: 9 seed-42 runs.\n"
        "- CI comparison ablations: 6 variants x 3 seeds = 18 runs.\n"
        "- Multiclass is deferred because this paper is centered on binary CI/DCI behavior.\n"
        "- Early stopping patience is 15 epochs to favor breadth; rerun the best models later with patience 30 if needed.\n\n"
        "## Runtime Estimate\n\n"
        "- Expected active training time with patience 15: about 24-26 hours\n"
        "- With 30 second pauses: about 24.5-26.5 hours wall-clock\n"
        "- Conservative upper bound: about 30 hours if full-channel runs train longer than the first batch.\n",
    )

    print(f"Generated {len(generated)} configs")


if __name__ == "__main__":
    main()
