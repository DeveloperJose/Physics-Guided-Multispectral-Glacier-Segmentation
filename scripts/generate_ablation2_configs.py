import yaml
from pathlib import Path

# Define the tasks
TASKS = {"clean_ice": "ci", "debris_ice": "dci", "multiclass": "mc"}

# Define the experiment configurations
# Each config defines the overrides for loader_opts and loss_opts
EXPERIMENTS = {
    "landsat_only": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": False,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": False,
            "velocity_channels": False,
        }
    },
    "landsat_spectral_indices": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": False,
            "spectral_indices_channels": True,
            "hsv_channels": False,
            "physics_channels": False,
            "velocity_channels": False,
        }
    },
    "full_physics_velocity_channels_loss": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": True,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": True,
            "velocity_channels": True,
        },
        "loss_opts": {"use_velocity_loss": True},
    },
    "landsat_dem_core": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": True,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": False,
            "velocity_channels": False,
        }
    },
    "landsat_flow_accumulation": {
        "loader_opts": {
            "landsat_channels": True,
            # Keep only the derived flow channel. DEM was used during preprocessing.
            "dem_channels": False,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": ["flow_accumulation"],
            "velocity_channels": False,
        }
    },
    "landsat_velocity_channels": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": False,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": False,
            "velocity_channels": True,
        }
    },
    "landsat_tpi": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": False,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": ["tpi"],
            "velocity_channels": False,
        }
    },
    "landsat_velocity_channels_loss": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": False,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": False,
            "velocity_channels": True,
        },
        "loss_opts": {"use_velocity_loss": True},
    },
    "landsat_roughness": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": False,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": ["roughness"],
            "velocity_channels": False,
        }
    },
    "landsat_full_physics": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": True,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": True,
            "velocity_channels": False,
        }
    },
    "landsat_curvature": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": False,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": ["plan_curvature"],
            "velocity_channels": False,
        }
    },
    "landsat_velocity_magnitude": {
        "loader_opts": {
            "landsat_channels": True,
            "dem_channels": False,
            "spectral_indices_channels": False,
            "hsv_channels": False,
            "physics_channels": False,
            "velocity_channels": ["velocity"],  # "velocity" is magnitude
        }
    },
}

# Define the server assignments
ASSIGNMENTS = {
    "bilbo": {
        0: ["landsat_only", "landsat_spectral_indices"],
        1: ["full_physics_velocity_channels_loss", "landsat_dem_core"],
    },
    "frodo": {
        0: ["landsat_only", "landsat_flow_accumulation"],
        1: ["landsat_velocity_channels", "landsat_tpi"],
        2: ["landsat_velocity_channels_loss", "landsat_roughness"],
        3: ["landsat_full_physics", "landsat_curvature"],
    },
    "desktop": {
        0: [
            "full_physics_velocity_channels_loss",
            "landsat_velocity_magnitude",
            "landsat_only",
        ]
    },
}


def generate_configs():
    base_path = Path("configs")

    for server, gpus in ASSIGNMENTS.items():
        for gpu_id, exp_list in gpus.items():
            for exp_name in exp_list:
                if exp_name not in EXPERIMENTS:
                    print(
                        f"Warning: Experiment {exp_name} not defined in EXPERIMENTS dict."
                    )
                    continue

                exp_config = EXPERIMENTS[exp_name]

                for task, task_abbr in TASKS.items():
                    # Construct filename
                    # Format: ablation2_{task_abbr}_{exp_name}_gpu{gpu_id}.yaml
                    filename = f"ablation2_{task_abbr}_{exp_name}_gpu{gpu_id}.yaml"

                    # Ensure directory exists
                    dir_path = base_path / server / task
                    dir_path.mkdir(parents=True, exist_ok=True)

                    file_path = dir_path / filename

                    # Construct content
                    run_name = f"ablation2_{task_abbr}_{exp_name}_gpu{gpu_id}"

                    content = {
                        "training_opts": {
                            "run_name": run_name,
                            "early_stopping": 30,
                            "early_stopping_min_delta": 0.001,
                        },
                        "loader_opts": exp_config["loader_opts"].copy(),
                        "scheduler_opts": {"args": {"max_lr": 0.0005}},
                    }

                    if "loss_opts" in exp_config:
                        content["loss_opts"] = exp_config["loss_opts"].copy()

                    # Write to file
                    with open(file_path, "w") as f:
                        yaml.dump(content, f, default_flow_style=False, sort_keys=False)

    print("Configuration generation complete.")


if __name__ == "__main__":
    generate_configs()
