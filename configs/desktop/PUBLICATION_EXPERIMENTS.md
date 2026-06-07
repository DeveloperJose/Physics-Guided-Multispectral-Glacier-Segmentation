# Publication Reproducibility Batch 1

This desktop batch is intentionally small. `run_sequential_training.sh` discovers
all YAML files under `configs/desktop`, so this directory currently contains only
the first set we want to run.

## MLflow

- Tracking URI: `https://mlflow.josegperez.com/`
- Experiment prefix: `reproducibility`
- Expected experiments: `reproducibility_debris_ice`,
  `reproducibility_clean_ice`

## Batch 1 Runs

### Debris Ice Core Ablation

1. `reproducibility_dci_01_landsat_only_seed42_gpu0.yaml`
2. `reproducibility_dci_02_landsat_dem_seed42_gpu0.yaml`
3. `reproducibility_dci_03_static_physics_seed42_gpu0.yaml`
4. `reproducibility_dci_04_velocity_channels_seed42_gpu0.yaml`
5. `reproducibility_dci_05_velocity_loss_seed42_gpu0.yaml`
6. `reproducibility_dci_06_physics_velocity_seed42_gpu0.yaml`

These runs isolate the key paper question for debris-covered ice: whether static
terrain physics, velocity inputs, and velocity-informed loss add value beyond
Landsat and DEM baselines.

### Clean Ice Check

1. `reproducibility_ci_01_landsat_only_seed42_gpu0.yaml`
2. `reproducibility_ci_03_static_physics_seed42_gpu0.yaml`
3. `reproducibility_ci_06_physics_velocity_seed42_gpu0.yaml`

These runs are a compact generalization check. If the debris-ice signal is weak
or inconsistent, this helps determine whether the physics/velocity effect is
task-specific or broadly unstable.

## Deferred Runs

- Additional seeds: `1337`, `2026`
- Multiclass experiments
- Velocity threshold sensitivity
- Robust physics scaling sensitivity

Run these only after Batch 1 shows which comparisons are worth replicating.
