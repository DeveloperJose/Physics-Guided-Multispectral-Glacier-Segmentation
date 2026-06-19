# AGENTS.md

## Development Commands

**IMPORTANT**: Use `uv` for everything (installs, running scripts). No raw `pip` or `python`.

## Agent Execution Policy

- Agents do **not** have permission to run training scripts.
- Agents may prepare configs, inspect past outputs, run analysis, run prediction, and run tests unless the user says otherwise.
- All training runs, including `scripts/train.py` and `run_sequential_training.sh`, are user-executed only.
- If a task would require new training, the agent must stop at config preparation and tell the user exactly what to run.

### Environment Setup
```bash
uv pip install -e .
uv pip install -e ".[dev]"
```

### Running Python
```bash
# Always through uv
uv run python script.py
uv run python -m module.name
```

### Linting & Formatting
```bash
ruff check .
ruff format .
```

### Testing
```bash
# Unit tests (velocity loss, slice functions, class weights)
uv run python scripts/test.py --unit

# Integration tests (end-to-end training pipeline)
uv run python scripts/test.py --server desktop --subset-size 5 --epochs 2
```

### Training
User reference only. Agents must not execute these commands.
```bash
uv run python scripts/train.py --config configs/desktop/debris_ice/sota_dci_06_bs12_seed42_gpu0.yaml --server desktop --gpu 0

# Sequential runs
uv run bash run_sequential_training.sh
```

### MLflow Upload
```bash
uv run python scripts/upload_to_mlflow.py output/run_name
uv run python scripts/upload_to_mlflow.py output/run_name --regenerate --high-res --val-viz-n 8 --test-eval-n 8
uv run python scripts/upload_to_mlflow.py --batch --experiment-type baseline_ci --output-dir output
```

### Prediction
```bash
# Single model
uv run python scripts/predict.py --ci-run-name run_name --server desktop --gpu 0 --split test

# Paired CI/DCI evaluation
uv run python scripts/predict.py --ci-run-name ci_run_name --deb-run-name dci_run_name --server desktop --gpu 0 --split test
```

## Configuration System
- 4-level merge: `configs/train.yaml` (global defaults) → `configs/servers.yaml` (server) → `configs/tasks/{task}.yaml` (task) → experiment file.
- Experiment files live under `configs/{server}/{task}/`. Only override what differs from upstream levels.
- Keep experiment configs minimal and descriptive.
- Always use MLflow experiment `sota_replication` for every training batch so all
  runs remain comparable in one experiment. Do not create batch-specific experiments.
- Keep MLflow metrics enabled by default, but leave `training_opts.mlflow_artifacts_enabled: false` unless the artifact store is writable from the training machine.

## Measured Training Performance

Evidence from desktop batches 23-32 on RTX 3060 Ti:

- Canonical loader: packed recipe-specific float32 NCHW `X.npy`/`y.npy` memmaps,
  batch size 12, 2 workers, pinned memory, persistent workers, prefetch factor 2.
- Do not restore prebatched arrays, RAM caching, forced contiguous copies, float16
  storage selection, or `torch.compile`. They were slower, unstable, or unproven.
- Batch size 24 was slower than 12 in isolated tests despite unused VRAM.
- More than 2 workers and prefetch factor 4 did not improve wall time.
- Keep fused AdamW; measured wall-time gain was small, but no regression occurred.
- Keep `channels_last` support. One full run improved about 15.79 to 13.74
  seconds/epoch (13%); confirm before making it the publication default.
- Use Lightning SimpleProfiler for whole-run attribution and PyTorchProfiler for
  CUDA kernels/transfers. AdvancedProfiler mostly duplicated their evidence.
- Do not add custom timing callbacks. Removed timer mixed validation/checkpoint time
  into `data_wait_ms` and produced misleading attribution.
- Full batch-32 profiling found train DataLoader wait near 1% and validation loading
  near 0.7%. Data loading is no longer primary bottleneck.
- Main measured bottleneck was `customloss`: about 89.5% of profiled
  `training_step` wall time and 82% of validation-step wall time. Avoid CUDA tensor
  values in Python conditionals because they force host synchronization.
- Rich progress reporting was charged about 100 ms/batch in PyTorch traces, but part
  may be deferred CUDA synchronization. Keep progress disabled by default and verify
  gains using profiler plus wall time.
- Reference, SimpleProfiler, AdvancedProfiler, and PyTorchProfiler batch-32 runs had
  identical test IoU, showing profiler instrumentation did not change results.

## Public Documentation
- Keep README and public docs focused on this repository's software, workflows, results, and assets.
- Keep implementation notes, local setup history, and maintenance context out of public docs.
- If public docs reference figures or supporting files, keep those files inside this repository under `docs/` and link only to repo-local paths.

## Project Structure
```
configs/
├── desktop/
├── tasks/
├── train.yaml
└── servers.yaml
glacier_mapping/
├── data/        # Dataset, slicing, physics channels
├── lightning/   # Module, datamodule, callbacks
├── model/       # UNet, losses, evaluation
└── utils/       # Config, logging, MLflow, GPU, viz
scripts/
├── train.py
├── predict.py
├── preprocess.py
├── upload_to_mlflow.py
├── test.py
├── app_gradio.py
└── create_velocity_from_itslive_mosaic.py
output/
```

## Code Style Guidelines
- Imports: stdlib → third-party → local; import `torch` before other ML libs; use absolute imports (e.g., `from glacier_mapping.lightning.glacier_module import GlacierSegmentationModule`).
- Formatting: follow `ruff` config; 88-char lines; double quotes.
- Type hints: required; use `torch.Tensor` for tensors, `Path` for paths.
- Naming: PascalCase classes, snake_case functions/vars, UPPER_SNAKE_CASE constants.
- Error handling: explicit for file I/O; validate configs early; wrap inference in `torch.no_grad()`.
- Critical patterns: run from repo root for config loading; treat label 255 as ignore mask; load models with `GlacierSegmentationModule.load_from_checkpoint()`.
