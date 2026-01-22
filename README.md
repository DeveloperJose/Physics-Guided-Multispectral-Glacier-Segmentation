# Glacier Mapping by Segmentation

U-Net models for glacier segmentation from satellite imagery.

## Setup

```bash
uv pip install -e .
uv pip install -e ".[dev]"
```

## Commands

```bash
# Train
uv run python scripts/train.py --config configs/frodo/clean_ice/base.yaml --server frodo --gpu 0

# Lint
ruff check .
ruff format .

# Type check
pyright glacier_mapping/

# Upload to MLflow
uv run python scripts/upload_to_mlflow.py output/run_name
```

## Structure

```
glacier_mapping/
├── lightning/     # PyTorch Lightning modules
├── model/         # U-Net, losses, metrics
├── data/          # Data loading, preprocessing
└── utils/         # Visualization, utilities

google_earth_scripts/  # Google Earth Engine scripts

scripts/
├── train.py
├── predict.py
├── preprocess.py
└── upload_to_mlflow.py

configs/
├── train.yaml           # Global defaults
├── servers.yaml         # Server configs
├── tasks/               # Task configs
    └── {server}/{task}/     # Experiment configs
```
