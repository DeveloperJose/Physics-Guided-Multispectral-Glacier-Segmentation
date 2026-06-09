# Physics-Guided Strategies for Enhancing Neural Networks Trained with Limited Data

This repository contains the glacier segmentation track of the doctoral dissertation by Jose G. Perez (UTEP, 2025), under the supervision of Dr. Olac Fuentes. We apply three physics-guided strategies to improve debris-covered ice mapping from Landsat satellite imagery without changing the underlying U-Net architecture.

## Strategies

1. **Physics-Informed Data Augmentation (Ch. 3)** -- Encode DEM-derived terrain features (flow accumulation, TPI, roughness, plan curvature) and spectral indices (NDVI, NDWI, NDSI) as additional input channels to guide the model toward physically plausible ice boundaries.

2. **Physics-Informed Loss Functions (Ch. 4)** -- Penalize predictions that violate glacier flow physics using a sigmoid-based velocity loss: if the model predicts background on a pixel moving faster than surrounding static terrain, it is penalized.

3. **Dynamic Velocity Integration (Ch. 4)** -- A production-grade pipeline that fuses ITS\_LIVE velocity datacubes with Landsat imagery via geometric discovery, 7-year temporal aggregation, cross-UTM-zone reprojection, and bilinear resampling from 120m to 30m.

## Key Results on the HKH Glacier Dataset

| Model | DCI IoU | DCI Prec. | DCI Rec. | CI IoU | CI Prec. | CI Rec. |
|---|---|---|---|---|---|---|
| Standard U-Net | 28.50 | -- | -- | 65.60 | -- | -- |
| Boundary-Aware (SOTA, Aryal et al. 2023) | 35.94 | 51.97 | 53.81 | **68.17** | 81.59 | 80.55 |
| Ours (Flow Only, Ch. 3) | 38.50 | 58.90 | 52.60 | 63.50 | 78.90 | 76.40 |
| Ours (Full Static Physics, Ch. 3) | **45.92** | **71.89** | **55.96** | **71.22** | **85.39** | **81.10** |
| Ours (Velocity Channels Only, Ch. 4) | 32.40 | 70.25 | 37.56 | 70.78 | 82.27 | 83.52 |
| Ours (Velocity Channels + Loss, Ch. 4) | 41.91 | 66.40 | 53.20 | 61.83 | 64.90 | **92.90** |
| Ours (Complete Physics-Informed, Ch. 4) | **46.07** | **71.95** | **56.16** | 65.85 | 72.36 | 87.98 |

- **Full Static Physics** (DEM flow accumulation + TPI + roughness + plan curvature) improves DCI IoU by **+9.98pp (27.8% relative)** over SOTA baseline.
- **Complete Physics-Informed** (static augmentation + velocity channels + velocity loss) achieves a new state-of-the-art DCI IoU of **46.07%**, a **+10.13pp (28.2% relative)** improvement.

## Visual Results

![Glacier segmentation example](dissertation/figures/glacier_mapping.png)

Good predictions on Clean Ice (top) and Debris-Covered Ice (bottom):

![Clean ice good prediction](dissertation/figures/ci_good.png)
![Debris-covered ice good prediction](dissertation/figures/dci_good.png)

Earth observation data augmentation channels derived from DEM:

![DEM physics channels](dissertation/figures/data_augmentation_dem.png)
![Physics augmentation result](dissertation/figures/data_augmentation_result.png)

## Quick Start

```bash
uv pip install -e .
uv pip install -e ".[dev]"

# Train
uv run python scripts/train.py --config configs/desktop/debris_ice/reproducibility2_dci_15_full_threshold_2p0_seed42_gpu0.yaml --server desktop --gpu 0

# Evaluate on test set
uv run python scripts/predict.py --ci-run-name <run> --deb-run-name <run> --server desktop --gpu 0 --split test

# Upload to MLflow
uv run python scripts/upload_to_mlflow.py output/<run_name>

# Tests
uv run python scripts/test.py --unit
uv run python scripts/test.py --server desktop --subset-size 5 --epochs 2

# Lint
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
configs/                 4-level config merge (global → server → task → experiment)
glacier_mapping/
  data/                  Dataset, slicing, physics channel computation
  lightning/             LightningModule, DataModule, callbacks
  model/                 U-Net, losses, evaluation pipeline
  utils/                 Config, MLflow, GPU, visualization
scripts/
  train.py               Training entry point
  predict.py             Test evaluation for paired CI/DCI models
  preprocess.py          Data preprocessing
  upload_to_mlflow.py    Post-training metrics and visualization upload
  test.py                Unit and integration tests
  app_gradio.py          Interactive demo
  create_velocity_from_itslive_mosaic.py  Velocity fusion pipeline
output/                  Run outputs (checkpoints, logs, metrics)
```

## Citation

```bibtex
@phdthesis{perez2025physics,
  title={Physics-Guided Strategies for Enhancing Neural Networks Trained with Limited Data},
  author={Perez, Jose G.},
  school={The University of Texas at El Paso},
  year={2025},
  advisor={Fuentes, Olac}
}
```
