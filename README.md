# Glacier Mapping by Segmentation

Modern PyTorch Lightning implementation for glacier segmentation using U-Net architecture with boundary-aware loss functions.

## 🚀 Quick Start

This project uses `uv` for package management. See **AGENTS.md** for complete development guide.

### Environment Setup
```bash
# Install dependencies
uv pip install -e .
uv pip install -e ".[dev]"
```

### Training (Gen6 Currently Executing)
```bash
# Frodo - Parallel GPU execution
./run_sequential_training.sh frodo --gpu 0 --tasks ci --gpu-filter gpu0     # Clean Ice
./run_sequential_training.sh frodo --gpu 1 --tasks dci --gpu-filter gpu1   # Debris Ice
./run_sequential_training.sh frodo --gpu 2 --tasks multi --gpu-filter gpu2 # Multi-class
./run_sequential_training.sh frodo --gpu 3 --tasks ci,dci,multi --gpu-filter gpu3 # Validation

# Bilbo - Heavy lifting (4090)
./run_sequential_training.sh bilbo --gpu 1 --tasks dci,ci,multi --priority dci,ci,multi

# Desktop - FREE for user experiments
./run_sequential_training.sh desktop --dry-run  # Shows "No config files found"
```

### Development Commands
```bash
# Linting and formatting
ruff check .
ruff format .

# Type checking
pyright glacier_mapping/

# MLflow upload
uv run python scripts/upload_to_mlflow.py output/run_name --regenerate
```

## 📁 Project Structure

```
Python-Glacier-Mapping-by-Segmentation/
├── configs/                    # Hierarchical configuration system
│   ├── bilbo/                 # 4090 heavy synthesis experiments
│   ├── frodo/                 # 4x 2080 Ti parallel testing
│   ├── desktop/               # Quick validation (currently empty)
│   ├── tasks/                 # Task-specific configurations
│   ├── _archive_gen*/          # Historical documentation
│   └── train.yaml             # Global base configuration
├── glacier_mapping/           # Main package (Lightning modules)
│   ├── lightning/             # PyTorch Lightning implementation
│   ├── model/                # U-Net architecture & losses
│   ├── data/                 # Data loading & preprocessing
│   └── utils/                # Utilities & analysis
├── scripts/                   # Training & utility scripts
├── AGENTS.md                  # Complete development guide
├── TODO.md                   # Active development notes
└── pyproject.toml            # Project configuration
```

## 📚 Documentation

### Essential Reading
- **AGENTS.md** - Complete development commands, architecture guide, and best practices
- **TODO.md** - Active development items and known issues

### Historical Documentation
- `archive/gen8_configs/` - 512x512 window hypothesis testing (with known regressions)
- `archive/gen7_configs/` - Physics channel optimization
- `archive/gen6/` - Current generation implementation and analysis
- `archive/gen3/` - 256px breakthrough experiments  
- `archive/gen1_gen2/` - Early generation findings

## 🔬 Research

Based on "Boundary Aware U-Net for Glacier Segmentation" (https://doi.org/10.7557/18.6789)

### Key Features
- **Boundary-aware loss** for improved debris-covered ice segmentation
- **Multi-spectral Landsat 7** imagery with physics-derived features
- **PyTorch Lightning** for simplified training and experimentation
- **MLflow integration** for experiment tracking and reproducibility
- **Multi-GPU support** with balanced workload distribution

### Tasks
- **Clean Ice (CI)**: Segment clean glacier ice
- **Debris-Covered Ice (DCI)**: Segment debris-covered glacier ice  
- **Multi-class**: Unified segmentation of all ice types

## 🖥️ Hardware Configuration

| **Server** | **GPU** | **Purpose** | **Status** |
|-------------|-----------|-------------|------------|
| **Desktop** | RTX 3060 Ti | User experiments | ✅ **FREE** |
| **Frodo** | 4x RTX 2080 Ti | Parallel testing | ✅ **ACTIVE** |
| **Bilbo** | RTX 4090 | Heavy synthesis | ✅ **ACTIVE** |

## 🤝 Contributing

1. Follow code style guidelines in **AGENTS.md**
2. Use `uv` for all package management
3. Run `ruff check .` and `ruff format .` before commits
4. Update **TODO.md** for any new development items

## 📄 License

This project builds on research published in the Proceedings of the Northern Lights Deep Learning Workshop.

---

**Status**: ✅ **Gen6 Currently Executing**  
**Total Configs**: 26 perfectly balanced experiments  
**Desktop**: ✅ **Free for user experiments**