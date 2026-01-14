# Directory Structure

This document explains the organization of the codebase.

## Root Directory

The root directory contains **essential files** - scripts you'll run frequently:

```
cmu-10799-diffusion/
├── train.py                 # Main training script
├── sample.py                # Generate samples from trained models
├── download_dataset.py      # Download CelebA dataset from HuggingFace
├── modal_app.py             # Modal cloud GPU setup
├── setup.sh                 # Setup script (pip + venv)
├── setup-uv.sh              # Setup script (uv - faster!)
├── pyproject.toml           # Python package configuration
└── README.md                # Main documentation (start here!)
```

**TODOs in this directory:**
1. train.py: incorporate your sampling scheme to the training pipeline and save generated samples as images for logging
2. sample.py: incorporate your sampling scheme to the training pipeline and save generated samples as images

## Subdirectories

### `src/` - Source Code

The main implementation:

```
src/
├── __init__.py
├── models/                 # U-Net architecture
│   ├── __init__.py
│   ├── blocks.py           # Building blocks (ResBlock, Attention, etc.)
│   └── unet.py             # Complete U-Net model, TODO
├── methods/                # Training methods
│   ├── __init__.py
│   ├── base.py             # Base method class
│   └── ddpm.py             # DDPM implementation, TODO
├── data/                   # Dataset loading
│   ├── __init__.py
│   └── celeba.py           # CelebA dataset, TODO
└── utils/                  # Utilities
    ├── __init__.py
    ├── ema.py              # Exponential Moving Average
    └── logging_utils.py    # Logging utilities
```

**TODOs in this directory:**
1. src/data/celeba.py: fill in your data transforms functions
2. src/methods/ddpm.py: implement everything in this file
3. src/models/unet.py: implement the unet model architecture and its forward pass

### `configs/` - Hyperparameter Configurations

Training hyperparameters and model settings:

```
configs/
├── ddpm_modal.yaml         # DDPM config for Modal cloud training
└── ddpm_babel.yaml         # DDPM config for Babel cluster
```

**TODOs in this directory:**
Create your own model configs

### `environments/` - Environment Configurations

Requirements files for different hardware setups:

```
environments/
├── requirements.txt           # Base dependencies
├── requirements-cpu.txt       # CPU-only PyTorch
├── requirements-cuda118.txt   # PyTorch + CUDA 11.8
├── requirements-cuda121.txt   # PyTorch + CUDA 12.1
├── requirements-cuda126.txt   # PyTorch + CUDA 12.6
├── requirements-cuda129.txt   # PyTorch + CUDA 12.9
└── requirements-rocm.txt      # PyTorch + ROCm (AMD)
```

**Which one to use?**
- Have a laptop/no GPU? → Use `requirements-cpu.txt` + Modal for training
- Have an NVIDIA GPU? → Use the matching CUDA version (check with `nvidia-smi`)

### `docs/` - Documentation

All guides and documentation:

```
docs/
├── SETUP.md                 # Detailed setup for all platforms (Modal, AWS, SLURM)
├── QUICKSTART-MODAL.md      # 5-minute quick start for Modal users
└── DIRECTORY-STRUCTURE.md   # This file - explains project organization
```

**Start here based on your setup:**
- **Laptop user (no GPU)?** → Read [QUICKSTART-MODAL.md](QUICKSTART-MODAL.md)
- **Have a GPU or cluster?** → Read [SETUP.md](SETUP.md)

### `scripts/` - Shell Scripts

Example scripts for cluster and Modal environments, feel free to modify:

```
scripts/
├── train.sh                          # SLURM job template (tested on babel)
├── evaluate_torch_fidelity.sh        # torch-fidelity evaluation on cluster or locally (tested on babel)
├── evaluate_modal_torch_fidelity.sh  # torch-fidelity evaluation on Modal
└── list_checkpoints.sh               # List Modal checkpoints
```

### `notebooks/` - Jupyter Notebooks

Helper notebooks skeleton for exploration and visualization, feel free to modify:

```
notebooks/
├── 01_1d_playground.ipynb            # 1D diffusion experiments for intuition
├── 02_dataset_exploration.ipynb      # Explore CelebA dataset
└── 03_sampling_visualization.ipynb   # Visualize sampling process
```
