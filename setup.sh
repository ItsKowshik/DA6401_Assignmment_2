#!/usr/bin/env bash
# ============================================================
# DA6401 Assignment 2 — Environment Setup Script
# ============================================================
# Run from the repo root:  bash setup.sh
# ============================================================
set -e

CONDA_ENV_NAME="da6401-a2"

echo "╔══════════════════════════════════════════════════════╗"
echo "║    DA6401 A2 — Environment Setup                    ║"
echo "╚══════════════════════════════════════════════════════╝"

# ── Step 0: Check CUDA ──────────────────────────────────────
echo ""
echo "── Step 0: Detecting CUDA ─────────────────────────────"
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' 2>/dev/null || echo "NOT FOUND")
echo "  System CUDA: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == 12* ]]; then
    CUDA_FLAG="pytorch-cuda=12.1"
elif [[ "$CUDA_VERSION" == 11* ]]; then
    CUDA_FLAG="pytorch-cuda=11.8"
else
    CUDA_FLAG="cpuonly"
    echo "  ⚠ No CUDA found — installing CPU-only PyTorch"
fi
echo "  Using: $CUDA_FLAG"

# ── Step 1: Choose creation method ─────────────────────────
echo ""
echo "── Step 1: Creating Conda Environment ─────────────────"
echo "  Choose method:"
echo "  [1] Fresh environment from environment.yml (RECOMMENDED)"
echo "  [2] Clone base environment, then add packages"
read -rp "  Enter choice [1/2] (default: 1): " METHOD
METHOD="${METHOD:-1}"

if [[ "$METHOD" == "2" ]]; then
    echo "  Cloning base environment (this may take a few minutes)..."
    conda create --name "$CONDA_ENV_NAME" --clone base -y
    echo "  ✓ Cloned base → $CONDA_ENV_NAME"
else
    echo "  Creating fresh environment from environment.yml..."
    sed "s/pytorch-cuda=12.1/$CUDA_FLAG/" environment.yml > /tmp/env_patched.yml
    conda env create -f /tmp/env_patched.yml -y
    echo "  ✓ Environment created: $CONDA_ENV_NAME"
fi

# ── Step 2: Install pip requirements (for clone path) ──────
if [[ "$METHOD" == "2" ]]; then
    echo ""
    echo "── Step 2: Installing PyTorch + pip requirements ───────"
    conda run -n "$CONDA_ENV_NAME" conda install \
        pytorch torchvision torchaudio "$CUDA_FLAG" \
        -c pytorch -c nvidia -y
    conda run -n "$CONDA_ENV_NAME" pip install -r requirements.txt
    echo "  ✓ Packages installed"
fi

# ── Step 3: Register Jupyter kernel ────────────────────────
echo ""
echo "── Step 3: Registering Jupyter Kernel ─────────────────"
conda run -n "$CONDA_ENV_NAME" python -m ipykernel install \
    --user \
    --name "$CONDA_ENV_NAME" \
    --display-name "DA6401 A2 (Python 3.10)"
echo "  ✓ Kernel registered: 'DA6401 A2 (Python 3.10)'"

# ── Step 4: W&B login ───────────────────────────────────────
echo ""
echo "── Step 4: Weights & Biases Login ─────────────────────"
echo "  ⚠ You will need your W&B API key from: https://wandb.ai/authorize"
read -rp "  Login to W&B now? [y/N]: " WANDB_LOGIN
if [[ "$WANDB_LOGIN" =~ ^[Yy]$ ]]; then
    conda run -n "$CONDA_ENV_NAME" wandb login
fi

# ── Step 5: Verify Installation ────────────────────────────
echo ""
echo "── Step 5: Verifying Installation ─────────────────────"
conda run -n "$CONDA_ENV_NAME" python - << 'PYCHECK'
import sys
import torch
import numpy as np
import matplotlib
import sklearn
import wandb
import albumentations as A

print(f"  Python      : {sys.version.split()[0]}")
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
print(f"  NumPy       : {np.__version__}")
print(f"  Matplotlib  : {matplotlib.__version__}")
print(f"  Scikit-learn: {sklearn.__version__}")
print(f"  W&B         : {wandb.__version__}")
print(f"  Albumentations: {A.__version__}")

if torch.cuda.is_available():
    x = torch.randn(4, 3, 224, 224).cuda()
    print(f"\n  ✓ GPU smoke test passed: {x.shape} on {x.device}")
else:
    print("\n  ⚠ Running on CPU only")

print("\n  ✓ All packages verified successfully!")
PYCHECK

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✓ Setup Complete!                                   ║"
echo "║                                                      ║"
echo "║  Activate:   conda activate da6401-a2                ║"
echo "║  Deactivate: conda deactivate                        ║"
echo "╚══════════════════════════════════════════════════════╝"
