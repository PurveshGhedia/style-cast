#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Virtual Fashion Fitting Room — Environment Setup Script
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh            # CPU (local machine)
#   ./setup_env.sh --gpu      # GPU (university cluster)
#
# Requirements: conda OR python 3.10+ with venv
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit immediately on any error

ENV_NAME="vfr-env"
PYTHON_VERSION="3.10"
GPU_MODE=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --gpu) GPU_MODE=true ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Virtual Fashion Fitting Room — Environment Setup   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Mode       : $([ "$GPU_MODE" = true ] && echo 'GPU (CUDA)' || echo 'CPU')"
echo "  Env name   : $ENV_NAME"
echo "  Python     : $PYTHON_VERSION"
echo ""

# ── STEP 1: Detect conda or venv ─────────────────────────────────────────────
if command -v conda &>/dev/null; then
    echo "► Conda detected. Using conda environment."
    USE_CONDA=true
else
    echo "► Conda not found. Using Python venv instead."
    USE_CONDA=false
fi

# ── STEP 2: Create environment ───────────────────────────────────────────────
if [ "$USE_CONDA" = true ]; then
    # Remove existing env if it exists
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true

    echo "► Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

    # Activate
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    # Venv fallback
    if [ -d "$ENV_NAME" ]; then
        echo "► Removing existing venv: $ENV_NAME"
        rm -rf "$ENV_NAME"
    fi

    echo "► Creating venv: $ENV_NAME"
    python3 -m venv "$ENV_NAME"
    source "$ENV_NAME/bin/activate"
fi

echo "► Environment active: $(which python)"
echo "► Python version: $(python --version)"
echo ""

# ── STEP 3: Upgrade pip ──────────────────────────────────────────────────────
echo "► Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

# ── STEP 4: Install PyTorch (CPU or GPU) ─────────────────────────────────────
echo ""
if [ "$GPU_MODE" = true ]; then
    echo "► Installing PyTorch with CUDA 11.8 support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "► Installing PyTorch (CPU only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ── STEP 5: Install project dependencies ─────────────────────────────────────
echo ""
echo "► Installing project dependencies from requirements.txt..."

# Install everything EXCEPT SAM (needs git URL)
grep -v "segment-anything" requirements.txt > /tmp/reqs_nosam.txt
pip install -r /tmp/reqs_nosam.txt

# Install SAM from GitHub
echo ""
echo "► Installing Segment Anything Model (SAM)..."
pip install git+https://github.com/facebookresearch/segment-anything.git

# ── STEP 6: Register Jupyter kernel ──────────────────────────────────────────
echo ""
echo "► Registering Jupyter kernel: $ENV_NAME"
python -m ipykernel install --user --name="$ENV_NAME" --display-name "VFR Project ($PYTHON_VERSION)"

# ── STEP 7: Download SAM checkpoint ──────────────────────────────────────────
echo ""
echo "► Downloading SAM ViT-H checkpoint (~2.4 GB)..."
mkdir -p checkpoints
if [ ! -f "checkpoints/sam_vit_h_4b8939.pth" ]; then
    curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
         -o checkpoints/sam_vit_h_4b8939.pth
    echo "  ✓ SAM checkpoint saved to checkpoints/sam_vit_h_4b8939.pth"
else
    echo "  ✓ SAM checkpoint already exists — skipping download."
fi

# ── STEP 8: Create project folder structure ───────────────────────────────────
echo ""
echo "► Creating project folder structure..."

mkdir -p data/viton_hd/{train,test}/{image,cloth,cloth-mask,image-parse,openpose_json,openpose_img}
mkdir -p outputs/{pose,segmentation,warping,tryon}
mkdir -p checkpoints
mkdir -p notebooks
mkdir -p src/{pose,segmentation,warping,tryon,evaluation,demo}

# Create placeholder __init__.py files
touch src/__init__.py
touch src/pose/__init__.py
touch src/segmentation/__init__.py
touch src/warping/__init__.py
touch src/tryon/__init__.py
touch src/evaluation/__init__.py
touch src/demo/__init__.py

echo "  ✓ Folder structure created."

# ── STEP 9: Verify installation ───────────────────────────────────────────────
echo ""
echo "► Verifying installation..."
python - <<'EOF'
import sys
ok = True

checks = [
    ("numpy",          "numpy"),
    ("opencv",         "cv2"),
    ("mediapipe",      "mediapipe"),
    ("torch",          "torch"),
    ("torchvision",    "torchvision"),
    ("kornia",         "kornia"),
    ("PIL",            "PIL"),
    ("scipy",          "scipy"),
    ("sklearn",        "sklearn"),
    ("lpips",          "lpips"),
    ("tensorboard",    "tensorboard"),
    ("flask",          "flask"),
    ("segment_anything","segment_anything"),
]

for display_name, import_name in checks:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  ✓ {display_name:<22} {ver}")
    except ImportError as e:
        print(f"  ✗ {display_name:<22} MISSING — {e}")
        ok = False

import torch
print(f"\n  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device    : {torch.cuda.get_device_name(0)}")

print()
if ok:
    print("  All dependencies installed successfully.")
else:
    print("  Some dependencies are missing — check errors above.")
    sys.exit(1)
EOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                   Setup complete!                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  To activate this environment next time:"
if [ "$USE_CONDA" = true ]; then
    echo "    conda activate $ENV_NAME"
else
    echo "    source $ENV_NAME/bin/activate"
fi
echo ""
echo "  To open Jupyter:"
echo "    jupyter notebook"
echo ""
echo "  Select kernel: 'VFR Project (3.10)' in Jupyter"
echo ""
