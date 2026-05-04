#!/bin/bash
# setup_sam2.sh  —  Deploy SAM2 WITHOUT any CUDA compilation
#
# Strategy: clone repo → inject into Python path via .pth file → download weights
# NO pip install -e, NO nvcc, NO torch.utils.cpp_extension
#
# Usage (from repo root):
#   bash contrastive_pretrain/setup_sam2.sh

set -e

CONDA_ENV="modeltrain"
SAM2_SRC="/data/home/shujia/CHD/model_train/sam2"
WEIGHTS_DIR="/data/home/shujia/CHD/model_train/RETFound_MAE-main/pretrained_weights/sam2"

echo "================================================================"
echo " SAM2 deployment (no-compile mode)"
echo " src     : $SAM2_SRC"
echo " weights : $WEIGHTS_DIR"
echo "================================================================"

# ── 1. Clone ──────────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Cloning SAM2..."
if [ -d "$SAM2_SRC" ]; then
    echo "  already exists, pulling latest..."
    git -C "$SAM2_SRC" pull --ff-only
else
    git clone --depth 1 https://github.com/facebookresearch/sam2.git "$SAM2_SRC"
    echo "  cloned to $SAM2_SRC"
fi

# ── 2. Inject into Python path (NO compilation) ───────────────────────────────
echo ""
echo "[2/4] Injecting SAM2 into Python path (no compile)..."
SITE_PKG=$(conda run -n $CONDA_ENV python -c \
    "import site; print(site.getsitepackages()[0])")
PTH_FILE="$SITE_PKG/sam2_src.pth"
echo "$SAM2_SRC" > "$PTH_FILE"
echo "  wrote $PTH_FILE → $SAM2_SRC"

# Install only the two missing pure-Python dependencies
echo "  installing hydra-core and iopath (pure Python, no CUDA)..."
conda run -n $CONDA_ENV pip install hydra-core iopath --quiet --no-deps
echo "  dependencies done"

# ── 3. Verify import ──────────────────────────────────────────────────────────
echo ""
echo "[3/4] Verifying import..."
conda run -n $CONDA_ENV python - <<'EOF'
import sys
try:
    import sam2
    from sam2.modeling.backbones.hieradet import Hiera
    print(f"  sam2 path : {sam2.__file__}")
    print(f"  Hiera     : OK")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)
EOF

# ── 4. Download weights ───────────────────────────────────────────────────────
echo ""
echo "[4/4] Downloading SAM2.1 weights..."
mkdir -p "$WEIGHTS_DIR"

download() {
    local fname="$1"
    local url="$2"
    local dest="$WEIGHTS_DIR/$fname"
    if [ -f "$dest" ]; then
        echo "  [skip] $fname ($(du -sh "$dest" | cut -f1))"
    else
        echo "  downloading $fname ..."
        wget --quiet --show-progress -O "$dest" "$url"
        echo "  [done] $fname ($(du -sh "$dest" | cut -f1))"
    fi
}

BASE="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
download "sam2.1_hiera_tiny.pt"      "$BASE/sam2.1_hiera_tiny.pt"
download "sam2.1_hiera_small.pt"     "$BASE/sam2.1_hiera_small.pt"
download "sam2.1_hiera_base_plus.pt" "$BASE/sam2.1_hiera_base_plus.pt"

# MedSAM2 (optional — skip quietly if URL is unavailable)
MEDSAM2="$WEIGHTS_DIR/MedSAM2_pretrain.pth"
if [ ! -f "$MEDSAM2" ]; then
    echo "  downloading MedSAM2 (~900 MB, may take a while)..."
    wget --quiet --show-progress \
        -O "$MEDSAM2" \
        "https://huggingface.co/jiayuanz3/MedSAM2/resolve/main/MedSAM2_pretrain.pth" \
        && echo "  [done] MedSAM2_pretrain.pth ($(du -sh "$MEDSAM2" | cut -f1))" \
        || { rm -f "$MEDSAM2"; echo "  [skip] MedSAM2 download failed — will add later"; }
else
    echo "  [skip] MedSAM2_pretrain.pth ($(du -sh "$MEDSAM2" | cut -f1))"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Done. Weights:"
ls -lh "$WEIGHTS_DIR/"
echo ""
echo " Next: verify encoder output shapes"
echo "   conda run -n $CONDA_ENV python contrastive_pretrain/test_sam2_encoder.py"
echo "================================================================"
