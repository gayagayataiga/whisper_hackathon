#!/usr/bin/env bash
# setup.sh - Whisper environment setup for Jetson AGX Orin
#
# Usage:
#   ./setup.sh        # Run all steps (idempotent; already-installed items are skipped)
#
# Prerequisites:
#   - Jetson AGX Orin (ARM64 + CUDA, JetPack 6.1, Python 3.10)
#   - Internet connection (to fetch NVIDIA wheels and pip packages)
#
# Steps:
#   1. apt packages          (tmux, python3-venv, libsndfile1)
#   2. Create .venv
#   3. Upgrade pip / setuptools / wheel
#   4. Install JetPack-specific PyTorch wheel
#   5. Install CTranslate2 (local whl preferred; falls back to NVIDIA index)
#   6. Install requirements.txt
#   7. Verify imports
#
# See the header of requirements.txt for detailed instructions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# JetPack 6.1 / cp310 / aarch64 dedicated PyTorch wheel
TORCH_WHL_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

CT2_LOCAL_WHL_DIR="${REPO_DIR}/CTranslate2/python/dist"
CT2_NVIDIA_INDEX="https://pypi.ngc.nvidia.com"

log()   { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[ OK ]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
err()   { printf "\033[1;31m[err ]\033[0m %s\n" "$*" >&2; }

# ── 0. Environment check ──────────────────────────────────────
if [[ "$(uname -m)" != "aarch64" ]]; then
    err "This script is for Jetson (ARM64) only. uname -m=$(uname -m)"
    exit 1
fi

# ── 1. apt packages ───────────────────────────────────────────
log "Step 1/7: Checking apt packages"
APT_PKGS=(tmux python3-venv libsndfile1)
MISSING=()
for pkg in "${APT_PKGS[@]}"; do
    if ! dpkg -s "${pkg}" &>/dev/null; then
        MISSING+=("${pkg}")
    fi
done
if (( ${#MISSING[@]} > 0 )); then
    log "  Missing: ${MISSING[*]} -> sudo apt install"
    sudo apt update
    sudo apt install -y "${MISSING[@]}"
else
    ok "  All apt packages present"
fi

# ── 2. Create .venv ───────────────────────────────────────────
log "Step 2/7: Virtual environment (.venv)"
if [[ ! -x "${PYTHON}" ]]; then
    log "  Creating ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
    ok "  Created"
else
    ok "  Already exists: ${VENV_DIR}"
fi

# ── 3. Upgrade pip etc. ───────────────────────────────────────
log "Step 3/7: Updating pip / setuptools / wheel"
"${PIP}" install --quiet --upgrade pip setuptools wheel
ok "  pip $(${PIP} --version | awk '{print $2}')"

# ── 4. PyTorch (JetPack-specific) ────────────────────────────
log "Step 4/7: PyTorch (JetPack 6.1)"
TORCH_OK=$("${PYTHON}" -c "import torch; print('+' in torch.__version__ and 'cpu' not in torch.__version__)" 2>/dev/null || echo "False")
if [[ "${TORCH_OK}" == "True" ]]; then
    TORCH_VER=$("${PYTHON}" -c "import torch; print(torch.__version__)")
    ok "  Already installed: torch ${TORCH_VER}"
else
    log "  Fetching JetPack wheel from NVIDIA"
    "${PIP}" install --no-cache-dir "${TORCH_WHL_URL}"
    ok "  Installed"
fi

# ── 5. CTranslate2 ───────────────────────────────────────────
log "Step 5/7: CTranslate2"
if "${PYTHON}" -c "import ctranslate2" &>/dev/null; then
    CT2_VER=$("${PYTHON}" -c "import ctranslate2; print(ctranslate2.__version__)")
    ok "  Already installed: ctranslate2 ${CT2_VER}"
else
    LOCAL_WHL=$(ls "${CT2_LOCAL_WHL_DIR}"/*.whl 2>/dev/null | head -1 || true)
    if [[ -n "${LOCAL_WHL}" ]]; then
        log "  Using local wheel: ${LOCAL_WHL}"
        "${PIP}" install "${LOCAL_WHL}"
    else
        warn "  No ${CT2_LOCAL_WHL_DIR}/*.whl found; fetching from NVIDIA index"
        "${PIP}" install ctranslate2 --extra-index-url "${CT2_NVIDIA_INDEX}"
    fi
    ok "  Installed"
fi

# ── 6. requirements.txt ──────────────────────────────────────
log "Step 6/7: requirements.txt"
"${PIP}" install -r "${REPO_DIR}/requirements.txt"
ok "  Done"

# ── 7. Verify imports ────────────────────────────────────────
log "Step 7/7: Verifying imports"
"${PYTHON}" - <<'PY'
import sys
import torch, ctranslate2, faster_whisper, fastapi, uvicorn, httpx, soundfile, numpy
print(f"  python         : {sys.version.split()[0]}")
print(f"  torch          : {torch.__version__}  (cuda={torch.cuda.is_available()})")
print(f"  ctranslate2    : {ctranslate2.__version__}")
print(f"  faster-whisper : {faster_whisper.__version__}")
print(f"  fastapi        : {fastapi.__version__}")
print(f"  numpy          : {numpy.__version__}")
PY

ok "Setup complete"
echo
echo "Next step:"
echo "  WHISPER_RASPI_URL=http://192.168.10.2:9000/command ./start.sh"
