#!/usr/bin/env bash
# setup.sh - Jetson AGX Orin 用 Whisper 環境セットアップ
#
# 使い方:
#   ./setup.sh        # 全ステップ実行(冪等。既に入っていればスキップ)
#
# 前提:
#   - Jetson AGX Orin (ARM64 + CUDA, JetPack 6.1, Python 3.10)
#   - インターネット接続あり (NVIDIA wheel と pip パッケージ取得のため)
#
# 実行内容:
#   1. apt パッケージ        (tmux, python3-venv, libsndfile1)
#   2. .venv 作成
#   3. pip / setuptools / wheel を最新化
#   4. JetPack 専用 PyTorch wheel をインストール
#   5. CTranslate2 をインストール (ローカル whl 優先、無ければ NVIDIA index)
#   6. requirements.txt をインストール
#   7. import 動作確認
#
# 詳細手順は requirements.txt 冒頭参照。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# JetPack 6.1 / cp310 / aarch64 専用 PyTorch wheel
TORCH_WHL_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

CT2_LOCAL_WHL_DIR="${REPO_DIR}/CTranslate2/python/dist"
CT2_NVIDIA_INDEX="https://pypi.ngc.nvidia.com"

log()   { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[ OK ]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
err()   { printf "\033[1;31m[err ]\033[0m %s\n" "$*" >&2; }

# ── 0. 環境確認 ──────────────────────────────────────────────
if [[ "$(uname -m)" != "aarch64" ]]; then
    err "このスクリプトは Jetson (ARM64) 専用です。uname -m=$(uname -m)"
    exit 1
fi

# ── 1. apt パッケージ ────────────────────────────────────────
log "Step 1/7: apt パッケージを確認"
APT_PKGS=(tmux python3-venv libsndfile1)
MISSING=()
for pkg in "${APT_PKGS[@]}"; do
    if ! dpkg -s "${pkg}" &>/dev/null; then
        MISSING+=("${pkg}")
    fi
done
if (( ${#MISSING[@]} > 0 )); then
    log "  不足: ${MISSING[*]} → sudo apt install"
    sudo apt update
    sudo apt install -y "${MISSING[@]}"
else
    ok "  apt パッケージは揃っています"
fi

# ── 2. .venv 作成 ────────────────────────────────────────────
log "Step 2/7: 仮想環境 (.venv)"
if [[ ! -x "${PYTHON}" ]]; then
    log "  ${VENV_DIR} を作成"
    python3 -m venv "${VENV_DIR}"
    ok "  作成完了"
else
    ok "  既に存在: ${VENV_DIR}"
fi

# ── 3. pip 等を最新化 ────────────────────────────────────────
log "Step 3/7: pip / setuptools / wheel を更新"
"${PIP}" install --quiet --upgrade pip setuptools wheel
ok "  pip $(${PIP} --version | awk '{print $2}')"

# ── 4. PyTorch (JetPack 専用) ────────────────────────────────
log "Step 4/7: PyTorch (JetPack 6.1)"
TORCH_OK=$("${PYTHON}" -c "import torch; print('+' in torch.__version__ and 'cpu' not in torch.__version__)" 2>/dev/null || echo "False")
if [[ "${TORCH_OK}" == "True" ]]; then
    TORCH_VER=$("${PYTHON}" -c "import torch; print(torch.__version__)")
    ok "  既にインストール済み: torch ${TORCH_VER}"
else
    log "  NVIDIA から JetPack 用 wheel を取得"
    "${PIP}" install --no-cache-dir "${TORCH_WHL_URL}"
    ok "  インストール完了"
fi

# ── 5. CTranslate2 ───────────────────────────────────────────
log "Step 5/7: CTranslate2"
if "${PYTHON}" -c "import ctranslate2" &>/dev/null; then
    CT2_VER=$("${PYTHON}" -c "import ctranslate2; print(ctranslate2.__version__)")
    ok "  既にインストール済み: ctranslate2 ${CT2_VER}"
else
    LOCAL_WHL=$(ls "${CT2_LOCAL_WHL_DIR}"/*.whl 2>/dev/null | head -1 || true)
    if [[ -n "${LOCAL_WHL}" ]]; then
        log "  ローカル wheel を使用: ${LOCAL_WHL}"
        "${PIP}" install "${LOCAL_WHL}"
    else
        warn "  ${CT2_LOCAL_WHL_DIR}/*.whl が無いので NVIDIA index から取得"
        "${PIP}" install ctranslate2 --extra-index-url "${CT2_NVIDIA_INDEX}"
    fi
    ok "  インストール完了"
fi

# ── 6. requirements.txt ──────────────────────────────────────
log "Step 6/7: requirements.txt"
"${PIP}" install -r "${REPO_DIR}/requirements.txt"
ok "  完了"

# ── 7. 動作確認 ──────────────────────────────────────────────
log "Step 7/7: import 動作確認"
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

ok "セットアップ完了"
echo
echo "次のステップ:"
echo "  WHISPER_RASPI_URL=http://192.168.10.2:9000/command ./start.sh"
