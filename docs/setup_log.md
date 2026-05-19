# Jetson AGX Orin Real-Time Speech Recognition — Environment Setup Log

## Environment

| Item | Value |
|---|---|
| Hardware | NVIDIA Jetson AGX Orin |
| GPU | Orin (compute capability 8.7, sm_87, VRAM 30GB) |
| CUDA | 12.6 |
| OS | Ubuntu 22.04 (JetPack) |
| Python | 3.10.12 |
| PyTorch | 2.5.0a0+872d972e41.nv24.08 (JetPack build) |
| CTranslate2 | 4.7.1 (CUDA-enabled, source build) |
| faster-whisper | 1.x |

---

## Key Files

```
~/whisper/
├── requirements.txt              # pip dependencies (excluding PyTorch)
├── Script/
│   ├── whisper_server.py         # Inference server (port 8001)
│   ├── interface.py              # Receiver server (port 8000)
│   ├── raspi_receiver.py         # Receiving server on the Raspi side
│   ├── mock_raspi.py             # Mock Raspi client
│   └── start.sh / stop.sh        # tmux-based start/stop
└── docs/
    └── setup_log.md              # This document
```

> The old ZeroMQ configuration (`jetson_inference.py` + `mock_sender.py`) has been retired; only the HTTP pipeline is used now.
> See `README.md` for an overview of the full system.

---

## Setup Steps

### 1. Maximize Jetson Performance

```bash
sudo nvpmodel -m 0      # MAX-N power mode
sudo jetson_clocks      # Lock CPU/GPU clocks
```

### 2. Install uv

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
source ~/.bashrc        # Add ~/.local/bin to PATH
```

**Note:** The installer appends a source line for `~/.local/bin/env` to `~/.bashrc`,
but existing terminals won't pick it up — run `source ~/.bashrc` or restart the terminal.

### 3. Create a Virtual Environment

```bash
cd ~/whisper
uv venv .venv --python python3
source .venv/bin/activate
```

### 4. Install Build Tools and CUDA Development Packages

```bash
sudo apt-get install -y \
    cmake build-essential python3-dev \
    cuda-nvcc-12-6 \
    libcublas-dev-12-6 \
    libcurand-dev-12-6 \
    libnvtoolsext1 \
    patchelf

export PATH=/usr/local/cuda-12.6/bin:$PATH
```

**Note:** The Jetson CUDA installation only includes runtime libraries;
`nvcc` and development headers (`cublas_v2.h`, `curand_kernel.h`) require separate `-dev` packages.

### 5. Source-Build CTranslate2 with CUDA Support

The standard pip wheel does not support ARM64 + CUDA, so build from source.

```bash
cd ~/whisper
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2

mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
    -DCUDA_INCLUDE_DIRS=/usr/local/cuda-12.6/targets/aarch64-linux/include \
    -DWITH_CUDNN=ON \
    -DWITH_MKL=OFF \
    -DOPENMP_RUNTIME=COMP \
    -DCMAKE_CUDA_ARCHITECTURES=87   # Jetson AGX Orin = sm_87

make -j$(nproc)       # Approx. 20-40 minutes
sudo make install     # Installs to /usr/local/lib, /usr/local/include
sudo ldconfig
```

### 6. Build the Python Bindings

```bash
cd ~/whisper/CTranslate2/python
CT2_ROOT=/usr/local pip install pybind11
CT2_ROOT=/usr/local pip install . --no-build-isolation
```

### 7. Create a `__cxa_call_terminate` Shim (ABI Compatibility Fix)

**Problem:** The bundled library shipped with the pip-installed ctranslate2 provided
`__cxa_call_terminate`, but the custom CUDA build does not include this symbol.
`__init__.py` silently swallows the ImportError, hiding the failure.

```bash
# Create the shim library
cat > /tmp/cxa_shim.cc << 'EOF'
#include <cstdlib>
extern "C" { void __cxa_call_terminate(void*) { std::abort(); } }
EOF
sudo g++ -shared -fPIC -o /usr/local/lib/libcxa_shim.so /tmp/cxa_shim.cc
sudo ldconfig

# Relink _ext.so to the CUDA build
EXT=$(find ~/whisper/.venv -name "_ext*.so")
patchelf --replace-needed \
    libctranslate2-ac01f8af.so.4.7.1 \
    libctranslate2.so.4 \
    "$EXT"
patchelf --set-rpath \
    "/usr/local/lib:/usr/local/cuda/targets/aarch64-linux/lib" \
    "$EXT"
patchelf --add-needed libcxa_shim.so "$EXT"
```

### 8. Persist Library Paths

```bash
# Register permanently via ldconfig
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/ctranslate2.conf
echo "/usr/local/cuda/targets/aarch64-linux/lib" | sudo tee -a /etc/ld.so.conf.d/ctranslate2.conf
sudo ldconfig

# Automatically set CTranslate2 LD_LIBRARY_PATH when the venv is activated
echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/targets/aarch64-linux/lib:$LD_LIBRARY_PATH' \
    >> ~/whisper/.venv/bin/activate
```

### 9. Expose PyTorch to the venv

PyTorch is already installed in the JetPack build (`~/.local`).
Add a `.pth` file so the venv can find it.

```bash
# Reference ~/.local site-packages from inside the venv
echo "$HOME/.local/lib/python3.10/site-packages" \
    > ~/whisper/.venv/lib/python3.10/site-packages/user-site.pth

# Add the CUDA library directory to sys.path (workaround for PyTorch _preload_cuda_deps)
echo "/usr/local/cuda/targets/aarch64-linux/lib" \
    >> ~/whisper/.venv/lib/python3.10/site-packages/cuda-libs.pth
```

### 10. Install cuSPARSELt

PyTorch 2.5.0 (JetPack) requires `libcusparseLt.so.0`, which is not in apt,
so obtain it from NVIDIA's official PyPI package.

```bash
pip install nvidia-cusparselt-cu12

echo "/home/tirobot/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib" \
    | sudo tee -a /etc/ld.so.conf.d/ctranslate2.conf
sudo ldconfig
```

### 11. Install Remaining Packages

```bash
pip install -r ~/whisper/requirements.txt
```

---

## Verified Commands

```bash
# Verify ctranslate2 CUDA support
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count()); print(ctranslate2.get_supported_compute_types('cuda'))"
# → 1
# → {'int8', 'float16', 'float32', ...}

# Verify PyTorch CUDA support
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# → 2.5.0a0+872d972e41.nv24.08 True

# Verify Faster-Whisper loads
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16'); print('OK')"
# → Faster-Whisper large-v3: OK
```

---

## Common Pitfalls

| # | Problem | Cause | Solution |
|---|---|---|---|
| 1 | `uv` not available inside the venv | `~/.local/bin` not in PATH | Run `source ~/.bashrc` to apply |
| 2 | ctranslate2 has no CUDA support | No CUDA wheel for ARM64 on pip | Build from source |
| 3 | cmake cannot find CUDA | `nvcc` not in PATH / `-dev` packages not installed | Install `cuda-nvcc-12-6` etc. |
| 4 | `cublas_v2.h` not found | `libcublas-dev-12-6` not installed | Install the `-dev` package |
| 5 | `__cxa_call_terminate` undefined | Symbol provided by pip bundled lib is absent from the custom build | Create `libcxa_shim.so` + patchelf |
| 6 | `StorageView` not found (silent failure) | CWD is `CTranslate2/python/` so source is imported with priority | Run from `~/whisper/` |
| 7 | `libnvToolsExt.so.1` not found | NVTX library not installed | `sudo apt install libnvtoolsext1` |
| 8 | `libcusparseLt.so.0` not found | Not available in apt | `pip install nvidia-cusparselt-cu12` |

---

## How to Run

```bash
cd ~/whisper
source .venv/bin/activate

# Start receiver (8000) + inference (8001) via tmux
cd Script
./start.sh
tmux attach -t whisper       # View logs (top pane: inference / bottom pane: receiver)
./status.sh                  # Health check
./stop.sh                    # Stop

# Send a WAV from another terminal using the mock Raspi client to verify operation
python mock_raspi.py path/to/audio.wav
```

### WAV Conversion for Testing (ffmpeg)

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 test.wav
```
