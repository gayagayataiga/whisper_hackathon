# Jetson AGX Orin リアルタイム音声認識 — 環境構築ログ

## 環境

| 項目 | 値 |
|---|---|
| ハードウェア | NVIDIA Jetson AGX Orin |
| GPU | Orin (compute capability 8.7, sm_87, VRAM 30GB) |
| CUDA | 12.6 |
| OS | Ubuntu 22.04 (JetPack) |
| Python | 3.10.12 |
| PyTorch | 2.5.0a0+872d972e41.nv24.08 (JetPack ビルド) |
| CTranslate2 | 4.7.1 (CUDA 対応、ソースビルド) |
| faster-whisper | 1.x |

---

## 主要ファイル

```
~/whisper/
├── requirements.txt              # pip 依存パッケージ（PyTorch 除く）
├── Script/
│   ├── whisper_server.py         # 推論サーバー (port 8001)
│   ├── interface.py              # 受け子サーバー (port 8000)
│   ├── raspi_receiver.py         # Raspi 側の受信サーバー
│   ├── mock_raspi.py             # Raspi 疑似クライアント
│   └── start.sh / stop.sh        # tmux による起動・停止
└── docs/
    └── setup_log.md              # 本ドキュメント
```

> 旧 ZeroMQ 構成（`jetson_inference.py` + `mock_sender.py`）は廃止し、現在は HTTP パイプラインのみ。
> 全体像は `README.md` を参照。

---

## 環境構築手順

### 1. Jetson パフォーマンス最大化

```bash
sudo nvpmodel -m 0      # MAX-N 電力モード
sudo jetson_clocks      # CPU/GPU クロック固定
```

### 2. uv のインストール

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
source ~/.bashrc        # PATH に ~/.local/bin を追加
```

**ポイント:** インストーラが `~/.bashrc` に `~/.local/bin/env` の source 行を追記するが、
既存のターミナルでは反映されないため `source ~/.bashrc` または再起動が必要。

### 3. 仮想環境の作成

```bash
cd ~/whisper
uv venv .venv --python python3
source .venv/bin/activate
```

### 4. ビルドツール・CUDA 開発パッケージのインストール

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

**ポイント:** Jetson の CUDA インストールはランタイムライブラリのみで、
`nvcc` や開発ヘッダ (`cublas_v2.h`, `curand_kernel.h`) は別途 `-dev` パッケージが必要。

### 5. CTranslate2 を CUDA 対応でソースビルド

pip の標準 wheel は ARM64 + CUDA 非対応のため、ソースからビルドする。

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

make -j$(nproc)       # 約 20〜40 分
sudo make install     # /usr/local/lib, /usr/local/include に配置
sudo ldconfig
```

### 6. Python バインディングのビルド

```bash
cd ~/whisper/CTranslate2/python
CT2_ROOT=/usr/local pip install pybind11
CT2_ROOT=/usr/local pip install . --no-build-isolation
```

### 7. __cxa_call_terminate シム の作成 (ABI 互換修正)

**問題:** pip でインストールされた ctranslate2 の bundled ライブラリが
`__cxa_call_terminate` を提供していたが、自前ビルドの CUDA 版にはこのシンボルがない。
`__init__.py` が ImportError をサイレントに飲み込むため、エラーが隠れる。

```bash
# シムライブラリ作成
cat > /tmp/cxa_shim.cc << 'EOF'
#include <cstdlib>
extern "C" { void __cxa_call_terminate(void*) { std::abort(); } }
EOF
sudo g++ -shared -fPIC -o /usr/local/lib/libcxa_shim.so /tmp/cxa_shim.cc
sudo ldconfig

# _ext.so のリンク先を CUDA 版に書き換え
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

### 8. ライブラリパスの永続登録

```bash
# ldconfig でシステムに恒久登録
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/ctranslate2.conf
echo "/usr/local/cuda/targets/aarch64-linux/lib" | sudo tee -a /etc/ld.so.conf.d/ctranslate2.conf
sudo ldconfig

# CTranslate2 の LD_LIBRARY_PATH を venv アクティベート時に自動設定
echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/targets/aarch64-linux/lib:$LD_LIBRARY_PATH' \
    >> ~/whisper/.venv/bin/activate
```

### 9. PyTorch の venv への公開

PyTorch は JetPack ビルド (`~/.local`) に既インストール済み。
venv からアクセスするため `.pth` ファイルで参照を追加。

```bash
# ~/.local のサイトパッケージを venv から参照
echo "$HOME/.local/lib/python3.10/site-packages" \
    > ~/whisper/.venv/lib/python3.10/site-packages/user-site.pth

# CUDA ライブラリディレクトリを sys.path に追加（PyTorch の _preload_cuda_deps 対策）
echo "/usr/local/cuda/targets/aarch64-linux/lib" \
    >> ~/whisper/.venv/lib/python3.10/site-packages/cuda-libs.pth
```

### 10. cuSPARSELt のインストール

PyTorch 2.5.0 (JetPack) が `libcusparseLt.so.0` を要求するが apt に存在しないため、
pip (PyPI) の NVIDIA 公式パッケージから取得。

```bash
pip install nvidia-cusparselt-cu12

echo "/home/tirobot/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib" \
    | sudo tee -a /etc/ld.so.conf.d/ctranslate2.conf
sudo ldconfig
```

### 11. その他パッケージのインストール

```bash
pip install -r ~/whisper/requirements.txt
```

---

## 動作確認済みコマンド

```bash
# ctranslate2 CUDA 確認
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count()); print(ctranslate2.get_supported_compute_types('cuda'))"
# → 1
# → {'int8', 'float16', 'float32', ...}

# PyTorch CUDA 確認
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# → 2.5.0a0+872d972e41.nv24.08 True

# Faster-Whisper ロード確認
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16'); print('OK')"
# → Faster-Whisper large-v3: OK
```

---

## ハマりポイントまとめ

| # | 問題 | 原因 | 解決策 |
|---|---|---|---|
| 1 | `uv` が venv 内で使えない | `~/.local/bin` が PATH 未設定 | `source ~/.bashrc` で反映 |
| 2 | ctranslate2 が CUDA 非対応 | ARM64 向け CUDA wheel が pip に存在しない | ソースビルド |
| 3 | cmake が CUDA を見つけられない | `nvcc` が PATH にない / `-dev` パッケージ未インストール | `cuda-nvcc-12-6` 等を追加 |
| 4 | `cublas_v2.h` 見つからない | `libcublas-dev-12-6` 未インストール | `-dev` パッケージ追加 |
| 5 | `__cxa_call_terminate` undefined | pip bundled ライブラリが提供するシンボルが自前ビルドにない | `libcxa_shim.so` 作成 + patchelf |
| 6 | `StorageView` not found (サイレント失敗) | CWD が `CTranslate2/python/` でソースが優先インポートされる | `~/whisper/` から実行 |
| 7 | `libnvToolsExt.so.1` not found | NVTX ライブラリ未インストール | `sudo apt install libnvtoolsext1` |
| 8 | `libcusparseLt.so.0` not found | apt に存在しない | `pip install nvidia-cusparselt-cu12` |

---

## 実行方法

```bash
cd ~/whisper
source .venv/bin/activate

# 受け子(8000) + 推論(8001) を tmux で起動
cd Script
./start.sh
tmux attach -t whisper       # ログ確認（上ペイン: 推論 / 下ペイン: 受け子）
./status.sh                  # 死活確認
./stop.sh                    # 停止

# 別ターミナルから疑似 Raspi で WAV を送信して動作確認
python mock_raspi.py path/to/audio.wav
```

### テスト用 WAV 変換（ffmpeg）

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 test.wav
```
