# whisper

Jetson AGX Orin 上で動作する Whisper 音声文字起こしパイプライン。
Raspberry Pi から送られてきた音声を受け取り、`faster-whisper` で推論し、結果を Raspi へ転送する。

## アーキテクチャ

```
Raspi
  │  HTTP POST (wav)
  ▼
受け子サーバー  Script/interface.py        :8000
  │  HTTP POST (wav)
  ▼
推論サーバー    Script/whisper_server.py   :8001
  │  HTTP Response (text)
  ▼
受け子サーバー
  ├─ タイムスタンプ付き txt 追記保存
  └─ HTTP POST (text) → Raspi
```

Raspi 側の受信・モックは `Script/raspi_receiver.py` / `Script/mock_raspi.py`。

## 主な構成

| パス | 役割 |
| --- | --- |
| `Script/interface.py` | 受け子サーバー (port 8000) |
| `Script/whisper_server.py` | Whisper 推論サーバー (port 8001) |
| `Script/raspi_receiver.py` | Raspi 側の文字起こし結果受信 |
| `Script/mock_raspi.py` | Raspi 録音→送信の疑似クライアント |
| `Script/modules/` | `whisper_runner.py` / `vram.py` などの共通処理 |
| `Script/tools/` | `convert_mp3_to_wav.py`, `resample_wav.py` |
| `Script/start.sh` / `stop.sh` / `status.sh` | tmux セッションでの起動・停止・死活確認 |
| `docs/` | セットアップ・ストレステスト等のメモ |
| `requirements.txt` | Python 依存（インストール手順はファイル冒頭参照） |

## モデル設定

`Script/whisper_server.py` 冒頭の定数で変更可能。

- モデル: `large-v3`
- デバイス: `cuda` (Jetson 統合 GPU)
- compute_type: `float16` (Tensor Core)
- 言語: `en` 固定
- beam_size: 5 / best_of: 5
- 直前の認識テキストを `initial_prompt` に渡してコンテキスト保持（最大 200 文字）

## セットアップ

依存のインストールには順序があるので `requirements.txt` 冒頭の手順に従う:

1. JetPack 用 PyTorch ホイール
2. ソースビルド済み CTranslate2 (`CTranslate2/python/dist/*.whl`)
3. `pip install -r requirements.txt`
4. `sudo apt install libsndfile1 tmux`

## 起動

```bash
cd Script
./start.sh                  # 推論(8001) + 受け子(8000) を tmux で起動
tmux attach -t whisper      # ログ確認（上ペイン: 推論 / 下ペイン: 受け子）
./status.sh                 # 死活確認
./stop.sh                   # 停止
```

## エンドポイント

### 受け子サーバー (`:8000`)
- `POST /audio` — Raspi から wav を受信、推論サーバーへ転送
- `POST /image` — Raspi から画像を保存
- `GET  /health`

### 推論サーバー (`:8001`)
- `POST /transcribe` — wav を受け取り文字起こし結果を返す
- `GET  /health` — モデルロード状態と VRAM 使用量

## ディレクトリ補足

`music/`, `results/`, `transcripts/`, `Script/audio/`, `Script/images/`, `.venv/`, `CTranslate2/`（vendored）は `.gitignore` で除外している。
