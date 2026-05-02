# Script 一覧

## メインサーバー

| スクリプト | ポート | 役割 |
|-----------|--------|------|
| `Script/whisper_server.py` | 8001 | Whisper 推論サーバー |
| `Script/interface.py` | 8000 | 受け子サーバー（中継） |

**`whisper_server.py`**
- WAV バイト列を受け取り Faster-Whisper (large-v3, float16, CUDA) でテキスト化
- `POST /transcribe` : WAV → `{"text", "language", "duration_s", "inference_s", "rtf"}`
- `GET  /health` : 死活確認 + モデルロード状態 + VRAM 使用量

**`interface.py`**
- Raspi から WAV を受け取り推論サーバー(8001)へ転送、結果を保存・Raspi 送信
- `POST /audio` : WAV → 推論 → `transcripts/transcript.txt` / `transcripts/transcript.jsonl` に追記 → Raspi(`RASPI_URL`)へ送信
- `POST /image` : 画像保存
- `GET  /health` : 死活確認

```
Raspi → POST /audio (8000) → POST /transcribe (8001) → text → transcripts/{txt,jsonl}
                                                              → Raspi (9000)
```

---

## テスト・送信ツール

**`mock_raspi.py`** — HTTP 経由の疑似 Raspi

```bash
python mock_raspi.py --generate                        # テストトーン生成して送信
python mock_raspi.py music/BASIC5000_0001.wav          # WAVファイル送信
python mock_raspi.py music/news_audio/xxx.wav          # 48kHzも自動リサンプリング
python mock_raspi.py audio.wav --split                 # VADシミュレートして分割送信
python mock_raspi.py audio.wav --loop --interval 1.0   # ループ送信
python mock_raspi.py --list                            # 使えるWAVファイル一覧
```

**`raspi_receiver.py`** — Raspi 側の受信サーバー
- 受け子サーバーから飛んでくる文字起こし結果を `received_transcripts.txt` に追記。
- `python raspi_receiver.py --port 9000`

**`Script/test/test_whisper_server.py`** — 推論サーバー単体への手動疎通スクリプト
- `python Script/test/test_whisper_server.py path/to/audio.wav`

---

## モジュール

| ファイル | 内容 |
|---------|------|
| `modules/whisper_runner.py` | Whisper モデルロード・`transcribe_audio()`（RTF計測付き） |
| `modules/vram.py` | CUDA ランタイム直呼びで VRAM 使用量取得（Jetson の NVML 非対応対策） |

---

## ベンチマーク

| ファイル | 内容 |
|---------|------|
| `benchmark/stress_test.py` | 全モデルサイズ × 10回推論、VRAM・RTF を JSON 保存 |
| `benchmark/accuracy_report.py` / `accuracy_report_en.py` | large-v3 を正解として CER/WER を集計 |
| `benchmark/combined_report.py` / `combined_report_en.py` | ストレステスト結果の統合レポート |
| `benchmark/news_benchmark.py` | ニュース音声を使ったベンチマーク |
| `benchmark/whisper_benchmark.py` | LibriSpeech WER 評価 |
| `benchmark/turbo_run.py` / `turbo_vram_profile.py` | turbo モデル特化のスクリプト |
| `benchmark/transcript_accuracy.py` | 転写精度集計ユーティリティ |
| `benchmark/wer_utils.py` | WER 計算共通関数 |

---

## ツール

| ファイル | 内容 |
|---------|------|
| `tools/convert_mp3_to_wav.py` | MP3 → WAV 変換（ffmpeg 使用） |
| `tools/resample_wav.py` | WAV のサンプルレート変換 |

---

## 起動制御

| ファイル | 役割 |
|---------|------|
| `start.sh` | 受け子(8000) + 推論(8001) を tmux で起動 |
| `stop.sh`  | tmux セッション停止 |
| `status.sh` | プロセス確認 + ヘルスチェック + 直近の文字起こし表示 |
