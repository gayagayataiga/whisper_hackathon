#!/usr/bin/env python3
"""
test_whisper_server.py - 推論サーバー(8001) /transcribe の手動疎通確認

multipart/form-data で WAV を POST し、文字起こし結果を表示する。
pytest は使わず、httpx で 1 回叩くだけの手動スクリプト。

使い方:
    python Script/test/test_whisper_server.py path/to/audio.wav
    python Script/test/test_whisper_server.py path/to/audio.wav --url http://localhost:8001/transcribe
"""

import argparse
import sys
from pathlib import Path

import httpx

DEFAULT_URL = "http://localhost:8001/transcribe"


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper 推論サーバーへ WAV を投げて結果を表示")
    parser.add_argument("wav_path", help="送信する WAV ファイル (16kHz / 16bit / モノラル)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"推論エンドポイント (default: {DEFAULT_URL})")
    args = parser.parse_args()

    wav = Path(args.wav_path)
    if not wav.is_file():
        print(f"[Error] file not found: {wav}", file=sys.stderr)
        sys.exit(1)

    with httpx.Client(timeout=60.0) as client:
        with wav.open("rb") as f:
            response = client.post(
                args.url,
                files={"file": (wav.name, f, "audio/wav")},
            )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
