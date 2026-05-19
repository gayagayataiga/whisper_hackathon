#!/usr/bin/env python3
"""
test_whisper_server.py - Manual connectivity check for inference server (8001) /transcribe

POST a WAV as multipart/form-data and display the transcription result.
No pytest — just a manual script that hits the endpoint once with httpx.

Usage:
    python Script/test/test_whisper_server.py path/to/audio.wav
    python Script/test/test_whisper_server.py path/to/audio.wav --url http://localhost:8001/transcribe
"""

import argparse
import sys
from pathlib import Path

import httpx

DEFAULT_URL = "http://localhost:8001/transcribe"


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a WAV to the Whisper inference server and display the result")
    parser.add_argument("wav_path", help="WAV file to send (16kHz / 16bit / mono)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Inference endpoint (default: {DEFAULT_URL})")
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
