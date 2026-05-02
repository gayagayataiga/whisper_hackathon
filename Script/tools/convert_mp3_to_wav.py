#!/usr/bin/env python3
"""Convert MP3 files to WAV format.

Usage:
    python3 convert_mp3_to_wav.py input.mp3 [output.wav]
    python3 convert_mp3_to_wav.py input_dir/ [output_dir/]

Requirements (one of the following):
    - pydub + ffmpeg:  pip install pydub  &&  apt install ffmpeg
    - subprocess ffmpeg only (no pip needed):  apt install ffmpeg
"""

import argparse
import subprocess
import sys
from pathlib import Path


def convert_with_ffmpeg(src: Path, dst: Path) -> None:
    """Convert using ffmpeg subprocess (no Python package needed)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr}")


def convert_with_pydub(src: Path, dst: Path) -> None:
    """Convert using pydub (requires pydub + ffmpeg)."""
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(str(src))
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(str(dst), format="wav")


def detect_backend() -> str:
    """Return the best available backend."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        try:
            import pydub  # noqa: F401
            return "pydub"
        except ImportError:
            return "ffmpeg"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    raise RuntimeError(
        "No conversion backend found.\n"
        "Install ffmpeg:  sudo apt install ffmpeg\n"
        "Optionally install pydub:  pip install pydub"
    )


def convert(src: Path, dst: Path, backend: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if backend == "pydub":
        convert_with_pydub(src, dst)
    else:
        convert_with_ffmpeg(src, dst)
    print(f"[OK] {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MP3 to WAV")
    parser.add_argument("input", help="MP3 file or directory containing MP3 files")
    parser.add_argument("output", nargs="?", help="Output WAV file or directory (optional)")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, metavar="HZ",
        help="Output sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--backend", choices=["ffmpeg", "pydub", "auto"], default="auto",
        help="Conversion backend (default: auto-detect)",
    )
    args = parser.parse_args()

    backend = detect_backend() if args.backend == "auto" else args.backend

    src_path = Path(args.input)

    if src_path.is_dir():
        mp3_files = sorted(src_path.glob("**/*.mp3"))
        if not mp3_files:
            print(f"No MP3 files found in {src_path}", file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.output) if args.output else src_path
        for mp3 in mp3_files:
            relative = mp3.relative_to(src_path)
            wav = out_dir / relative.with_suffix(".wav")
            convert(mp3, wav, backend)
        print(f"\nConverted {len(mp3_files)} file(s).")
    elif src_path.is_file():
        if args.output:
            dst_path = Path(args.output)
        else:
            dst_path = src_path.with_suffix(".wav")
        convert(src_path, dst_path, backend)
    else:
        print(f"Input not found: {src_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
