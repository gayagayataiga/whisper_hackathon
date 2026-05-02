#!/usr/bin/env python3
"""Resample WAV files to a target sample rate.

Usage:
    python3 resample_wav.py input.wav [output.wav] [--rate 16000]
    python3 resample_wav.py input_dir/ [output_dir/] [--rate 16000]
"""

import argparse
import audioop
import struct
import sys
import wave
from pathlib import Path


def read_wav(path: Path) -> tuple[bytes, int, int, int]:
    """Return (raw_pcm, channels, sampwidth, framerate)."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        data = wf.readframes(wf.getnframes())
    return data, channels, sampwidth, framerate


def write_wav(path: Path, data: bytes, channels: int, sampwidth: int, framerate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(data)


def resample_audioop(data: bytes, sampwidth: int, src_rate: int, dst_rate: int) -> bytes:
    """Resample using stdlib audioop (no extra dependencies)."""
    resampled, _ = audioop.ratecv(data, sampwidth, 1, src_rate, dst_rate, None)
    return resampled


def mix_to_mono(data: bytes, channels: int, sampwidth: int) -> bytes:
    """Mix multi-channel PCM down to mono."""
    if channels == 1:
        return data
    # interleaved samples -> average across channels
    fmt = {1: "b", 2: "h", 4: "i"}[sampwidth]
    samples = struct.unpack(f"{len(data) // sampwidth}{fmt}", data)
    mono = [
        sum(samples[i : i + channels]) // channels
        for i in range(0, len(samples), channels)
    ]
    return struct.pack(f"{len(mono)}{fmt}", *mono)


def resample(
    src: Path,
    dst: Path,
    target_rate: int,
    mono: bool,
) -> None:
    data, channels, sampwidth, src_rate = read_wav(src)

    if src_rate == target_rate and (channels == 1 or not mono):
        print(f"[SKIP] {src} (already {src_rate} Hz, channels={channels})")
        if src != dst:
            write_wav(dst, data, channels, sampwidth, src_rate)
        return

    if mono and channels > 1:
        data = mix_to_mono(data, channels, sampwidth)
        channels = 1

    if src_rate != target_rate:
        # audioop.ratecv works on single-channel data; process each channel
        if channels == 1:
            data = resample_audioop(data, sampwidth, src_rate, target_rate)
        else:
            # split channels, resample each, interleave back
            fmt = {1: "b", 2: "h", 4: "i"}[sampwidth]
            n_frames = len(data) // (sampwidth * channels)
            samples = struct.unpack(f"{n_frames * channels}{fmt}", data)
            ch_data = []
            for ch in range(channels):
                ch_samples = samples[ch::channels]
                ch_bytes = struct.pack(f"{len(ch_samples)}{fmt}", *ch_samples)
                ch_bytes = resample_audioop(ch_bytes, sampwidth, src_rate, target_rate)
                ch_data.append(ch_bytes)
            # interleave
            n_out = len(ch_data[0]) // sampwidth
            out_samples = []
            ch_unpacked = [
                struct.unpack(f"{n_out}{fmt}", c) for c in ch_data
            ]
            for i in range(n_out):
                for ch in range(channels):
                    out_samples.append(ch_unpacked[ch][i])
            data = struct.pack(f"{len(out_samples)}{fmt}", *out_samples)

    write_wav(dst, data, channels, sampwidth, target_rate)
    note = f"{src_rate} Hz -> {target_rate} Hz"
    if mono:
        note += ", mono"
    print(f"[OK] {src} -> {dst}  ({note})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample WAV files to a target sample rate")
    parser.add_argument("input", help="WAV file or directory")
    parser.add_argument("output", nargs="?", help="Output file or directory (default: overwrite)")
    parser.add_argument(
        "--rate", type=int, default=16000, metavar="HZ",
        help="Target sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--mono", action="store_true",
        help="Also convert to mono (mix down channels)",
    )
    args = parser.parse_args()

    src_path = Path(args.input)

    if src_path.is_dir():
        wav_files = sorted(src_path.glob("**/*.wav"))
        if not wav_files:
            print(f"No WAV files found in {src_path}", file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.output) if args.output else src_path
        for wav in wav_files:
            relative = wav.relative_to(src_path)
            dst = out_dir / relative
            resample(wav, dst, args.rate, args.mono)
        print(f"\nProcessed {len(wav_files)} file(s).")
    elif src_path.is_file():
        dst_path = Path(args.output) if args.output else src_path
        resample(src_path, dst_path, args.rate, args.mono)
    else:
        print(f"Input not found: {src_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
