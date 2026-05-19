#!/usr/bin/env python3
"""
mock_raspi.py - Mock Raspi sender script

Instead of an actual Raspi (recording + VAD),
reads a WAV file and HTTP POSTs it to the receiver server.

Usage:
  # Generate a test tone and send (no file needed)
  python mock_raspi.py --generate

  # Send a WAV file
  python mock_raspi.py audio.wav

  # Specify receiver server IP
  python mock_raspi.py audio.wav --host 192.168.1.10

  # Simulate VAD (split on 1.5s silence and send in order)
  python mock_raspi.py audio.wav --split

  # Loop sending (repeatedly send the same file)
  python mock_raspi.py audio.wav --loop

  # List available test files
  python mock_raspi.py --list
"""

import argparse
import sys
import time
import io
import wave
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

# ============================================================
# Configuration parameters
# ============================================================

DEFAULT_HOST     = "localhost"
DEFAULT_PORT     = 8000
RECEIVER_TIMEOUT = 60.0   # Set long to account for inference wait time

SAMPLE_RATE  = 16000  # Expected sample rate
CHANNELS     = 1      # Mono
SAMPLE_WIDTH = 2      # 16bit

# For VAD simulation
VAD_SILENCE_THRESHOLD = 0.01   # Float32 normalized amplitude
VAD_SILENCE_SEC       = 1.5    # Seconds of silence to cut the file
VAD_MIN_SPEECH_SEC    = 0.3    # Discard segments shorter than this

SCRIPT_DIR = Path(__file__).parent
MUSIC_DIR  = SCRIPT_DIR.parent / "music"

# ============================================================
# Helper functions
# ============================================================

def list_test_files() -> list[Path]:
    """List WAV files in the music directory."""
    if not MUSIC_DIR.exists():
        return []
    return sorted(MUSIC_DIR.rglob("*.wav"))


def generate_test_audio(duration: float = 2.0) -> np.ndarray:
    """
    Generate a test tone for pipeline connectivity checks.
    Synthesized sine wave of 440Hz + 880Hz (speech recognition result may be empty).
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    return audio


def load_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file with soundfile and return a 16kHz Float32 mono array.
    Automatically converts if the format differs.
    """
    path = Path(wav_path)
    if not path.exists():
        print(f"[Error] File not found: {wav_path}")
        files = list_test_files()
        if files:
            print(f"\n  Available test files (use --list for details):")
            for f in files[:5]:
                print(f"    {f.relative_to(SCRIPT_DIR.parent)}")
        sys.exit(1)

    try:
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception as e:
        print(f"[Error] Cannot read file: {e}")
        sys.exit(1)

    # Stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
        print(f"[Info] Converted stereo to mono")

    # Resampling (linear interpolation)
    if sr != SAMPLE_RATE:
        print(f"[Info] Resampling {sr}Hz → {SAMPLE_RATE}Hz")
        target_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(
            np.linspace(0, len(audio), target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        sr = SAMPLE_RATE

    return audio, sr


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a Float32 numpy array to 16bit WAV bytes."""
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def split_by_vad(audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
    """
    Split audio into utterance units using a simple amplitude-based VAD.
    Simulates the webrtcvad behavior of the production Raspi.
    """
    silence_samples    = int(VAD_SILENCE_SEC * sample_rate)
    min_speech_samples = int(VAD_MIN_SPEECH_SEC * sample_rate)
    frame_size         = sample_rate // 100  # 10ms

    segments    = []
    speech_buf  = []
    silence_cnt = 0
    in_speech   = False

    for i in range(0, len(audio), frame_size):
        frame = audio[i:i + frame_size]
        if len(frame) == 0:
            break
        is_speech = np.abs(frame).mean() > VAD_SILENCE_THRESHOLD

        if is_speech:
            speech_buf.append(frame)
            silence_cnt = 0
            in_speech   = True
        elif in_speech:
            silence_cnt += frame_size
            speech_buf.append(frame)
            if silence_cnt >= silence_samples:
                segment = np.concatenate(speech_buf)
                if len(segment) >= min_speech_samples:
                    segments.append(segment)
                speech_buf  = []
                silence_cnt = 0
                in_speech   = False

    if speech_buf:
        segment = np.concatenate(speech_buf)
        if len(segment) >= min_speech_samples:
            segments.append(segment)

    return segments


def send_wav(wav_bytes: bytes, host: str, port: int, label: str = "") -> dict | None:
    """HTTP POST WAV bytes to the receiver server."""
    url    = f"http://{host}:{port}/audio"
    prefix = f"[{label}] " if label else ""

    try:
        with httpx.Client(timeout=RECEIVER_TIMEOUT) as client:
            response = client.post(
                url,
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            )
            response.raise_for_status()
            result   = response.json()
            text       = result.get("text", "")
            duration   = result.get("duration_s", 0.0)
            raspi_sent = result.get("raspi_sent", False)

            if text:
                print(f"{prefix}✓ ({duration:.2f}s) → {text}")
                if raspi_sent:
                    print(f"{prefix}  Raspi send: success")
            else:
                print(f"{prefix}✓ ({duration:.2f}s) → (empty: silence or noise)")
            return result

    except httpx.TimeoutException:
        print(f"{prefix}✗ Timeout ({RECEIVER_TIMEOUT}s)")
        return None
    except httpx.ConnectError:
        print(f"{prefix}✗ Connection failed: cannot reach {url}")
        print(f"{prefix}  Check that interface.py (port {port}) is running")
        return None
    except httpx.HTTPStatusError as e:
        print(f"{prefix}✗ HTTP error: {e.response.status_code} {e.response.text}")
        return None

# ============================================================
# Main processing
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mock Raspi: send a WAV file to the receiver server"
    )
    parser.add_argument(
        "wav_file", nargs="?",
        help="Path to the WAV file to send (--generate required if omitted)"
    )
    parser.add_argument("--host",     default=DEFAULT_HOST, help=f"Receiver server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port",     default=DEFAULT_PORT, type=int, help=f"Receiver server port (default: {DEFAULT_PORT})")
    parser.add_argument("--split",    action="store_true",  help="Simulate VAD and split on 1.5s silence")
    parser.add_argument("--loop",     action="store_true",  help="Keep sending the file repeatedly (Ctrl+C to stop)")
    parser.add_argument("--interval", default=0.5, type=float, help="Interval between segments in loop mode [seconds] (default: 0.5)")
    parser.add_argument("--generate", action="store_true",  help="Generate a test tone (440Hz, 2s) and send it")
    parser.add_argument("--list",     action="store_true",  help=f"List available WAV files in {MUSIC_DIR.name}/")
    args = parser.parse_args()

    # ── --list ───────────────────────────────────────────────
    if args.list:
        files = list_test_files()
        if not files:
            print(f"[Info] No WAV files found in {MUSIC_DIR}")
        else:
            print(f"Available WAV files ({MUSIC_DIR}):")
            for f in files:
                info = sf.info(str(f))
                print(f"  {str(f.relative_to(MUSIC_DIR)):<40} {info.duration:.2f}s  {info.samplerate}Hz  {info.channels}ch")
        return

    # ── Audio data preparation ────────────────────────────────
    if args.generate:
        audio       = generate_test_audio(duration=2.0)
        sample_rate = SAMPLE_RATE
        source_name = "generated (440Hz tone, 2s)"
    elif args.wav_file:
        audio, sample_rate = load_wav(args.wav_file)
        source_name = args.wav_file
    else:
        parser.print_help()
        print("\n[Error] Specify a WAV file path or --generate")
        files = list_test_files()
        if files:
            print(f"\n  Available test files:")
            for f in files[:5]:
                print(f"    python mock_raspi.py {f.relative_to(SCRIPT_DIR.parent)}")
        sys.exit(1)

    duration = len(audio) / sample_rate

    print(f"{'='*55}")
    print(f"  Mock Raspi Sender")
    print(f"  Source    : {source_name} ({duration:.2f}s)")
    print(f"  Target    : http://{args.host}:{args.port}/audio")
    print(f"  VAD split : {'enabled' if args.split else 'disabled (send whole)'}")
    print(f"  Loop      : {'enabled' if args.loop else 'disabled'}")
    print(f"{'='*55}\n")

    # ── Segment preparation ───────────────────────────────────
    if args.split:
        segments = split_by_vad(audio, sample_rate)
        if not segments:
            print("[Error] No speech segments detected by VAD.")
            print("  → Try without --split or check the audio file.")
            sys.exit(1)
        print(f"[VAD] Split into {len(segments)} segments:")
        for i, seg in enumerate(segments):
            print(f"  [{i+1:02d}] {len(seg)/sample_rate:.2f}s")
        print()
        wav_list = [
            (f"seg{i+1:02d}", numpy_to_wav_bytes(seg, sample_rate))
            for i, seg in enumerate(segments)
        ]
    else:
        wav_list = [("full", numpy_to_wav_bytes(audio, sample_rate))]

    # ── Send loop ─────────────────────────────────────────────
    loop_count = 0
    try:
        while True:
            loop_count += 1
            if args.loop:
                print(f"--- Loop iteration {loop_count} ---")

            for label, wav_bytes in wav_list:
                send_wav(wav_bytes, args.host, args.port, label)
                if len(wav_list) > 1:
                    time.sleep(args.interval)

            if not args.loop:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n[Interrupted] Stopped after {loop_count} loop(s).")

    print("\nDone.")


if __name__ == "__main__":
    main()
