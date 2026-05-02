#!/usr/bin/env python3
"""
mock_raspi.py - 疑似 Raspi 送信スクリプト

実際の Raspi（録音 + VAD）の代わりに、
WAV ファイルを読み込んで受け子サーバーへ HTTP POST する。

使い方:
  # テストトーンを生成して送信（ファイル不要）
  python mock_raspi.py --generate

  # WAV ファイルを送信
  python mock_raspi.py audio.wav

  # 受け子サーバーの IP を指定
  python mock_raspi.py audio.wav --host 192.168.1.10

  # VAD をシミュレート（無音1.5秒で分割して順番に送信）
  python mock_raspi.py audio.wav --split

  # ループ送信（同じファイルを繰り返し送る）
  python mock_raspi.py audio.wav --loop

  # 利用可能なテストファイルを一覧表示
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
# 設定パラメータ
# ============================================================

DEFAULT_HOST     = "localhost"
DEFAULT_PORT     = 8000
RECEIVER_TIMEOUT = 60.0   # 推論待ちを含むため長めに設定

SAMPLE_RATE  = 16000  # 期待するサンプルレート
CHANNELS     = 1      # モノラル
SAMPLE_WIDTH = 2      # 16bit

# VAD シミュレート用
VAD_SILENCE_THRESHOLD = 0.01   # Float32 正規化済み振幅
VAD_SILENCE_SEC       = 1.5    # 無音継続でファイルを切る秒数
VAD_MIN_SPEECH_SEC    = 0.3    # これ未満のセグメントは捨てる

SCRIPT_DIR = Path(__file__).parent
MUSIC_DIR  = SCRIPT_DIR.parent / "music"

# ============================================================
# ヘルパー関数
# ============================================================

def list_test_files() -> list[Path]:
    """music ディレクトリ内の WAV ファイルを列挙する。"""
    if not MUSIC_DIR.exists():
        return []
    return sorted(MUSIC_DIR.rglob("*.wav"))


def generate_test_audio(duration: float = 2.0) -> np.ndarray:
    """
    パイプライン疎通確認用のテストトーンを生成する。
    440Hz + 880Hz のサイン波合成音（音声認識結果は空になる場合がある）。
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    return audio


def load_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """
    音声ファイルを soundfile で読み込み 16kHz Float32 モノラル配列を返す。
    フォーマットが異なる場合は自動変換する。
    """
    path = Path(wav_path)
    if not path.exists():
        print(f"[Error] File not found: {wav_path}")
        files = list_test_files()
        if files:
            print(f"\n  利用可能なテストファイル（--list で詳細確認）:")
            for f in files[:5]:
                print(f"    {f.relative_to(SCRIPT_DIR.parent)}")
        sys.exit(1)

    try:
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception as e:
        print(f"[Error] ファイルを読み込めません: {e}")
        sys.exit(1)

    # ステレオ → モノラル
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
        print(f"[Info] ステレオ → モノラルに変換しました")

    # リサンプリング（線形補間）
    if sr != SAMPLE_RATE:
        print(f"[Info] {sr}Hz → {SAMPLE_RATE}Hz にリサンプリングします")
        target_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(
            np.linspace(0, len(audio), target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        sr = SAMPLE_RATE

    return audio, sr


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Float32 numpy 配列を 16bit WAV バイト列に変換する。"""
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
    振幅ベースの簡易 VAD で音声を発話単位に分割する。
    本番 Raspi の webrtcvad 挙動をシミュレートする。
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
    """WAV バイト列を受け子サーバーへ HTTP POST する。"""
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
                    print(f"{prefix}  Raspi送信: 成功")
            else:
                print(f"{prefix}✓ ({duration:.2f}s) → (空: 無音または雑音)")
            return result

    except httpx.TimeoutException:
        print(f"{prefix}✗ タイムアウト ({RECEIVER_TIMEOUT}s)")
        return None
    except httpx.ConnectError:
        print(f"{prefix}✗ 接続失敗: {url} に到達できません")
        print(f"{prefix}  interface.py (port {port}) が起動しているか確認してください")
        return None
    except httpx.HTTPStatusError as e:
        print(f"{prefix}✗ HTTPエラー: {e.response.status_code} {e.response.text}")
        return None

# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="疑似 Raspi: WAV ファイルを受け子サーバーへ送信する"
    )
    parser.add_argument(
        "wav_file", nargs="?",
        help="送信する WAV ファイルのパス（省略時は --generate が必要）"
    )
    parser.add_argument("--host",     default=DEFAULT_HOST, help=f"受け子サーバーのホスト (デフォルト: {DEFAULT_HOST})")
    parser.add_argument("--port",     default=DEFAULT_PORT, type=int, help=f"受け子サーバーのポート (デフォルト: {DEFAULT_PORT})")
    parser.add_argument("--split",    action="store_true",  help="VAD をシミュレートして無音1.5秒で分割送信")
    parser.add_argument("--loop",     action="store_true",  help="ファイルを繰り返し送信し続ける (Ctrl+C で停止)")
    parser.add_argument("--interval", default=0.5, type=float, help="ループ時のセグメント間隔 [秒] (デフォルト: 0.5)")
    parser.add_argument("--generate", action="store_true",  help="テストトーン (440Hz, 2秒) を生成して送信")
    parser.add_argument("--list",     action="store_true",  help=f"{MUSIC_DIR.name}/ 内の利用可能な WAV ファイルを一覧表示")
    args = parser.parse_args()

    # ── --list ───────────────────────────────────────────────
    if args.list:
        files = list_test_files()
        if not files:
            print(f"[Info] {MUSIC_DIR} に WAV ファイルが見つかりません")
        else:
            print(f"利用可能な WAV ファイル ({MUSIC_DIR}):")
            for f in files:
                info = sf.info(str(f))
                print(f"  {str(f.relative_to(MUSIC_DIR)):<40} {info.duration:.2f}s  {info.samplerate}Hz  {info.channels}ch")
        return

    # ── 音声データ準備 ────────────────────────────────────────
    if args.generate:
        audio       = generate_test_audio(duration=2.0)
        sample_rate = SAMPLE_RATE
        source_name = "generated (440Hz tone, 2s)"
    elif args.wav_file:
        audio, sample_rate = load_wav(args.wav_file)
        source_name = args.wav_file
    else:
        parser.print_help()
        print("\n[Error] WAV ファイルパスか --generate を指定してください")
        files = list_test_files()
        if files:
            print(f"\n  利用可能なテストファイル:")
            for f in files[:5]:
                print(f"    python mock_raspi.py {f.relative_to(SCRIPT_DIR.parent)}")
        sys.exit(1)

    duration = len(audio) / sample_rate

    print(f"{'='*55}")
    print(f"  Mock Raspi Sender")
    print(f"  ソース    : {source_name} ({duration:.2f}s)")
    print(f"  送信先    : http://{args.host}:{args.port}/audio")
    print(f"  VAD分割   : {'有効' if args.split else '無効（全体を送信）'}")
    print(f"  ループ    : {'有効' if args.loop else '無効'}")
    print(f"{'='*55}\n")

    # ── セグメント準備 ────────────────────────────────────────
    if args.split:
        segments = split_by_vad(audio, sample_rate)
        if not segments:
            print("[Error] VAD で発話セグメントが検出されませんでした。")
            print("  → --split なしで試すか、音声ファイルを確認してください。")
            sys.exit(1)
        print(f"[VAD] {len(segments)} セグメントに分割しました:")
        for i, seg in enumerate(segments):
            print(f"  [{i+1:02d}] {len(seg)/sample_rate:.2f}s")
        print()
        wav_list = [
            (f"seg{i+1:02d}", numpy_to_wav_bytes(seg, sample_rate))
            for i, seg in enumerate(segments)
        ]
    else:
        wav_list = [("full", numpy_to_wav_bytes(audio, sample_rate))]

    # ── 送信ループ ────────────────────────────────────────────
    loop_count = 0
    try:
        while True:
            loop_count += 1
            if args.loop:
                print(f"--- ループ {loop_count} 回目 ---")

            for label, wav_bytes in wav_list:
                send_wav(wav_bytes, args.host, args.port, label)
                if len(wav_list) > 1:
                    time.sleep(args.interval)

            if not args.loop:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n[Interrupted] {loop_count} ループ送信して停止しました。")

    print("\n完了。")


if __name__ == "__main__":
    main()
