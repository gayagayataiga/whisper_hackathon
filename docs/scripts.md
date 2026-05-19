# Script List

## Main Servers

| Script | Port | Role |
|-----------|--------|------|
| `Script/whisper_server.py` | 8001 | Whisper inference server |
| `Script/interface.py` | 8000 | Receiver server (relay) |

**`whisper_server.py`**
- Accepts WAV bytes and transcribes them with Faster-Whisper (large-v3, float16, CUDA)
- `POST /transcribe` : WAV → `{"text", "language", "duration_s", "inference_s", "rtf"}`
- `GET  /health` : Health check + model load status + VRAM usage

**`interface.py`**
- Receives WAV from Raspi, forwards to inference server (8001), saves the result, and sends it back to Raspi
- `POST /audio` : WAV → inference → append to `transcripts/transcript.txt` / `transcripts/transcript.jsonl` → send to Raspi (`RASPI_URL`)
- `POST /image` : Save image
- `GET  /health` : Health check

```
Raspi → POST /audio (8000) → POST /transcribe (8001) → text → transcripts/{txt,jsonl}
                                                              → Raspi (9000)
```

---

## Testing / Sending Tools

**`mock_raspi.py`** — Mock Raspi over HTTP

```bash
python mock_raspi.py --generate                        # Generate test tone and send
python mock_raspi.py music/BASIC5000_0001.wav          # Send WAV file
python mock_raspi.py music/news_audio/xxx.wav          # 48kHz files are automatically resampled
python mock_raspi.py audio.wav --split                 # Split and send simulating VAD
python mock_raspi.py audio.wav --loop --interval 1.0   # Send in a loop
python mock_raspi.py --list                            # List available WAV files
```

**`raspi_receiver.py`** — Receiver server on the Raspi side
- Appends transcription results sent from the receiver server to `received_transcripts.txt`.
- `python raspi_receiver.py --port 9000`

**`Script/test/test_whisper_server.py`** — Manual connectivity test script for the inference server
- `python Script/test/test_whisper_server.py path/to/audio.wav`

---

## Modules

| File | Contents |
|---------|------|
| `modules/whisper_runner.py` | Whisper model loading and `transcribe_audio()` (with RTF measurement) |
| `modules/vram.py` | VRAM usage retrieval via direct CUDA runtime calls (workaround for Jetson's lack of NVML support) |

---

## Benchmarks

| File | Contents |
|---------|------|
| `benchmark/stress_test.py` | All model sizes × 10 inferences; saves VRAM and RTF as JSON |
| `benchmark/accuracy_report.py` / `accuracy_report_en.py` | Aggregates CER/WER using large-v3 as ground truth |
| `benchmark/combined_report.py` / `combined_report_en.py` | Combined report of stress test results |
| `benchmark/news_benchmark.py` | Benchmark using news audio |
| `benchmark/whisper_benchmark.py` | LibriSpeech WER evaluation |
| `benchmark/turbo_run.py` / `turbo_vram_profile.py` | Scripts specific to the turbo model |
| `benchmark/transcript_accuracy.py` | Transcription accuracy aggregation utility |
| `benchmark/wer_utils.py` | Shared WER calculation functions |

---

## Tools

| File | Contents |
|---------|------|
| `tools/convert_mp3_to_wav.py` | MP3 → WAV conversion (uses ffmpeg) |
| `tools/resample_wav.py` | WAV sample rate conversion |

---

## Startup Control

| File | Role |
|---------|------|
| `start.sh` | Start receiver (8000) + inference (8001) via tmux |
| `stop.sh`  | Stop the tmux session |
| `status.sh` | Process check + health check + display recent transcriptions |
