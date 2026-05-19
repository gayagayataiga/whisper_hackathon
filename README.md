# whisper

A Whisper speech transcription pipeline running on Jetson AGX Orin.
Receives audio sent from a Raspberry Pi, runs inference with `faster-whisper`, and forwards the results back to the Raspi.

## Architecture

```
Raspi
  │  HTTP POST (wav)
  ▼
Receiver server  Script/interface.py        :8000
  │  HTTP POST (wav)
  ▼
Inference server    Script/whisper_server.py   :8001
  │  HTTP Response (text)
  ▼
Receiver server
  ├─ Append to txt with timestamp
  └─ HTTP POST (text) → Raspi
```

The Raspi-side receiver and mock are `Script/raspi_receiver.py` / `Script/mock_raspi.py`.

## Main Structure

| Path | Role |
| --- | --- |
| `Script/interface.py` | Receiver server (port 8000) |
| `Script/whisper_server.py` | Whisper inference server (port 8001) |
| `Script/raspi_receiver.py` | Receives transcription results on the Raspi side |
| `Script/mock_raspi.py` | Mock client simulating Raspi recording and sending |
| `Script/modules/` | Shared utilities such as `whisper_runner.py` / `vram.py` |
| `Script/tools/` | `convert_mp3_to_wav.py`, `resample_wav.py` |
| `Script/start.sh` / `stop.sh` / `status.sh` | Start, stop, and health-check via tmux session |
| `docs/` | Notes on setup, stress tests, etc. |
| `requirements.txt` | Python dependencies (see top of file for installation steps) |

## Model Configuration

Adjustable via constants at the top of `Script/whisper_server.py`.

- Model: `large-v3`
- Device: `cuda` (Jetson integrated GPU)
- compute_type: `float16` (Tensor Core)
- Language: fixed to `en`
- beam_size: 5 / best_of: 5
- Previous recognition text is passed as `initial_prompt` to maintain context (up to 200 characters)

## Setup

Dependencies must be installed in a specific order; follow the instructions at the top of `requirements.txt`:

1. PyTorch wheel for JetPack
2. Pre-built CTranslate2 (`CTranslate2/python/dist/*.whl`)
3. `pip install -r requirements.txt`
4. `sudo apt install libsndfile1 tmux`

## Starting Up

```bash
cd Script
./start.sh                  # Start inference (8001) + receiver (8000) via tmux
tmux attach -t whisper      # View logs (top pane: inference / bottom pane: receiver)
./status.sh                 # Health check
./stop.sh                   # Stop
```

## Endpoints

### Receiver server (`:8000`)
- `POST /audio` — Receive wav from Raspi and forward to inference server
- `POST /image` — Save image received from Raspi
- `GET  /health`

### Inference server (`:8001`)
- `POST /transcribe` — Accept wav and return transcription result
- `GET  /health` — Model load status and VRAM usage

## Directory Notes

`music/`, `results/`, `transcripts/`, `Script/audio/`, `Script/images/`, `.venv/`, and `CTranslate2/` (vendored) are excluded via `.gitignore`.
