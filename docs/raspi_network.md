# Raspberry Pi Network Configuration Notes

The URL used by the receiver server (`Script/interface.py`) to send transcription results to the Raspi
is specified via the environment variable `WHISPER_RASPI_URL`. No edits to `Script/interface.py` are needed.

## IP Address List

| Connection type | IP Address      | Port | Notes                                |
| --------------- | --------------- | ---- | ------------------------------------ |
| Wireless        | `10.27.72.53`   | 9000 | Currently in use (2026-05-02)        |
| Wired           | `192.168.10.2`  | 9000 | For use when switching to wired      |

The Jetson side (wired) has `192.168.10.1/24` fixed on `eno1`.
Saved in NetworkManager profile `Wired connection 1`
(bound by MAC `3C:6D:66:B1:AB:FA`, autoconnect-priority=100).

## How to Start

Pass `WHISPER_RASPI_URL` as an environment variable when launching `Script/start.sh`:

```bash
# Wireless connection
WHISPER_RASPI_URL=http://10.27.72.53:9000/command ./start.sh

# Wired connection
WHISPER_RASPI_URL=http://192.168.10.2:9000/command ./start.sh
```

`WHISPER_RASPI_URL` is required. If not set, `start.sh` will exit with an error
(no default value is provided to prevent accidental IP mix-ups).

If specifying it manually every time is tedious, export it in your shell rc:

```bash
# Add to ~/.bashrc or similar
export WHISPER_RASPI_URL=http://10.27.72.53:9000/command
```

## Changing the Inference Server URL (optional)

Normally fixed to local (`http://localhost:8001`). Only specify `WHISPER_INFERENCE_URL`
when using a remote inference server (base URL only, no path):

```bash
WHISPER_INFERENCE_URL=http://other-host:8001 \
WHISPER_RASPI_URL=http://10.27.72.53:9000/command \
./start.sh
```

## Post-launch Verification

`./status.sh` can display the current `WHISPER_RASPI_URL`
(reads the value recorded at startup in `Script/.env.runtime`).
