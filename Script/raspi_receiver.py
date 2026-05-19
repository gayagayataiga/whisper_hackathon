#!/usr/bin/env python3
"""
raspi_receiver.py - Raspi receiver server

Receives transcription results from Jetson and displays/saves them.

Usage:
  python raspi_receiver.py
  python raspi_receiver.py --port 9000
"""

import argparse
import datetime
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ============================================================
# Configuration parameters
# ============================================================

DEFAULT_PORT    = 9000
TRANSCRIPT_FILE = Path("./received_transcripts.txt")

# ============================================================
# Logger configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# App initialization
# ============================================================

app = FastAPI(title="Raspi Receiver", version="1.0.0")

class CommandPayload(BaseModel):
    text: str

# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/command")
async def receive_command(payload: CommandPayload) -> JSONResponse:
    text = payload.text.strip()
    logger.info(f"Received: {text}")

    # Save with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp}\t{text}\n")

    return JSONResponse({"status": "ok", "received": text})

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspi receiver server")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help=f"Listening port (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    import uvicorn
    logger.info(f"Raspi receiver starting on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
