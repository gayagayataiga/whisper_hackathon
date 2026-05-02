import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000


@dataclass
class TranscribeResult:
    text: str
    inference_time_s: float
    audio_duration_s: float
    rtf: float
    language: str
    language_prob: float


def load_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type, num_workers=1)


def transcribe_audio(
    model: WhisperModel,
    audio: np.ndarray,
    language: str,
    initial_prompt: str | None = None,
    beam_size: int = 5,
) -> TranscribeResult:
    audio_duration_s = len(audio) / SAMPLE_RATE
    t_start = time.monotonic()
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
    )
    text = "".join(seg.text for seg in segments).strip()
    elapsed = time.monotonic() - t_start
    rtf = elapsed / audio_duration_s if audio_duration_s > 0 else float("inf")
    return TranscribeResult(
        text=text,
        inference_time_s=elapsed,
        audio_duration_s=audio_duration_s,
        rtf=rtf,
        language=info.language,
        language_prob=info.language_probability,
    )
