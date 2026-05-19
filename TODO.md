# TODO — Remaining tasks for B / C

At the time of the initial commit, everything up to "A: Fix broken / outdated descriptions" was already applied.
This document summarizes the unfinished items under B (behavior improvements) and C (externalizing configuration).

Completed (see `git log` for details):
- C2 (relax numpy version constraint)
- B1 (add `POST /reset_context` + `previous_text_chars` to `/health`)
- C1 (externalize URLs to environment variables `WHISPER_RASPI_URL` / `WHISPER_INFERENCE_URL`)

Only the 2 items below remain, both with **low priority (not planned for now)**.

## Low priority (not planned for now)

### B2. Migrate `@app.on_event("startup")` to `lifespan`
- **Motivation**: FastAPI treats `on_event` as deprecated. Starlette 0.36+ emits warnings and it will be removed in the future.
- **Reason for deferring**: Production runs stably with a single fixed model, so the practical benefit of migrating is slim. Will address when Starlette actually removes `on_event`.

- **Locations**:
  - `Script/whisper_server.py:116` (`startup_event` loads the model)
  - `Script/interface.py:92` (`startup_event` creates directories + logging)
- **What to do**:
  ```python
  from contextlib import asynccontextmanager

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      # Contents of the old startup_event
      yield
      # Shutdown processing if needed (e.g., model teardown)

  app = FastAPI(lifespan=lifespan, title=..., version=...)
  ```
- **Notes**:
  - Storing `whisper_model` in `app.state.whisper_model` makes it easier to swap during tests, but keeping it as a global variable works fine too. If you don't want to expand the scope, keeping the global is OK.
  - Explicit VRAM release on shutdown matters more for benchmarking (model switching), so treat it as a separate task (see B3 below); do not include it in the lifespan migration itself.

### B3. Explicitly release VRAM when switching models during benchmarking
- **Motivation**: When switching models A → B → C, the previous model's VRAM may not be released before the next one is loaded, potentially causing OOM or performance degradation.
- **Reason for deferring**: Not needed in production since only a single model is used. Will address when OOM or speed degradation is actually encountered during benchmarking.

- **Location**: `Script/benchmark/*` (wherever models are swapped in sequence)
- **What to do**: Insert the following into the model-switching logic.
  ```python
  import gc, torch
  del whisper_model
  gc.collect()
  torch.cuda.empty_cache()  # Effect is limited since it goes through ctranslate2, but harmless
  ```
- **Relationship with B2**: The lifespan shutdown hook only runs when the server exits, so it cannot be used for benchmark model switching. Treat them as separate concerns.
