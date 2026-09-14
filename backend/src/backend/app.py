"""Realtime Nemotron ASR with coalesced live cleanup of the recording so far."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import numpy as np
import traceback
import anyio
from functools import partial
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.asr import load_dotenv_files, load_model_in_background, runtime_status
from backend.cleanup import cleanup_span, CleanupValidationError

logger = logging.getLogger("uvicorn.error")

load_dotenv_files()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model_in_background()
    yield


app = FastAPI(title="Wisper realtime", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status():
    return runtime_status()


# NeMo's decoding configuration is mutable and shared by the loaded model.
# Admit one live session at a time instead of mixing languages/caches across clients.
model_slot = asyncio.Lock()


@app.websocket("/ws/transcribe")
async def transcribe_socket(ws: WebSocket) -> None:
    await ws.accept()
    session = None
    owns_slot = False
    drain_task = None
    sentence_id = 0
    recording_ids: list[int] = []
    cleanup_enabled = False
    cleanup_task = None
    pending_polish = None
    polish_revision = 0
    last_requested_raw = ""
    trace_id = uuid.uuid4().hex[:8]
    started_at = 0.0
    audio_received = 0
    inference_steps = 0
    inference_seconds = 0.0
    first_text_seen = False
    first_signal_seen = False

    async def send(payload: dict) -> None:
        await ws.send_json(payload)

    async def commit(force=False):
        nonlocal sentence_id
        raw = session.commit(force=force)
        if not raw:
            return
        sentence_id += 1
        recording_ids.append(sentence_id)
        await send({"type": "commit", "id": sentence_id, "raw": raw,
                    "cleanup_status": "deferred" if cleanup_enabled else "skipped"})
        if not cleanup_enabled:
            await send({"type": "cleaned", "id": sentence_id, "raw": raw,
                        "cleaned": raw, "cleanup_status": "skipped"})

    async def polish_worker():
        nonlocal pending_polish
        while pending_polish is not None:
            revision, raw, ids = pending_polish
            pending_polish = None
            await send({"type": "polishing", "ids": ids, "raw": raw, "revision": revision})
            cleanup_started = time.monotonic()
            failure_reason = ""
            try:
                cleaned = await asyncio.to_thread(cleanup_span, raw)
                result = "applied"
            except Exception as exc:
                cleaned = raw
                result = "failed"
                if isinstance(exc, CleanupValidationError):
                    failure_reason = str(exc)  # Our validator messages contain no transcript.
                elif getattr(exc, "status_code", None):
                    failure_reason = f"Provider HTTP {exc.status_code}"
                else:
                    failure_reason = type(exc).__name__
                logger.warning("cleanup_failed session=%s revision=%s reason=%s elapsed_ms=%.0f input_chars=%s",
                               trace_id, revision, failure_reason, (time.monotonic()-cleanup_started)*1000, len(raw))
            # Newer pause snapshots supersede unfinished edits. Never append or
            # show an old response after a newer correction has been requested.
            if revision != polish_revision:
                continue
            await send({"type": "polished", "ids": ids, "raw": raw,
                        "cleaned": cleaned, "cleanup_status": result, "revision": revision})
            if result == "failed":
                await send({"type": "warning", "message": f"Polishing failed ({failure_reason}). Your original words have been kept."})

    def request_polish(raw: str):
        nonlocal cleanup_task, pending_polish, polish_revision, last_requested_raw
        if not raw.strip() or not recording_ids or raw == last_requested_raw:
            return
        last_requested_raw = raw
        polish_revision += 1
        # One provider call at a time, plus only the newest waiting snapshot.
        # Inputs are canonical raw ASR text, never prior generated text.
        pending_polish = (polish_revision, raw, list(recording_ids))
        if cleanup_task is None or cleanup_task.done():
            cleanup_task = asyncio.create_task(polish_worker())

    async def drain_audio():
        nonlocal inference_steps, inference_seconds, first_text_seen
        try:
            while session is not None and session.has_chunk():
                began = time.monotonic()
                snap = await anyio.to_thread.run_sync(session.consume_chunks, 1)
                step_seconds = time.monotonic() - began
                inference_seconds += step_seconds
                inference_steps += 1
                if inference_steps == 1 or (snap["full"] and not first_text_seen):
                    stage = "first_text" if snap["full"] else "first_inference"
                    logger.info("asr_timing session=%s stage=%s elapsed_ms=%.0f step_ms=%.0f received_audio_ms=%.0f steps=%s inference_ms=%.0f",
                                trace_id, stage, (time.monotonic()-started_at)*1000, step_seconds*1000,
                                audio_received/32, inference_steps, inference_seconds*1000)
                if snap["full"]:
                    first_text_seen = True
                await send({"type": "partial", **snap})
        except Exception:
            await send({"type": "error", "message": "Audio processing failed. Please start a new recording."})
            await ws.close(code=1011)
            raise

    try:
        await send({"type": "ready", **runtime_status()})
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                    if not isinstance(payload, dict):
                        raise ValueError()
                except (ValueError, TypeError):
                    await send({"type": "warning", "message": "Invalid control message."})
                    continue
                kind = payload.get("type")
                if kind == "start":
                    if session is not None:
                        await send({"type": "warning", "message": "Recording is already active."})
                        continue
                    if runtime_status()["status"] != "ready":
                        await send({"type": "error", "message": "The speech model is not ready yet."})
                        continue
                    if model_slot.locked():
                        await send({"type": "error", "message": "Another recording is using the speech model. Please retry shortly."})
                        continue
                    await model_slot.acquire()
                    owns_slot = True
                    from backend.asr import LiveSession, NEMOTRON_DEFAULT_CHUNK
                    try:
                        session = await anyio.to_thread.run_sync(partial(
                            LiveSession, lang=payload.get("language") or "auto",
                            profile=payload.get("profile") or NEMOTRON_DEFAULT_CHUNK,
                        ))
                    except Exception:
                        model_slot.release()
                        owns_slot = False
                        await send({"type": "error", "message": "Could not initialize the speech session. Check the server logs and retry."})
                        traceback.print_exc()
                        continue
                    cleanup_enabled = bool(payload.get("cleanup", True))
                    recording_ids = []
                    pending_polish = None
                    last_requested_raw = ""
                    polish_revision = 0
                    started_at = time.monotonic()
                    audio_received = 0
                    inference_steps = 0
                    inference_seconds = 0.0
                    first_text_seen = False
                    first_signal_seen = False
                    trace_id = uuid.uuid4().hex[:8]
                    logger.info("asr_started session=%s language=%s profile=%s", trace_id, payload.get("language", "auto"), payload.get("profile", "Balanced"))
                    await send({"type": "started"})
                elif kind in ("commit", "end") and session is not None:
                    # Never flush or commit concurrently with an inference step.
                    if drain_task:
                        await drain_task
                        drain_task = None
                    if kind == "commit":
                        await commit(force=True)
                        if cleanup_enabled:
                            request_polish(session.snapshot()["full"])
                    else:
                        snap = await anyio.to_thread.run_sync(session.flush)
                        await send({"type": "partial", **snap})
                        await commit(force=True)
                        session = None
                        model_slot.release()
                        owns_slot = False
                        if cleanup_enabled:
                            request_polish(snap["full"])
                            if cleanup_task:
                                await cleanup_task
                                cleanup_task = None
                        await send({"type": "ended"})
                continue
            data = message.get("bytes")
            if data and session is not None:
                if not audio_received:
                    logger.info("asr_timing session=%s stage=first_audio elapsed_ms=%.0f", trace_id, (time.monotonic()-started_at)*1000)
                audio_received += len(data)
                if not first_signal_seen and len(data) >= 2:
                    pcm = np.frombuffer(data[:len(data) - len(data) % 2], dtype="<i2").astype(np.float32) / 32768
                    rms = float(np.sqrt(np.mean(pcm * pcm)))
                    if rms > 0.018:
                        first_signal_seen = True
                        logger.info("asr_timing session=%s stage=first_signal elapsed_ms=%.0f rms=%.4f",
                                    trace_id, (time.monotonic()-started_at)*1000, rms)
                session.add_pcm(data)
                if drain_task is None or drain_task.done():
                    if drain_task:
                        await drain_task
                    drain_task = asyncio.create_task(drain_audio())
    except WebSocketDisconnect:
        pass
    except Exception:
        traceback.print_exc()
        try:
            await send({"type": "error", "message": "Streaming interrupted. Your received text is preserved; please retry."})
            await ws.close(code=1011)
        except Exception:
            pass
    finally:
        # Thread work cannot be cancelled safely; let it finish before releasing the model.
        if cleanup_task:
            cleanup_task.cancel()
        with anyio.CancelScope(shield=True):
            if cleanup_task:
                await asyncio.gather(cleanup_task, return_exceptions=True)
            try:
                if drain_task:
                    await asyncio.gather(drain_task, return_exceptions=True)
            finally:
                if owns_slot:
                    model_slot.release()


frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="ui")


def main() -> None:
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
