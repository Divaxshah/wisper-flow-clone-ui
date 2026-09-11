"""Realtime Nemotron ASR + pause-delimited sentence cleanup."""

from __future__ import annotations

import asyncio
import json
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.asr import load_dotenv_files, load_model_in_background, runtime_status
from backend.cleanup import cleanup_span

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


@app.websocket("/ws/transcribe")
async def transcribe_socket(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    session = None
    cleanup_enabled = True
    prior_cleaned: list[str] = []
    sentence_id = 0
    pending_cleanup: set[asyncio.Task] = set()

    async def send(payload: dict) -> None:
        await ws.send_text(json.dumps(payload))

    async def run_cleanup(sid: int, raw: str) -> None:
        context = " ".join(prior_cleaned[-3:])
        try:
            cleaned = await loop.run_in_executor(None, cleanup_span, raw, context)
        except Exception as e:
            await send({"type": "error", "message": f"Cleanup failed: {e}"})
            cleaned = raw
        prior_cleaned.append(cleaned)
        await send({"type": "cleaned", "id": sid, "raw": raw, "cleaned": cleaned})

    try:
        meta = runtime_status()
        await send({"type": "ready", **meta})
        drain_task: asyncio.Task | None = None

        async def drain_audio() -> None:
            current = session
            if current is None:
                return
            snap = await loop.run_in_executor(None, current.consume_chunks)
            await send({"type": "partial", **snap})
            while current.has_chunk():
                snap = await loop.run_in_executor(None, current.consume_chunks)
                await send({"type": "partial", **snap})

        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    payload = {}
                kind = payload.get("type")

                if kind == "start" or (session is None and kind is None):
                    from backend.asr import LiveSession

                    lang = payload.get("language") or "auto"
                    profile = payload.get("profile") or "Lowest latency"
                    cleanup_enabled = bool(payload.get("cleanup", True))
                    try:
                        session = await loop.run_in_executor(
                            None, lambda: LiveSession(lang=lang, profile=profile)
                        )
                    except Exception as e:
                        await send({"type": "error", "message": str(e)})
                        continue
                    prior_cleaned = []
                    sentence_id = 0
                    await send({"type": "started", "language": lang, "profile": profile})
                    continue

                if kind == "commit":
                    if session is None:
                        continue
                    raw = session.commit(force=False)
                    if not raw:
                        continue
                    sentence_id += 1
                    await send({"type": "commit", "id": sentence_id, "raw": raw})
                    if cleanup_enabled:
                        task = asyncio.create_task(run_cleanup(sentence_id, raw))
                        pending_cleanup.add(task)
                        task.add_done_callback(pending_cleanup.discard)
                    else:
                        prior_cleaned.append(raw)
                        await send(
                            {"type": "cleaned", "id": sentence_id, "raw": raw, "cleaned": raw}
                        )
                    continue

                if kind == "end":
                    if session is not None:
                        snap = await loop.run_in_executor(None, session.flush)
                        await send({"type": "partial", **snap})
                        raw = session.commit(force=True)
                        if raw:
                            sentence_id += 1
                            await send({"type": "commit", "id": sentence_id, "raw": raw})
                            if cleanup_enabled:
                                task = asyncio.create_task(run_cleanup(sentence_id, raw))
                                pending_cleanup.add(task)
                                task.add_done_callback(pending_cleanup.discard)
                            else:
                                await send(
                                    {
                                        "type": "cleaned",
                                        "id": sentence_id,
                                        "raw": raw,
                                        "cleaned": raw,
                                    }
                                )
                    if pending_cleanup:
                        await asyncio.gather(*pending_cleanup, return_exceptions=True)
                    await send({"type": "ended"})
                    session = None
                    continue

                continue

            data = message.get("bytes")
            if not data or session is None:
                continue
            session.add_pcm(data)
            if drain_task is None or drain_task.done():
                drain_task = asyncio.create_task(drain_audio())

    except WebSocketDisconnect:
        pass
    except Exception:
        traceback.print_exc()
        try:
            await send({"type": "error", "message": "Streaming session failed."})
        except Exception:
            pass
    finally:
        for task in list(pending_cleanup):
            task.cancel()


frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="ui")


def main() -> None:
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
