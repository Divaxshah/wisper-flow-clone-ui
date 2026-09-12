# Wisper live dictation

Cache-aware Nemotron ASR streams text while you speak. Optional smart cleanup removes fillers and clear self-corrections after a 900 ms pause. Original text is always available in the transcript view.

## Run locally

From this repository, use two terminals:

```bash
# Backend (Python 3.10+, with a compatible PyTorch/NeMo environment)
cd backend
uv sync
uv pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git'
uv run backend
```

```bash
# Frontend, from this repository
cd frontend
npm ci
npm run dev
```

If you already have an environment with NeMo installed, run the backend from this repository with:

```bash
PYTHONPATH=backend/src /path/to/environment/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open the URL printed by Vite (normally http://127.0.0.1:5173). Vite proxies `/api` and `/ws` to port 8000; set `VITE_API_TARGET` to override it. A microphone requires localhost or HTTPS. For a remote browser, use an HTTPS tunnel to Vite.

Click **Start dictation** to toggle recording, or hold **Space** outside form controls and release to stop. You can also hold the recording button. Wait for **Finishing…** to complete before starting again. Subsequent recordings append to the same page. Text stays in memory until the page is closed or refreshed; copy anything you want to keep.

## Optional cleanup

Set `OPENROUTER_API_KEY` in a `.env` at this repository's root or in `backend/`. `OPENROUTER_MODEL` overrides the default `openai/gpt-4o-mini`. The key stays on the server. Cleanup sends transcript text to OpenRouter; audio stays with your ASR server. Without a key, smart cleanup is disabled. Provider errors retain the raw text. **Original text** switches between cleaned and original wording.

## Streaming behavior

- The browser sends mono, 16 kHz PCM16 frames during recording and flushes the worklet tail before sending `end`.
- A session starts only after the server acknowledges it. Each session has fresh encoder and decoder caches.
- **Balanced** uses 320 ms model chunks. The other profiles use 80, 560, or 1120 ms. Actual response time also depends on hardware and backlog.
- Feature extraction preserves waveform context across boundaries, uses 20 ms of future audio for the centered STFT, and passes exactly the new feature frames into the model cache.
- The server publishes a partial after each model step. Pause commits and final flush wait for queued inference, so they cannot mutate caches concurrently.
- One active recording per backend process is supported because NeMo has shared mutable model configuration. Concurrent clients receive a retry message.
- Audio backlog is bounded. A slow or interrupted connection reports an error and preserves text already received, but does not replay lost audio.
- Pause detection currently uses an energy threshold, not a trained voice activity model. Cleanup operates on pause-delimited spans, so corrections across previously committed spans are not rewritten.

## Checks

```bash
cd frontend
npm run build

# From the repository root, in an environment with backend dependencies:
python -m pip install pytest httpx
PYTHONPATH=backend/src python -m pytest backend/tests -q

# Real-model smoke test; requires NeMo and a 16 kHz mono PCM16 speech WAV:
PYTHONPATH=backend/src python backend/tests/smoke_asr.py /path/to/speech.wav
```

The regression suite checks partials before end, repeated recording, unique sentence IDs, stop/drain ordering, duplicate start, cleanup failure, disconnect recovery, and audio queue limits. The real-model check runs two independent recordings and requires nonempty partials before either ends.

## Serve the built UI

```bash
cd frontend
npm run build
```

Restart the backend; it serves `frontend/dist` at http://127.0.0.1:8000. This is a local single-user tool, without authentication or production deployment hardening.

## Reference guidance

- [NVIDIA model and cache-aware streaming guidance](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Wispr Flow smart formatting and backtrack](https://docs.wisprflow.ai/articles/5373093536-how-do-i-use-smart-formatting-and-backtrack)
- [Google Gboard Rambler](https://support.google.com/gboard/answer/17468539)
