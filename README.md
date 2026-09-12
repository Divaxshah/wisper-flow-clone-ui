# Wisper live dictation

Cache-aware Nemotron ASR streams text while you speak. Optional smart cleanup removes fillers and clear self-corrections while you record, after approximately 1.6 seconds of silence. Original text is always available in the transcript view.

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
- Pause detection currently uses an energy threshold, not a trained voice activity model. After a 1.6-second pause, cleanup uses the canonical raw transcript of the recording so far, so corrections across pauses can be resolved. It runs in the background while audio and partial text continue streaming. Earlier recordings remain unchanged.

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

## Workspace and cleanup checks

The workspace now has an internally scrolling transcript and a persistent recording bar. It follows live text only while you are at the bottom; **Back to live text** resumes following after you scroll up. The **Original / Polished** control preserves both versions. A status below the document distinguishes pending cleanup, applied cleanup, and a fallback to original wording.

Cleanup is instructed to retain complete sentences and speech intent rather than summarize text into topic labels. Empty, refused, and incomplete provider responses preserve the original text and report a failure. Cleanup runs after longer pauses using the recording so far, without any previous generated text. One request runs at a time, and only the newest waiting snapshot is kept. Superseded responses are discarded. Results replace only their committed prefix; newer live words stay visible. Stopping flushes remaining audio and waits for the latest required pass, without repeating unchanged work. A conservative output check rejects new vocabulary, unsupported symbols, and added word repetitions; rejected output leaves the full original recording visible. This is a guardrail, not a guarantee of semantic equivalence, and can reject otherwise reasonable paraphrases.

```bash
# Run Vite in a separate terminal, then from frontend/:
npx playwright install chromium
npm run test:ui
# Or use installed Chrome:
CHROME_PATH=/usr/bin/google-chrome npm run test:ui

# From the repository root, with a configured OpenRouter key:
PYTHONPATH=backend/src python backend/tests/smoke_cleanup.py
```

The browser suite mocks ASR and uses synthetic microphone audio to check layout stability with long transcripts, 320/390 px mobile layouts, language and recognition controls, cleanup status, original/polished switching, and recording restart. The optional cleanup smoke test sends public regression fixtures to the configured model and incurs API usage; read its outputs to assess meaning preservation.
