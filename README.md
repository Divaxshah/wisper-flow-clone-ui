# Wisper app

FastAPI app for real-time speech-to-text using NVIDIA Nemotron ASR. It accepts streamed PCM audio over WebSocket and can optionally clean transcripts with OpenRouter.

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/)
- A PyTorch/NeMo-compatible environment

## Setup

```bash
uv sync
uv pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git'
```

## Run

```bash
uv run app
```

The server starts at `http://127.0.0.1:8000`.

Check model readiness with curl:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/api/status
```

The response reports whether the speech model is loading, ready, or unavailable.

## API

### `GET /api/status`

Returns the current model/runtime status.

### `WS /ws/transcribe`

Streams mono, 16 kHz PCM16 audio for transcription.

1. Connect and wait for the `ready` message.
2. Send a JSON `start` message.
3. Send PCM16 audio as binary WebSocket frames.
4. Send `commit` to commit a pause, or `end` to finish the recording.

Example start message:

```json
{
  "type": "start",
  "language": "auto",
  "profile": "Most accurate",
  "cleanup": true
}
```

The server emits messages such as `started`, `partial`, `commit`, `polishing`, `polished`, `ended`, `warning`, and `error`.

For a short client handoff, see [Developer integration guide](docs/INTEGRATION.md).

Only one recording can use a backend process at a time because the NeMo model has shared mutable decoding state.

## Transcript cleanup

Add an OpenRouter key to `.env` in the repository root or `app/`:

```dotenv
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

`OPENROUTER_MODEL` is optional. If remote cleanup is unavailable or its output fails validation, the backend applies conservative local cleanup instead. Audio is never sent to OpenRouter.

## Tests

From the repository root:

```bash
uv pip install --python .venv/bin/python pytest
PYTHONPATH=app/src python -m pytest app/tests -q
```

Real-model smoke test with a mono, 16 kHz PCM16 WAV file:

```bash
PYTHONPATH=app/src python app/tests/smoke_asr.py /path/to/speech.wav
```

Optional OpenRouter cleanup smoke test:

```bash
PYTHONPATH=app/src python app/tests/smoke_cleanup.py
```
