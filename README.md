# Wisper realtime UI

Nemotron streams the words as you talk. After a short pause, that span is cleaned and locked into the transcript.

## Run

From the repo root, in two terminals:

```bash
# 1. API + ASR (loads nvidia/nemotron-3.5-asr-streaming-0.6b)
uv sync
uv pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"
uv run --package backend backend

# 2. UI
cd wisper-flow-clone-ui/frontend
npm install
npm run dev
```

Open the URL Vite prints. On this machine that is http://127.0.0.1:5173. From another computer or an EC2 public IP, use the **https://** Network address and accept the self-signed certificate — browsers hide `getUserMedia` on `http://<ip>`.

```bash
# from your laptop, if you would rather keep HTTP
ssh -L 5173:127.0.0.1:5173 -L 8000:127.0.0.1:8000 ubuntu@<host>
# then open http://127.0.0.1:5173
```

The ASR process must be running on the same host (`uv run --package backend backend`). Vite proxies `/api` and `/ws` to `127.0.0.1:8000`.

The pedal stays locked until `/api/status` reports `ready`. Hold it (or hold Space) to speak. A ~600ms silence commits the current line; OpenRouter then rewrites that span. A short click latches listening until you click again.

Cleanup uses `OPENROUTER_API_KEY` from a `.env` at the repo root or `src/wisper_flow_clone/.env`. Optional: `OPENROUTER_MODEL`.

To serve the UI from FastAPI instead of Vite:

```bash
cd wisper-flow-clone-ui/frontend && npm run build
uv run --package backend backend
```

Then open http://127.0.0.1:8000
