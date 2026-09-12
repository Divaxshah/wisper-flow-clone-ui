import "./styles.css";
import { micUnavailableReason, startCapture, type CaptureHandle } from "./audio";
import type { ServerEvent, StatusPayload } from "./types";

const languageEl = document.querySelector<HTMLSelectElement>("#language")!;
const profileEl = document.querySelector<HTMLSelectElement>("#profile")!;
const cleanupEl = document.querySelector<HTMLInputElement>("#cleanup")!;
const pedal = document.querySelector<HTMLButtonElement>("#pedal")!;
const pedalLabel = pedal.querySelector<HTMLSpanElement>(".pedal-label")!;
const transcriptEl = document.querySelector<HTMLElement>("#transcript")!;
const emptyEl = document.querySelector<HTMLElement>("#empty")!;
const liveEl = document.querySelector<HTMLElement>("#live")!;
const vuEl = document.querySelector<HTMLElement>("#vu")!;
const statusEl = document.querySelector<HTMLElement>("#model-status")!;
const errorEl = document.querySelector<HTMLElement>("#error")!;
const toastEl = document.querySelector<HTMLElement>("#toast")!;
const copyBtn = document.querySelector<HTMLButtonElement>("#copy")!;

const BARS = 24;
for (let i = 0; i < BARS; i += 1) {
  vuEl.appendChild(document.createElement("i"));
}
const vuBars = [...vuEl.querySelectorAll("i")];

let socket: WebSocket | null = null;
let capture: CaptureHandle | null = null;
let phase: "idle" | "starting" | "listening" | "stopping" = "idle";
let stopRequested = false;
let sessionNumber = 0;
let modelReady = false;
let statusLoaded = false;
let phaseTimer = 0;
let startAck: { resolve: () => void; reject: (error: Error) => void } | null = null;
const activityEl = document.querySelector<HTMLElement>("#activity")!;
const originalEl = document.querySelector<HTMLInputElement>("#original")!;
const wordCountEl = document.querySelector<HTMLElement>("#word-count")!;

function setPhase(next: typeof phase) {
  phase = next;
  setListening(next === "listening");
  pedal.disabled = !modelReady || next === "stopping";
  pedalLabel.textContent = { idle: "Start dictation", starting: "Connecting…", listening: "Stop dictation", stopping: "Finishing…" }[next];
  activityEl.textContent = { idle: "Ready when you are", starting: "Preparing microphone and speech model", listening: "Listening · words appear as you speak", stopping: "Finishing your transcript" }[next];
  document.body.dataset.phase = next;
  languageEl.disabled = profileEl.disabled = next !== "idle";
  cleanupEl.disabled = next !== "idle" || cleanupEl.dataset.available === "false";
}

function updateWords() {
  const text = [...transcriptEl.querySelectorAll<HTMLElement>(".sentence")].map(el => el.textContent).join(" ");
  wordCountEl.textContent = `${text.trim() ? text.trim().split(/\s+/u).length : 0} words`;
}

function renderSentence(el: HTMLElement) {
  el.textContent = originalEl.checked ? el.dataset.raw ?? "" : el.dataset.cleaned ?? el.dataset.raw ?? "";
  updateWords();
}

async function resetSession(message?: string) {
  window.clearTimeout(phaseTimer);
  startAck?.reject(new Error(message ?? "Recording cancelled."));
  startAck = null;
  const old = socket;
  socket = null;
  old?.close();
  const oldCapture = capture;
  capture = null;
  await oldCapture?.stop().catch(() => {});
  setPhase("idle");
  if (message) {
    if (!liveEl.hidden && liveEl.textContent?.trim()) {
      const el = document.createElement("p");
      el.className = "sentence";
      el.dataset.raw = liveEl.textContent;
      el.dataset.state = "settled";
      transcriptEl.appendChild(el);
      renderSentence(el);
      liveEl.hidden = true;
      liveEl.textContent = "";
      copyBtn.hidden = false;
    }
    showError(message);
  }
}
let holdStarted = 0;
let pressWasListening = false;
let spaceDown = false;
let toastTimer = 0;

function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/transcribe`;
}

function showError(message: string | null) {
  if (!message) {
    errorEl.hidden = true;
    errorEl.textContent = "";
    return;
  }
  errorEl.hidden = false;
  errorEl.textContent = message;
}

function toast(message: string) {
  toastEl.textContent = message;
  toastEl.hidden = false;
  toastEl.setAttribute("data-starting-style", "");
  requestAnimationFrame(() => {
    toastEl.removeAttribute("data-starting-style");
  });
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toastEl.hidden = true;
  }, 1800);
}

function setListening(on: boolean) {
  pedal.setAttribute("aria-pressed", String(on));
  pedalLabel.textContent = on ? pedalLabel.dataset.hot! : pedalLabel.dataset.idle!;
  vuEl.classList.toggle("is-hot", on);
  if (!on) {
    vuBars.forEach((bar) => bar.style.setProperty("--a", "0.12"));
    vuBars.forEach((bar) => {
      bar.style.transform = "scaleY(0.12)";
    });
  }
}

function ensureSocket(): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl());
    socket = ws;
    ws.binaryType = "arraybuffer";
    const timer = window.setTimeout(() => {
      reject(new Error("The speech server did not respond. Check the connection and retry."));
      ws.close();
    }, 10000);
    ws.addEventListener("open", () => { window.clearTimeout(timer); resolve(ws); });
    ws.addEventListener("error", () => reject(new Error("Could not connect to the speech server. Check that the backend is running.")));
    ws.addEventListener("message", (event) => {
      if (socket !== ws) return;
      try { onEvent(JSON.parse(event.data) as ServerEvent); } catch { /* malformed frame */ }
    });
    ws.addEventListener("close", () => {
      window.clearTimeout(timer);
      reject(new Error("The connection closed before recording started."));
      if (socket === ws) void resetSession("Connection lost. Received text is preserved. Start again to reconnect.");
    });
  });
}

function sentenceEl(id: number): HTMLElement {
  let el = transcriptEl.querySelector<HTMLElement>(`[data-id="${sessionNumber}-${id}"]`);
  if (!el) {
    emptyEl.hidden = true;
    el = document.createElement("p");
    el.className = "sentence";
    el.dataset.id = `${sessionNumber}-${id}`;
    transcriptEl.appendChild(el);
  }
  copyBtn.hidden = false;
  return el;
}

function onEvent(event: ServerEvent) {
  if (event.type === "started") {
    startAck?.resolve();
    startAck = null;
    return;
  }
  if (event.type === "ended") {
    void resetSession();
    return;
  }
  if (event.type === "warning") {
    showError(event.message);
    return;
  }
  if (event.type === "error") {
    void resetSession(event.message);
    return;
  }
  if (event.type === "partial") {
    if (event.live) {
      emptyEl.hidden = true;
      liveEl.hidden = false;
      liveEl.textContent = event.live;
    } else {
      liveEl.hidden = true;
      liveEl.textContent = "";
    }
    if (event.detected_lang) {
      const current = statusEl.dataset.base ?? "";
      statusEl.textContent = current
        ? `${current} · heard as ${event.detected_lang}`
        : `Heard as ${event.detected_lang}`;
    }
    return;
  }
  if (event.type === "commit") {
    const el = sentenceEl(event.id);
    el.dataset.state = "cleaning";
    el.dataset.raw = event.raw;
    renderSentence(el);
    liveEl.hidden = true;
    liveEl.textContent = "";
    return;
  }
  if (event.type === "cleaned") {
    const el = sentenceEl(event.id);
    el.dataset.raw = event.raw;
    el.dataset.cleaned = event.cleaned;
    el.dataset.state = "settled";
    renderSentence(el);
  }
}

async function startListening() {
  if (phase !== "idle" || !modelReady) return;
  setPhase("starting");
  stopRequested = false;
  sessionNumber += 1;
  showError(null);
  try {
    const blocked = micUnavailableReason();
    if (blocked) throw new Error(blocked);
    const ws = await ensureSocket();
    const started = new Promise<void>((resolve, reject) => { startAck = { resolve, reject }; });
    phaseTimer = window.setTimeout(() => startAck?.reject(new Error("Speech session startup timed out. Please retry.")), 30000);
    ws.send(JSON.stringify({ type: "start", language: languageEl.value || "auto", profile: profileEl.value || "Balanced", cleanup: cleanupEl.checked }));
    await started;
    window.clearTimeout(phaseTimer);
    if (stopRequested) {
      setPhase("listening");
      await stopListening();
      return;
    }
    const handle = await startCapture({
      onPcm: bytes => {
        if (socket !== ws || ws.readyState !== WebSocket.OPEN) return;
        if (ws.bufferedAmount > 16000 * 2 * 5) {
          void resetSession("The connection cannot keep up with audio. Please reconnect and try again.");
          return;
        }
        ws.send(bytes);
      },
      onLevel: values => values.forEach((value, i) => {
        if (vuBars[i]) vuBars[i].style.transform = `scaleY(${Math.max(0.08, Math.min(1, value * 3))})`;
      }),
      pauseMs: 900,
      onPause: () => { if (socket === ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "commit" })); },
      onEnded: () => { if (phase === "listening") void stopListening(); },
    });
    if (socket !== ws) { await handle.stop(); return; }
    capture = handle;
    setPhase("listening");
    if (stopRequested) await stopListening();
  } catch (err) {
    await resetSession(err instanceof Error ? err.message : "Microphone could not start.");
  }
}

async function stopListening() {
  if (phase === "starting") { stopRequested = true; return; }
  if (phase !== "listening") return;
  setPhase("stopping");
  try {
    await capture?.stop();
    capture = null;
    if (socket?.readyState !== WebSocket.OPEN) throw new Error("Connection lost. Please retry.");
    socket.send(JSON.stringify({ type: "end" }));
    phaseTimer = window.setTimeout(() => void resetSession("Finishing timed out. Received text is preserved; you can start again."), 60000);
  } catch (err) {
    await resetSession(err instanceof Error ? err.message : "Recording stopped unexpectedly.");
  }
}

function fillSelect(
  select: HTMLSelectElement,
  items: { label: string; value: string }[],
  selected: string,
) {
  select.replaceChildren();
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    if (item.value === selected) option.selected = true;
    select.appendChild(option);
  }
}

async function loadStatus() {
  const res = await fetch("/api/status");
  if (!res.ok) throw new Error("Status endpoint failed.");
  const data = (await res.json()) as StatusPayload;
  if (!statusLoaded) {
    fillSelect(languageEl, data.languages, "auto");
    fillSelect(
      profileEl,
      data.profiles.map((name) => ({ label: name, value: name })),
      data.default_profile,
    );

    cleanupEl.dataset.available = String(data.cleanup_available);
    if (!data.cleanup_available) cleanupEl.checked = false;
    cleanupEl.disabled = !data.cleanup_available;
    cleanupEl.title = data.cleanup_available ? "Clean up fillers and clear corrections" : "Set OPENROUTER_API_KEY on the server to enable cleanup";
    statusLoaded = true;
  }
  const apply = (payload: StatusPayload) => {
    if (payload.status === "ready") {
      statusEl.dataset.base = payload.device ? `Ready on ${payload.device}` : "Ready";
      statusEl.textContent = statusEl.dataset.base;
      modelReady = true;
      if (phase === "idle") setPhase("idle");
      return true;
    }
    if (payload.status === "error") {
      statusEl.textContent = "Model failed to load";
      showError(payload.error);
      pedal.disabled = true;
      return true;
    }
    statusEl.textContent = "Preparing the speech model…";
    pedal.disabled = true;
    return false;
  };

  if (!apply(data)) window.setTimeout(() => void refreshStatus(), 1500);
}

function onPedalRelease() {
  if (spaceDown) return;
  const held = performance.now() - holdStarted;
  if (held >= 220 || pressWasListening) {
    stopListening();
  }
}

pedal.addEventListener("pointerdown", (event) => {
  if (pedal.disabled) return;
  event.preventDefault();
  pedal.setPointerCapture(event.pointerId);
  holdStarted = performance.now();
  pressWasListening = phase !== "idle";
  if (phase === "idle") startListening().catch((err) => showError(err.message));
});

pedal.addEventListener("pointerup", onPedalRelease);
pedal.addEventListener("pointercancel", () => void stopListening());
pedal.addEventListener("click", event => {
  if (event.detail === 0) { if (phase === "idle") void startListening(); else void stopListening(); }
});
originalEl.addEventListener("change", () => transcriptEl.querySelectorAll<HTMLElement>(".sentence").forEach(renderSentence));
window.addEventListener("blur", () => {
  if (spaceDown) { spaceDown = false; void stopListening(); }
});

window.addEventListener("keydown", (event) => {
  if (event.code !== "Space") return;
  const tag = (event.target as HTMLElement | null)?.tagName;
  if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA" || tag === "BUTTON" || (event.target as HTMLElement)?.isContentEditable) return;
  event.preventDefault();
  if (spaceDown || pedal.disabled) return;
  spaceDown = true;
  startListening().catch((err) => showError(err.message));
});

window.addEventListener("keyup", (event) => {
  if (event.code !== "Space") return;
  if (!spaceDown) return;
  event.preventDefault();
  spaceDown = false;
  void stopListening();
});

copyBtn.addEventListener("click", async () => {
  const settled = [...transcriptEl.querySelectorAll<HTMLElement>(".sentence")]
    .map((el) => el.textContent?.trim() ?? "")
    .filter(Boolean)
    .join("\n");
  const live = liveEl.hidden ? "" : liveEl.textContent?.trim() ?? "";
  const text = [settled, live].filter(Boolean).join("\n");
  if (!text) return;
  try { await navigator.clipboard.writeText(text); toast("Transcript copied"); }
  catch { showError("Clipboard is unavailable. Select your transcript and copy it manually."); }
});

pedal.disabled = true;
const micBlock = micUnavailableReason();
if (micBlock) showError(micBlock);
async function refreshStatus() {
  try { await loadStatus(); }
  catch {
    statusEl.textContent = "Speech server unavailable · reconnecting…";
    window.setTimeout(() => void refreshStatus(), 3000);
  }
}
void refreshStatus();
