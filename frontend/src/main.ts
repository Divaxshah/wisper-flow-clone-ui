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
let listening = false;
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
  listening = on;
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
  if (socket && socket.readyState === WebSocket.OPEN) {
    return Promise.resolve(socket);
  }
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";
    const fail = () =>
      reject(
        new Error(
          "Could not reach the ASR server. Start it with `uv run --package backend backend` " +
            "so something is listening on 127.0.0.1:8000, then retry.",
        ),
      );
    ws.addEventListener("open", () => {
      socket = ws;
      resolve(ws);
    });
    ws.addEventListener("error", fail);
    ws.addEventListener("message", (event) => {
      try {
        onEvent(JSON.parse(event.data) as ServerEvent);
      } catch {
        /* ignore malformed frames */
      }
    });
    ws.addEventListener("close", () => {
      socket = null;
      if (listening) stopListening();
    });
  });
}

function sentenceEl(id: number): HTMLElement {
  let el = transcriptEl.querySelector<HTMLElement>(`[data-id="${id}"]`);
  if (!el) {
    emptyEl.hidden = true;
    el = document.createElement("p");
    el.className = "sentence";
    el.dataset.id = String(id);
    transcriptEl.appendChild(el);
  }
  copyBtn.hidden = false;
  return el;
}

function onEvent(event: ServerEvent) {
  if (event.type === "error") {
    showError(event.message);
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
    el.textContent = event.raw;
    liveEl.hidden = true;
    liveEl.textContent = "";
    return;
  }
  if (event.type === "cleaned") {
    const el = sentenceEl(event.id);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      el.dataset.state = "settled";
      el.textContent = event.cleaned;
      return;
    }
    el.dataset.state = "settling";
    window.setTimeout(() => {
      el.textContent = event.cleaned;
      el.dataset.state = "settled";
    }, 120);
  }
}

async function startListening() {
  if (listening) return;
  showError(null);
  const blocked = micUnavailableReason();
  if (blocked) throw new Error(blocked);
  const ws = await ensureSocket();
  ws.send(
    JSON.stringify({
      type: "start",
      language: languageEl.value || "auto",
      profile: profileEl.value || "Fast",
      cleanup: cleanupEl.checked,
    }),
  );
  try {
    capture = await startCapture({
      onPcm: (bytes) => {
        if (socket && socket.readyState === WebSocket.OPEN) socket.send(bytes);
      },
      onLevel: (values) => {
        values.forEach((value, i) => {
          const bar = vuBars[i];
          if (bar) bar.style.transform = `scaleY(${Math.max(0.08, Math.min(1, value * 3))})`;
        });
      },
      pauseMs: 900,
      onPause: () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "commit" }));
        }
      },
    });
  } catch (err) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "end" }));
      socket.close();
    }
    socket = null;
    throw err;
  }
  setListening(true);
}

async function stopListening() {
  if (!listening && !capture) return;
  try {
    await capture?.stop();
  } catch {
    /* already closed */
  }
  capture = null;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "end" }));
  }
  setListening(false);
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
  fillSelect(languageEl, data.languages, "auto");
  fillSelect(
    profileEl,
    data.profiles.map((name) => ({ label: name, value: name })),
    data.default_profile,
  );

  const apply = (payload: StatusPayload) => {
    if (payload.status === "ready") {
      statusEl.dataset.base = payload.device ? `Ready on ${payload.device}` : "Ready";
      statusEl.textContent = statusEl.dataset.base;
      pedal.disabled = false;
      return true;
    }
    if (payload.status === "error") {
      statusEl.textContent = "Model failed to load";
      showError(payload.error);
      pedal.disabled = true;
      return true;
    }
    statusEl.textContent = "Loading Nemotron… the pedal unlocks when the checkpoint is ready.";
    pedal.disabled = true;
    return false;
  };

  if (apply(data)) return;
  const poll = window.setInterval(async () => {
    const next = (await (await fetch("/api/status")).json()) as StatusPayload;
    if (apply(next)) window.clearInterval(poll);
  }, 1500);
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
  pressWasListening = listening;
  if (!listening) startListening().catch((err) => showError(err.message));
});

pedal.addEventListener("pointerup", onPedalRelease);
pedal.addEventListener("pointercancel", onPedalRelease);

window.addEventListener("keydown", (event) => {
  if (event.code !== "Space") return;
  const tag = (event.target as HTMLElement | null)?.tagName;
  if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;
  event.preventDefault();
  if (spaceDown || pedal.disabled) return;
  spaceDown = true;
  startListening().catch((err) => showError(err.message));
});

window.addEventListener("keyup", (event) => {
  if (event.code !== "Space") return;
  spaceDown = false;
  stopListening();
});

copyBtn.addEventListener("click", async () => {
  const settled = [...transcriptEl.querySelectorAll<HTMLElement>(".sentence")]
    .map((el) => el.textContent?.trim() ?? "")
    .filter(Boolean)
    .join("\n");
  const live = liveEl.hidden ? "" : liveEl.textContent?.trim() ?? "";
  const text = [settled, live].filter(Boolean).join("\n");
  if (!text) return;
  await navigator.clipboard.writeText(text);
  toast("Copied");
});

pedal.disabled = true;
const micBlock = micUnavailableReason();
if (micBlock) showError(micBlock);
loadStatus().catch((err) => showError(err.message));
