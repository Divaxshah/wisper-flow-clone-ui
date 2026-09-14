import "./styles.css";
import {
  micUnavailableReason,
  startCapture,
  type CaptureHandle,
} from "./audio";
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
let startAck: { resolve: () => void; reject: (error: Error) => void } | null =
  null;
const activityEl = document.querySelector<HTMLElement>("#activity")!;
const originalEl = document.querySelector<HTMLInputElement>("#original")!;
const wordCountEl = document.querySelector<HTMLElement>("#word-count")!;

const scrollEl = document.querySelector<HTMLElement>("#transcript-scroll")!;
const cleanupStatusEl = document.querySelector<HTMLElement>("#cleanup-status")!;
const languageButton =
  document.querySelector<HTMLButtonElement>("#language-button")!;
const languageDialog =
  document.querySelector<HTMLDialogElement>("#language-dialog")!;
const languageSearch =
  document.querySelector<HTMLInputElement>("#language-search")!;
const languageOptions =
  document.querySelector<HTMLElement>("#language-options")!;
const jumpButton = document.querySelector<HTMLButtonElement>("#jump-live")!;
const polishedButton =
  document.querySelector<HTMLButtonElement>("#view-polished")!;
const originalButton =
  document.querySelector<HTMLButtonElement>("#view-original")!;
let followLive = true;
let cleanupRequested = false;
let recordingBegan = 0;
let elapsedTimer = 0;

function followTranscript() {
  if (followLive)
    requestAnimationFrame(() => {
      scrollEl.scrollTop = scrollEl.scrollHeight;
    });
}
scrollEl.addEventListener("scroll", () => {
  followLive =
    scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 70;
  jumpButton.hidden = followLive || phase === "idle";
});
jumpButton.addEventListener("click", () => {
  followLive = true;
  followTranscript();
  jumpButton.hidden = true;
});

function syncSettings() {
  languageButton.disabled = !statusLoaded || phase !== "idle";
  document.querySelector<HTMLElement>("#language-label")!.textContent =
    languageEl.selectedOptions[0]?.textContent ?? "Auto-detect";
  document
    .querySelectorAll<HTMLInputElement>('input[name="recognition"]')
    .forEach((input) => {
      input.disabled = phase !== "idle";
      input.checked = input.value === profileEl.value;
    });
}

function renderLanguages() {
  const query = languageSearch.value.trim().toLocaleLowerCase();
  languageOptions.replaceChildren();
  for (const option of languageEl.options) {
    if (!option.textContent?.toLocaleLowerCase().includes(query)) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = option.textContent;
    button.setAttribute(
      "aria-pressed",
      String(option.value === languageEl.value),
    );
    button.addEventListener("click", () => {
      languageEl.value = option.value;
      syncSettings();
      languageDialog.close();
    });
    languageOptions.appendChild(button);
  }
  document.querySelector<HTMLElement>("#language-empty")!.hidden =
    languageOptions.childElementCount > 0;
}
languageButton.addEventListener("click", () => {
  languageSearch.value = "";
  renderLanguages();
  languageDialog.showModal();
  languageSearch.focus();
});
languageSearch.addEventListener("input", renderLanguages);
document
  .querySelector("#language-close")!
  .addEventListener("click", () => languageDialog.close());
languageDialog.addEventListener("click", (event) => {
  if (event.target === languageDialog) {
    const box = languageDialog.getBoundingClientRect();
    if (
      event.clientX < box.left ||
      event.clientX > box.right ||
      event.clientY < box.top ||
      event.clientY > box.bottom
    )
      languageDialog.close();
  }
});

function buildProfiles() {
  const descriptions: Record<string, [string, string, string]> = {
    "Lowest latency": ["Instant", "Words without the wait", "80 ms"],
    Balanced: ["Balanced", "A natural conversational pace", "320 ms"],
    Accurate: ["Considered", "A little more context", "560 ms"],
    "Most accurate": ["Precise", "More time for complex speech", "1.12 s"],
  };
  const container = document.querySelector("#profile-options")!;
  container.replaceChildren();
  for (const option of profileEl.options) {
    const [name, description, duration] = descriptions[option.value] ?? [
      option.value,
      "",
      "",
    ];
    const label = document.createElement("label");
    label.className = "profile-option";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "recognition";
    input.value = option.value;
    input.addEventListener("change", () => {
      profileEl.value = input.value;
      syncSettings();
    });
    const dot = document.createElement("span");
    dot.className = "profile-dot";
    dot.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    text.className = "profile-text";
    const strong = document.createElement("strong");
    strong.textContent = name;
    const small = document.createElement("small");
    small.textContent = description;
    text.append(strong, small);
    const ms = document.createElement("span");
    ms.className = "profile-ms";
    ms.textContent = duration;
    label.append(input, dot, text, ms);
    container.appendChild(label);
  }
  syncSettings();
}

function setPhase(next: typeof phase) {
  if (next === "listening" && phase !== "listening") {
    recordingBegan = performance.now();
    document.querySelector("#session-time")!.textContent = "00:00";
    elapsedTimer = window.setInterval(() => {
      const seconds = Math.floor((performance.now() - recordingBegan) / 1000);
      document.querySelector("#session-time")!.textContent =
        `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    }, 1000);
  }
  if (next !== "listening") window.clearInterval(elapsedTimer);
  phase = next;
  setListening(next === "listening");
  pedal.disabled = !modelReady || next === "stopping";
  pedalLabel.textContent = {
    idle: "Start recording",
    starting: "Connecting…",
    listening: "Stop recording",
    stopping: "Finishing…",
  }[next];
  activityEl.textContent = {
    idle: "Ready when you are",
    starting: "Getting ready…",
    listening: "Listening to you",
    stopping: "Finishing your words…",
  }[next];
  document.body.dataset.phase = next;
  languageEl.disabled = profileEl.disabled = next !== "idle";
  cleanupEl.disabled =
    next !== "idle" || cleanupEl.dataset.available === "false";
  syncSettings();
  updateWords();
  if (next === "idle") jumpButton.hidden = true;
}

function updateWords() {
  const rows = [...transcriptEl.querySelectorAll<HTMLElement>(".sentence")];
  const text = [
    ...rows.map((el) => el.textContent ?? ""),
    liveEl.hidden ? "" : (liveEl.textContent ?? ""),
  ]
    .join(" ")
    .trim();
  wordCountEl.textContent = `${text ? text.split(/\s+/u).length : 0} words`;
  copyBtn.disabled = !text;
  const pending = rows.filter((el) => el.dataset.cleanup === "pending").length;
  const failed = rows.some((el) => el.dataset.cleanup === "failed");
  cleanupStatusEl.dataset.state = pending
    ? "pending"
    : failed
      ? "failed"
      : "settled";
  cleanupStatusEl.textContent = pending
    ? `Polishing ${pending === 1 ? "your words" : `${pending} passages`}…`
    : originalEl.checked
      ? "Original words · no edits"
      : failed
        ? "Original kept · cleanup could not be applied"
        : rows.some((el) => el.dataset.cleanup === "applied")
          ? "Polished · original always available"
          : rows.some((el) => el.dataset.cleanup === "deferred")
            ? "Live words · polishing after a pause"
            : text
              ? "Original words preserved"
              : "Ready for your first thought";
}

function renderSentence(el: HTMLElement) {
  el.textContent = originalEl.checked
    ? (el.dataset.raw ?? "")
    : (el.dataset.cleaned ?? el.dataset.raw ?? "");
  updateWords();
  followTranscript();
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
    transcriptEl
      .querySelectorAll<HTMLElement>('[data-cleanup="pending"]')
      .forEach((el) => {
        el.dataset.cleanup = "failed";
        el.dataset.state = "settled";
      });
    updateWords();
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
  pedalLabel.textContent = on
    ? pedalLabel.dataset.hot!
    : pedalLabel.dataset.idle!;
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
      reject(
        new Error(
          "The speech server did not respond. Check the connection and retry.",
        ),
      );
      ws.close();
    }, 10000);
    ws.addEventListener("open", () => {
      window.clearTimeout(timer);
      resolve(ws);
    });
    ws.addEventListener("error", () =>
      reject(
        new Error(
          "Could not connect to the speech server. Check that the backend is running.",
        ),
      ),
    );
    ws.addEventListener("message", (event) => {
      if (socket !== ws) return;
      try {
        onEvent(JSON.parse(event.data) as ServerEvent);
      } catch {
        /* malformed frame */
      }
    });
    ws.addEventListener("close", () => {
      window.clearTimeout(timer);
      reject(new Error("The connection closed before recording started."));
      if (socket === ws)
        void resetSession(
          "Connection lost. Received text is preserved. Start again to reconnect.",
        );
    });
  });
}

function sentenceEl(id: number): HTMLElement {
  let el = transcriptEl.querySelector<HTMLElement>(
    `[data-id="${sessionNumber}-${id}"]`,
  );
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
  if (event.type === "polishing" || event.type === "polished") {
    if (!event.ids.length) return;
    if (event.type === "polishing") {
      // Keep the last polished version visible while the next pass runs.
      for (const id of event.ids) {
        const row = transcriptEl.querySelector<HTMLElement>(`[data-id="${sessionNumber}-${id}"]`);
        if (row) row.dataset.cleanup = "pending";
      }
      updateWords();
      return;
    }
    // Replace only this snapshot's committed prefix. Speech received while
    // cleanup was running remains visible in the live suffix and later rows.
    const first = sentenceEl(event.ids[0]);
    for (const id of event.ids.slice(1)) {
      transcriptEl
        .querySelector(`[data-id="${sessionNumber}-${id}"]`)
        ?.remove();
    }
    first.dataset.raw = event.raw;
    first.dataset.cleaned = event.cleaned;
    first.dataset.cleanup = event.cleanup_status;
    first.dataset.state = "settled";
    renderSentence(first);
    return;
  }
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
    updateWords();
    followTranscript();
    if (event.detected_lang) {
      const current = statusEl.dataset.base ?? "";
      statusEl.textContent = current
        ? `${current} · ${event.detected_lang}`
        : `Heard as ${event.detected_lang}`;
    }
    return;
  }
  if (event.type === "commit") {
    const el = sentenceEl(event.id);
    el.dataset.state = "cleaning";
    el.dataset.cleanup =
      event.cleanup_status ?? (cleanupRequested ? "pending" : "skipped");
    el.dataset.raw = event.raw;
    renderSentence(el);
    liveEl.hidden = true;
    liveEl.textContent = "";
    updateWords();
    followTranscript();
    return;
  }
  if (event.type === "cleaned") {
    const el = sentenceEl(event.id);
    el.dataset.raw = event.raw;
    el.dataset.cleaned = event.cleaned;
    el.dataset.cleanup =
      event.cleanup_status ?? (cleanupRequested ? "applied" : "skipped");
    el.dataset.state = "settled";
    renderSentence(el);
  }
}

async function startListening() {
  if (phase !== "idle" || !modelReady) return;
  setPhase("starting");
  stopRequested = false;
  sessionNumber += 1;
  const recording = sessionNumber;
  cleanupRequested = cleanupEl.checked;
  followLive = true;
  followTranscript();
  showError(null);
  try {
    const blocked = micUnavailableReason();
    if (blocked) throw new Error(blocked);
    const ws = await ensureSocket();
    const started = new Promise<void>((resolve, reject) => {
      startAck = { resolve, reject };
    });
    phaseTimer = window.setTimeout(
      () =>
        startAck?.reject(
          new Error("Speech session startup timed out. Please retry."),
        ),
      30000,
    );
    ws.send(
      JSON.stringify({
        type: "start",
        language: languageEl.value || "auto",
        profile: profileEl.value || "Balanced",
        cleanup: cleanupEl.checked,
      }),
    );
    await started;
    window.clearTimeout(phaseTimer);
    if (stopRequested) {
      setPhase("listening");
      await stopListening();
      return;
    }
    const handle = await startCapture({
      onPcm: (bytes) => {
        if (socket !== ws || ws.readyState !== WebSocket.OPEN) return;
        if (ws.bufferedAmount > 16000 * 2 * 5) {
          void resetSession(
            "The connection cannot keep up with audio. Please reconnect and try again.",
          );
          return;
        }
        ws.send(bytes);
      },
      onLevel: (values) =>
        values.forEach((value, i) => {
          if (vuBars[i])
            vuBars[i].style.transform =
              `scaleY(${Math.max(0.08, Math.min(1, value * 3))})`;
        }),
      pauseMs: 1600,
      onPause: () => {
        if (socket === ws && ws.readyState === WebSocket.OPEN)
          ws.send(JSON.stringify({ type: "commit" }));
      },
      onEnded: () => {
        if (phase === "listening") void stopListening();
      },
    });
    if (socket !== ws) {
      await handle.stop();
      return;
    }
    capture = handle;
    // A connected audio graph can still be waiting for its first microphone
    // samples. Do not invite speech until a PCM frame has actually been sent.
    let micTimer = 0;
    try {
      await Promise.race([
        handle.ready,
        new Promise<never>((_, reject) => {
          micTimer = window.setTimeout(() => reject(new Error("The microphone is not delivering audio. Check your input device and retry.")), 8000);
        }),
      ]);
    } finally {
      window.clearTimeout(micTimer);
    }
    if (socket !== ws) return;
    setPhase("listening");
    if (stopRequested) await stopListening();
  } catch (err) {
    if (sessionNumber !== recording) return;
    await resetSession(
      err instanceof Error ? err.message : "Microphone could not start.",
    );
  }
}

async function stopListening() {
  if (phase === "starting") {
    stopRequested = true;
    return;
  }
  if (phase !== "listening") return;
  setPhase("stopping");
  try {
    await capture?.stop();
    capture = null;
    if (socket?.readyState !== WebSocket.OPEN)
      throw new Error("Connection lost. Please retry.");
    socket.send(JSON.stringify({ type: "end" }));
    phaseTimer = window.setTimeout(
      () =>
        void resetSession(
          "Finishing timed out. Received text is preserved; you can start again.",
        ),
      60000,
    );
  } catch (err) {
    await resetSession(
      err instanceof Error ? err.message : "Recording stopped unexpectedly.",
    );
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
    cleanupEl.title = data.cleanup_available
      ? "Clean up fillers and clear corrections"
      : "Set OPENROUTER_API_KEY on the server to enable cleanup";
    document.querySelector<HTMLElement>("#cleanup-help")!.textContent =
      data.cleanup_available
        ? "Polished after a pause. Your original is kept."
        : "Unavailable · add a cleanup key on your server.";
    statusLoaded = true;
    buildProfiles();
  }
  const apply = (payload: StatusPayload) => {
    if (payload.status === "ready") {
      statusEl.dataset.base = "Connected";
      statusEl.title = payload.device
        ? `Speech engine: ${payload.device}`
        : "Speech engine ready";
      document.body.dataset.ready = "true";
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
pedal.addEventListener("click", (event) => {
  if (event.detail === 0) {
    if (phase === "idle") void startListening();
    else void stopListening();
  }
});
function changeView(original: boolean) {
  originalEl.checked = original;
  originalButton.setAttribute("aria-pressed", String(original));
  polishedButton.setAttribute("aria-pressed", String(!original));
  const oldScroll = scrollEl.scrollTop;
  const wasFollowing = followLive;
  // Switching versions should not force the reader to the end.
  followLive = false;
  transcriptEl
    .querySelectorAll<HTMLElement>(".sentence")
    .forEach(renderSentence);
  scrollEl.scrollTop = oldScroll;
  followLive = wasFollowing;
  updateWords();
}
originalEl.addEventListener("change", () => changeView(originalEl.checked));
originalButton.addEventListener("click", () => changeView(true));
polishedButton.addEventListener("click", () => changeView(false));
window.addEventListener("blur", () => {
  if (spaceDown) {
    spaceDown = false;
    void stopListening();
  }
});

window.addEventListener("keydown", (event) => {
  if (event.code !== "Space") return;
  const tag = (event.target as HTMLElement | null)?.tagName;
  if (languageDialog.open) return;
  if (
    tag === "SELECT" ||
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "BUTTON" ||
    (event.target as HTMLElement)?.isContentEditable
  )
    return;
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
  const live = liveEl.hidden ? "" : (liveEl.textContent?.trim() ?? "");
  const text = [settled, live].filter(Boolean).join("\n");
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    toast("Transcript copied");
  } catch {
    showError(
      "Clipboard is unavailable. Select your transcript and copy it manually.",
    );
  }
});

pedal.disabled = true;
const micBlock = micUnavailableReason();
if (micBlock) showError(micBlock);
async function refreshStatus() {
  try {
    await loadStatus();
  } catch {
    statusEl.textContent = "Speech server unavailable · reconnecting…";
    window.setTimeout(() => void refreshStatus(), 3000);
  }
}
void refreshStatus();
