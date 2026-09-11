const TARGET_SR = 16000;
const WORKLET = `
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) {
      this.port.postMessage(channel);
    }
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);
`;

function resampleTo16k(input: Float32Array, fromRate: number): Float32Array {
  if (fromRate === TARGET_SR) return input;
  const ratio = fromRate / TARGET_SR;
  const outLen = Math.max(1, Math.floor(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i += 1) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = x - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

function floatToPcm16(input: Float32Array): ArrayBuffer {
  const buf = new ArrayBuffer(input.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < input.length; i += 1) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

function rms(input: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < input.length; i += 1) sum += input[i] * input[i];
  return Math.sqrt(sum / Math.max(1, input.length));
}

export type CaptureHandle = {
  stop: () => Promise<void>;
};

export async function startCapture(options: {
  onPcm: (bytes: ArrayBuffer) => void;
  onLevel: (values: number[]) => void;
  onPause: () => void;
  pauseMs?: number;
}): Promise<CaptureHandle> {
  const pauseMs = options.pauseMs ?? 600;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);
  const analyser = context.createAnalyser();
  analyser.fftSize = 64;
  source.connect(analyser);

  const blob = new Blob([WORKLET], { type: "application/javascript" });
  const url = URL.createObjectURL(blob);
  await context.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);

  const node = new AudioWorkletNode(context, "capture-processor");
  const mute = context.createGain();
  mute.gain.value = 0;
  source.connect(node);
  node.connect(mute);
  mute.connect(context.destination);

  let spoken = false;
  let silentSince: number | null = null;
  let paused = false;
  const bins = new Uint8Array(analyser.frequencyBinCount);
  let raf = 0;

  const tick = () => {
    analyser.getByteTimeDomainData(bins);
    const levels: number[] = [];
    const step = Math.max(1, Math.floor(bins.length / 24));
    for (let i = 0; i < 24; i += 1) {
      const v = bins[i * step] ?? 128;
      levels.push(Math.abs(v - 128) / 128);
    }
    options.onLevel(levels);
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);

  node.port.onmessage = (event: MessageEvent<Float32Array>) => {
    const resampled = resampleTo16k(event.data, context.sampleRate);
    options.onPcm(floatToPcm16(resampled));

    const level = rms(resampled);
    const now = performance.now();
    if (level > 0.018) {
      spoken = true;
      silentSince = null;
      paused = false;
      return;
    }
    if (!spoken) return;
    if (silentSince === null) silentSince = now;
    if (!paused && now - silentSince >= pauseMs) {
      paused = true;
      options.onPause();
    }
  };

  return {
    stop: async () => {
      cancelAnimationFrame(raf);
      node.port.onmessage = null;
      node.disconnect();
      source.disconnect();
      mute.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      await context.close();
    },
  };
}
