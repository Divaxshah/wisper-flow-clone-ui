"""Nemotron cache-aware streaming ASR for live microphone sessions."""

from __future__ import annotations

import copy
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NEMOTRON_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
TARGET_SR = 16000

NEMOTRON_LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Arabic (ar-AR)", "ar-AR"),
    ("English (en-GB)", "en-GB"),
    ("English (en-US)", "en-US"),
    ("French (fr-CA)", "fr-CA"),
    ("French (fr-FR)", "fr-FR"),
    ("German (de-DE)", "de-DE"),
    ("Gujarati (gu-IN)", "gu-IN"),
    ("Hindi (hi-IN)", "hi-IN"),
    ("Italian (it-IT)", "it-IT"),
    ("Japanese (ja-JP)", "ja-JP"),
    ("Korean (ko-KR)", "ko-KR"),
    ("Mandarin (zh-CN)", "zh-CN"),
    ("Polish (pl-PL)", "pl-PL"),
    ("Portuguese (pt-BR)", "pt-BR"),
    ("Portuguese (pt-PT)", "pt-PT"),
    ("Russian (ru-RU)", "ru-RU"),
    ("Spanish (es-ES)", "es-ES"),
    ("Spanish (es-US)", "es-US"),
    ("Turkish (tr-TR)", "tr-TR"),
    ("Ukrainian (uk-UA)", "uk-UA"),
    ("Vietnamese (vi-VN)", "vi-VN"),
]
NEMOTRON_LANG_CODES = {value for _, value in NEMOTRON_LANG_CHOICES}

# Cache-aware streaming lookahead. Values are [left, right] in 80ms frames.
# This checkpoint only supports right context in {0, 3, 6, 13} — not 1 (160ms).
NEMOTRON_CHUNK_PROFILES = {
    "Lowest latency": [56, 0],
    "Balanced": [56, 3],
    "Accurate": [56, 6],
    "Most accurate": [56, 13],
}
NEMOTRON_DEFAULT_CHUNK = "Balanced"

_model = None
_model_error: str | None = None
_runtime: dict | None = None
_lock = threading.RLock()
_status = "idle"


def _select_runtime():
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device, f"cuda ({torch.cuda.get_device_name(0)}, float32)"
    return torch.device("cpu"), "cpu (float32)"


def runtime_status() -> dict:
    return {
        "status": _status,
        "error": _model_error,
        "device": (_runtime or {}).get("label"),
        "cleanup_available": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
        "model": NEMOTRON_MODEL_ID,
        "languages": [{"label": label, "value": value} for label, value in NEMOTRON_LANG_CHOICES],
        "profiles": list(NEMOTRON_CHUNK_PROFILES.keys()),
        "default_profile": NEMOTRON_DEFAULT_CHUNK,
    }


def get_nemotron_model():
    global _model, _model_error, _runtime, _status
    if _model is not None:
        return _model
    if _model_error is not None:
        raise RuntimeError(_model_error)
    try:
        import torch
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

        if not torch.cuda.is_available():
            os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

        device, label = _select_runtime()
        _status = "loading"
        print(f"Loading {NEMOTRON_MODEL_ID} on {label}...")
        model = nemo_asr.models.ASRModel.from_pretrained(NEMOTRON_MODEL_ID)
        if hasattr(model, "change_decoding_strategy") and hasattr(model, "joint"):
            decoding_cfg = RNNTDecodingConfig(fused_batch_size=-1)
            decoding_cfg.greedy.use_cuda_graph_decoder = False
            model.change_decoding_strategy(decoding_cfg)
        model = model.to(dtype=torch.float32).to(device).eval()
        _model = model
        _runtime = {"device": device, "label": label}
        _status = "warming"
        warm_up_streaming()
        _status = "ready"
        print(f"Nemotron ASR loaded on {label}.")
        return _model
    except Exception as e:
        _model = None
        _status = "error"
        _model_error = (
            f"Failed to load Nemotron ASR: {e}\n\n"
            "Common causes:\n"
            "- NeMo not installed (see the repo README)\n"
            "- Missing libsndfile\n"
            "- CUDA OOM (~2GB+ VRAM for this 0.6B checkpoint)"
        )
        raise RuntimeError(_model_error)


def warm_up_streaming() -> None:
    """Pay lazy feature/encoder/decoder initialization before advertising ready.

    Disposable sessions keep warm-up audio and hypotheses out of user recordings.
    Exercise every advertised chunk shape, including first and cached steps.
    """
    import torch

    for profile in NEMOTRON_CHUNK_PROFILES:
        warmup = LiveSession(lang="auto", profile=profile)
        audio = np.zeros(warmup.chunk_samples * 2 + warmup.feature_lookahead, dtype="<i2")
        warmup.feed_pcm16(audio.tobytes())
        warmup.flush()
        del warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _configure_nemotron(_model, "auto", NEMOTRON_DEFAULT_CHUNK)


def load_model_in_background() -> None:
    global _status
    if _model is None and _model_error is None:
        _status = "loading"

    def _run():
        try:
            get_nemotron_model()
        except Exception as e:
            print(f"Nemotron failed to load: {e}")

    threading.Thread(target=_run, daemon=True, name="nemotron-load").start()


def _configure_nemotron(asr_model, target_lang: str, chunk_profile: str) -> None:
    if target_lang not in NEMOTRON_LANG_CODES:
        raise ValueError(f"Unsupported language code: {target_lang}")
    att_context_size = NEMOTRON_CHUNK_PROFILES[chunk_profile]
    if hasattr(asr_model.encoder, "set_default_att_context_size"):
        asr_model.encoder.set_default_att_context_size(att_context_size=att_context_size)
    if hasattr(asr_model, "set_inference_prompt"):
        asr_model.set_inference_prompt(target_lang)
    if hasattr(asr_model, "decoding") and hasattr(asr_model.decoding, "set_strip_lang_tags"):
        asr_model.decoding.set_strip_lang_tags(False, lang_tag_pattern=None)


def _extract_transcriptions(hypotheses) -> list[str]:
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

    hypotheses = list(hypotheses)
    if not hypotheses:
        return [""]
    if isinstance(hypotheses[0], Hypothesis):
        return [hyp.text or "" for hyp in hypotheses]
    return [str(hyp) for hyp in hypotheses]


def _split_nemotron_lang_tag(text: str) -> tuple[str, str]:
    pattern = r"<([a-z]{2}(?:-[A-Za-z]{2})?)>"
    tags = re.findall(pattern, text)
    return re.sub(r"\s+", " ", re.sub(pattern, "", text)).strip(), tags[-1] if tags else ""


def _move_cache(value, device, dtype):
    import torch

    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, (list, tuple)):
        return type(value)(_move_cache(item, device, dtype) for item in value)
    return value


def _chunk_samples(profile: str) -> int:
    """Nemotron right-context is in 80ms frames. One GPU step = (right + 1) * 80ms of 16 kHz audio."""
    right = NEMOTRON_CHUNK_PROFILES[profile][1]
    return int(TARGET_SR * (right + 1) * 0.08)


def _make_preprocessor(model):
    from omegaconf import OmegaConf

    cfg = copy.deepcopy(model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    cfg.preprocessor.normalize = "None"
    preprocessor = model.from_config_dict(cfg.preprocessor)
    return preprocessor.to(next(model.parameters()).device)


@dataclass
class LiveSession:
    """PCM queue in, one model-sized chunk per GPU step (NVIDIA mic-streaming path)."""

    lang: str = "auto"
    profile: str = NEMOTRON_DEFAULT_CHUNK
    committed: str = ""
    full_text: str = ""
    detected_lang: str = ""
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    step_num: int = 0

    def __post_init__(self) -> None:
        import torch

        if self.profile not in NEMOTRON_CHUNK_PROFILES:
            raise ValueError(f"Unsupported chunk profile: {self.profile}")
        if self.lang not in NEMOTRON_LANG_CODES:
            raise ValueError(f"Unsupported language: {self.lang}")

        model = get_nemotron_model()
        with _lock:
            _configure_nemotron(model, self.lang, self.profile)

        self._model = model
        self._pcm_lock = threading.Lock()
        self.chunk_samples = _chunk_samples(self.profile)
        self._preprocessor = _make_preprocessor(model)
        device = next(model.parameters()).device
        dtype = torch.float32
        cache_last_channel, cache_last_time, cache_last_channel_len = (
            model.encoder.get_initial_cache_state(batch_size=1)
        )
        self.cache_last_channel = _move_cache(cache_last_channel, device, dtype)
        self.cache_last_time = _move_cache(cache_last_time, device, dtype)
        self.cache_last_channel_len = _move_cache(cache_last_channel_len, device, dtype)
        self.previous_hypotheses = None
        self.previous_pred_out = None
        pcs = model.encoder.streaming_cfg.pre_encode_cache_size
        self.pre_encode_cache_size = pcs[1] if isinstance(pcs, (list, tuple)) else pcs
        num_channels = int(model.cfg.preprocessor.features)
        self.cache_pre_encode = torch.zeros(
            (1, num_channels, self.pre_encode_cache_size), device=device, dtype=dtype
        )
        # Centered STFT needs future samples; retain waveform context across chunks
        # so chunk boundaries do not become artificial silence/reflect padding.
        self.feature_lookahead = 320
        self._wave_left = np.zeros(0, dtype=np.float32)
        self._stable_since = time.monotonic()

    def add_pcm(self, data: bytes) -> None:
        if not data or len(data) < 2:
            return
        usable = len(data) - (len(data) % 2)
        samples = np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0
        with self._pcm_lock:
            if len(self.pending) + len(samples) > TARGET_SR * 15:
                raise RuntimeError("Audio processing cannot keep up. Try a faster device or shorter recording.")
            self.pending = np.concatenate([self.pending, samples])

    def has_chunk(self) -> bool:
        with self._pcm_lock:
            return len(self.pending) >= self.chunk_samples + getattr(self, "feature_lookahead", 0)

    def consume_chunks(self, max_chunks: int | None = None) -> dict:
        consumed = 0
        while max_chunks is None or consumed < max_chunks:
            with self._pcm_lock:
                if len(self.pending) < self.chunk_samples + getattr(self, "feature_lookahead", 0):
                    break
                chunk = self.pending[: self.chunk_samples + getattr(self, "feature_lookahead", 0)].copy()
                self.pending = self.pending[self.chunk_samples :]
            self._infer(chunk, final=False)
            consumed += 1
        return self.snapshot()

    def feed_pcm16(self, data: bytes) -> dict:
        self.add_pcm(data)
        return self.consume_chunks()

    def flush(self) -> dict:
        with self._pcm_lock:
            tail = self.pending
            self.pending = np.zeros(0, dtype=np.float32)
        silence = np.zeros(max(self.chunk_samples, int(TARGET_SR * 0.4)), dtype=np.float32)
        audio = np.concatenate([tail, silence]) if tail.size else silence
        rem = len(audio) % self.chunk_samples
        if rem:
            audio = np.concatenate(
                [audio, np.zeros(self.chunk_samples - rem, dtype=np.float32)]
            )
        total = len(audio)
        audio = np.pad(audio, (0, self.feature_lookahead))
        for i in range(0, total, self.chunk_samples):
            last = i + self.chunk_samples >= total
            self._infer(audio[i : i + self.chunk_samples + self.feature_lookahead], final=last)
        return self.snapshot()

    def live_span(self) -> str:
        full = self.full_text
        committed = self.committed
        if not committed:
            return full
        if full.startswith(committed):
            return full[len(committed) :].strip()
        if committed in full:
            idx = full.rfind(committed)
            return full[idx + len(committed) :].strip()
        return full.strip()

    def commit(self, force: bool = False) -> str:
        span = self.live_span().strip()
        if not span:
            return ""
        if not force:
            if time.monotonic() - self._stable_since < 0.4:
                return ""
            if len(span) < 6 and " " not in span:
                return ""
        if self.full_text.startswith(self.committed):
            self.committed = self.full_text
        else:
            self.committed = (self.committed + " " + span).strip()
        return span

    def snapshot(self) -> dict:
        return {
            "full": self.full_text,
            "live": self.live_span(),
            "detected_lang": self.detected_lang,
        }

    def _infer(self, audio: np.ndarray, final: bool) -> None:
        import torch

        with _lock, torch.inference_mode():
            self._infer_step(audio, final)

    def _infer_step(self, audio: np.ndarray, final: bool) -> None:
        import torch

        model = self._model
        device = next(model.parameters()).device
        left_samples = self._wave_left.size
        waveform = np.concatenate([self._wave_left, audio])
        audio_signal = torch.from_numpy(waveform).unsqueeze(0).to(device)
        audio_signal_len = torch.tensor([waveform.size], device=device)
        features, _ = self._preprocessor(input_signal=audio_signal, length=audio_signal_len)
        # Keep exactly the new 10 ms feature frames. The centered STFT emits an
        # extra boundary frame, which must not enter the encoder or its cache.
        begin = left_samples // 160
        frames = self.chunk_samples // 160
        features = features[:, :, begin : begin + frames]
        self._wave_left = waveform[: left_samples + self.chunk_samples][-640:].copy()
        processed_signal = torch.cat([self.cache_pre_encode, features], dim=-1)
        processed_signal_length = torch.tensor([processed_signal.size(-1)], device=device)
        self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size :].clone()

        (
            self.previous_pred_out,
            transcribed_texts,
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
            self.previous_hypotheses,
        ) = model.conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=self.cache_last_channel,
            cache_last_time=self.cache_last_time,
            cache_last_channel_len=self.cache_last_channel_len,
            keep_all_outputs=final,
            previous_hypotheses=self.previous_hypotheses,
            previous_pred_out=self.previous_pred_out,
            drop_extra_pre_encoded=None,
            return_transcription=True,
        )
        raw = _extract_transcriptions(transcribed_texts)[0]
        text, lang = _split_nemotron_lang_tag(raw)
        if text != self.full_text:
            self._stable_since = time.monotonic()
        self.full_text = text
        if lang:
            self.detected_lang = lang
        self.step_num += 1


def load_dotenv_files() -> None:
    import dotenv

    here = Path(__file__).resolve()
    for env_file in (
        here.parents[2] / ".env",
        here.parents[3] / ".env",
        here.parents[4] / ".env",
        here.parents[4] / "src" / "wisper_flow_clone" / ".env",
        Path.cwd() / ".env",
    ):
        dotenv.load_dotenv(env_file)
