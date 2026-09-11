"""Nemotron cache-aware streaming ASR for live microphone sessions."""

from __future__ import annotations

import os
import re
import threading
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
NEMOTRON_CHUNK_PROFILES = {
    "Lowest latency": [56, 0],
    "Fast": [56, 1],
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
        _status = "ready"
        print(f"Nemotron ASR loaded on {label}.")
        return _model
    except Exception as e:
        _status = "error"
        _model_error = (
            f"Failed to load Nemotron ASR: {e}\n\n"
            "Common causes:\n"
            "- NeMo not installed (see the repo README)\n"
            "- Missing libsndfile\n"
            "- CUDA OOM (~2GB+ VRAM for this 0.6B checkpoint)"
        )
        raise RuntimeError(_model_error)


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


def _drop_extra_pre_encoded(asr_model, step_num: int, pad_and_drop_preencoded: bool) -> int:
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def _split_nemotron_lang_tag(text: str) -> tuple[str, str]:
    match = re.search(r"\s*<([a-z]{2}(?:-[A-Za-z]{2})?)>\s*$", text)
    if not match:
        return text.strip(), ""
    return text[: match.start()].strip(), match.group(1)


def _move_cache(value, device, dtype):
    import torch

    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, (list, tuple)):
        return type(value)(_move_cache(item, device, dtype) for item in value)
    return value


@dataclass
class LiveSession:
    """One WebSocket connection's cache-aware streaming state."""

    lang: str = "auto"
    profile: str = NEMOTRON_DEFAULT_CHUNK
    committed: str = ""
    full_text: str = ""
    detected_lang: str = ""
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    step_num: int = 0
    _stream_started: bool = False

    def __post_init__(self) -> None:
        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        if self.profile not in NEMOTRON_CHUNK_PROFILES:
            raise ValueError(f"Unsupported chunk profile: {self.profile}")
        if self.lang not in NEMOTRON_LANG_CODES:
            raise ValueError(f"Unsupported language: {self.lang}")

        model = get_nemotron_model()
        with _lock:
            _configure_nemotron(model, self.lang, self.profile)

        self._model = model
        self._buffer = CacheAwareStreamingAudioBuffer(
            model=model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )
        cache_last_channel, cache_last_time, cache_last_channel_len = (
            model.encoder.get_initial_cache_state(batch_size=1)
        )
        device = next(model.parameters()).device
        dtype = torch.float32
        self.cache_last_channel = _move_cache(cache_last_channel, device, dtype)
        self.cache_last_time = _move_cache(cache_last_time, device, dtype)
        self.cache_last_channel_len = _move_cache(cache_last_channel_len, device, dtype)
        self.previous_hypotheses = None
        self.previous_pred_out = None

    def feed_pcm16(self, data: bytes) -> dict:
        if not data or len(data) < 2:
            return self.snapshot()
        usable = len(data) - (len(data) % 2)
        samples = np.frombuffer(data[:usable], dtype=np.int16).astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, samples])
        min_samples = int(TARGET_SR * 0.08)
        if len(self.pending) < min_samples:
            return self.snapshot()
        audio = self.pending
        self.pending = np.zeros(0, dtype=np.float32)
        self._append(audio)
        self._step(final=False)
        return self.snapshot()

    def flush(self) -> dict:
        tail = self.pending
        self.pending = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(TARGET_SR * 0.4), dtype=np.float32)
        self._append(np.concatenate([tail, silence]) if tail.size else silence)
        self._step(final=True)
        return self.snapshot()

    def live_span(self) -> str:
        full = self.full_text
        committed = self.committed
        if not committed:
            return full
        if full.startswith(committed):
            return full[len(committed) :].strip()
        # Hypothesis was revised behind the commit point.
        if committed in full:
            idx = full.rfind(committed)
            return full[idx + len(committed) :].strip()
        return full.strip()

    def commit(self) -> str:
        """Lock the current live span as a pause-delimited sentence. Returns the raw span."""
        span = self.live_span().strip()
        if not span:
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

    def _append(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        stream_id = 0 if self._stream_started else -1
        self._buffer.append_audio(audio, stream_id=stream_id)
        self._stream_started = True

    def _step(self, final: bool) -> None:
        import torch

        if self._buffer.buffer is None:
            return
        model = self._model
        model_device = next(model.parameters()).device
        stream_dtype = torch.float32
        with _lock:
            for chunk_audio, chunk_lengths in self._buffer:
                keep_all = final and self._buffer.is_buffer_empty()
                with torch.inference_mode():
                    chunk_audio = chunk_audio.to(device=model_device, dtype=stream_dtype)
                    chunk_lengths = chunk_lengths.to(device=model_device)
                    (
                        self.previous_pred_out,
                        transcribed_texts,
                        self.cache_last_channel,
                        self.cache_last_time,
                        self.cache_last_channel_len,
                        self.previous_hypotheses,
                    ) = model.conformer_stream_step(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=self.cache_last_channel,
                        cache_last_time=self.cache_last_time,
                        cache_last_channel_len=self.cache_last_channel_len,
                        keep_all_outputs=keep_all,
                        previous_hypotheses=self.previous_hypotheses,
                        previous_pred_out=self.previous_pred_out,
                        drop_extra_pre_encoded=_drop_extra_pre_encoded(model, self.step_num, False),
                        return_transcription=True,
                    )
                raw = _extract_transcriptions(transcribed_texts)[0]
                text, lang = _split_nemotron_lang_tag(raw)
                self.full_text = text
                if lang:
                    self.detected_lang = lang
                self.step_num += 1


def load_dotenv_files() -> None:
    import dotenv

    here = Path(__file__).resolve()
    for env_file in (
        here.parents[2] / ".env",
        here.parents[4] / ".env",
        here.parents[4] / "src" / "wisper_flow_clone" / ".env",
        Path.cwd() / ".env",
    ):
        dotenv.load_dotenv(env_file)
