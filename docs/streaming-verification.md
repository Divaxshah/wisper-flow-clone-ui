# Streaming verification — 12 September 2026

## Fixed failure modes

- Browser capture previously started without waiting for `started`, and a release during microphone startup could leave recording active. The explicit idle/starting/listening/stopping lifecycle now handles both.
- Ending could flush while a background inference step was still mutating the same cache. Commit and flush now join the audio drain first.
- Every recording reused sentence IDs from 1, overwriting text. Browser recording prefixes and monotonic server IDs prevent collisions.
- Disconnect cancellation could leave the model slot held. Shielded cleanup waits for non-abandoned inference work before releasing it.
- Per-chunk spectrogram extraction introduced artificial waveform boundaries and an extra STFT frame into the pre-encoder cache. The corrected path retains 40 ms of left waveform context, waits for 20 ms of right context, and slices exactly the new 10 ms feature frames. Encoder and decoder caches continue to carry streaming history; the full recording is never retranscribed on each step.
- Language tags embedded between sentences now get stripped as well as trailing tags.

## Verification performed

- Nine backend regression tests: partials before end, three recordings on one connection, queued-audio/end ordering, short pause commits, duplicate start, cleanup failure, disconnect recovery, busy-client isolation, startup failure, PCM backlog limits, and language-tag stripping (some tests cover multiple cases).
- TypeScript check and production Vite build.
- Headless Chrome with its synthetic microphone: three click-to-record sessions, a Space release during delayed startup, 18 PCM frames observed during recording, unique transcript rows, original-text toggle, and no page errors. Desktop and 390 px mobile screenshots inspected; no horizontal overflow.
- Browser startup-error and mid-recording disconnect tests: received partial text preserved, controls unlocked, and a subsequent recording succeeded.
- Real cached `nvidia/nemotron-3.5-asr-streaming-0.6b` with NeMo `3.1.0+6ad981f26`, PyTorch `2.14.0+cu130`, CPU with four threads. Two independent Balanced sessions processed the public 11-second JFK speech sample. Each produced 18 changing partials before finalization and the complete wording, in approximately 3.6 seconds of processing after model loading. This is faster-than-real-time file feeding, not a measured microphone-to-screen latency benchmark.
- NVIDIA's installed reference `CacheAwareStreamingAudioBuffer` was used as a comparison. Before the feature fix, the same sample produced only “So my.”; afterward it produced the full two-sentence speech, matching the reference wording.

## Remaining practical limits

This is a focused reliability improvement, not a claim of Wispr Flow or Gboard accuracy parity. Real microphone/accent/noise coverage, GPU performance, and live OpenRouter cleanup quality have not been benchmarked. Pause detection remains energy based, cleanup cannot revise earlier committed spans, and one recording per model process is supported. Browser transcripts are in-memory, and failed connections do not replay lost audio. See the main README for setup and repeatable tests.

## Sources

- [NVIDIA Nemotron model guidance](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [NVIDIA cache-aware streaming example](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py)
- [Wispr Flow formatting and backtrack](https://docs.wisprflow.ai/articles/5373093536-how-do-i-use-smart-formatting-and-backtrack)
- [Google Rambler help](https://support.google.com/gboard/answer/17468539)
- [Public speech test sample](https://github.com/ggerganov/whisper.cpp/blob/master/samples/jfk.wav)
