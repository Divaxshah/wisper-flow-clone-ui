# Streaming verification — 12 September 2026

## Fixed failure modes

- Ending cannot flush while a background inference step is mutating the same cache. Commit and flush join the audio drain first.
- Sentence IDs remain monotonic for the lifetime of a connection, preventing collisions across repeated recordings.
- Disconnect cancellation cannot leave the model slot held. Shielded cleanup waits for non-abandoned inference work before releasing it.
- Feature extraction retains 40 ms of left waveform context, waits for 20 ms of right context, and slices exactly the new 10 ms feature frames. Encoder and decoder caches continue to carry streaming history.
- Language tags embedded between sentences are stripped along with trailing tags.

## Verification performed

- Backend regression tests cover partials before end, repeated recordings, queued-audio/end ordering, short pause commits, duplicate starts, cleanup failure, disconnect recovery, busy-client isolation, startup failure, PCM backlog limits, and language-tag stripping.
- Real cached `nvidia/nemotron-3.5-asr-streaming-0.6b` sessions processed the public 11-second JFK speech sample with changing partials before finalization and complete final wording.
- NVIDIA's `CacheAwareStreamingAudioBuffer` was used as a comparison for streaming output.

## Remaining practical limits

Real microphone, accent, noise, GPU performance, and live OpenRouter cleanup quality require environment-specific benchmarking. Pause detection remains energy based, cleanup cannot revise earlier committed spans, and one recording per model process is supported. Failed connections do not replay lost audio.

## Sources

- [NVIDIA Nemotron model guidance](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [NVIDIA cache-aware streaming example](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py)
- [Public speech test sample](https://github.com/ggerganov/whisper.cpp/blob/master/samples/jfk.wav)
