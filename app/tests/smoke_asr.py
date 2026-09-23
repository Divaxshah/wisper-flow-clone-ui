"""Run manually with a 16 kHz mono PCM16 speech WAV and installed NeMo.

PYTHONPATH=app/src python app/tests/smoke_asr.py /path/to/speech.wav
"""
import argparse
import time
import wave

import torch
from app.asr import LiveSession

parser = argparse.ArgumentParser()
parser.add_argument('wav')
parser.add_argument('--profile', default='Balanced')
parser.add_argument('--expect', help='Phrase that must occur in each final transcript')
args = parser.parse_args()
torch.set_num_threads(4)
with wave.open(args.wav, 'rb') as source:
    assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (16000, 1, 2)
    pcm = source.readframes(source.getnframes())

for recording in range(2):
    session = LiveSession(lang='en-US', profile=args.profile)
    began = time.monotonic()
    first_partial = None
    changes = 0
    previous = ''
    for offset in range(0, len(pcm), 2560):
        snap = session.feed_pcm16(pcm[offset:offset + 2560])
        if snap['full'] and snap['full'] != previous:
            first_partial = first_partial or (offset / 32000)
            previous = snap['full']
            changes += 1
            print(f'PARTIAL session={recording + 1} audio_s={offset / 32000:.2f}: {previous}', flush=True)
    assert changes > 0, 'No words appeared before end-of-stream'
    final = session.flush()['full']
    assert final, 'Final transcript is empty'
    if args.expect:
        assert args.expect.casefold() in final.casefold(), f'Missing expected phrase: {final}'
    print(f'FINAL session={recording + 1} steps={session.step_num} changes={changes} elapsed_s={time.monotonic() - began:.2f}: {final}', flush=True)
