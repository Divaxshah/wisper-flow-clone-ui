"""Protocol regression tests. Fake inference deliberately overlaps incoming messages."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend import app as server
from backend import asr


class FakeSession:
    instances = []

    def __init__(self, lang='auto', profile='Balanced'):
        self.pending = 0
        self.full = ''
        self.committed = ''
        self.active = False
        self.steps = 0
        self.instances.append(self)

    def add_pcm(self, data):
        self.pending += len(data) // 2

    def has_chunk(self):
        return self.pending > 0

    def consume_chunks(self, max_chunks=None):
        assert not self.active
        self.active = True
        time.sleep(.02)
        self.pending -= 1
        self.steps += 1
        self.full += ' word'
        self.active = False
        return self.snapshot()

    def snapshot(self):
        return {'full': self.full.strip(), 'live': self.full[len(self.committed):].strip(), 'detected_lang': 'en-US'}

    def commit(self, force=False):
        assert not self.active
        text = self.full[len(self.committed):].strip()
        self.committed = self.full
        return text

    def flush(self):
        assert not self.active, 'flush raced with inference'
        assert self.pending == 0, 'end lost queued audio'
        return self.snapshot()


@pytest.fixture
def client(monkeypatch):
    FakeSession.instances = []
    monkeypatch.setattr(asr, 'LiveSession', FakeSession)
    monkeypatch.setattr(server, 'runtime_status', lambda: {'status': 'ready'})
    return TestClient(server.app)


def start(ws, cleanup=False):
    ws.send_json({'type': 'start', 'cleanup': cleanup})
    assert ws.receive_json()['type'] == 'started'


def until(ws, kind):
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event['type'] == kind:
            return events


def test_streams_before_end_and_restarts_without_id_collision(client):
    with client.websocket_connect('/ws/transcribe') as ws:
        assert ws.receive_json()['type'] == 'ready'
        ids = []
        for _ in range(3):
            start(ws)
            ws.send_bytes(b'\x00\x01' * 3)
            first = ws.receive_json()
            assert first['type'] == 'partial'
            assert first['live'] == 'word'  # one partial per inference step
            ws.send_json({'type': 'end'})
            events = until(ws, 'ended')
            ids.extend(e['id'] for e in events if e['type'] == 'commit')
        assert ids == [1, 2, 3]
        assert all(s.steps == 3 for s in FakeSession.instances)


def test_commit_waits_for_pending_audio_and_keeps_short_utterance(client):
    with client.websocket_connect('/ws/transcribe') as ws:
        ws.receive_json()
        start(ws)
        ws.send_bytes(b'\x00\x01')
        ws.send_json({'type': 'commit'})
        events = until(ws, 'cleaned')
        assert events[-1]['raw'] == 'word'
        ws.send_json({'type': 'end'})
        until(ws, 'ended')


def test_cleanup_failure_preserves_raw_and_allows_restart(client, monkeypatch):
    def fail(*args):
        raise RuntimeError('provider failed')
    monkeypatch.setattr(server, 'cleanup_span', fail)
    with client.websocket_connect('/ws/transcribe') as ws:
        ws.receive_json()
        start(ws, cleanup=True)
        ws.send_bytes(b'\x00\x01')
        ws.send_json({'type': 'end'})
        events = until(ws, 'ended')
        assert next(e for e in events if e['type'] == 'polished')['cleaned'] == 'word'
        start(ws)
        ws.send_json({'type': 'end'})
        until(ws, 'ended')


def test_duplicate_start_does_not_replace_session(client):
    with client.websocket_connect('/ws/transcribe') as ws:
        ws.receive_json()
        start(ws)
        ws.send_json({'type': 'start'})
        assert ws.receive_json()['type'] == 'warning'
        assert len(FakeSession.instances) == 1
        ws.send_json({'type': 'end'})
        until(ws, 'ended')


def test_disconnect_releases_model_after_inference(client):
    with client.websocket_connect('/ws/transcribe') as ws:
        ws.receive_json()
        start(ws)
        ws.send_bytes(b'\x00\x01' * 2)
        ws.receive_json()
    assert not server.model_slot.locked()
    with client.websocket_connect('/ws/transcribe') as ws:
        ws.receive_json()
        start(ws)
        ws.send_json({'type': 'end'})
        until(ws, 'ended')


def test_pcm_queue_emits_one_step_and_rejects_backlog():
    # Bypass GPU initialization to exercise the real PCM queue.
    session = object.__new__(asr.LiveSession)
    import numpy as np
    session.pending = np.zeros(0, dtype=np.float32)
    session._pcm_lock = threading.Lock()
    session.chunk_samples = 2
    session.snapshot = lambda: {}
    chunks = []
    session._infer = lambda audio, final: chunks.append(audio)
    session.add_pcm(b'\x00\x40' * 4)
    session.consume_chunks(1)
    assert len(chunks) == 1
    assert session.has_chunk()
    assert chunks[0].tolist() == [.5, .5]
    with pytest.raises(RuntimeError, match='cannot keep up'):
        session.add_pcm(b'\0\0' * (16000 * 16))


def test_busy_client_cannot_change_active_model(client):
    with client.websocket_connect('/ws/transcribe') as first:
        first.receive_json()
        start(first)
        with client.websocket_connect('/ws/transcribe') as second:
            second.receive_json()
            second.send_json({'type': 'start', 'language': 'hi-IN'})
            assert 'Another recording' in second.receive_json()['message']
        assert len(FakeSession.instances) == 1
        first.send_json({'type': 'end'})
        until(first, 'ended')


def test_initialization_failure_releases_slot_for_retry(client, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError('initialization failed')
    monkeypatch.setattr(asr, 'LiveSession', fail)
    with client.websocket_connect('/ws/transcribe') as ws:
        ws.receive_json()
        ws.send_json({'type': 'start'})
        assert ws.receive_json()['type'] == 'error'
        assert not server.model_slot.locked()
        monkeypatch.setattr(asr, 'LiveSession', FakeSession)
        start(ws)
        ws.send_json({'type': 'end'})
        until(ws, 'ended')


def test_language_tags_are_removed_inside_and_after_text():
    assert asr._split_nemotron_lang_tag('Hello. <en-US> Namaste. <hi-IN>') == ('Hello. Namaste.', 'hi-IN')


def test_cleanup_arrives_live_and_stop_does_not_repeat_unchanged_work(client, monkeypatch):
    calls = []
    def clean(raw):
        calls.append(raw)
        return raw
    monkeypatch.setattr(server, 'cleanup_span', clean)
    with client.websocket_connect('/ws/transcribe') as ws:
        ws.receive_json()
        for recording in range(2):
            start(ws, cleanup=True)
            ids = []
            for pause in range(3):
                ws.send_bytes(b'\0\1')
                ws.send_json({'type': 'commit'})
                events = until(ws, 'polished')
                ids.append(next(e for e in events if e['type'] == 'commit')['id'])
                result = events[-1]
                assert result['ids'] == ids
                assert result['raw'] == ' '.join(['word'] * (pause + 1))
                assert result['cleaned'] == result['raw']
                assert all(e['type'] != 'ended' for e in events)
            ws.send_json({'type': 'end'})
            events = until(ws, 'ended')
            assert not any(e['type'] == 'polished' for e in events)
    assert calls == ['word', 'word word', 'word word word'] * 2


def test_slow_cleanup_does_not_block_audio_and_coalesces_newer_pauses(client, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []
    def clean(raw):
        calls.append(raw)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return raw
    monkeypatch.setattr(server, 'cleanup_span', clean)
    try:
        with client.websocket_connect('/ws/transcribe') as ws:
            ws.receive_json()
            start(ws, cleanup=True)
            ws.send_bytes(b'\0\1')
            ws.send_json({'type': 'commit'})
            until(ws, 'polishing')
            assert entered.wait(2)
            for _ in range(2):
                ws.send_bytes(b'\0\1')
                ws.send_json({'type': 'commit'})
                events = until(ws, 'commit')
                assert any(e['type'] == 'partial' for e in events)
            release.set()
            events = until(ws, 'polished')
            assert events[-1]['raw'] == 'word word word'
            assert events[-1]['revision'] == 3
            assert calls == ['word', 'word word word']
            ws.send_json({'type': 'end'})
            until(ws, 'ended')
    finally:
        release.set()


def test_warmup_exercises_disposable_sessions_for_every_profile(monkeypatch):
    import sys
    from types import SimpleNamespace
    sessions = []
    restored = []
    class WarmSession:
        def __init__(self, lang, profile):
            self.lang, self.profile = lang, profile
            self.chunk_samples = 4
            self.feature_lookahead = 2
            self.audio = None
            self.flushed = False
            sessions.append(self)
        def feed_pcm16(self, audio):
            self.audio = audio
        def flush(self):
            self.flushed = True
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(asr, 'LiveSession', WarmSession)
    monkeypatch.setattr(asr, '_configure_nemotron', lambda model, lang, profile: restored.append((lang, profile)))
    asr.warm_up_streaming()
    assert [s.profile for s in sessions] == list(asr.NEMOTRON_CHUNK_PROFILES)
    assert all(s.flushed and s.audio == bytes(20) for s in sessions)
    assert restored == [('auto', asr.NEMOTRON_DEFAULT_CHUNK)]


def test_timing_logs_and_cleanup_rejection_reason(client, monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO, logger='uvicorn.error')
    def reject(raw):
        raise server.CleanupValidationError('Cleanup introduced unsupported or repeated words.')
    monkeypatch.setattr(server, 'cleanup_span', reject)
    with client.websocket_connect('/ws/transcribe') as ws:
        ws.receive_json()
        start(ws, cleanup=True)
        ws.send_bytes(b'\0\x40')
        ws.send_json({'type': 'end'})
        events = until(ws, 'ended')
    assert 'stage=first_audio' in caplog.text
    assert 'stage=first_signal' in caplog.text
    assert 'stage=first_text' in caplog.text
    assert 'step_ms=' in caplog.text
    assert 'cleanup_fallback' in caplog.text
    assert 'revision=1' in caplog.text
