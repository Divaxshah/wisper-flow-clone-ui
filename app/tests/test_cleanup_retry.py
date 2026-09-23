from types import SimpleNamespace

import openai

from app.cleanup import cleanup_span


def test_validation_failure_gets_one_grounded_retry(monkeypatch):
    responses = [
        "Please send the report tomorrow.",
        "Please send the report.",
    ]
    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=responses.pop(0), refusal=None),
                )]
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setattr(openai, "OpenAI", Client)

    assert cleanup_span("Please send the report.") == "Please send the report."
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["role"] == "system"
