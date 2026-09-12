"""Opt-in live cleanup check: requires OPENROUTER_API_KEY and sends public fixtures.

PYTHONPATH=backend/src python backend/tests/smoke_cleanup.py
This makes billable requests to the configured OpenRouter model.
"""
from backend.asr import load_dotenv_files
load_dotenv_files()
from backend.cleanup import cleanup_span

cases = [
    ('Fragmented question and repeated emoji', "My name is Divatsh. My name is Divaks. Can you help me solve what is one plus one\nplease\n? What's up with all these text that you are giving me\n? Smiling emoji, smiling emoji, smiling emoji", ['Divaks', 'please', 'giving me', '😊😊😊']),
    ('Topic correction', "I talk about a humanizer model. Never mind, I want to talk about a model that I was working on before. Just forget about it. Let's talk about the AI detection model.", ['talk about', 'AI detection model']),
    ('Time correction', "Let's meet at 3pm, actually no, let's make it 2pm instead.", ['2']),
    ('Meaningful actually', 'I actually enjoyed the movie, and I like the ending.', ['enjoyed', 'ending']),
    ('Names and negation', "Don't send the invoice to Mira yet. The amount is 1,450 dollars.", ['Mira', '1,450']),
    ('Question remains a question', 'Um, can you explain how the detection model works?', ['?', 'detection model']),
]
for name, raw, required in cases:
    cleaned = cleanup_span(raw)
    print(f'{name}\n  Original: {raw}\n  Cleaned:  {cleaned}', flush=True)
    assert all(phrase.casefold() in cleaned.casefold() for phrase in required), f'Review failed case: {name}'
print('Basic checks passed. Review the results for semantic fidelity; this is not a full accuracy benchmark.')
