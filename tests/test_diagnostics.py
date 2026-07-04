from rag_eval.diagnostics import _key_check


def test_no_key_required_passes():
    c = _key_check("embedding key", "bge", None)
    assert c.ok is True
    assert "no API key" in c.detail


def test_key_present_passes(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    c = _key_check("answer llm key", "groq", "GROQ_API_KEY")
    assert c.ok is True


def test_key_missing_fails(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = _key_check("answer llm key", "openai", "OPENAI_API_KEY")
    assert c.ok is False
    assert "NOT set" in c.detail
