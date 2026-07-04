import pytest

from rag_eval.config import RetryConfig
from rag_eval.resilience import call_with_retry, is_transient


class _HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"http {status_code}")
        self.status_code = status_code


class RateLimitError(Exception):
    """Name-based transient detection (no status code)."""


def test_is_transient_by_status():
    assert is_transient(_HttpError(429)) is True
    assert is_transient(_HttpError(503)) is True
    assert is_transient(_HttpError(400)) is False
    assert is_transient(_HttpError(404)) is False


def test_is_transient_by_class_name():
    assert is_transient(RateLimitError()) is True
    assert is_transient(ValueError("nope")) is False


@pytest.fixture
def fast_retry(monkeypatch):
    # Don't actually sleep during tests.
    monkeypatch.setattr("rag_eval.resilience.time.sleep", lambda _s: None)
    return RetryConfig(max_attempts=4, base_delay=0.001, max_delay=0.001)


def test_retries_then_succeeds(fast_retry):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _HttpError(429)
        return "ok"

    assert call_with_retry(flaky, fast_retry) == "ok"
    assert calls["n"] == 3


def test_non_transient_raises_immediately(fast_retry):
    calls = {"n": 0}

    def fatal():
        calls["n"] += 1
        raise _HttpError(400)

    with pytest.raises(_HttpError):
        call_with_retry(fatal, fast_retry)
    assert calls["n"] == 1  # no retries on a 400


def test_exhausts_attempts(fast_retry):
    def always_429():
        raise _HttpError(429)

    with pytest.raises(_HttpError):
        call_with_retry(always_429, fast_retry)
