"""Avoid leaking env credentials into HTTP request construction or logs."""
from rsi_researcher.runtime.client import StrictDeepSeek


def test_deepseek_key_strips_windows_pipe_bom(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "\ufeffsk-test-only\r\n")
    transport = StrictDeepSeek(tmp_path)
    assert transport.key == "sk-test-only"
    assert not transport.log.exists()
