"""Smoke tests — carga de settings con override local y env (1.7)."""

from __future__ import annotations

import json

import pytest

from config import Settings


_BASE = {
    "unity_host": "127.0.0.1",
    "unity_port": 7777,
    "xmpp_host": "localhost",
    "xmpp_port": 5222,
    "llm_provider": "ollama",
    "llm_model": "qwen2.5:7b",
    "llm_base_url": "http://localhost:11434",
    "llm_temperature": 0.2,
    "llm_timeout": 300,
    "llm_max_retries": 2,
}


@pytest.fixture
def settings_dir(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps(_BASE), encoding="utf-8")
    return tmp_path


def _load_fresh(path):
    Settings._instance = None  # el singleton cachea; resetear para el test
    try:
        return Settings.load(path)
    finally:
        Settings._instance = None


def test_load_without_local_uses_defaults(settings_dir):
    s = _load_fresh(settings_dir / "settings.json")
    assert s.llm_model == "qwen2.5:7b"
    assert s.gemini_api_key == ""


def test_settings_local_overrides(settings_dir):
    (settings_dir / "settings.local.json").write_text(
        json.dumps({"gemini_api_key": "SECRET123", "llm_model": "llama3.1:8b"}),
        encoding="utf-8",
    )
    s = _load_fresh(settings_dir / "settings.json")
    assert s.gemini_api_key == "SECRET123"
    assert s.llm_model == "llama3.1:8b"


def test_env_var_overrides_gemini_key(settings_dir, monkeypatch):
    monkeypatch.setenv("NPC_GEMINI_API_KEY", "FROM_ENV")
    s = _load_fresh(settings_dir / "settings.json")
    assert s.gemini_api_key == "FROM_ENV"


def test_env_var_overrides_local(settings_dir, monkeypatch):
    (settings_dir / "settings.local.json").write_text(
        json.dumps({"gemini_api_key": "FROM_LOCAL"}), encoding="utf-8"
    )
    monkeypatch.setenv("NPC_GEMINI_API_KEY", "FROM_ENV")
    s = _load_fresh(settings_dir / "settings.json")
    # El env tiene prioridad sobre settings.local.json para claves sensibles.
    assert s.gemini_api_key == "FROM_ENV"
