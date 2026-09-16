"""Smoke tests — adaptador de spade-llm (Fase 2).

Requiere spade-llm 0.3.0 (venv 3.12). Mockean el provider interno, así que NO
hacen llamadas de red.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from llm.providers import SpadeLLMProviderAdapter, build_provider


def _ollama_settings(**over):
    base = {
        "llm_provider": "ollama",
        "llm_model": "qwen2.5:7b",
        "llm_base_url": "http://localhost:11434",
        "llm_temperature": 0.2,
        "llm_timeout": 300,
        "gemini_api_key": "",
    }
    base.update(over)
    return SimpleNamespace(**base)


# ===========================================================================
# complete(): pasa system+user y devuelve el texto
# ===========================================================================

@pytest.mark.asyncio
async def test_complete_passes_system_and_user_and_returns_text():
    adapter = SpadeLLMProviderAdapter(model="ollama_chat/qwen2.5:7b", timeout=5)
    captured = {}

    async def fake_get_response(context, tools=None):
        captured["prompt"] = context.get_prompt()
        return "RESPUESTA"

    adapter._provider.get_response = fake_get_response
    out = await adapter.complete("hola", system="eres un asistente")

    assert out == "RESPUESTA"
    roles = [(m["role"], m["content"]) for m in captured["prompt"]]
    assert ("system", "eres un asistente") in roles
    assert ("user", "hola") in roles


@pytest.mark.asyncio
async def test_complete_without_system_only_user():
    adapter = SpadeLLMProviderAdapter(model="ollama_chat/x", timeout=5)
    captured = {}

    async def fake_get_response(context, tools=None):
        captured["prompt"] = context.get_prompt()
        return "ok"

    adapter._provider.get_response = fake_get_response
    await adapter.complete("solo user")
    roles = [m["role"] for m in captured["prompt"]]
    assert "user" in roles
    assert "system" not in roles


@pytest.mark.asyncio
async def test_complete_none_response_returns_empty_string():
    adapter = SpadeLLMProviderAdapter(model="ollama_chat/x", timeout=5)

    async def fake_get_response(context, tools=None):
        return None

    adapter._provider.get_response = fake_get_response
    assert await adapter.complete("p") == ""


# ===========================================================================
# build_provider(): lee settings y construye el adaptador
# ===========================================================================

def test_build_provider_ollama_maps_model_and_params():
    a = build_provider(_ollama_settings())
    assert a.model == "ollama_chat/qwen2.5:7b"
    assert a.timeout == 300.0
    assert a._provider.base_url == "http://localhost:11434"
    assert a._provider.temperature == 0.2


def test_build_provider_gemini_prefix_and_key():
    s = _ollama_settings(llm_provider="gemini", llm_model="gemini-1.5-flash",
                         gemini_api_key="KEY123")
    a = build_provider(s)
    assert a.model == "gemini/gemini-1.5-flash"
    assert a._provider.api_key == "KEY123"


# ===========================================================================
# Propagación de errores y timeout (sin try/except que silencie)
# ===========================================================================

@pytest.mark.asyncio
async def test_complete_propagates_provider_errors():
    adapter = SpadeLLMProviderAdapter(model="ollama_chat/x", timeout=5)

    async def boom(context, tools=None):
        raise ValueError("provider boom")

    adapter._provider.get_response = boom
    with pytest.raises(ValueError, match="provider boom"):
        await adapter.complete("p")


@pytest.mark.asyncio
async def test_complete_timeout_raises_asyncio_timeout():
    adapter = SpadeLLMProviderAdapter(model="ollama_chat/x", timeout=0.05)

    async def slow(context, tools=None):
        await asyncio.sleep(1.0)
        return "late"

    adapter._provider.get_response = slow
    with pytest.raises(asyncio.TimeoutError):
        await adapter.complete("p")


# ===========================================================================
# complete_structured() — Fase 3
# ===========================================================================

@pytest.mark.asyncio
async def test_complete_structured_returns_typed_instance():
    from llm.schemas import ParseGoalsResponse, GoalItem
    adapter = SpadeLLMProviderAdapter(model="ollama_chat/x", timeout=5)
    expected = ParseGoalsResponse(goals=[GoalItem(
        sig="achieve_have_wheat", source_index=0, priority=1.0,
        reason="r", success_condition="has_item(wheat, 1)")])

    async def fake_get_llm_response(context, tools=None, conversation_id=None, output_schema=None):
        assert output_schema is ParseGoalsResponse
        return {"text": None, "tool_calls": [], "structured": expected}

    adapter._provider.get_llm_response = fake_get_llm_response
    out = await adapter.complete_structured("p", schema=ParseGoalsResponse)
    assert out is expected
    assert out.goals[0].sig == "achieve_have_wheat"


@pytest.mark.asyncio
async def test_complete_structured_wrong_type_raises():
    from llm.schemas import ParseGoalsResponse
    adapter = SpadeLLMProviderAdapter(model="ollama_chat/x", timeout=5)

    async def fake_get_llm_response(context, tools=None, conversation_id=None, output_schema=None):
        return {"text": None, "tool_calls": [], "structured": None}

    adapter._provider.get_llm_response = fake_get_llm_response
    with pytest.raises(ValueError, match="structured output"):
        await adapter.complete_structured("p", schema=ParseGoalsResponse)
