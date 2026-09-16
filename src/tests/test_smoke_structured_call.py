"""Smoke tests — call_llm (structured opcional según firma del callable)."""

from __future__ import annotations

import pytest

from llm.pipeline.structured_call import call_llm, _accepts_schema


def test_accepts_schema_with_keyword():
    async def f(user, system, *, schema=None):  # noqa: ANN001
        return ""
    assert _accepts_schema(f) is True


def test_accepts_schema_with_var_keyword():
    async def f(user, system, **kw):  # noqa: ANN001
        return ""
    assert _accepts_schema(f) is True


def test_accepts_schema_two_arg_false():
    async def f(user, system):  # noqa: ANN001
        return ""
    assert _accepts_schema(f) is False


@pytest.mark.asyncio
async def test_call_llm_uses_schema_when_supported():
    captured = {}

    async def f(user, system, *, schema=None):  # noqa: ANN001
        captured["schema"] = schema
        return "ok"

    out = await call_llm(f, ("u", "s"), schema=str)
    assert out == "ok"
    assert captured["schema"] is str


@pytest.mark.asyncio
async def test_call_llm_falls_back_for_two_arg_mock():
    async def f(user, system):  # noqa: ANN001 — mock de test de 2 args
        return f"{user}|{system}"

    # Aunque se pase schema, el callable de 2 args recibe solo (user, system).
    out = await call_llm(f, ("u", "s"), schema=str)
    assert out == "u|s"


@pytest.mark.asyncio
async def test_call_llm_no_schema_is_plain():
    async def f(user, system, *, schema=None):  # noqa: ANN001
        return "plain" if schema is None else "structured"

    assert await call_llm(f, ("u", "s"), schema=None) == "plain"
