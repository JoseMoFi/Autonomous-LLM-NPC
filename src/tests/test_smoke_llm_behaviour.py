from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from npc.llm_behaviour import LLMBehaviour


@pytest.mark.asyncio
async def test_llm_behaviour_ignores_mismatched_thread_reply() -> None:
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    behaviour = LLMBehaviour(
        task={"task": "generate_plan", "_timeout": 1},
        result_future=future,
        llm_jid="llm_planning@localhost",
        thread_id="thread-ok",
    )

    sent = []
    replies = iter(
        [
            SimpleNamespace(
                thread="thread-other",
                body=json.dumps({"ok": True, "result": ["wrong"]}),
            ),
            SimpleNamespace(
                thread="thread-ok",
                body=json.dumps({"ok": True, "result": {"sig": "achieve_ok"}}),
            ),
        ]
    )

    async def _fake_send(msg) -> None:
        sent.append(msg)

    async def _fake_receive(timeout=None):
        return next(replies, None)

    behaviour.send = _fake_send  # type: ignore[method-assign]
    behaviour.receive = _fake_receive  # type: ignore[method-assign]

    await behaviour.run()

    assert sent[0].thread == "thread-ok"
    assert future.result() == {"sig": "achieve_ok"}