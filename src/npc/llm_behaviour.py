from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import spade

log = logging.getLogger(__name__)


class LLMBehaviour(spade.behaviour.OneShotBehaviour):
    """
    Instanciado on-demand por BDIBehaviour.
    Envía la tarea al LLMPlanningAgent por XMPP y espera la respuesta.
    El NPC queda bloqueado solo durante este tiempo (puntual).
    """

    def __init__(
        self,
        task: dict,
        result_future: asyncio.Future,
        llm_jid: str,
        thread_id: str,
    ):
        super().__init__()
        self.task = task
        self.result_future = result_future
        self.llm_jid = llm_jid
        self.thread_id = thread_id

    async def run(self) -> None:
        # Fase 11 (T5): marca de tiempo de encolado. El LLMPlanningAgent es
        # compartido entre todos los NPCs (CyclicBehaviour serializado) — esta
        # marca permite separar "tiempo esperando cola" de "tiempo trabajando",
        # imprescindible para leer con honestidad los experimentos multi-agente.
        # setdefault: si el llamante ya la puso (reintento), no se pisa.
        self.task.setdefault("_enqueued_at", time.time())
        msg = spade.message.Message(
            to=self.llm_jid,
            thread=self.thread_id,
            body=json.dumps(self.task),
        )
        await self.send(msg)
        log.debug(
            "[LLM_BEHAV] Tarea '%s' enviada a %s (thread=%s)",
            self.task.get("task"),
            self.llm_jid,
            self.thread_id,
        )

        timeout = float(self.task.get("_timeout", 120))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        reply = None
        while True:
            remaining = max(0.0, deadline - loop.time())
            reply = await self.receive(timeout=remaining)
            if reply is None:
                break
            if reply.thread == self.thread_id:
                break
            log.debug(
                "[LLM_BEHAV] Reply ignorado para tarea '%s': thread=%r esperado=%r",
                self.task.get("task"),
                reply.thread,
                self.thread_id,
            )

        if reply is None:
            exc = TimeoutError(f"LLMPlanningAgent no respondió en {timeout}s")
            if not self.result_future.done():
                self.result_future.set_exception(exc)
            return

        try:
            payload: dict[str, Any] = json.loads(reply.body)
        except json.JSONDecodeError as exc:
            if not self.result_future.done():
                self.result_future.set_exception(exc)
            return

        if payload.get("ok"):
            if not self.result_future.done():
                self.result_future.set_result(payload["result"])
        else:
            errors = payload.get("errors", ["respuesta LLM inválida"])
            if not self.result_future.done():
                self.result_future.set_exception(
                    ValueError(f"LLM validation errors: {errors}")
                )
