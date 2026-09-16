from __future__ import annotations

"""PeerCoordBehaviour — canal de coordinación NPC↔NPC (Fase 12).

CyclicBehaviour separado del BDIBehaviour (que solo habla con Unity y con el
LLMPlanningAgent). Escucha mensajes XMPP con metadata
`{"protocol": "npc-coord"}` de otros NPCAgent y:

  - RECEPTOR de `query-if`  → evalúa contra SUS PROPIAS creencias/perfil y
    responde `inform`. Nunca escribe nada en el emisor.
  - RECEPTOR de `request`   → aplica los guardarraíles (profundidad, condición
    parseable, dedup, límite de peticiones, capability) y responde
    `agree`/`refuse`. Si acepta, adopta un Goal `achieve_deliver_to_peer` por
    el MISMO camino que un trigger reactivo (`_adopt_goal_from_trigger`) — el
    peticionario NUNCA inyecta un plan, solo pide.
  - RECEPTOR de `inform`/`agree`/`refuse` → son RESPUESTAS a algo que ESTE NPC
    preguntó/pidió — se guardan en `agent.peer_results[conversation_id]` para
    que el Waiter de la acción ASL correspondiente (`.ask_peer`/
    `.request_peer`, en bdi.py) las recoja.
  - RECEPTOR de `inform-done`/`failure` → notificación asíncrona de que un
    goal delegado (en el receptor original) terminó — escribe belief
    `peer_done`/`peer_failed` (+ `peer_item_available` si `inform-done` trae
    entrega física) directamente, sin correlación por conversation_id (no hay
    Waiter esperando: `.await_peer` polla la creencia, no el dict).

Todo mensaje malformado o performativa/predicado fuera de la allowlist se
loguea y traza — NUNCA excepción que tumbe el behaviour cíclico.
"""

import json
import logging

import spade

from protocol.peer_messages import PeerMessage, PROTOCOL_METADATA, try_parse_peer_message
from utils.trace_logger import trace as _trace, TraceLogger

log = logging.getLogger(__name__)


def _npc_id_from_jid(jid: str) -> str:
    """'npc_001@localhost/resource' -> 'npc_001'."""
    return str(jid).split("@")[0].split("/")[0]


class PeerCoordBehaviour(spade.behaviour.CyclicBehaviour):
    async def run(self) -> None:
        agent = self.agent
        msg = await self.receive(timeout=2)
        if not msg:
            return

        sender_jid = str(msg.sender)
        sender_id = _npc_id_from_jid(sender_jid)

        try:
            payload = json.loads(msg.body)
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("[PEER:%s] Mensaje no-JSON de %s: %s", agent.npc_id, sender_jid, exc)
            _trace("peer_msg_malformed", npc_id=agent.npc_id, from_npc=sender_id, reason="invalid_json")
            return

        pmsg, err = try_parse_peer_message(payload)
        tl = TraceLogger.get()
        if tl is not None:
            tl.metrics.npc(agent.npc_id).peer_msgs_received += 1
        if pmsg is None:
            log.warning("[PEER:%s] Mensaje inválido de %s: %s", agent.npc_id, sender_jid, err)
            _trace("peer_msg_malformed", npc_id=agent.npc_id, from_npc=sender_id, reason=err)
            return

        _trace(
            "peer_msg_in", npc_id=agent.npc_id, from_npc=sender_id,
            performative=pmsg.performative, conversation_id=pmsg.conversation_id,
        )

        try:
            if pmsg.performative == "query-if":
                await self._handle_query(pmsg, sender_jid, sender_id)
            elif pmsg.performative == "request":
                await self._handle_request(pmsg, sender_jid, sender_id)
            elif pmsg.performative in ("inform", "agree", "refuse"):
                # Respuesta a algo que ESTE NPC preguntó/pidió — el Waiter de
                # la acción ASL correspondiente la recoge de este dict.
                agent.peer_results[pmsg.conversation_id] = pmsg
            elif pmsg.performative == "inform-done":
                agent.beliefs.apply_peer_done(
                    sender_id, pmsg.goal_sig or "",
                    item_id=pmsg.item, qty=pmsg.qty, x=pmsg.x, y=pmsg.y,
                )
                _trace(
                    "peer_request_completed", npc_id=agent.npc_id,
                    from_npc=sender_id, goal_sig=pmsg.goal_sig,
                )
                if tl is not None:
                    tl.metrics.npc(agent.npc_id).peer_requests_completed += 1
                    if pmsg.item is not None:
                        tl.metrics.npc(agent.npc_id).items_received += 1
            elif pmsg.performative == "failure":
                agent.beliefs.apply_peer_failed(
                    sender_id, pmsg.goal_sig or "", pmsg.reason or "unknown",
                )
                _trace(
                    "peer_request_completed", npc_id=agent.npc_id,
                    from_npc=sender_id, goal_sig=pmsg.goal_sig,
                    failed=True, reason=pmsg.reason,
                )
                if tl is not None:
                    tl.metrics.npc(agent.npc_id).peer_requests_failed += 1
        except Exception:
            # Regla de no-silencio pero SIN tumbar el behaviour cíclico: un
            # fallo al procesar un mensaje de coordinación no debe cortar el
            # canal para el resto de la sesión.
            log.exception(
                "[PEER:%s] Error procesando mensaje de %s (performativa=%s)",
                agent.npc_id, sender_jid, pmsg.performative,
            )
            _trace(
                "peer_msg_handler_error", npc_id=agent.npc_id,
                from_npc=sender_id, performative=pmsg.performative,
            )

    # ------------------------------------------------------------------
    # query-if → inform
    # ------------------------------------------------------------------

    async def _handle_query(self, pmsg: PeerMessage, sender_jid: str, sender_id: str) -> None:
        agent = self.agent
        pred = pmsg.pred
        args = pmsg.args or []

        if pred == "can_make":
            item = str(args[0]).lower() if args else ""
            value = agent.beliefs.has("recipe_output", None, None, item, None)
        elif pred == "has_item":
            item = str(args[0]).lower() if args else ""
            rows = agent.beliefs.query("has_item", item)
            value = rows[0][1] if rows else 0
        elif pred == "knows_zone":
            zone = str(args[0]).lower() if args else ""
            value = agent.beliefs.has("zone_center", zone, None, None)
        elif pred == "busy":
            value = agent.intention is not None
        else:
            # No debería llegar aquí (try_parse_peer_message ya valida contra
            # la allowlist) — defensivo, sin excepción.
            log.warning("[PEER:%s] query-if con predicado inesperado: %s", agent.npc_id, pred)
            return

        await self._reply(sender_jid, PeerMessage(
            performative="inform", conversation_id=pmsg.conversation_id,
            pred=pred, args=args, value=value,
        ))
        _trace(
            "peer_query", npc_id=agent.npc_id, from_npc=sender_id,
            pred=pred, args=args, value=value,
        )

    # ------------------------------------------------------------------
    # request → agree | refuse (+ adopción del goal si agree)
    # ------------------------------------------------------------------

    async def _handle_request(self, pmsg: PeerMessage, sender_jid: str, sender_id: str) -> None:
        from config import settings
        from llm.canonical import call_args_from_condition

        agent = self.agent
        goal_sig = pmsg.goal_sig or ""
        condition = pmsg.condition or ""

        if pmsg.depth > settings.peer_max_depth:
            await self._refuse(pmsg, sender_jid, sender_id, "max_depth")
            return
        if not condition:
            await self._refuse(pmsg, sender_jid, sender_id, "no_success_condition")
            return

        # MVP: solo se sabe delegar la familia has_item(Item, Qty) — es la
        # única con variante `delegate` (family_plan.py). Cualquier otra
        # condición se rechaza explícitamente en vez de fingir soporte.
        args = call_args_from_condition(condition)
        if len(args) != 2 or not isinstance(args[1], (int, float)):
            await self._refuse(pmsg, sender_jid, sender_id, "unsupported_condition")
            return
        item, qty = str(args[0]).lower(), args[1]

        # Dedup: ¿ya hay un goal activo cumpliendo ESTA petición exacta
        # (mismo peticionario, mismo goal_sig, mismo binding)? → agree
        # idempotente, no se duplica el compromiso.
        already = any(
            getattr(g, "goal_source", None) == "peer"
            and getattr(g, "peer_requester_jid", None) == sender_jid
            and getattr(g, "peer_origin_sig", None) == goal_sig
            and list(getattr(g, "call_args", []) or []) == [sender_id, item, qty]
            for g in agent.goals
        )
        if already:
            await self._agree(pmsg, sender_jid, sender_id, dedup=True)
            return

        # Fase 17t: la 17s hace que una petición nueva replanifique desde cero en vez
        # de heredar el nodo FAILED; sin tope, peticionario y receptor repetían el
        # mismo encargo fallido (~3 min por ciclo) hasta el timeout de la sesión.
        from llm.canonical import goal_identity_key
        failed_key = goal_identity_key("achieve_deliver_to_peer", [sender_id, item, qty])
        failed_before = (getattr(agent, "failed_peer_deliveries", None) or {}).get(failed_key, 0)
        if failed_before >= settings.peer_max_failed_attempts:
            await self._refuse(pmsg, sender_jid, sender_id, "already_failed")
            return

        tl = TraceLogger.get()
        accepted_so_far = tl.metrics.npc(agent.npc_id).peer_requests_accepted if tl is not None else 0
        if accepted_so_far >= settings.max_peer_requests:
            await self._refuse(pmsg, sender_jid, sender_id, "too_many_requests")
            return

        # Capability check ligero: ¿tengo ALGÚN camino para Item (spawn propio
        # o receta propia)? Si no, no finjo que puedo — igual que la familia
        # canónica evalúa el mundo antes de comprometerse.
        has_spawn = agent.beliefs.has("item_spawn", item, None)
        has_recipe = agent.beliefs.has("recipe_output", None, None, item, None)
        if not has_spawn and not has_recipe:
            await self._refuse(pmsg, sender_jid, sender_id, "no_capability")
            return

        await self._agree(pmsg, sender_jid, sender_id, dedup=False)

        agent._adopt_goal_from_trigger(
            "achieve_deliver_to_peer",
            condition=f"delivered_to_peer({sender_id}, {item}, {qty})",
            origin="peer",
            call_args=[sender_id, item, qty],
            peer_requester_jid=sender_jid,
            peer_request_depth=pmsg.depth,
            peer_origin_sig=goal_sig,
        )

    async def _agree(self, pmsg: PeerMessage, sender_jid: str, sender_id: str, *, dedup: bool) -> None:
        agent = self.agent
        await self._reply(sender_jid, PeerMessage(
            performative="agree", conversation_id=pmsg.conversation_id, goal_sig=pmsg.goal_sig,
        ))
        _trace(
            "peer_request_accepted", npc_id=agent.npc_id, from_npc=sender_id,
            goal_sig=pmsg.goal_sig, condition=pmsg.condition, dedup=dedup,
        )
        tl = TraceLogger.get()
        if tl is not None and not dedup:
            tl.metrics.npc(agent.npc_id).peer_requests_accepted += 1

    async def _refuse(self, pmsg: PeerMessage, sender_jid: str, sender_id: str, reason: str) -> None:
        agent = self.agent
        await self._reply(sender_jid, PeerMessage(
            performative="refuse", conversation_id=pmsg.conversation_id,
            goal_sig=pmsg.goal_sig, reason=reason,
        ))
        _trace(
            "peer_request_refused", npc_id=agent.npc_id, from_npc=sender_id,
            goal_sig=pmsg.goal_sig, reason=reason,
        )
        tl = TraceLogger.get()
        if tl is not None:
            tl.metrics.npc(agent.npc_id).peer_requests_refused += 1

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    async def _reply(self, to_jid: str, pmsg: PeerMessage) -> None:
        agent = self.agent
        msg = spade.message.Message(
            to=to_jid, metadata=dict(PROTOCOL_METADATA), body=json.dumps(pmsg.to_dict()),
        )
        # await directo (no fire-and-forget): estamos en un método async, y
        # una respuesta perdida por no esperar a que el scheduler la procese
        # es exactamente el tipo de fallo silencioso que este proyecto evita.
        await self.send(msg)
        tl = TraceLogger.get()
        if tl is not None:
            tl.metrics.npc(agent.npc_id).peer_msgs_sent += 1
