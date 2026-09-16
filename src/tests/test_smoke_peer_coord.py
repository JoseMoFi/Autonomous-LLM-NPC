"""Smoke tests — coordinación NPC↔NPC (Fase 12).

Cubren:
  - protocol/peer_messages.py: parseo/validación (malformado, performativa
    desconocida, predicado fuera de allowlist, request sin condition).
  - PeerCoordBehaviour: query-if → inform; request → agree/refuse con los
    guardarraíles (depth, condición, dedup, límite, capability); request
    aceptado adopta un Goal con success_condition y goal_source="peer" SIN
    inyectar plan; inform-done/failure escriben belief directamente.
  - PeerReplyWaiter / PeerDoneWaiter: timeout se traduce en fallo trazado,
    nunca cuelga.
  - family_plan.build_has_item_family: la variante `delegate` solo aparece
    con coordination_enabled=True.
  - Registro de acciones ASL: .ask_peer/.request_peer/.await_peer/
    .deliver_to_peer solo se registran si se llama _register_peer_actions
    (coordination_enabled) — con el flag off no forman parte del vocabulario.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import agentspeak
import agentspeak.stdlib  # noqa: F401
from agentspeak import runtime as asp_runtime
import pytest

from npc.agent import NPCAgent
from npc.behaviours.bdi import BDIBehaviour, PeerDoneWaiter, PeerReplyWaiter
from npc.behaviours.peer_coord import PeerCoordBehaviour
from protocol.peer_messages import (
    PeerMessage,
    parse_peer_message,
    try_parse_peer_message,
    PeerMessageError,
)
from llm.family_plan import build_has_item_family


# ===========================================================================
# protocol/peer_messages.py — parseo y validación
# ===========================================================================

def test_parse_rejects_non_dict():
    with pytest.raises(PeerMessageError):
        parse_peer_message("not a dict")


def test_parse_rejects_unknown_performative():
    with pytest.raises(PeerMessageError):
        parse_peer_message({"performative": "shout", "conversation_id": "c1"})


def test_parse_rejects_query_predicate_outside_allowlist():
    with pytest.raises(PeerMessageError):
        parse_peer_message({
            "performative": "query-if", "conversation_id": "c1",
            "pred": "steal_secrets", "args": [],
        })


def test_parse_accepts_allowlisted_predicate():
    msg = parse_peer_message({
        "performative": "query-if", "conversation_id": "c1",
        "pred": "can_make", "args": ["flour"],
    })
    assert msg.pred == "can_make"
    assert msg.args == ["flour"]


def test_parse_rejects_request_without_condition():
    with pytest.raises(PeerMessageError):
        parse_peer_message({
            "performative": "request", "conversation_id": "c1",
            "goal_sig": "achieve_has_item",
        })


def test_try_parse_never_raises():
    msg, err = try_parse_peer_message({"performative": "bogus", "conversation_id": "c1"})
    assert msg is None
    assert err is not None


# ===========================================================================
# PeerCoordBehaviour — fixtures
# ===========================================================================

def _make_agent(npc_id: str = "npc_baker") -> NPCAgent:
    return NPCAgent(
        jid=f"{npc_id}@localhost", password="x", npc_id=npc_id,
        send_to_unity=lambda m: None, llm_planning_jid="llm@localhost",
    )


def _behaviour_for(agent: NPCAgent, incoming_payload: dict, sender: str = "npc_miller@localhost"):
    """PeerCoordBehaviour con .send mockeado y UN solo mensaje entrante."""
    b = PeerCoordBehaviour()
    b.agent = agent
    sent: list = []

    async def _fake_send(msg) -> None:
        sent.append(msg)

    replies = iter([SimpleNamespace(sender=sender, body=json.dumps(incoming_payload))])

    async def _fake_receive(timeout=None):
        return next(replies, None)

    b.send = _fake_send  # type: ignore[method-assign]
    b.receive = _fake_receive  # type: ignore[method-assign]
    return b, sent


def _sent_payload(sent: list, index: int = 0) -> dict:
    return json.loads(sent[index].body)


# ===========================================================================
# query-if → inform
# ===========================================================================

@pytest.mark.asyncio
async def test_query_can_make_replies_inform_with_true():
    agent = _make_agent()
    agent.beliefs.apply_recipe("bread_from_flour_recipe", "bakeri",
                                [{"itemId": "flour", "qty": 1}], [{"itemId": "bread", "qty": 1}])
    b, sent = _behaviour_for(agent, {
        "performative": "query-if", "conversation_id": "c1",
        "pred": "can_make", "args": ["bread"],
    })
    await b.run()
    assert len(sent) == 1
    payload = _sent_payload(sent)
    assert payload["performative"] == "inform"
    assert payload["value"] is True


@pytest.mark.asyncio
async def test_query_can_make_false_when_no_recipe():
    agent = _make_agent()
    b, sent = _behaviour_for(agent, {
        "performative": "query-if", "conversation_id": "c1",
        "pred": "can_make", "args": ["bread"],
    })
    await b.run()
    payload = _sent_payload(sent)
    assert payload["value"] is False


@pytest.mark.asyncio
async def test_malformed_message_does_not_crash_and_sends_nothing():
    agent = _make_agent()
    b = PeerCoordBehaviour()
    b.agent = agent

    async def _fake_receive(timeout=None):
        return SimpleNamespace(sender="npc_miller@localhost", body="not-json{{{")

    sent: list = []

    async def _fake_send(msg) -> None:
        sent.append(msg)

    b.receive = _fake_receive  # type: ignore[method-assign]
    b.send = _fake_send  # type: ignore[method-assign]

    await b.run()  # no debe lanzar
    assert sent == []


# ===========================================================================
# request → agree | refuse
# ===========================================================================

@pytest.mark.asyncio
async def test_request_without_condition_dropped_at_parse_stage():
    """`condition` vacío/ausente ya lo rechaza `parse_peer_message` (T1) —
    el mensaje ni llega a `_handle_request`, se descarta como malformado y no
    se envía ningún `refuse` (no hay nada válido que correlacionar)."""
    agent = _make_agent()
    b, sent = _behaviour_for(agent, {
        "performative": "request", "conversation_id": "c1", "depth": 1,
        "goal_sig": "achieve_has_item", "condition": "",
    })
    await b.run()
    assert sent == []
    assert agent.goals == []


@pytest.mark.asyncio
async def test_request_depth_exceeds_max_refused():
    agent = _make_agent()
    agent.beliefs.apply_item_spawn("wheat", "farmland")
    with patch("config.settings.peer_max_depth", 1):
        b, sent = _behaviour_for(agent, {
            "performative": "request", "conversation_id": "c1", "depth": 5,
            "goal_sig": "achieve_has_item", "condition": "has_item(wheat, 2)",
        })
        await b.run()
    payload = _sent_payload(sent)
    assert payload["performative"] == "refuse"
    assert payload["reason"] == "max_depth"
    assert agent.goals == []


@pytest.mark.asyncio
async def test_request_unsupported_condition_refused():
    agent = _make_agent()
    b, sent = _behaviour_for(agent, {
        "performative": "request", "conversation_id": "c1", "depth": 1,
        "goal_sig": "achieve_at_zone", "condition": "at_zone(bakeri)",
    })
    await b.run()
    payload = _sent_payload(sent)
    assert payload["performative"] == "refuse"
    assert payload["reason"] == "unsupported_condition"


@pytest.mark.asyncio
async def test_request_no_capability_refused():
    agent = _make_agent()  # sin item_spawn ni recipe_output para flour
    b, sent = _behaviour_for(agent, {
        "performative": "request", "conversation_id": "c1", "depth": 1,
        "goal_sig": "achieve_has_item", "condition": "has_item(flour, 2)",
    })
    await b.run()
    payload = _sent_payload(sent)
    assert payload["performative"] == "refuse"
    assert payload["reason"] == "no_capability"
    assert agent.goals == []


@pytest.mark.asyncio
async def test_request_accepted_adopts_goal_with_peer_source_and_no_plan_injection():
    agent = _make_agent()
    agent.beliefs.apply_item_spawn("wheat", "farmland")  # capability: puede recolectar
    b, sent = _behaviour_for(agent, {
        "performative": "request", "conversation_id": "conv-1", "depth": 1,
        "goal_sig": "achieve_has_item", "condition": "has_item(wheat, 2)",
    }, sender="npc_baker@localhost")
    await b.run()

    payload = _sent_payload(sent)
    assert payload["performative"] == "agree"

    assert len(agent.goals) == 1
    goal = agent.goals[0]
    assert goal.sig == "achieve_deliver_to_peer"
    assert goal.call_args == ["npc_baker", "wheat", 2]
    assert goal.goal_source == "peer"
    assert goal.peer_requester_jid == "npc_baker@localhost"
    assert goal.peer_origin_sig == "achieve_has_item"
    assert goal.success_condition == "delivered_to_peer(npc_baker, wheat, 2)"
    # No se inyectó ningún plan: plan_graph sigue vacío (la generación es
    # responsabilidad del ciclo BDI normal, no de PeerCoordBehaviour).
    assert len(agent.plan_graph.nodes) == 0


@pytest.mark.asyncio
async def test_request_dedup_does_not_duplicate_goal():
    agent = _make_agent()
    agent.beliefs.apply_item_spawn("wheat", "farmland")
    payload_in = {
        "performative": "request", "conversation_id": "conv-1", "depth": 1,
        "goal_sig": "achieve_has_item", "condition": "has_item(wheat, 2)",
    }
    b1, sent1 = _behaviour_for(agent, payload_in, sender="npc_baker@localhost")
    await b1.run()
    assert len(agent.goals) == 1

    b2, sent2 = _behaviour_for(agent, {**payload_in, "conversation_id": "conv-2"},
                                sender="npc_baker@localhost")
    await b2.run()
    assert len(agent.goals) == 1  # no duplicado
    assert _sent_payload(sent2)["performative"] == "agree"


@pytest.mark.asyncio
async def test_max_peer_requests_respected():
    from utils.trace_logger import TraceLogger
    agent = _make_agent()
    agent.beliefs.apply_item_spawn("wheat", "farmland")
    TraceLogger._instance = None
    try:
        with patch("config.settings.max_peer_requests", 0):
            b, sent = _behaviour_for(agent, {
                "performative": "request", "conversation_id": "c1", "depth": 1,
                "goal_sig": "achieve_has_item", "condition": "has_item(wheat, 2)",
            })
            await b.run()
        payload = _sent_payload(sent)
        assert payload["performative"] == "refuse"
        assert payload["reason"] == "too_many_requests"
    finally:
        TraceLogger._instance = None


# ===========================================================================
# inform-done / failure — notificación asíncrona (sin Waiter)
# ===========================================================================

@pytest.mark.asyncio
async def test_inform_done_writes_peer_done_and_item_available():
    agent = _make_agent()
    b, sent = _behaviour_for(agent, {
        "performative": "inform-done", "conversation_id": "c1",
        "goal_sig": "achieve_has_item", "item": "flour", "qty": 2, "x": 5, "y": 6,
    }, sender="npc_miller@localhost")
    await b.run()
    assert agent.beliefs.has("peer_done", "npc_miller", "achieve_has_item")
    assert agent.beliefs.has("peer_item_available", "npc_miller", "flour", 2, 5, 6)
    assert sent == []  # inform-done no espera respuesta


@pytest.mark.asyncio
async def test_failure_writes_peer_failed():
    agent = _make_agent()
    b, sent = _behaviour_for(agent, {
        "performative": "failure", "conversation_id": "c1",
        "goal_sig": "achieve_has_item", "reason": "no_capability",
    }, sender="npc_miller@localhost")
    await b.run()
    assert agent.beliefs.has("peer_failed", "npc_miller", "achieve_has_item", "no_capability")


@pytest.mark.asyncio
async def test_inform_agree_refuse_stored_in_peer_results_for_waiter():
    agent = _make_agent()
    b, sent = _behaviour_for(agent, {
        "performative": "agree", "conversation_id": "conv-x", "goal_sig": "achieve_has_item",
    }, sender="npc_miller@localhost")
    await b.run()
    assert "conv-x" in agent.peer_results
    assert agent.peer_results["conv-x"].performative == "agree"


# ===========================================================================
# PeerReplyWaiter / PeerDoneWaiter — timeout no cuelga
# ===========================================================================

def test_peer_reply_waiter_resolves_on_match():
    agent = SimpleNamespace(peer_results={})
    resolved = []
    waiter = PeerReplyWaiter("cid-1", agent, lambda pmsg: resolved.append(pmsg), timeout_s=5)
    assert waiter.poll(None) is False  # aún nada
    agent.peer_results["cid-1"] = PeerMessage(performative="agree", conversation_id="cid-1")
    assert waiter.poll(None) is True
    assert resolved[0].performative == "agree"


def test_peer_reply_waiter_times_out():
    agent = SimpleNamespace(peer_results={})
    resolved = []
    # timeout_s negativo: garantiza "expirado" desde el primer poll() sin
    # depender de la resolución del reloj (time.time() puede no avanzar
    # entre dos llamadas consecutivas en Windows).
    waiter = PeerReplyWaiter("cid-1", agent, lambda pmsg: resolved.append(pmsg), timeout_s=-1)
    assert waiter.poll(None) is True
    assert resolved == [None]  # None = timeout, no cuelga


def test_peer_done_waiter_resolves_on_belief():
    agent_beliefs_only = SimpleNamespace(beliefs=_FakeBeliefs())
    resolved = []
    waiter = PeerDoneWaiter(
        "npc_miller", "achieve_has_item", agent_beliefs_only,
        lambda ok, reason: resolved.append((ok, reason)), timeout_s=5,
    )
    assert waiter.poll(None) is False
    agent_beliefs_only.beliefs.done.add(("npc_miller", "achieve_has_item"))
    assert waiter.poll(None) is True
    assert resolved[0] == (True, None)


def test_peer_done_waiter_times_out_without_hanging():
    agent_beliefs_only = SimpleNamespace(beliefs=_FakeBeliefs())
    resolved = []
    waiter = PeerDoneWaiter(
        "npc_miller", "achieve_has_item", agent_beliefs_only,
        lambda ok, reason: resolved.append((ok, reason)), timeout_s=-1,
    )
    assert waiter.poll(None) is True
    assert resolved == [(False, "timeout")]


class _FakeBeliefs:
    def __init__(self) -> None:
        self.done: set[tuple] = set()

    def has(self, pred, *args):
        if pred == "peer_done":
            return tuple(args) in self.done
        return False

    def query(self, pred, *args):
        return []


# ===========================================================================
# family_plan.build_has_item_family — variante delegate condicionada
# ===========================================================================

def test_delegate_variant_absent_without_coordination():
    variants = build_has_item_family([], [{"itemId": "wheat"}], coordination_enabled=False)
    assert not any("peer(" in v.guard for v in variants)


def test_delegate_variant_present_with_coordination():
    variants = build_has_item_family([], [], coordination_enabled=True, peer_request_timeout_s=60)
    delegate = [v for v in variants if "peer(" in v.guard]
    assert len(delegate) == 1
    assert "not item_spawn(Item, _)" in delegate[0].guard
    assert "not recipe_output(_, _, Item, _)" in delegate[0].guard
    assert ".ask_peer(P, can_make, Item)" in delegate[0].asl
    assert ".request_peer(P, achieve_has_item, Item, Qty)" in delegate[0].asl
    assert "60" in delegate[0].asl  # timeout inlineado


# ===========================================================================
# Registro de acciones ASL — solo con coordination_enabled
# ===========================================================================

def test_peer_actions_registered_only_when_called():
    b = BDIBehaviour()
    b.agent = SimpleNamespace(npc_id="npc_test")
    b._actions = agentspeak.Actions(agentspeak.stdlib.actions)
    b._env = asp_runtime.Environment()
    b._asp = b._env.build_agent(agentspeak.StringSource("<bdi>", ""), b._actions)
    b._pending_failure = None
    b._attempt_counts = {}
    b._register_unity_actions()

    for name, arity in [(".ask_peer", 3), (".request_peer", 4), (".await_peer", 3), (".deliver_to_peer", 3)]:
        assert (name, arity) not in b._actions.actions

    b._register_peer_actions()
    for name, arity in [(".ask_peer", 3), (".request_peer", 4), (".await_peer", 3), (".deliver_to_peer", 3)]:
        assert (name, arity) in b._actions.actions
