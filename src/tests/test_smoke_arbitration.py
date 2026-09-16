"""Smoke tests — arbitraje de goals / preempción (Fase 14).

Cubren:
  - `_maybe_arbitrate`: no-op salvo que TODAS las condiciones de disparo se
    cumplan (flag off, sin coordinación, cooldown, max_switches, sin
    candidatos, bloqueado en un waiter que NO es PeerDoneWaiter).
  - `_arbitrate_by_rule`: detecta el ciclo de espera de longitud 2 vía
    `peer_requester_jid`, y devuelve None cuando no hay ciclo.
  - `_decide_arbitration`: modo "rule" nunca llama al LLM; modo "llm" cae a
    la regla (RULE_FALLBACK) si `run_llm_task` lanza.
  - `_validate_arbitrate`: rechaza `goal_sig` no ofrecido, `decision`
    inválida, resultado no-dict — política de no-fabricación.
  - `_park_intention` / `_restore_intention`: conservan la pila de intención
    y desplazan el deadline del waiter por el tiempo aparcado (sin esto el
    waiter expiraría de inmediato al volver).
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import agentspeak
import agentspeak.stdlib  # noqa: F401
from agentspeak import runtime as asp_runtime
import pytest

from npc.agent import NPCAgent, Goal
from npc.behaviours.bdi import BDIBehaviour, PeerDoneWaiter, ActionResultWaiter
from llm.planning_agent import _validate_arbitrate


def _make_agent(npc_id: str = "npc_miller") -> NPCAgent:
    return NPCAgent(
        jid=f"{npc_id}@localhost", password="x", npc_id=npc_id,
        send_to_unity=lambda m: None, llm_planning_jid="llm@localhost",
    )


def _bdi_for(agent: NPCAgent) -> BDIBehaviour:
    """BDIBehaviour mínimo, sin pasar por on_start() (mismo patrón que
    test_smoke_peer_coord.py::test_peer_actions_registered_only_when_called)."""
    b = BDIBehaviour()
    b.agent = agent
    b._actions = agentspeak.Actions(agentspeak.stdlib.actions)
    b._env = asp_runtime.Environment()
    b._asp = b._env.build_agent(agentspeak.StringSource("<bdi>", ""), b._actions)
    b._pending_failure = None
    b._attempt_counts = {}
    b._arbitration_switch_count = 0
    b._last_arbitration_ts = 0.0
    return b


def _push_waiting_intention(bdi: BDIBehaviour, waiter) -> None:
    """Simula que la intención activa está bloqueada en `waiter`: una pila con
    UNA Intention cuyo .waiter es el dado, en self._asp.intentions."""
    import collections
    intention = asp_runtime.Intention()
    intention.waiter = waiter
    stack = collections.deque([intention])
    bdi._asp.intentions.append(stack)


# ===========================================================================
# _maybe_arbitrate — condiciones de disparo
# ===========================================================================

@pytest.mark.asyncio
async def test_no_preempt_when_arbitration_disabled():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    agent.goals = [goal, Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])]
    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))

    with patch("config.settings.goal_arbitration_enabled", False), \
         patch("config.settings.coordination_enabled", True):
        switched = await bdi._maybe_arbitrate(goal)
    assert switched is False
    assert agent.intention is None  # no lo tocó


@pytest.mark.asyncio
async def test_no_preempt_without_coordination():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    agent.goals = [goal, Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])]
    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))

    with patch("config.settings.goal_arbitration_enabled", True), \
         patch("config.settings.coordination_enabled", False):
        switched = await bdi._maybe_arbitrate(goal)
    assert switched is False


@pytest.mark.asyncio
async def test_no_preempt_without_candidates():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    agent.goals = [goal]  # sin candidatos
    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))

    with patch("config.settings.goal_arbitration_enabled", True), \
         patch("config.settings.coordination_enabled", True):
        switched = await bdi._maybe_arbitrate(goal)
    assert switched is False


@pytest.mark.asyncio
async def test_no_preempt_on_unity_action_waiter():
    """La intención activa está a mitad de un MoveTo real (ActionResultWaiter),
    no esperando a un peer — NO es preemptible."""
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    agent.goals = [goal, Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])]
    _push_waiting_intention(bdi, ActionResultWaiter(
        "cmd-1", {}, asp_runtime.Intention(), agent.npc_id, goal.sig, "MoveTo", bdi,
    ))

    with patch("config.settings.goal_arbitration_enabled", True), \
         patch("config.settings.coordination_enabled", True):
        switched = await bdi._maybe_arbitrate(goal)
    assert switched is False


@pytest.mark.asyncio
async def test_no_preempt_when_not_blocked_at_all():
    """Sin ninguna intención viva (goal recién seleccionado, plan aún no
    inyectado) -- nada que preemptar."""
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    agent.goals = [goal, Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])]

    with patch("config.settings.goal_arbitration_enabled", True), \
         patch("config.settings.coordination_enabled", True):
        switched = await bdi._maybe_arbitrate(goal)
    assert switched is False


@pytest.mark.asyncio
async def test_cooldown_blocks_second_consult():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    candidate = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])
    candidate.peer_requester_jid = "npc_baker@localhost"
    agent.goals = [goal, candidate]
    bdi._last_arbitration_ts = time.time()  # "acabo de consultar"

    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))
    with patch("config.settings.goal_arbitration_enabled", True), \
         patch("config.settings.coordination_enabled", True), \
         patch("config.settings.arbitration_cooldown_s", 999.0):
        switched = await bdi._maybe_arbitrate(goal)
    assert switched is False


@pytest.mark.asyncio
async def test_max_switches_respected():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    candidate = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])
    candidate.peer_requester_jid = "npc_baker@localhost"
    agent.goals = [goal, candidate]
    bdi._arbitration_switch_count = 4

    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))
    with patch("config.settings.goal_arbitration_enabled", True), \
         patch("config.settings.coordination_enabled", True), \
         patch("config.settings.arbitration_max_switches", 4):
        switched = await bdi._maybe_arbitrate(goal)
    assert switched is False


@pytest.mark.asyncio
async def test_returning_to_the_same_goal_does_not_spend_the_switch_cap():
    # Fase 17w (piloto 17v, CO6/ATOM): los replans de un mismo encargo agotaban el tope.
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    candidate = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])
    candidate.peer_requester_jid = "npc_baker@localhost"
    agent.goals = [goal, candidate]
    agent.intention = goal
    bdi._arbitration_switch_count = 4
    bdi._last_switch_key = "achieve_deliver_to_peer__npc_baker_flour_1"

    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))
    with patch("config.settings.goal_arbitration_enabled", True), \
         patch("config.settings.coordination_enabled", True), \
         patch("config.settings.goal_arbitration_mode", "rule"), \
         patch("config.settings.arbitration_max_switches", 4):
        switched = await bdi._maybe_arbitrate(goal)
    assert switched is True and agent.intention is candidate
    assert bdi._arbitration_switch_count == 4


@pytest.mark.asyncio
async def test_switch_happens_via_rule_and_parks_previous_intention():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    candidate = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])
    candidate.peer_requester_jid = "npc_baker@localhost"
    agent.goals = [goal, candidate]
    agent.intention = goal

    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))
    with patch("config.settings.goal_arbitration_enabled", True), \
         patch("config.settings.coordination_enabled", True), \
         patch("config.settings.goal_arbitration_mode", "rule"):
        switched = await bdi._maybe_arbitrate(goal)

    assert switched is True
    assert agent.intention is candidate
    assert bdi._arbitration_switch_count == 1
    # la pila del goal original quedó aparcada, no perdida
    assert getattr(goal, "_parked_stack", None)
    assert len(bdi._asp.intentions) == 0  # el motor quedó vacío, listo para inyectar `candidate`


# ===========================================================================
# _arbitrate_by_rule — detección de ciclo de espera de longitud 2
# ===========================================================================

def test_rule_detects_cycle():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    matching = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])
    matching.peer_requester_jid = "npc_baker@localhost/res"
    other = Goal(sig="achieve_deliver_to_peer", call_args=["npc_other", "wheat", 1])
    other.peer_requester_jid = "npc_other@localhost"

    chosen = bdi._arbitrate_by_rule("npc_baker", [other, matching])
    assert chosen is matching


def test_rule_returns_none_without_cycle():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    other = Goal(sig="achieve_deliver_to_peer", call_args=["npc_other", "wheat", 1])
    other.peer_requester_jid = "npc_other@localhost"

    chosen = bdi._arbitrate_by_rule("npc_baker", [other])
    assert chosen is None


def test_rule_ignores_candidates_without_requester():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    designer_goal = Goal(sig="achieve_has_item", call_args=["wheat", 2])  # goal_source="designer", sin requester
    chosen = bdi._arbitrate_by_rule("npc_baker", [designer_goal])
    assert chosen is None


# ===========================================================================
# _decide_arbitration — despacho de modo + fallback
# ===========================================================================

@pytest.mark.asyncio
async def test_mode_rule_never_calls_llm():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    candidate = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])
    candidate.peer_requester_jid = "npc_baker@localhost"

    called = []
    agent.run_llm_task = lambda task: called.append(task) or {}  # no debería invocarse

    with patch("config.settings.goal_arbitration_mode", "rule"):
        chosen, source, reason = await bdi._decide_arbitration(goal, "npc_baker", [candidate])
    assert chosen is candidate
    assert source == "RULE"
    assert called == []


@pytest.mark.asyncio
async def test_mode_llm_falls_back_to_rule_on_exception():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    candidate = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])
    candidate.peer_requester_jid = "npc_baker@localhost"

    async def _boom(task):
        raise RuntimeError("ollama down")
    agent.run_llm_task = _boom
    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))

    with patch("config.settings.goal_arbitration_mode", "llm"):
        chosen, source, reason = await bdi._decide_arbitration(goal, "npc_baker", [candidate])
    assert chosen is candidate  # la regla SÍ encuentra el ciclo
    assert source == "RULE_FALLBACK"
    assert "llm_error" in reason


@pytest.mark.asyncio
async def test_mode_llm_switch_uses_offered_goal_sig():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    candidate = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])

    async def _fake_llm(task):
        assert task["task"] == "arbitrate"
        return {"decision": "switch", "goal_sig": "achieve_deliver_to_peer", "reason": "unblocks peer"}
    agent.run_llm_task = _fake_llm
    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))

    with patch("config.settings.goal_arbitration_mode", "llm"):
        chosen, source, reason = await bdi._decide_arbitration(goal, "npc_baker", [candidate])
    assert chosen is candidate
    assert source == "LLM"
    assert reason == "unblocks peer"


@pytest.mark.asyncio
async def test_mode_llm_wait_keeps_current_goal():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    candidate = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])

    async def _fake_llm(task):
        return {"decision": "wait", "reason": "not related"}
    agent.run_llm_task = _fake_llm
    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))

    with patch("config.settings.goal_arbitration_mode", "llm"):
        chosen, source, reason = await bdi._decide_arbitration(goal, "npc_baker", [candidate])
    assert chosen is None
    assert source == "LLM"


@pytest.mark.asyncio
async def test_llm_hallucinated_goal_sig_falls_back_to_rule():
    """El LLM devuelve un goal_sig que NO estaba entre los candidatos
    ofrecidos -- _arbitrate_via_llm lo rechaza (política de no-fabricación) y
    el caller cae a la regla, nunca ejecuta el goal fabricado."""
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    candidate = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])
    candidate.peer_requester_jid = "npc_baker@localhost"

    async def _fake_llm(task):
        return {"decision": "switch", "goal_sig": "achieve_invented_goal", "reason": "..."}
    agent.run_llm_task = _fake_llm
    _push_waiting_intention(bdi, PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, 120))

    with patch("config.settings.goal_arbitration_mode", "llm"):
        chosen, source, reason = await bdi._decide_arbitration(goal, "npc_baker", [candidate])
    assert chosen is candidate  # la regla SÍ resuelve el mismo caso
    assert source == "RULE_FALLBACK"


# ===========================================================================
# _validate_arbitrate — política de no-fabricación
# ===========================================================================

def _payload(candidates_sigs):
    return {"task": "arbitrate", "candidates": [{"sig": s} for s in candidates_sigs]}


def test_validate_arbitrate_accepts_wait():
    assert _validate_arbitrate({"decision": "wait", "reason": "x"}, _payload(["a"])) == []


def test_validate_arbitrate_accepts_switch_with_offered_sig():
    errors = _validate_arbitrate(
        {"decision": "switch", "goal_sig": "a", "reason": "x"}, _payload(["a", "b"]),
    )
    assert errors == []


def test_validate_arbitrate_rejects_non_dict():
    assert _validate_arbitrate(["not", "a", "dict"], _payload(["a"])) != []


def test_validate_arbitrate_rejects_invalid_decision():
    assert _validate_arbitrate({"decision": "maybe"}, _payload(["a"])) != []


def test_validate_arbitrate_rejects_unoffered_goal_sig():
    errors = _validate_arbitrate(
        {"decision": "switch", "goal_sig": "achieve_invented", "reason": "x"}, _payload(["a", "b"]),
    )
    assert errors != []
    assert "not among the offered candidates" in errors[0]


def test_validate_arbitrate_rejects_switch_without_goal_sig():
    errors = _validate_arbitrate({"decision": "switch"}, _payload(["a"]))
    assert errors != []


# ===========================================================================
# _park_intention / _restore_intention — conserva pila y desplaza deadline
# ===========================================================================

def test_park_and_restore_preserve_stack_and_shift_waiter_deadline():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])

    waiter = PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, timeout_s=5.0)
    t0_before = waiter._t0
    _push_waiting_intention(bdi, waiter)

    bdi._park_intention(goal)
    assert len(bdi._asp.intentions) == 0  # el motor queda vacío
    assert goal._parked_stack  # la pila se guardó en el goal

    # Simula que estuvo aparcada 10s reales (más que el timeout de 5s del
    # waiter) -- sin el desplazamiento de deadline, expiraría al instante.
    goal._parked_at = time.time() - 10.0
    restored = bdi._restore_intention(goal)

    assert restored is True
    assert len(bdi._asp.intentions) == 1  # la pila volvió
    assert getattr(goal, "_parked_stack", None) is None  # se limpió
    # El deadline se desplazó ~10s -- el waiter YA NO debe expirar de inmediato.
    assert waiter._t0 > t0_before
    assert waiter.poll(None) is False  # sigue esperando, no expiró por el aparcado


def test_restore_without_parked_stack_is_noop():
    agent = _make_agent()
    bdi = _bdi_for(agent)
    goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    assert bdi._restore_intention(goal) is False


@pytest.mark.asyncio
async def test_select_intention_restores_parked_goal():
    """Integración corta: cuando el goal preemptor cierra y _select_intention
    vuelve a elegir el goal original, su pila aparcada se restaura (no se
    re-inyecta desde cero)."""
    agent = _make_agent()
    bdi = _bdi_for(agent)
    parked_goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    parked_goal.priority = 1.0

    waiter = PeerDoneWaiter("npc_baker", "achieve_has_item", agent, lambda *a: None, timeout_s=120.0)
    _push_waiting_intention(bdi, waiter)
    bdi._park_intention(parked_goal)

    agent.goals = [parked_goal]
    agent.intention = None
    await bdi._select_intention()

    assert agent.intention is parked_goal
    assert len(bdi._asp.intentions) == 1  # restaurada, no vacía
