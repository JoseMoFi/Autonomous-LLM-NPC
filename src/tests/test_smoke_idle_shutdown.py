"""Smoke tests — apagado por inactividad (shutdown_when_idle).

Cubren la lógica que decide cuándo el sistema ha resuelto todo el trabajo:
  - NPCAgent.is_work_done()
  - NPCRegistry.work_done()
"""

from __future__ import annotations

from types import SimpleNamespace

from gateway.registry import NPCRegistry
from npc.agent import NPCAgent


def _make_agent() -> NPCAgent:
    return NPCAgent(
        jid="npc_x@localhost",
        password="x",
        npc_id="npc_x",
        send_to_unity=lambda m: None,
        llm_planning_jid="llm@localhost",
    )


# ===========================================================================
# NPCAgent.is_work_done
# ===========================================================================

def test_is_work_done_false_before_having_goals():
    a = _make_agent()
    # Aún no ha tenido goals (estado inicial) → no es "trabajo terminado".
    assert a.is_work_done() is False


def test_is_work_done_true_when_all_goals_resolved():
    a = _make_agent()
    a._had_goals = True
    a.goals = []
    a.intention = None
    assert a.is_work_done() is True


def test_is_work_done_false_with_active_goal():
    a = _make_agent()
    a._had_goals = True
    a.goals = [object()]
    a.intention = None
    assert a.is_work_done() is False


def test_is_work_done_false_with_intention_in_progress():
    a = _make_agent()
    a._had_goals = True
    a.goals = []
    a.intention = object()
    assert a.is_work_done() is False


# ===========================================================================
# NPCRegistry.work_done
# ===========================================================================

def _stub(done: bool):
    return SimpleNamespace(is_work_done=lambda: done)


def test_registry_work_done_false_when_no_agents():
    r = NPCRegistry("llm@localhost")
    assert r.work_done() is False


def test_registry_work_done_true_when_all_done():
    r = NPCRegistry("llm@localhost")
    r._agents = {"a": _stub(True), "b": _stub(True)}
    assert r.work_done() is True


def test_registry_work_done_false_when_one_pending():
    r = NPCRegistry("llm@localhost")
    r._agents = {"a": _stub(True), "b": _stub(False)}
    assert r.work_done() is False
