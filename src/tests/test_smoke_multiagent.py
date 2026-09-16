"""Smoke tests — multi-agente independiente (Fase 11).

Cubren, sin necesitar SPADE/XMPP en vivo:
  - MessageRouter despacha ActionResult al NPCAgent correcto; npc_id
    desconocido no revienta ni se filtra a otro agente.
  - Aislamiento de creencias entre agentes (BeliefStore por instancia).
  - Aislamiento de plan memory: dos agentes con el mismo run_root escriben en
    subcarpetas separadas y no se cargan planes cruzados.
  - NPCRegistry.work_done() exige que TODOS los agentes hayan terminado.
  - NPCRegistry.announce_peers(): peer/2 sembrado cruzado, sin auto-peer.
  - Métrica planning_wait_s (Fase 11 T5): se acumula y aparece en metrics.json.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.registry import NPCRegistry
from gateway.router import MessageRouter
from npc.agent import NPCAgent
from utils.plan_memory import PlanMemory
from utils.trace_logger import TraceLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_trace_logger_singleton():
    TraceLogger._instance = None
    yield
    if TraceLogger._instance is not None:
        try:
            TraceLogger._instance._trace_fh.close()
            TraceLogger._instance._prompt_fh.close()
            TraceLogger._instance._unity_fh.close()
        except Exception:
            pass
        TraceLogger._instance = None


def _make_agent(npc_id: str, plan_memory_root: Path | None = None) -> NPCAgent:
    return NPCAgent(
        jid=f"{npc_id}@localhost",
        password="x",
        npc_id=npc_id,
        send_to_unity=lambda m: None,
        llm_planning_jid="llm_planning@localhost",
        plan_memory_root=plan_memory_root,
    )


# ===========================================================================
# MessageRouter — despacho por npc_id, sin filtrado cruzado
# ===========================================================================

class _FakeWriter:
    def __init__(self) -> None:
        self.closed = False
        self.written: list[bytes] = []

    def is_closing(self) -> bool:
        return self.closed

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        return None


def test_router_delivers_action_result_to_correct_agent():
    registry = NPCRegistry(llm_planning_jid="llm@localhost")
    a = _make_agent("npc_001")
    b = _make_agent("npc_002")
    registry._agents = {"npc_001": a, "npc_002": b}
    router = MessageRouter(registry)

    asyncio.run(router.route(
        {"type": "ActionResult", "npc_id": "npc_002", "status": "Success"},
        _FakeWriter(),
    ))

    assert a.inbox.qsize() == 0
    assert b.inbox.qsize() == 1


def test_router_ignores_unknown_npc_without_crashing():
    registry = NPCRegistry(llm_planning_jid="llm@localhost")
    a = _make_agent("npc_001")
    registry._agents = {"npc_001": a}
    router = MessageRouter(registry)

    # No debe lanzar excepción ni entregar nada al agente existente.
    asyncio.run(router.route(
        {"type": "ActionResult", "npc_id": "npc_999", "status": "Success"},
        _FakeWriter(),
    ))
    assert a.inbox.qsize() == 0


# ===========================================================================
# Aislamiento de creencias
# ===========================================================================

def test_beliefs_isolated_between_agents():
    a = _make_agent("npc_001")
    b = _make_agent("npc_002")

    a.beliefs.apply_has_item("wheat", 3)

    assert a.beliefs.has("has_item", "wheat", 3)
    assert not b.beliefs.has("has_item", "wheat", 3)
    assert not b.beliefs.query("has_item")


# ===========================================================================
# Aislamiento de plan memory
# ===========================================================================

def _variants() -> list[dict]:
    return [{"guard": "true", "steps": [".wait(1)"], "full_asl": "+!achieve_wait : true <- .wait(1)."}]


def test_plan_memory_isolated_between_agents(tmp_path):
    # Cada NPC persiste bajo <run_root>/<npc_id>/ — un reuso de A no debe
    # cargar los planes aprobados de B.
    seed_a = PlanMemory("npc_001", tmp_path, promote_threshold=1, promote_min_rate=0.5)
    seed_a.store("achieve_wait", _variants(), _variants()[0]["full_asl"])
    assert seed_a.record_success("achieve_wait") is True

    # npc_002 no tiene nada guardado.
    agent_a = _make_agent("npc_001", plan_memory_root=tmp_path)
    agent_b = _make_agent("npc_002", plan_memory_root=tmp_path)
    agent_a._load_memory_plans()
    agent_b._load_memory_plans()

    assert agent_a.plan_graph.has_node("achieve_wait")
    assert not agent_b.plan_graph.has_node("achieve_wait")

    # Los ficheros de cada NPC viven en subcarpetas separadas; npc_002 puede
    # tener su propia carpeta (creada al construir PlanMemory) pero sin
    # ningún registro aprobado dentro — no debe "ver" los planes de npc_001.
    assert agent_b.plan_memory.load_all_approved() == []


# ===========================================================================
# NPCRegistry.work_done — exige que TODOS terminen
# ===========================================================================

def _stub(done: bool):
    return SimpleNamespace(is_work_done=lambda: done)


def test_work_done_requires_all_npcs_finished():
    r = NPCRegistry("llm@localhost")
    r._agents = {"npc_001": _stub(True), "npc_002": _stub(False)}
    assert r.work_done() is False

    r._agents["npc_002"] = _stub(True)
    assert r.work_done() is True


def test_one_agent_failing_all_goals_does_not_mark_global_done_early():
    # npc_001 ya resolvió todo (fallado sin repair); npc_002 sigue trabajando.
    r = NPCRegistry("llm@localhost")
    a = _make_agent("npc_001")
    a._had_goals = True
    a.goals = []
    a.intention = None  # is_work_done() == True

    b = _make_agent("npc_002")
    b._had_goals = True
    b.goals = [object()]  # sigue con goals pendientes

    r._agents = {"npc_001": a, "npc_002": b}
    assert r.work_done() is False


# ===========================================================================
# announce_peers — directorio de agentes (Fase 11 T6)
# ===========================================================================

def test_announce_peers_seeds_belief_both_ways_without_self_peer():
    r = NPCRegistry("llm@localhost")
    a = _make_agent("npc_001")
    r._agents = {"npc_001": a}

    b = _make_agent("npc_002")
    r._agents["npc_002"] = b
    r.announce_peers("npc_002")

    assert a.beliefs.has("peer", "npc_002", "unknown")
    assert b.beliefs.has("peer", "npc_001", "unknown")
    # Ningún agente se anuncia a sí mismo.
    assert not a.beliefs.has("peer", "npc_001", "unknown")
    assert not b.beliefs.has("peer", "npc_002", "unknown")


def test_announce_peers_uses_known_role_when_profile_present():
    r = NPCRegistry("llm@localhost")
    a = _make_agent("npc_001")
    a.profile = SimpleNamespace(role="baker")
    r._agents = {"npc_001": a}

    b = _make_agent("npc_002")
    r._agents["npc_002"] = b
    r.announce_peers("npc_002")

    assert b.beliefs.has("peer", "npc_001", "baker")


def test_announce_peers_noop_for_single_agent():
    r = NPCRegistry("llm@localhost")
    a = _make_agent("npc_001")
    r._agents = {"npc_001": a}
    r.announce_peers("npc_001")  # no debe lanzar; no hay a nadie que anunciar
    assert not a.beliefs.query("peer")


# ===========================================================================
# planning_wait_s — cola de planificación compartida (Fase 11 T5)
# ===========================================================================

def test_planning_queue_wait_accumulates_in_metrics(tmp_path):
    import time
    from llm.planning_agent import PlanningRequestBehaviour

    tl = TraceLogger.init_session(logs_root=tmp_path / "logs")
    try:
        payload = {"task": "generate_plan", "npc_id": "npc_001", "_enqueued_at": time.time() - 0.05}
        PlanningRequestBehaviour._trace_queue_wait(payload, "generate_plan")

        assert tl.metrics.npc("npc_001").planning_wait_s > 0

        tl.finalize()
        data = json.loads((tl.session_dir / "metrics.json").read_text(encoding="utf-8"))
        assert data["npcs"]["npc_001"]["planning_wait_s"] > 0
    finally:
        pass


def test_planning_queue_wait_noop_without_enqueued_at(tmp_path):
    from llm.planning_agent import PlanningRequestBehaviour

    tl = TraceLogger.init_session(logs_root=tmp_path / "logs")
    # Sin _enqueued_at (mensaje que no pasó por LLMBehaviour.run) — no debe
    # fabricar un tiempo de espera ni lanzar excepción.
    PlanningRequestBehaviour._trace_queue_wait({"task": "generate_plan", "npc_id": "npc_001"}, "generate_plan")
    assert tl.metrics.npc("npc_001").planning_wait_s == 0.0
