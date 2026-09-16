"""Integración — Fase 17: dos NPC planificados por el pipeline LLM se ayudan.

Escenarios E5 (delegación simple) y E6 (ayuda en los dos sentidos, con
arbitraje) en los dos brazos de la ablación:

  - SUB: sub-planes macro disponibles (move_to_and_pickup, craft_item,
    obtain_from_peer, collect_from_peer).
  - ATOM: sin ellos; solo acciones primitivas y de coordinación
    (ask_peer/request_peer/await_peer).

Motor agentspeak, BDIBehaviour y PeerCoordBehaviour REALES; bus y mundo en
memoria; pipeline LLM REAL (`PlanningRequestBehaviour._handle_pipeline`) con
un LLM de oráculo que siempre responde el cuerpo correcto de cada peldaño (ver
`coord_harness.py`). No mide al LLM: comprueba que la maquinaria permite que dos
agentes planificados por el pipeline se ayuden en ambos brazos.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from npc.agent import Goal
from tests.coord_harness import (
    ABLATED,
    BREAD_FROM_FLOUR_RECIPE,
    FLOUR_RECIPE,
    Bus,
    OracleLLM,
    install_llm,
    make_agent,
    run_until,
)


@contextlib.contextmanager
def _coordination_settings(mode: str):
    values = {
        "coordination_enabled": True,
        "coordination_planner": "llm",
        "builtin_subplans_enabled": mode == "SUB",
        "canonical_reuse_enabled": False,
        "canonical_family_plan": False,
        "use_refinement": True,
        "isolate_capability_contracts": False,
        "peer_max_depth": 2,
        "peer_request_timeout_s": 3.0,
        "peer_query_timeout_s": 3.0,
        "action_timeout_s": 3.0,
        "goal_arbitration_enabled": True,
        "goal_arbitration_mode": "rule",
        "arbitration_cooldown_s": 0.0,
    }
    with contextlib.ExitStack() as stack:
        for key, value in values.items():
            stack.enter_context(patch(f"config.settings.{key}", value))
        yield


async def _two_npcs(mode: str, tmp_path):
    bus = Bus()
    sub = mode == "SUB"
    miller = await make_agent("npc_miller", "miller", [FLOUR_RECIPE], bus, builtin_subplans=sub)
    baker = await make_agent("npc_baker", "baker", [BREAD_FROM_FLOUR_RECIPE], bus, builtin_subplans=sub)
    # Directorio de peers (en producción lo siembra NPCRegistry.announce_peers).
    miller[0].beliefs.apply_peer("npc_baker", "baker")
    baker[0].beliefs.apply_peer("npc_miller", "miller")
    oracles = {"npc_miller": OracleLLM(mode), "npc_baker": OracleLLM(mode)}
    for agent, _bdi, _peer in (miller, baker):
        install_llm(agent, oracles[agent.npc_id], tmp_path)
    return miller, baker, oracles


def _owner_goal() -> Goal:
    # Goal de diseñador tal como sale de parse_goals (sin reuso canónico).
    goal = Goal(sig="achieve_bake_bread", priority=1.0)
    goal.success_condition = "has_item(bread, 1)"
    goal.expected_condition = "has_item(bread, 1)"
    return goal


def _assert_arm(mode: str, oracles: dict, agents: list) -> None:
    used = {
        step["name"]
        for oracle in oracles.values()
        for _failing, steps in oracle.plans
        for step in steps
        if step["type"] == "subgoal"
    }
    if mode == "ATOM":
        assert not used & set(ABLATED)
        for agent in agents:
            assert not set(ABLATED) & set(agent.plan_graph.nodes), agent.npc_id
    else:
        assert {"obtain_from_peer", "move_to_and_pickup", "craft_item"} <= used


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["SUB", "ATOM"])
async def test_co5_llm_planned_delegation(mode: str, tmp_path):
    (miller, m_bdi, m_peer), (baker, b_bdi, b_peer), oracles = await _two_npcs(mode, tmp_path)
    baker.goals.append(_owner_goal())

    with _coordination_settings(mode):
        await run_until(
            [(miller, m_bdi, m_peer), (baker, b_bdi, b_peer)],
            lambda: baker.beliefs.has("has_item", "bread", 1) and not baker.goals,
        )

    assert baker.beliefs.has("has_item", "bread", 1)
    assert baker.intention is None
    # El miller aceptó, fabricó la harina y la entregó (peldaño fijo de la escalera).
    assert miller.beliefs.has("delivered_to_peer", "npc_baker", "flour", 1)
    assert baker.beliefs.has("peer_done", "npc_miller", "achieve_has_item")
    # La entrega del miller la planificó el pipeline, no el plan determinista.
    assert oracles["npc_miller"].plans
    _assert_arm(mode, oracles, [miller, baker])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["SUB", "ATOM"])
async def test_co6_llm_planned_bidirectional_help(mode: str, tmp_path):
    (miller, m_bdi, m_peer), (baker, b_bdi, b_peer), oracles = await _two_npcs(mode, tmp_path)
    miller.goals.append(_owner_goal())

    with _coordination_settings(mode):
        await run_until(
            [(miller, m_bdi, m_peer), (baker, b_bdi, b_peer)],
            lambda: miller.beliefs.has("has_item", "bread", 1) and not miller.goals,
        )

    assert miller.beliefs.has("has_item", "bread", 1)
    assert miller.intention is None
    # Ayuda en los dos sentidos: el baker entregó el pan y el miller la harina.
    assert baker.beliefs.has("delivered_to_peer", "npc_miller", "bread", 1)
    assert miller.beliefs.has("delivered_to_peer", "npc_baker", "flour", 1)
    # Sin arbitraje el miller se quedaría esperando al baker (límite de Fase 13).
    assert m_bdi._arbitration_switch_count >= 1
    _assert_arm(mode, oracles, [miller, baker])
