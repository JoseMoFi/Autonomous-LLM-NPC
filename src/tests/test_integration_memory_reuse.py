"""Integración — batería de memoria (A4/A5 × SUB/ATOM, primera y segunda sesión).

La batería `memory_reuse` compara una sesión que planifica con el LLM (1.ª) con
otra que arranca con la memoria de planes de la primera (2.ª). Estos tests
comprueban, con el motor agentspeak, BDIBehaviour y el pipeline REALES (LLM de
oráculo, mundo estricto de A5), lo que la batería da por hecho:

  - el plan de escalera de SUB y de ATOM se guarda en la 1.ª sesión SIN clave
    canónica (el sig de parse_goals es estable: 32/32 en la tanda larga);
  - la 2.ª sesión lo carga como `from_memory`, completa el goal sin pedir plan
    al LLM y traza `plan_memory_reuse`.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from npc.agent import Goal
from tests.coord_harness import Bus, OracleLLM, install_llm, make_agent, run_until
from tests.test_integration_atom_gather_ladder import BREAD_RECIPE, FARMLAND, _bakery_world
from utils.plan_memory import PlanMemory


@contextlib.contextmanager
def _settings(builtin_subplans: bool):
    values = {
        "coordination_enabled": False,
        "builtin_subplans_enabled": builtin_subplans,
        "canonical_reuse_enabled": False,
        "canonical_family_plan": False,
        "use_refinement": True,
        "isolate_capability_contracts": False,
        "action_timeout_s": 3.0,
        "plan_memory_reuse": True,
    }
    with contextlib.ExitStack() as stack:
        for key, value in values.items():
            stack.enter_context(patch(f"config.settings.{key}", value))
        yield


async def _bake_three(mode: str, memory_root, contracts_dir, events: list):
    builtin = mode == "SUB"
    agent, bdi, peer = await make_agent("npc_001", "baker", [BREAD_RECIPE], Bus(), builtin_subplans=builtin)
    agent.plan_memory = PlanMemory("npc_001", memory_root, promote_threshold=2, promote_min_rate=0.5)
    agent.beliefs.clear_item_at("wheat")
    agent.beliefs.apply_zone_discovery("farmland", *FARMLAND)
    actions: list = []
    agent.send_to_unity = _bakery_world(agent, actions)
    oracle = OracleLLM(mode)
    install_llm(agent, oracle, contracts_dir)

    goal = Goal(sig="achieve_bake_bread", priority=1.0)
    goal.success_condition = "has_item(bread, 3)"
    goal.expected_condition = goal.success_condition
    agent.goals.append(goal)

    with _settings(builtin), \
         patch("npc.behaviours.bdi._trace", lambda ev, **kw: events.append((ev, kw))), \
         patch("utils.trace_logger.trace", lambda ev, **kw: events.append((ev, kw))):
        agent._load_memory_plans()
        await run_until(
            [(agent, bdi, peer)],
            lambda: agent.beliefs.has("has_item", "bread", 3) and not agent.goals,
            max_seconds=90,
        )
    return agent, oracle


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["SUB", "ATOM"])
async def test_second_session_reuses_the_plan_learned_in_the_first(mode: str, tmp_path):
    memory_root = tmp_path / "pair"
    first_events: list = []
    first, first_oracle = await _bake_three(mode, memory_root, tmp_path, first_events)
    assert first.beliefs.has("has_item", "bread", 3)
    assert first_oracle.plans, "la 1.ª sesión planifica con el LLM"
    assert (memory_root / "npc_001" / "pending" / "achieve_bake_bread.json").exists()

    second_events: list = []
    second, second_oracle = await _bake_three(mode, memory_root, tmp_path, second_events)
    assert second.beliefs.has("has_item", "bread", 3)
    assert second_oracle.plans == [], "la 2.ª sesión no pide plan al LLM"
    assert [kw["goal"] for ev, kw in second_events if ev == "plan_memory_reuse"] == ["achieve_bake_bread"]
    assert not [ev for ev, _kw in second_events if ev == "plan_requested"]
