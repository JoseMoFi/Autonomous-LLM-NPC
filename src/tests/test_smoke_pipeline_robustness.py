"""Smoke tests — Fase 17c: robustez del pipeline cuando no hay sub-planes.

Defectos que destapó la tanda ATOM real (2026-09-15, 25 de 32 sesiones):
  - args como objeto o lista de objetos → la rama no compilaba (A3);
  - sub-goals inventados por el LLM: su plan (Paso 6) se descartaba y el padre
    fallaba con "no applicable plan" (A1/A2/A4);
  - un goal FAILED desaparecía sin traza (el análisis lo marcaba `unknown`).
El cuarto (ids de receta en mayúscula = variable ASL) está cubierto en
test_smoke_coordination_llm.py.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import analyze_ablation as AA  # noqa: E402
from llm.pipeline.pipeline_runner import _normalize_step_args, run_full_pipeline  # noqa: E402
from npc.agent import Goal  # noqa: E402
from npc.plan_graph import GoalNode, NodeStatus  # noqa: E402
from tests.coord_harness import Bus, make_agent, run_until  # noqa: E402


# ===========================================================================
# args como objeto
# ===========================================================================

def test_object_args_become_positional():
    # Formas reales de las respuestas de qwen3 en la tanda ATOM.
    steps = [
        {"type": "subgoal", "name": "achieve_explore_zone", "args": [{"zoneTag": "farmland"}]},
        {"type": "action", "name": "MoveTo", "args": [{"X": 9, "Y": -6}]},
        {"type": "action", "name": "MoveTo", "args": {"X": 3, "Y": 4}},
        {"type": "action", "name": "Craft", "args": [{"itemId": "wheat", "targetId": "bread_recipe", "qty": 2}]},
        {"type": "action", "name": "request_peer",
         "args": {"qty": 1, "npcId": "npc_miller", "itemId": "flour", "goalSig": "achieve_has_item"}},
        {"type": "subgoal", "name": "achieve_custom", "args": {"b": 2, "a": 1}},
        {"type": "action", "name": "MoveTo", "args": [{"var": "X"}, {"var": "Y"}]},
        {"type": "action", "name": "PickUp", "args": ["wheat"]},
    ]
    _normalize_step_args(steps)
    assert [s["args"] for s in steps] == [
        ["farmland"],
        [9, -6],
        [3, 4],
        ["wheat", "bread_recipe", 2],
        ["npc_miller", "achieve_has_item", "flour", 1],   # orden de la firma, no el escrito
        [2, 1],                                           # sin firma: orden escrito por el LLM
        [{"var": "X"}, {"var": "Y"}],                     # marcadores de variable: intactos
        ["wheat"],
    ]


@pytest.mark.asyncio
async def test_pipeline_compiles_object_args(tmp_path):
    response = json.dumps({"steps": [
        {"type": "action", "name": "MoveTo", "args": {"X": 3, "Y": 4}},
        {"type": "action", "name": "PickUp", "args": [{"itemId": "wheat"}]},
    ]})

    async def llm(_user: str, _system: str) -> str:
        return response

    result = await run_full_pipeline(
        goal_sig="achieve_collect_wheat",
        npc_statement="Collect 1 wheat",
        existing_goals=[],
        llm_call=llm,
        description="The NPC must collect 1 wheat.",
        success_condition="has_item(wheat, 1)",
        use_refinement=False,
        builtin_subplans=False,
    )
    body = result.variants[1].asl
    assert ".moveto(3, 4)" in body and ".pickup(wheat)" in body
    assert "{" not in body


# ===========================================================================
# sub-goals inventados por el LLM
# ===========================================================================

def _plan_with_invented_subgoal() -> dict:
    return {
        "sig": "achieve_reach_spot",
        "description": "reach the spot",
        "variants": [
            {"guard": "current_position(1, 1)", "steps": [],
             "asl": "+!achieve_reach_spot : current_position(1, 1) <- true."},
            {"guard": "not current_position(1, 1)",
             "steps": [{"type": "subgoal", "name": "achieve_go_there", "args": []}],
             "asl": "+!achieve_reach_spot : not current_position(1, 1) <-\n    !achieve_go_there."},
        ],
        "subgoals_to_expand": [{"sig": "achieve_go_there", "description": "go to the spot"}],
        "sub_results": {
            "achieve_go_there": {
                "sig": "achieve_go_there",
                "description": "go to the spot",
                "variants": [
                    {"guard": "true", "steps": [{"type": "action", "name": "MoveTo", "args": [1, 1]}],
                     "asl": "+!achieve_go_there : true <-\n    .moveto(1, 1)."},
                ],
                "sub_results": {},
            },
        },
        "dag": {"nodes": [], "edges": [["achieve_reach_spot", "achieve_go_there"]]},
    }


@pytest.mark.asyncio
async def test_invented_subgoal_plan_is_loaded_and_called_inline():
    agent, bdi, peer = await make_agent("npc_001", "farmer", [], Bus(), builtin_subplans=False)
    goal = Goal(sig="achieve_reach_spot", priority=1.0)
    goal.success_condition = "current_position(1, 1)"
    agent.goals.append(goal)

    async def run_llm_task(task: dict):
        assert task["task"] == "generate_plan", task["task"]
        return _plan_with_invented_subgoal()

    agent.run_llm_task = run_llm_task
    events: list[tuple[str, dict]] = []
    with patch("npc.behaviours.bdi._trace", lambda ev, **kw: events.append((ev, kw))), \
         patch("config.settings.action_timeout_s", 3.0):
        await run_until([(agent, bdi, peer)], lambda: not agent.goals, max_seconds=20)

    assert agent.beliefs.has("current_position", 1, 1)
    assert [ev for ev, _ in events if ev == "goal_completed"]
    # El sub-goal no se encoló como goal suelto: su plan se cargó y el padre lo llamó.
    assert any(ev == "subplan_loaded" and kw["subgoal"] == "achieve_go_there" for ev, kw in events)
    assert not [ev for ev, _ in events if ev == "goal_failed"]


# ===========================================================================
# goal FAILED con traza
# ===========================================================================

@pytest.mark.asyncio
async def test_failed_goal_leaves_a_trace():
    agent, bdi, _peer = await make_agent("npc_001", "farmer", [], Bus(), builtin_subplans=True)
    goal = Goal(sig="achieve_x", priority=1.0)
    goal.success_condition = "has_item(bread, 1)"
    agent.goals.append(goal)
    agent.intention = goal
    node = GoalNode(sig="achieve_x", status=NodeStatus.FAILED)
    node.failure_history.append("plan_failure")
    agent.plan_graph.add_node("achieve_x", data=node)

    events: list[tuple[str, dict]] = []
    with patch("npc.behaviours.bdi._trace", lambda ev, **kw: events.append((ev, kw))):
        await bdi.run()

    assert agent.goals == []
    failed = [kw for ev, kw in events if ev == "goal_failed"]
    assert failed and failed[0]["goal"] == "achieve_x" and failed[0]["reason"] == "plan_failure"


def test_analyzer_uses_goal_failed_as_cause(tmp_path):
    manifests = {"A2": {"id": "A2", "goals": {"npc_001": [{"nl": "Collect 2 @wheat", "condition": "has_item(wheat, 2)"}]}}}
    events = [
        {"t": 0.0, "ev": "session_start", "experiment_id": "A2", "config_label": "ATOM", "builtin_subplans": False},
        {"t": 0.0, "ev": "unity_in", "npc": "npc_001", "msg_type": "RegisterNPC"},
        {"t": 30.0, "ev": "goal_failed", "npc": "npc_001", "goal": "achieve_collect_wheat", "reason": "plan_failure"},
        {"t": 31.0, "ev": "shutdown_idle"},
        {"t": 31.5, "ev": "session_end", "duration_s": 31.5},
    ]
    d = tmp_path / "s"
    d.mkdir()
    (d / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    summary = AA.summarize_session(d, manifests, ("SUB", "ATOM"))
    assert summary["primary_cause"] == "failed:plan_failure"


# ===========================================================================
# Fase 17e (tanda 2 ATOM): PickUp en el simulador y replan tras fallo de acción
# ===========================================================================

def _act(name, *args):
    return {"type": "action", "name": name, "args": list(args)}


def test_simulator_credits_primitive_pickup_with_literal_qty():
    from llm.pipeline.plan_simulator import active_subgoal_guarantees, check_plan_reachability
    g = active_subgoal_guarantees(False)
    one = [_act("MoveTo", 10, -8), _act("PickUp", "wheat")]
    assert check_plan_reachability(one, {}, "has_item(wheat, 1)", subgoal_guarantees=g).reachable
    assert not check_plan_reachability(one, {}, "has_item(wheat, 2)", subgoal_guarantees=g).reachable
    two = one + [_act("MoveTo", 11, -5), _act("PickUp", "wheat")]
    assert check_plan_reachability(two, {}, "has_item(wheat, 2)", subgoal_guarantees=g).reachable
    # MoveTo + Search siguen sin dar inventario.
    assert not check_plan_reachability(
        [_act("MoveTo", 9, -6), _act("Search", "wheat")], {}, "has_item(wheat, 1)",
        subgoal_guarantees=g,
    ).reachable


def test_simulator_does_not_mutate_llm_dict_args():
    from llm.pipeline.plan_simulator import simulate_plan
    step = {"type": "action", "name": "PickUp", "args": {"itemId": "wheat"}}
    simulate_plan([step], {})
    assert step["args"] == {"itemId": "wheat"}


_PICKUP_FAIL = {
    "status": "Failure", "errorCode": "NotFound",
    "errorMessage": "PickUp failed (NotFound): No item on current cell.",
}


async def _goal_at_failure_limit(condition):
    agent, bdi, _peer = await make_agent("npc_001", "farmer", [], Bus(), builtin_subplans=False)
    goal = Goal(sig="achieve_collect_wheat", priority=1.0)
    goal.success_condition = condition
    agent.goals.append(goal)
    agent.intention = goal
    node = GoalNode(sig="achieve_collect_wheat", status=NodeStatus.READY)
    node.failure_count = 2
    agent.plan_graph.add_node("achieve_collect_wheat", data=node)
    return agent, bdi, goal, node


@pytest.mark.asyncio
async def test_action_failure_limit_replans_with_unity_error():
    agent, bdi, goal, node = await _goal_at_failure_limit("has_item(wheat, 1)")
    events: list[tuple[str, dict]] = []
    with patch("npc.behaviours.bdi._trace", lambda ev, **kw: events.append((ev, kw))):
        await bdi._handle_action_failure(goal, dict(_PICKUP_FAIL))
    assert node.status == NodeStatus.NEEDS_REPLAN
    assert goal.replan_count == 1 and node.failure_count == 0
    assert "No item on current cell" in goal.replan_hint
    assert any(g is goal for g in agent.goals)
    assert [kw["reason"] for ev, kw in events if ev == "goal_replan_required"] == ["action_failed:NotFound"]


@pytest.mark.asyncio
async def test_action_failure_after_replan_budget_fails_goal():
    agent, bdi, goal, node = await _goal_at_failure_limit("has_item(wheat, 1)")
    goal.replan_count = 3
    await bdi._handle_action_failure(goal, dict(_PICKUP_FAIL))
    assert node.status == NodeStatus.FAILED
    assert not any(g is goal for g in agent.goals)


# ===========================================================================
# Fase 17s: MoveTo a una casilla ocupada por otro NPC
# ===========================================================================

def test_moveto_retry_candidates():
    from npc.behaviours.bdi import moveto_retry_candidates
    assert moveto_retry_candidates(-11, 5)[:3] == [(-11, 5), (-10, 5), (-12, 5)]
    assert len(moveto_retry_candidates(-11, 5)) == 9
    assert moveto_retry_candidates(3, 4, include_neighbours=False) == [(3, 4)]


@pytest.mark.asyncio
async def test_moveto_path_not_found_retries_nearby_cells():
    agent, bdi, _peer = await make_agent("npc_001", "farmer", [], Bus(), builtin_subplans=False)
    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    agent.send_to_unity = send
    blocked = {"status": "Failure", "errorCode": "PathNotFound"}
    events: list[tuple[str, dict]] = []
    with patch("npc.behaviours.bdi._trace", lambda ev, **kw: events.append((ev, kw))):
        first = bdi._retry_moveto_nearby((-11, 5), 0, blocked)
        second = bdi._retry_moveto_nearby((-11, 5), 1, blocked)
        assert bdi._retry_moveto_nearby((-11, 5), 9, blocked) is None
        assert bdi._retry_moveto_nearby((-11, 5), 0, {"status": "Failure", "errorCode": "timeout"}) is None
    await asyncio.sleep(0)
    assert first and second and first != second
    assert [m["args"] for m in sent] == [{"x": -11, "y": 5}, {"x": -10, "y": 5}]
    retries = [kw for ev, kw in events if ev == "moveto_retry"]
    assert [r["retry"] for r in retries] == [[-11, 5], [-10, 5]] and retries[0]["source"] == "CODE"
    # Con un item en el destino, solo se reintenta la misma casilla (PickUp la necesita).
    agent.beliefs.apply_item_at("flour", 2, 2, current_tick=0)
    assert bdi._retry_moveto_nearby((2, 2), 0, blocked)
    assert bdi._retry_moveto_nearby((2, 2), 1, blocked) is None


@pytest.mark.asyncio
async def test_moveto_waiter_resends_instead_of_failing():
    from npc.behaviours.bdi import ActionResultWaiter
    agent, bdi, _peer = await make_agent("npc_001", "farmer", [], Bus(), builtin_subplans=False)

    async def send(msg):
        pass

    agent.send_to_unity = send

    class _Intention:
        instr = "x"

    intention = _Intention()
    waiter = ActionResultWaiter(
        cmd_id="c1", results=agent.action_results, intention=intention, npc_id="npc_001",
        goal_sig="g", action_type="MoveTo", bdi=bdi, move_target=(-11, 5),
    )
    agent.action_results["c1"] = {"status": "Failure", "errorCode": "PathNotFound"}
    with patch("npc.behaviours.bdi._trace", lambda ev, **kw: None):
        assert waiter.poll(None) is False
    assert intention.instr == "x" and bdi._pending_failure is None
    assert waiter._cmd_id != "c1" and waiter._move_attempt == 1
    agent.action_results[waiter._cmd_id] = {"status": "Success"}
    with patch("npc.behaviours.bdi._trace", lambda ev, **kw: None):
        assert waiter.poll(None) is True
    assert bdi._pending_failure is None


async def _goal_whose_plan_cannot_be_built(condition, replan_count=0):
    agent, bdi, _peer = await make_agent("npc_001", "farmer", [], Bus(), builtin_subplans=False)
    goal = Goal(sig="achieve_collect_wheat", priority=1.0)
    goal.success_condition = condition
    goal.replan_count = replan_count
    agent.goals.append(goal)

    async def run_llm_task(task: dict):
        raise ValueError("LLM validation errors: No executable steps for condition 'x'")

    agent.run_llm_task = run_llm_task
    return agent, bdi, goal


@pytest.mark.asyncio
async def test_plan_generation_failure_uses_the_replan_budget():
    # Fase 17r (piloto 17q, CO5/ATOM): antes el goal moría en el primer fallo del pipeline.
    agent, bdi, goal = await _goal_whose_plan_cannot_be_built("has_item(wheat, 1)")
    events: list[tuple[str, dict]] = []
    with patch("npc.behaviours.bdi._trace", lambda ev, **kw: events.append((ev, kw))):
        await bdi._request_plan(goal)
    node = agent.plan_graph.nodes["achieve_collect_wheat"]["data"]
    assert node.status == NodeStatus.NEEDS_REPLAN
    assert goal.replan_count == 1 and "No executable steps" in goal.replan_hint
    assert any(g is goal for g in agent.goals)
    assert [kw["reason"] for ev, kw in events if ev == "goal_replan_required"] == ["plan_failed"]
    assert [ev for ev, _ in events if ev == "plan_failed"]


@pytest.mark.asyncio
async def test_plan_generation_failure_after_budget_fails_goal():
    agent, bdi, goal = await _goal_whose_plan_cannot_be_built("has_item(wheat, 1)", replan_count=3)
    await bdi._request_plan(goal)
    assert agent.plan_graph.nodes["achieve_collect_wheat"]["data"].status == NodeStatus.FAILED
    assert not any(g is goal for g in agent.goals)


@pytest.mark.asyncio
async def test_plan_generation_failure_without_condition_fails_as_before():
    agent, bdi, goal = await _goal_whose_plan_cannot_be_built(None)
    await bdi._request_plan(goal)
    assert agent.plan_graph.nodes["achieve_collect_wheat"]["data"].status == NodeStatus.FAILED
    assert goal.replan_count == 0


@pytest.mark.asyncio
async def test_action_failure_without_condition_fails_directly_as_before():
    _agent, bdi, goal, node = await _goal_at_failure_limit(None)
    await bdi._handle_action_failure(goal, dict(_PICKUP_FAIL))
    assert node.status == NodeStatus.FAILED
    assert goal.replan_count == 0


def test_craft_catalog_says_qty_counts_ingredient_units():
    # Fase 17j (tanda corta, A3/A4): con la descripción anterior de qty ("defaults to the
    # recipe amount") el LLM mandaba Craft(wheat, Bread_recipe, 1) pensando en panes y
    # Unity respondía "No matching recipe" (qty son unidades del ingrediente).
    from llm.catalogs import ACTIONS_CATALOG

    assert "counts units of the INGREDIENT itemId to consume (NOT units of the output)" in ACTIONS_CATALOG
    assert "it must equal the recipe's input quantity, so omit it" in ACTIONS_CATALOG
    assert "defaults to recipe's required amount" not in ACTIONS_CATALOG
    # Fase 17u (piloto 17t): "produces ItemToCraft" hacía leer itemId como el producto.
    assert "produces the recipe's OUTPUT item in inventory (has_item); consumes the ingredient itemId" in ACTIONS_CATALOG
    assert "ItemToCraft" not in ACTIONS_CATALOG
