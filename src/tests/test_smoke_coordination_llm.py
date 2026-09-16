"""Smoke tests — Fase 17: coordinación planificada por el LLM (SUB/ATOM/DET).

Cubren (sin LLM ni Unity):
  - step1b: sin coordinación la escalera no cambia; con coordinación, peldaños
    de pedir/recoger para lo que solo da otro NPC, y escalera de entrega con
    peldaño fijo (.drop + .deliver_to_peer).
  - Carga de builtins y sub-planes terminales según sub-planes y coordinación.
  - Prompt de step3: bloque de coordinación solo con coordinación; ATOM sin macros.
  - Acciones de coordinación: primitivas para classify_steps, alias
    normalizados, aridad validada.
  - Apagado por inactividad multi-NPC (registry.work_done / is_idle_for_shutdown).
  - Settings NPC_COORDINATION_PLANNER; suite y manifiestos CO5/CO6; config DET.
  - analyze_ablation con tres brazos y métricas de coordinación.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import analyze_ablation as AA  # noqa: E402
from gateway.registry import NPCRegistry  # noqa: E402
from llm.pipeline.builtins import (  # noqa: E402
    active_terminal_subplans,
    builtin_file_exclusions,
    terminal_subplans,
)
from llm.pipeline.pipeline_runner import (  # noqa: E402
    _lowercase_asl_constants,
    _normalize_step_types,
    _steps_to_asl_body,
)
from llm.pipeline.step1b_decompose import build_need_variants  # noqa: E402
from llm.pipeline.step3_validator import validate_steps  # noqa: E402
from llm.pipeline.step4_classify import classify_steps  # noqa: E402
from llm.prompts.planning import build_prompt  # noqa: E402
from npc.agent import Goal, NPCAgent  # noqa: E402
from npc.builtin_loader import load_builtin_plans  # noqa: E402

_BUILTIN_DIR = _REPO / "src" / "plans" / "builtin"
_EXPERIMENTS = _REPO / "tools" / "experiments"
_MACROS = ("move_to_and_pickup", "craft_item", "obtain_from_peer", "collect_from_peer")

_BREAD = [{"recipeId": "Bread_recipe", "zone": "bakeri",
           "inputs": [{"itemId": "wheat", "qty": 2}], "outputs": [{"itemId": "bread", "qty": 1}]}]
_FLOUR = [{"recipeId": "Flour_recipe", "zone": "bakeri",
           "inputs": [{"itemId": "wheat", "qty": 2}], "outputs": [{"itemId": "flour", "qty": 1}]}]
_BREAD_FROM_FLOUR = [{"recipeId": "Bread_from_flour_recipe", "zone": "bakeri",
                      "inputs": [{"itemId": "flour", "qty": 1}], "outputs": [{"itemId": "bread", "qty": 1}]}]
_SPAWNS = [{"itemId": "wheat", "zones": ["farmland"]}]
_MILLER = [("npc_miller", "miller")]
_BAKER = [("npc_baker", "baker")]


# ===========================================================================
# step1b
# ===========================================================================

def test_step1b_without_coordination_is_unchanged():
    variants = build_need_variants("has_item(bread, 1)", _BREAD, _SPAWNS)
    assert [v.variant_guard for v in variants] == [
        "not has_item(bread, 1) & not has_item(wheat, 2)",
        "not has_item(bread, 1) & has_item(wheat, 2) & not at_zone(bakeri)",
        "not has_item(bread, 1) & has_item(wheat, 2) & at_zone(bakeri)",
    ]
    assert variants[0].problem_nl == (
        "The NPC does not have wheat (needs 2). It can be gathered at farmland. "
        "Gather 2 wheat before crafting bread."
    )
    assert variants[2].problem_nl == (
        "The NPC has all ingredients (has_item(wheat, 2)) and is at bakeri. Use Bread_recipe to craft bread."
    )
    assert variants[2].known_facts == [
        "recipe_output(Bread_recipe, bakeri, bread, 1)", "recipe_input(Bread_recipe, wheat, 2)",
    ]
    assert [v.unsatisfied_condition for v in variants] == [
        "not has_item(wheat, 2)", "not at_zone(bakeri)", "not has_item(bread, 1)",
    ]
    assert all(not v.bound_variables and not v.fixed_steps for v in variants)
    # Sin coordinación los peers se ignoran y no hay escalera de entrega.
    assert build_need_variants("has_item(bread, 1)", _BREAD_FROM_FLOUR, _SPAWNS, peers=_MILLER) == \
        build_need_variants("has_item(bread, 1)", _BREAD_FROM_FLOUR, _SPAWNS)
    assert build_need_variants("delivered_to_peer(npc_baker, flour, 1)", _FLOUR, _SPAWNS) == []


def test_co5_baker_ladder_asks_a_peer_for_flour():
    variants = build_need_variants(
        "has_item(bread, 1)", _BREAD_FROM_FLOUR, _SPAWNS, coordination=True, peers=_MILLER,
    )
    assert [v.variant_guard for v in variants] == [
        "not has_item(bread, 1) & not has_item(flour, 1) & not peer_item_available(_, flour, _, _, _)",
        "not has_item(bread, 1) & not has_item(flour, 1) & peer_item_available(P, flour, Q, X, Y)",
        "not has_item(bread, 1) & has_item(flour, 1) & not at_zone(bakeri)",
        "not has_item(bread, 1) & has_item(flour, 1) & at_zone(bakeri)",
    ]
    assert "npc_miller" in variants[0].problem_nl
    assert variants[0].known_facts == ["peer(npc_miller, miller)"]
    assert variants[1].bound_variables == ["P", "Q", "X", "Y"]
    assert not any(v.fixed_steps for v in variants)


# item_spawns tal como los manda Unity (Builds_Fase13_test): flour/bread son craft-only.
_UNITY_SPAWNS = [
    {"itemId": "wheat", "zones": ["farmland"], "targetCount": 4, "weight": 1},
    {"itemId": "bread", "zones": ["none"], "targetCount": 0, "weight": 1},
    {"itemId": "flour", "zones": ["none"], "targetCount": 0, "weight": 1},
    {"itemId": "yeast", "zones": [], "targetCount": 0, "weight": 0},
]


@pytest.mark.parametrize("builtin_subplans", [True, False])
def test_unity_none_spawn_sentinel_is_not_a_gather_zone(builtin_subplans: bool):
    # Fase 17j (tanda corta, A5): "none" se tomaba por zona real y flour parecía
    # recolectable → no había peldaños de pedirlo al peer.
    variants = build_need_variants(
        "has_item(bread, 1)", _BREAD_FROM_FLOUR, _UNITY_SPAWNS,
        builtin_subplans=builtin_subplans, coordination=True, peers=_MILLER,
    )
    guards = [v.variant_guard for v in variants]
    assert any("not peer_item_available(_, flour, _, _, _)" in g for g in guards)
    assert any("peer_item_available(P, flour, Q, X, Y)" in g for g in guards)
    assert not any("item_at(flour" in g for g in guards)
    assert not any("none" in fact for v in variants for fact in v.known_facts)


def test_co6_miller_top_goal_only_reachable_through_a_peer():
    variants = build_need_variants("has_item(bread, 1)", _FLOUR, _SPAWNS, coordination=True, peers=_BAKER)
    assert [v.variant_guard for v in variants] == [
        "not has_item(bread, 1) & not peer_item_available(_, bread, _, _, _)",
        "not has_item(bread, 1) & peer_item_available(P, bread, Q, X, Y)",
    ]
    # Sin peers conocidos o con un item recolectable, camino de variante única.
    assert build_need_variants("has_item(bread, 1)", _FLOUR, _SPAWNS, coordination=True, peers=[]) == []
    assert build_need_variants("has_item(wheat, 2)", _FLOUR, _SPAWNS, coordination=True, peers=_BAKER) == []


def test_delivery_ladder_obtains_the_item_and_ends_with_a_fixed_handover():
    variants = build_need_variants(
        "delivered_to_peer(npc_baker, flour, 1)", _FLOUR, _SPAWNS, coordination=True, peers=_BAKER,
    )
    prefix = "not delivered_to_peer(npc_baker, flour, 1)"
    assert [v.variant_guard for v in variants] == [
        f"{prefix} & not has_item(flour, 1) & not has_item(wheat, 2)",
        f"{prefix} & not has_item(flour, 1) & has_item(wheat, 2) & not at_zone(bakeri)",
        f"{prefix} & not has_item(flour, 1) & has_item(wheat, 2) & at_zone(bakeri)",
        f"{prefix} & has_item(flour, 1)",
    ]
    assert variants[2].unsatisfied_condition == "not has_item(flour, 1)"
    assert variants[-1].fixed_steps == [
        {"type": "action", "name": "Drop", "args": ["flour", 1]},
        {"type": "action", "name": "deliver_to_peer", "args": ["npc_baker", "flour", 1]},
    ]
    assert _steps_to_asl_body(variants[-1].fixed_steps) == [
        ".drop(flour, 1)", ".deliver_to_peer(npc_baker, flour, 1)",
    ]


def test_co6_baker_delivery_asks_back_for_flour():
    variants = build_need_variants(
        "delivered_to_peer(npc_miller, bread, 1)", _BREAD_FROM_FLOUR, _SPAWNS,
        coordination=True, peers=_MILLER,
    )
    guards = [v.variant_guard for v in variants]
    assert len(variants) == 5
    assert guards[0].endswith("not has_item(flour, 1) & not peer_item_available(_, flour, _, _, _)")
    assert "peer_item_available(P, flour, Q, X, Y)" in guards[1]
    assert guards[-1] == "not delivered_to_peer(npc_miller, bread, 1) & has_item(bread, 1)"


# ===========================================================================
# builtins
# ===========================================================================

def test_builtin_exclusions_and_terminal_subplans():
    assert builtin_file_exclusions(True, False) == {"obtain_from_peer.asl"}
    assert builtin_file_exclusions(False, False) == {"obtain_from_peer.asl", "move_to_and_pickup.asl", "craft_item.asl"}
    assert builtin_file_exclusions(True, True) == frozenset()
    assert builtin_file_exclusions(False, True) == {
        "move_to_and_pickup.asl", "craft_item.asl", "obtain_from_peer.asl", "collect_from_peer.asl",
    }
    assert active_terminal_subplans(True, False) == terminal_subplans(True)
    assert {"obtain_from_peer", "collect_from_peer"} <= active_terminal_subplans(True, True)
    assert active_terminal_subplans(False, True) == frozenset({"achieve_explore_zone"})

    loaded = load_builtin_plans(nx.DiGraph(), _BUILTIN_DIR, exclude_files=builtin_file_exclusions(True, True))
    assert "obtain_from_peer" in loaded
    atom = load_builtin_plans(nx.DiGraph(), _BUILTIN_DIR, exclude_files=builtin_file_exclusions(False, True))
    assert not set(_MACROS) & set(atom)


# ===========================================================================
# prompt de step3
# ===========================================================================

def _payload(**overrides) -> dict:
    payload = {
        "task": "step3_steps",
        "goal_name": "achieve_bake_bread",
        "npc_statement": "Bake 1 bread",
        "neg_guard": "not has_item(bread, 1) & not has_item(flour, 1) & not peer_item_available(_, flour, _, _, _)",
        "unsatisfied_condition": "not has_item(flour, 1)",
        "problem_nl": "The NPC needs 1 flour. Other NPCs it can ask: npc_miller.",
        "known_facts": ["peer(npc_miller, miller)"],
        "existing_subgoals": ["achieve_explore_zone"],
        "entity_catalog": {"item_ids": ["flour", "bread"], "zone_ids": ["bakeri"]},
        "atomic_only": True,
    }
    payload.update(overrides)
    return payload


def test_prompt_without_coordination_is_identical():
    base = build_prompt(_payload())
    assert build_prompt(_payload(coordination=False, peers=[["npc_miller", "miller"]])) == base
    assert "request_peer" not in base[0] + base[1]


@pytest.mark.parametrize("atomic_only", [True, False])
def test_prompt_with_coordination_lists_peer_actions(atomic_only: bool):
    user, _system = build_prompt(_payload(
        atomic_only=atomic_only, coordination=True, peers=[["npc_miller", "miller"]],
    ))
    for name in ("ask_peer", "request_peer", "await_peer"):
        assert name in user
    assert "npc_miller (miller)" in user


def test_atom_coordination_prompt_has_no_macros():
    user, system = build_prompt(_payload(
        coordination=True, builtin_subplans=False, peers=[["npc_miller", "miller"]],
    ))
    for name in _MACROS:
        assert name not in user + system


def test_sub_coordination_prompt_labels_the_coordination_macros():
    user, _system = build_prompt(_payload(
        coordination=True, peers=[["npc_miller", "miller"]],
        existing_subgoals=["obtain_from_peer", "collect_from_peer"],
    ))
    assert "obtain_from_peer(peerId, itemId, qty)" in user
    assert "collect_from_peer(peerId, itemId, qty)" in user


@pytest.mark.parametrize("builtin_subplans", [True, False])
def test_completeness_rule_allows_await_peer_only_with_coordination(builtin_subplans: bool):
    # Fase 17n (piloto CO5/ATOM): la regla 13 exigía PickUp/Craft/Drop y el LLM
    # recolectaba trigo en vez de pedir la harina.
    base = {
        "task": "step3_steps", "goal_name": "achieve_bake_bread", "npc_statement": "Bake 1 bread",
        "neg_guard": "not has_item(bread, 1) & not has_item(flour, 1) & not peer_item_available(_, flour, _, _, _)",
        "unsatisfied_condition": "not has_item(flour, 1)", "problem_nl": "needs flour",
        "known_facts": [], "existing_subgoals": [], "entity_catalog": {},
        "atomic_only": True, "builtin_subplans": builtin_subplans,
    }
    plain, _ = build_prompt(dict(base))
    coord, _ = build_prompt(dict(base, coordination=True, peers=[["npc_miller", "miller"]]))
    assert "await_peer" not in plain
    assert "13. If the failing condition requires has_item, the plan must end with" in plain
    assert "Coordination guarantees" not in plain
    # Fase 17q: sin "otherwise" que obligue a PickUp/Craft/Drop en los peldaños peer_*.
    assert "otherwise" not in coord
    assert "or with await_peer when another NPC must produce the item" in coord
    assert "If it requires peer_promised, end with request_peer" in coord
    assert "if it requires peer_item_available, end with await_peer" in coord
    assert "Coordination guarantees: request_peer → peer_promised" in coord
    assert "Effect when it agrees: peer_promised(NpcId, GoalSig)" in coord


def test_rule7_is_replaced_only_for_peer_failing_conditions():
    # Fase 17q: la regla 7 (MoveTo + Search) no aplica a peer_promised/peer_item_available.
    base = {
        "task": "step3_steps", "goal_name": "achieve_bake_bread", "npc_statement": "Bake 1 bread",
        "neg_guard": "not has_item(bread, 1) & not has_item(flour, 1) & not peer_promised(_, _)",
        "unsatisfied_condition": "not peer_promised(_, _)", "problem_nl": "needs flour",
        "known_facts": [], "existing_subgoals": [], "entity_catalog": {},
        "atomic_only": True, "builtin_subplans": False,
        "coordination": True, "peers": [["npc_miller", "miller"]],
    }
    peer, _ = build_prompt(dict(base))
    assert "7. The failing condition is about another NPC" in peer
    assert "7. If zone_center(ZoneTag" not in peer
    item, _ = build_prompt(dict(base, unsatisfied_condition="not has_item(flour, 1)"))
    assert "7. If zone_center(ZoneTag, X, Y) is already known" in item
    assert "about another NPC" not in item
    # Sin coordinación la regla 7 no cambia aunque la condición sea peer_*.
    plain, _ = build_prompt(dict(base, coordination=False))
    assert "about another NPC" not in plain
    # Fase 17r: regla 1 igual (solo en peldaños peer_* con coordinación).
    assert "1. The reusable sub-goals above cover item acquisition" in peer
    assert "1. PREFER a reusable sub-goal" not in peer
    assert "1. PREFER a reusable sub-goal" in item
    assert "1. PREFER a reusable sub-goal" in plain


def test_refuse_withdraws_the_promise_of_that_request():
    from npc.beliefs import BeliefStore
    beliefs = BeliefStore()
    beliefs.apply_peer_promised("npc_baker", "achieve_has_item")
    beliefs.apply_peer_promised("npc_baker", "achieve_other")
    beliefs.apply_peer_refused("npc_baker", "achieve_has_item", "already_failed")
    assert not beliefs.has("peer_promised", "npc_baker", "achieve_has_item")
    assert beliefs.has("peer_promised", "npc_baker", "achieve_other")


def test_peer_done_waiter_stops_on_refusal():
    from npc.beliefs import BeliefStore
    from npc.behaviours.bdi import PeerDoneWaiter
    agent = SimpleNamespace(beliefs=BeliefStore())
    resolved: list = []
    waiter = PeerDoneWaiter("npc_baker", "achieve_has_item", agent,
                            lambda ok, reason: resolved.append((ok, reason)), timeout_s=300)
    assert waiter.poll(None) is False
    agent.beliefs.apply_peer_refused("npc_baker", "achieve_has_item", "already_failed")
    assert waiter.poll(None) is True
    assert resolved == [(False, "refused:already_failed")]


@pytest.mark.asyncio
async def test_await_peer_without_a_promise_fails_at_once():
    # Fase 17x (tanda coop 17w, CO6/ATOM): esperaba 300 s a un encargo rechazado.
    from unittest.mock import patch
    import agentspeak
    from tests.coord_harness import Bus, make_agent
    agent, bdi, _peer = await make_agent("npc_miller", "miller", [], Bus(), builtin_subplans=False)
    intention = SimpleNamespace(scope={}, waiter=None)
    term = agentspeak.Literal("await_peer", (
        agentspeak.Literal("npc_baker"), agentspeak.Literal("achieve_has_item"), 300,
    ))
    action = bdi._actions.lookup(".await_peer", 3)
    events: list = []
    with patch("npc.behaviours.bdi._trace", lambda ev, **kw: events.append((ev, kw))):
        for _ in action(bdi._asp, term, intention):
            pass
    assert intention.waiter is None
    assert bdi._pending_failure == "peer_wait_failed:not_promised"
    # Con la promesa, espera como antes.
    bdi._pending_failure = None
    agent.beliefs.apply_peer_promised("npc_baker", "achieve_has_item")
    for _ in action(bdi._asp, term, intention):
        pass
    assert intention.waiter is not None and bdi._pending_failure is None


def test_craft_item_must_be_an_ingredient_of_its_recipe():
    from llm.pipeline.step3_steps import wrong_craft_ingredients
    facts = ["recipe_input(Bread_from_flour_recipe, flour, 1)", "recipe(flour_recipe, bakeri, wheat, 2)"]
    bad = [{"type": "action", "name": "Craft", "args": ["bread", "Bread_from_flour_recipe", 1]}]
    assert wrong_craft_ingredients(bad, facts) == [("bread", "Bread_from_flour_recipe", ["flour"])]
    good = [{"type": "action", "name": "Craft", "args": {"itemId": "wheat", "targetId": "Flour_recipe"}}]
    assert wrong_craft_ingredients(good, facts) == []
    # Receta desconocida: no se comprueba.
    assert wrong_craft_ingredients(
        [{"type": "action", "name": "Craft", "args": ["x", "Other_recipe"]}], facts,
    ) == []


@pytest.mark.asyncio
async def test_step3_asks_again_when_craft_uses_the_product():
    from llm.pipeline.step3_steps import run_step3
    answers = iter([
        {"steps": [{"type": "action", "name": "Craft", "args": ["bread", "Bread_from_flour_recipe", 1]}]},
        {"steps": [{"type": "action", "name": "Craft", "args": ["flour", "Bread_from_flour_recipe", 1]}]},
    ])
    users: list[str] = []

    async def llm(user: str, _system: str) -> str:
        users.append(user)
        return json.dumps(next(answers))

    steps = await run_step3(
        "achieve_bake_bread", "d", "not has_item(bread, 1) & has_item(flour, 1) & at_zone(bakeri)", "p",
        ["recipe_input(Bread_from_flour_recipe, flour, 1)"], [], llm,
        atomic_only=True, builtin_subplans=False,
        already_satisfied=["has_item(flour, 1)", "at_zone(bakeri)"],
        unsatisfied_condition="not has_item(bread, 1)",
    )
    assert steps[0]["args"][0] == "flour"
    assert "itemId must be the INGREDIENT that Bread_from_flour_recipe consumes (flour), not 'bread'" in users[1]


def test_active_goals_are_not_offered_as_reusable_subgoals():
    # Fase 17w (piloto 17v, CO6): el miller veía achieve_bake_bread como sub-goal.
    import agentspeak
    from agentspeak import runtime as asp_runtime
    from npc.behaviours.bdi import BDIBehaviour
    from npc.plan_graph import GoalNode, NodeStatus
    agent = NPCAgent(jid="npc_miller@localhost", password="x", npc_id="npc_miller",
                     send_to_unity=None, llm_planning_jid="llm@localhost")
    bdi = BDIBehaviour()
    bdi.agent = agent
    for key in ("achieve_explore_zone", "achieve_bake_bread",
                "achieve_deliver_to_peer__npc_baker_flour_1", "achieve_go_there"):
        agent.plan_graph.add_node(key, data=GoalNode(sig=key.split("__")[0], status=NodeStatus.READY))
    agent.goals = [Goal(sig="achieve_bake_bread"),
                   Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])]
    agent.intention = agent.goals[1]
    assert bdi._reusable_known_goals() == ["achieve_explore_zone", "achieve_go_there"]


def test_peer_delivery_goal_has_a_concrete_statement():
    # Fase 17u (piloto 17t): el planificador recibía "Deliver to peer".
    from npc.behaviours.bdi import goal_statement
    delivery = Goal(sig="achieve_deliver_to_peer", call_args=["npc_baker", "flour", 1])
    assert goal_statement(delivery) == (
        "Produce 1 flour for npc_baker and deliver it to them (npc_baker requested it)"
    )
    # El resto de goals no cambia: el planificador sigue derivándolo del sig.
    assert goal_statement(Goal(sig="achieve_bake_bread")) == ""


def test_rung_state_facts_follow_the_rung_guard():
    # Fase 17u (piloto 17t): la rama "tengo el trigo y estoy en bakeri" veía el NPC en (0, 0).
    from llm.pipeline.pipeline_runner import rung_state_facts
    snapshot = [
        "current_position(0, 0)", "at_zone(farmland)", "has_item(bread, 1)",
        "zone_center(bakeri, -11, 5)", "item_at(wheat, 8, -5)",
    ]
    facts = rung_state_facts(snapshot, ["has_item(wheat, 2)", "at_zone(bakeri)"])
    assert facts == [
        "has_item(wheat, 2)", "at_zone(bakeri)", "zone_center(bakeri, -11, 5)", "item_at(wheat, 8, -5)",
    ]
    # Sin nada garantizado, solo se quitan los hechos de estado del NPC.
    assert rung_state_facts(snapshot, []) == ["zone_center(bakeri, -11, 5)", "item_at(wheat, 8, -5)"]
    # Fase 17v: tampoco el estado del protocolo del snapshot; el directorio de peers sí.
    protocol = ["peer(npc_miller, miller)", "peer_can_make(npc_miller, flour)",
                "peer_promised(npc_miller, achieve_has_item)", "peer_failed(npc_miller, g, r)"]
    assert rung_state_facts(protocol, []) == ["peer(npc_miller, miller)"]


def test_peer_rungs_must_use_the_action_that_guarantees_their_condition():
    from llm.pipeline.step3_steps import missing_peer_action
    explore = {"type": "subgoal", "name": "achieve_explore_zone", "args": ["bakeri"]}
    request = {"type": "action", "name": "request_peer", "args": ["npc_miller", "g", "flour", 1]}
    wait = {"type": "action", "name": "await_peer", "args": ["npc_miller", "g", 300]}
    waiting = "not peer_item_available(P, flour, _, _, _)"
    assert missing_peer_action([request, explore], waiting) == ("peer_item_available", "await_peer")
    assert missing_peer_action([request, wait], waiting) is None
    assert missing_peer_action([explore], "not peer_promised(_, _)") == ("peer_promised", "request_peer")
    assert missing_peer_action([request], "not peer_promised(_, _)") is None
    # Fuera de los peldaños del protocolo no aplica.
    assert missing_peer_action([explore], "not has_item(flour, 1)") is None


@pytest.mark.asyncio
async def test_step3_asks_again_when_the_wait_rung_does_not_await():
    from llm.pipeline.step3_steps import run_step3
    answers = iter([
        {"steps": [{"type": "action", "name": "request_peer", "args": ["npc_miller", "g", "flour", 1]},
                   {"type": "subgoal", "name": "achieve_explore_zone", "args": ["bakeri"]}]},
        {"steps": [{"type": "action", "name": "await_peer", "args": ["npc_miller", "g", 300]}]},
    ])
    users: list[str] = []

    async def llm(user: str, _system: str) -> str:
        users.append(user)
        return json.dumps(next(answers))

    steps = await run_step3(
        "achieve_bake_bread", "d", "not has_item(flour, 1) & peer_promised(P, G)", "p", [],
        ["achieve_explore_zone"], llm, atomic_only=True, builtin_subplans=False, coordination=True,
        unsatisfied_condition="not peer_item_available(P, flour, _, _, _)",
    )
    assert [s["name"] for s in steps] == ["await_peer"]
    assert "only await_peer guarantees" in users[1]


def test_regathering_an_item_the_guard_already_holds_is_detected():
    # Fase 17t (piloto 17s, CO5): rama con has_item(wheat, 2) & at_zone(bakeri).
    from llm.pipeline.step3_steps import regathered_satisfied_items
    bad = [
        {"type": "subgoal", "name": "achieve_explore_zone", "args": ["farmland"]},
        {"type": "action", "name": "MoveTo", "args": [9, -6]},
        {"type": "action", "name": "Search", "args": ["wheat"]},
        {"type": "action", "name": "PickUp", "args": {"itemId": "wheat"}},
        {"type": "action", "name": "Craft", "args": ["wheat", "Flour_recipe", 2]},
    ]
    held = ["has_item(wheat, 2)", "at_zone(bakeri)"]
    assert regathered_satisfied_items(bad, held, "not has_item(flour, 1)") == ["wheat"]
    good = [{"type": "action", "name": "Craft", "args": ["wheat", "Flour_recipe", 2]}]
    assert regathered_satisfied_items(good, held, "not has_item(flour, 1)") == []
    # Si falta MÁS del mismo item, recoger es correcto.
    assert regathered_satisfied_items(bad, ["has_item(wheat, 1)"], "not has_item(wheat, 2)") == []
    assert regathered_satisfied_items(bad, [], "not has_item(flour, 1)") == []


@pytest.mark.asyncio
async def test_step3_asks_again_when_the_plan_regathers_a_held_item():
    from llm.pipeline.step3_steps import run_step3
    answers = iter([
        {"steps": [{"type": "action", "name": "PickUp", "args": ["wheat"]},
                   {"type": "action", "name": "Craft", "args": ["wheat", "Flour_recipe", 2]}]},
        {"steps": [{"type": "action", "name": "Craft", "args": ["wheat", "Flour_recipe", 2]}]},
    ])
    users: list[str] = []

    async def llm(user: str, _system: str) -> str:
        users.append(user)
        return json.dumps(next(answers))

    steps = await run_step3(
        "achieve_deliver_to_peer", "d",
        "not has_item(flour, 1) & has_item(wheat, 2) & at_zone(bakeri)", "p", [], [], llm,
        atomic_only=True, builtin_subplans=False,
        already_satisfied=["has_item(wheat, 2)", "at_zone(bakeri)"],
        unsatisfied_condition="not has_item(flour, 1)",
    )
    assert [s["name"] for s in steps] == ["Craft"]
    assert "Do NOT Search or PickUp wheat again" in users[1]


@pytest.mark.asyncio
async def test_receiver_refuses_a_delivery_that_already_failed_twice():
    # Fase 17t: sin tope, el mismo encargo fallido se repetía hasta el timeout.
    from unittest.mock import AsyncMock
    from npc.behaviours.peer_coord import PeerCoordBehaviour
    from protocol.peer_messages import PeerMessage
    agent = NPCAgent(jid="npc_miller@localhost", password="x", npc_id="npc_miller",
                     send_to_unity=None, llm_planning_jid="llm@localhost")
    agent.beliefs.apply_item_spawn("wheat", "farmland")
    agent.beliefs.apply_recipe("Flour_recipe", "bakeri", [{"itemId": "wheat", "qty": 2}],
                               [{"itemId": "flour", "qty": 1}])
    coord = PeerCoordBehaviour()
    coord.agent = agent
    coord._agree = AsyncMock()
    coord._refuse = AsyncMock()
    request = PeerMessage(performative="request", conversation_id="c", goal_sig="achieve_has_item",
                          condition="has_item(flour, 1)", depth=1)
    agent.failed_peer_deliveries["achieve_deliver_to_peer__npc_baker_flour_1"] = 1
    await coord._handle_request(request, "npc_baker@localhost", "npc_baker")
    assert coord._agree.await_count == 1 and coord._refuse.await_count == 0
    agent.goals.clear()
    agent.failed_peer_deliveries["achieve_deliver_to_peer__npc_baker_flour_1"] = 2
    await coord._handle_request(request, "npc_baker@localhost", "npc_baker")
    assert coord._refuse.await_count == 1
    assert coord._refuse.await_args.args[-1] == "already_failed"


@pytest.mark.asyncio
async def test_new_peer_request_replans_a_failed_delivery():
    # Fase 17s (piloto 17r): el receptor heredaba el nodo FAILED y cerraba el goal en 0,1 s.
    from npc.plan_graph import GoalNode, NodeStatus
    agent = NPCAgent(jid="npc_miller@localhost", password="x", npc_id="npc_miller",
                     send_to_unity=None, llm_planning_jid="llm@localhost")
    key = "achieve_deliver_to_peer__npc_baker_flour_1"
    node = GoalNode(sig="achieve_deliver_to_peer", status=NodeStatus.FAILED)
    node.failure_count = 3
    agent.plan_graph.add_node(key, data=node)
    agent._adopt_goal_from_trigger(
        "achieve_deliver_to_peer", condition="delivered_to_peer(npc_baker, flour, 1)",
        origin="peer", call_args=["npc_baker", "flour", 1],
    )
    assert node.status == NodeStatus.NEEDS_REPLAN and node.failure_count == 0
    assert agent.goals and agent.goals[-1].replan_count == 0
    # Un trigger (no peer) no toca el nodo.
    node.status = NodeStatus.FAILED
    agent.goals.clear()
    agent._adopt_goal_from_trigger(
        "achieve_deliver_to_peer", condition="delivered_to_peer(npc_baker, flour, 1)",
        origin="trigger", call_args=["npc_baker", "flour", 1],
    )
    assert node.status == NodeStatus.FAILED


def test_own_goal_with_binding_is_not_a_reusable_subgoal():
    # Fase 17r (piloto 17q, CO6): el receptor veía `achieve_deliver_to_peer__npc_miller_bread_1`.
    from llm.pipeline.step3_steps import is_same_goal_sig
    assert is_same_goal_sig("achieve_deliver_to_peer", "achieve_deliver_to_peer")
    assert is_same_goal_sig("achieve_deliver_to_peer__npc_miller_bread_1", "achieve_deliver_to_peer")
    assert not is_same_goal_sig("achieve_deliver_to_peer_fast", "achieve_deliver_to_peer")
    assert not is_same_goal_sig("achieve_explore_zone", "achieve_deliver_to_peer")


@pytest.mark.asyncio
async def test_step3_rejects_own_goal_with_binding_as_subgoal():
    from llm.pipeline.step3_steps import run_step3
    goal = "achieve_deliver_to_peer"
    answers = iter([
        {"steps": [{"type": "subgoal", "name": f"{goal}__npc_miller_bread_1", "args": []}]},
        {"steps": [{"type": "action", "name": "Drop", "args": ["bread", 1]}]},
    ])
    users: list[str] = []

    async def llm(user: str, _system: str) -> str:
        users.append(user)
        return json.dumps(next(answers))

    steps = await run_step3(
        goal, "d", "not delivered_to_peer(npc_miller, bread, 1)", "p", [],
        [f"{goal}__npc_miller_bread_1"], llm, atomic_only=True, builtin_subplans=False,
    )
    assert [s["name"] for s in steps] == ["Drop"]
    assert "infinite recursion" in users[1]


def test_atom_peer_rungs_split_the_request_protocol():
    # Fase 17o: recoger → esperar → reintentar → pedir, solo sin sub-planes.
    rungs = build_need_variants(
        "has_item(bread, 1)", _BREAD_FROM_FLOUR, _SPAWNS,
        builtin_subplans=False, coordination=True, peers=_MILLER,
    )
    base = "not has_item(bread, 1) & not has_item(flour, 1)"
    flour = [v for v in rungs if v.variant_guard.startswith(base)]
    no_delivery = "not peer_item_available(_, flour, _, _, _)"
    assert [v.variant_guard for v in flour] == [
        f"{base} & peer_item_available(P, flour, Q, X, Y)",
        f"{base} & {no_delivery} & peer_promised(P, G) & not peer_failed(P, G, _)",
        f"{base} & {no_delivery} & peer_promised(P, G) & peer_failed(P, G, R)",
        f"{base} & {no_delivery} & not peer_promised(_, _)",
    ]
    assert [v.unsatisfied_condition for v in flour] == [
        "not has_item(flour, 1)", "not peer_item_available(P, flour, _, _, _)",
        "not peer_promised(_, _)", "not peer_promised(_, _)",
    ]
    assert [v.bound_variables for v in flour] == [["P", "Q", "X", "Y"], ["P", "G"], ["P", "G", "R"], []]
    # Ningún peldaño del protocolo pide resolver has_item salvo el de recoger.
    assert all("has_item" not in v.unsatisfied_condition for v in flour[1:])
    # Fase 17p: hechos del estado (qué puede conseguir el NPC solo), sin nombrar acciones.
    note = ("Gathering wheat or crafting cannot give this NPC flour: none of its recipes produces "
            "flour (they produce bread). Only another NPC can provide the flour.")
    for rung in flour[1:]:
        assert note in rung.problem_nl
        assert not any(name in rung.problem_nl for name in (
            "request_peer", "await_peer", "ask_peer", "MoveTo", "Search", "PickUp", "Craft(",
        ))
    # SUB: sin cambios (obtain_from_peer hace pedir → esperar → recoger).
    sub = build_need_variants("has_item(bread, 1)", _BREAD_FROM_FLOUR, _SPAWNS, coordination=True, peers=_MILLER)
    assert len([v for v in sub if v.variant_guard.startswith(base)]) == 2
    assert not any("cannot give this NPC" in v.problem_nl for v in sub)


def test_new_agreement_forgets_the_previous_request_outcome():
    # Fase 17o: un peer_failed viejo hacía fallar al instante el await_peer de la petición nueva.
    from npc.beliefs import BeliefStore

    beliefs = BeliefStore()
    beliefs.apply_peer_failed("npc_baker", "achieve_has_item", "node_failed")
    beliefs.apply_peer_done("npc_baker", "achieve_has_item")
    beliefs.apply_peer_refused("npc_baker", "achieve_has_item", "busy")
    beliefs.apply_peer_failed("npc_miller", "achieve_has_item", "timeout")
    beliefs.clear_peer_request("npc_baker", "achieve_has_item")
    assert not beliefs.query("peer_failed", "npc_baker", "achieve_has_item", None)
    assert not beliefs.query("peer_refused", "npc_baker", "achieve_has_item", None)
    # Una entrega ya hecha sigue siendo válida (SUB/DET reenvían al retomar una intención).
    assert beliefs.has("peer_done", "npc_baker", "achieve_has_item")
    # Otro peer: intacto.
    assert beliefs.query("peer_failed", "npc_miller", "achieve_has_item", None)


def test_simulator_credits_obtain_from_peer_with_explicit_peer():
    from llm.pipeline.plan_simulator import active_subgoal_guarantees, check_plan_reachability

    steps = [{"type": "subgoal", "name": "obtain_from_peer", "args": ["npc_miller", "flour", 1]}]
    guarantees = active_subgoal_guarantees(True, True)
    assert check_plan_reachability(steps, {}, "has_item(flour, 1)", subgoal_guarantees=guarantees).reachable


# ===========================================================================
# acciones de coordinación en el pipeline
# ===========================================================================

def test_peer_actions_are_primitives_and_names_are_normalized():
    steps = [
        {"type": "subgoal", "name": "AskPeer", "args": ["npc_miller", "can_make", "flour"]},
        {"type": "action", "name": "request-peer", "args": ["npc_miller", "achieve_has_item", "flour", 1]},
    ]
    _normalize_step_types(steps, set())
    assert [s["name"] for s in steps] == ["ask_peer", "request_peer"]
    assert all(s["type"] == "action" for s in steps)
    primitives, to_expand = classify_steps(steps)
    assert len(primitives) == 2 and to_expand == []
    assert _steps_to_asl_body(steps) == [
        ".ask_peer(npc_miller, can_make, flour)",
        ".request_peer(npc_miller, achieve_has_item, flour, 1)",
    ]


def test_peer_action_arity_is_validated():
    errors = validate_steps(
        [{"type": "action", "name": "await_peer", "args": ["npc_miller", "achieve_has_item"]}], [], [], set(),
    )
    assert errors and "await_peer" in errors[0].errors[0]


# ===========================================================================
# apagado por inactividad con varios NPCs
# ===========================================================================

class _FakeAgent:
    def __init__(self, had_goals: bool, idle: bool) -> None:
        self._had_goals = had_goals
        self._idle = idle

    def is_idle_for_shutdown(self) -> bool:
        return self._idle


def test_work_done_with_a_helper_npc_without_own_goals():
    registry = NPCRegistry(llm_planning_jid="llm@localhost")
    assert registry.work_done() is False
    registry._agents = {"npc_baker": _FakeAgent(True, True), "npc_miller": _FakeAgent(False, True)}
    assert registry.work_done() is True
    registry._agents["npc_miller"] = _FakeAgent(False, False)   # sigue entregando
    assert registry.work_done() is False
    registry._agents = {"a": _FakeAgent(False, True), "b": _FakeAgent(False, True)}
    assert registry.work_done() is False                        # nadie tuvo goals todavía


def test_is_idle_for_shutdown_rules():
    agent = NPCAgent(
        jid="npc_x@localhost", password="x", npc_id="npc_x",
        send_to_unity=None, llm_planning_jid="llm@localhost",
    )
    assert agent.is_idle_for_shutdown() is False                 # sin perfil aún
    agent.profile = SimpleNamespace(goals_nl=[])
    assert agent.is_idle_for_shutdown() is True                  # ayudante sin goals propios
    agent.profile = SimpleNamespace(goals_nl=["Bake 1 @bread"])
    assert agent.is_idle_for_shutdown() is False                 # todavía parseando
    agent._had_goals = True
    assert agent.is_idle_for_shutdown() is True
    agent.goals.append(Goal(sig="achieve_deliver_to_peer"))
    assert agent.is_idle_for_shutdown() is False


# ===========================================================================
# settings, suite, manifiestos y configs
# ===========================================================================

def test_settings_coordination_planner_env(monkeypatch):
    from config import Settings

    saved = Settings._instance
    try:
        Settings._instance = None
        monkeypatch.setenv("NPC_COORDINATION_PLANNER", "llm")
        assert Settings.load().coordination_planner == "llm"
        Settings._instance = None
        monkeypatch.setenv("NPC_COORDINATION_PLANNER", "otra_cosa")
        assert Settings.load().coordination_planner == "family"
    finally:
        Settings._instance = saved


def test_coordination_suite_manifests_and_configs_are_consistent():
    suite = json.loads((_EXPERIMENTS / "suites" / "coordination_builtins.json").read_text(encoding="utf-8"))
    assert suite["configs"] == ["SUB", "ATOM", "DET"]
    assert suite["runs"] % len(suite["configs"]) == 0
    assert set(suite["ablated_subplans"]) == set(_MACROS)
    for exp_id in suite["experiments"]:
        manifest = json.loads((_EXPERIMENTS / f"{exp_id}.json").read_text(encoding="utf-8"))
        assert manifest["npcs"] == 2 and manifest["configs"] == suite["configs"]
        assert manifest["env"]["NPC_COORDINATION"] == "1"
        assert manifest["env"]["NPC_ARBITRATION_MODE"] == "rule"
        # El escenario no puede fijar la variable del brazo.
        assert "NPC_COORDINATION_PLANNER" not in manifest["env"]
        assert "NPC_BUILTIN_SUBPLANS" not in manifest["env"]
        assert manifest["build_exe"] == "builds\\coop\\My project.exe"

    configs = json.loads((_EXPERIMENTS / "configs.json").read_text(encoding="utf-8"))
    assert configs["DET"]["env"]["NPC_COORDINATION_PLANNER"] == "family"
    assert configs["DET"]["env"]["NPC_CANONICAL_FAMILY"] == "1"
    assert configs["SUB"]["env"]["NPC_COORDINATION_PLANNER"] == "llm"
    assert configs["ATOM"]["env"]["NPC_COORDINATION_PLANNER"] == "llm"


# ===========================================================================
# analyze_ablation con tres brazos
# ===========================================================================

def test_analyzer_three_arms_and_coordination_metrics(tmp_path):
    manifests = {"CO5": {"id": "CO5", "goals": {"npc_baker": [{"nl": "Bake 1 @bread", "condition": "has_item(bread, 1)"}]}}}
    suite = {
        "id": "c", "experiments": ["CO5"], "configs": ["SUB", "ATOM", "DET"],
        "config_builtin_subplans": {"SUB": True, "ATOM": False, "DET": True},
        "ablated_subplans": list(_MACROS),
    }

    def session(name: str, t: float, cfg: str, ok: bool) -> None:
        loaded = ["achieve_explore_zone"] + ([] if cfg == "ATOM" else list(_MACROS))
        events = [
            {"t": t, "ev": "session_start", "experiment_id": "CO5", "config_label": cfg,
             "builtin_subplans": cfg != "ATOM", "git_sha": "x"},
            {"t": t, "ev": "unity_in", "npc": "npc_miller", "msg_type": "RegisterNPC"},
            {"t": t, "ev": "unity_in", "npc": "npc_baker", "msg_type": "RegisterNPC"},
            {"t": t + 1, "ev": "builtin_plans_loaded", "npc": "npc_baker", "sigs": loaded},
            {"t": t + 2, "ev": "peer_request_sent", "npc": "npc_baker", "depth": 1},
            {"t": t + 3, "ev": "peer_request_accepted", "npc": "npc_miller"},
            {"t": t + 4, "ev": "action_sent", "npc": "npc_miller", "action": "PickUp"},
        ]
        if ok:
            events += [
                {"t": t + 5, "ev": "peer_transfer", "npc": "npc_miller"},
                {"t": t + 9, "ev": "goal_completed", "npc": "npc_baker",
                 "expected_condition": "has_item(bread, 1)", "goal_belief_met": True},
                {"t": t + 10, "ev": "shutdown_idle"},
            ]
        else:
            events += [
                {"t": t + 50, "ev": "goal_failed_after_replans", "npc": "npc_baker", "reason": "ladder_stuck"},
                {"t": t + 51, "ev": "shutdown_idle"},
            ]
        events.append({"t": t + 52, "ev": "session_end", "duration_s": 52})
        d = tmp_path / name
        d.mkdir()
        (d / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    session("s1", 100.0, "SUB", True)
    session("s2", 200.0, "ATOM", False)
    session("s3", 300.0, "DET", True)

    summaries, agg, comparisons = AA.analyze(tmp_path, suite, manifests)
    assert len(summaries) == 3
    assert all(s["manipulation_ok"] is True for s in summaries)
    assert {(c["base"], c["treatment"]) for c in comparisons if c["experiment_id"] == "CO5"} == {
        ("SUB", "ATOM"), ("SUB", "DET"), ("ATOM", "DET"),
    }
    assert agg[("CO5", "SUB")]["sessions_with_transfer"] == 1
    assert agg[("CO5", "ATOM")]["sessions_with_transfer"] == 0
    assert agg[("CO5", "SUB")]["max_peer_depth"] == 1
    md = AA.to_markdown(summaries, agg, comparisons, suite)
    assert "Coordinación entre NPCs" in md and "SUB vs ATOM" in md


def test_world_constants_are_lowercased_before_compiling_asl():
    # `.craft(wheat, Bread_recipe)` compilaba con Bread_recipe como VARIABLE sin ligar.
    steps = [
        {"type": "action", "name": "Craft", "args": ["wheat", "Bread_recipe"]},
        {"type": "action", "name": "MoveTo", "args": ["X", "Y"]},
        {"type": "subgoal", "name": "craft_item", "args": ["Bread_recipe", 1]},
        {"type": "action", "name": "Drop", "args": {"itemId": "flour"}},
    ]
    _lowercase_asl_constants(steps)
    assert steps[0]["args"] == ["wheat", "bread_recipe"]
    assert steps[1]["args"] == ["X", "Y"]
    assert steps[2]["args"] == ["Bread_recipe", 1]   # sub-goal: intacto (unifica sin error)
    assert _steps_to_asl_body(steps[:1]) == [".craft(wheat, bread_recipe)"]
