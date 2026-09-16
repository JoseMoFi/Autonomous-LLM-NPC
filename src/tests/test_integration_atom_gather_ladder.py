"""Integración — Fase 17f: la escalera de precondiciones de PickUp en ejecución.

Sin sub-planes (brazo ATOM), step1b desgrana recolectar en los `requires` del
contrato de PickUp: item no observado → observado pero lejos → en su casilla.
Este test ejecuta esos peldaños con el motor agentspeak, BDIBehaviour y el
pipeline REALES (LLM de oráculo, ver `coord_harness.py`) contra un mundo
ESTRICTO que reproduce lo que falló en la tanda 2 real:

  - Search solo observa trigo si el NPC está en farmland, y lo sitúa en casillas
    distintas del centro de la zona;
  - PickUp falla con NotFound si el NPC no está en la casilla del item, y al
    recoger limpia item_at (como unity_events).

No mide al LLM: comprueba que los guards ligan X/Y y que un cuerpo que los usa
llega al objetivo sin un solo PickUp fallido.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from npc.agent import Goal
from protocol.messages import RecipeItemPayload, RecipePayload
from tests.coord_harness import BAKERI, Bus, OracleLLM, install_llm, make_agent, run_until

FARMLAND = (9, -6)
WHEAT_TILES = ((11, -5), (10, -8))


def _strict_world(agent, actions: list[tuple[str, dict, str]]):
    tiles = list(WHEAT_TILES)

    async def _send_to_unity(msg: dict) -> None:
        if msg.get("type") != "ActionCommand":
            return
        action, args, cmd_id = msg["actionType"], dict(msg.get("args", {})), msg["commandId"]
        rows = agent.beliefs.query("current_position")
        here = tuple(rows[0]) if rows else None
        status, error = "Success", None

        if action == "MoveTo":
            agent.beliefs.apply_current_position(int(args["x"]), int(args["y"]))
        elif action == "Search":
            if here == FARMLAND:
                for x, y in tiles:
                    agent.beliefs.apply_item_at(args["itemId"], x, y, current_tick=0)
        elif action == "PickUp":
            item = args["itemId"]
            if here in tiles:
                tiles.remove(here)
                agent.beliefs.clear_item_at(item)
                held = agent.beliefs.query("has_item", item)
                agent.beliefs.apply_has_item(item, (held[0][1] if held else 0) + 1)
            else:
                status, error = "Failure", "NotFound"
        else:
            status, error = "Failure", "unsupported_in_fake_world"

        actions.append((action, args, status))
        agent.action_results[cmd_id] = (
            {"status": "Success", "payload": {}} if status == "Success"
            else {"status": "Failure", "errorCode": error, "errorMessage": "No item on current cell."}
        )

    return _send_to_unity


@contextlib.contextmanager
def _atom_settings():
    values = {
        "coordination_enabled": False,
        "builtin_subplans_enabled": False,
        "canonical_reuse_enabled": False,
        "canonical_family_plan": False,
        "use_refinement": True,
        "isolate_capability_contracts": False,
        "action_timeout_s": 3.0,
    }
    with contextlib.ExitStack() as stack:
        for key, value in values.items():
            stack.enter_context(patch(f"config.settings.{key}", value))
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("qty", [1, 2])
async def test_atom_gather_ladder_moves_to_the_observed_tile(qty: int, tmp_path):
    agent, bdi, peer = await make_agent("npc_001", "farmer", [], Bus(), builtin_subplans=False)
    agent.beliefs.clear_item_at("wheat")                      # nada observado al empezar
    agent.beliefs.apply_zone_discovery("farmland", *FARMLAND)
    actions: list[tuple[str, dict, str]] = []
    agent.send_to_unity = _strict_world(agent, actions)
    oracle = OracleLLM("ATOM")
    install_llm(agent, oracle, tmp_path)

    goal = Goal(sig="achieve_collect_wheat", priority=1.0)
    goal.success_condition = f"has_item(wheat, {qty})"
    goal.expected_condition = goal.success_condition
    agent.goals.append(goal)

    with _atom_settings():
        await run_until(
            [(agent, bdi, peer)],
            lambda: agent.beliefs.has("has_item", "wheat", qty) and not agent.goals,
            max_seconds=60,
        )

    assert agent.beliefs.has("has_item", "wheat", qty)
    # Los tres peldaños derivados del contrato, con X/Y del guard en el cuerpo.
    failing = [f for f, _steps in oracle.plans]
    assert "not item_at(wheat, _, _)" in failing
    assert "not current_position(X, Y)" in failing
    assert f"not has_item(wheat, {qty})" in failing
    # En ejecución: fue a las casillas observadas y ningún PickUp falló.
    pickups = [(a, s) for a, _args, s in actions if a == "PickUp"]
    assert pickups == [("PickUp", "Success")] * qty
    moves = [(int(args["x"]), int(args["y"])) for a, args, _s in actions if a == "MoveTo"]
    assert set(WHEAT_TILES[:qty]) <= set(moves) or len(set(moves) & set(WHEAT_TILES)) == qty


# ===========================================================================
# Fase 17k (A5): tres hornadas con una receta de 1 pan
# ===========================================================================

BREAD_RECIPE = RecipePayload(
    recipeId="Bread_recipe", zone="bakeri",
    inputs=[RecipeItemPayload(itemId="wheat", qty=2)],
    outputs=[RecipeItemPayload(itemId="bread", qty=1)],
)


def _bakery_world(agent, actions: list[tuple[str, dict, str]]):
    """Mundo estricto de A5: el trigo se repone al buscar si no queda, PickUp solo en
    la casilla del item y Craft solo en bakeri con 2 trigos (qty, si viene, debe ser 2)."""
    tiles = list(WHEAT_TILES)

    def have(item: str) -> int:
        rows = agent.beliefs.query("has_item", item)
        return rows[0][1] if rows else 0

    async def _send_to_unity(msg: dict) -> None:
        if msg.get("type") != "ActionCommand":
            return
        action, args, cmd_id = msg["actionType"], dict(msg.get("args", {})), msg["commandId"]
        rows = agent.beliefs.query("current_position")
        here = tuple(rows[0]) if rows else None
        status, error = "Success", None

        if action == "MoveTo":
            x, y = int(args["x"]), int(args["y"])
            agent.beliefs.apply_current_position(x, y)
            if (x, y) == BAKERI:
                agent.beliefs.apply_at_zone("bakeri")
            else:
                agent.beliefs.remove_at_zone("bakeri")
        elif action == "Search":
            if here == FARMLAND:
                if not tiles:
                    tiles.extend(WHEAT_TILES)   # el trigo se repone
                for x, y in tiles:
                    agent.beliefs.apply_item_at(args["itemId"], x, y, current_tick=0)
        elif action == "PickUp":
            if here in tiles:
                tiles.remove(here)
                agent.beliefs.clear_item_at(args["itemId"])
                agent.beliefs.apply_has_item(args["itemId"], have(args["itemId"]) + 1)
            else:
                status, error = "Failure", "NotFound"
        elif action == "Craft":
            if args.get("qty") not in (None, 2) or not agent.beliefs.has("at_zone", "bakeri") or have("wheat") < 2:
                status, error = "Failure", "InvalidArgs"
            else:
                agent.beliefs.apply_has_item("wheat", have("wheat") - 2)
                agent.beliefs.apply_has_item("bread", have("bread") + 1)
        else:
            status, error = "Failure", "unsupported_in_fake_world"

        actions.append((action, args, status))
        agent.action_results[cmd_id] = (
            {"status": "Success", "payload": {}} if status == "Success"
            else {"status": "Failure", "errorCode": error, "errorMessage": error}
        )

    return _send_to_unity


@pytest.mark.asyncio
async def test_atom_bakes_three_breads_regathering_between_crafts(tmp_path):
    agent, bdi, peer = await make_agent("npc_001", "baker", [BREAD_RECIPE], Bus(), builtin_subplans=False)
    agent.beliefs.clear_item_at("wheat")
    agent.beliefs.apply_zone_discovery("farmland", *FARMLAND)
    actions: list[tuple[str, dict, str]] = []
    agent.send_to_unity = _bakery_world(agent, actions)
    oracle = OracleLLM("ATOM")
    install_llm(agent, oracle, tmp_path)

    goal = Goal(sig="achieve_bake_bread", priority=1.0)
    goal.success_condition = "has_item(bread, 3)"
    goal.expected_condition = goal.success_condition
    agent.goals.append(goal)

    with _atom_settings():
        await run_until(
            [(agent, bdi, peer)],
            lambda: agent.beliefs.has("has_item", "bread", 3) and not agent.goals,
            max_seconds=90,
        )

    assert agent.beliefs.has("has_item", "bread", 3)
    assert [s for a, _args, s in actions if a == "Craft"] == ["Success"] * 3
    assert [s for a, _args, s in actions if a == "PickUp"] == ["Success"] * 6
