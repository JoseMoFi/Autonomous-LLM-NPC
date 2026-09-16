"""Tests de mini_repair (Fase 6.5): preguntar al LLM en vez de fabricar."""

import asyncio

import pytest

from llm.pipeline.mini_repair import repair_plan_gaps, MiniRepairError


def _llm(responses):
    state = {"n": 0}

    async def call(user, system):
        r = responses[state["n"]] if state["n"] < len(responses) else "{}"
        state["n"] += 1
        return r

    return call, state


def test_qty_ausente_se_repara_preguntando_al_llm():
    steps = [{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat"]}]
    call, state = _llm(['{"args": ["wheat", 2]}'])
    out = asyncio.run(repair_plan_gaps(steps, "achieve_has_item", call))
    assert out[0]["args"] == ["wheat", 2]
    assert state["n"] == 1  # se preguntó al LLM


def test_qty_ausente_irreparable_falla_ruidoso():
    steps = [{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat"]}]
    call, _ = _llm(["no hay json aquí"])
    with pytest.raises(MiniRepairError):
        asyncio.run(repair_plan_gaps(steps, "sig", call))


def test_steps_completos_no_llaman_al_llm():
    steps = [{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 2]}]
    call, state = _llm([])
    out = asyncio.run(repair_plan_gaps(steps, "sig", call))
    assert out == steps
    assert state["n"] == 0  # nada que reparar → cero llamadas


def test_trailing_se_quita_solo_si_el_llm_lo_confirma():
    steps = [
        {"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 2]},
        {"type": "action", "name": "MoveTo", "args": [1, 2]},
    ]
    call, _ = _llm(['{"redundant": true}'])
    out = asyncio.run(repair_plan_gaps(steps, "sig", call))
    assert [s["name"] for s in out] == ["move_to_and_pickup"]


def test_trailing_se_mantiene_si_el_llm_lo_niega():
    steps = [
        {"type": "subgoal", "name": "craft_item", "args": ["bread_recipe", 1]},
        {"type": "action", "name": "Drop", "args": ["bread", 1]},
    ]
    call, _ = _llm(['{"redundant": false}'])
    out = asyncio.run(repair_plan_gaps(steps, "sig", call))
    assert len(out) == 2  # no se trunca contenido del LLM


def test_sin_builtin_terminal_no_se_toca():
    steps = [{"type": "action", "name": "MoveTo", "args": [1, 2]}]
    call, state = _llm([])
    out = asyncio.run(repair_plan_gaps(steps, "sig", call))
    assert out == steps
    assert state["n"] == 0


def test_trailing_primitivo_subsumido_si_pregunta_al_llm():
    # Un PickUp crudo tras move_to_and_pickup ES el patrón de redundancia → se
    # pregunta al LLM (subsumido por el builtin).
    steps = [
        {"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 2]},
        {"type": "action", "name": "PickUp", "args": ["wheat", 2]},
    ]
    call, state = _llm(['{"redundant": true}'])
    out = asyncio.run(repair_plan_gaps(steps, "sig", call))
    assert [s["name"] for s in out] == ["move_to_and_pickup"]
    assert state["n"] == 1  # se acotó al caso real → se preguntó


def test_trailing_tras_explore_zone_no_se_pregunta_si_mueve_o_busca():
    # Fase 17j: achieve_explore_zone solo garantiza zone_center; un MoveTo/Search
    # posterior no es redundante y no se pregunta (el LLM "confirmaba" que sobraba
    # y el peldaño "sin observar" quedaba sin acciones).
    steps = [
        {"type": "subgoal", "name": "achieve_explore_zone", "args": ["farmland"]},
        {"type": "action", "name": "MoveTo", "args": [9, -6]},
        {"type": "action", "name": "Search", "args": ["wheat"]},
    ]
    call, state = _llm(['{"redundant": true}'])
    out = asyncio.run(repair_plan_gaps(steps, "sig", call))
    assert [s["name"] for s in out] == ["achieve_explore_zone", "MoveTo", "Search"]
    assert state["n"] == 0


def test_trailing_no_subsumido_se_conserva_sin_llamada():
    # Un Drop tras craft_item NO está subsumido por el builtin → se conserva SIN
    # gastar llamada LLM (acotación del caso de redundancia).
    steps = [
        {"type": "subgoal", "name": "craft_item", "args": ["bread_recipe", 1]},
        {"type": "action", "name": "Drop", "args": ["bread", 1]},
    ]
    call, state = _llm([])
    out = asyncio.run(repair_plan_gaps(steps, "sig", call))
    assert len(out) == 2
    assert state["n"] == 0


def test_trailing_subgoal_se_conserva_sin_llamada():
    # Un subgoal colgante no es el patrón de redundancia (no es acción cruda
    # subsumida) → se conserva sin preguntar al LLM.
    steps = [
        {"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 2]},
        {"type": "subgoal", "name": "craft_item", "args": ["bread_recipe", 1]},
    ]
    call, state = _llm([])
    out = asyncio.run(repair_plan_gaps(steps, "sig", call))
    assert [s["name"] for s in out] == ["move_to_and_pickup", "craft_item"]
    assert state["n"] == 0
