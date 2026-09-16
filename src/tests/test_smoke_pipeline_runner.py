"""Smoke tests — pipeline_runner orquestador (v3, 7 pasos).

Cubre (sin LLM real):
  - run_step0 (step0_name): sig + description
  - PipelineResult / PlanVariant / SubgoalEntry / ContingencyPlan serialización
  - run_full_pipeline: camino feliz (paso0→1→bucle[2→3→4→5]→paso6 recursivo)
  - subgoals_to_expand correctamente poblado
  - paso 6: sub_results con recursión
  - _steps_to_asl_body helper
  - description pre-passed skips paso 0
  - Compat: main_asl / contingency_plans / steps / facts / guards properties
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.pipeline.pipeline_runner import (
    PipelineResult,
    PlanVariant,
    SubgoalEntry,
    ContingencyPlan,
    _steps_to_asl_body,
    run_full_pipeline,
)
from llm.pipeline.step0_reasoning import run_step0


# ===========================================================================
# Fixtures — respuestas de LLM sintéticas
# ===========================================================================

_STEP0_RESP = '{"sig": "achieve_craft_bread", "description": "The NPC must craft bread at the bakery zone."}'

_STEP1_RESP = (
    '{"success_model": ['
    '{"facts": ["has_item(bread, 1)"], "guards": [], "done_fragment": "has_item(bread, 1)"}'
    '], '
    '"success_conditions": ["has_item(bread, 1)"], '
    '"done_guard": "has_item(bread, 1)", '
    '"done_asl": "+!achieve_craft_bread : has_item(bread, 1) <- true."}'
)

_STEP1_LEGACY_RESP = '{"success_conditions": ["has_item(bread, 1)"], "done_guard": "has_item(bread, 1)", "done_asl": "+!achieve_craft_bread : has_item(bread, 1) <- true."}'

_STEP2_RESP = '{"problem_nl": "The NPC does not have bread yet.", "known_facts": ["item_spawn(wheat, farmland)"]}'

# Steps with concrete args so step4 validator passes without needing repair
_STEP3_RESP = '{"steps": [{"type": "action", "name": "MoveTo", "args": [10, 5]}, {"type": "action", "name": "Craft", "args": ["wheat", "bread_recipe"]}]}'

# step4 (repair) is only invoked when step3 produces semantically invalid steps.
# For the happy-path tests, step3 produces valid steps so step4 is a no-op (0 LLM calls).


def _make_sequential_llm(*responses: str):
    """Devuelve un mock LLM que responde en secuencia."""
    it = iter(responses)

    async def _llm(user_prompt: str, system_prompt: str) -> str:
        return next(it, "{}")

    return _llm




class TestRunStep0:
    @pytest.mark.asyncio
    async def test_valid_response_returned(self) -> None:
        resp = '{"sig": "achieve_craft_bread", "description": "The NPC must craft bread."}'
        llm = _make_sequential_llm(resp)
        result = await run_step0("craft bread", llm)
        assert result is not None
        assert result["sig"] == "achieve_craft_bread"
        assert "bread" in result["description"].lower()

    @pytest.mark.asyncio
    async def test_invalid_json_raises_value_error(self) -> None:
        llm = _make_sequential_llm("not valid json", "also not json")
        with pytest.raises(ValueError):
            await run_step0("craft bread", llm)

    @pytest.mark.asyncio
    async def test_missing_description_retried(self) -> None:
        resp_bad  = '{"sig": "achieve_craft_bread", "description": ""}'
        resp_good = '{"sig": "achieve_craft_bread", "description": "The NPC must craft bread."}'
        llm = _make_sequential_llm(resp_bad, resp_good)
        result = await run_step0("craft bread", llm)
        assert result["description"] != ""

    @pytest.mark.asyncio
    async def test_npc_profile_keyword_accepted(self) -> None:
        resp = '{"sig": "achieve_craft_sword", "description": "The smith must forge a sword."}'
        llm = _make_sequential_llm(resp)
        result = await run_step0("forge a sword", llm, npc_profile={"role": "blacksmith"})
        assert result["sig"] == "achieve_craft_sword"


# ===========================================================================
# _steps_to_asl_body helper
# ===========================================================================

class TestStepsToAslBody:
    def test_action_step(self) -> None:
        steps = [{"type": "action", "name": "MoveTo", "args": ["10", "20"]}]
        body = _steps_to_asl_body(steps)
        assert body == [".moveto(10, 20)"]

    def test_subgoal_step(self) -> None:
        steps = [{"type": "subgoal", "name": "achieve_gather_wheat", "args": []}]
        body = _steps_to_asl_body(steps)
        assert body == ["!achieve_gather_wheat"]

    def test_subgoal_with_args(self) -> None:
        steps = [{"type": "subgoal", "name": "achieve_gather", "args": ["2", "wheat"]}]
        body = _steps_to_asl_body(steps)
        assert body == ["!achieve_gather(2, wheat)"]

    def test_mixed_steps(self) -> None:
        steps = [
            {"type": "action",  "name": "MoveTo",               "args": ["X", "Y"]},
            {"type": "subgoal", "name": "achieve_gather_wheat", "args": []},
            {"type": "action",  "name": "Craft",                  "args": ["wheat", "recipe"]},
        ]
        body = _steps_to_asl_body(steps)
        assert body[0] == ".moveto(X, Y)"
        assert body[1] == "!achieve_gather_wheat"
        assert body[2] == ".craft(wheat, recipe)"

    def test_empty_steps(self) -> None:
        assert _steps_to_asl_body([]) == []



# ===========================================================================
# PipelineResult / PlanVariant / SubgoalEntry / ContingencyPlan serialización
# ===========================================================================

class TestPipelineResultSerialization:
    def _make_result(self) -> PipelineResult:
        v_done = PlanVariant(
            guard="has_item(bread, 1)",
            steps=[],
            asl="+!achieve_craft_bread : has_item(bread, 1) <- true.",
        )
        v_exec = PlanVariant(
            guard="not has_item(bread, 1)",
            steps=[{"type": "action", "name": "Craft", "args": ["bread_recipe"]}],
            asl="+!achieve_craft_bread : not has_item(bread, 1) <- .craft(bread_recipe) .",
        )
        return PipelineResult(
            sig="achieve_craft_bread",
            description="Craft bread at bakery.",
            success_conditions=["has_item(bread, 1)"],
            variants=[v_done, v_exec],
            success_model=[{"facts": ["has_item(bread, 1)"], "guards": [], "done_fragment": "has_item(bread, 1)", "bound_variables": []}],
            subgoals_to_expand=[],
            dag_nodes=[{"id": "achieve_craft_bread", "status": "main"}],
            dag_edges=[],
        )

    def test_to_dict_keys(self) -> None:
        d = self._make_result().to_dict()
        assert d["sig"] == "achieve_craft_bread"
        assert "variants" in d
        assert "success_conditions" in d
        assert "success_model" in d
        assert "dag" in d
        assert "main_asl" in d  # compat property

    def test_main_asl_contains_sig(self) -> None:
        result = self._make_result()
        assert "+!achieve_craft_bread" in result.main_asl

    def test_steps_property_returns_exec_variant(self) -> None:
        result = self._make_result()
        assert len(result.steps) == 1
        assert result.steps[0]["name"] == "Craft"

    def test_compat_facts_guards_contingency_empty(self) -> None:
        result = self._make_result()
        assert result.facts == []
        assert result.guards == []
        assert result.contingency_plans == []

    def test_subgoal_entry_to_dict(self) -> None:
        se = SubgoalEntry(sig="achieve_gather_wheat", description="Gather wheat.")
        d = se.to_dict()
        assert d["sig"] == "achieve_gather_wheat"
        assert d["description"] == "Gather wheat."

    def test_contingency_plan_stub(self) -> None:
        cp = ContingencyPlan(
            fact={"functor": "has_item", "args": ["wheat", "N"]},
            guard_expression="has_item(wheat, N) & N >= 2",
            steps=[],
            asl="",
        )
        d = cp.to_dict()
        assert d["guard_expression"] == "has_item(wheat, N) & N >= 2"

    def test_to_dict_is_json_serializable(self) -> None:
        import json
        result = self._make_result()
        json.dumps(result.to_dict())  # must not raise


# ===========================================================================
# run_full_pipeline — camino feliz
# ===========================================================================

@pytest.mark.asyncio
async def test_run_full_pipeline_happy_path() -> None:
    """Pipeline completo: step0→step1(1 cond)→step2→step3."""
    llm = _make_sequential_llm(
        _STEP0_RESP,   # step0_name
        _STEP1_RESP,   # step1_success
        _STEP2_RESP,   # step2_problem for cond 0
        _STEP3_RESP,   # step3_steps for cond 0
    )

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=[],
        llm_call=llm,
    )

    assert result.sig == "achieve_craft_bread"
    assert len(result.success_conditions) == 1
    assert len(result.success_model) == 1
    assert len(result.variants) == 2        # done + 1 exec
    assert "+!achieve_craft_bread" in result.main_asl
    assert len(result.steps) == 2          # exec variant steps
    assert result.variants[1].facts == ["has_item(bread, 1)"]
    assert result.dag_nodes[0]["id"] == "achieve_craft_bread"


@pytest.mark.asyncio
async def test_run_full_pipeline_uses_explicit_zone_explore_capability_when_available() -> None:
    llm = _make_sequential_llm(
        '{"steps": ['
        '{"type": "action", "name": "ExploreArea", "args": ["farmland"]}, '
        '{"type": "action", "name": "Search", "args": ["wheat"]}, '
        '{"type": "action", "name": "PickUp", "args": ["wheat"]}'
        ']}'
    )

    result = await run_full_pipeline(
        goal_sig="achieve_have_item_wheat",
        npc_statement="Have item wheat",
        existing_goals=["achieve_explore_zone", "move_to_and_pickup"],
        llm_call=llm,
        description="The NPC must collect item wheat in order to fulfill the objective.",
        use_refinement=True,
        success_condition="has_item(wheat, 1)",
        beliefs={
            "item_spawn": [["wheat", "farmland"]],
        },
    )

    assert result.steps == [
        {
            "type": "subgoal",
            "name": "achieve_explore_zone",
            "args": ["farmland"],
            "description": "Ensure the center coordinates of zone farmland are known.",
        },
        {
            "type": "subgoal",
            "name": "move_to_and_pickup",
            "args": ["wheat", 1],
            "description": "Obtain at least 1 units of wheat.",
        },
    ]


@pytest.mark.asyncio
async def test_run_full_pipeline_description_skips_step0() -> None:
    """Si description se pasa directamente, step0 no se invoca."""
    calls: list[str] = []

    async def counting_llm(user_prompt: str, system_prompt: str) -> str:
        calls.append(user_prompt[:40])
        if len(calls) == 1:
            return _STEP1_RESP
        if len(calls) == 2:
            return _STEP2_RESP
        if len(calls) == 3:
            return _STEP3_RESP
        return "{}"

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=[],
        llm_call=counting_llm,
        description="The NPC must craft bread at the bakery.",
    )

    # step0 would be 1st call → only 3 calls expected (step1+step2+step3)
    assert len(calls) == 3
    assert result.sig == "achieve_craft_bread"
    assert len(result.variants) == 2


@pytest.mark.asyncio
async def test_run_full_pipeline_step0_failure_continues() -> None:
    """Si step0 falla, el pipeline sigue usando npc_statement como descripción."""
    call_count = {"n": 0}

    async def llm(user_prompt: str, system_prompt: str) -> str:
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return "not valid json"   # step0 → 2 intentos = ValueError → capturado
        if call_count["n"] == 3:
            return _STEP1_RESP
        if call_count["n"] == 4:
            return _STEP2_RESP
        if call_count["n"] == 5:
            return _STEP3_RESP
        return "{}"

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=[],
        llm_call=llm,
    )

    assert result.sig == "achieve_craft_bread"
    assert len(result.steps) == 2



@pytest.mark.asyncio
async def test_reachability_repair_verifies_against_success_condition() -> None:
    """0.A3 — Con success_condition presente, un plan inalcanzable dispara el
    repair y la verificación POST-repair se evalúa contra success_condition.

    El borrador (Search) no garantiza inventario → inalcanzable contra
    has_item(bread, 1). El repair devuelve un Craft(bread, ...) que sí lo
    garantiza, por lo que debe aceptarse y sustituir al borrador.
    """
    async def llm(user_prompt: str, system_prompt: str) -> str:
        # step0 se omite (description provista). Orden: step1, step2, step3, repair.
        if "success_model" in system_prompt or "success_conditions" in system_prompt:
            return (
                '{"success_model": [{"facts": ["has_item(bread, 1)"], "guards": [], '
                '"done_fragment": "has_item(bread, 1)"}], '
                '"success_conditions": ["has_item(bread, 1)"], '
                '"done_guard": "has_item(bread, 1)", '
                '"done_asl": "+!achieve_craft_bread : has_item(bread, 1) <- true."}'
            )
        if "problem_nl" in system_prompt:
            return _STEP2_RESP
        if "repair" in system_prompt.lower() or "missing" in system_prompt.lower():
            return '{"steps": [{"type": "action", "name": "Craft", "args": ["bread", "bread_recipe", 1]}]}'
        # step3 (borrador inalcanzable)
        return '{"steps": [{"type": "action", "name": "Search", "args": ["bread"]}]}'

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="craft bread",
        existing_goals=[],
        llm_call=llm,
        description="The NPC must craft bread.",
        use_refinement=True,
        success_condition="has_item(bread, 1)",
        beliefs={"recipe": [["bread_recipe", "bakery", "wheat", 2]]},
    )

    # El repair (Craft) debe haber sustituido al borrador (Search) en los steps.
    step_names = {s.get("name") for s in result.steps}
    assert "Craft" in step_names
    assert "Search" not in step_names


@pytest.mark.asyncio
async def test_run_full_pipeline_fails_on_empty_exec_steps() -> None:
    """Si step3/step4 deja una variante sin steps, el pipeline falla en fail-fast."""
    step3_only_subgoal = (
        '{"steps": ['
        '{"type": "subgoal", "name": "achieve_collect_wheat", "args": []}'
        ']}'
    )

    llm = _make_sequential_llm(
        _STEP0_RESP,
        _STEP1_RESP,
        _STEP2_RESP,
        step3_only_subgoal,  # intento 1 de step3: inválido (solo subgoals)
        "{}",               # intento 2 de step3: inválido (steps vacío)
    )

    with pytest.raises(ValueError, match="No executable steps"):
        await run_full_pipeline(
            goal_sig="achieve_craft_bread",
            npc_statement="I want to craft bread",
            existing_goals=[],
            llm_call=llm,
        )


@pytest.mark.asyncio
async def test_run_full_pipeline_subgoals_discovered() -> None:
    """Sub-goals en steps deben aparecer en subgoals_to_expand."""
    step1_one_cond = (
        '{"success_model": ['
        '{"facts": ["has_item(bread, 1)"], "guards": [], "done_fragment": "has_item(bread,1)"}'
        '], '
        '"success_conditions": ["has_item(bread,1)"], '
        '"done_guard": "has_item(bread,1)", '
        '"done_asl": "+!achieve_craft_bread : has_item(bread,1) <- true."}'
    )
    step2_resp = '{"problem_nl": "No bread.", "known_facts": []}'
    steps_with_subgoal = '{"steps": [{"type": "action", "name": "MoveTo", "args": ["bakery"], "description": "Go to bakery."}, {"type": "subgoal", "name": "achieve_gather_wheat", "args": [], "description": "Gather wheat."}]}'

    llm = _make_sequential_llm(
        _STEP0_RESP,
        step1_one_cond,
        step2_resp,
        steps_with_subgoal,
    )

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=[],
        llm_call=llm,
    )

    sigs = [s.sig for s in result.subgoals_to_expand]
    assert "achieve_gather_wheat" in sigs


@pytest.mark.asyncio
async def test_run_full_pipeline_existing_goals_not_in_to_expand() -> None:
    """Goals ya existentes no se re-añaden a subgoals_to_expand."""
    step1 = (
        '{"success_model": ['
        '{"facts": ["has_item(bread, 1)"], "guards": [], "done_fragment": "has_item(bread,1)"}'
        '], '
        '"success_conditions": ["has_item(bread,1)"], '
        '"done_guard": "has_item(bread,1)", '
        '"done_asl": "+!achieve_craft_bread : has_item(bread,1) <- true."}'
    )
    step2 = '{"problem_nl": "No bread.", "known_facts": []}'
    steps_resp = '{"steps": [{"type": "action", "name": "MoveTo", "args": ["bakery"], "description": "Go to bakery."}, {"type": "subgoal", "name": "achieve_gather_wheat", "args": []}]}'

    llm = _make_sequential_llm(
        _STEP0_RESP,
        step1,
        step2,
        steps_resp,
    )

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=["achieve_gather_wheat"],   # ya existe
        llm_call=llm,
    )

    sigs = [s.sig for s in result.subgoals_to_expand]
    assert "achieve_gather_wheat" not in sigs


@pytest.mark.asyncio
async def test_run_full_pipeline_duplicate_subgoal_not_duplicated() -> None:
    """El mismo sub-goal no debe reencolarse dos veces aunque aparezca repetido."""
    step1 = (
        '{"success_model": ['
        '{"facts": ["has_item(bread, 1)"], "guards": [], "done_fragment": "has_item(bread,1)"}'
        '], '
        '"success_conditions": ["has_item(bread,1)"], '
        '"done_guard": "has_item(bread,1)", '
        '"done_asl": "+!achieve_craft_bread : has_item(bread,1) <- true."}'
    )
    step2 = '{"problem_nl": "No bread.", "known_facts": []}'
    steps_resp = (
        '{"steps": ['
        '{"type": "action", "name": "MoveTo", "args": ["bakery"], "description": "Go."},'
        '{"type": "subgoal", "name": "achieve_gather_wheat", "args": []},'
        '{"type": "subgoal", "name": "achieve_gather_wheat", "args": []}'
        ']}'
    )

    llm = _make_sequential_llm(_STEP0_RESP, step1, step2, steps_resp)

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=[],
        llm_call=llm,
    )

    sigs = [s.sig for s in result.subgoals_to_expand]
    assert sigs == ["achieve_gather_wheat"]


@pytest.mark.asyncio
async def test_run_full_pipeline_cycle_subgoal_discarded() -> None:
    """step3 rechaza planes con el mismo goal como sub-goal (ciclo directo).

    En la nueva semántica, el ciclo se detecta en step3 (no en el DAG):
    el primer intento es rechazado con un mensaje de error y el LLM recibe
    una segunda oportunidad con un plan válido.
    """
    step1 = (
        '{"success_model": ['
        '{"facts": ["has_item(bread, 1)"], "guards": [], "done_fragment": "has_item(bread,1)"}'
        '], '
        '"success_conditions": ["has_item(bread,1)"], '
        '"done_guard": "has_item(bread,1)", '
        '"done_asl": "+!achieve_craft_bread : has_item(bread,1) <- true."}'
    )
    step2 = '{"problem_nl": "No bread.", "known_facts": []}'
    # Intento 1: ciclo — step3 lo rechaza y reintenta
    steps_cycle = (
        '{"steps": ['
        '{"type": "action", "name": "MoveTo", "args": [1, 2]}, '
        '{"type": "subgoal", "name": "achieve_craft_bread", "args": []}'
        ']}'
    )
    # Intento 2: plan válido sin ciclo
    steps_valid = _STEP3_RESP

    llm = _make_sequential_llm(_STEP0_RESP, step1, step2, steps_cycle, steps_valid)

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=[],
        llm_call=llm,
    )

    # Plan final no debe contener el sub-goal autorreferencial
    assert all(
        s.get("name") != "achieve_craft_bread"
        for s in result.steps
        if s.get("type") == "subgoal"
    )
    assert result.subgoals_to_expand == []


@pytest.mark.asyncio
async def test_run_full_pipeline_two_conditions() -> None:
    """Pipeline con 2 success_conditions genera 3 variantes (done + 2 exec)."""
    step1_two = (
        '{"success_model": ['
        '{"facts": ["has_item(wheat,1)"], "guards": [], "done_fragment": "has_item(wheat,1)"},'
        '{"facts": ["at_location(bakery)"], "guards": [], "done_fragment": "at_location(bakery)"}'
        '], '
        '"success_conditions": ["has_item(wheat,1)", "at_location(bakery)"],'
        ' "done_guard": "has_item(wheat,1) & at_location(bakery)",'
        ' "done_asl": "+!achieve_craft_bread : has_item(wheat,1) & at_location(bakery) <- true."}'
    )
    step2a = '{"problem_nl": "No wheat.", "known_facts": []}'
    step3a = '{"steps": [{"type": "action", "name": "PickUp", "args": ["wheat"]}]}'
    step2b = '{"problem_nl": "Not at bakery.", "known_facts": []}'
    step3b = '{"steps": [{"type": "action", "name": "MoveTo", "args": [5, 3]}]}'

    llm = _make_sequential_llm(
        _STEP0_RESP,
        step1_two,
        step2a, step3a,
        step2b, step3b,
    )

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=[],
        llm_call=llm,
    )

    assert len(result.success_conditions) == 2
    assert len(result.success_model) == 2
    assert len(result.variants) == 3    # done + cond1 + cond2
    assert "+!achieve_craft_bread" in result.main_asl


# ===========================================================================
# Paso 4 — step4_map: validación + repair contra catálogo
# ===========================================================================

@pytest.mark.asyncio
async def test_step4_valid_steps_no_llm_call() -> None:
    """Paso 4 no invoca LLM si los steps ya son válidos."""
    from llm.pipeline.step4_map import run_step4

    calls = []

    async def counting_llm(user: str, system: str) -> str:
        calls.append(user)
        return "{}"

    # MoveTo con args concretos → válido en el catálogo
    steps = [
        {"type": "action",  "name": "MoveTo", "args": [10, 5]},
        {"type": "action",  "name": "Craft",  "args": ["wheat", "bread_recipe"]},
    ]
    result = await run_step4("achieve_craft_bread", steps, [], counting_llm)

    assert result == steps          # sin cambios
    assert len(calls) == 0          # no se invocó el LLM


@pytest.mark.asyncio
async def test_step4_invalid_variable_triggers_repair() -> None:
    """Paso 4 invoca repair LLM cuando hay variables sin ligar."""
    from llm.pipeline.step4_map import run_step4

    repair_resp = '{"fixes": [{"index": 0, "step": {"type": "action", "name": "MoveTo", "args": [10, 5]}}]}'

    async def llm(user: str, system: str) -> str:
        return repair_resp

    # MoveTo con variables no ligadas → validator debe señalarlo
    steps = [{"type": "action", "name": "MoveTo", "args": ["X", "Y"]}]
    result = await run_step4("achieve_craft_bread", steps, [], llm)

    # Resultado debe tener args concretos tras repair
    assert result[0]["args"] == [10, 5]


@pytest.mark.asyncio
async def test_step4_atomic_only_allows_subgoals() -> None:
    """Paso 4 debe permitir subgoals incluso en modo atomic_only (validación solo aplica a actions)."""
    from llm.pipeline.step4_map import run_step4

    async def llm(_user: str, _system: str) -> str:
        return "{}"

    steps = [{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 1]}]

    result = await run_step4("achieve_craft_bread", steps, ["move_to_and_pickup"], llm, atomic_only=True)
    assert result == steps


@pytest.mark.asyncio
async def test_step5_refine_generates_new_subplan_when_no_capability() -> None:
    """Si no hay capability reusable, step5_refine puede crear sub-plan con LLM."""
    from llm.pipeline.step5_refine import run_step5_refine

    async def llm(_user: str, _system: str) -> str:
        return (
            '{"subgoal": {'
            '"sig": "achieve_collect_wheat", '
            '"description": "Collect wheat needed for crafting.", '
            '"args": [2]'
            '}}'
        )

    steps = [{"type": "action", "name": "Craft", "args": ["wheat", "bread_recipe", 1]}]
    result = await run_step5_refine(
        "achieve_craft_bread",
        steps,
        existing_subgoals=[],
        llm_call=llm,
        beliefs={"recipe": [["bread_recipe", "bakeri", "wheat", 2]]},
    )

    assert len(result.steps) == 2
    assert result.steps[0]["type"] == "subgoal"
    assert result.steps[0]["name"] == "achieve_collect_wheat"
    assert len(result.created_plans) == 1


@pytest.mark.asyncio
async def test_step5_refine_prefers_reusable_capability_over_llm() -> None:
    """Con capability reusable, no debe generar plan nuevo con LLM."""
    from llm.pipeline.step5_refine import run_step5_refine

    calls = {"n": 0}

    async def llm(_user: str, _system: str) -> str:
        calls["n"] += 1
        return '{"subgoal": {"sig": "achieve_collect_wheat", "description": "x", "args": [2]}}'

    steps = [{"type": "action", "name": "Craft", "args": ["wheat", "bread_recipe", 1]}]
    result = await run_step5_refine(
        "achieve_craft_bread",
        steps,
        existing_subgoals=["move_to_and_pickup"],
        llm_call=llm,
        beliefs={"recipe": [["bread_recipe", "bakeri", "wheat", 2]]},
    )

    assert result.steps[0]["name"] == "move_to_and_pickup"
    assert result.created_plans == []
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_step5_refine_uses_persisted_contract_without_llm() -> None:
    """Si existe contrato persistido, se reutiliza sin invocar generación LLM."""
    from llm.pipeline.step5_refine import run_step5_refine

    calls = {"n": 0}

    async def llm(_user: str, _system: str) -> str:
        calls["n"] += 1
        return '{}'

    steps = [{"type": "action", "name": "Craft", "args": ["wheat", "bread_recipe", 1]}]
    contracts = {
        "version": 1,
        "contracts": {
            "achieve_collect_wheat": {
                "description": "Collect wheat.",
                "provides": [
                    {
                        "kind": "inventory_at_least",
                        "constraints": {"item": "wheat", "qty": 2},
                    }
                ],
                "param_names": ["qty"],
                "source": "step5_need_plan",
            }
        },
    }

    result = await run_step5_refine(
        "achieve_craft_bread",
        steps,
        existing_subgoals=[],
        llm_call=llm,
        capability_contracts=contracts,
        beliefs={"recipe": [["bread_recipe", "bakery", "wheat", 2]]},
    )

    assert result.steps[0]["type"] == "subgoal"
    assert result.steps[0]["name"] == "achieve_collect_wheat"
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_run_full_pipeline_refines_craft_with_builtin_capabilities(tmp_path: Path) -> None:
    """Política reuse-first a nivel de pipeline completo.

    Al refinar un plan Craft, los needs (inventario del ingrediente y at_zone)
    se cubren con los builtins reutilizables `move_to_and_pickup` y `craft_item`,
    NO creando un sub-plan nuevo vía LLM. Por tanto no se persiste ningún
    contrato redundante en disco.

    (La creación+persistencia de planes vía LLM cuando NO hay capability se
    cubre en test_step5_refine_generates_new_subplan_when_no_capability y en
    test_upsert_created_plan_contracts_* — ahí sí se ejercita esa ruta.)
    """
    contracts_path = tmp_path / "capability_contracts.json"

    llm = _make_sequential_llm(
        _STEP0_RESP,
        _STEP1_RESP,
        _STEP2_RESP,
        '{"steps": [{"type": "action", "name": "Craft", "args": ["wheat", "bread_recipe", 1]}]}',
    )

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=[],
        llm_call=llm,
        use_refinement=True,
        capability_contracts_path=str(contracts_path),
        beliefs={"recipe": [["bread_recipe", "bakery", "wheat", 2]]},
    )

    expanded = {s.sig for s in result.subgoals_to_expand}
    assert "move_to_and_pickup" in expanded
    assert "craft_item" in expanded
    # Reuse-first: no se inventa achieve_collect_wheat ni se persiste contrato nuevo.
    assert not any(s.sig == "achieve_collect_wheat" for s in result.subgoals_to_expand)
    if contracts_path.exists():
        data = json.loads(contracts_path.read_text(encoding="utf-8"))
        assert "achieve_collect_wheat" not in data.get("contracts", {})


# ===========================================================================
# Paso 6 — recursión: sub_results en PipelineResult
# ===========================================================================

@pytest.mark.asyncio
async def test_run_full_pipeline_paso6_recurse_subgoal() -> None:
    """Paso 6: un sub-goal nuevo genera sub_results con su propio PipelineResult."""
    # Subgoal steps + complete sub-pipeline for achieve_gather_wheat
    step1_main = (
        '{"success_model": ['
        '{"facts": ["has_item(bread,1)"], "guards": [], "done_fragment": "has_item(bread,1)"}'
        '], '
        '"success_conditions": ["has_item(bread,1)"], "done_guard": "has_item(bread,1)", '
        '"done_asl": "+!achieve_craft_bread : has_item(bread,1) <- true."}'
    )
    step2_main = '{"problem_nl": "No bread.", "known_facts": []}'
    step3_main = '{"steps": [{"type": "action", "name": "MoveTo", "args": [5, 10], "description": "Go to bakery."}, {"type": "subgoal", "name": "achieve_gather_wheat", "args": [], "description": "Gather wheat from farmland."}]}'
    # Sub-pipeline for achieve_gather_wheat (paso 6, depth=1)
    # description passed → skips paso 0
    step1_sub  = (
        '{"success_model": ['
        '{"facts": ["has_item(wheat,1)"], "guards": [], "done_fragment": "has_item(wheat,1)"}'
        '], '
        '"success_conditions": ["has_item(wheat,1)"], "done_guard": "has_item(wheat,1)", '
        '"done_asl": "+!achieve_gather_wheat : has_item(wheat,1) <- true."}'
    )
    step2_sub  = '{"problem_nl": "No wheat.", "known_facts": []}'
    step3_sub  = '{"steps": [{"type": "action", "name": "MoveTo", "args": [10, -5]}, {"type": "action", "name": "PickUp", "args": ["wheat", 1]}]}'

    llm = _make_sequential_llm(
        _STEP0_RESP,    # paso 0 main
        step1_main,     # paso 1 main
        step2_main,     # paso 2 main (cond 0)
        step3_main,     # paso 3 main (cond 0) → subgoal achieve_gather_wheat
        # paso 6 → sub-pipeline achieve_gather_wheat (description provided → skip paso 0)
        step1_sub,      # paso 1 sub
        step2_sub,      # paso 2 sub
        step3_sub,      # paso 3 sub
    )

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=[],
        llm_call=llm,
    )

    assert "achieve_gather_wheat" in result.sub_results
    sub = result.sub_results["achieve_gather_wheat"]
    assert sub.sig == "achieve_gather_wheat"
    assert len(sub.variants) == 2   # done + exec
    assert "+!achieve_gather_wheat" in sub.main_asl


@pytest.mark.asyncio
async def test_run_full_pipeline_paso6_no_recurse_existing_goal() -> None:
    """Paso 6: goals ya existentes no se re-expanden."""
    step1  = (
        '{"success_model": ['
        '{"facts": ["has_item(bread,1)"], "guards": [], "done_fragment": "has_item(bread,1)"}'
        '], '
        '"success_conditions": ["has_item(bread,1)"], "done_guard": "has_item(bread,1)", '
        '"done_asl": "+!achieve_craft_bread : has_item(bread,1) <- true."}'
    )
    step2  = '{"problem_nl": "No bread.", "known_facts": []}'
    step3  = (
        '{"steps": ['
        '{"type": "action", "name": "MoveTo", "args": [5, 3]}, '
        '{"type": "subgoal", "name": "achieve_gather_wheat", "args": []}'
        ']}'
    )

    llm = _make_sequential_llm(
        _STEP0_RESP,
        step1,
        step2,
        step3,
        # No more responses needed — achieve_gather_wheat is in existing_goals
    )

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=["achieve_gather_wheat"],
        llm_call=llm,
    )

    assert "achieve_gather_wheat" not in result.sub_results


@pytest.mark.asyncio
async def test_run_full_pipeline_sub_results_in_to_dict() -> None:
    """sub_results aparece en to_dict()."""
    v = PlanVariant(guard="true", steps=[], asl="+!x : true <- true.")
    result = PipelineResult(
        sig="achieve_x",
        description="desc",
        success_conditions=[],
        variants=[v],
    )
    d = result.to_dict()
    assert "sub_results" in d
    assert d["sub_results"] == {}


@pytest.mark.asyncio
async def test_run_full_pipeline_step1_legacy_response_still_supported() -> None:
    """El runner acepta responses legacy de step1 y deriva success_model localmente."""
    llm = _make_sequential_llm(
        _STEP0_RESP,
        _STEP1_LEGACY_RESP,
        _STEP2_RESP,
        _STEP3_RESP,
    )

    result = await run_full_pipeline(
        goal_sig="achieve_craft_bread",
        npc_statement="I want to craft bread",
        existing_goals=[],
        llm_call=llm,
    )

    assert len(result.success_model) == 1
    assert result.success_model[0]["facts"] == ["has_item(bread, 1)"]



# ===========================================================================
# Iter 6 — deteccion de needs basada en contratos (_detect_unmet_needs)
# ===========================================================================

class TestStep5DetectUnmetNeeds:
    """Tests unitarios de _detect_unmet_needs usando CONTRACT_REGISTRY."""

    def _needs(self, steps, beliefs=None):
        from llm.pipeline.step5_refine import _detect_unmet_needs
        return _detect_unmet_needs(steps, beliefs or {})

    # --- Craft ---

    def test_craft_genera_inventory_at_least_para_ingrediente(self) -> None:
        steps = [{"type": "action", "name": "Craft", "args": ["wheat", "bread_recipe", 1]}]
        beliefs = {"recipe": [["bread_recipe", "bakeri", "wheat", 2]]}
        needs = self._needs(steps, beliefs)
        inv = [n for n in needs if n.kind == "inventory_at_least"]
        assert len(inv) == 1
        assert inv[0].payload["item"] == "wheat"
        assert inv[0].payload["qty"] == 2

    def test_craft_sin_receta_no_genera_need(self) -> None:
        """Si no hay receta en beliefs, Craft no puede derivar ingrediente."""
        steps = [{"type": "action", "name": "Craft", "args": ["wheat", "unknown_recipe"]}]
        needs = self._needs(steps, {})
        inv = [n for n in needs if n.kind == "inventory_at_least"]
        assert len(inv) == 0

    def test_craft_con_inventario_suficiente_no_genera_need(self) -> None:
        steps = [{"type": "action", "name": "Craft", "args": ["wheat", "bread_recipe"]}]
        beliefs = {
            "recipe": [["bread_recipe", "bakeri", "wheat", 1]],
            "has_item": [["wheat", 5]],
        }
        needs = self._needs(steps, beliefs)
        inv = [n for n in needs if n.kind == "inventory_at_least"]
        assert len(inv) == 0

    # --- PickUp ---

    def test_pickup_genera_inventory_at_least(self) -> None:
        steps = [{"type": "action", "name": "PickUp", "args": ["wheat"]}]
        needs = self._needs(steps, {})
        inv = [n for n in needs if n.kind == "inventory_at_least"]
        assert len(inv) == 1
        assert inv[0].payload["item"] == "wheat"
        assert inv[0].payload["qty"] == 1

    def test_pickup_fuente_step_es_pickup(self) -> None:
        steps = [{"type": "action", "name": "PickUp", "args": ["wheat"]}]
        needs = self._needs(steps, {})
        inv = [n for n in needs if n.kind == "inventory_at_least"]
        assert inv[0].source_step["name"] == "PickUp"

    # --- ExploreArea ---

    def test_explore_genera_know_zone_si_zona_desconocida(self) -> None:
        steps = [{"type": "action", "name": "ExploreArea", "args": ["farmland"]}]
        needs = self._needs(steps, {})
        kz = [n for n in needs if n.kind == "know_zone"]
        assert len(kz) == 1
        assert kz[0].payload["zone"] == "farmland"

    def test_explore_no_genera_need_si_zona_conocida(self) -> None:
        steps = [{"type": "action", "name": "ExploreArea", "args": ["farmland"]}]
        beliefs = {"zone_center": [["farmland", 10, 20]]}
        needs = self._needs(steps, beliefs)
        kz = [n for n in needs if n.kind == "know_zone"]
        assert len(kz) == 0

    def test_explore_args_numericos_no_genera_need(self) -> None:
        """ExploreArea con coords numericas (no zone tag) no genera know_zone."""
        steps = [{"type": "action", "name": "ExploreArea", "args": [10, 20]}]
        needs = self._needs(steps, {})
        kz = [n for n in needs if n.kind == "know_zone"]
        assert len(kz) == 0

    # --- Drop (nuevo en Iter 6) ---

    def test_drop_sin_item_genera_inventory_at_least(self) -> None:
        """Drop(wheat) sin has_item en beliefs genera inventory_at_least."""
        steps = [{"type": "action", "name": "Drop", "args": ["wheat"]}]
        needs = self._needs(steps, {})
        inv = [n for n in needs if n.kind == "inventory_at_least"]
        assert len(inv) == 1
        assert inv[0].payload["item"] == "wheat"
        assert inv[0].payload["qty"] == 1

    def test_drop_con_item_suficiente_no_genera_need(self) -> None:
        """Drop(wheat) con has_item(wheat, 3) en beliefs no genera need."""
        steps = [{"type": "action", "name": "Drop", "args": ["wheat"]}]
        beliefs = {"has_item": [["wheat", 3]]}
        needs = self._needs(steps, beliefs)
        inv = [n for n in needs if n.kind == "inventory_at_least"]
        assert len(inv) == 0

    def test_drop_con_item_insuficiente_genera_need(self) -> None:
        """Drop(wheat) con has_item(wheat, 0) en beliefs genera need."""
        steps = [{"type": "action", "name": "Drop", "args": ["wheat"]}]
        beliefs = {"has_item": [["wheat", 0]]}
        needs = self._needs(steps, beliefs)
        inv = [n for n in needs if n.kind == "inventory_at_least"]
        assert len(inv) == 1

    def test_drop_consumer_step_index_correcto(self) -> None:
        steps = [
            {"type": "action", "name": "MoveTo", "args": [5, 5]},
            {"type": "action", "name": "Drop", "args": ["wheat"]},
        ]
        needs = self._needs(steps, {})
        inv = [n for n in needs if n.kind == "inventory_at_least"]
        assert len(inv) == 1
        assert inv[0].consumer_step_index == 1

    # --- Legacy MoveTo(zone_string) ---

    def test_moveto_zone_string_genera_know_zone(self) -> None:
        steps = [{"type": "action", "name": "MoveTo", "args": ["bakery"]}]
        needs = self._needs(steps, {})
        kz = [n for n in needs if n.kind == "know_zone"]
        assert len(kz) == 1 and kz[0].payload["zone"] == "bakery"

    def test_moveto_numericos_no_genera_needs(self) -> None:
        steps = [{"type": "action", "name": "MoveTo", "args": [10, 20]}]
        needs = self._needs(steps, {})
        assert len(needs) == 0

    # --- Casos borde ---

    def test_subgoal_step_ignorado(self) -> None:
        steps = [{"type": "subgoal", "name": "Craft", "args": ["wheat"]}]
        needs = self._needs(steps, {})
        assert len(needs) == 0

    def test_accion_sin_contrato_ignorada(self) -> None:
        steps = [{"type": "action", "name": "UnknownAction", "args": ["x"]}]
        needs = self._needs(steps, {})
        assert len(needs) == 0

    def test_lista_vacia_de_steps(self) -> None:
        needs = self._needs([], {})
        assert len(needs) == 0


@pytest.mark.asyncio
async def test_step5_drop_sin_item_resuelto_por_move_to_and_pickup() -> None:
    """Drop(wheat) sin inventario: move_to_and_pickup lo resuelve si esta disponible."""
    from llm.pipeline.step5_refine import run_step5_refine

    steps = [{"type": "action", "name": "Drop", "args": ["wheat"]}]
    result = await run_step5_refine(
        "achieve_drop_wheat",
        steps,
        existing_subgoals=["move_to_and_pickup"],
        beliefs={},
    )
    subgoals = [s for s in result.steps if s.get("type") == "subgoal"]
    assert len(subgoals) == 1
    assert subgoals[0]["name"] == "move_to_and_pickup"
    assert subgoals[0]["args"][0] == "wheat"
    # Al menos una need resuelta de tipo inventory_at_least (puede haber mas por reintentos).
    assert any(n.kind == "inventory_at_least" for n in result.resolved_needs)
    assert result.unresolved_needs == []


@pytest.mark.asyncio
async def test_step5_drop_con_item_no_genera_subgoal() -> None:
    """Drop(wheat) con inventario suficiente: no se inserta ningun subgoal."""
    from llm.pipeline.step5_refine import run_step5_refine

    steps = [{"type": "action", "name": "Drop", "args": ["wheat"]}]
    result = await run_step5_refine(
        "achieve_drop_wheat",
        steps,
        existing_subgoals=["move_to_and_pickup"],
        beliefs={"has_item": [["wheat", 2]]},
    )
    assert len(result.steps) == 1
    assert result.steps[0]["name"] == "Drop"
    assert result.resolved_needs == []
    assert result.unresolved_needs == []


@pytest.mark.asyncio
async def test_step5_drop_sin_capability_genera_unresolved() -> None:
    """Drop(wheat) sin inventario y sin capability: queda como unresolved."""
    from llm.pipeline.step5_refine import run_step5_refine

    steps = [{"type": "action", "name": "Drop", "args": ["wheat"]}]
    result = await run_step5_refine(
        "achieve_drop_wheat",
        steps,
        existing_subgoals=[],
        beliefs={},
    )
    assert len(result.unresolved_needs) == 1
    assert result.unresolved_needs[0].kind == "inventory_at_least"


# ---------------------------------------------------------------------------
# TestIter7ExhaustionIntegration — C2 fix: CONTRACT_REGISTRY llega al helper
# ---------------------------------------------------------------------------


class TestIter7ExhaustionIntegration:
    """Verifica que _emit_exhaustion_branches recibe CONTRACT_REGISTRY (no
    capability_contracts) y por tanto emite variantes de agotamiento reales."""

    def _make_dag(self):
        import networkx as nx
        return nx.DiGraph()

    def test_explorearea_genera_variante_exhaustion(self) -> None:
        """Un step con ExploreArea debe producir al menos 1 variante exhaustion."""
        from llm.pipeline.pipeline_runner import _emit_exhaustion_branches
        import networkx as nx
        dag = nx.DiGraph()
        dag.add_node("test_goal", status="main")
        steps = [{"name": "ExploreArea", "args": ["farmland"]}]
        result = _emit_exhaustion_branches(
            "test_goal",
            "not knows_wheat_zone & not zone_center(farmland, ZX, ZY)",
            steps,
            dag,
            None,   # sin override; el helper usa _ACTION_CONTRACT_REGISTRY internamente
        )
        # Sin override (None) → vacío (backward compat)
        assert result == []

    def test_emit_con_contract_registry_explorearea(self) -> None:
        """Con CONTRACT_REGISTRY explícito, ExploreArea produce 1 variante exhaustion."""
        from llm.pipeline.pipeline_runner import _emit_exhaustion_branches
        from protocol.action_semantics import CONTRACT_REGISTRY
        import networkx as nx
        dag = nx.DiGraph()
        dag.add_node("test_goal", status="main")
        steps = [{"name": "ExploreArea", "args": ["farmland"]}]
        result = _emit_exhaustion_branches(
            "test_goal",
            "not has_item(wheat, N) & not zone_center(farmland, ZX, ZY)",
            steps,
            dag,
            CONTRACT_REGISTRY,
        )
        assert len(result) == 1
        v = result[0]
        assert "exhausted(ExploreArea, 2)" in v.guard
        assert "replan_goal" in v.asl or "print" in v.asl  # replan_goal → stub .print

    def test_emit_con_contract_registry_search(self) -> None:
        """Con CONTRACT_REGISTRY explícito, Search produce 1 variante exhaustion."""
        from llm.pipeline.pipeline_runner import _emit_exhaustion_branches
        from protocol.action_semantics import CONTRACT_REGISTRY
        import networkx as nx
        dag = nx.DiGraph()
        dag.add_node("test_goal2", status="main")
        steps = [{"name": "Search", "args": ["wheat"]}]
        result = _emit_exhaustion_branches(
            "test_goal2",
            "not has_item(wheat, N) & not item_at(wheat, WX, WY)",
            steps,
            dag,
            CONTRACT_REGISTRY,
        )
        assert len(result) == 1
        assert "exhausted(Search, 3)" in result[0].guard

    def test_emit_moveto_no_exhaustion(self) -> None:
        """MoveTo no tiene may_observe → sin variantes de agotamiento."""
        from llm.pipeline.pipeline_runner import _emit_exhaustion_branches
        from protocol.action_semantics import CONTRACT_REGISTRY
        import networkx as nx
        dag = nx.DiGraph()
        steps = [{"name": "MoveTo", "args": ["5", "5"]}]
        result = _emit_exhaustion_branches("g", "not done", steps, dag, CONTRACT_REGISTRY)
        assert result == []

    def test_emit_guard_contiene_neg_original(self) -> None:
        """El guard de la variante exhaustion contiene el guard normal completo."""
        from llm.pipeline.pipeline_runner import _emit_exhaustion_branches
        from protocol.action_semantics import CONTRACT_REGISTRY
        import networkx as nx
        dag = nx.DiGraph()
        neg_guard = "not has_item(wheat, N) & item_spawn(wheat, Z) & not zone_center(Z, ZX, ZY)"
        steps = [{"name": "ExploreArea", "args": ["Z"]}]
        result = _emit_exhaustion_branches("goal_wheat", neg_guard, steps, dag, CONTRACT_REGISTRY)
        assert len(result) == 1
        # El guard exhaustion empieza por el neg_guard original
        assert result[0].guard.startswith(neg_guard)
        assert result[0].guard.endswith("exhausted(ExploreArea, 2)")
