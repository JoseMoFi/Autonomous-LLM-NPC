"""Smoke tests — step3_validator y step3_repair.

Cubre (sin LLM real):
  - build_fact_var_map: extrae variables de facts con origen correcto
  - _is_variable: convención Prolog/ASL (mayúscula = variable)
  - check_arg_count (H1): correcto, pocos args, demasiados, acción desconocida, subgoal
  - check_unbound_vars (H2): var ligada OK, var no ligada → error con sugerencia
  - check_variable_reuse (W1): reservado, no produce warnings por defecto
  - validate_steps: integración completa
  - repair_failing_steps: mock LLM aplica fix, revierte si empeora, ignora índices inválidos
"""

from __future__ import annotations

import pytest

from llm.pipeline.step3_validator import (
    ACTION_SIGNATURES,
    StepError,
    _is_variable,
    build_fact_var_map,
    check_arg_count,
    check_unbound_vars,
    check_variable_reuse,
    gather_variant_underconstrained,
    validate_steps,
)
from llm.pipeline.step3_repair import repair_failing_steps


# ===========================================================================
# W2 (T5) — variante de recolección sin acotar por estado del mundo
# ===========================================================================

def test_gather_sin_guard_es_permisiva():
    steps = [{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 2]}]
    assert gather_variant_underconstrained("true", steps) is True
    assert gather_variant_underconstrained("", steps) is True


def test_gather_acotada_por_estado_no_avisa():
    steps = [{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 2]}]
    # step1b acota por ausencia del item (has_item) → no se avisa
    assert gather_variant_underconstrained("not has_item(wheat, 2)", steps) is False
    assert gather_variant_underconstrained("item_spawn(wheat, _)", steps) is False
    assert gather_variant_underconstrained("recipe_output(r, _, wheat, _)", steps) is False


def test_sin_paso_de_recoleccion_nunca_avisa():
    steps = [{"type": "subgoal", "name": "craft_item", "args": ["bread_recipe", 1]}]
    assert gather_variant_underconstrained("true", steps) is False
    # acción cruda no-gather tampoco
    assert gather_variant_underconstrained("true", [{"type": "action", "name": "Drop", "args": []}]) is False


def test_accion_cruda_de_gather_tambien_avisa():
    assert gather_variant_underconstrained(
        "true", [{"type": "action", "name": "PickUp", "args": ["wheat"]}]) is True


# ===========================================================================
# Fixtures comunes
# ===========================================================================

FACTS = [
    {"functor": "has_item", "args": ["wheat", "N"], "reason": "need item"},
    {"functor": "zone_center", "args": ["bakery", "X", "Y"], "reason": "need location"},
]

GUARDS = [
    {"expr": "N >= 2", "reason": "enough wheat"},
]

BOUND_VARS: set[str] = {"N", "X", "Y"}


def _make_llm_returning(response: str):
    """Mock LLM async que devuelve siempre la misma cadena."""
    async def _llm(user_prompt: str, system_prompt: str) -> str:
        return response
    return _llm


# ===========================================================================
# build_fact_var_map
# ===========================================================================

class TestBuildFactVarMap:
    def test_extracts_variables_from_facts(self) -> None:
        var_map = build_fact_var_map(FACTS)
        assert "N" in var_map
        assert "X" in var_map
        assert "Y" in var_map

    def test_origin_message_contains_functor(self) -> None:
        var_map = build_fact_var_map(FACTS)
        assert "has_item" in var_map["N"]
        assert "zone_center" in var_map["X"]
        assert "zone_center" in var_map["Y"]

    def test_origin_message_contains_arg_index(self) -> None:
        var_map = build_fact_var_map(FACTS)
        # N es el arg[1] de has_item
        assert "arg[1]" in var_map["N"]
        # X es el arg[1] de zone_center, Y es el arg[2]
        assert "arg[1]" in var_map["X"]
        assert "arg[2]" in var_map["Y"]

    def test_constants_not_included(self) -> None:
        var_map = build_fact_var_map(FACTS)
        # 'wheat' y 'bakery' son constantes (minúscula)
        assert "wheat" not in var_map
        assert "bakery" not in var_map

    def test_empty_facts_returns_empty(self) -> None:
        assert build_fact_var_map([]) == {}

    def test_no_duplicate_overwrite(self) -> None:
        """Primera ocurrencia de una variable gana en el mapa."""
        facts = [
            {"functor": "fact_a", "args": ["X"], "reason": ""},
            {"functor": "fact_b", "args": ["X"], "reason": ""},
        ]
        var_map = build_fact_var_map(facts)
        assert var_map["X"] == "fact_a arg[0]"


# ===========================================================================
# _is_variable
# ===========================================================================

class TestIsVariable:
    def test_uppercase_start_is_variable(self) -> None:
        assert _is_variable("X") is True
        assert _is_variable("N") is True
        assert _is_variable("MyVar") is True
        # Single-word mixed-case identifiers are ASL variables.
        assert _is_variable("ZoneTag") is True

    def test_pascal_case_with_underscore_lowercase_is_constant(self) -> None:
        # PascalCase con `_minúscula` (recipe IDs / delivery point IDs) se tratan
        # como constantes, NO como variables ASL — ver docstring de _is_variable.
        assert _is_variable("Bread_recipe") is False
        assert _is_variable("Bakeri_point") is False
        assert _is_variable("Zone_tag") is False

    def test_lowercase_is_not_variable(self) -> None:
        assert _is_variable("wheat") is False
        assert _is_variable("bakery") is False
        assert _is_variable("itemId") is False

    def test_empty_string_is_not_variable(self) -> None:
        assert _is_variable("") is False

    def test_digit_start_is_not_variable(self) -> None:
        assert _is_variable("1X") is False

    def test_underscore_start_is_not_variable(self) -> None:
        assert _is_variable("_X") is False

    def test_numeric_suffix_is_variable(self) -> None:
        assert _is_variable("X1") is True
        assert _is_variable("Var2") is True


# ===========================================================================
# check_arg_count (H1)
# ===========================================================================

class TestCheckArgCount:
    def _make_action(self, name: str, args: list) -> dict:
        return {"type": "action", "name": name, "args": args}

    def test_correct_arg_count_no_error(self) -> None:
        step = self._make_action("MoveTo", ["X", "Y"])
        assert check_arg_count(step) == []

    def test_too_few_args_returns_error(self) -> None:
        step = self._make_action("MoveTo", ["X"])  # needs 2
        errors = check_arg_count(step)
        assert len(errors) == 1
        assert "MoveTo" in errors[0]
        assert "2" in errors[0]

    def test_too_many_args_returns_error(self) -> None:
        step = self._make_action("PickUp", ["wheat", "extra"])  # needs exactly 1
        errors = check_arg_count(step)
        assert len(errors) == 1

    def test_optional_arg_at_min_ok(self) -> None:
        # Craft requires 2-3 args; 2 is OK
        step = self._make_action("Craft", ["wheat", "recipe"])
        assert check_arg_count(step) == []

    def test_optional_arg_at_max_ok(self) -> None:
        # Craft can have 3 args
        step = self._make_action("Craft", ["wheat", "recipe", "N"])
        assert check_arg_count(step) == []

    def test_unknown_action_skipped(self) -> None:
        step = self._make_action("UnknownAction", ["a", "b", "c"])
        assert check_arg_count(step) == []

    def test_subgoal_type_skipped(self) -> None:
        step = {"type": "subgoal", "name": "achieve_gather_wheat", "args": []}
        assert check_arg_count(step) == []

    def test_case_insensitive_lookup(self) -> None:
        # MoveTo in mixed case vs MOVETO in ACTION_SIGNATURES
        step_lower = self._make_action("moveto", ["X", "Y"])
        step_upper = self._make_action("MOVETO", ["X", "Y"])
        assert check_arg_count(step_lower) == []
        assert check_arg_count(step_upper) == []

    def test_all_known_actions_in_signatures(self) -> None:
        """Verifica que las signaturas cubren las acciones primitivas esperadas."""
        for action in ["MOVETO", "EXPLOREAREA", "PICKUP", "CRAFT", "DROP", "WAIT"]:
            assert action in ACTION_SIGNATURES


# ===========================================================================
# check_unbound_vars (H2)
# ===========================================================================

class TestCheckUnboundVars:
    def _make_step(self, args: list, action_type: str = "action", name: str = "MoveTo") -> dict:
        return {"type": action_type, "name": name, "args": args}

    def _fact_var_map(self) -> dict[str, str]:
        return build_fact_var_map(FACTS)

    def test_bound_var_is_ok(self) -> None:
        step = self._make_step(["X", "Y"])
        errors = check_unbound_vars(step, BOUND_VARS, self._fact_var_map())
        assert errors == []

    def test_constant_arg_ok(self) -> None:
        step = self._make_step(["wheat"])
        errors = check_unbound_vars(step, BOUND_VARS, self._fact_var_map())
        assert errors == []

    def test_unbound_uppercase_var_error(self) -> None:
        step = self._make_step(["A", "B"])  # A, B not in BOUND_VARS
        errors = check_unbound_vars(step, BOUND_VARS, self._fact_var_map())
        assert len(errors) == 2
        assert "A" in errors[0]
        assert "B" in errors[1]

    def test_error_includes_available_vars(self) -> None:
        step = self._make_step(["Z"])
        errors = check_unbound_vars(step, BOUND_VARS, self._fact_var_map())
        assert len(errors) == 1
        # El mensaje debe citar algunas variables disponibles
        msg = errors[0]
        assert "N" in msg or "X" in msg or "Y" in msg

    def test_mixed_bound_and_unbound(self) -> None:
        step = self._make_step(["X", "Z"])  # X bound, Z unbound
        errors = check_unbound_vars(step, BOUND_VARS, self._fact_var_map())
        assert len(errors) == 1
        assert "Z" in errors[0]

    def test_empty_bound_vars_flags_all_uppercase(self) -> None:
        step = self._make_step(["X", "Y"])
        errors = check_unbound_vars(step, set(), {})
        assert len(errors) == 2

    def test_non_string_arg_skipped(self) -> None:
        step = {"type": "action", "name": "Wait", "args": [5]}
        errors = check_unbound_vars(step, BOUND_VARS, {})
        assert errors == []


# ===========================================================================
# check_variable_reuse (W1) — actualmente reservado
# ===========================================================================

class TestCheckVariableReuse:
    def test_consistent_step_no_warnings(self) -> None:
        step = {"type": "action", "name": "MoveTo", "args": ["X", "Y"]}
        warnings = check_variable_reuse(step, BOUND_VARS, build_fact_var_map(FACTS))
        assert warnings == []

    def test_constant_only_step_no_warnings(self) -> None:
        step = {"type": "action", "name": "PickUp", "args": ["wheat"]}
        warnings = check_variable_reuse(step, BOUND_VARS, build_fact_var_map(FACTS))
        assert warnings == []


# ===========================================================================
# validate_steps — integración
# ===========================================================================

class TestValidateSteps:
    def test_clean_steps_returns_empty_list(self) -> None:
        steps = [
            {"type": "action", "name": "MoveTo", "args": ["X", "Y"]},
            {"type": "action", "name": "PickUp", "args": ["wheat"]},
        ]
        errors = validate_steps(steps, FACTS, GUARDS, BOUND_VARS)
        assert errors == []

    def test_step_with_arg_count_error_detected(self) -> None:
        steps = [
            {"type": "action", "name": "MoveTo", "args": ["X"]},  # falta un arg
        ]
        errors = validate_steps(steps, FACTS, GUARDS, BOUND_VARS)
        assert len(errors) == 1
        assert errors[0].step_index == 0
        assert errors[0].has_errors is True

    def test_step_with_unbound_var_detected(self) -> None:
        steps = [
            {"type": "action", "name": "MoveTo", "args": ["A", "B"]},  # A, B no ligados
        ]
        errors = validate_steps(steps, FACTS, GUARDS, BOUND_VARS)
        assert len(errors) == 1
        assert len(errors[0].errors) == 2  # A y B

    def test_mixed_clean_and_failing_steps(self) -> None:
        steps = [
            {"type": "action", "name": "MoveTo", "args": ["X", "Y"]},   # OK
            {"type": "action", "name": "PickUp", "args": ["A"]},          # A unbound
            {"type": "action", "name": "Craft", "args": ["wheat", "recipe"]},  # OK
        ]
        errors = validate_steps(steps, FACTS, GUARDS, BOUND_VARS)
        assert len(errors) == 1
        assert errors[0].step_index == 1

    def test_step_error_contains_step_reference(self) -> None:
        steps = [
            {"type": "action", "name": "Drop", "args": []},  # arg count error
        ]
        errors = validate_steps(steps, FACTS, GUARDS, BOUND_VARS)
        assert errors[0].step == steps[0]

    def test_subgoal_step_accepted(self) -> None:
        steps = [
            {"type": "subgoal", "name": "achieve_gather_wheat", "args": []},
        ]
        errors = validate_steps(steps, FACTS, GUARDS, BOUND_VARS)
        assert errors == []

    def test_step_error_has_errors_property(self) -> None:
        se = StepError(step_index=0, step={}, errors=["some error"], warnings=[])
        assert se.has_errors is True

    def test_step_error_without_errors(self) -> None:
        se = StepError(step_index=0, step={}, errors=[], warnings=["soft warning"])
        assert se.has_errors is False
        assert se.has_warnings is True


# ===========================================================================
# repair_failing_steps — async, mock LLM
# ===========================================================================

class TestRepairFailingSteps:

    STEPS_WITH_ERRORS = [
        {"type": "action", "name": "MoveTo", "args": ["A", "B"]},   # A, B unbound
        {"type": "action", "name": "PickUp", "args": ["wheat"]},    # OK
    ]

    @pytest.mark.asyncio
    async def test_valid_fix_applied(self) -> None:
        step_errors = validate_steps(self.STEPS_WITH_ERRORS, FACTS, GUARDS, BOUND_VARS)
        hard_errors = [e for e in step_errors if e.has_errors]

        fix_response = '{"fixes": [{"index": 0, "step": {"type": "action", "name": "MoveTo", "args": ["X", "Y"]}}]}'
        llm = _make_llm_returning(fix_response)

        result = await repair_failing_steps(
            goal_name="achieve_craft_bread",
            steps=self.STEPS_WITH_ERRORS,
            step_errors=hard_errors,
            facts=FACTS,
            guards=GUARDS,
            bound_vars=BOUND_VARS,
            existing_subgoals=[],
            llm_call=llm,
        )

        assert result[0]["args"] == ["X", "Y"]
        assert result[1] == self.STEPS_WITH_ERRORS[1]  # original intacto

    @pytest.mark.asyncio
    async def test_invalid_index_ignored(self) -> None:
        step_errors = validate_steps(self.STEPS_WITH_ERRORS, FACTS, GUARDS, BOUND_VARS)
        hard_errors = [e for e in step_errors if e.has_errors]

        # Índice 99 no está en los failing steps
        fix_response = '{"fixes": [{"index": 99, "step": {"type": "action", "name": "MoveTo", "args": ["X", "Y"]}}]}'
        llm = _make_llm_returning(fix_response)

        result = await repair_failing_steps(
            goal_name="achieve_craft_bread",
            steps=self.STEPS_WITH_ERRORS,
            step_errors=hard_errors,
            facts=FACTS,
            guards=GUARDS,
            bound_vars=BOUND_VARS,
            existing_subgoals=[],
            llm_call=llm,
        )

        # No se aplica el fix → devuelve originales
        assert result == self.STEPS_WITH_ERRORS

    @pytest.mark.asyncio
    async def test_repair_worsens_result_reverts_to_original(self) -> None:
        step_errors = validate_steps(self.STEPS_WITH_ERRORS, FACTS, GUARDS, BOUND_VARS)
        hard_errors = [e for e in step_errors if e.has_errors]

        # El fix introduce OTRO error (C, D tampoco son bound_vars) → más errores que antes
        bad_fix = '{"fixes": [{"index": 0, "step": {"type": "action", "name": "MoveTo", "args": ["C", "D"]}}]}'
        llm = _make_llm_returning(bad_fix)

        result = await repair_failing_steps(
            goal_name="achieve_craft_bread",
            steps=self.STEPS_WITH_ERRORS,
            step_errors=hard_errors,
            facts=FACTS,
            guards=GUARDS,
            bound_vars=BOUND_VARS,
            existing_subgoals=[],
            llm_call=llm,
        )

        # Revertido a originales porque la reparación no mejoró nada
        assert result == self.STEPS_WITH_ERRORS

    @pytest.mark.asyncio
    async def test_unparseable_llm_response_returns_original(self) -> None:
        step_errors = validate_steps(self.STEPS_WITH_ERRORS, FACTS, GUARDS, BOUND_VARS)
        hard_errors = [e for e in step_errors if e.has_errors]

        llm = _make_llm_returning("Lo siento, no puedo procesarlo.")

        result = await repair_failing_steps(
            goal_name="achieve_craft_bread",
            steps=self.STEPS_WITH_ERRORS,
            step_errors=hard_errors,
            facts=FACTS,
            guards=GUARDS,
            bound_vars=BOUND_VARS,
            existing_subgoals=[],
            llm_call=llm,
        )

        assert result == self.STEPS_WITH_ERRORS

    @pytest.mark.asyncio
    async def test_no_hard_errors_no_repair_call(self) -> None:
        clean_steps = [
            {"type": "action", "name": "MoveTo", "args": ["X", "Y"]},
        ]
        # No hay hard errors → step_errors=[] → reparar no hace nada
        llm = _make_llm_returning('{"fixes": []}')
        result = await repair_failing_steps(
            goal_name="achieve_craft_bread",
            steps=clean_steps,
            step_errors=[],
            facts=FACTS,
            guards=GUARDS,
            bound_vars=BOUND_VARS,
            existing_subgoals=[],
            llm_call=llm,
        )
        assert result == clean_steps

    @pytest.mark.asyncio
    async def test_reasoning_context_passed_through(self) -> None:
        """repair acepta reasoning sin fallar."""
        step_errors = validate_steps(self.STEPS_WITH_ERRORS, FACTS, GUARDS, BOUND_VARS)
        hard_errors = [e for e in step_errors if e.has_errors]

        fix_response = '{"fixes": [{"index": 0, "step": {"type": "action", "name": "MoveTo", "args": ["X", "Y"]}}]}'
        llm = _make_llm_returning(fix_response)
        reasoning = {"preconditions": ["need wheat"], "action_plan": ["go to bakery"]}

        result = await repair_failing_steps(
            goal_name="achieve_craft_bread",
            steps=self.STEPS_WITH_ERRORS,
            step_errors=hard_errors,
            facts=FACTS,
            guards=GUARDS,
            bound_vars=BOUND_VARS,
            existing_subgoals=["achieve_gather_wheat"],
            llm_call=llm,
            reasoning=reasoning,
        )

        assert result[0]["args"] == ["X", "Y"]
