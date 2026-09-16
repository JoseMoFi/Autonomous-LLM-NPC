from __future__ import annotations

"""Tests de humo para plan_simulator.py.

Cubre los escenarios de la auditoría:
  - [MoveTo, Search] con has_item → reachable=False
  - [MoveTo, Search, MoveTo, PickUp] → reachable=True
  - [Craft] con inventario suficiente → reachable=True
  - Plan vacío → reachable=False (sin success_condition → True)
  - Contrato faltante → se ignora el step, no explota
"""

import pytest

from llm.pipeline.plan_simulator import (
    Beliefs,
    PlanReachability,
    check_plan_reachability,
    simulate_plan,
    verify_success_condition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_beliefs(**kwargs) -> Beliefs:
    """Crea un snapshot de creencias desde kwargs: functor=[tuple, ...]."""
    return {k: set(v) for k, v in kwargs.items()}


# ---------------------------------------------------------------------------
# simulate_plan
# ---------------------------------------------------------------------------

class TestSimulatePlan:
    def test_empty_plan_returns_initial_beliefs(self):
        initial = make_beliefs(current_position={("9", "-6")})
        result = simulate_plan([], initial)
        assert result == initial
        # deep copy: no es el mismo objeto
        assert result is not initial

    def test_moveto_sets_current_position(self):
        initial = make_beliefs(current_position={("0", "0")})
        steps = [{"type": "action", "name": "MoveTo", "args": {"x": "9", "y": "-6"}}]
        result = simulate_plan(steps, initial)
        assert ("9", "-6") in result.get("current_position", set())

    def test_moveto_invalidates_old_position(self):
        initial = make_beliefs(current_position={("0", "0")})
        steps = [{"type": "action", "name": "MoveTo", "args": {"x": "9", "y": "-6"}}]
        result = simulate_plan(steps, initial)
        assert ("0", "0") not in result.get("current_position", set())

    def test_search_does_not_add_has_item(self):
        """Search es observacional: no garantiza has_item."""
        initial = make_beliefs(current_position={("9", "-6")})
        steps = [{"type": "action", "name": "Search", "args": {"itemId": "wheat"}}]
        result = simulate_plan(steps, initial)
        assert "has_item" not in result

    def test_pickup_adds_has_item(self):
        initial = make_beliefs(
            current_position={("9", "-6")},
            item_at={("wheat", "9", "-6")},
        )
        steps = [{"type": "action", "name": "PickUp", "args": {"itemId": "wheat"}}]
        result = simulate_plan(steps, initial)
        # PickUp guarantees has_item(ItemId, N)
        assert "has_item" in result
        # item_at debe estar invalidado
        assert ("wheat", "9", "-6") not in result.get("item_at", set())

    def test_positional_args_list(self):
        """Args como lista posicional (formato ASL) funcionan igual."""
        initial = make_beliefs(current_position={("0", "0")})
        steps = [{"type": "action", "name": "MoveTo", "args": ["5", "3"]}]
        result = simulate_plan(steps, initial)
        assert ("5", "3") in result.get("current_position", set())

    def test_subgoal_steps_are_ignored(self):
        initial = make_beliefs(current_position={("0", "0")})
        steps = [{"type": "subgoal", "name": "achieve_something", "args": []}]
        result = simulate_plan(steps, initial)
        assert result == initial

    def test_unknown_action_is_skipped(self):
        initial = make_beliefs(current_position={("0", "0")})
        steps = [{"type": "action", "name": "DoMagic", "args": {}}]
        result = simulate_plan(steps, initial)  # no debe lanzar
        assert result == initial

    def test_chain_moveto_search_no_has_item(self):
        """MoveTo + Search: no produce has_item (Search es observacional)."""
        initial = make_beliefs(current_position={("0", "0")})
        steps = [
            {"type": "action", "name": "MoveTo",  "args": {"x": "9", "y": "-6"}},
            {"type": "action", "name": "Search",  "args": {"itemId": "wheat"}},
        ]
        result = simulate_plan(steps, initial)
        assert "has_item" not in result
        assert ("9", "-6") in result.get("current_position", set())

    def test_chain_moveto_search_moveto_pickup(self):
        """MoveTo + Search + MoveTo + PickUp: produce has_item."""
        initial = make_beliefs(
            current_position={("0", "0")},
            item_at={("wheat", "9", "-6")},
        )
        steps = [
            {"type": "action", "name": "MoveTo",  "args": {"x": "9", "y": "-6"}},
            {"type": "action", "name": "Search",  "args": {"itemId": "wheat"}},
            {"type": "action", "name": "MoveTo",  "args": {"x": "9", "y": "-6"}},
            {"type": "action", "name": "PickUp",  "args": {"itemId": "wheat"}},
        ]
        result = simulate_plan(steps, initial)
        assert "has_item" in result


# ---------------------------------------------------------------------------
# verify_success_condition
# ---------------------------------------------------------------------------

class TestVerifySuccessCondition:
    def test_none_condition_is_always_true(self):
        assert verify_success_condition({}, None) is True

    def test_empty_condition_is_always_true(self):
        assert verify_success_condition({}, "") is True

    def test_simple_ground_predicate_true(self):
        beliefs = make_beliefs(has_item={("wheat", "1")})
        assert verify_success_condition(beliefs, "has_item(wheat, 1)") is True

    def test_simple_ground_predicate_false(self):
        beliefs = make_beliefs(has_item={("bread", "1")})
        assert verify_success_condition(beliefs, "has_item(wheat, 1)") is False

    def test_missing_functor_is_false(self):
        assert verify_success_condition({}, "has_item(wheat, 1)") is False

    def test_variable_arg_matches_any(self):
        """Variable uppercase actúa como existencial: cualquier valor satisface."""
        beliefs = make_beliefs(has_item={("wheat", "3")})
        assert verify_success_condition(beliefs, "has_item(wheat, N)") is True

    def test_variable_arg_no_match_if_functor_absent(self):
        assert verify_success_condition({}, "has_item(wheat, N)") is False

    def test_conjunction_both_true(self):
        beliefs = make_beliefs(
            has_item={("wheat", "1")},
            current_position={("5", "5")},
        )
        assert verify_success_condition(
            beliefs, "has_item(wheat, 1) & current_position(5, 5)"
        ) is True

    def test_conjunction_one_false(self):
        beliefs = make_beliefs(has_item={("wheat", "1")})
        assert verify_success_condition(
            beliefs, "has_item(wheat, 1) & current_position(5, 5)"
        ) is False

    def test_position_condition(self):
        beliefs = make_beliefs(current_position={("9", "-6")})
        assert verify_success_condition(beliefs, "current_position(9, -6)") is True


# ---------------------------------------------------------------------------
# check_plan_reachability — escenarios principales del audit
# ---------------------------------------------------------------------------

class TestCheckPlanReachability:
    def test_moveto_search_not_reachable_for_has_item(self):
        """
        Escenario del bug: plan [MoveTo, Search] para goal has_item(wheat, 1).
        Debe devolver reachable=False con missing_predicates que mencione has_item.
        """
        initial = make_beliefs(current_position={("0", "0")})
        steps = [
            {"type": "action", "name": "MoveTo", "args": {"x": "9", "y": "-6"}},
            {"type": "action", "name": "Search", "args": {"itemId": "wheat"}},
        ]
        result = check_plan_reachability(steps, initial, "has_item(wheat, 1)")

        assert isinstance(result, PlanReachability)
        assert result.reachable is False
        assert any("has_item" in p for p in result.missing_predicates)
        assert "has_item" in result.hint

    def test_moveto_search_moveto_pickup_reachable(self):
        """Plan completo con PickUp → reachable=True."""
        initial = make_beliefs(
            current_position={("0", "0")},
            item_at={("wheat", "9", "-6")},
        )
        steps = [
            {"type": "action", "name": "MoveTo",  "args": {"x": "9", "y": "-6"}},
            {"type": "action", "name": "Search",  "args": {"itemId": "wheat"}},
            {"type": "action", "name": "MoveTo",  "args": {"x": "9", "y": "-6"}},
            {"type": "action", "name": "PickUp",  "args": {"itemId": "wheat"}},
        ]
        result = check_plan_reachability(steps, initial, "has_item(wheat, N)")
        assert result.reachable is True
        assert result.missing_predicates == []
        assert result.hint == ""

    def test_craft_reachable(self):
        """Craft garantiza has_item del output → reachable para has_item."""
        initial = make_beliefs(
            current_position={("5", "5")},
            has_item={("wheat", "3")},
        )
        steps = [
            {
                "type": "action",
                "name": "Craft",
                "args": {"itemId": "wheat", "targetId": "bread_recipe"},
            }
        ]
        # Craft garantiza has_item(ItemToCraft, N): el simulador añade has_item
        # con args resueltos desde el contrato (ItemToCraft no es resolvible
        # sin la receta, pero el functor sí aparece).
        result = check_plan_reachability(steps, initial, "has_item(W, N)")
        # Con variable W y N, cualquier tupla en has_item satisface
        assert result.reachable is True

    def test_empty_plan_not_reachable_for_has_item(self):
        """Plan vacío: no produce has_item → reachable=False."""
        result = check_plan_reachability([], {}, "has_item(wheat, 1)")
        assert result.reachable is False
        assert result.missing_predicates != []

    def test_no_condition_always_reachable(self):
        """Sin success_condition el plan siempre es válido."""
        result = check_plan_reachability([], {}, None)
        assert result.reachable is True
        assert result.hint == ""

    def test_hint_contains_missing_and_advice(self):
        """El hint menciona la condición faltante y una acción que la garantice."""
        steps = [
            {"type": "action", "name": "MoveTo", "args": {"x": "5", "y": "5"}},
        ]
        result = check_plan_reachability(steps, {}, "has_item(wheat, 1)")
        assert "has_item" in result.hint
        assert "PickUp" in result.hint or "Craft" in result.hint

    def test_moveto_reachable_for_position_condition(self):
        """MoveTo garantiza current_position → reachable=True para esa condición."""
        steps = [
            {"type": "action", "name": "MoveTo", "args": {"x": "9", "y": "-6"}},
        ]
        result = check_plan_reachability(steps, {}, "current_position(9, -6)")
        assert result.reachable is True
