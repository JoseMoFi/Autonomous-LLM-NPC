"""Smoke tests — triggers reactivos (0.B1).

Cubren:
  - TriggerRegistry.evaluate() evalúa el guard (no solo el sig).
  - BeliefStore._execute_trigger_body soporta .log y adopt_goal.
  - adopt_goal encola un goal una sola vez (idempotente).
  - Una acción no soportada genera warning, no excepción.
"""

from __future__ import annotations

import logging

import pytest

from npc.beliefs import BeliefStore
from npc.trigger_registry import TriggerRegistry, TriggerRule


def _registry(*rules: TriggerRule) -> TriggerRegistry:
    reg = TriggerRegistry()
    for r in rules:
        reg.add(r)
    return reg


def _rule(sig: str, guard: str, body: list[str], head: str) -> TriggerRule:
    return TriggerRule(sig=sig, guard=guard, body=body, full_asl=head, source="test")


# ===========================================================================
# evaluate(): evaluación de guard
# ===========================================================================

def test_evaluate_fires_when_guard_passes() -> None:
    bs = BeliefStore()
    reg = _registry(_rule(
        "+has_item", "N >= 2", [".log('ok')"], "+has_item(ItemId, N)",
    ))
    bs._facts["has_item"] = {("wheat", 3)}
    fired = reg.evaluate("has_item", bs)
    assert len(fired) == 1


def test_evaluate_does_not_fire_when_guard_fails() -> None:
    bs = BeliefStore()
    reg = _registry(_rule(
        "+has_item", "N >= 5", [".log('ok')"], "+has_item(ItemId, N)",
    ))
    bs._facts["has_item"] = {("wheat", 2)}
    fired = reg.evaluate("has_item", bs)
    assert fired == []


def test_evaluate_true_guard_always_fires() -> None:
    bs = BeliefStore()
    reg = _registry(_rule(
        "+zone_center", "true", [".log('z')"], "+zone_center(ZoneTag, _, _)",
    ))
    bs._facts["zone_center"] = {("bakeri", 5, 5)}
    fired = reg.evaluate("zone_center", bs)
    assert len(fired) == 1


def test_evaluate_ignores_non_matching_sig() -> None:
    bs = BeliefStore()
    reg = _registry(_rule(
        "+has_item", "true", [".log('x')"], "+has_item(ItemId, N)",
    ))
    assert reg.evaluate("zone_center", bs) == []


def test_evaluate_head_constant_matches_by_position() -> None:
    # Cabeza con constante ground: +has_item(wheat, N). N debe ligarse a la
    # POSICIÓN 1 (qty), y la constante wheat debe casar la posición 0.
    bs = BeliefStore()
    reg = _registry(_rule(
        "+has_item", "N >= 2", ["adopt_goal(achieve_bake_bread, 'has_item(bread, 1)')"],
        "+has_item(wheat, N)",
    ))
    # wheat con 2 → dispara
    bs._facts["has_item"] = {("wheat", 2)}
    assert len(reg.evaluate("has_item", bs)) == 1
    # wheat con 1 → no dispara (N=1 < 2)
    bs._facts["has_item"] = {("wheat", 1)}
    assert reg.evaluate("has_item", bs) == []
    # otro item (bread) con 2 → la constante wheat NO casa → no dispara
    bs._facts["has_item"] = {("bread", 2)}
    assert reg.evaluate("has_item", bs) == []


# ===========================================================================
# _execute_trigger_body
# ===========================================================================

def test_adopt_goal_enqueues_once() -> None:
    bs = BeliefStore()
    adopted: list[str] = []
    seen: set[str] = set()

    def adopter(sig: str, cond: str | None = None) -> None:
        if sig in seen:
            return
        seen.add(sig)
        adopted.append(sig)

    bs.set_goal_adopter(adopter)
    bs._execute_trigger_body(["adopt_goal(achieve_test)"])
    bs._execute_trigger_body(["adopt_goal(achieve_test)"])
    assert adopted == ["achieve_test"]


def test_adopt_goal_with_condition_passes_success_condition() -> None:
    # Fase 9: adopt_goal(sig, 'cond') → el goal reactivo lleva success_condition.
    bs = BeliefStore()
    got: list[tuple] = []
    bs.set_goal_adopter(lambda sig, cond=None: got.append((sig, cond)))
    bs._execute_trigger_body(["adopt_goal(achieve_has_item, 'has_item(bread, 1)')"])
    assert got == [("achieve_has_item", "has_item(bread, 1)")]


def test_log_body_does_not_crash(caplog: pytest.LogCaptureFixture) -> None:
    bs = BeliefStore()
    with caplog.at_level(logging.INFO):
        bs._execute_trigger_body([".log('hola mundo')"])
    assert any("hola mundo" in r.message for r in caplog.records)


def test_unsupported_action_warns(caplog: pytest.LogCaptureFixture) -> None:
    bs = BeliefStore()
    with caplog.at_level(logging.WARNING):
        bs._execute_trigger_body(["+some_belief(x)"])
    assert any("not supported" in r.message for r in caplog.records)


def test_adopt_goal_without_adopter_warns(caplog: pytest.LogCaptureFixture) -> None:
    bs = BeliefStore()
    with caplog.at_level(logging.WARNING):
        bs._execute_trigger_body(["adopt_goal(achieve_x)"])
    assert any("sin goal_adopter" in r.message for r in caplog.records)


# ===========================================================================
# Integración: belief change → trigger → adopt_goal
# ===========================================================================

def test_belief_change_fires_trigger_and_adopts_goal() -> None:
    bs = BeliefStore()
    reg = _registry(_rule(
        "+has_item", "N >= 1", ["adopt_goal(achieve_reactivo)"],
        "+has_item(ItemId, N)",
    ))
    bs.set_trigger_registry(reg)
    adopted: list[str] = []
    bs.set_goal_adopter(lambda sig, cond=None: adopted.append(sig))

    bs.apply_has_item("bread", 1)
    assert "achieve_reactivo" in adopted
