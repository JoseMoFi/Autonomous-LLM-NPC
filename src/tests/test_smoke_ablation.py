"""Smoke tests — tools/ablation_monolithic.py (validador puro, sin LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import ablation_monolithic as AB  # noqa: E402

_KNOWN = {"move_to_and_pickup", "craft_item"}


def test_valid_plan_passes():
    plan = {
        "sig": "achieve_have_wheat",
        "success_condition": "has_item(wheat, 1)",
        "steps": [
            {"type": "action", "name": "MoveTo", "args": [9, -6]},
            {"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 1]},
        ],
    }
    ok, errors = AB.validate_monolithic(plan, _KNOWN)
    assert ok is True, errors


def test_empty_steps_fails():
    ok, errors = AB.validate_monolithic({"steps": []}, _KNOWN)
    assert ok is False
    assert any("vac" in e for e in errors)


def test_unknown_action_fails():
    plan = {"steps": [{"type": "action", "name": "Teleport", "args": [1, 2]}]}
    ok, errors = AB.validate_monolithic(plan, _KNOWN)
    assert ok is False
    assert any("desconocida" in e for e in errors)


def test_unknown_subgoal_fails():
    plan = {"steps": [{"type": "subgoal", "name": "achieve_magic", "args": []}]}
    ok, errors = AB.validate_monolithic(plan, _KNOWN)
    assert ok is False
    assert any("sub-goal desconocido" in e for e in errors)


def test_non_dict_fails():
    ok, errors = AB.validate_monolithic(["not", "a", "dict"], _KNOWN)
    assert ok is False


def test_build_prompt_includes_catalog():
    user, system = AB.build_monolithic_prompt("Get wheat", {"zone_ids": ["farmland"], "item_ids": ["wheat"]})
    assert "Get wheat" in user
    assert "farmland" in user
    assert "JSON" in system or "json" in system.lower()
