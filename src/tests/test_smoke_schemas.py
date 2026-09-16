"""Smoke tests — esquemas Pydantic del pipeline (Fase 3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm.schemas import (
    GoalItem,
    ParseGoalsResponse,
    Step0Response,
    Step1Response,
    Step2Response,
    Step3Response,
    StepItem,
    SuccessVariant,
)


def _goal(**over):
    base = {
        "sig": "achieve_have_wheat",
        "source_index": 0,
        "priority": 1.0,
        "reason": "need wheat",
        "success_condition": "has_item(wheat, 1)",
    }
    base.update(over)
    return base


def test_parse_goals_response_valid():
    r = ParseGoalsResponse(goals=[_goal(), _goal(sig="achieve_explore_farmland", source_index=1)])
    assert len(r.goals) == 2
    assert isinstance(r.goals[0], GoalItem)
    assert r.goals[1].source_index == 1


def test_goal_item_missing_field_raises():
    bad = _goal()
    del bad["success_condition"]
    with pytest.raises(ValidationError):
        ParseGoalsResponse(goals=[bad])


def test_goal_item_wrong_type_raises():
    with pytest.raises(ValidationError):
        ParseGoalsResponse(goals=[_goal(source_index="zero")])


def test_goal_item_extra_field_forbidden():
    with pytest.raises(ValidationError):
        ParseGoalsResponse(goals=[_goal(unexpected="x")])


def test_model_dump_shape_matches_downstream():
    r = ParseGoalsResponse(goals=[_goal()])
    dumped = [g.model_dump() for g in r.goals]
    assert dumped[0]["sig"] == "achieve_have_wheat"
    assert dumped[0]["source_index"] == 0
    assert set(dumped[0].keys()) == {
        "sig", "source_index", "priority", "reason", "success_condition",
    }


# ===========================================================================
# Step0 / Step2 / Step1
# ===========================================================================

def test_step0_response_valid_and_missing():
    assert Step0Response(sig="achieve_x", description="do the thing").sig == "achieve_x"
    with pytest.raises(ValidationError):
        Step0Response(sig="achieve_x")  # falta description


def test_step2_response_known_facts_optional():
    s = Step2Response(problem_nl="no wheat yet")
    assert s.known_facts == []
    s2 = Step2Response(problem_nl="x", known_facts=["item_spawn(wheat, farmland)"])
    assert s2.known_facts == ["item_spawn(wheat, farmland)"]


def test_step1_response_with_success_model():
    s = Step1Response(success_model=[
        SuccessVariant(facts=["has_item(bread, 1)"], guards=[], done_fragment="has_item(bread, 1)")
    ], done_guard="has_item(bread, 1)")
    assert s.success_model[0].done_fragment == "has_item(bread, 1)"
    assert s.success_conditions == []


def test_step1_variant_done_fragment_optional():
    # done_fragment puede omitirse (run_step1 lo deriva de facts+guards).
    v = SuccessVariant(facts=["has_item(wheat, 1)"])
    assert v.done_fragment == ""


def test_step3_response_heterogeneous_args():
    r = Step3Response(steps=[
        StepItem(type="action", name="MoveTo", args=[10, 5]),
        StepItem(type="subgoal", name="move_to_and_pickup", args=["wheat", 1],
                 description="get wheat"),
    ])
    assert r.steps[0].args == [10, 5]
    assert r.steps[1].args == ["wheat", 1]
    assert r.steps[1].description == "get wheat"


def test_step3_item_ignores_extra_fields():
    # extra="ignore": campos no declarados no rompen (p.ej. replaces_steps).
    item = StepItem(type="action", name="Craft", args=["wheat", "bread_recipe", 1],
                    replaces_steps=[1, 2])
    assert not hasattr(item, "replaces_steps")
    assert item.name == "Craft"
