"""Smoke tests — pipeline lógica Python pura (sin LLM).

Cubre:
  - step4_classify: classify_steps + enqueue_if_needed
  - step2_guards: build_guard_expression + extract_bound_variables
  - llm/parser: parse_llm_response
  - llm/validator: validate_goal_name, validate_guards_json
"""

from __future__ import annotations

import collections
import pytest
import networkx as nx

from llm.pipeline.step4_classify import classify_steps, enqueue_if_needed
from llm.pipeline.step2_guards import build_guard_expression, extract_bound_variables
from llm.parser import parse_llm_response
from llm.validator import validate_goal_name, validate_guards_json, validate_parse_goals_json
from llm.catalogs import PRIMITIVE_ACTIONS


# ===========================================================================
# step4_classify
# ===========================================================================

class TestClassifySteps:
    def test_primitive_actions_recognized(self) -> None:
        steps = [
            {"type": "action", "name": "MoveTo", "args": ["10", "20"]},
            {"type": "action", "name": "PickUp", "args": ["wheat"]},
            {"type": "action", "name": "Wait",   "args": []},
        ]
        primitives, to_expand = classify_steps(steps)
        assert len(primitives) == 3
        assert to_expand == []

    def test_subgoal_not_primitive(self) -> None:
        steps = [
            {"type": "subgoal", "name": "acquire_wheat", "args": []},
        ]
        primitives, to_expand = classify_steps(steps)
        assert primitives == []
        assert [s["name"] for s in to_expand] == ["acquire_wheat"]

    def test_unknown_action_type_becomes_subgoal(self) -> None:
        steps = [
            {"type": "action", "name": "MysteryAction", "args": []},
        ]
        primitives, to_expand = classify_steps(steps)
        assert primitives == []
        assert any(s["name"] == "MysteryAction" for s in to_expand)

    def test_mixed_steps(self) -> None:
        steps = [
            {"type": "action",  "name": "MoveTo",        "args": []},
            {"type": "subgoal", "name": "acquire_wheat", "args": []},
            {"type": "action",  "name": "Craft",         "args": []},
        ]
        primitives, to_expand = classify_steps(steps)
        assert len(primitives) == 2
        assert [s["name"] for s in to_expand] == ["acquire_wheat"]


class TestEnqueueIfNeeded:
    def _setup(self):
        dag = nx.DiGraph()
        dag.add_node("parent")
        plan_library = {}
        queue = collections.deque()
        return dag, plan_library, queue

    def test_new_goal_enqueued(self) -> None:
        dag, lib, queue = self._setup()
        result = enqueue_if_needed("child", "parent", dag, lib, queue)
        assert result is True
        assert "child" in queue
        assert dag.has_node("child")

    def test_existing_in_library_not_enqueued(self) -> None:
        dag, lib, queue = self._setup()
        lib["child"] = {"asl": "..."}
        result = enqueue_if_needed("child", "parent", dag, lib, queue)
        assert result is True
        assert len(queue) == 0
        assert dag.has_edge("parent", "child")

    def test_already_in_dag_not_enqueued(self) -> None:
        dag, lib, queue = self._setup()
        dag.add_node("child", status="pending")
        result = enqueue_if_needed("child", "parent", dag, lib, queue)
        assert result is True
        assert len(queue) == 0

    def test_cycle_detection_returns_false(self) -> None:
        dag, lib, queue = self._setup()
        # "parent"→"child_a" already. Adding "parent" as a child of "child_a"
        # would create: child_a → parent → child_a (cycle).
        dag.add_node("child_a")
        dag.add_edge("parent", "child_a")      # parent → child_a
        # Now try to add "parent" as child of "child_a" → would make cycle
        result = enqueue_if_needed("parent", "child_a", dag, lib, queue)
        assert result is False
        # The cycle edge must not remain
        assert not dag.has_edge("child_a", "parent")


# ===========================================================================
# step2_guards helpers
# ===========================================================================

class TestBuildGuardExpression:
    def test_facts_only(self) -> None:
        facts = [{"functor": "knows_recipe", "args": ["bread", "2", "wheat"]}]
        guards = []
        expr = build_guard_expression(facts, guards)
        assert expr == "knows_recipe(bread, 2, wheat)"

    def test_facts_and_guard(self) -> None:
        facts = [{"functor": "has_item", "args": ["wheat", "N"]}]
        guards = [{"expr": "N >= 2"}]
        expr = build_guard_expression(facts, guards)
        assert expr == "has_item(wheat, N) & N >= 2"

    def test_empty_returns_true(self) -> None:
        assert build_guard_expression([], []) == "true"

    def test_multiple_facts(self) -> None:
        facts = [
            {"functor": "knows_recipe", "args": ["bread"]},
            {"functor": "has_item",     "args": ["wheat", "N"]},
        ]
        guards = [{"expr": "N >= 2"}]
        expr = build_guard_expression(facts, guards)
        assert "knows_recipe(bread)" in expr
        assert "has_item(wheat, N)" in expr
        assert "N >= 2" in expr


class TestExtractBoundVariables:
    def test_uppercase_args_extracted(self) -> None:
        facts = [{"functor": "has_item", "args": ["wheat", "N"]}]
        guards = []
        variables = extract_bound_variables(facts, guards)
        assert "N" in variables
        assert "wheat" not in variables

    def test_no_variables(self) -> None:
        facts = [{"functor": "has_item", "args": ["wheat", "2"]}]
        variables = extract_bound_variables(facts, [])
        assert variables == []

    def test_multiple_variables(self) -> None:
        facts = [{"functor": "item_at", "args": ["TargetItem", "X", "Y"]}]
        variables = extract_bound_variables(facts, [])
        assert set(variables) == {"TargetItem", "X", "Y"}

    def test_deduplication(self) -> None:
        facts = [
            {"functor": "f1", "args": ["N"]},
            {"functor": "f2", "args": ["N"]},
        ]
        variables = extract_bound_variables(facts, [])
        assert variables.count("N") == 1


# ===========================================================================
# llm/parser
# ===========================================================================

class TestParseLlmResponse:
    def test_clean_json_object(self) -> None:
        raw = '{"goal": "achieve_craft_bread", "npc_statement": "I want to craft bread"}'
        result = parse_llm_response(raw, "step1_name")
        assert isinstance(result, dict)
        assert result["goal"] == "achieve_craft_bread"

    def test_json_array(self) -> None:
        raw = '[{"sig": "achieve_x", "priority": 1.0}]'
        result = parse_llm_response(raw, "parse_goals")
        assert isinstance(result, list)
        assert result[0]["sig"] == "achieve_x"

    def test_strips_markdown_fences(self) -> None:
        raw = "```json\n{\"goal\": \"achieve_x\"}\n```"
        result = parse_llm_response(raw, "step1_name")
        assert isinstance(result, dict)
        assert result["goal"] == "achieve_x"

    def test_extra_text_around_json(self) -> None:
        raw = 'Sure! Here is the result:\n{"goal": "achieve_x"}\nHope that helps.'
        result = parse_llm_response(raw, "step1_name")
        assert isinstance(result, dict)

    def test_invalid_json_returns_empty_dict(self) -> None:
        raw = "I cannot help with that."
        result = parse_llm_response(raw, "step1_name")
        assert result == {}


# ===========================================================================
# llm/validator
# ===========================================================================

class TestValidateGoalName:
    def test_valid_achieve(self) -> None:
        assert validate_goal_name("achieve_craft_bread") == []

    def test_valid_achieve_long(self) -> None:
        assert validate_goal_name("achieve_deliver_bread") == []

    def test_valid_get_prefix(self) -> None:
        assert validate_goal_name("get_wheat") == []

    def test_valid_find_prefix(self) -> None:
        assert validate_goal_name("find_zone_farmland") == []

    def test_valid_craft_prefix(self) -> None:
        assert validate_goal_name("craft_bread") == []

    def test_valid_deliver_prefix(self) -> None:
        assert validate_goal_name("deliver_bread") == []

    def test_valid_flee_prefix(self) -> None:
        assert validate_goal_name("flee_danger") == []

    def test_valid_explore_prefix(self) -> None:
        assert validate_goal_name("explore_farmland") == []

    def test_unknown_prefix_invalid(self) -> None:
        errors = validate_goal_name("do_something_important")
        assert errors != []

    def test_empty_string_invalid(self) -> None:
        assert validate_goal_name("") != []

    def test_spaces_invalid(self) -> None:
        assert validate_goal_name("achieve craft bread") != []

    def test_camelcase_invalid(self) -> None:
        assert validate_goal_name("AchieveCraftBread") != []


class TestValidateGuardsJson:
    def test_no_guards_valid(self) -> None:
        payload = {
            "facts": [{"functor": "has_item", "args": ["wheat", "3"]}],
            "guards": [],
        }
        errors = validate_guards_json(payload)
        assert errors == []

    def test_bound_variable_used_correctly(self) -> None:
        payload = {
            "facts": [{"functor": "has_item", "args": ["wheat", "N"]}],
            "guards": [{"expr": "N >= 2"}],
        }
        errors = validate_guards_json(payload)
        assert errors == []

    def test_unbound_variable_in_guard(self) -> None:
        payload = {
            "facts": [],
            "guards": [{"expr": "N >= 2"}],
        }
        errors = validate_guards_json(payload)
        assert any("N" in e for e in errors)

    def test_unknown_functor_errors(self) -> None:
        payload = {
            "facts": [{"functor": "nonexistent_belief", "args": []}],
            "guards": [],
        }
        errors = validate_guards_json(payload)
        assert errors != []


class TestValidateParseGoalsJson:
    def test_parse_goals_rejects_placeholder_condition(self) -> None:
        payload = [
            {
                "sig": "achieve_have_wheat",
                "priority": 1.0,
                "reason": "Need wheat",
                "success_condition": "belief(condition_met)",
            }
        ]
        errors = validate_parse_goals_json(payload)
        assert any("placeholder" in e.lower() for e in errors)

    def test_parse_goals_rejects_adjacent_atoms_without_and(self) -> None:
        payload = [
            {
                "sig": "achieve_have_wheat",
                "priority": 1.0,
                "reason": "Need wheat",
                "success_condition": "has_item(wheat, 1) knows_zone(farmland)",
            }
        ]
        errors = validate_parse_goals_json(payload)
        assert any("must join predicates with ' & '" in e for e in errors)

    def test_parse_goals_accepts_valid_condition(self) -> None:
        payload = [
            {
                "sig": "achieve_have_wheat",
                "priority": 1.0,
                "reason": "Need wheat",
                "success_condition": "has_item(wheat, 1)",
            }
        ]
        errors = validate_parse_goals_json(payload)
        assert errors == []

    def test_parse_goals_accepts_valid_source_index(self) -> None:
        payload = [
            {
                "sig": "achieve_have_wheat",
                "source_index": 0,
                "priority": 1.0,
                "reason": "Need wheat",
                "success_condition": "has_item(wheat, 1)",
            }
        ]
        assert validate_parse_goals_json(payload, n_goals=1) == []

    def test_parse_goals_rejects_out_of_range_source_index(self) -> None:
        payload = [
            {
                "sig": "achieve_have_wheat",
                "source_index": 5,
                "priority": 1.0,
                "reason": "Need wheat",
                "success_condition": "has_item(wheat, 1)",
            }
        ]
        errors = validate_parse_goals_json(payload, n_goals=2)
        assert any("source_index" in e and "out of range" in e for e in errors)

    def test_parse_goals_rejects_non_integer_source_index(self) -> None:
        payload = [
            {
                "sig": "achieve_have_wheat",
                "source_index": "0",
                "priority": 1.0,
                "reason": "Need wheat",
                "success_condition": "has_item(wheat, 1)",
            }
        ]
        errors = validate_parse_goals_json(payload, n_goals=1)
        assert any("source_index must be an integer" in e for e in errors)


class TestValidatePrioritize:
    """0.C8 — validación mínima del resultado de prioritize."""

    def test_valid_prioritize_no_errors(self) -> None:
        from llm.planning_agent import _validate_prioritize
        result = [{"sig": "achieve_a", "score": 0.9}, {"sig": "achieve_b", "score": 0.2}]
        assert _validate_prioritize(result) == []

    def test_prioritize_rejects_non_list(self) -> None:
        from llm.planning_agent import _validate_prioritize
        assert _validate_prioritize({"sig": "x"}) != []

    def test_prioritize_rejects_missing_sig(self) -> None:
        from llm.planning_agent import _validate_prioritize
        errors = _validate_prioritize([{"score": 0.5}])
        assert any("sig" in e for e in errors)

    def test_prioritize_rejects_non_numeric_score(self) -> None:
        from llm.planning_agent import _validate_prioritize
        errors = _validate_prioritize([{"sig": "achieve_a", "score": "high"}])
        assert any("score" in e for e in errors)


# ===========================================================================
# _format_entity_catalog / prompt injection
# ===========================================================================

class TestEntityCatalogPromptInjection:
    """Tests for _format_entity_catalog utility and its injection into prompts."""

    def _catalog(self) -> dict:
        return {
            "zone_ids": ["bakery", "farmland"],
            "item_ids": ["bread", "wheat"],
            "delivery_tags": ["tavern_point"],
            "recipe_ids": ["bread_recipe"],
        }

    def test_format_entity_catalog_nonempty(self) -> None:
        from llm.prompts.planning import _format_entity_catalog
        text = _format_entity_catalog(self._catalog())
        assert "bakery" in text
        assert "farmland" in text
        assert "wheat" in text
        assert "bread" in text
        assert "tavern_point" in text
        assert "bread_recipe" in text

    def test_format_entity_catalog_empty_returns_empty_string(self) -> None:
        from llm.prompts.planning import _format_entity_catalog
        assert _format_entity_catalog({}) == ""
        assert _format_entity_catalog({"zone_ids": [], "item_ids": []}) == ""

    def test_step2_guards_prompt_contains_catalog(self) -> None:
        from llm.prompts.planning import build_prompt
        p = {
            "task": "step2_guards",
            "goal_name": "achieve_deliver_bread",
            "npc_statement": "I want to deliver bread.",
            "entity_catalog": self._catalog(),
        }
        user, _ = build_prompt(p)
        assert "farmland" in user
        assert "wheat" in user

    def test_step3_steps_prompt_contains_catalog(self) -> None:
        from llm.prompts.planning import build_prompt
        p = {
            "task": "step3_steps",
            "goal_name": "achieve_deliver_bread",
            "npc_statement": "I want to deliver bread.",
            "facts": [],
            "guards": [],
            "bound_variables": [],
            "existing_subgoals": [],
            "entity_catalog": self._catalog(),
        }
        user, _ = build_prompt(p)
        assert "farmland" in user
        assert "wheat" in user

    def test_step3_steps_atomic_prompt_allows_subgoals(self) -> None:
        from llm.prompts.planning import build_prompt
        p = {
            "task": "step3_steps",
            "goal_name": "achieve_deliver_bread",
            "npc_statement": "I want to deliver bread.",
            "facts": [],
            "guards": [],
            "bound_variables": [],
            "existing_subgoals": ["move_to_and_pickup"],
            "entity_catalog": self._catalog(),
            "atomic_only": True,
        }
        user, system = build_prompt(p)
        assert "at least one primitive action" in system
        assert "PREFER a reusable sub-goal" in user
        assert '"type": "subgoal"' in user
        assert "move_to_and_pickup" in user

    def test_step3_steps_atomic_prompt_explains_subgoal_usage(self) -> None:
        from llm.prompts.planning import build_prompt
        p = {
            "task": "step3_steps",
            "goal_name": "achieve_deliver_bread",
            "npc_statement": "I want to deliver bread.",
            "facts": [],
            "guards": [],
            "bound_variables": [],
            "existing_subgoals": ["move_to_and_pickup"],
            "belief_gap_hints": ["has_item(bread, N) is NOT in beliefs."],
            "entity_catalog": self._catalog(),
            "atomic_only": True,
        }
        user, _ = build_prompt(p)
        assert "PREFER a reusable sub-goal" in user
        assert "Avoid hardcoding raw coordinates" in user
        assert "move_to_and_pickup" in user

    def test_step3_steps_prompt_lists_zone_explore_subgoal_when_available(self) -> None:
        from llm.prompts.planning import build_prompt
        p = {
            "task": "step3_steps",
            "goal_name": "achieve_have_item_wheat",
            "npc_statement": "I want wheat.",
            "facts": [],
            "guards": [],
            "bound_variables": [],
            "existing_subgoals": ["achieve_explore_zone", "move_to_and_pickup"],
            "entity_catalog": self._catalog(),
        }
        user, _ = build_prompt(p)
        assert "achieve_explore_zone(zoneTag)" in user
        assert "move_to_and_pickup(itemId, qty)" in user

    def test_step3_steps_atomic_prompt_forbids_explorearea_when_zone_known(self) -> None:
        from llm.prompts.planning import build_prompt
        p = {
            "task": "step3_steps",
            "goal_name": "achieve_have_item_wheat",
            "npc_statement": "I want wheat.",
            "facts": [],
            "guards": [],
            "bound_variables": [],
            "existing_subgoals": ["move_to_and_pickup"],
            "belief_gap_hints": [
                "has_item(wheat, N) is NOT in beliefs. Relevant support facts: item_spawn(wheat, farmland), zone_center(farmland, X, Y)."
            ],
            "entity_catalog": self._catalog(),
            "atomic_only": True,
        }
        user, _ = build_prompt(p)
        assert "do NOT use ExploreArea(ZoneTag)" in user
        assert "prefer MoveTo(X, Y) and Search(itemId)" in user

    def test_step1_success_prompt_requests_structured_success_model(self) -> None:
        from llm.prompts.planning import build_prompt
        p = {
            "task": "step1_success",
            "sig": "achieve_deliver_bread",
            "description": "The NPC must deliver bread to the tavern.",
            "entity_catalog": self._catalog(),
        }
        user, _ = build_prompt(p)
        assert "success_model" in user
        assert "done_fragment" in user

    def test_step2_problem_prompt_contains_catalog_and_variant_context(self) -> None:
        from llm.prompts.planning import build_prompt
        p = {
            "task": "step2_problem",
            "sig": "achieve_deliver_bread",
            "description": "The NPC must deliver bread to the tavern.",
            "neg_guard": "not has_item(bread, N) & N >= 1",
            "facts": ["has_item(bread, N)"],
            "guards": ["N >= 1"],
            "bound_variables": ["N"],
            "entity_catalog": self._catalog(),
        }
        user, _ = build_prompt(p)
        assert "Available primitive actions" in user
        assert "Variant facts" in user
        assert "Bound variables" in user

    def test_step5_need_plan_prompt_is_available(self) -> None:
        from llm.prompts.planning import build_prompt
        p = {
            "task": "step5_need_plan",
            "goal_name": "achieve_craft_bread",
            "need_kind": "inventory_at_least",
            "need_payload": {"item": "wheat", "qty": 2},
            "need_rationale": "Craft needs wheat",
            "existing_subgoals": ["move_to_and_pickup"],
            "entity_catalog": self._catalog(),
        }
        user, system = build_prompt(p)
        assert "Generate ONE new reusable sub-goal" in system
        assert "need_kind" not in user  # should render value, not literal key name
        assert "inventory_at_least" in user
        assert "wheat" in user

    def test_step2_no_catalog_does_not_crash(self) -> None:
        """Catalog is optional — prompts must work without it."""
        from llm.prompts.planning import build_prompt
        p = {
            "task": "step2_guards",
            "goal_name": "achieve_deliver_bread",
            "npc_statement": "I want to deliver bread.",
        }
        user, _ = build_prompt(p)
        # No entity_catalog key → no catalog block, but prompt still works
        assert "Goal:" in user


# ===========================================================================
# step5 contracts store
# ===========================================================================

class TestCapabilityContracts:
    def test_validate_contracts_accepts_minimal_valid_shape(self) -> None:
        from llm.pipeline.capability_contracts import validate_contracts

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
                }
            },
        }
        assert validate_contracts(contracts) == []

    def test_upsert_created_plan_contracts_registers_new_sig(self) -> None:
        from llm.pipeline.capability_contracts import upsert_created_plan_contracts

        contracts = {"version": 1, "contracts": {}}
        changed = upsert_created_plan_contracts(
            contracts,
            [
                {
                    "sig": "achieve_collect_wheat",
                    "description": "Collect wheat.",
                    "provides": [
                        {
                            "kind": "inventory_at_least",
                            "constraints": {"item": "wheat", "qty": 2},
                        }
                    ],
                    "param_names": ["qty"],
                }
            ],
        )

        assert changed is True
        assert "achieve_collect_wheat" in contracts["contracts"]

    def test_upsert_created_plan_contracts_idempotent(self) -> None:
        from llm.pipeline.capability_contracts import upsert_created_plan_contracts

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
        changed = upsert_created_plan_contracts(
            contracts,
            [
                {
                    "sig": "achieve_collect_wheat",
                    "description": "Collect wheat.",
                    "provides": [
                        {
                            "kind": "inventory_at_least",
                            "constraints": {"item": "wheat", "qty": 2},
                        }
                    ],
                    "param_names": ["qty"],
                }
            ],
        )

        assert changed is False
