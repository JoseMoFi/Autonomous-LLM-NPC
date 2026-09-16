from __future__ import annotations

"""
test_smoke_dependency_graph.py — tests de Iter 5.

Cubre:
- DependencyNode: construccion, validacion, belief_str.
- parse_step_breakdown: conversion de dicts a DependencyNodes.
- build_dependency_graph: cadena con goal al final, validacion de producers.
- derive_variants: ejemplo canonico del trigo (DOC/PIPELINE_OBJETIVO_USUARIO.md),
  casos borde (sin adquiribles, solo goal, varios required).
- Variant: guard, parts, is_final, producer_action.
"""

import pytest

from llm.pipeline.dependency_graph import (
    DependencyNode,
    Variant,
    build_dependency_graph,
    derive_variants,
    parse_step_breakdown,
)


# ---------------------------------------------------------------------------
# DependencyNode
# ---------------------------------------------------------------------------


class TestDependencyNode:
    def test_construccion_minima(self) -> None:
        n = DependencyNode(functor="has_item", args=["wheat", "N"], state="required")
        assert n.functor == "has_item"
        assert n.args == ["wheat", "N"]
        assert n.state == "required"
        assert n.producer == ""

    def test_functor_se_normaliza_lowercase(self) -> None:
        n = DependencyNode(functor="HAS_ITEM", args=[], state="required")
        assert n.functor == "has_item"

    def test_state_validos(self) -> None:
        for s in ("required", "provided", "observed"):
            n = DependencyNode(functor="x", state=s)
            assert n.state == s

    def test_state_invalido_lanza(self) -> None:
        with pytest.raises(ValueError, match="invalido"):
            DependencyNode(functor="x", state="guaranteed")

    def test_functor_vacio_lanza(self) -> None:
        with pytest.raises(ValueError, match="vacio"):
            DependencyNode(functor="")

    def test_belief_str_con_args(self) -> None:
        n = DependencyNode(functor="zone_center", args=["Z", "ZX", "ZY"], state="observed")
        assert n.belief_str() == "zone_center(Z, ZX, ZY)"

    def test_belief_str_sin_args(self) -> None:
        n = DependencyNode(functor="idle", state="required")
        assert n.belief_str() == "idle"


# ---------------------------------------------------------------------------
# parse_step_breakdown
# ---------------------------------------------------------------------------


class TestParseStepBreakdown:
    def _wheat_breakdown(self) -> list[dict]:
        return [
            {"functor": "item_spawn",  "args": ["wheat", "Z"],        "state": "required", "producer": ""},
            {"functor": "zone_center", "args": ["Z", "ZX", "ZY"],     "state": "observed", "producer": "ExploreArea"},
            {"functor": "item_at",     "args": ["wheat", "WX", "WY"], "state": "observed", "producer": "Search"},
        ]

    def test_parse_tres_nodos(self) -> None:
        nodes = parse_step_breakdown(self._wheat_breakdown())
        assert len(nodes) == 3

    def test_parse_estado_y_producer(self) -> None:
        nodes = parse_step_breakdown(self._wheat_breakdown())
        assert nodes[0].state == "required"
        assert nodes[1].state == "observed"
        assert nodes[1].producer == "ExploreArea"
        assert nodes[2].producer == "Search"

    def test_parse_functor_normalizado_lowercase(self) -> None:
        nodes = parse_step_breakdown([{"functor": "HAS_ITEM", "args": ["X"], "state": "provided"}])
        assert nodes[0].functor == "has_item"

    def test_parse_estado_desconocido_degrada_a_required(self) -> None:
        nodes = parse_step_breakdown([{"functor": "x", "state": "unknown_state"}])
        assert nodes[0].state == "required"

    def test_parse_functor_vacio_se_omite(self) -> None:
        nodes = parse_step_breakdown([{"functor": ""}, {"functor": "has_item", "state": "provided"}])
        assert len(nodes) == 1
        assert nodes[0].functor == "has_item"

    def test_parse_no_dict_se_omite(self) -> None:
        nodes = parse_step_breakdown(["not_a_dict", {"functor": "x", "state": "required"}])
        assert len(nodes) == 1

    def test_parse_lista_vacia(self) -> None:
        assert parse_step_breakdown([]) == []


# ---------------------------------------------------------------------------
# build_dependency_graph
# ---------------------------------------------------------------------------


class TestBuildDependencyGraph:
    def _wheat_chain(self) -> list[DependencyNode]:
        return parse_step_breakdown([
            {"functor": "item_spawn",  "args": ["wheat", "Z"],        "state": "required"},
            {"functor": "zone_center", "args": ["Z", "ZX", "ZY"],     "state": "observed", "producer": "ExploreArea"},
            {"functor": "item_at",     "args": ["wheat", "WX", "WY"], "state": "observed", "producer": "Search"},
        ])

    def test_goal_al_final(self) -> None:
        chain = build_dependency_graph("has_item(wheat, N)", self._wheat_chain())
        assert chain[-1].functor == "has_item"
        assert chain[-1].args == ["wheat", "N"]

    def test_longitud_chain(self) -> None:
        chain = build_dependency_graph("has_item(wheat, N)", self._wheat_chain())
        assert len(chain) == 4  # 3 breakdown + 1 goal

    def test_goal_state_provided(self) -> None:
        chain = build_dependency_graph("has_item(wheat, N)", self._wheat_chain())
        assert chain[-1].state == "provided"

    def test_goal_sin_args(self) -> None:
        chain = build_dependency_graph("idle", [])
        assert len(chain) == 1
        assert chain[0].functor == "idle"
        assert chain[0].args == []

    def test_producer_invalido_lanza_sin_registry(self) -> None:
        """Sin registry, no se validan producers."""
        nodes = [DependencyNode(functor="x", state="observed", producer="AccionInexistente")]
        # Sin registry: no lanza
        chain = build_dependency_graph("goal", nodes, contracts=None)
        assert len(chain) == 2

    def test_producer_invalido_lanza_con_registry(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        nodes = [DependencyNode(functor="x", state="observed", producer="AccionInexistente")]
        with pytest.raises(ValueError, match="AccionInexistente"):
            build_dependency_graph("goal", nodes, contracts=CONTRACT_REGISTRY)

    def test_producer_valido_no_lanza_con_registry(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        nodes = [DependencyNode(functor="zone_center", args=["Z","X","Y"],
                                state="observed", producer="ExploreArea")]
        chain = build_dependency_graph("has_item(wheat, N)", nodes, contracts=CONTRACT_REGISTRY)
        assert len(chain) == 2


# ---------------------------------------------------------------------------
# derive_variants — ejemplo canonico del trigo
# ---------------------------------------------------------------------------

# Variantes exactas esperadas por DOC/PIPELINE_OBJETIVO_USUARIO.md
_WHEAT_VARIANTS_EXPECTED = [
    "not has_item(wheat, N) & item_spawn(wheat, Z) & not zone_center(Z, ZX, ZY)",
    "not has_item(wheat, N) & item_spawn(wheat, Z) & zone_center(Z, ZX, ZY) & not item_at(wheat, WX, WY)",
    "not has_item(wheat, N) & item_spawn(wheat, Z) & zone_center(Z, ZX, ZY) & item_at(wheat, WX, WY)",
]


def _wheat_chain() -> list[DependencyNode]:
    breakdown = parse_step_breakdown([
        {"functor": "item_spawn",  "args": ["wheat", "Z"],        "state": "required"},
        {"functor": "zone_center", "args": ["Z", "ZX", "ZY"],     "state": "observed", "producer": "ExploreArea"},
        {"functor": "item_at",     "args": ["wheat", "WX", "WY"], "state": "observed", "producer": "Search"},
    ])
    return build_dependency_graph("has_item(wheat, N)", breakdown)


class TestDeriveVariants:

    # --- ejemplo trigo ---

    def test_trigo_genera_tres_variantes(self) -> None:
        variants = derive_variants(_wheat_chain())
        assert len(variants) == 3

    def test_trigo_variante_1_guard(self) -> None:
        variants = derive_variants(_wheat_chain())
        assert variants[0].guard == _WHEAT_VARIANTS_EXPECTED[0]

    def test_trigo_variante_2_guard(self) -> None:
        variants = derive_variants(_wheat_chain())
        assert variants[1].guard == _WHEAT_VARIANTS_EXPECTED[1]

    def test_trigo_variante_3_guard(self) -> None:
        variants = derive_variants(_wheat_chain())
        assert variants[2].guard == _WHEAT_VARIANTS_EXPECTED[2]

    def test_trigo_v1_producer_explorearea(self) -> None:
        variants = derive_variants(_wheat_chain())
        assert variants[0].producer_action == "ExploreArea"

    def test_trigo_v2_producer_search(self) -> None:
        variants = derive_variants(_wheat_chain())
        assert variants[1].producer_action == "Search"

    def test_trigo_v3_is_final(self) -> None:
        """La ultima variante es final: todos los pre-goal beliefs positivos."""
        variants = derive_variants(_wheat_chain())
        assert variants[2].is_final is True

    def test_trigo_v1_v2_not_final(self) -> None:
        variants = derive_variants(_wheat_chain())
        assert variants[0].is_final is False
        assert variants[1].is_final is False

    def test_trigo_v1_negated_belief(self) -> None:
        variants = derive_variants(_wheat_chain())
        assert variants[0].negated_belief == "zone_center(Z, ZX, ZY)"

    def test_trigo_v2_negated_belief(self) -> None:
        variants = derive_variants(_wheat_chain())
        assert variants[1].negated_belief == "item_at(wheat, WX, WY)"

    def test_trigo_v3_negated_belief_vacio(self) -> None:
        variants = derive_variants(_wheat_chain())
        assert variants[2].negated_belief == ""

    def test_trigo_todas_las_variantes_niegan_el_goal(self) -> None:
        variants = derive_variants(_wheat_chain())
        for v in variants:
            assert "not has_item(wheat, N)" in v.guard

    def test_trigo_item_spawn_siempre_positivo(self) -> None:
        """item_spawn es 'required': nunca se niega."""
        variants = derive_variants(_wheat_chain())
        for v in variants:
            assert "not item_spawn" not in v.guard
            assert "item_spawn(wheat, Z)" in v.guard

    # --- casos borde ---

    def test_chain_vacia_devuelve_vacio(self) -> None:
        assert derive_variants([]) == []

    def test_solo_goal_un_variant_final(self) -> None:
        """Con chain=[goal] no hay beliefs previos: una sola variante final."""
        chain = [DependencyNode(functor="idle", state="provided")]
        variants = derive_variants(chain)
        assert len(variants) == 1
        assert variants[0].is_final is True
        assert variants[0].guard == "not idle"

    def test_todos_required_un_variant_final(self) -> None:
        """Si todos los pre-goal beliefs son 'required', no se generan variantes intermedias."""
        chain = build_dependency_graph("has_item(wheat, N)", parse_step_breakdown([
            {"functor": "item_spawn", "args": ["wheat", "Z"], "state": "required"},
            {"functor": "recipe",     "args": ["bread"],       "state": "required"},
        ]))
        variants = derive_variants(chain)
        assert len(variants) == 1
        assert variants[0].is_final is True

    def test_chain_solo_observed_sin_required(self) -> None:
        """Con dos beliefs observacionales y goal: genera 3 variantes."""
        breakdown = parse_step_breakdown([
            {"functor": "zone_center", "args": ["Z","ZX","ZY"], "state": "observed", "producer": "ExploreArea"},
            {"functor": "item_at",     "args": ["wheat","WX","WY"],"state": "observed","producer": "Search"},
        ])
        chain = build_dependency_graph("has_item(wheat, N)", breakdown)
        variants = derive_variants(chain)
        assert len(variants) == 3

    def test_parts_consistente_con_guard(self) -> None:
        """Variant.guard debe ser ' & '.join(parts)."""
        variants = derive_variants(_wheat_chain())
        for v in variants:
            assert " & ".join(v.parts) == v.guard


# ---------------------------------------------------------------------------
# TestIter7ExhaustionBranches — derive_variants con contracts
# ---------------------------------------------------------------------------


class TestIter7ExhaustionBranches:
    """Tests de Iter 7: variantes de agotamiento cuando contracts es proporcionado."""

    def _wheat_chain_with_contracts(self) -> list[DependencyNode]:
        breakdown = parse_step_breakdown([
            {"functor": "item_spawn",  "args": ["wheat", "Z"],        "state": "required"},
            {"functor": "zone_center", "args": ["Z", "ZX", "ZY"],     "state": "observed", "producer": "ExploreArea"},
            {"functor": "item_at",     "args": ["wheat", "WX", "WY"], "state": "observed", "producer": "Search"},
        ])
        return build_dependency_graph("has_item(wheat, N)", breakdown)

    def test_con_contracts_genera_cinco_variantes(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        variants = derive_variants(self._wheat_chain_with_contracts(), contracts=CONTRACT_REGISTRY)
        assert len(variants) == 5

    def test_sin_contracts_sigue_generando_tres(self) -> None:
        """Backward compat: contracts=None => mismas 3 variantes de Iter 5."""
        variants = derive_variants(self._wheat_chain_with_contracts())
        assert len(variants) == 3

    def test_dos_exhaustion_branches(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        variants = derive_variants(self._wheat_chain_with_contracts(), contracts=CONTRACT_REGISTRY)
        exh = [v for v in variants if v.is_exhaustion_branch]
        assert len(exh) == 2

    def test_exhaustion_explorearea_guard(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        variants = derive_variants(self._wheat_chain_with_contracts(), contracts=CONTRACT_REGISTRY)
        exh = [v for v in variants if v.is_exhaustion_branch and v.producer_action == "ExploreArea"]
        assert len(exh) == 1
        assert "exhausted(ExploreArea, 2)" in exh[0].guard
        assert "not zone_center(Z, ZX, ZY)" in exh[0].guard

    def test_exhaustion_search_guard(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        variants = derive_variants(self._wheat_chain_with_contracts(), contracts=CONTRACT_REGISTRY)
        exh = [v for v in variants if v.is_exhaustion_branch and v.producer_action == "Search"]
        assert len(exh) == 1
        assert "exhausted(Search, 3)" in exh[0].guard
        assert "not item_at(wheat, WX, WY)" in exh[0].guard

    def test_exhaustion_branch_no_is_final(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        variants = derive_variants(self._wheat_chain_with_contracts(), contracts=CONTRACT_REGISTRY)
        for v in variants:
            if v.is_exhaustion_branch:
                assert v.is_final is False

    def test_orden_variantes_normal_luego_exhaustion(self) -> None:
        """La variante de agotamiento va justo despues de la normal."""
        from protocol.action_semantics import CONTRACT_REGISTRY
        variants = derive_variants(self._wheat_chain_with_contracts(), contracts=CONTRACT_REGISTRY)
        # index 0: normal ExploreArea, index 1: exhaustion ExploreArea
        assert variants[0].is_exhaustion_branch is False
        assert variants[1].is_exhaustion_branch is True
        assert variants[1].producer_action == "ExploreArea"
        # index 2: normal Search, index 3: exhaustion Search
        assert variants[2].is_exhaustion_branch is False
        assert variants[3].is_exhaustion_branch is True
        assert variants[3].producer_action == "Search"
        # index 4: final
        assert variants[4].is_final is True

    def test_attempt_budget_propagado(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        variants = derive_variants(self._wheat_chain_with_contracts(), contracts=CONTRACT_REGISTRY)
        normal_explore = variants[0]
        assert normal_explore.attempt_budget == 2  # ExploreArea.attempt_budget
        normal_search = variants[2]
        assert normal_search.attempt_budget == 3   # Search.attempt_budget

    def test_exhaustion_parts_consistente_con_guard(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        variants = derive_variants(self._wheat_chain_with_contracts(), contracts=CONTRACT_REGISTRY)
        for v in variants:
            assert " & ".join(v.parts) == v.guard

    def test_goal_negado_en_todas(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        variants = derive_variants(self._wheat_chain_with_contracts(), contracts=CONTRACT_REGISTRY)
        for v in variants:
            assert "not has_item(wheat, N)" in v.guard

    def test_exhaustion_policy_asl_goal_failed(self) -> None:
        from llm.pipeline.dependency_graph import exhaustion_policy_asl
        from llm.pipeline.dependency_graph import Variant
        v = Variant.build(
            parts=["not goal", "not some_belief", "exhausted(TestAction, 1)"],
            negated_belief="some_belief",
            producer_action="TestAction",
            is_final=False,
            attempt_budget=1,
            on_exhaustion="goal_failed",
            is_exhaustion_branch=True,
        )
        asl = exhaustion_policy_asl("goal(X)", v)
        assert ".fail" in asl
        assert "+!goal(X)" in asl

    def test_exhaustion_policy_asl_replan_stub(self) -> None:
        from llm.pipeline.dependency_graph import exhaustion_policy_asl
        from llm.pipeline.dependency_graph import Variant
        v = Variant.build(
            parts=["not goal", "not some_belief", "exhausted(TestAction, 2)"],
            negated_belief="some_belief",
            producer_action="TestAction",
            is_final=False,
            attempt_budget=2,
            on_exhaustion="replan_goal",
            is_exhaustion_branch=True,
        )
        asl = exhaustion_policy_asl("goal(X)", v)
        assert ".print" in asl
        assert "replan_goal" in asl

    def test_required_nodes_no_generan_exhaustion(self) -> None:
        """Nodes con state=required nunca generan exhaustion branches."""
        from protocol.action_semantics import CONTRACT_REGISTRY
        chain = build_dependency_graph("has_item(wheat, N)", parse_step_breakdown([
            {"functor": "item_spawn", "args": ["wheat", "Z"], "state": "required"},
            {"functor": "recipe", "args": ["bread"], "state": "required"},
        ]))
        variants = derive_variants(chain, contracts=CONTRACT_REGISTRY)
        exh = [v for v in variants if v.is_exhaustion_branch]
        assert len(exh) == 0
