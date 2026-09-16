"""Tests del generador de familia paramétrica (Fase 6.5, T4).

Valida sin Ollama/Unity: (a) cada variante COMPILA como plan ASL ejecutable;
(b) los guards ENRUTAN correctamente cada binding (wheat→gather, bread→craft);
(c) la rama craft reúne los ingredientes de forma recursiva.
"""

import agentspeak
import agentspeak.stdlib

from llm.family_plan import build_has_item_family, FAMILY_SIG
from npc.asp_guard import compile_plan_with_actions, GuardEvaluator

# Mundo de prueba: bread se craftea de wheat×2; wheat se recolecta.
RECIPES = [
    {"recipeId": "bread_recipe", "zone": "bakeri",
     "inputs": [{"itemId": "wheat", "qty": 2}],
     "outputs": [{"itemId": "bread", "qty": 1}]},
]
SPAWNS = [{"itemId": "wheat", "zones": ["farmland"]}]


def _snapshot(**preds):
    """Construye un snapshot de beliefs: pred=[(args...), ...]."""
    return {k: [tuple(row) for row in v] for k, v in preds.items()}


def test_estructura_done_gather_craft():
    variants = build_has_item_family(RECIPES, SPAWNS)
    # done + gather + 1 craft (una receta)
    assert len(variants) == 3
    assert variants[0].guard == "has_item(Item, Qty)"
    assert "item_spawn(Item, _)" in variants[1].guard
    assert "recipe_output(bread_recipe, _, Item, _)" in variants[2].guard
    # la rama craft reúne wheat recursivamente y delega en el builtin craft_item
    assert f"!{FAMILY_SIG}(wheat, 2)" in variants[2].asl
    assert "!craft_item(bread_recipe, Qty)" in variants[2].asl
    # todo el plan es andamiaje CODE (no contenido del LLM)
    assert all(v.source == "CODE" for v in variants)


def test_todas_las_variantes_compilan():
    actions = agentspeak.Actions(agentspeak.stdlib.actions)
    for v in build_has_item_family(RECIPES, SPAWNS):
        plan = compile_plan_with_actions(v.asl, actions)
        assert plan is not None, f"no compila: {v.asl}"
        assert plan.head.functor == FAMILY_SIG
        assert len(plan.head.args) == 2  # (Item, Qty)


def test_enrutado_wheat_va_por_gather():
    ev = GuardEvaluator()
    snap = _snapshot(item_spawn=[("wheat", "farmland")],
                     recipe_output=[("bread_recipe", "bakeri", "bread", 1)])
    variants = build_has_item_family(RECIPES, SPAWNS)
    gather_guard = variants[1].guard
    craft_guard = variants[2].guard
    # wheat (spawneable, sin receta de output) → gather sí, craft no
    ok_g, _ = ev.eval_guard(gather_guard, ["Item", "Qty"], ["wheat", 2], snap)
    ok_c, _ = ev.eval_guard(craft_guard, ["Item", "Qty"], ["wheat", 2], snap)
    assert ok_g is True
    assert ok_c is False


def test_enrutado_bread_va_por_craft():
    ev = GuardEvaluator()
    snap = _snapshot(item_spawn=[("wheat", "farmland")],
                     recipe_output=[("bread_recipe", "bakeri", "bread", 1)])
    variants = build_has_item_family(RECIPES, SPAWNS)
    gather_guard = variants[1].guard
    craft_guard = variants[2].guard
    # bread (no spawneable, output de receta) → craft sí, gather no
    ok_g, _ = ev.eval_guard(gather_guard, ["Item", "Qty"], ["bread", 1], snap)
    ok_c, _ = ev.eval_guard(craft_guard, ["Item", "Qty"], ["bread", 1], snap)
    assert ok_g is False
    assert ok_c is True


def test_done_casa_si_ya_se_tiene():
    ev = GuardEvaluator()
    snap = _snapshot(has_item=[("bread", 1)])
    done_guard = build_has_item_family(RECIPES, SPAWNS)[0].guard
    ok, _ = ev.eval_guard(done_guard, ["Item", "Qty"], ["bread", 1], snap)
    assert ok is True
    # y NO casa si no se tiene
    ok2, _ = ev.eval_guard(done_guard, ["Item", "Qty"], ["bread", 1], _snapshot())
    assert ok2 is False


def test_sin_spawns_no_emite_gather():
    variants = build_has_item_family(RECIPES, item_spawns=[])
    guards = [v.guard for v in variants]
    assert not any("item_spawn" in g for g in guards)
    # done + craft (sin gather)
    assert len(variants) == 2


def test_sin_recetas_solo_gather():
    variants = build_has_item_family([], SPAWNS)
    assert len(variants) == 2  # done + gather
    assert not any("recipe_output" in v.guard for v in variants)
