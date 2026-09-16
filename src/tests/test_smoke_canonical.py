"""Tests del módulo de canonicalización (Fase 6.5, Tarea 1)."""

from types import SimpleNamespace

from llm.canonical import (
    canonical_key,
    derive_family_sig,
    call_args_from_condition,
    param_names_from_condition,
    canonicalize_goals,
    family_capabilities,
    goal_identity_key,
)
from protocol.action_semantics import CONTRACT_REGISTRY

# En el runtime, canonical_key siempre recibe el entity_catalog del perfil
# (derive_entity_catalog). Con él, las constantes se nombran por su tipo.
CAT = {
    "item_ids": ["wheat", "bread", "flour", "apple"],
    "zone_ids": ["bakeri", "farmland"],
    "recipe_ids": ["bread_recipe"],
    "delivery_tags": ["tavern_delivery"],
}


def test_misma_estructura_mismo_esquema_distinto_binding():
    s1, b1 = canonical_key("has_item(wheat, 2)", CAT)
    s2, b2 = canonical_key("has_item(bread, 1)", CAT)
    assert s1 == s2 == "has_item(Item, Qty)"
    assert b1 == {"Item": "wheat", "Qty": 2}
    assert b2 == {"Item": "bread", "Qty": 1}


def test_sin_catalog_mismo_esquema_nombres_genericos():
    # Sin catálogo los nombres son genéricos (Const) PERO el esquema sigue
    # siendo idéntico entre estructuras iguales → el matching no depende del nombre.
    s1, _ = canonical_key("has_item(wheat, 2)")
    s2, _ = canonical_key("has_item(bread, 1)")
    assert s1 == s2 == "has_item(Const, Qty)"


def test_family_sig_estable_desde_predicado():
    # El sig de familia NO depende del NL: misma estructura → mismo sig.
    assert derive_family_sig("has_item(wheat, 2)") == "achieve_has_item"
    assert derive_family_sig("has_item(bread, 1)") == "achieve_has_item"
    sch, _ = canonical_key("has_item(apple, 3)", CAT)
    assert derive_family_sig(sch) == "achieve_has_item"


def test_constantes_iguales_comparten_variable():
    # El mismo 'wheat' en recipe y has_item debe ligar a la MISMA variable.
    schema, binds = canonical_key("recipe(R, Z, wheat, 2) & has_item(wheat, N)", CAT)
    assert schema == "recipe(R, Z, Item, Qty) & has_item(Item, N)"
    assert binds == {"Item": "wheat", "Qty": 2}


def test_tipado_con_entity_catalog():
    schema, binds = canonical_key(
        "recipe_output(bread_recipe, bakeri, bread, 1)", entity_catalog=CAT
    )
    assert schema == "recipe_output(Recipe, Zone, Item, Qty)"
    assert binds == {
        "Recipe": "bread_recipe", "Zone": "bakeri", "Item": "bread", "Qty": 1,
    }


def test_conjuncion():
    schema, binds = canonical_key("has_item(bread, 1) & at_zone(bakeri)", CAT)
    assert schema == "has_item(Item, Qty) & at_zone(Zone)"
    assert binds == {"Item": "bread", "Qty": 1, "Zone": "bakeri"}


def test_negacion_se_preserva():
    schema, _ = canonical_key("not has_item(bread, 1)", CAT)
    assert schema == "not has_item(Item, Qty)"


def test_call_args_solo_constantes():
    assert call_args_from_condition("has_item(wheat, 2)") == ["wheat", 2]
    # variables libres se omiten
    assert call_args_from_condition("has_item(Item, N)") == []


def test_clausula_no_parseable_verbatim():
    # Un guard numérico no es un predicado: se deja tal cual, sin romper.
    schema, binds = canonical_key("has_item(wheat, N) & N >= 2", CAT)
    assert schema == "has_item(Item, N) & N >= 2"
    assert binds == {"Item": "wheat"}


def test_dos_items_distintos_variables_distintas():
    schema, binds = canonical_key("has_item(wheat, 2) & has_item(flour, 3)", CAT)
    assert schema == "has_item(Item, Qty) & has_item(Item2, Qty2)"
    assert binds == {"Item": "wheat", "Qty": 2, "Item2": "flour", "Qty2": 3}


def _goal(sig, sc):
    return SimpleNamespace(sig=sig, success_condition=sc, call_args=[])


def test_param_names_paralelos_a_call_args():
    # Mismos índices, misma longitud que call_args; nombres tipados por catálogo.
    assert param_names_from_condition("has_item(wheat, 2)", CAT) == ["Item", "Qty"]
    assert call_args_from_condition("has_item(wheat, 2)") == ["wheat", 2]
    # Sin catálogo: el tipo de los strings cae a Const, los enteros a Qty.
    assert param_names_from_condition("has_item(wheat, 2)") == ["Const", "Qty"]


def test_param_names_variables_libres_se_omiten():
    # Igual que call_args: las variables libres no son parámetros del binding.
    assert param_names_from_condition("has_item(Item, N)", CAT) == []
    assert param_names_from_condition("", CAT) == []
    assert param_names_from_condition("N >= 2", CAT) == []


def test_canonicalize_goals_renombra_y_dedup():
    g1 = _goal("achieve_fabricate_bread", "has_item(bread, 1)")
    g2 = _goal("achieve_produce_bread", "has_item(bread, 1)")   # duplicado exacto
    g3 = _goal("achieve_get_wheat", "has_item(wheat, 2)")       # misma familia, otro binding
    out = canonicalize_goals([g1, g2, g3], CAT)
    assert [g.sig for g in out] == ["achieve_has_item", "achieve_has_item"]
    assert out[0].call_args == ["bread", 1]
    assert out[1].call_args == ["wheat", 2]
    # param_names paralelos a call_args (aridad de la cabeza paramétrica).
    assert out[0].param_names == ["Item", "Qty"]
    assert out[1].param_names == ["Item", "Qty"]
    assert len(out) == 2  # g2 deduplicado (misma familia + mismos bindings que g1)


def test_canonicalize_goals_sin_success_condition_intacto():
    g = _goal("achieve_explore_zone", None)
    out = canonicalize_goals([g], CAT)
    assert len(out) == 1
    assert out[0].sig == "achieve_explore_zone"
    assert out[0].call_args == []


def test_family_capabilities_has_item():
    caps = family_capabilities("has_item(Item, Qty)", CONTRACT_REGISTRY)
    # Acciones que garantizan has_item en su contrato semántico.
    assert "PickUp" in caps
    assert "Craft" in caps


def test_family_capabilities_sin_match():
    caps = family_capabilities("at_zone(Zone)", CONTRACT_REGISTRY)
    # Ninguna acción declara guarantees_on_success at_zone.
    assert caps == []


def test_goal_identity_key_incluye_binding():
    # Dos bindings de la misma familia → claves distintas (coexisten como nodos).
    assert goal_identity_key("achieve_has_item", ["bread", 1]) == "achieve_has_item__bread_1"
    assert goal_identity_key("achieve_has_item", ["wheat", 2]) == "achieve_has_item__wheat_2"
    assert (goal_identity_key("achieve_has_item", ["bread", 1])
            != goal_identity_key("achieve_has_item", ["wheat", 2]))
    # Sin binding → el sig tal cual (builtins/sub-goals, no-regresión).
    assert goal_identity_key("achieve_explore_zone", []) == "achieve_explore_zone"
    assert goal_identity_key("move_to_and_pickup", None) == "move_to_and_pickup"
