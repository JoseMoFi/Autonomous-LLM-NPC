"""Smoke tests — BeliefStore.

Verifica que BeliefStore actualiza y consulta predicados correctamente
sin depender de SPADE, Ollama ni pyjabber.
"""

from __future__ import annotations

import pytest
from npc.beliefs import BeliefStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bs() -> BeliefStore:
    return BeliefStore()


# ---------------------------------------------------------------------------
# zone_center
# ---------------------------------------------------------------------------

def test_apply_zone_discovery_stored(bs: BeliefStore) -> None:
    bs.apply_zone_discovery("farmland", 10, 20)
    results = bs.query("zone_center", "farmland")
    assert len(results) == 1
    assert results[0] == ("farmland", 10, 20)


def test_apply_zone_discovery_normalise_tag(bs: BeliefStore) -> None:
    """Tag se normaliza a minusculas."""
    bs.apply_zone_discovery("FARMLAND", 10, 20)
    assert bs.query("zone_center", "farmland") != []
    assert bs.query("zone_center", "FARMLAND") == []


def test_apply_zone_discovery_wildcard(bs: BeliefStore) -> None:
    bs.apply_zone_discovery("farmland", 10, 20)
    bs.apply_zone_discovery("mill", 30, 40)
    results = bs.query("zone_center", None)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# has_item (inventario)
# ---------------------------------------------------------------------------

def test_apply_inventory_update_stored(bs: BeliefStore) -> None:
    bs.apply_inventory_update([{"itemId": "wheat", "qty": 3}])
    results = bs.query("has_item", "wheat")
    assert results == [("wheat", 3)]


def test_apply_inventory_update_replaces(bs: BeliefStore) -> None:
    """Segunda update reemplaza la primera completamente."""
    bs.apply_inventory_update([{"itemId": "wheat", "qty": 3}])
    bs.apply_inventory_update([{"itemId": "bread", "qty": 1}])
    assert bs.query("has_item", "wheat") == []
    assert bs.query("has_item", "bread") == [("bread", 1)]


def test_apply_inventory_multiple_items(bs: BeliefStore) -> None:
    items = [{"itemId": "wheat", "qty": 3}, {"itemId": "flour", "qty": 2}]
    bs.apply_inventory_update(items)
    assert len(bs.query("has_item", None)) == 2


# ---------------------------------------------------------------------------
# delivery_point
# ---------------------------------------------------------------------------

def test_apply_delivery_point(bs: BeliefStore) -> None:
    bs.apply_delivery_point("tavern", 50, 60)
    results = bs.query("delivery_point", "tavern")
    assert results == [("tavern", 50, 60)]


# ---------------------------------------------------------------------------
# recipe
# ---------------------------------------------------------------------------

def test_apply_recipe(bs: BeliefStore) -> None:
    bs.apply_recipe(
        "bread", "bakery",
        [{"itemId": "wheat", "qty": 2}],
        [{"itemId": "bread", "qty": 1}],
    )
    results = bs.query("recipe", "bread")
    assert len(results) >= 1
    assert results[0][0] == "bread"
    assert results[0][1] == "bakery"


def test_apply_recipe_stores_outputs(bs: BeliefStore) -> None:
    """apply_recipe almacena predicados recipe_output con los items producidos."""
    bs.apply_recipe(
        "bread", "bakery",
        [{"itemId": "wheat", "qty": 2}],
        [{"itemId": "bread", "qty": 1}],
    )
    results = bs.query("recipe_output", "bread")
    assert len(results) == 1, f"Expected 1 recipe_output for bread, got: {results}"
    assert results[0][0] == "bread"
    assert results[0][1] == "bakery"
    assert results[0][2] == "bread"
    assert results[0][3] == 1


# ---------------------------------------------------------------------------
# item_spawn
# ---------------------------------------------------------------------------

def test_apply_item_spawn(bs: BeliefStore) -> None:
    bs.apply_item_spawn("wheat", "farmland")
    results = bs.query("item_spawn", "wheat", "farmland")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# current_position  (Iter 2)
# ---------------------------------------------------------------------------

def test_apply_current_position_stored(bs: BeliefStore) -> None:
    bs.apply_current_position(10, 20)
    results = bs.query("current_position")
    assert len(results) == 1
    assert results[0] == (10, 20)


def test_apply_current_position_upsert_no_duplicates(bs: BeliefStore) -> None:
    """Segunda llamada reemplaza la primera; nunca hay dos entradas."""
    bs.apply_current_position(10, 20)
    bs.apply_current_position(30, 40)
    results = bs.query("current_position")
    assert len(results) == 1
    assert results[0] == (30, 40)


def test_apply_current_position_coerces_float_to_int(bs: BeliefStore) -> None:
    """Inputs float se convierten a int para evitar mezcla de tipos."""
    bs.apply_current_position(5.9, 7.1)  # type: ignore[arg-type]
    results = bs.query("current_position")
    assert len(results) == 1
    x, y = results[0]
    assert isinstance(x, int)
    assert isinstance(y, int)
    assert x == 5
    assert y == 7


def test_apply_current_position_no_float_in_snapshot(bs: BeliefStore) -> None:
    """El snapshot no debe contener floats en current_position."""
    bs.apply_current_position(3, 8)
    snap = bs.snapshot()
    for tup in snap.get("current_position", []):
        for v in tup:
            assert isinstance(v, int), f"Valor no-int en current_position: {v!r}"


def test_apply_current_position_appears_in_recent_additions(bs: BeliefStore) -> None:
    bs.apply_current_position(1, 2)
    recent = bs.recent_additions()
    assert "current_position" in recent


def test_apply_current_position_zero_coords(bs: BeliefStore) -> None:
    """Coordenadas (0, 0) son validas."""
    bs.apply_current_position(0, 0)
    results = bs.query("current_position")
    assert results == [(0, 0)]


# ---------------------------------------------------------------------------
# search_range — eliminado (Iter 3)
# ---------------------------------------------------------------------------

def test_search_range_not_in_snapshot_after_profile_beliefs(bs: BeliefStore) -> None:
    """Tras poblar beliefs de perfil tipicos, search_range no debe aparecer."""
    bs.apply_zone_discovery("farmland", 10, 20)
    bs.apply_inventory_update([{"itemId": "wheat", "qty": 1}])
    bs.apply_current_position(5, 5)
    snap = bs.snapshot()
    assert "search_range" not in snap


def test_apply_search_range_method_no_longer_exists() -> None:
    """apply_search_range fue eliminado en Iter 3."""
    bs = BeliefStore()
    assert not hasattr(bs, "apply_search_range"), (
        "apply_search_range todavia existe en BeliefStore; debe haberse eliminado en Iter 3"
    )


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

def test_snapshot_empty(bs: BeliefStore) -> None:
    snap = bs.snapshot()
    assert isinstance(snap, dict)
    assert snap == {}


def test_snapshot_populated(bs: BeliefStore) -> None:
    bs.apply_zone_discovery("farmland", 10, 20)
    bs.apply_inventory_update([{"itemId": "wheat", "qty": 1}])
    snap = bs.snapshot()
    assert "zone_center" in snap
    assert "has_item" in snap
