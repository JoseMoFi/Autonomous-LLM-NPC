"""Smoke tests — UnityEventBehaviour dispatch (async, sin SPADE real).

Prueba el comportamiento de despacho de mensajes de Unity al estado del agente,
mockeando el agente como un simple objeto con los atributos necesarios.
No requiere XMPP ni red real.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import networkx as nx
import pytest

from npc.beliefs import BeliefStore
from npc.behaviours.unity_events import UnityEventBehaviour


# ===========================================================================
# Mock agent helper
# ===========================================================================

class MockAgent:
    """Mínimo estado de NPCAgent para probar UnityEventBehaviour."""

    def __init__(self):
        self.npc_id = "npc_test"
        self.beliefs = BeliefStore()
        self.profile = None
        self.paused = False
        self.inbox: asyncio.Queue[dict] = asyncio.Queue()
        self.goals: list[Any] = []
        self.intention = None
        self.plan_graph = nx.DiGraph()
        self.action_results: dict[str, dict] = {}
        self.sent_commands: dict[str, dict] = {}
        self.completed_goal_count = 0
        self.phase = "idle"
        self._sent: list[dict] = []
        self._bootstrap_task = None
        self._profile_fingerprint = None
        self._bootstrapped_profile_fingerprint = None

    async def send_to_unity(self, msg: dict) -> None:
        self._sent.append(msg)

    async def push_status(self) -> None:
        await self.send_to_unity({"type": "GoalsUpdate", "npcId": self.npc_id})

    async def run_llm_task(self, task: dict) -> Any:
        """Stub: devuelve lista vacía para parse_goals."""
        return []


def _make_behaviour() -> tuple[UnityEventBehaviour, MockAgent]:
    agent = MockAgent()
    b = UnityEventBehaviour()
    b.agent = agent  # type: ignore[assignment]
    return b, agent


# ===========================================================================
# NPCProfile dispatch
# ===========================================================================

_PROFILE_DICT = {
    "role": "baker",
    "display_name": "Lars",
    "home_pos": {"x": 5, "y": 10},
    "background": "A master baker.",
    "personality": "Friendly",
    "llm_context": "",
    "map_bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
    "zones": [
        {"tag": "bakery", "center": {"x": 50, "y": 50},
         "bounds": {"x_min": 40, "x_max": 60, "y_min": 40, "y_max": 60}},
    ],
    "delivery_points": {"tavern": {"x": 80, "y": 20}},
    "recipes": [
        {
            "recipeId": "bread",
            "zone": "bakery",
            "inputs": [{"itemId": "wheat", "qty": 2}],
            "outputs": [{"itemId": "bread", "qty": 1}],
        }
    ],
    "goals_nl": ["Deliver bread to the tavern"],
    "item_spawns": [{"itemId": "wheat", "zones": ["farmland"], "targetCount": 5}],
}


@pytest.mark.asyncio
async def test_npc_profile_sets_agent_profile() -> None:
    b, agent = _make_behaviour()
    msg = {"type": "NPCProfile", "profile": _PROFILE_DICT.copy()}
    with patch.object(agent, "run_llm_task", new_callable=AsyncMock, return_value=[]):
        await b._dispatch(msg)
    assert agent.profile is not None
    assert agent.profile.role == "baker"


@pytest.mark.asyncio
async def test_npc_profile_populates_zone_beliefs() -> None:
    b, agent = _make_behaviour()
    msg = {"type": "NPCProfile", "profile": _PROFILE_DICT.copy()}
    with patch.object(agent, "run_llm_task", new_callable=AsyncMock, return_value=[]):
        await b._dispatch(msg)
    snap = agent.beliefs.snapshot()
    assert any("bakery" in str(t) for t in snap.get("zone_center", []))


@pytest.mark.asyncio
async def test_npc_profile_populates_delivery_point_beliefs() -> None:
    b, agent = _make_behaviour()
    msg = {"type": "NPCProfile", "profile": _PROFILE_DICT.copy()}
    with patch.object(agent, "run_llm_task", new_callable=AsyncMock, return_value=[]):
        await b._dispatch(msg)
    snap = agent.beliefs.snapshot()
    assert any("tavern" in str(t) for t in snap.get("delivery_point", []))


@pytest.mark.asyncio
async def test_npc_profile_does_not_send_unknown_ready_message() -> None:
    b, agent = _make_behaviour()
    msg = {"type": "NPCProfile", "profile": _PROFILE_DICT.copy()}
    with patch.object(agent, "run_llm_task", new_callable=AsyncMock, return_value=[]):
        await b._dispatch(msg)
    assert not any(m.get("type") == "NPCReady" for m in agent._sent)


@pytest.mark.asyncio
async def test_npc_profile_malformed_does_not_crash() -> None:
    """Profile malformado → error silencioso, profile no se establece."""
    b, agent = _make_behaviour()
    msg = {"type": "NPCProfile", "profile": {"role": "MISSING_REQUIRED_FIELDS"}}
    with patch.object(agent, "run_llm_task", new_callable=AsyncMock, return_value=[]):
        await b._dispatch(msg)
    # El agente sigue funcionando; profile puede estar None o tener error manejado
    # No debe propagarse excepción


@pytest.mark.asyncio
async def test_duplicate_npc_profile_does_not_relaunch_parse_goals() -> None:
    b, agent = _make_behaviour()
    msg = {"type": "NPCProfile", "profile": _PROFILE_DICT.copy()}

    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def _fake_run_llm_task(task: dict) -> Any:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return []

    with patch.object(agent, "run_llm_task", new=_fake_run_llm_task):
        await b._dispatch(msg)
        await asyncio.wait_for(started.wait(), timeout=1)
        await b._dispatch(msg)
        await asyncio.sleep(0)
        assert calls == 1

        release.set()
        if agent._bootstrap_task is not None:
            await agent._bootstrap_task


# ===========================================================================
# 0.B3 — source_index: emparejamiento robusto de expected_condition
# ===========================================================================

@pytest.mark.asyncio
async def test_bootstrap_goals_pairs_by_source_index_when_reordered() -> None:
    """El LLM devuelve los goals en orden inverso pero con source_index: la
    expected_condition debe emparejarse por source_index, no por posición."""
    from protocol.messages import NPCProfilePayload
    b, agent = _make_behaviour()
    profile_dict = _PROFILE_DICT.copy()
    profile_dict["goals_nl"] = ["Collect wheat", "Deliver bread"]
    profile_dict["goal_conditions"] = ["has_item(wheat, 1)", "has_item(bread, 0)"]
    agent.profile = NPCProfilePayload.from_dict(profile_dict)

    # Respuesta desordenada: primero el goal del objetivo índice 1, luego el 0.
    reordered = [
        {"sig": "achieve_deliver_bread", "priority": 1.0,
         "success_condition": "has_item(bread, 0)", "source_index": 1},
        {"sig": "achieve_collect_wheat", "priority": 0.9,
         "success_condition": "has_item(wheat, 1)", "source_index": 0},
    ]
    with patch.object(agent, "run_llm_task", new_callable=AsyncMock, return_value=reordered):
        await b._bootstrap_goals("fp")

    by_sig = {g.sig: g.expected_condition for g in agent.goals}
    assert by_sig["achieve_collect_wheat"] == "has_item(wheat, 1)"
    assert by_sig["achieve_deliver_bread"] == "has_item(bread, 0)"


@pytest.mark.asyncio
async def test_bootstrap_goals_falls_back_to_position_without_source_index() -> None:
    """Sin source_index (modelo antiguo) se cae al emparejamiento posicional."""
    from protocol.messages import NPCProfilePayload
    b, agent = _make_behaviour()
    profile_dict = _PROFILE_DICT.copy()
    profile_dict["goals_nl"] = ["Collect wheat", "Deliver bread"]
    profile_dict["goal_conditions"] = ["has_item(wheat, 1)", "has_item(bread, 0)"]
    agent.profile = NPCProfilePayload.from_dict(profile_dict)

    legacy = [
        {"sig": "achieve_collect_wheat", "priority": 1.0, "success_condition": "has_item(wheat, 1)"},
        {"sig": "achieve_deliver_bread", "priority": 0.9, "success_condition": "has_item(bread, 0)"},
    ]
    with patch.object(agent, "run_llm_task", new_callable=AsyncMock, return_value=legacy):
        await b._bootstrap_goals("fp")

    by_sig = {g.sig: g.expected_condition for g in agent.goals}
    assert by_sig["achieve_collect_wheat"] == "has_item(wheat, 1)"
    assert by_sig["achieve_deliver_bread"] == "has_item(bread, 0)"


# ===========================================================================
# ZoneDiscovery dispatch
# ===========================================================================

@pytest.mark.asyncio
async def test_zone_discovery_updates_beliefs() -> None:
    b, agent = _make_behaviour()
    msg = {
        "type": "ZoneDiscovery",
        "zones": [
            {"tag": "farmland", "center": {"x": 30.0, "y": 40.0}},
            {"tag": "mill",     "center": {"x": 60.0, "y": 70.0}},
        ],
    }
    await b._dispatch(msg)
    snap = agent.beliefs.snapshot()
    zone_tags = {t[0] for t in snap.get("zone_center", [])}
    assert "farmland" in zone_tags
    assert "mill" in zone_tags


@pytest.mark.asyncio
async def test_zone_discovery_empty_zones_no_crash() -> None:
    b, agent = _make_behaviour()
    msg = {"type": "ZoneDiscovery", "zones": []}
    await b._dispatch(msg)  # must not raise


# ===========================================================================
# InventoryUpdate dispatch
# ===========================================================================

@pytest.mark.asyncio
async def test_inventory_update_changes_beliefs() -> None:
    b, agent = _make_behaviour()
    msg = {
        "type": "InventoryUpdate",
        "items": [
            {"itemId": "wheat", "qty": 5},
            {"itemId": "bread", "qty": 2},
        ],
    }
    await b._dispatch(msg)
    snap = agent.beliefs.snapshot()
    inv = {t[0]: t[1] for t in snap.get("has_item", [])}
    assert inv.get("wheat") == 5
    assert inv.get("bread") == 2


# ===========================================================================
# ActionResult dispatch
# ===========================================================================

@pytest.mark.asyncio
async def test_action_result_resolves_future() -> None:
    b, agent = _make_behaviour()
    cmd_id = "cmd-123"

    msg = {
        "type": "ActionResult",
        "commandId": cmd_id,
        "status": "Success",
        "npcId": "npc_test",
    }
    await b._dispatch(msg)
    assert agent.action_results[cmd_id]["status"] == "Success"


@pytest.mark.asyncio
async def test_action_result_unknown_cmd_does_not_crash() -> None:
    b, agent = _make_behaviour()
    msg = {
        "type": "ActionResult",
        "commandId": "nonexistent-cmd",
        "status": "Failure",
    }
    await b._dispatch(msg)  # debe ignorarse sin error


# ===========================================================================
# ActionResult — current_position
# ===========================================================================

@pytest.mark.asyncio
async def test_moveto_success_with_payload_updates_current_position() -> None:
    """MoveTo Success con payload.current_position actualiza el belief."""
    b, agent = _make_behaviour()
    msg = {
        "type": "ActionResult",
        "commandId": "cmd-moveto-1",
        "actionType": "MoveTo",
        "status": "Success",
        "endedTick": 10,
        "payload": {"current_position": {"x": 42, "y": 17}},
    }
    await b._dispatch(msg)
    results = agent.beliefs.query("current_position")
    assert len(results) == 1
    assert results[0] == (42, 17)


@pytest.mark.asyncio
async def test_moveto_failure_does_not_update_current_position() -> None:
    """MoveTo Failure NO modifica current_position."""
    b, agent = _make_behaviour()
    # Establecer una posicion inicial
    agent.beliefs.apply_current_position(5, 5)
    msg = {
        "type": "ActionResult",
        "commandId": "cmd-moveto-fail",
        "actionType": "MoveTo",
        "status": "Failure",
        "endedTick": 11,
        "payload": {"current_position": {"x": 99, "y": 99}},
    }
    await b._dispatch(msg)
    results = agent.beliefs.query("current_position")
    # La posicion no debe haber cambiado
    assert results == [(5, 5)]


@pytest.mark.asyncio
async def test_explorearea_success_with_payload_updates_current_position() -> None:
    """ExploreArea Success con payload.current_position actualiza el belief."""
    b, agent = _make_behaviour()
    msg = {
        "type": "ActionResult",
        "commandId": "cmd-explore-1",
        "actionType": "ExploreArea",
        "status": "Success",
        "endedTick": 20,
        "payload": {"current_position": {"x": 55, "y": 33}},
    }
    await b._dispatch(msg)


# ===========================================================================
# 0.A1 — ZoneEntry routing + at_zone (mensaje explícito y fallback por posición)
# ===========================================================================

@pytest.mark.asyncio
async def test_zone_entry_sets_at_zone() -> None:
    """ZoneEntry entered=True assertea at_zone(zone_tag)."""
    b, agent = _make_behaviour()
    msg = {"type": "ZoneEntry", "npc_id": "npc_test", "zone_tag": "Bakeri", "entered": True}
    await b._dispatch(msg)
    assert agent.beliefs.has("at_zone", "bakeri")


@pytest.mark.asyncio
async def test_zone_entry_exit_removes_at_zone() -> None:
    """ZoneEntry entered=False retracta at_zone(zone_tag)."""
    b, agent = _make_behaviour()
    agent.beliefs.apply_at_zone("bakeri")
    msg = {"type": "ZoneEntry", "npc_id": "npc_test", "zone_tag": "bakeri", "entered": False}
    await b._dispatch(msg)
    assert not agent.beliefs.has("at_zone", "bakeri")


@pytest.mark.asyncio
async def test_at_zone_derived_from_position_when_inside_bounds() -> None:
    """Fallback C1: si Unity no emite ZoneEntry, at_zone se deriva al entrar en
    los bounds de la zona tras un MoveTo Success."""
    b, agent = _make_behaviour()
    # Profile carga los bounds de la zona bakery (40..60, 40..60)
    msg_profile = {"type": "NPCProfile", "profile": _PROFILE_DICT.copy()}
    with patch.object(agent, "run_llm_task", new_callable=AsyncMock, return_value=[]):
        await b._dispatch(msg_profile)

    assert not agent.beliefs.has("at_zone", "bakery")
    # MoveTo al centro de la zona → posición dentro de bounds → at_zone derivado
    msg_move = {
        "type": "ActionResult",
        "commandId": "cmd-move-bakery",
        "actionType": "MoveTo",
        "status": "Success",
        "endedTick": 5,
        "payload": {"current_position": {"x": 50, "y": 50}},
    }
    await b._dispatch(msg_move)
    assert agent.beliefs.has("at_zone", "bakery")


@pytest.mark.asyncio
async def test_at_zone_not_derived_outside_bounds() -> None:
    """Posición fuera de cualquier zona no assertea at_zone."""
    b, agent = _make_behaviour()
    msg_profile = {"type": "NPCProfile", "profile": _PROFILE_DICT.copy()}
    with patch.object(agent, "run_llm_task", new_callable=AsyncMock, return_value=[]):
        await b._dispatch(msg_profile)
    msg_move = {
        "type": "ActionResult",
        "commandId": "cmd-move-out",
        "actionType": "MoveTo",
        "status": "Success",
        "endedTick": 6,
        "payload": {"current_position": {"x": 5, "y": 5}},
    }
    await b._dispatch(msg_move)
    assert not agent.beliefs.snapshot().get("at_zone")


# ===========================================================================
# 0.A2 — Inventario tras Craft Success (provisional desde la receta)
# ===========================================================================

@pytest.mark.asyncio
async def test_craft_success_updates_inventory_from_recipe() -> None:
    """Craft Success produce el output y consume los inputs según la receta."""
    b, agent = _make_behaviour()
    # Receta bread: 2 wheat -> 1 bread, en zona bakery.
    agent.beliefs.apply_recipe(
        "bread", "bakery",
        [{"itemId": "wheat", "qty": 2}],
        [{"itemId": "bread", "qty": 1}],
    )
    agent.beliefs.apply_has_item("wheat", 2)
    cmd_id = "cmd-craft-1"
    agent.sent_commands[cmd_id] = {
        "actionType": "Craft",
        "args": {"itemId": "bread", "targetId": "bread", "qty": 1},
    }
    msg = {
        "type": "ActionResult",
        "commandId": cmd_id,
        "actionType": "Craft",
        "status": "Success",
        "endedTick": 30,
        "payload": {},
    }
    await b._dispatch(msg)
    inv = {t[0]: t[1] for t in agent.beliefs.snapshot().get("has_item", [])}
    assert inv.get("bread") == 1
    assert inv.get("wheat", 0) == 0  # 2 consumidos
    # El comando se libera tras el resultado terminal
    assert cmd_id not in agent.sent_commands


@pytest.mark.asyncio
async def test_craft_success_uses_unity_consumed_and_produced() -> None:
    """Fase 17l: manda el payload de Unity. Craft(wheat, Bread_recipe, qty=2) es UNA
    hornada (1 pan, 2 trigos); antes se apuntaban 2 panes y 4 trigos."""
    b, agent = _make_behaviour()
    agent.beliefs.apply_recipe(
        "bread", "bakery",
        [{"itemId": "wheat", "qty": 2}],
        [{"itemId": "bread", "qty": 1}],
    )
    agent.beliefs.apply_has_item("wheat", 4)
    cmd_id = "cmd-craft-3"
    agent.sent_commands[cmd_id] = {
        "actionType": "Craft",
        "args": {"itemId": "wheat", "targetId": "bread", "qty": 2},
    }
    await b._dispatch({
        "type": "ActionResult",
        "commandId": cmd_id,
        "actionType": "Craft",
        "status": "Success",
        "endedTick": 32,
        "payload": {
            "recipeId": "bread",
            "consumed": [{"itemId": "wheat", "qty": 1}, {"itemId": "wheat", "qty": 1}],
            "produced": [{"itemId": "bread", "qty": 1}],
        },
    })
    inv = {t[0]: t[1] for t in agent.beliefs.snapshot().get("has_item", [])}
    assert inv.get("bread") == 1
    assert inv.get("wheat") == 2


@pytest.mark.asyncio
async def test_craft_qty_counts_ingredient_units_not_batches() -> None:
    """Sin listas en el payload, qty=2 con una receta de 2 trigos sigue siendo 1 hornada."""
    b, agent = _make_behaviour()
    agent.beliefs.apply_recipe(
        "bread", "bakery",
        [{"itemId": "wheat", "qty": 2}],
        [{"itemId": "bread", "qty": 1}],
    )
    agent.beliefs.apply_has_item("wheat", 4)
    cmd_id = "cmd-craft-4"
    agent.sent_commands[cmd_id] = {
        "actionType": "Craft",
        "args": {"itemId": "wheat", "targetId": "bread", "qty": 2},
    }
    await b._dispatch({
        "type": "ActionResult",
        "commandId": cmd_id,
        "actionType": "Craft",
        "status": "Success",
        "endedTick": 33,
        "payload": {},
    })
    inv = {t[0]: t[1] for t in agent.beliefs.snapshot().get("has_item", [])}
    assert inv.get("bread") == 1
    assert inv.get("wheat") == 2


@pytest.mark.asyncio
async def test_craft_success_resolves_recipe_from_payload_item() -> None:
    """Si no hay sent_commands, la receta se resuelve por el itemId del payload."""
    b, agent = _make_behaviour()
    agent.beliefs.apply_recipe(
        "bread", "bakery",
        [{"itemId": "wheat", "qty": 2}],
        [{"itemId": "bread", "qty": 1}],
    )
    agent.beliefs.apply_has_item("wheat", 4)
    msg = {
        "type": "ActionResult",
        "commandId": "cmd-craft-2",
        "actionType": "Craft",
        "status": "Success",
        "endedTick": 31,
        "payload": {"itemId": "bread", "qty": 1},
    }
    await b._dispatch(msg)
    inv = {t[0]: t[1] for t in agent.beliefs.snapshot().get("has_item", [])}
    assert inv.get("bread") == 1
    assert inv.get("wheat") == 2  # 4 - 2
