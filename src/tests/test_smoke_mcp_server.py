"""Smoke tests — tools/mcp_world_server.py (Fase 6, MCP opción A).

Se testea la LÓGICA DE NEGOCIO (funciones puras), no el transporte MCP, usando
fixtures: el catálogo real del proyecto y sesiones/memoria sintéticas en tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# tools/ no es un paquete; añadirlo al path (igual que el smoke de analyze_sessions).
_TOOLS = Path(__file__).resolve().parent.parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import mcp_world_server as W  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_session(root: Path, name: str = "sess1") -> Path:
    sd = root / name
    sd.mkdir(parents=True)
    events = [
        {"ev": "session_start", "model": "qwen3:8b"},
        {"ev": "npc_profile", "npc": "npc_001", "goals_nl": ["Fabrica 1 @bread"]},
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "current_position", "args": [1, 2]},
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "current_position", "args": [9, -6]},
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "zone_center", "args": ["farmland", 5, 5]},
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "at_zone", "args": ["bakery"]},
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "has_item", "args": ["wheat", 3]},
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "has_item", "args": ["bread", 1], "source": "craft_output"},
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "has_item", "args": ["wheat", -2], "source": "craft_input"},
        {"ev": "belief_updated", "npc": "npc_002", "predicate": "has_item", "args": ["wood", 5]},
        {"ev": "llm_call", "step": "parse_goals", "latency_s": 2.0, "structured": "ParseGoalsResponse"},
        {"ev": "action_result", "action": "MoveTo", "status": "Success", "latency_s": 1.0},
        {"ev": "goal_completed", "npc": "npc_001", "goal": "achieve_x", "goal_belief_met": True, "intent_match": True},
    ]
    (sd / "trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    (sd / "prompts.jsonl").write_text('{"step": "parse_goals"}\n', encoding="utf-8")
    (sd / "plan.asl").write_text("+!achieve_x : true <- .wait(1).\n", encoding="utf-8")
    metrics = {
        "session_id": name,
        "duration_s": 50.0,
        "npcs": {"npc_001": {"goals_started": 1, "goals_completed": 1}},
    }
    (sd / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return sd


def _make_memory(root: Path) -> Path:
    npc = root / "npc_001"
    (npc / "approved").mkdir(parents=True)
    (npc / "pending").mkdir(parents=True)
    (npc / "approved" / "achieve_bread.json").write_text(json.dumps({
        "goal_sig": "achieve_bread", "status": "approved",
        "uses_success": 3, "uses_error": 0, "success_rate": 1.0,
        "description": "hace pan", "guard": "true",
    }), encoding="utf-8")
    (npc / "pending" / "achieve_wheat.json").write_text(json.dumps({
        "goal_sig": "achieve_wheat", "status": "pending",
        "uses_success": 1, "uses_error": 1, "success_rate": 0.5,
    }), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Catálogo y contratos
# ---------------------------------------------------------------------------

def test_action_catalog_shape():
    cat = W.get_action_catalog()
    assert cat["count"] == len(cat["actions"]) > 0
    names = {a["name"] for a in cat["actions"]}
    assert {"MoveTo", "Craft", "PickUp"} <= names
    craft = next(a for a in cat["actions"] if a["name"] == "Craft")
    assert craft["synopsis"].startswith("Craft(")
    assert craft["contract"]["attempt_budget"] == 1
    assert any(b["functor"] == "at_zone" for b in craft["contract"]["requires"])


def test_capability_contracts_real_and_missing(tmp_path):
    real = W.get_capability_contracts()
    assert "contracts" in real
    missing = W.get_capability_contracts(tmp_path / "nope.json")
    assert missing["contracts"] == {} and "_note" in missing


# ---------------------------------------------------------------------------
# Beliefs
# ---------------------------------------------------------------------------

def test_get_beliefs_reconstruction(tmp_path):
    _make_session(tmp_path)
    out = W.get_beliefs("npc_001", session="sess1", sessions_root=tmp_path)
    b = out["beliefs"]
    assert b["current_position"] == [9, -6]              # último valor
    assert b["zone_center"]["farmland"] == [5, 5]
    assert b["at_zone"] == ["bakery"]
    assert b["has_item"]["wheat"] == 1                   # 3 absoluto, -2 delta craft_input
    assert b["has_item"]["bread"] == 1                   # delta craft_output
    assert "wood" not in b["has_item"]                   # era de npc_002


def test_get_beliefs_latest_session(tmp_path):
    _make_session(tmp_path, "a")
    _make_session(tmp_path, "b")
    out = W.get_beliefs("npc_001", sessions_root=tmp_path)
    assert out["session"] in ("a", "b")


# ---------------------------------------------------------------------------
# Plan memory
# ---------------------------------------------------------------------------

def test_get_plan_memory(tmp_path):
    _make_memory(tmp_path)
    out = W.get_plan_memory("npc_001", memory_root=tmp_path)
    assert out["count"] == 2
    sigs = {p["goal_sig"] for p in out["plans"]}
    assert sigs == {"achieve_bread", "achieve_wheat"}
    approved = next(p for p in out["plans"] if p["goal_sig"] == "achieve_bread")
    assert approved["status"] == "approved" and approved["uses_success"] == 3


def test_get_plan_memory_runs_layout(tmp_path):
    run = tmp_path / "runs" / "20260101_000000"
    _make_memory(run)
    out = W.get_plan_memory("npc_001", memory_root=tmp_path)
    assert out["count"] == 2


def test_get_plan_memory_empty(tmp_path):
    out = W.get_plan_memory("ghost", memory_root=tmp_path)
    assert out["count"] == 0 and out["plans"] == []


# ---------------------------------------------------------------------------
# Validador ASL
# ---------------------------------------------------------------------------

def test_validate_asl_ok():
    out = W.validate_asl("+!achieve_get_wheat : has_item(wheat, N) <- .moveto(1, 2).")
    assert out["ok"] is True and out["errors"] == []


def test_validate_asl_bad_action():
    out = W.validate_asl("+!achieve_x : true <- .teleport(1, 2).")
    assert out["ok"] is False
    assert any("PRIMITIVE_ACTIONS" in e for e in out["errors"])


def test_validate_asl_warning_is_not_error():
    # +belief en el body → WARNING, no error duro.
    out = W.validate_asl("+!achieve_x : true <- +has_item(wheat, 1).")
    assert out["ok"] is True
    assert out["warnings"] and all(w.startswith("WARNING") for w in out["warnings"])


# ---------------------------------------------------------------------------
# Observabilidad de sesiones
# ---------------------------------------------------------------------------

def test_list_sessions(tmp_path):
    _make_session(tmp_path, "a")
    _make_session(tmp_path, "b")
    out = W.list_sessions(tmp_path)
    assert out["count"] == 2
    assert {s["session"] for s in out["sessions"]} == {"a", "b"}


def test_list_sessions_missing_root(tmp_path):
    out = W.list_sessions(tmp_path / "nope")
    assert out["count"] == 0


def test_get_session_summary(tmp_path):
    _make_session(tmp_path)
    out = W.get_session_summary("sess1", sessions_root=tmp_path)
    assert out["session_id"] == "sess1"
    assert out["goals_completed"] == 1


def test_get_trace_events_filter(tmp_path):
    _make_session(tmp_path)
    out = W.get_trace_events("sess1", ev_filter="belief_updated", sessions_root=tmp_path)
    assert out["total"] == out["returned"] > 0
    assert all(e["ev"] == "belief_updated" for e in out["events"])


def test_get_trace_events_npc_filter_keeps_globals(tmp_path):
    _make_session(tmp_path)
    out = W.get_trace_events("sess1", npc="npc_001", sessions_root=tmp_path)
    npcs = {e.get("npc") for e in out["events"]}
    assert "npc_002" not in npcs
    assert any("npc" not in e for e in out["events"])  # eventos globales conservados


def test_get_trace_events_limit(tmp_path):
    _make_session(tmp_path)
    out = W.get_trace_events("sess1", limit=2, sessions_root=tmp_path)
    assert out["returned"] == 2 and out["total"] > 2


def test_resolve_session_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        W.get_session_summary("does_not_exist", sessions_root=tmp_path)


# ---------------------------------------------------------------------------
# Artefactos
# ---------------------------------------------------------------------------

def test_read_session_artifact(tmp_path):
    _make_session(tmp_path)
    out = W.read_session_artifact("sess1", "plan.asl", sessions_root=tmp_path)
    assert "achieve_x" in out["content"]


def test_read_session_artifact_rejects_bad_suffix(tmp_path):
    _make_session(tmp_path)
    (tmp_path / "sess1" / "secret.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        W.read_session_artifact("sess1", "secret.txt", sessions_root=tmp_path)


def test_read_session_artifact_rejects_traversal(tmp_path):
    _make_session(tmp_path)
    with pytest.raises(ValueError):
        W.read_session_artifact("sess1", "../../etc/passwd.json", sessions_root=tmp_path)
