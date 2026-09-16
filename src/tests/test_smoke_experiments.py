"""Smoke tests — batería experimental (Fase 13).

Cubren:
  - Manifiestos (tools/experiments/E1..E6.json): campos obligatorios,
    n_opt presente, goals con forma {npc_id: [{nl,condition}, ...]}.
  - configs.json: traducción etiqueta -> flags (las 5 configs del protocolo).
  - analyze_experiments.py: cálculo de η_exec/detour/redundancias sobre una
    traza sintética con resultado conocido; agregado sobre directorio vacío
    sin excepción; sesiones sin experiment_id se ignoran.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parent.parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import analyze_experiments as AE  # noqa: E402

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "tools" / "experiments"
_EXPECTED_IDS = ["E1", "E2", "E3", "E4", "E5", "E6"]


# ===========================================================================
# Manifiestos
# ===========================================================================

@pytest.mark.parametrize("exp_id", _EXPECTED_IDS)
def test_manifest_has_required_fields(exp_id: str):
    path = _EXPERIMENTS_DIR / f"{exp_id}.json"
    assert path.exists(), f"falta el manifiesto {path}"
    data = json.loads(path.read_text(encoding="utf-8"))

    for field in ("id", "label", "npcs", "goals", "n_opt", "configs", "runs", "timeout_s"):
        assert field in data, f"{exp_id}: falta el campo '{field}'"
    assert data["id"] == exp_id
    assert isinstance(data["npcs"], int) and data["npcs"] >= 1
    assert isinstance(data["n_opt"], (int, float)) and data["n_opt"] > 0
    assert isinstance(data["configs"], list) and data["configs"]
    # runs debe tener una entrada por cada config declarada
    for cfg in data["configs"]:
        assert cfg in data["runs"], f"{exp_id}: config '{cfg}' sin N en 'runs'"
        assert data["runs"][cfg] >= 1


@pytest.mark.parametrize("exp_id", _EXPECTED_IDS)
def test_manifest_goals_shape(exp_id: str):
    path = _EXPERIMENTS_DIR / f"{exp_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    goals = data["goals"]
    assert isinstance(goals, dict) and goals, f"{exp_id}: 'goals' vacío o no es un dict"
    for npc_id, goal_list in goals.items():
        assert isinstance(goal_list, list) and goal_list
        for g in goal_list:
            assert "nl" in g and "condition" in g


def test_e5_e6_are_two_npcs_with_single_goal_owner():
    for exp_id in ("E5", "E6"):
        data = json.loads((_EXPERIMENTS_DIR / f"{exp_id}.json").read_text(encoding="utf-8"))
        assert data["npcs"] == 2
        assert len(data["goals"]) == 1, f"{exp_id}: el goal debe ir a UN solo NPC"


def test_configs_json_covers_protocol_labels():
    data = json.loads((_EXPERIMENTS_DIR / "configs.json").read_text(encoding="utf-8"))
    for label in ("M0", "M1", "M2", "C1", "X1"):
        assert label in data, f"falta la config '{label}'"
        assert "env" in data[label]

    # C1 y X1 activan la familia determinista; M0 no activa nada.
    assert data["C1"]["env"].get("NPC_CANONICAL_FAMILY") == "1"
    assert data["X1"]["env"].get("NPC_COORDINATION") == "1"
    assert data["M0"]["env"].get("NPC_CANONICAL_REUSE") == "0"


# ===========================================================================
# analyze_experiments.py — cálculo sobre traza sintética
# ===========================================================================

def _write_session(tmp_path: Path, name: str, events: list[dict], metrics: dict) -> Path:
    sd = tmp_path / name
    sd.mkdir()
    (sd / "trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    (sd / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return sd


def _base_metrics(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "duration_s": 42.0,
        "npcs": {
            "npc_001": {
                "goals_started": 1, "goals_completed": 1, "goals_belief_met": 1,
                "plans_from_memory": 0, "plans_from_llm": 1,
                "llm_calls": 4, "llm_total_s": 8.0,
                "actions_sent": 6, "actions_ok": 6, "actions_failed": 0,
            }
        },
    }


def test_eta_exec_and_detour_computed_from_n_opt():
    events = [
        {"ev": "session_start", "experiment_id": "E1", "config_label": "M0", "n_opt": 3, "git_sha": "abc123"},
        {"ev": "action_sent", "npc": "npc_001", "action": "Search", "args": {"itemId": "wheat"}},
        {"ev": "action_sent", "npc": "npc_001", "action": "MoveTo", "args": {"x": 9, "y": -6}},
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "current_position", "args": [9, -6]},
        {"ev": "action_sent", "npc": "npc_001", "action": "PickUp", "args": {"itemId": "wheat"}},
        {"ev": "goal_completed", "npc": "npc_001", "goal": "achieve_collect_wheat", "goal_belief_met": True, "intent_match": True},
    ]

    metrics = _base_metrics("sessA")
    metrics["npcs"]["npc_001"]["actions_sent"] = 3  # n_act viene de metrics.json, no de la traza

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sd = _write_session(Path(td), "sessA", events, metrics)
        s = AE.summarize_experiment_session(sd)

    assert s is not None
    assert s["experiment_id"] == "E1"
    assert s["config_label"] == "M0"
    assert s["git_sha"] == "abc123"
    assert s["n_opt"] == 3
    assert s["n_act"] == 3  # actions_sent del metrics.json
    assert s["eta_exec"] == 1.0  # 3/3 -- optimo exacto
    assert s["detour"] == 0


def test_session_without_experiment_id_is_ignored():
    events = [{"ev": "session_start", "model": "qwen3:8b"}]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sd = _write_session(Path(td), "sessB", events, _base_metrics("sessB"))
        s = AE.summarize_experiment_session(sd)
    assert s is None


def test_redundant_moveto_detected():
    events = [
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "current_position", "args": [5, 5]},
        {"ev": "action_sent", "npc": "npc_001", "action": "MoveTo", "args": {"x": 5, "y": 5}},
    ]
    redundant = AE.count_redundant_actions(events)
    assert redundant["npc_001"]["redundant_moveto"] == 1


def test_redundant_search_detected_without_movement():
    events = [
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "at_zone", "args": ["farmland"]},
        {"ev": "action_sent", "npc": "npc_001", "action": "Search", "args": {"itemId": "wheat"}},
        {"ev": "action_sent", "npc": "npc_001", "action": "Search", "args": {"itemId": "wheat"}},
    ]
    redundant = AE.count_redundant_actions(events)
    assert redundant["npc_001"]["redundant_search"] == 1


def test_search_not_redundant_after_moving():
    events = [
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "at_zone", "args": ["farmland"]},
        {"ev": "action_sent", "npc": "npc_001", "action": "Search", "args": {"itemId": "wheat"}},
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "at_zone", "args": ["bakeri"]},
        {"ev": "action_sent", "npc": "npc_001", "action": "Search", "args": {"itemId": "wheat"}},
    ]
    redundant = AE.count_redundant_actions(events)
    # Un NPC sin ninguna accion redundante puede no tener entrada en el dict
    # (solo se crea al contar algo) -- .get(...) es la forma correcta de leerlo.
    assert redundant.get("npc_001", {}).get("redundant_search", 0) == 0


def test_redundant_explorearea_when_zone_already_known():
    events = [
        {"ev": "belief_updated", "npc": "npc_001", "predicate": "zone_center", "args": ["farmland", 9, -6]},
        {"ev": "action_sent", "npc": "npc_001", "action": "ExploreArea", "args": {"zoneTag": "farmland"}},
    ]
    redundant = AE.count_redundant_actions(events)
    assert redundant["npc_001"]["redundant_explorearea"] == 1


def test_explorearea_not_redundant_when_zone_unknown():
    events = [
        {"ev": "action_sent", "npc": "npc_001", "action": "ExploreArea", "args": {"zoneTag": "farmland"}},
    ]
    redundant = AE.count_redundant_actions(events)
    assert redundant.get("npc_001", {}).get("redundant_explorearea", 0) == 0


def test_analyze_empty_root_no_crash(tmp_path):
    summaries, agg = AE.analyze(tmp_path)
    assert summaries == []
    assert agg == {}
    md = AE.to_markdown(summaries, agg)
    assert "Sesiones de batería analizadas: **0**" in md


def test_aggregate_median_and_iqr(tmp_path):
    for i, (n_opt, n_act) in enumerate([(3, 3), (3, 4), (3, 6)]):
        events = [
            {"ev": "session_start", "experiment_id": "E1", "config_label": "M0", "n_opt": n_opt},
            {"ev": "goal_completed", "npc": "npc_001", "goal_belief_met": True, "intent_match": True},
        ]
        metrics = _base_metrics(f"sess{i}")
        metrics["npcs"]["npc_001"]["actions_sent"] = n_act
        metrics["npcs"]["npc_001"]["actions_ok"] = n_act
        _write_session(tmp_path, f"sess{i}", events, metrics)

    summaries, agg = AE.analyze(tmp_path)
    assert len(summaries) == 3
    key = "E1/M0"
    assert key in agg
    assert agg[key]["n"] == 3
    # eta_exec: 3/3=1.0, 3/4=0.75, 3/6=0.5 -> mediana 0.75
    assert agg[key]["eta_exec_median"] == 0.75
