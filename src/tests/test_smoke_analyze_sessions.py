"""Smoke tests — tools/analyze_sessions.py (Fase 5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# tools/ no es un paquete; añadirlo al path.
_TOOLS = Path(__file__).resolve().parent.parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import analyze_sessions as A  # noqa: E402


def _make_session(tmp_path: Path) -> Path:
    sd = tmp_path / "sess1"
    sd.mkdir()
    events = [
        {"ev": "session_start"},
        {"ev": "llm_call", "step": "parse_goals", "latency_s": 2.0, "structured": "ParseGoalsResponse"},
        {"ev": "llm_call", "step": "pipeline", "latency_s": 4.0, "structured": "Step0Response"},
        {"ev": "llm_call", "step": "pipeline", "latency_s": 6.0},
        {"ev": "action_sent", "action": "MoveTo"},
        {"ev": "action_result", "action": "MoveTo", "status": "Success", "latency_s": 1.5},
        {"ev": "action_result", "action": "Craft", "status": "Failure"},
        {"ev": "goal_replan_required", "goal": "achieve_x"},
        {"ev": "plan_failed", "goal": "achieve_y", "error": "No executable steps"},
        {"ev": "goal_completed", "goal": "achieve_x", "goal_belief_met": True, "intent_match": True},
    ]
    (sd / "trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    metrics = {
        "session_id": "sess1",
        "duration_s": 100.0,
        "npcs": {
            "npc_001": {
                "goals_started": 2, "goals_completed": 1, "goals_belief_met": 1,
                "plans_from_memory": 0, "plans_from_llm": 1,
                "llm_calls": 3, "llm_total_s": 12.0,
                "actions_sent": 2, "actions_ok": 1, "actions_failed": 1,
            }
        },
    }
    (sd / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return sd


def test_find_sessions(tmp_path):
    sd = _make_session(tmp_path)
    found = A.find_sessions(tmp_path)
    assert sd in found


def test_summarize_session(tmp_path):
    _make_session(tmp_path)
    s = A.summarize_session(tmp_path / "sess1")
    assert s["goals_started"] == 2
    assert s["goals_completed"] == 1
    assert s["goals_belief_met"] == 1
    assert s["intent_match_n"] == 1 and s["intent_match_total"] == 1
    assert s["plans_from_llm"] == 1
    assert s["plan_failed"] == 1
    assert s["replan_required"] == 1
    assert s["llm_count_by_step"]["pipeline"] == 2
    assert s["llm_latency_by_step"]["pipeline"] == 5.0  # media de 4 y 6
    assert s["structured_calls"]["ParseGoalsResponse"] == 1
    assert s["action_by_type"]["MoveTo"] == {"ok": 1, "fail": 0}
    assert s["action_by_type"]["Craft"] == {"ok": 0, "fail": 1}


def test_aggregate_and_markdown(tmp_path):
    _make_session(tmp_path)
    summaries, agg = A.analyze(tmp_path)
    assert agg["sessions"] == 1
    assert agg["goals_started"] == 2
    assert agg["plan_failed"] == 1
    md = A.to_markdown(summaries, agg)
    assert "# Análisis de sesiones" in md
    assert "Planning" in md and "Pipeline" in md and "Ejecución" in md
    assert "ParseGoalsResponse" in md


def test_csv_written(tmp_path):
    _make_session(tmp_path)
    summaries, _ = A.analyze(tmp_path)
    out = tmp_path / "out"
    A.write_csv(summaries, out)
    content = (out / "sessions.csv").read_text(encoding="utf-8")
    assert "session_id" in content
    assert "sess1" in content


def test_empty_root_no_crash(tmp_path):
    summaries, agg = A.analyze(tmp_path)
    assert summaries == []
    assert agg["sessions"] == 0
