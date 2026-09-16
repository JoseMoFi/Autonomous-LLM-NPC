from __future__ import annotations

"""Tests de humo para TraceLogger y SessionMetrics."""

import json
import time
from pathlib import Path

import pytest

from utils.trace_logger import TraceLogger, NPCMetrics, SessionMetrics, GoalRecord, trace, unity_log


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    """Devuelve un directorio temporal de sesión limpio."""
    d = tmp_path / "sessions" / "20260426" / "120000"
    d.mkdir(parents=True)
    return d


@pytest.fixture(autouse=True)
def reset_singleton():
    """Garantiza que el singleton de TraceLogger está limpio antes y después de cada test."""
    TraceLogger._instance = None
    yield
    if TraceLogger._instance is not None:
        try:
            TraceLogger._instance._trace_fh.close()
            TraceLogger._instance._prompt_fh.close()
            TraceLogger._instance._unity_fh.close()
        except Exception:
            pass
        TraceLogger._instance = None


# ---------------------------------------------------------------------------
# NPCMetrics
# ---------------------------------------------------------------------------

class TestNPCMetrics:
    def test_to_dict_basic(self):
        m = NPCMetrics(npc_id="npc_001")
        d = m.to_dict()
        assert d["npc_id"] == "npc_001"
        assert d["goals_started"] == 0
        # Sin acciones ni goals no debe haber métricas derivadas
        assert "action_success_rate" not in d
        assert "goal_success_rate" not in d
        assert "llm_avg_latency_s" not in d

    def test_action_success_rate(self):
        m = NPCMetrics(npc_id="npc_001", actions_sent=4, actions_ok=3, actions_failed=1)
        d = m.to_dict()
        assert d["action_success_rate"] == pytest.approx(0.75)

    def test_goal_success_rate(self):
        m = NPCMetrics(npc_id="npc_001", goals_started=2, goals_completed=1, goals_failed=1)
        d = m.to_dict()
        assert d["goal_success_rate"] == pytest.approx(0.5)

    def test_llm_avg_latency(self):
        m = NPCMetrics(npc_id="npc_001", llm_calls=4, llm_total_s=8.0)
        d = m.to_dict()
        assert d["llm_avg_latency_s"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# SessionMetrics
# ---------------------------------------------------------------------------

class TestSessionMetrics:
    def test_npc_creates_on_first_access(self):
        sm = SessionMetrics(session_id="test/session")
        m = sm.npc("npc_001")
        assert isinstance(m, NPCMetrics)
        assert m.npc_id == "npc_001"

    def test_npc_returns_same_instance(self):
        sm = SessionMetrics(session_id="test/session")
        a = sm.npc("npc_001")
        b = sm.npc("npc_001")
        assert a is b

    def test_to_dict_has_duration(self):
        sm = SessionMetrics(session_id="test/session")
        time.sleep(0.01)
        d = sm.to_dict()
        assert d["duration_s"] >= 0.0
        assert "started_at" in d
        assert "npcs" in d


# ---------------------------------------------------------------------------
# TraceLogger
# ---------------------------------------------------------------------------

class TestTraceLogger:
    def test_init_creates_files(self, session_dir: Path):
        tl = TraceLogger(session_dir)
        tl._trace_fh.close()
        tl._prompt_fh.close()
        tl._unity_fh.close()
        assert (session_dir / "trace.jsonl").exists()
        assert (session_dir / "prompts.jsonl").exists()
        assert (session_dir / "unity_tcp.log").exists()

    def test_trace_writes_jsonl(self, session_dir: Path):
        tl = TraceLogger(session_dir)
        tl.trace("goal_started", npc_id="npc_001", goal="achieve_test")
        tl._trace_fh.close()
        tl._prompt_fh.close()
        tl._unity_fh.close()

        lines = (session_dir / "trace.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["ev"] == "goal_started"
        assert rec["npc"] == "npc_001"
        assert rec["goal"] == "achieve_test"
        assert "t" in rec

    def test_trace_without_npc_id(self, session_dir: Path):
        tl = TraceLogger(session_dir)
        tl.trace("session_start", model="qwen2.5:7b")
        tl._trace_fh.close()
        tl._prompt_fh.close()
        tl._unity_fh.close()

        rec = json.loads((session_dir / "trace.jsonl").read_text(encoding="utf-8").strip())
        assert rec["ev"] == "session_start"
        assert "npc" not in rec
        assert rec["model"] == "qwen2.5:7b"

    def test_log_prompt_writes_jsonl(self, session_dir: Path):
        tl = TraceLogger(session_dir)
        tl.log_prompt(
            npc_id="npc_001",
            goal="achieve_harvest",
            step="step3_steps",
            prompt="USER_PROMPT",
            system="SYS_PROMPT",
            response="LLM_RESPONSE",
            latency_s=1.5,
        )
        tl._trace_fh.close()
        tl._prompt_fh.close()
        tl._unity_fh.close()

        rec = json.loads((session_dir / "prompts.jsonl").read_text(encoding="utf-8").strip())
        assert rec["step"] == "step3_steps"
        assert rec["prompt"] == "USER_PROMPT"
        assert rec["response"] == "LLM_RESPONSE"
        assert rec["latency_s"] == pytest.approx(1.5)

    def test_log_unity_writes_jsonl(self, session_dir: Path):
        tl = TraceLogger(session_dir)
        tl.log_unity(direction="in", npc_id="npc_001", msg_type="Tick", payload={"type": "Tick"})
        tl._trace_fh.close()
        tl._prompt_fh.close()
        tl._unity_fh.close()

        rec = json.loads((session_dir / "unity_tcp.log").read_text(encoding="utf-8").strip())
        assert rec["dir"] == "in"
        assert rec["npc"] == "npc_001"
        assert rec["msg_type"] == "Tick"
        assert rec["payload"]["type"] == "Tick"

    def test_finalize_writes_metrics_json(self, session_dir: Path):
        tl = TraceLogger(session_dir)
        tl.metrics.npc("npc_001").goals_completed = 2
        tl.finalize()

        assert (session_dir / "metrics.json").exists()
        data = json.loads((session_dir / "metrics.json").read_text(encoding="utf-8"))
        assert "npcs" in data
        assert data["npcs"]["npc_001"]["goals_completed"] == 2
        assert "duration_s" in data

    def test_finalize_resets_singleton(self, session_dir: Path):
        TraceLogger._instance = TraceLogger(session_dir)
        TraceLogger._instance.finalize()
        assert TraceLogger.get() is None

    def test_init_session_creates_directory(self, tmp_path: Path):
        tl = TraceLogger.init_session(logs_root=tmp_path / "logs")
        assert tl.session_dir.exists()
        assert (tl.session_dir / "trace.jsonl").exists()
        tl.finalize()

    def test_init_session_singleton(self, tmp_path: Path):
        tl1 = TraceLogger.init_session(logs_root=tmp_path / "logs")
        tl2 = TraceLogger.init_session(logs_root=tmp_path / "logs")
        assert tl1 is tl2
        tl1.finalize()


# ---------------------------------------------------------------------------
# Module-level trace() helper
# ---------------------------------------------------------------------------

class TestModuleTrace:
    def test_trace_noop_without_session(self):
        """trace() debe ser un no-op si no hay sesión activa."""
        assert TraceLogger.get() is None
        trace("some_event", npc_id="npc_001")  # no debe lanzar
        unity_log(direction="event", note="no_session")

    def test_trace_writes_when_session_active(self, session_dir: Path):
        TraceLogger._instance = tl = TraceLogger(session_dir)
        trace("test_event", npc_id="npc_001", value=42)
        tl._trace_fh.close()
        tl._prompt_fh.close()
        tl._unity_fh.close()
        TraceLogger._instance = None

        rec = json.loads((session_dir / "trace.jsonl").read_text(encoding="utf-8").strip())
        assert rec["ev"] == "test_event"
        assert rec["value"] == 42

    def test_unity_log_writes_when_session_active(self, session_dir: Path):
        TraceLogger._instance = tl = TraceLogger(session_dir)
        unity_log(direction="event", note="connected")
        tl._trace_fh.close()
        tl._prompt_fh.close()
        tl._unity_fh.close()
        TraceLogger._instance = None

        rec = json.loads((session_dir / "unity_tcp.log").read_text(encoding="utf-8").strip())
        assert rec["dir"] == "event"
        assert rec["note"] == "connected"


# ---------------------------------------------------------------------------
# T4 - GoalRecord y métricas de creencias
# ---------------------------------------------------------------------------

class TestGoalRecord:
    def test_to_dict_completed(self):
        r = GoalRecord(
            sig="achieve_have_wheat",
            success_condition="has_item(wheat, 1)",
            belief_met=True,
            replan_count=0,
            final_status="completed",
        )
        d = r.to_dict()
        assert d["sig"] == "achieve_have_wheat"
        assert d["belief_met"] is True
        assert d["replan_count"] == 0
        assert d["final_status"] == "completed"

    def test_to_dict_failed(self):
        r = GoalRecord(
            sig="achieve_have_wheat",
            success_condition="has_item(wheat, 1)",
            belief_met=False,
            replan_count=3,
            final_status="failed",
        )
        d = r.to_dict()
        assert d["belief_met"] is False
        assert d["replan_count"] == 3
        assert d["final_status"] == "failed"

    def test_to_dict_unverified(self):
        r = GoalRecord(
            sig="achieve_something",
            success_condition=None,
            belief_met=None,
            replan_count=0,
            final_status="unverified",
        )
        d = r.to_dict()
        assert d["success_condition"] is None
        assert d["belief_met"] is None
        assert d["final_status"] == "unverified"


class TestNPCMetricsBeliefCounters:
    def test_initial_belief_counters_zero(self):
        m = NPCMetrics(npc_id="npc_001")
        assert m.goals_belief_met == 0
        assert m.goals_belief_missing == 0
        assert m.goals_replan_attempted == 0
        assert m.goals_replan_success == 0
        assert m.goals_unverified == 0
        assert m.goals_detail == []

    def test_record_goal_belief_met(self):
        m = NPCMetrics(npc_id="npc_001", goals_started=1)
        r = GoalRecord("achieve_have_wheat", "has_item(wheat, 1)", True, 0, "completed")
        m.record_goal(r)
        assert m.goals_belief_met == 1
        assert m.goals_completed == 1
        assert m.goals_belief_missing == 0
        assert m.goals_unverified == 0
        assert len(m.goals_detail) == 1

    def test_record_goal_belief_missing_failed(self):
        m = NPCMetrics(npc_id="npc_001", goals_started=1)
        r = GoalRecord("achieve_have_wheat", "has_item(wheat, 1)", False, 3, "failed")
        m.record_goal(r)
        assert m.goals_belief_missing == 1
        assert m.goals_failed == 1
        assert m.goals_belief_met == 0
        assert m.goals_completed == 0

    def test_record_goal_unverified(self):
        m = NPCMetrics(npc_id="npc_001", goals_started=1)
        r = GoalRecord("achieve_something", None, None, 0, "unverified")
        m.record_goal(r)
        assert m.goals_unverified == 1
        assert m.goals_belief_met == 0
        assert m.goals_belief_missing == 0

    def test_belief_success_rate_in_to_dict(self):
        m = NPCMetrics(npc_id="npc_001", goals_started=4)
        m.goals_belief_met = 3
        d = m.to_dict()
        assert "belief_success_rate" in d
        assert d["belief_success_rate"] == pytest.approx(0.75)

    def test_belief_success_rate_absent_without_goals(self):
        m = NPCMetrics(npc_id="npc_001")  # goals_started=0
        d = m.to_dict()
        assert "belief_success_rate" not in d

    def test_goals_detail_serialized_in_to_dict(self, session_dir: Path):
        m = NPCMetrics(npc_id="npc_001", goals_started=2)
        m.record_goal(GoalRecord("achieve_a", "has_item(a, 1)", True, 0, "completed"))
        m.record_goal(GoalRecord("achieve_b", "has_item(b, 1)", False, 3, "failed"))
        d = m.to_dict()
        assert isinstance(d["goals_detail"], list)
        assert len(d["goals_detail"]) == 2
        assert d["goals_detail"][0]["sig"] == "achieve_a"
        assert d["goals_detail"][1]["final_status"] == "failed"

    def test_goals_detail_persisted_in_metrics_json(self, session_dir: Path):
        tl = TraceLogger(session_dir)
        npc = tl.metrics.npc("npc_001")
        npc.goals_started = 1
        npc.record_goal(GoalRecord("achieve_wheat", "has_item(wheat, 1)", True, 0, "completed"))
        tl.finalize()

        data = json.loads((session_dir / "metrics.json").read_text(encoding="utf-8"))
        detail = data["npcs"]["npc_001"]["goals_detail"]
        assert len(detail) == 1
        assert detail[0]["sig"] == "achieve_wheat"
        assert detail[0]["belief_met"] is True

    def test_two_goals_belief_success_rate(self, session_dir: Path):
        """Un goal cumplido y otro fallido → belief_success_rate = 0.5."""
        tl = TraceLogger(session_dir)
        npc = tl.metrics.npc("npc_001")
        npc.goals_started = 2
        npc.record_goal(GoalRecord("achieve_a", "has_item(a, 1)", True, 0, "completed"))
        npc.record_goal(GoalRecord("achieve_b", "has_item(b, 1)", False, 3, "failed"))
        tl.finalize()

        data = json.loads((session_dir / "metrics.json").read_text(encoding="utf-8"))
        assert data["npcs"]["npc_001"]["belief_success_rate"] == pytest.approx(0.5)
        assert data["npcs"]["npc_001"]["goals_belief_met"] == 1
        assert data["npcs"]["npc_001"]["goals_belief_missing"] == 1
