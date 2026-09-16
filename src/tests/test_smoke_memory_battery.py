"""Smoke tests — batería de memoria de planes (suite memory_reuse).

Cubren (sin LLM ni Unity): la suite y sus lanzadores, el emparejado 1.ª → 2.ª
sesión por directorio de memoria, el agregado y Wilcoxon de rangos con signo.
El reuso real del plan (motor + pipeline) está en test_integration_memory_reuse.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import analyze_memory as AM  # noqa: E402

_SUITE = json.loads((_TOOLS / "experiments" / "suites" / "memory_reuse.json").read_text(encoding="utf-8"))


def test_suite_pairs_both_arms_on_a4_and_a5():
    assert _SUITE["experiments"] == ["A4", "A5"]
    assert _SUITE["arms"] == ["SUB", "ATOM"] and _SUITE["passes"] == ["M1", "M2"]
    assert AM.suite_configs(_SUITE) == ["SUB_M1", "SUB_M2", "ATOM_M1", "ATOM_M2"]
    assert _SUITE["config_builtin_subplans"] == {
        "SUB_M1": True, "SUB_M2": True, "ATOM_M1": False, "ATOM_M2": False,
    }
    # 2 experimentos × 2 brazos × 2 sesiones × runs
    assert 2 * 2 * 2 * _SUITE["runs"] == 16


def test_launchers_know_the_memory_configs():
    exp = (_TOOLS / "run_experiment.ps1").read_text(encoding="utf-8")
    for cfg in ("SUB_M1", "SUB_M2", "ATOM_M1", "ATOM_M2"):
        assert f'"{cfg}"' in exp
    # Mismo brazo que SUB/ATOM y directorio de memoria por experimento, brazo y repetición.
    assert '"ATOM_M1" { $env:NPC_BUILTIN_SUBPLANS = "0"' in exp
    assert '"SUB_M2"  { $env:NPC_BUILTIN_SUBPLANS = "1"' in exp
    assert "mem_${Id}_${arm}_rep$pairIdx" in exp
    suite = (_TOOLS / "run_suite_memory.ps1").read_text(encoding="utf-8")
    assert "analyze_memory.py" in suite and '"memory_reuse"' in suite
    assert suite.isascii()


def _session(root: Path, name: str, exp: str, cfg: str, run_dir: str, *, t_goal: float, llm: int,
             reuse: bool, success: bool = True, duration: float = 60.0) -> None:
    d = root / name
    d.mkdir(parents=True)
    goals = {"A4": [("npc_001", "has_item(wheat, 2)"), ("npc_001", "has_item(bread, 1)")],
             "A5": [("npc_001", "has_item(bread, 3)")]}[exp]
    events = [
        {"t": 1000.0, "ev": "session_start", "experiment_id": exp, "config_label": cfg,
         "git_sha": "abc1234", "builtin_subplans": cfg.startswith("SUB")},
        {"t": 1000.1, "ev": "plan_memory_run", "mode": "reuse", "run_dir": run_dir},
        {"t": 1000.2, "ev": "unity_in", "npc": "npc_001", "msg_type": "RegisterNPC"},
        {"t": 1000.3, "ev": "plan_memory_load", "npc": "npc_001", "count": 1 if reuse else 0},
    ]
    events += [{"t": 1001.0, "ev": "llm_call", "npc": "npc_001", "latency_s": 1.0}] * llm
    if reuse:
        events.append({"t": 1002.0, "ev": "plan_memory_reuse", "npc": "npc_001", "goal": "achieve_bake_bread"})
    else:
        events.append({"t": 1002.0, "ev": "plan_requested", "npc": "npc_001", "goal": "achieve_bake_bread"})
    if success:
        events += [
            {"t": 1000.0 + t_goal, "ev": "goal_completed", "npc": npc, "goal_belief_met": True,
             "expected_condition": cond}
            for npc, cond in goals
        ]
    events += [{"t": 1000.0 + duration, "ev": "shutdown_idle"},
               {"t": 1000.0 + duration, "ev": "session_end", "duration_s": duration}]
    (d / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def test_pairs_are_matched_by_memory_dir_and_compared(tmp_path):
    root = tmp_path / "sessions"
    manifests = AM.AA.load_manifests(["A4", "A5"])
    for rep, (t1, t2) in enumerate(((90.0, 30.0), (100.0, 40.0), (80.0, 35.0)), start=1):
        mem = f"plans/memory/runs/mem_A5_ATOM_rep{rep}"
        _session(root, f"s{rep}a", "A5", "ATOM_M1", mem, t_goal=t1, llm=19, reuse=False, duration=t1 + 5)
        _session(root, f"s{rep}b", "A5", "ATOM_M2", mem, t_goal=t2, llm=1, reuse=True, duration=t2 + 5)
    # Un M1 fallido: su M2 no puede reusar.
    _session(root, "s4a", "A5", "SUB_M1", "plans/memory/runs/mem_A5_SUB_rep1", t_goal=0, llm=9, reuse=False,
             success=False, duration=245)
    _session(root, "s4b", "A5", "SUB_M2", "plans/memory/runs/mem_A5_SUB_rep1", t_goal=94, llm=9, reuse=False)
    # Sesión ajena a la suite: se ignora.
    _session(root, "s5", "A5", "ATOM", "x", t_goal=50, llm=10, reuse=False)

    summaries = AM.summarize(root, _SUITE, manifests)
    assert len(summaries) == 8
    pairs = AM.build_pairs(summaries)
    assert [(p["arm"], Path(p["run_dir"]).name) for p in pairs] == [
        ("ATOM", "mem_A5_ATOM_rep1"), ("ATOM", "mem_A5_ATOM_rep2"), ("ATOM", "mem_A5_ATOM_rep3"),
        ("SUB", "mem_A5_SUB_rep1"),
    ]

    agg = AM.aggregate(summaries)
    assert agg[("A5", "ATOM", "M2")]["memory_reused"] == 3
    assert agg[("A5", "ATOM", "M1")]["t_success"][0] == 90.0
    assert agg[("A5", "SUB", "M1")]["success"] == 0

    rows = {(r["experiment_id"], r["arm"]): r for r in AM.compare_pairs(pairs)}
    atom = rows[("A5", "ATOM")]
    assert atom["pairs"] == 3 and atom["pairs_both_success"] == 3 and atom["second_reused"] == 3
    assert atom["t_success_delta"][0] == -60.0
    assert atom["llm_calls_delta"][0] == -18.0
    assert atom["t_success_p"] == pytest.approx(0.25)     # 3 pares, todos a la baja: p exacto = 2/8
    sub = rows[("A5", "SUB")]
    assert sub["pairs_both_success"] == 0 and sub["t_success_n"] == 0

    md = AM.to_markdown(summaries, agg, pairs, AM.compare_pairs(pairs), _SUITE)
    assert "| A5 | ATOM | 3/3 | 3/3 |" in md and "| 3/3 |" in md


def test_wilcoxon_signed_rank_exact_and_edge_cases():
    assert AM.wilcoxon_signed_rank([]) is None
    assert AM.wilcoxon_signed_rank([0.0, None]) is None
    w, p = AM.wilcoxon_signed_rank([-5, -3, -8, -1, -2, -7])
    assert w == 0 and p == pytest.approx(2 / 64)
    w, p = AM.wilcoxon_signed_rank([-1, 1])
    assert w == pytest.approx(1.5) and p == pytest.approx(1.0)
