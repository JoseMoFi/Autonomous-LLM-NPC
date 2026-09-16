"""Smoke tests — Fase 16: ablación de sub-planes (move_to_and_pickup/craft_item).

Cubren (sin LLM ni Unity):
  - builtin_loader omite los ficheros ablacionados.
  - step1b: la rama de zona no prescribe craft_item sin sub-planes.
  - Prompt de step3: sin menciones a los sub-planes con builtin_subplans=False
    (modos atómico y no atómico); cada sustitución casa con el texto real.
  - run_full_pipeline: ningún prompt los menciona y la escalera recibe las
    creencias; con True siguen ofreciéndose.
  - step3 / mini_repair / plan_simulator respetan el conjunto activo.
  - Settings: NPC_BUILTIN_SUBPLANS / NPC_ISOLATE_CONTRACTS / NPC_SESSION_MAX_S.
  - session_timeout_watchdog dispara el cierre ordenado.
  - Suite, manifiestos A1..A4 y configs SUB/ATOM (una sola variable).
  - analyze_ablation: estadística y clasificación de sesiones sintéticas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import analyze_ablation as AA  # noqa: E402
from gateway.shutdown import session_timeout_watchdog  # noqa: E402
from llm.pipeline.builtins import ABLATABLE_BUILTIN_FILES, terminal_subplans  # noqa: E402
from llm.pipeline.mini_repair import repair_plan_gaps  # noqa: E402
from llm.pipeline.pipeline_runner import run_full_pipeline  # noqa: E402
from llm.pipeline.plan_simulator import active_subgoal_guarantees, check_plan_reachability  # noqa: E402
from llm.pipeline.step1b_decompose import build_need_variants  # noqa: E402
from llm.pipeline.step3_steps import run_step3  # noqa: E402
from llm.prompts.planning import _NO_BUILTIN_SUBPLAN_REPLACEMENTS, build_prompt  # noqa: E402
from npc.builtin_loader import load_builtin_plans  # noqa: E402

_ABLATED = ("move_to_and_pickup", "craft_item")
_BUILTIN_DIR = _REPO / "src" / "plans" / "builtin"
_EXPERIMENTS = _REPO / "tools" / "experiments"
_RECIPES = [{
    "recipeId": "Bread_recipe",
    "zone": "bakeri",
    "inputs": [{"itemId": "wheat", "qty": 2}],
    "outputs": [{"itemId": "bread", "qty": 1}],
}]
_SPAWNS = [{"itemId": "wheat", "zones": ["farmland"]}]
_CATALOG = {"item_ids": ["wheat", "bread"], "zone_ids": ["farmland", "bakeri"], "recipe_ids": ["Bread_recipe"]}
_ATOMIC_STEPS = json.dumps({"steps": [
    {"type": "action", "name": "MoveTo", "args": [3, 4]},
    {"type": "action", "name": "PickUp", "args": ["wheat"]},
]})


def _mentions_ablated(text: str) -> list[str]:
    return [name for name in _ABLATED if name in text]


# ===========================================================================
# Carga de builtins
# ===========================================================================

def test_builtin_loader_excludes_ablated_files():
    full = load_builtin_plans(nx.DiGraph(), _BUILTIN_DIR)
    assert {"move_to_and_pickup", "craft_item", "achieve_explore_zone"} <= set(full)

    ablated = load_builtin_plans(nx.DiGraph(), _BUILTIN_DIR, exclude_files=ABLATABLE_BUILTIN_FILES)
    assert not {"move_to_and_pickup", "craft_item", "craft_item_check"} & set(ablated)
    assert "achieve_explore_zone" in ablated


def test_terminal_subplans_without_ablated():
    assert {"move_to_and_pickup", "craft_item"} <= terminal_subplans(True)
    assert terminal_subplans(False) == frozenset({"achieve_explore_zone"})


# ===========================================================================
# step1b y prompts
# ===========================================================================

def test_step1b_zone_variant_does_not_prescribe_craft_item_without_subplans():
    on = build_need_variants("has_item(bread, 1)", _RECIPES, _SPAWNS)
    off = build_need_variants("has_item(bread, 1)", _RECIPES, _SPAWNS, builtin_subplans=False)
    assert any("craft_item" in v.problem_nl for v in on)
    assert not any(_mentions_ablated(v.problem_nl) for v in off)
    # Zona y crafteo: misma escalera de guards en ambos brazos. El ingrediente
    # recolectable se desgrana en ATOM en las precondiciones de PickUp (Fase 17f);
    # en SUB esa estructura la aporta move_to_and_pickup.asl.
    assert [v.variant_guard for v in on][1:] == [v.variant_guard for v in off][3:]
    assert [v.variant_guard for v in on][0] == "not has_item(bread, 1) & not has_item(wheat, 2)"
    base = "not has_item(bread, 1) & not has_item(wheat, 2)"
    assert [v.variant_guard for v in off][:3] == [
        f"{base} & item_at(wheat, X, Y) & current_position(X, Y)",
        f"{base} & item_at(wheat, X, Y) & not current_position(X, Y)",
        f"{base} & not item_at(wheat, _, _)",
    ]


def test_step1b_repeats_the_recipe_when_one_craft_is_not_enough():
    # Fase 17k (A5: 3 panes con una receta de 1 pan). Antes no había escalera.
    for builtin in (True, False):
        variants = build_need_variants("has_item(bread, 3)", _RECIPES, _SPAWNS, builtin_subplans=builtin)
        assert variants, builtin
        assert all(v.variant_guard.startswith("not has_item(bread, 3)") for v in variants)
        outputs = [f for v in variants for f in v.known_facts if f.startswith("recipe_output(")]
        assert outputs and set(outputs) == {"recipe_output(Bread_recipe, bakeri, bread, 1)"}
        assert "runs again" in variants[-1].problem_nl
        assert variants[-1].unsatisfied_condition == "not has_item(bread, 3)"
    # Objetivo de 1 unidad: escalera como antes, sin nota de repetición.
    assert not any("runs again" in v.problem_nl for v in build_need_variants("has_item(bread, 1)", _RECIPES, _SPAWNS))


def test_step1b_gather_rungs_follow_the_pickup_contract_without_subplans():
    # SUB: sin cambios (sin receta que produzca wheat → variante única).
    assert build_need_variants("has_item(wheat, 1)", _RECIPES, _SPAWNS) == []
    rungs = build_need_variants("has_item(wheat, 1)", _RECIPES, _SPAWNS, builtin_subplans=False)
    # Del estado más avanzado al menos: agentspeak elige el primer plan aplicable
    # y, con varios ejemplares vistos, "en la casilla" debe ganar a "visto, lejos".
    assert [v.variant_guard for v in rungs] == [
        "not has_item(wheat, 1) & item_at(wheat, X, Y) & current_position(X, Y)",
        "not has_item(wheat, 1) & item_at(wheat, X, Y) & not current_position(X, Y)",
        "not has_item(wheat, 1) & not item_at(wheat, _, _)",
    ]
    assert [v.unsatisfied_condition for v in rungs] == [
        "not has_item(wheat, 1)", "not current_position(X, Y)", "not item_at(wheat, _, _)",
    ]
    assert [v.bound_variables for v in rungs] == [["X", "Y"], ["X", "Y"], []]
    assert "farmland" in rungs[2].problem_nl
    assert "not at a crafting zone" in rungs[2].problem_nl
    assert all(v.known_facts == ["item_spawn(wheat, farmland)"] for v in rungs)
    assert not any(_mentions_ablated(v.problem_nl) for v in rungs)
    # Sin spawn conocido no hay de dónde recolectar: sin escalera.
    assert build_need_variants("has_item(stone, 1)", _RECIPES, _SPAWNS, builtin_subplans=False) == []


def _step3_payload(**overrides) -> dict:
    payload = {
        "task": "step3_steps",
        "goal_name": "achieve_collect_wheat",
        "npc_statement": "Collect 1 wheat",
        "neg_guard": "not has_item(wheat, 1)",
        "problem_nl": "The NPC has no wheat.",
        "known_facts": ["item_spawn(wheat, farmland)"],
        "existing_subgoals": ["achieve_explore_zone"],
        "entity_catalog": _CATALOG,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("atomic_only", [True, False])
def test_step3_prompt_has_no_ablated_subplans(atomic_only: bool):
    user, system = build_prompt(_step3_payload(atomic_only=atomic_only, builtin_subplans=False))
    assert _mentions_ablated(user + system) == []


@pytest.mark.parametrize("atomic_only", [True, False])
def test_step3_prompt_default_keeps_subplan_guidance(atomic_only: bool):
    user, system = build_prompt(_step3_payload(atomic_only=atomic_only))
    assert "move_to_and_pickup" in user + system


def test_every_prompt_replacement_matches_rendered_text():
    """Si alguien cambia el prompt, una sustitución dejaría de casar en silencio."""
    rendered = "".join(
        user + system
        for atomic in (True, False)
        for user, system in [build_prompt(_step3_payload(atomic_only=atomic))]
    )
    for old, _new in _NO_BUILTIN_SUBPLAN_REPLACEMENTS:
        assert old in rendered, old[:80]


# ===========================================================================
# Pipeline completo (LLM falso que captura prompts)
# ===========================================================================

_CRAFT_STEPS = json.dumps({"steps": [
    {"type": "action", "name": "MoveTo", "args": [3, 4]},
    {"type": "action", "name": "Craft", "args": ["wheat", "Bread_recipe"]},
]})


def _capturing_llm(prompts: list[str]):
    async def _llm(user_prompt: str, system_prompt: str) -> str:
        prompts.append(user_prompt + "\n" + system_prompt)
        # Fase 17t: step3 rechaza volver a recoger un item que el guard ya garantiza.
        if "  - has_item(wheat" in user_prompt:
            return _CRAFT_STEPS
        return _ATOMIC_STEPS
    return _llm


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_pipeline_offers_subplans_only_when_enabled(tmp_path, enabled: bool):
    prompts: list[str] = []
    await run_full_pipeline(
        goal_sig="achieve_collect_wheat",
        npc_statement="Collect 1 wheat",
        existing_goals=["achieve_explore_zone"],
        llm_call=_capturing_llm(prompts),
        description="The NPC must collect 1 wheat.",
        success_condition="has_item(wheat, 1)",
        use_refinement=True,
        capability_contracts_path=str(tmp_path / "contracts.json"),
        entity_catalog=_CATALOG,
        beliefs={"zone_center": [["farmland", 9, -6]]},
        builtin_subplans=enabled,
    )
    joined = "\n".join(prompts)
    assert prompts
    if enabled:
        assert "move_to_and_pickup" in joined
    else:
        assert _mentions_ablated(joined) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_ladder_rungs_receive_beliefs_only_without_subplans(tmp_path, enabled: bool):
    prompts: list[str] = []
    await run_full_pipeline(
        goal_sig="achieve_bake_bread",
        npc_statement="Bake 1 bread",
        existing_goals=["achieve_explore_zone"],
        llm_call=_capturing_llm(prompts),
        description="The NPC must bake 1 bread.",
        success_condition="has_item(bread, 1)",
        use_refinement=True,
        capability_contracts_path=str(tmp_path / "contracts.json"),
        entity_catalog=_CATALOG,
        beliefs={"zone_center": [["farmland", 9, -6]]},
        recipes=_RECIPES,
        item_spawns=_SPAWNS,
        builtin_subplans=enabled,
    )
    joined = "\n".join(prompts)
    assert ("zone_center(farmland, 9, -6)" in joined) is (not enabled)
    if not enabled:
        assert _mentions_ablated(joined) == []


# ===========================================================================
# step3 / mini_repair / simulador
# ===========================================================================

@pytest.mark.asyncio
async def test_step3_all_subgoal_plan_needs_an_active_terminal():
    response = json.dumps({"steps": [{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 1]}]})

    async def llm(_user: str, _system: str) -> str:
        return response

    steps = await run_step3(
        "achieve_collect_wheat", "d", "not has_item(wheat, 1)", "p", [],
        ["move_to_and_pickup"], llm, atomic_only=True,
    )
    assert steps[0]["name"] == "move_to_and_pickup"

    with pytest.raises(ValueError):
        await run_step3(
            "achieve_collect_wheat", "d", "not has_item(wheat, 1)", "p", [],
            [], llm, atomic_only=True, builtin_subplans=False,
        )


@pytest.mark.asyncio
async def test_mini_repair_ignores_ablated_names():
    async def llm(_user: str, _system: str) -> str:
        raise AssertionError("no debe consultar al LLM")

    steps = [
        {"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat"]},
        {"type": "action", "name": "PickUp", "args": ["wheat"]},
    ]
    out = await repair_plan_gaps(
        [dict(s) for s in steps], "achieve_x", llm, terminal_subplans=terminal_subplans(False),
    )
    assert out == steps


def test_simulator_credits_ablated_subplans_only_when_active():
    steps = [{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 1]}]
    assert check_plan_reachability(steps, {}, "has_item(wheat, 1)").reachable
    assert not check_plan_reachability(
        steps, {}, "has_item(wheat, 1)", subgoal_guarantees=active_subgoal_guarantees(False),
    ).reachable


# ===========================================================================
# Settings y cierre ordenado
# ===========================================================================

def test_settings_env_overrides(monkeypatch):
    from config import Settings

    saved = Settings._instance
    try:
        Settings._instance = None
        monkeypatch.setenv("NPC_BUILTIN_SUBPLANS", "0")
        monkeypatch.setenv("NPC_ISOLATE_CONTRACTS", "1")
        monkeypatch.setenv("NPC_SESSION_MAX_S", "335")
        s = Settings.load()
        assert s.builtin_subplans_enabled is False
        assert s.isolate_capability_contracts is True
        assert s.session_max_s == 335.0

        Settings._instance = None
        for var in ("NPC_BUILTIN_SUBPLANS", "NPC_ISOLATE_CONTRACTS", "NPC_SESSION_MAX_S"):
            monkeypatch.delenv(var)
        s = Settings.load()
        assert s.builtin_subplans_enabled is True
        assert s.isolate_capability_contracts is False
        assert s.session_max_s == 0.0
    finally:
        Settings._instance = saved


class _FakeShutdown:
    def __init__(self, triggered: bool = False) -> None:
        self.triggered = triggered
        self.calls = 0

    def is_triggered(self) -> bool:
        return self.triggered

    def _trigger(self) -> None:
        self.calls += 1
        self.triggered = True


class _FakeRegistry:
    def open_goals(self):
        class _Goal:
            sig = "achieve_collect_wheat"
            success_condition = "has_item(wheat, 1)"
            replan_count = 2
        return [("npc_001", _Goal())]


@pytest.mark.asyncio
async def test_session_timeout_watchdog_triggers_shutdown():
    shutdown = _FakeShutdown()
    await session_timeout_watchdog(_FakeRegistry(), shutdown, 0.01)
    assert shutdown.calls == 1

    already = _FakeShutdown(triggered=True)
    await session_timeout_watchdog(_FakeRegistry(), already, 0.01)
    assert already.calls == 0


# ===========================================================================
# Suite, manifiestos y configs
# ===========================================================================

@pytest.mark.parametrize("suite_id", ["ablation_builtins", "ablation_long", "ablation_short"])
def test_suite_and_manifests_are_consistent(suite_id: str):
    suite = json.loads((_EXPERIMENTS / "suites" / f"{suite_id}.json").read_text(encoding="utf-8"))
    assert suite["configs"] == ["SUB", "ATOM"]
    assert suite["config_builtin_subplans"] == {"SUB": True, "ATOM": False}
    # Múltiplo del nº de brazos: cada brazo abre las mismas veces por experimento.
    assert suite["runs"] % len(suite["configs"]) == 0
    for exp_id in suite["experiments"]:
        manifest = json.loads((_EXPERIMENTS / f"{exp_id}.json").read_text(encoding="utf-8"))
        assert manifest["id"] == exp_id
        assert manifest["configs"] == suite["configs"]
        assert all(manifest["runs"][cfg] >= 1 for cfg in suite["configs"])
        # Fase 17l: nada que funcione tarda mas de ~110 s; el tope corta los atascos.
        assert 120 <= manifest["timeout_s"] <= 270
        for goals in manifest["goals"].values():
            assert all("nl" in g and "condition" in g for g in goals)
        # El escenario no puede fijar la variable del brazo.
        assert not {"NPC_BUILTIN_SUBPLANS", "NPC_COORDINATION_PLANNER"} & set(manifest.get("env") or {})


def test_long_suite_extends_the_ablation_with_the_two_npc_chain():
    base = json.loads((_EXPERIMENTS / "suites" / "ablation_builtins.json").read_text(encoding="utf-8"))
    long = json.loads((_EXPERIMENTS / "suites" / "ablation_long.json").read_text(encoding="utf-8"))
    assert long["experiments"] == base["experiments"] + ["A5"]
    assert long["runs"] == 16
    # A5: un solo NPC, 3 panes (6 trigos, 3 crafteos). La cooperación va en CO5/CO6.
    a5 = json.loads((_EXPERIMENTS / "A5.json").read_text(encoding="utf-8"))
    assert a5["npcs"] == 1
    assert a5["goals"] == {"npc_001": [{"nl": "Bake 3 @bread", "condition": "has_item(bread, 3)"}]}
    assert a5["build_exe"] == "builds\\single\\My project.exe"
    assert "env" not in a5 and "ablated_subplans" not in a5
    # El lanzador largo encadena las dos fases: ablación y cooperación.
    long_ps1 = (_REPO / "tools" / "run_suite_long.ps1").read_text(encoding="utf-8")
    assert 'Suite "ablation_long"' in long_ps1
    assert 'Suite "coordination_builtins"' in long_ps1
    assert "MaxHours" in long_ps1 and "$Phase" in long_ps1
    assert (_EXPERIMENTS / "suites" / "coordination_builtins.json").exists()
    # Tanda corta de comprobación: mismo diseño que la larga, 2 repeticiones.
    short = json.loads((_EXPERIMENTS / "suites" / "ablation_short.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in short.items() if k not in ("id", "label", "runs", "notes")} ==         {k: v for k, v in long.items() if k not in ("id", "label", "runs", "notes")}
    assert short["runs"] == 2
    assert (_REPO / "tools" / "run_suite_short.ps1").exists()


def _loaded_session(tmp_path, exp_id: str, npc: str, sigs: list[str]) -> Path:
    events = [
        {"t": 0.0, "ev": "session_start", "experiment_id": exp_id, "config_label": "ATOM",
         "builtin_subplans": False},
        {"t": 0.0, "ev": "unity_in", "npc": npc, "msg_type": "RegisterNPC"},
        {"t": 0.1, "ev": "builtin_plans_loaded", "npc": npc, "sigs": sigs},
        {"t": 1.0, "ev": "session_end", "duration_s": 1.0},
    ]
    d = tmp_path / f"{exp_id}_{len(sigs)}"
    d.mkdir()
    (d / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return d


def test_manipulation_check_uses_the_manifest_ablated_list(tmp_path):
    manifests = {
        "A1": {"id": "A1", "goals": {"npc_001": [{"nl": "Collect 1 @wheat", "condition": "has_item(wheat, 1)"}]}},
        "CO5": {"id": "CO5", "goals": {"npc_baker": [{"nl": "Bake 1 @bread", "condition": "has_item(bread, 1)"}]},
               "ablated_subplans": ["move_to_and_pickup", "craft_item", "obtain_from_peer", "collect_from_peer"]},
    }
    flags = {"SUB": True, "ATOM": False}

    def check(session_dir: Path):
        return AA.summarize_session(session_dir, manifests, ("SUB", "ATOM"), expected_flags=flags)["manipulation_ok"]

    # A1 (sin coordinación): collect_from_peer se carga sin usarse y no es de este experimento.
    assert check(_loaded_session(tmp_path, "A1", "npc_001", ["achieve_explore_zone", "collect_from_peer"])) is True
    # Escenario de coordinación que declara su lista: los macros de coordinación cuentan.
    assert check(_loaded_session(tmp_path, "CO5", "npc_baker", ["achieve_explore_zone"])) is True
    assert check(_loaded_session(tmp_path, "CO5", "npc_baker", ["achieve_explore_zone", "collect_from_peer"])) is False


def test_sub_and_atom_differ_only_in_builtin_subplans():
    data = json.loads((_EXPERIMENTS / "configs.json").read_text(encoding="utf-8"))
    sub, atom = data["SUB"]["env"], data["ATOM"]["env"]
    assert set(sub) == set(atom)
    assert {k for k in sub if sub[k] != atom[k]} == {"NPC_BUILTIN_SUBPLANS"}
    assert (sub["NPC_BUILTIN_SUBPLANS"], atom["NPC_BUILTIN_SUBPLANS"]) == ("1", "0")
    assert sub["NPC_PLAN_MEMORY_ENABLED"] == "0" and sub["NPC_CANONICAL_FAMILY"] == "0"


# ===========================================================================
# analyze_ablation
# ===========================================================================

def test_fisher_exact_reference_values():
    assert AA.fisher_exact_two_sided(3, 1, 1, 3) == pytest.approx(34 / 70)
    assert AA.fisher_exact_two_sided(4, 4, 4, 4) == pytest.approx(1.0)
    assert AA.fisher_exact_two_sided(0, 0, 0, 0) == 1.0


def test_wilson_all_successes():
    lo, hi = AA.wilson_ci(8, 8)
    assert hi == pytest.approx(1.0)
    assert 0.67 < lo < 0.68


def test_mann_whitney_separated_groups():
    u, p = AA.mann_whitney_u([1, 2, 3], [4, 5, 6])
    assert u == 0
    assert 0.05 < p < 0.1
    assert AA.mann_whitney_u([], [1]) is None


_MANIFESTS = {"A1": {"id": "A1", "goals": {"npc_001": [{"nl": "Collect 1 @wheat", "condition": "has_item(wheat, 1)"}]}}}
_FLAGS = {"SUB": True, "ATOM": False}


def _write_trace(session_dir: Path, events: list[dict], *, register: tuple[str, ...] = ("npc_001",)) -> Path:
    # Tras session_start, Unity registra los NPCs (RegisterNPC): el analizador lo exige.
    session_dir.mkdir(parents=True)
    t0 = events[0].get("t", 0.0) if events else 0.0
    registered = [{"t": t0, "ev": "unity_in", "npc": npc, "msg_type": "RegisterNPC"} for npc in register]
    body = events[:1] + registered + events[1:]
    (session_dir / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in body), encoding="utf-8")
    return session_dir


def _start(t: float, cfg: str) -> dict:
    return {"t": t, "ev": "session_start", "experiment_id": "A1", "config_label": cfg,
            "builtin_subplans": _FLAGS[cfg], "git_sha": "abc123"}


def _loaded(t: float, cfg: str) -> dict:
    sigs = ["achieve_explore_zone", "move_to_and_pickup", "craft_item"] if _FLAGS[cfg] else ["achieve_explore_zone"]
    return {"t": t, "ev": "builtin_plans_loaded", "npc": "npc_001", "sigs": sigs, "subplans_enabled": _FLAGS[cfg]}


def test_success_session(tmp_path):
    d = _write_trace(tmp_path / "s", [
        _start(0.0, "ATOM"), _loaded(1.0, "ATOM"),
        {"t": 2.0, "ev": "llm_call", "latency_s": 1.5},
        {"t": 3.0, "ev": "plan_requested", "npc": "npc_001"},
        {"t": 4.0, "ev": "action_sent", "npc": "npc_001", "action": "MoveTo"},
        {"t": 5.0, "ev": "action_result", "npc": "npc_001", "action": "MoveTo", "status": "Success"},
        {"t": 9.0, "ev": "goal_completed", "npc": "npc_001", "expected_condition": "has_item(wheat,1)",
         "goal_belief_met": True},
        {"t": 10.0, "ev": "shutdown_idle"},
        {"t": 10.5, "ev": "session_end", "duration_s": 10.5},
    ])
    s = AA.summarize_session(d, _MANIFESTS, ("SUB", "ATOM"), expected_flags=_FLAGS)
    assert s["success"] is True and s["first_plan_success"] is True
    assert s["t_success"] == 9.0 and s["llm_calls"] == 1 and s["n_act"] == 1
    assert s["manipulation_ok"] is True and s["primary_cause"] is None


def test_timeout_session_counts_as_failure(tmp_path):
    d = _write_trace(tmp_path / "s", [
        _start(0.0, "ATOM"), _loaded(1.0, "ATOM"),
        {"t": 5.0, "ev": "goal_replan_required", "npc": "npc_001", "reason": "ladder_stuck"},
        {"t": 6.0, "ev": "action_result", "npc": "npc_001", "action": "PickUp", "status": "Failure"},
        {"t": 300.0, "ev": "shutdown_timeout", "max_s": 300},
        {"t": 301.0, "ev": "session_end", "duration_s": 301.0},
    ])
    s = AA.summarize_session(d, _MANIFESTS, ("SUB", "ATOM"), expected_flags=_FLAGS)
    assert s["success"] is False and s["end_reason"] == "timeout"
    assert s["primary_cause"] == "timeout"
    assert "action_failures" in s["signals"]


def test_failed_after_replans_and_killed_sessions(tmp_path):
    failed = _write_trace(tmp_path / "f", [
        _start(0.0, "ATOM"),
        {"t": 50.0, "ev": "goal_failed_after_replans", "npc": "npc_001", "reason": "ladder_stuck"},
        {"t": 51.0, "ev": "shutdown_idle"},
        {"t": 51.5, "ev": "session_end", "duration_s": 51.5},
    ])
    s = AA.summarize_session(failed, _MANIFESTS, ("SUB", "ATOM"), expected_flags=_FLAGS)
    assert s["primary_cause"] == "failed_after_replans:ladder_stuck"
    assert s["manipulation_ok"] is None  # sin evento builtin_plans_loaded

    killed = _write_trace(tmp_path / "k", [_start(0.0, "SUB"), {"t": 3.0, "ev": "plan_requested"}])
    s = AA.summarize_session(killed, _MANIFESTS, ("SUB", "ATOM"), expected_flags=_FLAGS)
    assert s["success"] is False and s["end_reason"] == "killed" and s["primary_cause"] == "timeout"


def test_foreign_sessions_are_ignored(tmp_path):
    d = _write_trace(tmp_path / "e1", [
        {"t": 0.0, "ev": "session_start", "experiment_id": "E1", "config_label": "M0"},
    ])
    assert AA.summarize_session(d, _MANIFESTS, ("SUB", "ATOM")) is None


def test_analyze_compares_arms_and_filters_by_time(tmp_path):
    _write_trace(tmp_path / "20260915" / "100000", [
        _start(100.0, "SUB"), _loaded(101.0, "SUB"),
        {"t": 110.0, "ev": "goal_completed", "npc": "npc_001", "expected_condition": "has_item(wheat, 1)",
         "goal_belief_met": True},
        {"t": 111.0, "ev": "shutdown_idle"},
        {"t": 111.5, "ev": "session_end", "duration_s": 11.5},
    ])
    _write_trace(tmp_path / "20260915" / "100500", [
        _start(200.0, "ATOM"), _loaded(201.0, "ATOM"),
        {"t": 250.0, "ev": "goal_failed_after_replans", "npc": "npc_001", "reason": "ladder_stuck"},
        {"t": 251.0, "ev": "shutdown_idle"},
        {"t": 251.5, "ev": "session_end", "duration_s": 51.5},
    ])
    suite = {"id": "t", "experiments": ["A1"], "configs": ["SUB", "ATOM"], "config_builtin_subplans": _FLAGS}

    summaries, agg, comparisons = AA.analyze(tmp_path, suite, _MANIFESTS)
    assert len(summaries) == 2
    assert agg[("A1", "SUB")]["successes"] == 1 and agg[("A1", "ATOM")]["successes"] == 0
    assert all(s["manipulation_ok"] is True for s in summaries)
    assert comparisons[0]["experiment_id"] == "A1" and comparisons[0]["fisher_p"] == pytest.approx(1.0)

    md = AA.to_markdown(summaries, agg, comparisons, suite)
    assert "Fisher" in md and "failed_after_replans:ladder_stuck" in md

    later, _, _ = AA.analyze(tmp_path, suite, _MANIFESTS, since_epoch=150.0)
    assert [s["config_label"] for s in later] == ["ATOM"]

    AA.write_csv(summaries, agg, comparisons, tmp_path / "out")
    assert (tmp_path / "out" / "ablation_sessions.csv").exists()


def test_npc_not_registered_is_infra_error_and_excluded(tmp_path):
    # Caso real del piloto (2026-09-15): la build de coordinación generó npc_miller
    # y el goal era de npc_001 → el NPC no hizo nada y la sesión venció por tiempo.
    _write_trace(tmp_path / "20260915" / "153823", [
        _start(100.0, "ATOM"),
        {"t": 435.0, "ev": "shutdown_timeout", "max_s": 335},
        {"t": 435.5, "ev": "session_end", "duration_s": 335.5},
    ], register=("npc_miller",))
    _write_trace(tmp_path / "20260915" / "160000", [
        _start(500.0, "SUB"), _loaded(501.0, "SUB"),
        {"t": 510.0, "ev": "goal_completed", "npc": "npc_001", "expected_condition": "has_item(wheat, 1)",
         "goal_belief_met": True},
        {"t": 511.0, "ev": "shutdown_idle"},
        {"t": 511.5, "ev": "session_end", "duration_s": 11.5},
    ])
    suite = {"id": "t", "experiments": ["A1"], "configs": ["SUB", "ATOM"], "config_builtin_subplans": _FLAGS}

    summaries, agg, comparisons = AA.analyze(tmp_path, suite, _MANIFESTS)
    infra = [s for s in summaries if s["infra_error"]]
    assert len(infra) == 1
    assert infra[0]["primary_cause"] == "infra:npc_not_registered"
    assert infra[0]["missing_npcs"] == ["npc_001"]
    assert ("A1", "ATOM") not in agg and agg[("A1", "SUB")]["n"] == 1
    assert "EXCLUIDAS" in AA.to_markdown(summaries, agg, comparisons, suite)


def test_manifests_pin_the_right_unity_build():
    for exp_id in ("E1", "E2", "E3", "E4", "A1", "A2", "A3", "A4", "A5"):
        manifest = json.loads((_EXPERIMENTS / f"{exp_id}.json").read_text(encoding="utf-8"))
        assert manifest["build_exe"] == "builds\\single\\My project.exe", exp_id
    for exp_id in ("E5", "E6"):
        manifest = json.loads((_EXPERIMENTS / f"{exp_id}.json").read_text(encoding="utf-8"))
        assert manifest["build_exe"] == "builds\\coop\\My project.exe", exp_id


# ===========================================================================
# Fase 17i: peldaños con variables ligadas en el prompt de step3
# ===========================================================================

def test_bound_guard_facts_are_the_positive_literals_that_bind_the_variables():
    from llm.pipeline.pipeline_runner import _bound_guard_facts

    far = "not has_item(wheat, 1) & item_at(wheat, X, Y) & not current_position(X, Y)"
    here = "not has_item(wheat, 1) & item_at(wheat, X, Y) & current_position(X, Y)"
    peer = "not has_item(flour, 1) & peer_item_available(P, flour, Q, X, Y)"
    assert _bound_guard_facts(far, ["X", "Y"]) == ["item_at(wheat, X, Y)"]
    assert _bound_guard_facts(here, ["X", "Y"]) == ["item_at(wheat, X, Y)", "current_position(X, Y)"]
    assert _bound_guard_facts(peer, ["P", "Q", "X", "Y"]) == ["peer_item_available(P, flour, Q, X, Y)"]
    assert _bound_guard_facts("not has_item(wheat, 1) & not item_at(wheat, _, _)", []) == []


_RULE7 = (
    "7. If zone_center(ZoneTag, X, Y) is already known for the target zone, prefer MoveTo(X, Y) "
    "and Search(itemId); do NOT use ExploreArea(ZoneTag) in that case."
)


@pytest.mark.parametrize("builtin_subplans", [True, False])
def test_step3_prompt_explains_bound_variables_only_when_there_are_some(builtin_subplans: bool):
    # Sin variables ligadas (todo SUB en A1–A4): el prompt no cambia.
    plain_user, _ = build_prompt(_step3_payload(atomic_only=True, builtin_subplans=builtin_subplans))
    assert _RULE7 in plain_user
    # ("ZX, ZY" ya aparece en la descripción de Craft del catálogo: se mira la regla 7.)
    assert "ALREADY BOUND" not in plain_user and "prefer MoveTo(ZX, ZY)" not in plain_user

    user, _ = build_prompt(_step3_payload(
        atomic_only=True, builtin_subplans=builtin_subplans,
        neg_guard="not has_item(wheat, 1) & item_at(wheat, X, Y) & not current_position(X, Y)",
        unsatisfied_condition="not current_position(X, Y)",
        facts=[{"functor": "item_at", "args": ["wheat", "X", "Y"]}],
        bound_variables=["X", "Y"],
    ))
    assert "  Facts:\n  item_at(wheat, X, Y)\n" in user
    assert "Bound variables: X, Y\n  These variables are ALREADY BOUND" in user
    assert _RULE7 not in user and "prefer MoveTo(ZX, ZY)" in user
    assert "8. Avoid hardcoding raw coordinates. The bound variables (X, Y) are not hardcoded" in user
    assert "(bound variables are not invented: the branch guard binds them)" in user


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_gather_rungs_show_the_guard_facts_to_step3(tmp_path, enabled: bool):
    prompts: list[str] = []
    await run_full_pipeline(
        goal_sig="achieve_collect_wheat",
        npc_statement="Collect 1 wheat",
        existing_goals=["achieve_explore_zone"],
        llm_call=_capturing_llm(prompts),
        description="The NPC must collect 1 wheat.",
        success_condition="has_item(wheat, 1)",
        use_refinement=True,
        capability_contracts_path=str(tmp_path / "contracts.json"),
        entity_catalog=_CATALOG,
        beliefs={"zone_center": [["farmland", 9, -6]]},
        recipes=_RECIPES,
        item_spawns=_SPAWNS,
        builtin_subplans=enabled,
    )
    joined = "\n".join(prompts)
    # SUB: sin escalera de recolección ni variables ligadas → prompt como antes.
    assert ("  Facts:\n  item_at(wheat, X, Y)" in joined) is (not enabled)
    assert ("ALREADY BOUND" in joined) is (not enabled)
