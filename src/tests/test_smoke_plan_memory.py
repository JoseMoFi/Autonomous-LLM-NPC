from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx

from npc.plan_graph import GoalNode, NodeStatus, PlanVariant
from utils.plan_memory import (
    PlanMemory,
    build_plan_memory,
    resolve_run_root,
)


def _variants():
    return [{
        "guard": "true",
        "steps": [".wait(1)"],
        "full_asl": "+!achieve_wait : true <- .wait(1).",
    }]


def test_store_resets_counters_only_when_the_plan_changes(tmp_path):
    mem = PlanMemory("npc_001", tmp_path, promote_threshold=2, promote_min_rate=0.5)
    mem.store("achieve_bake_bread", _variants(), _variants()[0]["full_asl"])
    mem.record_failure("achieve_bake_bread")          # el plan antiguo falla y se replanifica
    # Mismo plan: se conservan los contadores.
    mem.store("achieve_bake_bread", _variants(), _variants()[0]["full_asl"])
    assert mem.load_pending("achieve_bake_bread")["uses_error"] == 1
    # Plan distinto (replan): contadores a cero, y su éxito lo hace reusable.
    new = [{"guard": "true", "steps": [".wait(2)"], "full_asl": "+!achieve_bake_bread : true <- .wait(2)."}]
    mem.store("achieve_bake_bread", new, new[0]["full_asl"])
    record = mem.load_pending("achieve_bake_bread")
    assert record["uses_error"] == 0 and record["uses_success"] == 0 and record["plan_revision"] == 1
    mem.record_success("achieve_bake_bread")
    assert [r["goal_sig"] for r in mem.load_all_reusable_pending()] == ["achieve_bake_bread"]


def _pm_settings(**over):
    base = {
        "plan_memory_enabled": True,
        "plan_memory_reuse": False,
        "plan_memory_dir": "",
        "plan_memory_approval_rate": 0.5,
        "plan_memory_min_uses": 2,
    }
    base.update(over)
    return SimpleNamespace(**base)


# ===========================================================================
# Fase 4 — enabled flag (kill-switch)
# ===========================================================================

def test_disabled_memory_is_noop(tmp_path):
    mem = PlanMemory("npc_x", tmp_path, enabled=False)
    mem.store("achieve_wait", _variants(), _variants()[0]["full_asl"])
    # No crea nada en disco ni registra.
    assert mem.record_success("achieve_wait") is False
    assert mem.load_approved("achieve_wait") is None
    assert mem.load_all_approved() == []
    assert not (tmp_path / "npc_x").exists()


# ===========================================================================
# Fase 4 — load_all_approved
# ===========================================================================

def test_load_all_approved_returns_only_approved(tmp_path):
    # min_uses=1, rate=0.5 → con 1 éxito se promueve.
    mem = PlanMemory("npc_x", tmp_path, promote_threshold=1, promote_min_rate=0.5)
    mem.store("achieve_approved", _variants(), _variants()[0]["full_asl"])
    mem.store("achieve_pending", _variants(), _variants()[0]["full_asl"])
    promoted = mem.record_success("achieve_approved")
    assert promoted is True

    approved = mem.load_all_approved()
    sigs = {r["goal_sig"] for r in approved}
    assert "achieve_approved" in sigs
    assert "achieve_pending" not in sigs


def test_load_all_reusable_pending_solo_con_evidencia(tmp_path):
    # Reuso cross-sesion: pending con 1 exito y sin errores SÍ es reusable
    # (umbral de promocion = 2 nunca se alcanza en una sesion corta).
    mem = PlanMemory("npc_x", tmp_path, promote_threshold=2, promote_min_rate=0.5)
    mem.store("achieve_ok", _variants(), _variants()[0]["full_asl"])
    mem.store("achieve_sin_uso", _variants(), _variants()[0]["full_asl"])
    mem.store("achieve_fallido", _variants(), _variants()[0]["full_asl"])
    mem.record_success("achieve_ok")          # 1 exito, sin errores → reusable
    mem.record_failure("achieve_fallido")     # tiene un error → NO reusable
    # achieve_sin_uso queda con 0 usos → NO reusable

    sigs = {r["goal_sig"] for r in mem.load_all_reusable_pending()}
    assert sigs == {"achieve_ok"}
    # ninguno se promovio (umbral 2) → approved sigue vacio
    assert mem.load_all_approved() == []


def test_store_persiste_y_restaura_call_args(tmp_path):
    # Identidad por binding: el plan guarda su call_args y build_goalnode lo
    # restaura, para distinguir bindings de la misma familia (bread vs wheat).
    mem = PlanMemory("npc_x", tmp_path, promote_threshold=1, promote_min_rate=0.5)
    mem.store("achieve_has_item", _variants(), _variants()[0]["full_asl"],
              param_names=["Item", "Qty"], call_args=["bread", 1])
    rec = mem.load_pending("achieve_has_item")
    assert rec["call_args"] == ["bread", 1]
    assert rec["param_names"] == ["Item", "Qty"]
    node = mem.build_goalnode("achieve_has_item", rec)
    assert node.call_args == ["bread", 1]
    assert node.param_names == ["Item", "Qty"]


def test_dos_bindings_misma_familia_no_colisionan(tmp_path):
    # Persistir bread y wheat (misma familia) bajo CLAVE de identidad → dos ficheros
    # separados, cada uno con su functor real. Antes (filename por sig) se pisaban.
    mem = PlanMemory("npc_x", tmp_path, promote_threshold=2, promote_min_rate=0.5)
    mem.store("achieve_has_item__bread_1", _variants(), _variants()[0]["full_asl"],
              call_args=["bread", 1], functor="achieve_has_item")
    mem.store("achieve_has_item__wheat_2", _variants(), _variants()[0]["full_asl"],
              call_args=["wheat", 2], functor="achieve_has_item")
    r_bread = mem.load_pending("achieve_has_item__bread_1")
    r_wheat = mem.load_pending("achieve_has_item__wheat_2")
    # los dos coexisten en disco (no se pisaron), cada uno con su functor real
    assert r_bread is not None and r_wheat is not None
    assert r_bread["call_args"] == ["bread", 1] and r_bread["functor"] == "achieve_has_item"
    assert r_wheat["call_args"] == ["wheat", 2] and r_wheat["functor"] == "achieve_has_item"


def test_promotion_uses_configured_policy(tmp_path):
    # política: min_uses=2, rate=0.5. Un éxito no basta; dos sí.
    s = _pm_settings(plan_memory_min_uses=2, plan_memory_approval_rate=0.5)
    mem = build_plan_memory("npc_x", s, tmp_path)
    mem.store("achieve_wait", _variants(), _variants()[0]["full_asl"])
    assert mem.record_success("achieve_wait") is False  # 1 uso
    assert mem.record_success("achieve_wait") is True    # 2 usos → promovido
    assert mem.is_approved("achieve_wait")


# ===========================================================================
# Fase 4 — build_plan_memory
# ===========================================================================

def test_build_plan_memory_disabled_when_root_none():
    mem = build_plan_memory("npc_x", _pm_settings(), None)
    assert mem.enabled is False


def test_build_plan_memory_uses_settings_policy(tmp_path):
    mem = build_plan_memory("npc_x", _pm_settings(plan_memory_min_uses=5,
                                                  plan_memory_approval_rate=0.9), tmp_path)
    assert mem.enabled is True
    assert mem.promote_threshold == 5
    assert mem.promote_min_rate == 0.9


# ===========================================================================
# Fase 4 — resolve_run_root (memoria por ejecución)
# ===========================================================================

def test_resolve_run_root_disabled(tmp_path):
    root, mode = resolve_run_root(_pm_settings(plan_memory_enabled=False), tmp_path)
    assert root is None
    assert mode == "disabled"


def test_resolve_run_root_new_creates_run(tmp_path):
    root, mode = resolve_run_root(_pm_settings(plan_memory_reuse=False), tmp_path)
    assert mode == "new"
    assert root is not None and root.exists()
    assert root.parent == tmp_path / "runs"


def test_resolve_run_root_reuse_explicit_dir(tmp_path):
    existing = tmp_path / "runs" / "20260101_000000"
    existing.mkdir(parents=True)
    root, mode = resolve_run_root(
        _pm_settings(plan_memory_reuse=True, plan_memory_dir=str(existing)), tmp_path)
    assert mode == "reuse"
    assert root == existing


def test_resolve_run_root_reuse_latest(tmp_path):
    runs = tmp_path / "runs"
    (runs / "20260101_000000").mkdir(parents=True)
    newer = runs / "20260601_000000"
    newer.mkdir(parents=True)
    root, mode = resolve_run_root(_pm_settings(plan_memory_reuse=True), tmp_path)
    assert mode == "reuse"
    assert root == newer


def test_resolve_run_root_reuse_no_runs_falls_back_to_new(tmp_path):
    root, mode = resolve_run_root(_pm_settings(plan_memory_reuse=True), tmp_path)
    assert mode == "new"
    assert root is not None and root.exists()


def test_resolve_run_root_reuse_missing_dir_falls_back_to_new(tmp_path):
    root, mode = resolve_run_root(
        _pm_settings(plan_memory_reuse=True, plan_memory_dir=str(tmp_path / "nope")),
        tmp_path)
    assert mode == "new"
    assert root is not None and root.exists()


# ===========================================================================
# Fase 4 — carga en el NPCAgent (from_memory)
# ===========================================================================

def test_agent_loads_approved_plans_as_from_memory(tmp_path):
    from npc.agent import NPCAgent

    # Preparar un run con un plan aprobado.
    seed = PlanMemory("npc_001", tmp_path, promote_threshold=1, promote_min_rate=0.5)
    seed.store("achieve_wait", _variants(), _variants()[0]["full_asl"], description="Wait.")
    assert seed.record_success("achieve_wait") is True  # promovido

    agent = NPCAgent(
        jid="npc_001@localhost", password="x", npc_id="npc_001",
        send_to_unity=lambda m: None, llm_planning_jid="llm@localhost",
        plan_memory_root=tmp_path,
    )
    agent._load_memory_plans()

    assert agent.plan_graph.has_node("achieve_wait")
    node = agent.plan_graph.nodes["achieve_wait"]["data"]
    assert node.from_memory is True
    assert node.status == NodeStatus.READY


def test_agent_disabled_memory_loads_nothing(tmp_path):
    from npc.agent import NPCAgent

    seed = PlanMemory("npc_001", tmp_path, promote_threshold=1, promote_min_rate=0.5)
    seed.store("achieve_wait", _variants(), _variants()[0]["full_asl"])
    seed.record_success("achieve_wait")

    # plan_memory_root=None → memoria desactivada.
    agent = NPCAgent(
        jid="npc_001@localhost", password="x", npc_id="npc_001",
        send_to_unity=lambda m: None, llm_planning_jid="llm@localhost",
        plan_memory_root=None,
    )
    agent._load_memory_plans()
    assert not agent.plan_graph.has_node("achieve_wait")


def test_plan_memory_store_writes_goal_asl_and_bundle(tmp_path) -> None:
    memory = PlanMemory("npc_test", tmp_path)
    variants = [
        {
            "guard": "true",
            "steps": [".wait(1)"],
            "full_asl": "+!achieve_wait : true <- .wait(1).",
        },
        {
            "guard": "has_item(wheat, N) & N >= 1",
            "steps": [".drop(wheat, 1)"],
            "full_asl": "+!achieve_wait : has_item(wheat, N) & N >= 1 <- .drop(wheat, 1).",
        },
    ]

    memory.store(
        goal_sig="achieve_wait",
        variants=variants,
        main_asl=variants[0]["full_asl"],
        description="Wait or drop.",
        param_names=[],
    )

    goal_text = memory.goal_asl_path("achieve_wait").read_text(encoding="utf-8")
    assert "+!achieve_wait : true <- .wait(1)." in goal_text
    assert "+!achieve_wait : has_item(wheat, N) & N >= 1 <- .drop(wheat, 1)." in goal_text

    bundle_text = memory.bundle_path.read_text(encoding="utf-8")
    assert "// goal: achieve_wait" in bundle_text
    assert "+!achieve_wait : true <- .wait(1)." in bundle_text

    record = json.loads((tmp_path / "npc_test" / "pending" / "achieve_wait.json").read_text(encoding="utf-8"))
    assert record["description"] == "Wait or drop."
    assert len(record["variants"]) == 2


def test_record_unverified_does_not_promote(tmp_path) -> None:
    """0.C5 — record_unverified cuenta aparte (uses_unverified) y NO toca
    uses_success ni promueve el plan a aprobado."""
    memory = PlanMemory("npc_test", tmp_path)
    memory.store(
        goal_sig="achieve_wait",
        variants=[{"guard": "true", "steps": [".wait(1)"],
                   "full_asl": "+!achieve_wait : true <- .wait(1)."}],
        main_asl="+!achieve_wait : true <- .wait(1).",
        description="Wait.",
        param_names=[],
    )
    for _ in range(10):
        memory.record_unverified("achieve_wait")

    record = json.loads(
        (tmp_path / "npc_test" / "pending" / "achieve_wait.json").read_text(encoding="utf-8")
    )
    assert record["uses_unverified"] == 10
    assert record["uses_success"] == 0
    assert record["status"] == "pending"  # nunca promovido


def test_plan_memory_rebuild_bundle_merges_disk_and_plan_graph(tmp_path) -> None:
    memory = PlanMemory("npc_test", tmp_path)
    memory.store(
        goal_sig="persisted_goal",
        variants=[
            {
                "guard": "true",
                "steps": ["true"],
                "full_asl": "+!persisted_goal : true <- true.",
            }
        ],
        main_asl="+!persisted_goal : true <- true.",
        param_names=[],
    )

    dag = nx.DiGraph()
    dag.add_node(
        "builtin_goal",
        data=GoalNode(
            sig="builtin_goal",
            variants=[
                PlanVariant(
                    guard="true",
                    steps=["true"],
                    full_asl="+!builtin_goal : true <- true.",
                )
            ],
            status=NodeStatus.READY,
            is_builtin=True,
        ),
    )

    memory.rebuild_bundle_from_graph(dag)

    bundle_text = memory.bundle_path.read_text(encoding="utf-8")
    assert "// goal: persisted_goal" in bundle_text
    assert "// goal: builtin_goal" in bundle_text
    assert "+!persisted_goal : true <- true." in bundle_text
    assert "+!builtin_goal : true <- true." in bundle_text


def test_plan_memory_reconstructs_variant_asl_when_full_asl_missing(tmp_path) -> None:
    memory = PlanMemory("npc_test", tmp_path)

    memory.store(
        goal_sig="move_to_and_pickup",
        variants=[
            {
                "guard": "item_at(ItemId, X, Y)",
                "steps": [".moveto(X, Y)", ".pickup(ItemId)"],
                "full_asl": "",
            }
        ],
        main_asl="",
        param_names=["ItemId"],
    )

    goal_text = memory.goal_asl_path("move_to_and_pickup").read_text(encoding="utf-8")
    assert "+!move_to_and_pickup(ItemId) : item_at(ItemId, X, Y) <-" in goal_text
    assert ".moveto(X, Y);" in goal_text
    assert ".pickup(ItemId)." in goal_text