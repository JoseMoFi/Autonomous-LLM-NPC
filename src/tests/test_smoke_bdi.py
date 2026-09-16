"""Smoke tests — BDI cycle (sin SPADE real ni LLM real).

Cubre los métodos del BDIBehaviour migrado a agentspeak.runtime:
  - compile_plan_with_actions + carga en asp_agent
  - _sync_beliefs: BeliefStore → asp_agent.beliefs
  - _inject_goal: agentspeak.call(+!goal)
  - _eval_guard: success_condition evaluation
  - full cycle: goal READY, plan compilado, asp.step(), goal completado
  - full cycle: .moveto → send_to_unity → ActionResultWaiter → completado
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import agentspeak
import agentspeak.stdlib  # noqa: F401
from agentspeak import runtime as asp_runtime
import networkx as nx
import pytest

from npc.asp_guard import compile_plan_with_actions
from npc.beliefs import BeliefStore
from npc.plan_graph import GoalNode, NodeStatus, PlanVariant
from npc.behaviours.bdi import BDIBehaviour, ActionResultWaiter
from npc.agent import Goal
from utils.plan_memory import PlanMemory


# ===========================================================================
# Mock agent helper
# ===========================================================================

class MockAgent:
    def __init__(self):
        self.npc_id = "npc_test"
        self.beliefs = BeliefStore()
        self.profile = MagicMock()  # profile presente → ciclo activo
        self.paused = False
        self.goals: list[Any] = []
        self.intention = None
        self.plan_graph = nx.DiGraph()
        self.action_results: dict[str, dict] = {}
        self.sent_commands: dict[str, dict] = {}
        self.completed_goal_count = 0
        self.phase = "idle"
        self._sent: list[dict] = []

    async def send_to_unity(self, msg: dict) -> None:
        self._sent.append(msg)

    async def push_status(self) -> None:
        await self.send_to_unity({"type": "GoalsUpdate", "npcId": self.npc_id})

    async def run_llm_task(self, task: dict) -> Any:
        return []


def _make_bdi() -> tuple[BDIBehaviour, MockAgent]:
    """Crea BDIBehaviour con asp engine inicializado (sin cargar builtin plans)."""
    agent = MockAgent()
    b = BDIBehaviour()
    b.agent = agent  # type: ignore[assignment]
    # Inicializar asp sin llamar on_start() (evita leer ficheros del disco)
    b._actions = agentspeak.Actions(agentspeak.stdlib.actions)
    b._env = asp_runtime.Environment()
    b._asp = b._env.build_agent(agentspeak.StringSource("<bdi>", ""), b._actions)
    b._pending_failure = None
    b._attempt_counts = {}
    b._register_unity_actions()
    return b, agent


def _inject_goal(agent: MockAgent, sig: str, priority: float = 1.0) -> Any:
    from npc.agent import Goal
    g = Goal(sig=sig, priority=priority)
    agent.goals.append(g)
    return g


def _load_plan(b: BDIBehaviour, asl_text: str) -> bool:
    """Compila y carga un plan ASL en b._asp. Devuelve True si tuvo éxito."""
    plan = compile_plan_with_actions(asl_text, b._actions)
    if plan is None:
        return False
    key = (plan.trigger, plan.goal_type, plan.head.functor, len(plan.head.args))
    b._asp.plans[key].append(plan)
    return True


def _setup_ready_plan(b: BDIBehaviour, agent: MockAgent, sig: str, asl_text: str) -> Any:
    """Configura goal con plan READY: carga en asp_agent Y en plan_graph.

    Esto simula el estado después de que _request_plan() completó y
    _load_plans_into_asp() se llamó. El ciclo run() puede ejecutar
    directamente sin llamar al LLM.
    """
    from npc.agent import Goal

    ok = _load_plan(b, asl_text)
    assert ok, f"No se pudo compilar: {asl_text[:60]}"

    variant = PlanVariant(guard="true", steps=[], full_asl=asl_text)
    node = GoalNode(sig=sig, status=NodeStatus.READY, variants=[variant])
    agent.plan_graph.add_node(sig, data=node)

    goal = Goal(sig=sig, priority=1.0)
    agent.goals.append(goal)
    agent.intention = goal
    return goal


# ===========================================================================
# compile_plan_with_actions
# ===========================================================================

class TestCanonicalHeadParams:
    """Fase 6.5: las cabezas aridad-0 del pipeline se parametrizan bajo reuso
    canónico para casar con el goal despachado `sig(args)`."""

    def test_full_asl_arity0_se_parametriza(self) -> None:
        b, _ = _make_bdi()
        node = GoalNode(sig="achieve_has_item", status=NodeStatus.READY,
                        param_names=["Item", "Qty"])
        variant = PlanVariant(
            guard="has_item(bread, 1)", steps=[],
            full_asl="+!achieve_has_item : has_item(bread, 1) <- .craft(bread_recipe, 1).",
        )
        asl = b._make_asl_from_variant(node, variant)
        assert asl.startswith("+!achieve_has_item(Item, Qty) :")
        # guard y cuerpo intactos (solo la cabeza cambia)
        assert "has_item(bread, 1) <- .craft(bread_recipe, 1)." in asl
        # y compila con la aridad correcta
        plan = compile_plan_with_actions(asl, b._actions)
        assert plan is not None and len(plan.head.args) == 2

    def test_sin_param_names_no_se_toca(self) -> None:
        b, _ = _make_bdi()
        node = GoalNode(sig="achieve_bake_bread", status=NodeStatus.READY)
        original = "+!achieve_bake_bread : true <- .craft(bread_recipe, 1)."
        variant = PlanVariant(guard="true", steps=[], full_asl=original)
        assert b._make_asl_from_variant(node, variant) == original

    def test_cabeza_ya_parametrizada_no_se_duplica(self) -> None:
        b, _ = _make_bdi()
        node = GoalNode(sig="achieve_has_item", status=NodeStatus.READY,
                        param_names=["Item", "Qty"])
        already = "+!achieve_has_item(Item, Qty) : item_spawn(Item, _) <- .moveto(1, 2)."
        variant = PlanVariant(guard="item_spawn(Item, _)", steps=[], full_asl=already)
        assert b._make_asl_from_variant(node, variant) == already

    def test_call_args_da_cabeza_ground(self) -> None:
        # Con call_args, la cabeza lleva el binding GROUND (no variables) → el plan
        # es exclusivo de su binding (coexistencia segura de bread y wheat).
        b, _ = _make_bdi()
        node = GoalNode(sig="achieve_has_item", status=NodeStatus.READY,
                        param_names=["Item", "Qty"], call_args=["bread", 1])
        variant = PlanVariant(
            guard="has_item(bread, 1)", steps=[],
            full_asl="+!achieve_has_item : has_item(bread, 1) <- .craft(bread_recipe, 1).",
        )
        asl = b._make_asl_from_variant(node, variant)
        assert asl.startswith("+!achieve_has_item(bread, 1) :")
        plan = compile_plan_with_actions(asl, b._actions)
        assert plan is not None and len(plan.head.args) == 2
        # un goal de otro binding (wheat) NO debe unificar con esta cabeza ground
        wnode = GoalNode(sig="achieve_has_item", status=NodeStatus.READY,
                         call_args=["wheat", 2])
        wasl = b._make_asl_from_variant(wnode, PlanVariant(
            guard="has_item(wheat, 2)", steps=[],
            full_asl="+!achieve_has_item : has_item(wheat, 2) <- .search(wheat)."))
        assert wasl.startswith("+!achieve_has_item(wheat, 2) :")


class TestCompilePlanWithActions:
    def test_simple_plan_compiles(self) -> None:
        actions = agentspeak.Actions(agentspeak.stdlib.actions)
        plan_text = "+!achieve_test : true <- true."
        plan = compile_plan_with_actions(plan_text, actions)
        assert plan is not None

    def test_plan_with_moveto_compiles(self) -> None:
        b, _ = _make_bdi()
        plan_text = "+!achieve_moveto : true <- .moveto(10, 20)."
        plan = compile_plan_with_actions(plan_text, b._actions)
        assert plan is not None
        assert plan.head.functor == "achieve_moveto"

    def test_plan_with_params_compiles(self) -> None:
        b, _ = _make_bdi()
        plan_text = "+!move_to_and_pickup(ItemId, N) : item_at(ItemId, X, Y) <- .moveto(X, Y); .pickup(ItemId)."
        plan = compile_plan_with_actions(plan_text, b._actions)
        assert plan is not None
        assert plan.head.functor == "move_to_and_pickup"
        assert len(plan.head.args) == 2

    def test_invalid_plan_returns_none(self) -> None:
        actions = agentspeak.Actions()
        plan = compile_plan_with_actions("this is not asl", actions)
        assert plan is None


@pytest.mark.asyncio
async def test_load_builtin_plans_uses_plan_graph_variants() -> None:
    b, agent = _make_bdi()
    node = GoalNode(
        sig="move_to_and_pickup",
        status=NodeStatus.READY,
        is_builtin=True,
        param_names=["ItemId", "N"],
        variants=[
            PlanVariant(
                guard="has_item(ItemId, Qty) & Qty >= N",
                steps=[],
                full_asl="+!move_to_and_pickup(ItemId, N) : has_item(ItemId, Qty) & Qty >= N <- true.",
            ),
            PlanVariant(
                guard="item_at(ItemId, X, Y)",
                steps=[".moveto(X, Y)", ".pickup(ItemId)"],
                full_asl="+!move_to_and_pickup(ItemId, N) : item_at(ItemId, X, Y) <- .moveto(X, Y); .pickup(ItemId).",
            ),
        ],
    )
    agent.plan_graph.add_node("move_to_and_pickup", data=node)

    await b._load_builtin_plans()

    loaded = sum(
        len(plans) for key, plans in b._asp.plans.items() if key[2] == "move_to_and_pickup"
    )
    assert loaded == 2


# ===========================================================================
# _sync_beliefs
# ===========================================================================

class TestSyncBeliefs:
    def test_empty_beliefs_clears_asp(self) -> None:
        b, agent = _make_bdi()
        # Manually add a belief then clear
        term = agentspeak.Literal("has_item", (agentspeak.Literal("wheat"), 3), frozenset())
        b._asp.beliefs[("has_item", 2)].add(term)
        agent.beliefs = BeliefStore()  # empty
        b._sync_beliefs()
        assert len(b._asp.beliefs) == 0

    def test_beliefs_transferred_correctly(self) -> None:
        b, agent = _make_bdi()
        agent.beliefs.apply_zone_discovery("farmland", 9, -6)
        b._sync_beliefs()
        # Should have zone_center(farmland, 9, -6) in asp beliefs
        assert ("zone_center", 3) in b._asp.beliefs
        bset = b._asp.beliefs[("zone_center", 3)]
        assert len(bset) == 1

    def test_inventory_belief_transferred(self) -> None:
        b, agent = _make_bdi()
        agent.beliefs.apply_inventory_update([{"itemId": "wheat", "qty": 3}])
        b._sync_beliefs()
        assert ("has_item", 2) in b._asp.beliefs


# ===========================================================================
# _inject_goal
# ===========================================================================

class TestInjectGoal:
    def test_inject_goal_with_no_plans_raises(self) -> None:
        b, agent = _make_bdi()
        from npc.agent import Goal
        goal = Goal(sig="achieve_nonexistent")
        with pytest.raises(agentspeak.AslError):
            b._inject_goal(goal)

    def test_inject_goal_with_plan_adds_intention(self) -> None:
        b, agent = _make_bdi()
        ok = _load_plan(b, "+!achieve_test : true <- true.")
        assert ok
        from npc.agent import Goal
        goal = Goal(sig="achieve_test")
        b._inject_goal(goal)
        assert len(b._asp.intentions) == 1

    def test_inject_goal_with_args(self) -> None:
        b, _ = _make_bdi()
        ok = _load_plan(b, "+!move_to_and_pickup(ItemId, N) : true <- true.")
        assert ok
        from npc.agent import Goal
        goal = Goal(sig="move_to_and_pickup", call_args=["wheat", 1])
        b._inject_goal(goal)
        assert len(b._asp.intentions) == 1


# ===========================================================================
# _eval_guard
# ===========================================================================

class TestEvalGuard:
    def test_true_string_always_passes(self) -> None:
        b, _ = _make_bdi()
        assert b._eval_guard("true") is True

    def test_empty_string_always_passes(self) -> None:
        b, _ = _make_bdi()
        assert b._eval_guard("") is True

    def test_not_true_fails(self) -> None:
        b, _ = _make_bdi()
        assert b._eval_guard("not (true)") is False

    def test_fact_present(self) -> None:
        b, agent = _make_bdi()
        agent.beliefs.apply_zone_discovery("farmland", 10, 20)
        assert b._eval_guard("zone_center(farmland, X, Y)") is True

    def test_fact_absent(self) -> None:
        b, _ = _make_bdi()
        assert b._eval_guard("zone_center(farmland, X, Y)") is False

    def test_negated_fact_absent(self) -> None:
        b, _ = _make_bdi()
        assert b._eval_guard("not (zone_center(farmland, X, Y))") is True

    def test_negated_fact_present(self) -> None:
        b, agent = _make_bdi()
        agent.beliefs.apply_zone_discovery("farmland", 10, 20)
        assert b._eval_guard("not (zone_center(farmland, X, Y))") is False

    def test_conjunction_both_true(self) -> None:
        b, agent = _make_bdi()
        agent.beliefs.apply_inventory_update([{"itemId": "wheat", "qty": 3}])
        assert b._eval_guard("has_item(wheat, N) & N >= 2") is True

    def test_conjunction_numeric_fails(self) -> None:
        b, agent = _make_bdi()
        agent.beliefs.apply_inventory_update([{"itemId": "wheat", "qty": 1}])
        assert b._eval_guard("has_item(wheat, N) & N >= 2") is False

    def test_conjunction_second_atom_absent(self) -> None:
        b, agent = _make_bdi()
        agent.beliefs.apply_zone_discovery("farmland", 10, 20)
        assert b._eval_guard("zone_center(farmland, X, Y) & delivery_point(tavern, DX, DY)") is False


# ===========================================================================
# Full BDI cycle — lifecycle tests (no Unity actions)
# ===========================================================================

@pytest.mark.asyncio
async def test_bdi_completes_goal_with_true_plan() -> None:
    """Un plan con body 'true' (agentspeak stdlib action) completa el goal."""
    b, agent = _make_bdi()
    goal = _setup_ready_plan(b, agent, "achieve_done", "+!achieve_done : true <- true.")

    # `true` body necesita ~5 pasos agentspeak para completarse
    for _ in range(10):
        if agent.completed_goal_count > 0:
            break
        await b.run()

    assert agent.completed_goal_count == 1
    assert agent.intention is None


@pytest.mark.asyncio
async def test_bdi_discards_failed_goal() -> None:
    """Un goal con plan FAILED se elimina de la lista."""
    b, agent = _make_bdi()
    node = GoalNode(sig="achieve_x", status=NodeStatus.FAILED)
    agent.plan_graph.add_node("achieve_x", data=node)
    goal = _inject_goal(agent, "achieve_x", priority=1.0)
    agent.intention = goal

    await b.run()

    assert not any(g.sig == "achieve_x" for g in agent.goals)
    assert agent.intention is None


@pytest.mark.asyncio
async def test_bdi_waits_when_generating() -> None:
    """Un goal cuyo plan está GENERATING debe esperar sin crash."""
    b, agent = _make_bdi()
    node = GoalNode(sig="achieve_x", status=NodeStatus.GENERATING)
    agent.plan_graph.add_node("achieve_x", data=node)
    goal = _inject_goal(agent, "achieve_x", priority=1.0)
    agent.intention = goal

    # Debe retornar sin lanzar excepciones
    await b.run()
    # El goal sigue activo
    assert any(g.sig == "achieve_x" for g in agent.goals)


@pytest.mark.asyncio
async def test_bdi_paused_does_not_execute() -> None:
    """Cuando paused=True, el ciclo duerme y no modifica el estado."""
    b, agent = _make_bdi()
    goal = _setup_ready_plan(b, agent, "achieve_x", "+!achieve_x : true <- true.")
    agent.paused = True

    await b.run()

    assert agent.completed_goal_count == 0


@pytest.mark.asyncio
async def test_bdi_selects_intention_when_none() -> None:
    """Sin intención activa, selecciona el goal con mayor prioridad."""
    b, agent = _make_bdi()
    _inject_goal(agent, "achieve_low", priority=0.3)
    _inject_goal(agent, "achieve_high", priority=0.9)
    agent.intention = None

    await b.run()

    assert agent.intention is not None
    assert agent.intention.sig == "achieve_high"


# ===========================================================================
# Full BDI cycle — Unity action (ActionResultWaiter)
# ===========================================================================

@pytest.mark.asyncio
async def test_bdi_fires_moveto_to_unity() -> None:
    """BDI engine llama send_to_unity cuando .moveto está en el body del plan."""
    b, agent = _make_bdi()
    _setup_ready_plan(
        b, agent, "achieve_moveto", "+!achieve_moveto : true <- .moveto(10, 20)."
    )

    # La ejecución del body necesita varios pasos asp para llegar a .moveto:
    # noop → push_query → next_or_fail(ejecuta acción → ensure_future → yield)
    for _ in range(5):
        await b.run()
        await asyncio.sleep(0)  # deja al event loop ejecutar ensure_future
        if any(m.get("actionType") == "MoveTo" for m in agent._sent):
            break

    moveto_cmds = [m for m in agent._sent if m.get("actionType") == "MoveTo"]
    assert len(moveto_cmds) == 1
    assert moveto_cmds[0]["args"]["x"] == 10
    assert moveto_cmds[0]["args"]["y"] == 20


@pytest.mark.asyncio
async def test_craft_restores_original_recipe_id_case_for_unity() -> None:
    """El comando Craft envía el recipe id con su case original del perfil
    (los beliefs lo guardan en minúscula; Unity lo registra como 'Bread_recipe')."""
    from types import SimpleNamespace
    b, agent = _make_bdi()
    agent.profile = SimpleNamespace(recipes=[SimpleNamespace(recipeId="Bread_recipe")])
    _setup_ready_plan(
        b, agent, "achieve_craft",
        "+!achieve_craft : true <- .craft(bread, bread_recipe, 1).",
    )
    for _ in range(5):
        await b.run()
        await asyncio.sleep(0)
        if any(m.get("actionType") == "Craft" for m in agent._sent):
            break
    craft_cmds = [m for m in agent._sent if m.get("actionType") == "Craft"]
    assert len(craft_cmds) == 1
    assert craft_cmds[0]["args"]["targetId"] == "Bread_recipe"


@pytest.mark.asyncio
async def test_bdi_completes_goal_after_unity_success() -> None:
    """Después de que Unity responde Success, el goal se completa."""
    b, agent = _make_bdi()
    _setup_ready_plan(
        b, agent, "achieve_moveto", "+!achieve_moveto : true <- .moveto(10, 20)."
    )

    # Avanzar hasta que .moveto sea enviado (necesita varios ciclos asp)
    for _ in range(5):
        await b.run()
        await asyncio.sleep(0)
        if any(m.get("type") == "ActionCommand" for m in agent._sent):
            break

    # Simular respuesta de Unity
    cmds = [m for m in agent._sent if m.get("type") == "ActionCommand"]
    assert len(cmds) == 1, f"Expected ActionCommand, got {agent._sent}"
    cmd_id = cmds[0]["commandId"]
    agent.action_results[cmd_id] = {"status": "Success", "commandId": cmd_id}

    # Ciclos siguientes: waiter se vacía, plan continúa, goal se completa
    for _ in range(10):
        if agent.completed_goal_count > 0:
            break
        await b.run()

    assert agent.completed_goal_count == 1
    assert agent.intention is None


@pytest.mark.asyncio
async def test_bdi_handles_unity_failure() -> None:
    """Unity Failure → plan falla → _handle_action_failure incrementa failure_count."""
    b, agent = _make_bdi()
    asl_text = "+!achieve_moveto : true <- .moveto(10, 20)."
    _setup_ready_plan(b, agent, "achieve_moveto", asl_text)
    # El nodo ya fue creado por _setup_ready_plan, obtenerlo
    node: GoalNode = agent.plan_graph.nodes["achieve_moveto"]["data"]
    goal = agent.intention

    # Avanzar hasta que .moveto sea enviado
    for _ in range(5):
        await b.run()
        await asyncio.sleep(0)
        if any(m.get("type") == "ActionCommand" for m in agent._sent):
            break

    # Simular respuesta de Unity con Failure
    cmds = [m for m in agent._sent if m.get("type") == "ActionCommand"]
    assert len(cmds) == 1
    cmd_id = cmds[0]["commandId"]
    agent.action_results[cmd_id] = {
        "status": "Failure", "errorCode": "item_not_found", "commandId": cmd_id
    }

    # Ciclo siguiente: waiter detecta Failure → plan falla → _handle_action_failure
    await b.run()

    assert node.failure_count >= 1

    # Ciclo 1: step() envía .moveto
    await b.run()
    await asyncio.sleep(0)

    # Simular respuesta de Unity con Failure
    cmds = [m for m in agent._sent if m.get("type") == "ActionCommand"]
    assert len(cmds) == 1
    cmd_id = cmds[0]["commandId"]
    agent.action_results[cmd_id] = {"status": "Failure", "errorCode": "item_not_found", "commandId": cmd_id}

    # Ciclo siguiente: waiter detecta Failure → plan falla → _handle_action_failure
    await b.run()

    assert node.failure_count >= 1


# ===========================================================================
# 0.C1 — Replan reutiliza el GoalNode (conserva contadores)
# ===========================================================================

@pytest.mark.asyncio
async def test_request_plan_reuses_goalnode_preserving_failure_count() -> None:
    b, agent = _make_bdi()
    agent.profile = None  # derive_entity_catalog → {}
    node = GoalNode(sig="achieve_x", status=NodeStatus.NEEDS_REPLAN)
    node.failure_count = 2
    node.failure_history = ["e1", "e2"]
    agent.plan_graph.add_node("achieve_x", data=node)

    goal = Goal(sig="achieve_x")
    with patch.object(agent, "run_llm_task", new_callable=AsyncMock, return_value={}):
        await b._request_plan(goal)

    # Mismo nodo (no recreado) y contadores conservados.
    assert agent.plan_graph.nodes["achieve_x"]["data"] is node
    assert node.failure_count == 2
    assert node.failure_history == ["e1", "e2"]


# ===========================================================================
# 0.C4 — Timeout del waiter configurable
# ===========================================================================

def test_waiter_respects_configured_timeout() -> None:
    b, agent = _make_bdi()

    class _Int:
        instr = object()

    intention = _Int()
    # timeout_s negativo = ya expirado (evita depender de la resolución del reloj).
    w = ActionResultWaiter(
        cmd_id="c", results={}, intention=intention, npc_id="npc_test",
        goal_sig="achieve_x", action_type="MoveTo", bdi=b, timeout_s=-1.0,
    )
    assert w.poll(None) is True
    assert b._pending_failure == "timeout"

    # Con timeout amplio y sin resultado disponible, sigue esperando (False).
    w2 = ActionResultWaiter(
        cmd_id="c2", results={}, intention=_Int(), npc_id="npc_test",
        goal_sig="achieve_x", action_type="MoveTo", bdi=b, timeout_s=999.0,
    )
    assert w2.poll(None) is False


# ===========================================================================
# 0.B2 — exhausted: contador de fallos de acciones observacionales
# ===========================================================================

def test_observational_failure_asserts_exhausted_at_budget() -> None:
    """Tras agotar el attempt_budget de Search (3), se assertea exhausted(Search, 3)."""
    b, agent = _make_bdi()
    # 2 fallos: aún no agotado
    b._note_observational_failure("achieve_have_wheat", "Search")
    b._note_observational_failure("achieve_have_wheat", "Search")
    assert not agent.beliefs.has("exhausted", "Search")
    # 3er fallo: agotado
    b._note_observational_failure("achieve_have_wheat", "Search")
    assert agent.beliefs.has("exhausted", "Search", 3)


def test_non_observational_action_never_exhausts() -> None:
    """MoveTo no es observacional (sin may_observe) → nunca assertea exhausted."""
    b, agent = _make_bdi()
    for _ in range(5):
        b._note_observational_failure("achieve_moveto", "MoveTo")
    assert not agent.beliefs.snapshot().get("exhausted")


def test_reset_attempt_counts_clears_exhausted() -> None:
    b, agent = _make_bdi()
    for _ in range(3):
        b._note_observational_failure("achieve_have_wheat", "Search")
    assert agent.beliefs.has("exhausted", "Search", 3)
    b._reset_attempt_counts("achieve_have_wheat")
    assert not agent.beliefs.snapshot().get("exhausted")
    # El contador se reinicia: hace falta otra tanda completa para re-agotar.
    b._note_observational_failure("achieve_have_wheat", "Search")
    assert not agent.beliefs.has("exhausted", "Search", 3)


def test_exhausted_belief_makes_guard_applicable() -> None:
    """Un guard exhausted(Search, 3) evalúa True una vez asertado el belief."""
    from npc.asp_guard import GuardEvaluator
    b, agent = _make_bdi()
    for _ in range(3):
        b._note_observational_failure("achieve_have_wheat", "Search")
    ok, _ = GuardEvaluator().eval_guard(
        "exhausted(Search, 3)", [], [], agent.beliefs.snapshot()
    )
    assert ok is True


@pytest.mark.asyncio
async def test_bdi_skips_explorearea_when_zone_center_already_known() -> None:
    b, agent = _make_bdi()
    agent.beliefs.apply_zone_discovery("farmland", 9, -6)
    _setup_ready_plan(
        b,
        agent,
        "achieve_explore_known_zone",
        "+!achieve_explore_known_zone : true <- .explorearea(farmland).",
    )

    for _ in range(5):
        await b.run()
        await asyncio.sleep(0)
        if agent.completed_goal_count > 0:
            break

    assert not any(m.get("actionType") == "ExploreArea" for m in agent._sent)
    assert agent.completed_goal_count == 1


@pytest.mark.asyncio
async def test_bdi_sends_explorearea_when_zone_center_unknown() -> None:
    b, agent = _make_bdi()
    _setup_ready_plan(
        b,
        agent,
        "achieve_explore_unknown_zone",
        "+!achieve_explore_unknown_zone : true <- .explorearea(farmland).",
    )

    for _ in range(5):
        await b.run()
        await asyncio.sleep(0)
        if any(m.get("actionType") == "ExploreArea" for m in agent._sent):
            break

    explore_cmds = [m for m in agent._sent if m.get("actionType") == "ExploreArea"]
    assert len(explore_cmds) == 1
    assert explore_cmds[0]["args"]["zoneTag"] == "farmland"


# ===========================================================================
# Variable resolution via agentspeak unification
# ===========================================================================

@pytest.mark.asyncio
async def test_bdi_resolves_variables_from_beliefs() -> None:
    """Agentspeak resuelve ZX, ZY desde beliefs (no Python string matching)."""
    b, agent = _make_bdi()
    agent.beliefs.apply_zone_discovery("farmland", 9, -6)

    # Plan que usa variables ligadas por el guard: ZX, ZY de zone_center
    plan_text = (
        "+!navigate_to_farm : zone_center(farmland, ZX, ZY) <- .moveto(ZX, ZY)."
    )
    _setup_ready_plan(b, agent, "navigate_to_farm", plan_text)

    # Avanzar hasta que .moveto sea enviado
    for _ in range(5):
        await b.run()
        await asyncio.sleep(0)
        if any(m.get("actionType") == "MoveTo" for m in agent._sent):
            break

    moveto_cmds = [m for m in agent._sent if m.get("actionType") == "MoveTo"]
    assert len(moveto_cmds) == 1
    # ZX=9, ZY=-6 deben llegar resueltos, NO como strings "ZX", "ZY"
    assert moveto_cmds[0]["args"]["x"] == 9
    assert moveto_cmds[0]["args"]["y"] == -6


# ===========================================================================
# _request_plan — processing a synthetic PipelineResult dict
# ===========================================================================

@pytest.mark.asyncio
async def test_request_plan_builds_variants_from_pipeline_result(tmp_path) -> None:
    """_request_plan convierte un PipelineResult dict en GoalNode READY con variantes."""
    b, agent = _make_bdi()

    pipeline_result = {
        "sig": "achieve_craft_bread",
        "description": "Craft item_beta.",
        "variants": [
            {
                "guard": "has_item(item_alpha, N) & N >= 2",
                "steps": [],
                "asl": "+!achieve_craft_bread : has_item(item_alpha, N) & N >= 2 <- true.",
            },
            {
                "guard": "not has_item(item_alpha, N)",
                "steps": [{"type": "subgoal", "name": "achieve_collect_item_alpha", "args": []}],
                "asl": "+!achieve_craft_bread : not has_item(item_alpha, N) <- !achieve_collect_item_alpha .",
            }
        ],
        "subgoals_to_expand": [{"sig": "achieve_collect_item_alpha", "description": "Collect item_alpha."}],
        "dag": {
            "nodes": [
                {"id": "achieve_craft_bread", "status": "main"},
                {"id": "achieve_collect_item_alpha", "status": "pending"},
            ],
            "edges": [["achieve_craft_bread", "achieve_collect_item_alpha"]],
        },
    }
    agent.plan_memory = PlanMemory("npc_test", tmp_path)

    async def fake_llm(task: dict):
        return pipeline_result

    agent.run_llm_task = fake_llm  # type: ignore

    from npc.agent import Goal
    goal = Goal(sig="achieve_craft_bread")
    await b._request_plan(goal)

    assert agent.plan_graph.has_node("achieve_craft_bread")
    node: GoalNode = agent.plan_graph.nodes["achieve_craft_bread"]["data"]
    assert node.status == NodeStatus.READY
    assert len(node.variants) == 2
    assert node.variants[0].guard == "has_item(item_alpha, N) & N >= 2"
    # Sub-goal encolado
    assert any(g.sig == "achieve_collect_item_alpha" for g in agent.goals)
    # Plan cargado en asp_agent
    assert b._is_plan_loaded("achieve_craft_bread")
    assert agent.plan_memory.goal_asl_path("achieve_craft_bread").exists()
    bundle_text = agent.plan_memory.bundle_path.read_text(encoding="utf-8")
    assert "achieve_craft_bread" in bundle_text
    assert "+!achieve_craft_bread : has_item(item_alpha, N) & N >= 2 <- true." in bundle_text


@pytest.mark.asyncio
async def test_request_plan_fails_on_empty_result() -> None:
    """Cuando run_llm_task devuelve un dict sin 'variants' ni 'steps', el node queda en FAILED."""
    b, agent = _make_bdi()

    async def fake_llm(task: dict):
        return {"sig": "x"}

    agent.run_llm_task = fake_llm  # type: ignore

    from npc.agent import Goal
    goal = Goal(sig="achieve_x")
    await b._request_plan(goal)

    node: GoalNode = agent.plan_graph.nodes["achieve_x"]["data"]
    assert node.status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_request_plan_fails_on_sig_mismatch() -> None:
    """Si el pipeline devuelve un sig distinto al goal solicitado, falla seguro."""
    b, agent = _make_bdi()

    async def fake_llm(task: dict):
        return {
            "sig": "achieve_other_goal",
            "description": "Wrong sig.",
            "variants": [
                {"guard": "true", "steps": [], "asl": "+!achieve_other_goal : true <- true."}
            ],
        }

    agent.run_llm_task = fake_llm  # type: ignore

    from npc.agent import Goal
    goal = Goal(sig="achieve_expected_goal")
    await b._request_plan(goal)

    node: GoalNode = agent.plan_graph.nodes["achieve_expected_goal"]["data"]
    assert node.status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_no_applicable_variant_fuse_marks_goal_failed() -> None:
    """El fusible de no-variant debe marcar FAILED y soltar la intención."""
    b, agent = _make_bdi()

    from npc.agent import Goal
    goal = Goal(sig="achieve_have_wheat")
    agent.intention = goal
    node = GoalNode(sig=goal.sig, status=NodeStatus.READY, variants=[])
    agent.plan_graph.add_node(goal.sig, data=node)

    goal.no_variant_hits = 19
    await b._handle_no_applicable_variant(goal)

    assert node.status == NodeStatus.FAILED
    assert agent.intention is None


# ===========================================================================
# Escalera de variantes: avanzar re-inyectando, no replanificar (cambio BDI)
# ===========================================================================

def _ladder_node(sig="achieve_x"):
    node = GoalNode(sig=sig, status=NodeStatus.READY)
    node.variants = [
        PlanVariant(guard="has_item(x, 1)", steps=[],
                    full_asl=f"+!{sig} : has_item(x, 1) <- true."),
    ]
    return node


def test_trigger_replan_within_budget_sets_needs_replan():
    b, agent = _make_bdi()
    g = Goal(sig="achieve_x"); g.success_condition = "has_item(x, 1)"
    agent.goals = [g]; agent.intention = g
    node = _ladder_node()
    agent.plan_graph.add_node("achieve_x", data=node)
    asyncio.run(b._trigger_replan_or_fail(g, node, reason="test"))
    assert g.replan_count == 1
    assert node.status == NodeStatus.NEEDS_REPLAN
    assert agent.intention is None


def test_trigger_replan_exhausted_marks_failed():
    b, agent = _make_bdi()
    g = Goal(sig="achieve_x"); g.success_condition = "has_item(x, 1)"; g.replan_count = 3
    agent.goals = [g]; agent.intention = g
    node = _ladder_node()
    agent.plan_graph.add_node("achieve_x", data=node)
    asyncio.run(b._trigger_replan_or_fail(g, node, reason="test"))
    assert node.status == NodeStatus.FAILED
    assert all(x.sig != "achieve_x" for x in agent.goals)


def test_ladder_stuck_guard_forces_replan_after_no_progress():
    b, agent = _make_bdi()
    g = Goal(sig="achieve_x"); g.success_condition = "has_item(x, 1)"
    agent.goals = [g]; agent.intention = g
    node = _ladder_node()
    agent.plan_graph.add_node("achieve_x", data=node)
    # Mismas beliefs (vacías) repetidas: baseline + 3 sin progreso → replan.
    for _ in range(4):
        asyncio.run(b._note_ladder_progress(g, node))
    assert g.replan_count == 1
    assert node.status == NodeStatus.NEEDS_REPLAN


def test_ladder_progress_resets_counter_on_belief_change():
    b, agent = _make_bdi()
    g = Goal(sig="achieve_x"); g.success_condition = "has_item(x, 1)"
    agent.goals = [g]; agent.intention = g
    node = _ladder_node()
    agent.plan_graph.add_node("achieve_x", data=node)
    asyncio.run(b._note_ladder_progress(g, node))
    asyncio.run(b._note_ladder_progress(g, node))
    agent.beliefs.apply_has_item("wheat", 1)  # progreso → resetea el contador
    asyncio.run(b._note_ladder_progress(g, node))
    assert node.status == NodeStatus.READY
    assert g.replan_count == 0


def test_catch_all_variant_loaded_for_main_goal():
    b, agent = _make_bdi()
    node = _ladder_node()
    b._load_plans_into_asp(node)
    plans = [p for key, plist in b._asp.plans.items()
             if key[2] == "achieve_x" for p in plist]
    assert len(plans) >= 2  # la variante real + el catch-all (: true)


def test_catch_all_not_loaded_for_builtin():
    b, agent = _make_bdi()
    node = GoalNode(sig="move_to_and_pickup", status=NodeStatus.READY, is_builtin=True)
    node.variants = [PlanVariant(guard="true", steps=[],
                                 full_asl="+!move_to_and_pickup(I, N) : true <- .search(I).")]
    b._load_plans_into_asp(node)
    plans = [p for key, plist in b._asp.plans.items()
             if key[2] == "move_to_and_pickup" for p in plist]
    assert len(plans) == 1  # sin catch-all para builtins
