"""Integración — Fase 14, escenario E6 (cooperación bidireccional CON arbitraje).

Mismo patrón que `test_integration_peer_delegation.py` (E5: bus falso +
mundo falso, agentspeak/BDI/PeerCoordBehaviour REALES, sin mocks de la
lógica de coordinación) pero con el escenario que en Fase 13 quedó
caracterizado como límite arquitectónico (0/5 real, ver
`DOC/Evaluacion/EVALUACION_RESULTADOS.md` §5): `npc_miller` tiene el ÚNICO
goal (`has_item(bread,1)`) pero solo sabe hacer flour; `npc_baker` sabe hacer
bread desde flour pero no tiene goal propio — así que cuando baker acepta el
pedido de bread y a su vez le pide flour a miller (depth=2), miller queda con
DOS goals: el suyo (bloqueado en `.await_peer` a baker) y el que acaba de
aceptar de baker (`achieve_deliver_to_peer`, sin arrancar). Sin arbitraje
(Fase 13) esto no converge dentro del presupuesto de reintentos. Este test
prueba que CON arbitraje (modo "rule", determinista — no depende de un LLM
real) SÍ converge, en un puñado de ticks.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from npc.agent import NPCAgent, Goal
from npc.behaviours.bdi import BDIBehaviour
from npc.behaviours.peer_coord import PeerCoordBehaviour
from npc.builtin_loader import load_builtin_plans
from protocol.messages import RecipePayload, RecipeItemPayload, ItemSpawnSummary

BAKERI = (0, 0)


class _Bus:
    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Queue] = {}

    def register(self, npc_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.queues[npc_id] = q
        return q

    async def deliver(self, msg) -> None:
        to_npc = str(msg.to).split("@")[0]
        q = self.queues.get(to_npc)
        if q is not None:
            await q.put(msg)


def _wire_transport(behaviour, npc_id: str, bus: _Bus, queue: asyncio.Queue) -> None:
    async def _send(msg) -> None:
        msg.sender = f"{npc_id}@localhost"
        await bus.deliver(msg)

    async def _receive(timeout=None):
        try:
            return queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    behaviour.send = _send  # type: ignore[method-assign]
    behaviour.receive = _receive  # type: ignore[method-assign]


class _FakeWorld:
    def make_executor(self, agent):
        async def _send_to_unity(msg: dict) -> None:
            if msg.get("type") != "ActionCommand":
                return
            action = msg["actionType"]
            args = msg.get("args", {})
            cmd_id = msg["commandId"]

            if action == "MoveTo":
                x, y = int(args["x"]), int(args["y"])
                agent.beliefs.apply_current_position(x, y)
                if (x, y) == BAKERI:
                    agent.beliefs.apply_at_zone("bakeri")
                else:
                    agent.beliefs.remove_at_zone("bakeri")
                agent.action_results[cmd_id] = {"status": "Success", "payload": {}}

            elif action == "PickUp":
                item_id = args["itemId"]
                rows = agent.beliefs.query("has_item", item_id)
                have = rows[0][1] if rows else 0
                agent.beliefs.apply_has_item(item_id, have + 1)
                agent.action_results[cmd_id] = {"status": "Success", "payload": {}}

            elif action == "Craft":
                recipe = _find_recipe(agent.profile.recipes, args["targetId"])
                if recipe is None:
                    agent.action_results[cmd_id] = {"status": "Failure", "errorCode": "no_recipe"}
                    return
                inp, outp = recipe.inputs[0], recipe.outputs[0]
                rows = agent.beliefs.query("has_item", inp.itemId.lower())
                have = rows[0][1] if rows else 0
                if have < inp.qty:
                    agent.action_results[cmd_id] = {"status": "Failure", "errorCode": "missing_items"}
                    return
                agent.beliefs.apply_has_item(inp.itemId.lower(), have - inp.qty)
                out_rows = agent.beliefs.query("has_item", outp.itemId.lower())
                out_have = out_rows[0][1] if out_rows else 0
                agent.beliefs.apply_has_item(outp.itemId.lower(), out_have + outp.qty)
                agent.action_results[cmd_id] = {"status": "Success", "payload": {}}

            elif action == "Drop":
                item_id, qty = args["itemId"], int(args.get("qty", 1))
                rows = agent.beliefs.query("has_item", item_id)
                have = rows[0][1] if rows else 0
                agent.beliefs.apply_has_item(item_id, max(0, have - qty))
                agent.action_results[cmd_id] = {"status": "Success", "payload": {}}

            else:
                agent.action_results[cmd_id] = {"status": "Failure", "errorCode": "unsupported_in_fake_world"}

        return _send_to_unity


def _find_recipe(recipes, target: str):
    t = str(target).lower()
    for r in recipes:
        if r.recipeId.lower() == t:
            return r
    return None


async def _make_agent(npc_id: str, recipes: list[RecipePayload], item_spawns: list[str],
                       bus: _Bus, world: _FakeWorld) -> tuple[NPCAgent, BDIBehaviour, PeerCoordBehaviour]:
    agent = NPCAgent(
        jid=f"{npc_id}@localhost", password="x", npc_id=npc_id,
        send_to_unity=None, llm_planning_jid="llm@localhost",
    )
    agent.send_to_unity = world.make_executor(agent)
    profile_spawns = [
        ItemSpawnSummary(itemId=i, zones=["farmland"], targetCount=1, weight=1)
        for i in item_spawns
    ]
    agent.profile = type("P", (), {"recipes": recipes, "item_spawns": profile_spawns})()
    agent.beliefs.npc_id = npc_id
    for r in recipes:
        agent.beliefs.apply_recipe(
            r.recipeId, r.zone,
            [{"itemId": i.itemId, "qty": i.qty} for i in r.inputs],
            [{"itemId": o.itemId, "qty": o.qty} for o in r.outputs],
        )
    for item_id in item_spawns:
        agent.beliefs.apply_item_spawn(item_id, "farmland")
        agent.beliefs.apply_item_at(item_id, 10, 10, current_tick=0)
    agent.beliefs.apply_zone_discovery("bakeri", BAKERI[0], BAKERI[1])

    from pathlib import Path
    plans_dir = Path(__file__).resolve().parent.parent / "plans" / "builtin"
    load_builtin_plans(agent.plan_graph, plans_dir)

    bdi = BDIBehaviour()
    bdi.agent = agent
    import agentspeak
    import agentspeak.stdlib  # noqa: F401
    from agentspeak import runtime as asp_runtime
    bdi._actions = agentspeak.Actions(agentspeak.stdlib.actions)
    bdi._env = asp_runtime.Environment()
    bdi._asp = bdi._env.build_agent(agentspeak.StringSource("<bdi>", ""), bdi._actions)
    bdi._pending_failure = None
    bdi._attempt_counts = {}
    bdi._arbitration_switch_count = 0
    bdi._last_arbitration_ts = 0.0
    bdi._register_unity_actions()
    bdi._register_peer_actions()
    await bdi._load_builtin_plans()

    peer_q = bus.register(npc_id)
    peer = PeerCoordBehaviour()
    peer.agent = agent
    _wire_transport(peer, npc_id, bus, peer_q)
    _wire_transport(bdi, npc_id, bus, peer_q)

    return agent, bdi, peer


async def _run_until(agents_bdi_peer, condition, *, max_ticks: int = 800) -> None:
    for _ in range(max_ticks):
        if condition():
            return
        for _agent, bdi, peer in agents_bdi_peer:
            await bdi.run()
            await peer.run()
        await asyncio.sleep(0)
    raise AssertionError(f"condición no alcanzada tras {max_ticks} ticks")


@pytest.mark.asyncio
async def test_e6_bidirectional_cooperation_converges_with_rule_arbitration():
    bus = _Bus()
    world = _FakeWorld()

    flour_recipe = RecipePayload(
        recipeId="Flour_recipe", zone="bakeri",
        inputs=[RecipeItemPayload(itemId="wheat", qty=2)],
        outputs=[RecipeItemPayload(itemId="flour", qty=1)],
    )
    bread_recipe = RecipePayload(
        recipeId="Bread_from_flour_recipe", zone="bakeri",
        inputs=[RecipeItemPayload(itemId="flour", qty=1)],
        outputs=[RecipeItemPayload(itemId="bread", qty=1)],
    )

    # E6 (invertido respecto a E5): miller tiene el goal, baker no.
    miller_agent, miller_bdi, miller_peer = await _make_agent(
        "npc_miller", [flour_recipe], ["wheat"], bus, world,
    )
    baker_agent, baker_bdi, baker_peer = await _make_agent(
        "npc_baker", [bread_recipe], [], bus, world,
    )

    miller_agent.beliefs.apply_peer("npc_baker", "baker")
    baker_agent.beliefs.apply_peer("npc_miller", "miller")

    miller_goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    miller_goal.success_condition = "has_item(bread, 1)"
    miller_agent.goals.append(miller_goal)

    with patch("config.settings.coordination_enabled", True), \
         patch("config.settings.canonical_reuse_enabled", True), \
         patch("config.settings.canonical_family_plan", True), \
         patch("config.settings.peer_max_depth", 2), \
         patch("config.settings.peer_request_timeout_s", 3.0), \
         patch("config.settings.peer_query_timeout_s", 3.0), \
         patch("config.settings.action_timeout_s", 3.0), \
         patch("config.settings.goal_arbitration_enabled", True), \
         patch("config.settings.goal_arbitration_mode", "rule"), \
         patch("config.settings.arbitration_cooldown_s", 0.0):

        agents = [
            (miller_agent, miller_bdi, miller_peer),
            (baker_agent, baker_bdi, baker_peer),
        ]
        await _run_until(
            agents,
            lambda: miller_agent.beliefs.has("has_item", "bread", 1) and not miller_agent.goals,
        )

    # --- Resultado: el GOAL DEL DUEÑO (miller) cierra con el pan de vuelta ---
    assert miller_agent.beliefs.has("has_item", "bread", 1)
    assert miller_agent.goals == []
    assert miller_agent.intention is None

    # --- Mecanismo: el arbitraje SÍ intervino (sin él, Fase 13 mostró que
    # esto no converge -- ver EVALUACION_RESULTADOS.md §5) ---
    assert miller_bdi._arbitration_switch_count >= 1

    # --- Intercambio bidireccional real: baker recibió flour, miller bread ---
    baker_flour = baker_agent.beliefs.query("has_item", "flour")
    assert not baker_flour or baker_flour[0][1] == 0  # la consumió crafteando bread
    assert baker_agent.beliefs.has("peer_done", "npc_miller", "achieve_has_item")
    assert miller_agent.beliefs.has("peer_done", "npc_baker", "achieve_has_item")
