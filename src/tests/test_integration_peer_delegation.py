"""Integración — Fase 12, escenario E5 (delegación simple, miller→baker).

Dos `NPCAgent` REALES (motor `agentspeak` real, `BDIBehaviour` +
`PeerCoordBehaviour` reales, sin mocks de la lógica de coordinación), con:

  - Bus de mensajes FALSO en vez de XMPP real (`_Bus`) — enruta
    `Message.send()` entre las dos `PeerCoordBehaviour`/`BDIBehaviour` por
    npc_id. La FORMA del mensaje (JSON, metadata, performativas) es la real;
    solo el transporte de red se sustituye.
  - Mundo FALSO en vez de Unity (`_FakeWorld`) — intercepta
    `agent.send_to_unity` y resuelve MoveTo/PickUp/Craft/Drop contra un
    estado mínimo (inventario, posición, recetas del perfil), escribiendo
    `agent.action_results` con el mismo contrato que lee
    `ActionResultWaiter` en producción.

Por qué esto y no Unity+Ollama real: el escenario E5 necesita un ítem
craft-only nuevo (`flour`) cuyo prefab + `ItemSpawnRule` en el Editor de
Unity siguen pendientes (ver `PLAN_EJECUCION/FASE_11_MULTIAGENTE_INDEPENDIENTE.md`
§"T10 — Resultado" y `FASE_13_BATERIA_EXPERIMENTAL.md` §T0) — no es seguro
completarlos por edición de texto. Este test ejercita TODO el código Python
nuevo de la Fase 12 de extremo a extremo (family_plan `delegate`,
PeerCoordBehaviour, las 4 acciones ASL de coordinación, `achieve_deliver_to_peer`,
`collect_from_peer`) sin necesitar el Editor ni Ollama — es la validación más
fuerte posible mientras ese prerrequisito siga abierto.
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


# ---------------------------------------------------------------------------
# Bus de mensajes (sustituye XMPP)
# ---------------------------------------------------------------------------

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
        # No-bloqueante SIEMPRE, ignorando el `timeout` que pida el llamante
        # (PeerCoordBehaviour.run() pide 2s reales — en este test el bucle
        # `_run_until` ya alterna agentes explícitamente; esperar de verdad
        # aquí multiplicaría el tiempo de test por nada). "Nada en cola todavía"
        # -> None inmediato, como un timeout expirado al instante.
        try:
            return queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    behaviour.send = _send  # type: ignore[method-assign]
    behaviour.receive = _receive  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Mundo falso (sustituye Unity)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Construcción de agentes
# ---------------------------------------------------------------------------

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
    # zone_center de bakeri conocido desde el arranque (en producción llega
    # con el NPCProfile, global — ver hallazgo Fase 11). Sin esto craft_item.asl
    # cae en su Rule C3 (.explorearea) que el mundo falso no implementa.
    agent.beliefs.apply_zone_discovery("bakeri", BAKERI[0], BAKERI[1])

    _builtin_dir = load_builtin_plans.__module__  # placeholder to keep import used
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
    bdi._register_unity_actions()
    bdi._register_peer_actions()
    await bdi._load_builtin_plans()

    peer_q = bus.register(npc_id)
    peer = PeerCoordBehaviour()
    peer.agent = agent
    _wire_transport(peer, npc_id, bus, peer_q)
    _wire_transport(bdi, npc_id, bus, peer_q)  # las acciones ASL también envían por self.send

    return agent, bdi, peer


async def _run_until(agents_bdi_peer, condition, *, max_ticks: int = 400) -> None:
    """Alterna un tick de BDI y un intento de recepción de PeerCoordBehaviour
    por cada agente, hasta que `condition()` sea True o se agoten los ticks."""
    for _ in range(max_ticks):
        if condition():
            return
        for _agent, bdi, peer in agents_bdi_peer:
            await bdi.run()
            await peer.run()  # timeout corto (0.05s) — no bloquea si no hay mensaje
        await asyncio.sleep(0)
    raise AssertionError(f"condición no alcanzada tras {max_ticks} ticks")


# ---------------------------------------------------------------------------
# El test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e5_delegation_miller_bakes_flour_for_baker():
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

    miller_agent, miller_bdi, miller_peer = await _make_agent(
        "npc_miller", [flour_recipe], ["wheat"], bus, world,
    )
    baker_agent, baker_bdi, baker_peer = await _make_agent(
        "npc_baker", [bread_recipe], [], bus, world,
    )

    # Directorio de peers (Fase 11 T6) — sembrado por NPCRegistry en producción.
    miller_agent.beliefs.apply_peer("npc_baker", "baker")
    baker_agent.beliefs.apply_peer("npc_miller", "miller")

    # Goal del baker: has_item(bread, 1). Se inyecta directamente (sin
    # parse_goals/LLM) — este test valida coordinación, no el pipeline NL.
    baker_goal = Goal(sig="achieve_has_item", call_args=["bread", 1])
    baker_goal.success_condition = "has_item(bread, 1)"
    baker_agent.goals.append(baker_goal)

    with patch("config.settings.coordination_enabled", True), \
         patch("config.settings.canonical_reuse_enabled", True), \
         patch("config.settings.canonical_family_plan", True), \
         patch("config.settings.peer_max_depth", 2), \
         patch("config.settings.peer_request_timeout_s", 3.0), \
         patch("config.settings.peer_query_timeout_s", 3.0), \
         patch("config.settings.action_timeout_s", 3.0):

        agents = [
            (miller_agent, miller_bdi, miller_peer),
            (baker_agent, baker_bdi, baker_peer),
        ]
        await _run_until(
            agents,
            lambda: baker_agent.beliefs.has("has_item", "bread", 1) and not baker_agent.goals,
        )

    # --- Aserciones de resultado ---
    assert baker_agent.beliefs.has("has_item", "bread", 1)
    assert baker_agent.goals == []
    assert baker_agent.intention is None

    # --- Aserciones de MECANISMO (no solo el resultado final) ---
    # El miller craftea la harina y NO se queda con ella (delivered).
    miller_flour = miller_agent.beliefs.query("has_item", "flour")
    assert not miller_flour or miller_flour[0][1] == 0

    # El baker recibió la harina vía el belief de entrega (no por atajo).
    assert baker_agent.beliefs.has("peer_done", "npc_miller", "achieve_has_item")

    # El miller consumió trigo real (recolección real, no simulada a medias).
    miller_wheat = miller_agent.beliefs.query("has_item", "wheat")
    # Pudo consumir las 2 unidades exactas (0 restante) — recolectó justo lo
    # necesario para la receta de harina.
    assert not miller_wheat or miller_wheat[0][1] == 0
