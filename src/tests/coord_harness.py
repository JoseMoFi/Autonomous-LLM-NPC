"""Arnés compartido para tests de coordinación con dos NPCAgent reales (Fase 17).

Mismo patrón que `test_integration_peer_delegation.py` y
`test_integration_e6_arbitration.py`: motor agentspeak, `BDIBehaviour` y
`PeerCoordBehaviour` REALES; bus de mensajes en memoria (sustituye XMPP) y mundo
mínimo en memoria (sustituye Unity). Añade lo necesario para planificar con el
pipeline LLM REAL (`PlanningRequestBehaviour._handle_pipeline`, el mismo mapeo
de payload que en producción) respondiendo con `OracleLLM`: un LLM determinista
que devuelve el cuerpo "correcto" de cada peldaño en modo SUB (sub-planes macro)
o ATOM (solo acciones primitivas y de coordinación).

No mide al LLM (el oráculo siempre acierta): comprueba que la MAQUINARIA permite
que dos agentes planificados por el pipeline se ayuden, en ambos brazos.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import agentspeak
import agentspeak.stdlib  # noqa: F401
from agentspeak import runtime as asp_runtime

from llm.pipeline.builtins import builtin_file_exclusions
from npc.agent import NPCAgent
from npc.behaviours.bdi import BDIBehaviour
from npc.behaviours.peer_coord import PeerCoordBehaviour
from npc.builtin_loader import load_builtin_plans
from protocol.messages import ItemSpawnSummary, RecipeItemPayload, RecipePayload

BAKERI = (0, 0)
WHEAT_AT = (10, 10)
PLANS_DIR = Path(__file__).resolve().parent.parent / "plans" / "builtin"
ABLATED = ("move_to_and_pickup", "craft_item", "obtain_from_peer", "collect_from_peer")

FLOUR_RECIPE = RecipePayload(
    recipeId="Flour_recipe", zone="bakeri",
    inputs=[RecipeItemPayload(itemId="wheat", qty=2)],
    outputs=[RecipeItemPayload(itemId="flour", qty=1)],
)
BREAD_FROM_FLOUR_RECIPE = RecipePayload(
    recipeId="Bread_from_flour_recipe", zone="bakeri",
    inputs=[RecipeItemPayload(itemId="flour", qty=1)],
    outputs=[RecipeItemPayload(itemId="bread", qty=1)],
)


# ---------------------------------------------------------------------------
# Transporte y mundo falsos
# ---------------------------------------------------------------------------

class Bus:
    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Queue] = {}

    def register(self, npc_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self.queues[npc_id] = queue
        return queue

    async def deliver(self, msg) -> None:
        queue = self.queues.get(str(msg.to).split("@")[0])
        if queue is not None:
            await queue.put(msg)


def _wire_transport(behaviour, npc_id: str, bus: Bus, queue: asyncio.Queue) -> None:
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


def _find_recipe(recipes, target: str):
    for recipe in recipes:
        if recipe.recipeId.lower() == str(target).lower():
            return recipe
    return None


def _world_executor(agent):
    async def _send_to_unity(msg: dict) -> None:
        if msg.get("type") != "ActionCommand":
            return
        action, args, cmd_id = msg["actionType"], msg.get("args", {}), msg["commandId"]

        def have(item: str) -> int:
            rows = agent.beliefs.query("has_item", item.lower())
            return rows[0][1] if rows else 0

        if action == "MoveTo":
            x, y = int(args["x"]), int(args["y"])
            agent.beliefs.apply_current_position(x, y)
            if (x, y) == BAKERI:
                agent.beliefs.apply_at_zone("bakeri")
            else:
                agent.beliefs.remove_at_zone("bakeri")
            agent.action_results[cmd_id] = {"status": "Success", "payload": {}}
        elif action == "PickUp":
            agent.beliefs.apply_has_item(args["itemId"], have(args["itemId"]) + 1)
            agent.action_results[cmd_id] = {"status": "Success", "payload": {}}
        elif action == "Craft":
            recipe = _find_recipe(agent.profile.recipes, args["targetId"])
            if recipe is None or not agent.beliefs.has("at_zone", recipe.zone):
                agent.action_results[cmd_id] = {"status": "Failure", "errorCode": "no_recipe_or_zone"}
                return
            inp, out = recipe.inputs[0], recipe.outputs[0]
            if have(inp.itemId) < inp.qty:
                agent.action_results[cmd_id] = {"status": "Failure", "errorCode": "missing_items"}
                return
            agent.beliefs.apply_has_item(inp.itemId.lower(), have(inp.itemId) - inp.qty)
            agent.beliefs.apply_has_item(out.itemId.lower(), have(out.itemId) + out.qty)
            agent.action_results[cmd_id] = {"status": "Success", "payload": {}}
        elif action == "Drop":
            item, qty = args["itemId"], int(args.get("qty", 1))
            agent.beliefs.apply_has_item(item, max(0, have(item) - qty))
            agent.action_results[cmd_id] = {"status": "Success", "payload": {}}
        else:
            agent.action_results[cmd_id] = {"status": "Failure", "errorCode": "unsupported_in_fake_world"}

    return _send_to_unity


class FakeProfile:
    """Lo mínimo de NPCProfilePayload que usan _request_plan y derive_entity_catalog."""

    def __init__(self, recipes, role: str) -> None:
        self.recipes = list(recipes)
        # item_spawns y zonas son GLOBALES en Unity (hallazgo Fase 11).
        # Como Unity: flour/bread llegan con el centinela "none" (craft-only, targetCount 0).
        self.item_spawns = [
            ItemSpawnSummary(itemId="wheat", zones=["farmland"], targetCount=2, weight=1),
            ItemSpawnSummary(itemId="flour", zones=["none"], targetCount=0, weight=1),
            ItemSpawnSummary(itemId="bread", zones=["none"], targetCount=0, weight=1),
        ]
        self.zones = [SimpleNamespace(tag="farmland"), SimpleNamespace(tag="bakeri")]
        self.delivery_points: dict = {}
        self.role = role
        self.goals_nl: list[str] = []

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "recipes": [r.to_dict() for r in self.recipes],
            "item_spawns": [s.to_dict() for s in self.item_spawns],
        }


async def make_agent(npc_id: str, role: str, recipes, bus: Bus, *, builtin_subplans: bool):
    agent = NPCAgent(
        jid=f"{npc_id}@localhost", password="x", npc_id=npc_id,
        send_to_unity=None, llm_planning_jid="llm@localhost",
    )
    agent.send_to_unity = _world_executor(agent)
    agent.profile = FakeProfile(recipes, role)
    agent.beliefs.npc_id = npc_id
    for r in recipes:
        agent.beliefs.apply_recipe(
            r.recipeId, r.zone,
            [{"itemId": i.itemId, "qty": i.qty} for i in r.inputs],
            [{"itemId": o.itemId, "qty": o.qty} for o in r.outputs],
        )
    agent.beliefs.apply_item_spawn("wheat", "farmland")
    agent.beliefs.apply_item_at("wheat", WHEAT_AT[0], WHEAT_AT[1], current_tick=0)
    agent.beliefs.apply_zone_discovery("bakeri", BAKERI[0], BAKERI[1])
    agent.beliefs.apply_current_position(5, 5)

    load_builtin_plans(
        agent.plan_graph, PLANS_DIR,
        exclude_files=builtin_file_exclusions(builtin_subplans, coordination=True),
    )

    bdi = BDIBehaviour()
    bdi.agent = agent
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

    queue = bus.register(npc_id)
    peer = PeerCoordBehaviour()
    peer.agent = agent
    _wire_transport(peer, npc_id, bus, queue)
    _wire_transport(bdi, npc_id, bus, queue)
    return agent, bdi, peer


async def run_until(agents_bdi_peer, condition, *, max_seconds: float = 90.0, idle_ticks: int = 30) -> None:
    """Alterna un tick de BDI y de coordinación por agente hasta `condition()`.

    Falla pronto (en vez de agotar el tiempo) si TODOS los agentes quedan sin
    goals ni intención durante `idle_ticks` seguidos sin cumplir la condición:
    el escenario ya terminó en fracaso."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_seconds
    idle = 0
    while loop.time() < deadline:
        if condition():
            return
        if all(not agent.goals and agent.intention is None for agent, _bdi, _peer in agents_bdi_peer):
            idle += 1
            if idle >= idle_ticks:
                raise AssertionError("todos los NPC ociosos sin cumplir la condición (el escenario falló)")
        else:
            idle = 0
        for _agent, bdi, peer in agents_bdi_peer:
            await bdi.run()
            await peer.run()
        await asyncio.sleep(0)
    raise AssertionError(f"condición no alcanzada en {max_seconds:.0f} s")


# ---------------------------------------------------------------------------
# LLM de oráculo
# ---------------------------------------------------------------------------

_ITEM_QTY_RE = re.compile(r"has_item\((\w+),\s*(\d+)\)")


def _line_after(text: str, header: str) -> str:
    idx = text.find(header)
    if idx < 0:
        return ""
    rest = text[idx + len(header):].lstrip("\n")
    return rest.split("\n", 1)[0].strip()


def _block_after(text: str, header: str) -> list[str]:
    idx = text.find(header)
    if idx < 0:
        return []
    lines = []
    for line in text[idx + len(header):].split("\n")[1:]:
        if not line.strip():
            break
        lines.append(line.strip())
    return lines


def _fact_args(facts: list[str], functor: str) -> list[list[str]]:
    out = []
    for fact in facts:
        if fact.startswith(functor + "("):
            out.append([a.strip() for a in fact[len(functor) + 1:-1].split(",")])
    return out


def _action(name: str, *args) -> dict:
    return {"type": "action", "name": name, "args": list(args)}


def _subgoal(name: str, *args) -> dict:
    return {"type": "subgoal", "name": name, "args": list(args)}


class OracleLLM:
    """Responde a los prompts del pipeline con el cuerpo correcto del peldaño.

    `plans` guarda (fallo, pasos) de cada respuesta de step3 para que el test
    compruebe qué se usó en cada brazo."""

    def __init__(self, mode: str) -> None:
        assert mode in ("SUB", "ATOM")
        self.mode = mode
        self.plans: list[tuple[str, list[dict]]] = []

    async def __call__(self, user: str, system: str) -> str:
        if "BDI goal signature and a detailed description" in system:
            return json.dumps({"sig": "achieve_do_task", "description": "The NPC must complete its task."})
        if "Generate the ordered action steps" in system:
            return json.dumps({"steps": self._step3(user)})
        if "trailing plan steps are redundant" in system:
            return json.dumps({"redundant": False})
        if "Fix flagged steps" in system:
            return json.dumps({"fixes": []})
        raise AssertionError(f"prompt inesperado para el oráculo: {system[:120]!r}")

    def _step3(self, user: str) -> list[dict]:
        failing = _line_after(user, "Failing condition (the ONE sub-problem these steps must resolve):")
        guard = _line_after(
            user, "Full variant guard (for reference — DO NOT add steps for already-satisfied atoms):",
        ) or failing
        facts = _block_after(
            user, "Known relevant facts (use these values to reason — do NOT invent coordinates or names):",
        )
        sub = self.mode == "SUB"

        if "peer_item_available(P," in guard:
            item, qty = _ITEM_QTY_RE.search(failing).groups()
            steps = ([_subgoal("collect_from_peer", "P", item, int(qty))] if sub
                     else [_action("MoveTo", "X", "Y"), _action("PickUp", item)])
        # Fase 17o (ATOM): peldaños de esperar, reintentar y pedir.
        elif not sub and "peer_promised(P, G)" in guard and "not peer_failed(P, G, _)" in guard:
            steps = [_action("await_peer", "P", "G", 30)]
        elif not sub and ("not peer_promised(_, _)" in guard or "peer_failed(P, G, R)" in guard):
            item = re.search(r"peer_item_available\(_, (\w+),", guard).group(1)
            qty = re.search(rf"has_item\({item}, (\d+)\)", guard).group(1)
            peer = re.search(r"Other NPCs it can ask: ([\w]+)", user).group(1)
            steps = [
                _action("ask_peer", peer, "can_make", item),
                _action("request_peer", peer, "achieve_has_item", item, int(qty)),
            ]
        elif "not peer_item_available(" in guard:
            item, qty = _ITEM_QTY_RE.search(failing).groups()
            peer = re.search(r"Other NPCs it can ask: ([\w]+)", user).group(1)
            steps = ([_subgoal("obtain_from_peer", peer, item, int(qty))] if sub else [
                _action("ask_peer", peer, "can_make", item),
                _action("request_peer", peer, "achieve_has_item", item, int(qty)),
                _action("await_peer", peer, "achieve_has_item", 30),
            ])
        # Fase 17f (ATOM): peldaños de precondiciones de PickUp.
        elif failing.startswith("not item_at("):
            item = failing[len("not item_at("):].split(",")[0].strip()
            zone = next(a[1] for a in _fact_args(facts, "item_spawn") if a[0] == item)
            centers = [a for a in _fact_args(facts, "zone_center") if a[0] == zone]
            steps = ([_action("MoveTo", int(centers[0][1]), int(centers[0][2])), _action("Search", item)]
                     if centers else [_action("ExploreArea", zone)])
        elif failing.startswith("not current_position("):
            steps = [_action("MoveTo", "X", "Y")]
        elif failing.startswith("not has_item(") and not sub and (
            "current_position(X, Y)" in guard or re.search(r"Bound variables:[^\n]*\bX\b", user)
        ):
            steps = [_action("PickUp", _ITEM_QTY_RE.search(failing).group(1))]
        elif failing.startswith("not at_zone("):
            zone = failing[len("not at_zone("):-1]
            if sub:
                recipe = _fact_args(facts, "recipe_output")[0][0]
                steps = [_subgoal("craft_item", recipe, 1)]
            else:
                _z, zx, zy = next(a for a in _fact_args(facts, "zone_center") if a[0] == zone)
                steps = [_action("MoveTo", int(zx), int(zy))]
        elif re.search(r"(?:^|& )at_zone\(", guard):
            recipe = _fact_args(facts, "recipe_output")[0][0]
            if sub:
                steps = [_subgoal("craft_item", recipe, 1)]
            else:
                ingredient = next(a[1] for a in _fact_args(facts, "recipe_input") if a[0] == recipe)
                steps = [_action("Craft", ingredient, recipe)]
        elif failing.startswith("not has_item("):
            item, qty = _ITEM_QTY_RE.search(failing).groups()
            if sub:
                steps = [_subgoal("move_to_and_pickup", item, int(qty))]
            else:
                _i, x, y = next(a for a in _fact_args(facts, "item_at") if a[0] == item)
                steps = [_action("MoveTo", int(x), int(y))] + [_action("PickUp", item)] * int(qty)
        else:
            raise AssertionError(f"peldaño no reconocido por el oráculo: {failing!r} / {guard!r}")

        self.plans.append((failing, steps))
        return steps


def install_llm(agent, oracle: OracleLLM, contracts_dir: Path) -> None:
    """Sustituye `agent.run_llm_task` por el pipeline REAL con `oracle` como LLM."""
    from llm.planning_agent import PlanningRequestBehaviour

    async def traced(prompt, system="", **_kw):
        return await oracle(prompt, system)

    async def structured(prompt, system, schema, **_kw):
        return schema.model_validate_json(await oracle(prompt, system))

    stub = SimpleNamespace(agent=SimpleNamespace(llm_call_traced=traced, llm_call_structured_traced=structured))

    async def run_llm_task(task: dict):
        kind = task.get("task")
        if kind == "prioritize":
            return [{"sig": g.get("sig"), "score": 1.0} for g in task.get("goals", [])]
        if kind == "generate_plan":
            payload = dict(task)
            payload["capability_contracts_path"] = str(contracts_dir / f"{agent.npc_id}_contracts.json")
            result, errors = await PlanningRequestBehaviour._handle_pipeline(stub, payload)
            if errors:
                raise AssertionError(f"pipeline con errores: {errors}")
            return result
        raise AssertionError(f"tarea LLM inesperada en el test: {kind}")

    agent.run_llm_task = run_llm_task
