"""BDI behaviour — ciclo deliberativo del NPC.

Usa agentspeak.runtime.Agent para ejecutar planes ASL de forma nativa:
  - Variable resolution: agentspeak unification (no Python regex)
  - Guard evaluation:    agentspeak context matching (no Python dict lookup)
  - Body execution:      agentspeak step() engine (no Python loop over steps)
  - Sub-goal adoption:   agentspeak !sub_goal call (no Python re-injection)

Las acciones Unity (.moveto, .pickup, …) se registran como generators de
agentspeak que:
  1. Envían el comando a Unity (fire-and-forget via asyncio.ensure_future)
  2. Fijan intention.waiter = ActionResultWaiter(cmd_id) para bloquear el
     plan hasta que llegue el ActionResult de Unity.
  3. Hacen yield — next_or_fail() devuelve True (acción "lanzada").
  4. El waiter se revisa en cada asp.step(): devuelve False mientras espera,
     True cuando llega el resultado. Si el resultado es Failure, inyecta una
     instrucción pop_and_fail antes de que continúe el plan.

El Python BDI sigue siendo responsable de:
  - Orquestación de planes (plan_graph, NodeStatus, LLM synthesis)
  - Selección de intención con priorización LLM
  - Gestión de métricas y trazas
  - Fuse de seguridad para variantes no aplicables
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

import agentspeak
import agentspeak.stdlib  # noqa: F401
from agentspeak import runtime as asp_runtime
from agentspeak.runtime import Instruction
import spade
from spade_bdi.bdi import BDIAgent

from npc.asp_guard import GuardEvaluator, compile_plan_with_actions
from npc.plan_graph import GoalNode, NodeStatus, PlanVariant
from utils.trace_logger import trace as _trace, TraceLogger, GoalRecord

log = logging.getLogger(__name__)

_MAX_NO_VARIANT_HITS = 20

# Re-inyecciones consecutivas de un goal sin que cambien las beliefs (escalera de
# variantes atascada) antes de forzar un replan. Evita bucles cuando una rama
# casa y se ejecuta pero su resultado no progresa el estado.
_MAX_LADDER_STUCK = 3


# ---------------------------------------------------------------------------
# ActionResultWaiter — pausa la intención hasta que llegue ActionResult
# ---------------------------------------------------------------------------

def goal_statement(goal: Any) -> str:
    """Fase 17u: enunciado NL que recibe el planificador (`npc_statement`).

    El payload de plan no lo enviaba y el planificador lo reconstruía del sig:
    `achieve_deliver_to_peer` → "Deliver to peer", y step0 describía "deliver the
    item to the peer ... maintain the relationship", sin item, cantidad ni
    destinatario. Con ese objetivo el LLM ignoraba incluso la corrección explícita
    de step3 y volvía a recolectar trigo en la rama de craftear (piloto 17t).
    El encargo de un peer se enuncia con su binding; el resto de goals no cambia
    (cadena vacía → el planificador sigue usando el sig).
    """
    call_args = list(getattr(goal, "call_args", []) or [])
    if getattr(goal, "sig", "") == "achieve_deliver_to_peer" and len(call_args) == 3:
        requester, item, qty = call_args
        return f"Produce {qty} {item} for {requester} and deliver it to them ({requester} requested it)"
    return ""


def moveto_retry_candidates(x: int, y: int, *, include_neighbours: bool = True) -> list[tuple[int, int]]:
    """Fase 17s: destinos de reintento tras un MoveTo con PathNotFound.

    Unity cancela el MoveTo (PathNotFound) cuando la casilla destino o una del
    camino está ocupada por otro NPC (GridMover → WorldSpatialIndex.TryMoveAgent).
    Primero el mismo destino (bloqueo de paso, transitorio) y después las 8
    vecinas (destino ocupado: para Craft basta estar dentro de la zona). Sin
    vecinas si en el destino hay un item: PickUp exige estar en su casilla.
    """
    same = [(x, y)]
    if not include_neighbours:
        return same
    return same + [
        (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
        (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1), (x - 1, y - 1),
    ]


class ActionResultWaiter:
    """
    Waiter personalizado (compatible agentspeak.runtime.Waiter) que bloquea
    una intención agentspeak hasta que Unity responde al comando.

    Cuando el resultado llega:
      - Si status == "Success": deja que pop_query_instr se ejecute normalmente.
      - Si status != "Success": fija intention.instr=None (termina la intención
        limpiamente) y guarda el error en bdi._pending_failure para que run()
        llame a _handle_action_failure en lugar de _complete_goal.

    El timeout (60s) se comprueba en cada poll() para evitar esperas infinitas.
    """

    def __init__(
        self,
        cmd_id: str,
        results: dict,
        intention: asp_runtime.Intention,
        npc_id: str,
        goal_sig: str | None,
        action_type: str,
        bdi: Any,
        timeout_s: float = 60.0,
        move_target: tuple[int, int] | None = None,
    ) -> None:
        self._cmd_id = cmd_id
        # Fase 17s: destino original de un MoveTo y reintentos hechos.
        self._move_target = move_target
        self._move_attempt = 0
        self._results = results
        self._intention = intention
        self._npc_id = npc_id
        self._goal_sig = goal_sig
        self._action_type = action_type
        self._bdi = bdi
        self._timeout_s = timeout_s
        self._t0 = time.time()
        # Requeridos por agentspeak.runtime.Agent.call() y shortest_deadline()
        self.event = None
        self.until = None

    def poll(self, env: Any) -> bool:
        # ── Timeout (configurable por acción/settings — 0.C4) ──────────────
        if time.time() - self._t0 > self._timeout_s:
            result: dict = {"status": "Failure", "errorCode": "timeout"}
        elif self._cmd_id not in self._results:
            return False  # aun esperando
        else:
            result = self._results.pop(self._cmd_id)

        status = result.get("status", "Failure")
        latency = round(time.time() - self._t0, 3)

        _trace(
            "action_result",
            npc_id=self._npc_id,
            goal=self._goal_sig,
            action=self._action_type,
            status=status,
            latency_s=latency,
            cmd_id=self._cmd_id,
        )
        tl = TraceLogger.get()
        if tl is not None:
            m = tl.metrics.npc(self._npc_id)
            if status == "Success":
                m.actions_ok += 1
            else:
                m.actions_failed += 1

        if status != "Success" and self._move_target is not None:
            retry_cmd = self._bdi._retry_moveto_nearby(self._move_target, self._move_attempt, result)
            if retry_cmd is not None:
                self._move_attempt += 1
                self._cmd_id = retry_cmd
                self._t0 = time.time()
                return False

        if status != "Success":
            error_code = (
                result.get("errorCode") or result.get("errorMessage") or "unity_action_failed"
            )
            # 0.B2 — contar el fallo si la acción es observacional y, llegado el
            # presupuesto, assertar exhausted(action, budget).
            self._bdi._note_observational_failure(self._goal_sig, self._action_type)
            # Terminar la intención limpiamente (instr=None) y señalar fallo
            # en BDIBehaviour para que run() llame a _handle_action_failure
            # en lugar de _complete_goal. Evita AggregatedError del runtime.
            self._intention.instr = None
            self._bdi._pending_failure = error_code
            message = result.get("errorMessage") or ""
            self._bdi._pending_failure_detail = (
                f"{self._action_type} failed ({error_code})" + (f": {message}" if message else "")
            )

        return True


# ---------------------------------------------------------------------------
# Waiters de coordinación NPC↔NPC (Fase 12) — mismo contrato que
# ActionResultWaiter (poll(env) -> bool, atributos event/until), pero sobre
# canales de coordinación en vez de resultados de acción Unity.
# ---------------------------------------------------------------------------

class PeerReplyWaiter:
    """Espera una respuesta de coordinación (inform/agree/refuse) correlacionada
    por `conversation_id`, que `PeerCoordBehaviour` escribe en
    `agent.peer_results` al llegar. Usada por `.ask_peer` y `.request_peer`.

    Al resolver (respuesta real o timeout) invoca `on_resolve(pmsg)` — `pmsg`
    es `None` en timeout — para que la acción escriba las beliefs `peer_*`
    correspondientes y, si procede, marque el fallo (mismo mecanismo que
    ActionResultWaiter: `bdi._pending_failure`).
    """

    def __init__(self, conversation_id: str, agent: Any, on_resolve, timeout_s: float) -> None:
        self._cid = conversation_id
        self._agent = agent
        self._on_resolve = on_resolve
        self._timeout_s = timeout_s
        self._t0 = time.time()
        self.event = None
        self.until = None

    def poll(self, env: Any) -> bool:
        pmsg = self._agent.peer_results.pop(self._cid, None)
        if pmsg is not None:
            self._on_resolve(pmsg)
            return True
        if time.time() - self._t0 > self._timeout_s:
            self._on_resolve(None)
            return True
        return False

    def shift_deadline(self, delta_s: float) -> None:
        """Fase 14: desplaza el reloj interno para que un intervalo en el que
        el waiter estuvo APARCADO (sin poll-earse, ver BDIBehaviour._park_intention)
        no cuente como tiempo de espera. Sin esto, una intención que vuelve de
        estar aparcada expiraría de inmediato aunque el peer siga trabajando
        correctamente."""
        self._t0 += delta_s


class PeerDoneWaiter:
    """Espera a que `agent.beliefs` tenga `peer_done(Npc, GoalSig)` (éxito) o
    `peer_failed(Npc, GoalSig, Reason)` (fallo) — las escribe
    `PeerCoordBehaviour` al recibir `inform-done`/`failure`. Usada por
    `.await_peer`. Sin correlación por conversation_id: solo hay UNA petición
    activa por (peer, goal_sig) en el alcance de esta fase (dedup lo garantiza
    en el receptor)."""

    def __init__(
        self, npc_id: str, goal_sig: str, agent: Any, on_resolve, timeout_s: float,
    ) -> None:
        self._npc_id = npc_id
        self._goal_sig = goal_sig
        self._agent = agent
        self._on_resolve = on_resolve
        self._timeout_s = timeout_s
        self._t0 = time.time()
        self.event = None
        self.until = None

    def poll(self, env: Any) -> bool:
        beliefs = self._agent.beliefs
        if beliefs.has("peer_done", self._npc_id, self._goal_sig):
            self._on_resolve(True, None)
            return True
        failed = beliefs.query("peer_failed", self._npc_id, self._goal_sig, None)
        if failed:
            self._on_resolve(False, failed[0][2])
            return True
        # Fase 17x: un encargo rechazado no se va a cumplir; no esperar al timeout.
        refused = beliefs.query("peer_refused", self._npc_id, self._goal_sig, None)
        if refused:
            self._on_resolve(False, f"refused:{refused[0][2]}")
            return True
        if time.time() - self._t0 > self._timeout_s:
            self._on_resolve(False, "timeout")
            return True
        return False

    def shift_deadline(self, delta_s: float) -> None:
        """Ver PeerReplyWaiter.shift_deadline — mismo motivo, mismo mecanismo."""
        self._t0 += delta_s


# ---------------------------------------------------------------------------
# BDIBehaviour
# ---------------------------------------------------------------------------

_guard_evaluator = GuardEvaluator()


class BDIBehaviour(BDIAgent.BDIBehaviour):
    """
    Ciclo deliberativo principal del NPC.

    Usa agentspeak.runtime.Agent (self._asp) para ejecutar planes ASL de
    forma nativa: variable resolution, guard evaluation y body execution.

    Hereda de `spade_bdi.bdi.BDIAgent.BDIBehaviour` (uso real, no de
    fachada: `on_start` expone el MISMO `agentspeak.Agent` bajo los nombres
    que la clase base espera — `self.agent.bdi_agent/bdi_env/bdi_actions` —
    así que sus métodos heredados y no reescritos (`get_belief`/`set_belief`/
    `get_beliefs`/`print_beliefs`) son genuinamente funcionales sobre nuestro
    motor, aunque el resto de este proyecto siga usando `BeliefStore` como
    fuente de verdad). `run()` SÍ se reescribe entero (ver más abajo): el de
    la clase base espera un mensaje SPADE con `{"performative": "BDI"}` y una
    fuerza ilocucionaria (tell/achieve/askHow...) — un protocolo de
    coordinación entre agentes distinto y no compatible con el propio de
    este proyecto (Fase 12, `protocol/peer_messages.py`), y no dirige en
    absoluto la orquestación de planes (plan_graph/NodeStatus/replan/
    memoria/arbitraje) que sí necesitamos aquí — el mismo motivo por el que
    la v1 de este proyecto (pre-tfm-v2) ya reescribía `run()` al usar
    `BDIAgent` directamente.
    """

    # --- lifecycle ----------------------------------------------------------

    async def on_start(self) -> None:
        self._actions = agentspeak.Actions(agentspeak.stdlib.actions)
        self._env = asp_runtime.Environment()
        # Agente asp vacío (se cargarán planes al arrancar y al sintetizar)
        self._asp = self._env.build_agent(agentspeak.StringSource("<bdi>", ""), self._actions)
        # Exponer el mismo motor bajo los nombres que BDIAgent espera, para
        # que sus helpers heredados operen sobre el agente real (ver
        # docstring de la clase) en vez de sobre `None`.
        self.agent.bdi_env = self._env
        self.agent.bdi_actions = self._actions
        self.agent.bdi_agent = self._asp
        self.agent.bdi_enabled = True
        self.agent.bdi = self
        self._pending_failure: str | None = None
        # Fase 17e: accion + mensaje de Unity del ultimo fallo, para el hint de replan.
        self._pending_failure_detail: str | None = None
        # 0.B2 — contador de intentos fallidos por (goal_sig, action_type) para
        # acciones observacionales. Al alcanzar el attempt_budget del contrato se
        # assertea exhausted(action, budget) en BeliefStore.
        self._attempt_counts: dict[tuple[str, str], int] = {}
        # Fase 14: arbitraje de goals — estado del punto de preempción.
        self._arbitration_switch_count: int = 0
        self._last_arbitration_ts: float = 0.0
        # Fase 17w: clave del último goal al que se cambió (volver a él no gasta tope).
        self._last_switch_key: str | None = None
        # Registrar acciones Unity antes de cargar planes
        self._register_unity_actions()
        # Fase 12: acciones de coordinación NPC↔NPC — SOLO si el flag está
        # activo. Con el flag off, .ask_peer/.request_peer/.await_peer/
        # .deliver_to_peer no se registran en absoluto (no solo no se usan):
        # cualquier ASL que las invocara fallaría a compilar, igual que
        # cualquier otra acción desconocida — cero superficie nueva.
        from config import settings as _settings
        if getattr(_settings, "coordination_enabled", False):
            self._register_peer_actions()
        # Cargar planes built-in
        await self._load_builtin_plans()

    async def run(self) -> None:
        agent = self.agent

        if agent.paused or agent.profile is None:
            await asyncio.sleep(0.1)
            return

        if not agent.goals:
            await asyncio.sleep(0.1)
            return

        # Sin intención activa → seleccionar una (con priorización LLM)
        if agent.intention is None:
            await self._select_intention()
            return

        goal = agent.intention

        # Short-circuit: success_condition ya satisfecha
        if goal.success_condition and self._eval_guard(goal.success_condition):
            log.info(
                f"[BDI:{agent.npc_id}] Goal '{goal.sig}' ya satisfecho (.done) — completando"
            )
            self._asp.intentions.clear()
            await self._complete_goal(goal)
            return

        # Fase 14: arbitraje de goals — si `goal` está bloqueado esperando a un
        # peer y hay otro goal pendiente que desbloquearía a ESE MISMO peer,
        # posible cambio de intención (preempción cooperativa, no concurrencia
        # real: nunca hay más de una pila viva). Ver _maybe_arbitrate.
        if await self._maybe_arbitrate(goal):
            return

        # ── Gestión de planes (plan_graph) ──────────────────────────────────

        # Identidad por binding (Fase 6.5): el plan_graph se indexa por
        # sig+call_args (goal_key), no solo por el `sig` de familia. Así bread y
        # wheat (misma familia achieve_has_item) son nodos SEPARADOS y coexisten;
        # y "reusar solo si existe el plan que casa" sale gratis: un binding sin
        # nodo propio → has_node False → se genera (no se reusa el de otro item).
        goal_key = self._goal_key(goal)
        if not agent.plan_graph.has_node(goal_key):
            await self._request_plan(goal)
            return

        node: GoalNode = agent.plan_graph.nodes[goal_key]["data"]

        if node.status == NodeStatus.FAILED:
            log.warning(f"[BDI:{agent.npc_id}] Goal '{goal.sig}' FAILED — descartando")
            # Fase 17c: cierre por fallo VISIBLE. Antes el goal desaparecía sin
            # traza (fallos de acción, fusible o error de plan) y el análisis no
            # podía atribuir la causa.
            _history = getattr(node, "failure_history", None) or []
            _trace(
                "goal_failed", npc_id=agent.npc_id, goal=goal.sig,
                success_condition=goal.success_condition,
                reason=str(_history[-1]) if _history else "node_failed",
            )
            self._notify_peer_on_failure(goal, reason="node_failed")
            agent.goals = [g for g in agent.goals if not self._same_goal(g, goal)]
            agent.intention = None
            self._asp.intentions.clear()
            return

        if node.status == NodeStatus.PENDING:
            log.info(f"[BDI:{agent.npc_id}] Goal '{goal.sig}' PENDING — generando plan")
            await self._request_plan(goal)
            return

        if node.status == NodeStatus.GENERATING:
            await asyncio.sleep(0.1)
            return

        if node.status == NodeStatus.NEEDS_REPLAN:
            log.info(f"[BDI:{agent.npc_id}] Goal '{goal.sig}' NEEDS_REPLAN — re-solicitando plan")
            node.status = NodeStatus.PENDING
            self._unload_plan(goal_key)
            self._asp.intentions.clear()
            await self._request_plan(goal)
            return

        if node.status != NodeStatus.READY:
            await asyncio.sleep(0.1)
            return

        # ── Cargar variantes en asp_agent si es la primera vez ───────────────
        if not self._is_plan_loaded(goal_key):
            self._load_plans_into_asp(node, goal_key)
            # Fase 4: si el plan vino de memoria (no del LLM), trazarlo y contarlo
            # — distinguible de builtin y de LLM-fresh (regla de no-silencio).
            if getattr(node, "from_memory", False):
                _sc = node.success_count + node.failure_count
                _rate = round(node.success_count / _sc, 3) if _sc else None
                _trace(
                    "plan_memory_reuse",
                    npc_id=agent.npc_id,
                    goal=goal.sig,
                    success_rate=_rate,
                    uses_success=node.success_count,
                    uses_error=node.failure_count,
                )
                log.info(
                    "[MEMORY:%s] Reusando plan aprendido '%s' (tasa=%s) — sin LLM",
                    agent.npc_id, goal.sig, _rate,
                )
                _tlm = TraceLogger.get()
                if _tlm is not None:
                    _tlm.metrics.npc(agent.npc_id).plans_from_memory += 1

        # ── Sincronizar beliefs BeliefStore → asp_agent ──────────────────────
        # Debe hacerse ANTES de _inject_goal para que los guards puedan
        # evaluar beliefs actuales (ej. zone_center, zone_discovered, etc.)
        self._sync_beliefs()

        # ── Inyectar goal en asp_engine si no hay intenciones activas ─────────
        if not self._asp.intentions:
            try:
                self._inject_goal(goal)
            except agentspeak.AslError:
                # Ninguna variante/guard aplica en las beliefs actuales
                await self._handle_no_applicable_variant(goal)
                return
            # Reset del fuse al poder inyectar
            goal.no_variant_hits = 0

        # ── Ejecutar un paso del plan ─────────────────────────────────────────
        had_intentions = bool(self._asp.intentions)
        try:
            ran = self._asp.step()
        except agentspeak.AslError as exc:
            await self._handle_asp_error(goal, node, exc)
            return
        except agentspeak.AggregatedError as exc:
            # El runtime envuelve AslError en AggregatedError; tratar como
            # fallo genérico del plan.
            await self._handle_asp_error(
                goal, node, agentspeak.AslError(f"plan_failure:{exc}")
            )
            return

        # Comprobar fallo de acción Unity señalado por ActionResultWaiter
        if self._pending_failure is not None:
            error_code = self._pending_failure
            self._pending_failure = None
            # getattr: hay arneses que crean el behaviour sin pasar por su init.
            detail = getattr(self, "_pending_failure_detail", None)
            self._pending_failure_detail = None
            self._asp.intentions.clear()
            log.warning(
                "[BDI:%s] Unity action failed in '%s': %s",
                self.agent.npc_id, goal.sig, error_code,
            )
            await self._handle_action_failure(goal, {
                "status": "Failure",
                "errorCode": error_code,
                "errorMessage": detail or f"unity_action_failed:{error_code}",
            })
            return

        # Catch-all de la escalera: la rama `: true <- .request_replan` disparó
        # porque ninguna variante real casaba con el estado actual → replan.
        if self._replan_requested:
            self._replan_requested = False
            log.info(
                "[BDI:%s] Ninguna variante aplica para '%s' (catch-all) → replan",
                self.agent.npc_id, goal.sig,
            )
            await self._trigger_replan_or_fail(goal, node, reason="no_applicable_variant")
            return

        if had_intentions and not self._asp.intentions:
            # Una RAMA (variante) del plan terminó su cuerpo. Esto NO implica que el
            # goal esté hecho: el plan es una escalera de variantes guardadas. Si la
            # success_condition no se cumple, NO replanificamos: dejamos que el
            # siguiente tick re-inyecte el goal y AgentSpeak dispare la siguiente
            # variante aplicable (avanza la escalera ejecutando ASL, sin LLM).
            await self._on_variant_completed(goal, node)
        elif not ran:
            await asyncio.sleep(0.05)

    # --- identidad de goal (sig + binding) ----------------------------------

    def _goal_key(self, goal: Any) -> str:
        """Clave de identidad del goal para indexar plan_graph y planes cargados:
        incluye el binding (sig+call_args) cuando existe, si no el sig. Hace que
        bread y wheat (misma familia achieve_has_item) sean entidades separadas."""
        from llm.canonical import goal_identity_key
        return goal_identity_key(goal.sig, getattr(goal, "call_args", []) or [])

    def _reusable_known_goals(self) -> list[str]:
        """Fase 17w: planes que el LLM puede reutilizar como sub-goal (`known_goals`).

        Antes era `plan_graph.nodes` entero, que incluye los goals ACTIVOS del NPC
        (los suyos y los encargos de peers). La 17s quitó las claves con binding
        (`sig__args`) en el pipeline, pero un goal sin argumentos seguía en la lista:
        en CO6 el miller veía `achieve_bake_bread` como sub-goal al planificar el
        encargo de harina y escribía Craft(flour, Flour_recipe) 5/5; sin él,
        Craft(wheat, …) 5/5 (piloto 17v, reproduciendo el prompt).
        """
        agent = self.agent
        active = list(agent.goals) + ([agent.intention] if agent.intention is not None else [])
        excluded = {self._goal_key(g) for g in active} | {g.sig for g in active}
        return [node for node in agent.plan_graph.nodes if node not in excluded]

    @staticmethod
    def _same_goal(g: Any, goal: Any) -> bool:
        """Identidad de goal por sig + binding (no solo sig): completar `wheat` no
        debe evictar `bread` (misma familia, distinto call_args)."""
        return g.sig == goal.sig and (
            list(getattr(g, "call_args", []) or []) == list(getattr(goal, "call_args", []) or [])
        )

    # --- Fase 12: notificar al peticionario cuando un goal delegado falla ---

    def _notify_peer_on_failure(self, goal: Any, *, reason: str) -> None:
        """Si `goal` fue delegado por otro NPC (`goal_source == "peer"`), avisa
        al peticionario original con `failure` al cerrarse SIN éxito.

        Se llama en los TRES puntos donde un goal se cierra definitivamente por
        fallo (`run()` dispatch de node FAILED, `_trigger_replan_or_fail`,
        `_complete_goal` con budget de replan agotado) — cubre también los
        caminos que solo marcan `node.status=FAILED` sin cerrar por sí mismos
        (`_handle_action_failure`, `_handle_no_applicable_variant`), porque
        ambos terminan pasando por el dispatch de `run()`.

        Sin esto, el `.await_peer` del peticionario colgaría hasta su propio
        timeout en vez de enterarse del fallo real — regla de no-silencio
        aplicada a la coordinación. No-op si el goal no es de origen peer
        (coste cero fuera de la Fase 12).
        """
        if getattr(goal, "goal_source", None) != "peer":
            return
        # Fase 17t: contar el fallo del encargo (PeerCoord rechaza al pasar el tope).
        failed = getattr(self.agent, "failed_peer_deliveries", None)
        if isinstance(failed, dict):
            key = self._goal_key(goal)
            failed[key] = failed.get(key, 0) + 1
        requester_jid = getattr(goal, "peer_requester_jid", None)
        if not requester_jid:
            log.warning(
                "[BDI:%s] Goal delegado '%s' falló sin peer_requester_jid — "
                "no se puede notificar al peticionario",
                self.agent.npc_id, goal.sig,
            )
            return

        from protocol.peer_messages import PeerMessage, PROTOCOL_METADATA

        origin_sig = getattr(goal, "peer_origin_sig", None) or goal.sig
        payload = PeerMessage(
            performative="failure",
            conversation_id=str(uuid.uuid4()),
            goal_sig=origin_sig,
            reason=reason,
        ).to_dict()
        msg = spade.message.Message(
            to=requester_jid, metadata=dict(PROTOCOL_METADATA), body=json.dumps(payload),
        )
        asyncio.ensure_future(self.send(msg))
        log.info(
            "[BDI:%s] Notificando fallo a %s: goal_sig=%s reason=%s",
            self.agent.npc_id, requester_jid, origin_sig, reason,
        )
        _trace(
            "peer_request_failed_notify",
            npc_id=self.agent.npc_id, to=requester_jid,
            goal_sig=origin_sig, reason=reason,
        )
        tl = TraceLogger.get()
        if tl is not None:
            tl.metrics.npc(self.agent.npc_id).peer_msgs_sent += 1

    # --- registro de acciones Unity -----------------------------------------

    def _send_and_wait(
        self,
        action_type: str,
        args: dict,
        intention: asp_runtime.Intention,
    ) -> None:
        """Helper compartido: valida, envía a Unity y fija el waiter. Método
        de instancia (no closure local) para que tanto las acciones Unity
        (_register_unity_actions) como las de coordinación
        (_register_peer_actions — p.ej. `.deliver_to_peer` moviéndose fuera
        del punto de entrega tras dejar el objeto) puedan reusarlo."""
        from protocol.action_contract import validate_action, ActionContractError

        agent = self.agent

        # Los beliefs/ASL normalizan los recipe id a minúscula, pero Unity
        # registra la receta con su case original (p.ej. "Bread_recipe"). Si
        # se envía "bread_recipe" Unity responde InvalidArgs/"No matching recipe".
        # Restaurar el case original del targetId desde el perfil antes de enviar.
        if action_type == "Craft" and isinstance(args.get("targetId"), str):
            profile = getattr(agent, "profile", None)
            recipes = getattr(profile, "recipes", None) if profile else None
            if recipes:
                tid_low = args["targetId"].lower()
                for r in recipes:
                    if r.recipeId.lower() == tid_low:
                        args = {**args, "targetId": r.recipeId}
                        break

        try:
            validate_action(action_type, args)
        except ActionContractError as exc:
            raise agentspeak.AslError(f"invalid_action:{exc}") from exc

        cmd_id = self._dispatch_unity_command(action_type, args)

        # 0.C4 — timeout por acción: ExploreArea (con WanderRetry) usa uno más largo.
        from config import settings
        if action_type == "ExploreArea":
            _timeout_s = float(settings.explore_timeout_s)
        else:
            _timeout_s = float(settings.action_timeout_s)

        intention.waiter = ActionResultWaiter(
            cmd_id=cmd_id,
            results=agent.action_results,
            intention=intention,
            npc_id=agent.npc_id,
            goal_sig=agent.intention.sig if agent.intention else None,
            action_type=action_type,
            bdi=self,
            timeout_s=_timeout_s,
            move_target=(int(args["x"]), int(args["y"])) if action_type == "MoveTo" else None,
        )

    def _retry_moveto_nearby(
        self, target: tuple[int, int], attempt: int, result: dict,
    ) -> str | None:
        """Fase 17s: reenvía un MoveTo que falló con PathNotFound (casilla ocupada
        por otro NPC, piloto 17r CO5: 12 PathNotFound al centro de bakeri con el
        baker parado encima). Devuelve el cmd_id del reintento o None si no aplica
        o se agotaron los destinos. Trazado como CODE (`moveto_retry`)."""
        error_code = result.get("errorCode") or (result.get("payload") or {}).get("errorCode")
        if error_code != "PathNotFound":
            return None
        x, y = target
        candidates = moveto_retry_candidates(
            x, y, include_neighbours=not self.agent.beliefs.has("item_at", None, x, y),
        )
        if attempt >= len(candidates):
            return None
        rx, ry = candidates[attempt]
        agent = self.agent
        _trace(
            "moveto_retry", npc_id=agent.npc_id,
            goal=agent.intention.sig if agent.intention else None,
            target=[x, y], retry=[rx, ry], attempt=attempt + 1, source="CODE",
        )
        return self._dispatch_unity_command("MoveTo", {"x": rx, "y": ry})

    def _dispatch_unity_command(self, action_type: str, args: dict) -> str:
        """Envía un ActionCommand a Unity y lo traza; devuelve el cmd_id."""
        agent = self.agent
        cmd_id = str(uuid.uuid4())
        # Recordar los args del comando para correlacionarlos con el
        # ActionResult (p.ej. la receta de un Craft — ver unity_events).
        agent.sent_commands[cmd_id] = {"actionType": action_type, "args": dict(args)}
        asyncio.ensure_future(agent.send_to_unity({
            "type": "ActionCommand",
            "msg_id": cmd_id,
            "commandId": cmd_id,
            "npcId": agent.npc_id,
            "actionType": action_type,
            "issuedTicks": 0,
            "timeoutTicks": 300,
            "args": args,
        }))
        _trace(
            "action_sent",
            npc_id=agent.npc_id,
            goal=agent.intention.sig if agent.intention else None,
            action=action_type,
            args=args,
            cmd_id=cmd_id,
        )
        tl = TraceLogger.get()
        if tl is not None:
            tl.metrics.npc(agent.npc_id).actions_sent += 1
        return cmd_id

    def _register_unity_actions(self) -> None:
        """Registra .moveto, .pickup, .drop, .craft, .explorearea, .search, .wait."""
        # Flag del catch-all de la escalera (lo activa la acción .request_replan).
        self._replan_requested = False
        # Planes cargados en el motor ASL por CLAVE de identidad (sig+call_args):
        # {clave: [(asp_key, plan), ...]}. Permite carga/descarga selectiva por
        # binding y que dos goals de la misma familia coexistan (ver _goal_key).
        if not hasattr(self, "_loaded_plans_by_key"):
            self._loaded_plans_by_key = {}

        _send_and_wait = self._send_and_wait

        # .moveto(X, Y)
        @self._actions.add(".moveto", 2)
        def _moveto(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            x = int(agentspeak.grounded(term.args[0], intention.scope))
            y = int(agentspeak.grounded(term.args[1], intention.scope))
            _send_and_wait("MoveTo", {"x": x, "y": y}, intention)
            yield

        # .pickup(ItemId)
        @self._actions.add(".pickup", 1)
        def _pickup(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            item_id = str(agentspeak.grounded(term.args[0], intention.scope))
            _send_and_wait("PickUp", {"itemId": item_id}, intention)
            yield

        # .drop(ItemId, Qty)
        @self._actions.add(".drop", 2)
        def _drop2(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            item_id = str(agentspeak.grounded(term.args[0], intention.scope))
            qty = int(agentspeak.grounded(term.args[1], intention.scope))
            _send_and_wait("Drop", {"itemId": item_id, "qty": qty}, intention)
            yield

        # .drop(ItemId, Qty, TargetId)
        @self._actions.add(".drop", 3)
        def _drop3(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            item_id = str(agentspeak.grounded(term.args[0], intention.scope))
            qty = int(agentspeak.grounded(term.args[1], intention.scope))
            target_id = str(agentspeak.grounded(term.args[2], intention.scope))
            _send_and_wait(
                "Drop",
                {"itemId": item_id, "qty": qty, "targetId": target_id},
                intention,
            )
            yield

        # .craft(ItemId, TargetId) — without optional qty
        @self._actions.add(".craft", 2)
        def _craft(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            item_id = str(agentspeak.grounded(term.args[0], intention.scope))
            target_id = str(agentspeak.grounded(term.args[1], intention.scope))
            _send_and_wait("Craft", {"itemId": item_id, "targetId": target_id}, intention)
            yield

        # .craft(ItemId, TargetId, Qty) — with optional qty
        @self._actions.add(".craft", 3)
        def _craft3(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            item_id = str(agentspeak.grounded(term.args[0], intention.scope))
            target_id = str(agentspeak.grounded(term.args[1], intention.scope))
            qty_raw = agentspeak.grounded(term.args[2], intention.scope)
            try:
                qty = int(qty_raw)
            except (TypeError, ValueError):
                qty = None
            args: dict = {"itemId": item_id, "targetId": target_id}
            if qty is not None:
                args["qty"] = qty
            _send_and_wait("Craft", args, intention)
            yield

        # .explorearea(ZoneTag)
        @self._actions.add(".explorearea", 1)
        def _explore(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            zone_tag = str(agentspeak.grounded(term.args[0], intention.scope))
            known_zone_rows = self.agent.beliefs.query("zone_center", zone_tag)
            if known_zone_rows:
                log.info(
                    "[BDI:%s] .explorearea(%s) omitido: zone_center ya conocido",
                    self.agent.npc_id,
                    zone_tag,
                )
                yield
                return
            _send_and_wait("ExploreArea", {"zoneTag": zone_tag}, intention)
            yield

        # .search(ItemId) — range is handled internally by Unity
        @self._actions.add(".search", 1)
        def _search(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            item_id = str(agentspeak.grounded(term.args[0], intention.scope))
            _send_and_wait("Search", {"itemId": item_id}, intention)
            yield

        # .wait(Ticks) — Unity Wait action (overrides stdlib .wait)
        @self._actions.add(".wait", 1)
        def _wait(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            ticks = int(agentspeak.grounded(term.args[0], intention.scope))
            _send_and_wait("Wait", {"ticks": ticks}, intention)
            yield

        # .request_replan — acción interna del catch-all de la escalera de
        # variantes. La dispara la rama final `+!sig : true <- .request_replan.`
        # cuando NINGUNA otra variante casa con el estado actual → señala al ciclo
        # BDI que debe pedir un plan nuevo (en vez de dejar el goal sin plan).
        @self._actions.add(".request_replan", 0)
        def _request_replan(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            self._replan_requested = True
            yield

    # --- registro de acciones de coordinación NPC↔NPC (Fase 12) ------------

    def _register_peer_actions(self) -> None:
        """Registra .ask_peer/.request_peer/.await_peer/.deliver_to_peer.

        Solo se llama si `settings.coordination_enabled` (ver on_start). No
        son primitivas de mundo: no envían nada a Unity, no tocan
        `PRIMITIVE_ACTIONS`/`ACTION_ALLOWLIST`. Hablan por XMPP directo con
        otro `NPCAgent` (mismo patrón que `llm_behaviour.py` con el
        planificador), con el mismo mecanismo de Waiter que las acciones
        Unity (bloquean la intención hasta la respuesta o el timeout).
        """
        from protocol.peer_messages import PeerMessage, PROTOCOL_METADATA

        # .ask_peer(Npc, Pred, Arg) — query-if + espera inform, escribe belief.
        @self._actions.add(".ask_peer", 3)
        def _ask_peer(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            from config import settings

            npc = str(agentspeak.grounded(term.args[0], intention.scope))
            pred = str(agentspeak.grounded(term.args[1], intention.scope))
            arg = agentspeak.grounded(term.args[2], intention.scope)
            arg = str(arg) if isinstance(arg, agentspeak.Literal) else arg

            agent = self.agent
            cid = str(uuid.uuid4())
            payload = PeerMessage(
                performative="query-if", conversation_id=cid, pred=pred, args=[arg],
            ).to_dict()
            msg = spade.message.Message(
                to=f"{npc}@localhost", metadata=dict(PROTOCOL_METADATA), body=json.dumps(payload),
            )
            asyncio.ensure_future(self.send(msg))
            tl = TraceLogger.get()
            if tl is not None:
                tl.metrics.npc(agent.npc_id).peer_msgs_sent += 1

            def _resolve(pmsg) -> None:
                if pmsg is None:
                    _trace("peer_query_timeout", npc_id=agent.npc_id, to=npc, pred=pred, arg=arg)
                    self._pending_failure = "peer_query_timeout"
                    return
                value = pmsg.value
                if pred == "can_make":
                    agent.beliefs.apply_peer_can_make(npc, str(arg), bool(value))
                elif pred == "has_item":
                    agent.beliefs.apply_peer_has_item(npc, str(arg), int(value or 0))
                elif pred == "knows_zone":
                    agent.beliefs.apply_peer_knows_zone(npc, str(arg), bool(value))
                elif pred == "busy":
                    agent.beliefs.apply_peer_busy(npc, bool(value))

            intention.waiter = PeerReplyWaiter(cid, agent, _resolve, settings.peer_query_timeout_s)
            yield

        # .request_peer(Npc, GoalSig, Item, Qty) — request + espera agree/refuse.
        # MVP acotado a condiciones has_item(Item, Qty) (la única familia con
        # variante `delegate`, ver llm/family_plan.py) — construir el string de
        # condición aquí (en Python, con Item/Qty ya ligados) en vez de intentar
        # interpolar variables dentro de un literal ASL, que agentspeak no hace.
        @self._actions.add(".request_peer", 4)
        def _request_peer(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            from config import settings

            npc = str(agentspeak.grounded(term.args[0], intention.scope))
            goal_sig = str(agentspeak.grounded(term.args[1], intention.scope))
            item = str(agentspeak.grounded(term.args[2], intention.scope))
            qty = int(agentspeak.grounded(term.args[3], intention.scope))

            agent = self.agent
            cid = str(uuid.uuid4())
            # Profundidad heredada: si la intención actual YA viene de una
            # delegación de otro peer, este request es un salto MÁS en la
            # cadena (anti-bucle, ver settings.peer_max_depth).
            inherited_depth = getattr(agent.intention, "peer_request_depth", 0) or 0
            depth = inherited_depth + 1
            condition = f"has_item({item}, {qty})"
            payload = PeerMessage(
                performative="request", conversation_id=cid, depth=depth,
                goal_sig=goal_sig, condition=condition,
            ).to_dict()
            msg = spade.message.Message(
                to=f"{npc}@localhost", metadata=dict(PROTOCOL_METADATA), body=json.dumps(payload),
            )
            # Fase 17o: la petición nueva empieza sin el FALLO de la anterior al mismo
            # peer y goal. Se limpia al ENVIAR, no al recibir el agree: el resultado de
            # esta petición solo puede llegar después, mientras que PeerCoord puede
            # procesar un inform-done antes de que el BDI lea el agree (E6 se colgaba).
            agent.beliefs.clear_peer_request(npc, goal_sig)
            asyncio.ensure_future(self.send(msg))
            _trace(
                "peer_request_sent", npc_id=agent.npc_id, to=npc,
                goal_sig=goal_sig, condition=condition, depth=depth,
            )
            tl = TraceLogger.get()
            if tl is not None:
                m = tl.metrics.npc(agent.npc_id)
                m.peer_msgs_sent += 1
                m.peer_requests_sent += 1

            def _resolve(pmsg) -> None:
                if pmsg is None:
                    _trace("peer_request_timeout", npc_id=agent.npc_id, to=npc, goal_sig=goal_sig)
                    self._pending_failure = "peer_request_timeout"
                    return
                if pmsg.performative == "agree":
                    agent.beliefs.apply_peer_promised(npc, goal_sig)
                else:
                    reason = pmsg.reason or "unknown"
                    agent.beliefs.apply_peer_refused(npc, goal_sig, reason)
                    self._pending_failure = f"peer_refused:{reason}"

            intention.waiter = PeerReplyWaiter(cid, agent, _resolve, settings.peer_request_timeout_s)
            yield

        # .await_peer(Npc, GoalSig, TimeoutS) — bloquea hasta peer_done/peer_failed.
        @self._actions.add(".await_peer", 3)
        def _await_peer(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            from config import settings

            npc = str(agentspeak.grounded(term.args[0], intention.scope))
            goal_sig = str(agentspeak.grounded(term.args[1], intention.scope))
            timeout_raw = agentspeak.grounded(term.args[2], intention.scope)
            try:
                timeout_s = float(timeout_raw)
            except (TypeError, ValueError):
                timeout_s = float(settings.peer_request_timeout_s)

            agent = self.agent
            t0 = time.time()

            def _resolve(success: bool, reason: str | None) -> None:
                latency = round(time.time() - t0, 3)
                tl = TraceLogger.get()
                if tl is not None:
                    tl.metrics.npc(agent.npc_id).peer_wait_s += latency
                if not success:
                    ev = "peer_wait_timeout" if reason == "timeout" else "peer_request_failed_recv"
                    _trace(ev, npc_id=agent.npc_id, peer=npc, goal_sig=goal_sig, reason=reason)
                    self._pending_failure = f"peer_wait_failed:{reason}"

            # Fase 17x: esperar un encargo que ese peer no ha prometido (rechazado, o
            # etiqueta goalSig distinta de la del request) no puede terminar bien;
            # antes consumía el timeout entero (300 s) antes de replanificar.
            beliefs = agent.beliefs
            npc_l = npc.lower()
            if not (
                beliefs.has("peer_promised", npc_l, goal_sig)
                or beliefs.has("peer_done", npc_l, goal_sig)
                or beliefs.query("peer_failed", npc_l, goal_sig, None)
                or beliefs.query("peer_refused", npc_l, goal_sig, None)
            ):
                _resolve(False, "not_promised")
                yield
                return

            intention.waiter = PeerDoneWaiter(npc, goal_sig, agent, _resolve, timeout_s)
            yield

        # .deliver_to_peer(Requester, Item, Qty) — SOLO tras .drop: avisa al
        # peticionario dónde recoger (Nivel 1: drop+aviso en un mensaje) y
        # escribe la creencia local que verifica el cierre del propio goal
        # `achieve_deliver_to_peer` (success_condition = delivered_to_peer/3).
        #
        # Captura x,y ANTES de moverse (misma posición que el .drop anterior
        # — es la ubicación real del objeto, la que necesita el peticionario)
        # y SOLO DESPUÉS dispara el "apartarse" (hallazgo Fase 13, sesión E5
        # 2026-08-11: el que entrega se queda quieto en el punto de entrega y
        # bloquea el pathfinding del peticionario). El "apartarse" se manda
        # como comando de Unity independiente (self._send_and_wait, fuera del
        # ciclo de la intención): en cuanto se fija `delivered_to_peer` el
        # goal se completa y `run()` limpia las intenciones — el ActionCommand
        # YA se envió y Unity lo ejecuta igual, pero nada en el BDI espera su
        # resultado (unity_events.py sigue actualizando current_position al
        # recibirlo, no requiere un waiter activo).
        @self._actions.add(".deliver_to_peer", 3)
        def _deliver_to_peer(
            asp_agent: asp_runtime.Agent,
            term: agentspeak.Literal,
            intention: asp_runtime.Intention,
        ):
            requester = str(agentspeak.grounded(term.args[0], intention.scope))
            item = str(agentspeak.grounded(term.args[1], intention.scope))
            qty = int(agentspeak.grounded(term.args[2], intention.scope))

            agent = self.agent
            pos_rows = agent.beliefs.query("current_position")
            x, y = (pos_rows[0][0], pos_rows[0][1]) if pos_rows else (None, None)
            # goal_sig que el peticionario reconoce en su .await_peer — heredado
            # del request original (ver PeerCoordBehaviour._handle_request).
            goal_sig = getattr(agent.intention, "peer_origin_sig", None) or "achieve_has_item"

            payload = PeerMessage(
                performative="inform-done", conversation_id=str(uuid.uuid4()),
                goal_sig=goal_sig, item=item, qty=qty, x=x, y=y,
            ).to_dict()
            msg = spade.message.Message(
                to=f"{requester}@localhost", metadata=dict(PROTOCOL_METADATA),
                body=json.dumps(payload),
            )
            asyncio.ensure_future(self.send(msg))
            _trace(
                "peer_transfer", npc_id=agent.npc_id, to=requester,
                item=item, qty=qty, x=x, y=y,
            )
            tl = TraceLogger.get()
            if tl is not None:
                m = tl.metrics.npc(agent.npc_id)
                m.peer_msgs_sent += 1
                m.items_given += 1

            # Apartarse a (0,0) — única celda transitable verificada en
            # decenas de sesiones reales de este mundo. El goal se cierra
            # inmediatamente después (belief ya se fija a continuación), así
            # que nada espera a que este MoveTo resuelva — es best-effort,
            # pero el comando ya viaja a Unity.
            self._send_and_wait("MoveTo", {"x": 0, "y": 0}, intention)

            agent.beliefs._set("delivered_to_peer", (requester, item, int(qty)))
            agent.beliefs._recent_additions.append("delivered_to_peer")
            yield

    # --- builtin plan loading -----------------------------------------------

    async def _load_builtin_plans(self) -> None:
        """Carga planes built-in ya parseados en plan_graph dentro del runtime ASP."""
        loaded = 0
        for _sig, attrs in self.agent.plan_graph.nodes(data=True):
            node = attrs.get("data") if isinstance(attrs, dict) else None
            if not isinstance(node, GoalNode) or not node.is_builtin:
                continue
            if self._is_plan_loaded(node.sig):
                continue

            before = sum(
                len(plans) for key, plans in self._asp.plans.items() if key[2] == node.sig
            )
            self._load_plans_into_asp(node)
            after = sum(
                len(plans) for key, plans in self._asp.plans.items() if key[2] == node.sig
            )
            loaded += max(0, after - before)

        if loaded:
            log.info("[BDI] Cargados %d planes built-in desde plan_graph", loaded)

    # --- belief sync --------------------------------------------------------

    def _sync_beliefs(self) -> None:
        """Sincroniza BeliefStore → asp_agent.beliefs (pull model por ciclo).

        CONTRATO (0.C2): las beliefs del runtime ASL se RECONSTRUYEN por completo
        desde BeliefStore en cada tick (clear + re-add). Por tanto los planes ASL
        NO deben usar +belief/-belief para mantener estado persistente: cualquier
        cambio que hagan se descarta en el siguiente tick. El estado canónico vive
        en BeliefStore y se actualiza vía los eventos de Unity (UnityEventBehaviour).
        El validador (validate_plan_asl) avisa si un body usa +/-belief.
        """
        self._asp.beliefs.clear()
        for pred, tuples in self.agent.beliefs.snapshot().items():
            for tup in tuples:
                args = tuple(
                    agentspeak.Literal(str(v), (), frozenset()) if isinstance(v, str) else v
                    for v in tup
                )
                term = agentspeak.Literal(pred, args, frozenset())
                self._asp.beliefs[(pred, len(args))].add(term)

    # --- plan management ----------------------------------------------------

    def _is_plan_loaded(self, key: str) -> bool:
        """¿Están cargados los planes de ESTE goal/binding? Se rastrea por clave de
        identidad (sig+call_args), no por functor: dos bindings de la misma familia
        comparten functor en el motor ASL, así que la presencia por functor no
        distingue bread de wheat. `_loaded_plans_by_key` mapea clave → planes."""
        return key in self._loaded_plans_by_key

    def _unload_plan(self, key: str) -> None:
        """Descarga SOLO los planes de este goal/binding (no los de un hermano de la
        misma familia): quita los objetos plan concretos que se cargaron bajo esta
        clave de sus buckets del motor ASL."""
        for asp_key, plan in self._loaded_plans_by_key.pop(key, []):
            bucket = self._asp.plans.get(asp_key)
            if bucket is not None:
                try:
                    bucket.remove(plan)
                except ValueError:
                    pass

    def _load_plans_into_asp(self, node: GoalNode, key: str | None = None) -> None:
        """Compila y carga las variantes del GoalNode en asp_agent.

        `key` = clave de identidad (sig+call_args) bajo la que se rastrean los planes
        para carga/descarga selectiva por binding. Por defecto `node.sig` (builtins
        y sub-goals, sin binding)."""
        key = key or node.sig
        tracked: list = self._loaded_plans_by_key.setdefault(key, [])
        loaded = 0
        for variant in node.variants:
            asl_text = self._make_asl_from_variant(node, variant)
            if not asl_text:
                continue
            plan = compile_plan_with_actions(asl_text, self._actions)
            if plan is None:
                log.warning(
                    "[BDI:%s] No se pudo compilar variante de '%s': %s",
                    self.agent.npc_id,
                    node.sig,
                    asl_text[:80],
                )
                continue
            asp_key = (plan.trigger, plan.goal_type, plan.head.functor, len(plan.head.args))
            self._asp.plans[asp_key].append(plan)
            tracked.append((asp_key, plan))
            loaded += 1
        log.debug(
            "[BDI:%s] Planes cargados para '%s': %d variante(s)",
            self.agent.npc_id,
            node.sig,
            loaded,
        )
        # No-silencio (Fase 6.5): si la familia es canónica (goal del LLM con
        # param_names), las cabezas aridad-0 del pipeline se parametrizaron
        # (transform estructural CODE). Se traza una vez por carga. Los builtins ya
        # son paramétricos de fábrica (no hay transform) → se excluyen.
        node_params = getattr(node, "param_names", []) or []
        if node_params and not getattr(node, "is_builtin", False):
            _trace(
                "canonical_family_heads",
                npc_id=self.agent.npc_id, goal=node.sig,
                params=node_params, variants=loaded, source="CODE",
            )
        # Catch-all de la escalera: una rama final `: true <- .request_replan` que
        # AgentSpeak prueba la ÚLTIMA y solo dispara cuando ninguna variante real
        # casa. Permite que la escalera de variantes avance re-inyectando el goal
        # (sin LLM) y solo replanifica cuando de verdad no hay rama aplicable. Solo
        # para goals principales — los sub-planes builtin no se replanifican así.
        if not getattr(node, "is_builtin", False):
            cap = self._load_catch_all_variant(node)
            if cap is not None:
                tracked.append(cap)

    def _head_args(self, node: GoalNode) -> str:
        """Args de la CABEZA del plan episódico: el binding GROUND (e.g. 'bread, 1').

        Con la cabeza ground (`+!achieve_has_item(bread, 1)`), el goal despachado
        `achieve_has_item(bread, 1)` casa SOLO con el plan de bread y no con el de
        wheat → dos bindings de la misma familia COEXISTEN sin contaminarse en el
        motor ASL (mismo functor+aridad). Cae a `param_names` (familia paramétrica
        de T4, que ya trae su cabeza) si no hay call_args. Vacío = sin reuso canónico.
        """
        call_args = getattr(node, "call_args", []) or []
        if call_args:
            return ", ".join(str(a) for a in call_args)
        param_names = getattr(node, "param_names", []) or []
        return ", ".join(param_names)

    def _load_catch_all_variant(self, node: GoalNode) -> None:
        """Carga `+!sig(binding) : true <- .request_replan.` como última variante.

        Con binding ground el catch-all solo dispara para ESTE goal (no para otro
        binding de la misma familia)."""
        args = self._head_args(node)
        head_args = f"({args})" if args else ""
        asl = f"+!{node.sig}{head_args} : true <- .request_replan."
        plan = compile_plan_with_actions(asl, self._actions)
        if plan is None:
            return None
        key = (plan.trigger, plan.goal_type, plan.head.functor, len(plan.head.args))
        self._asp.plans[key].append(plan)  # appended LAST → AgentSpeak lo prueba el último
        return (key, plan)

    def _with_head_binding(self, asl_text: str, node: GoalNode) -> str:
        """Inyecta el binding GROUND en la CABEZA del asl si no lo lleva ya:
        `+!sig` → `+!sig(bread, 1)`. Transform ESTRUCTURAL (no toca guard ni cuerpo)
        para que la aridad case con el goal despachado y el plan sea exclusivo de su
        binding. Sin call_args/param_names o si la cabeza ya tiene args → intacto."""
        args = self._head_args(node)
        if not args:
            return asl_text
        prefix = f"+!{node.sig}"
        if not asl_text.startswith(prefix):
            return asl_text  # forma inesperada → no tocar
        rest = asl_text[len(prefix):]
        if rest[:1] == "(":
            return asl_text  # cabeza ya con args (familia paramétrica o ya reescrita)
        return f"{prefix}({args}){rest}"

    def _make_asl_from_variant(self, node: GoalNode, variant: PlanVariant) -> str:
        """Devuelve el texto ASL de la variante; reconstruye si full_asl está vacío."""
        if variant.full_asl:
            # El pipeline genera cabezas aridad-0 (`+!sig : guard <- body`). Bajo
            # reuso canónico se les inyecta el binding ground para casar el dispatch.
            return self._with_head_binding(variant.full_asl.strip(), node)
        # Reconstruir desde guard + steps
        _hargs = self._head_args(node)
        head_args = f"({_hargs})" if _hargs else ""
        guard = (variant.guard or "true").strip()
        if not variant.steps:
            # Fase 6.5: NO se fabrica un cuerpo `true` que el LLM no puso. Variante
            # malformada (sin asl y sin steps) → se descarta de forma VISIBLE.
            from utils.trace_logger import plan_transform as _pt
            _pt(
                "empty_variant_skipped", "CODE", npc_id=self.agent.npc_id,
                reason=f"variante de '{node.sig}' sin asl ni steps — descartada (no se fabrica body)",
            )
            log.warning(
                "[BDI:%s] Variante de '%s' sin cuerpo — descartada (no se fabrica)",
                self.agent.npc_id, node.sig,
            )
            return ""
        body = ";\n    ".join(variant.steps)
        return f"+!{node.sig}{head_args} : {guard} <-\n    {body}."

    def _persist_generated_plan(self, node: GoalNode) -> None:
        """Guarda las variantes del goal en plans/memory/<npc_id>/asl/ y refresca _bundle.asl."""
        plan_memory = getattr(self.agent, "plan_memory", None)
        if plan_memory is None or not node.variants:
            return

        variants_payload = [
            {
                "guard": variant.guard,
                "steps": list(variant.steps),
                "full_asl": self._make_asl_from_variant(node, variant),
            }
            for variant in node.variants
        ]

        try:
            # El FICHERO se indexa por clave de identidad (sig+call_args) para que
            # dos bindings de la misma familia (bread y wheat) no se pisen en disco;
            # `functor` guarda el sig real (achieve_has_item) para reconstruir el nodo.
            from llm.canonical import goal_identity_key
            node_call_args = list(getattr(node, "call_args", []) or [])
            store_key = goal_identity_key(node.sig, node_call_args)
            plan_memory.store(
                goal_sig=store_key,
                variants=variants_payload,
                main_asl=variants_payload[0]["full_asl"],
                description=node.description,
                param_names=list(node.param_names),
                call_args=node_call_args,
                functor=node.sig,
            )
            plan_memory.rebuild_bundle_from_graph(self.agent.plan_graph)
        except Exception as exc:
            log.warning(
                "[BDI:%s] No se pudo persistir plan '%s' en plan memory: %s",
                self.agent.npc_id,
                node.sig,
                exc,
            )

    def _record_plan_memory_success(self, goal_sig: str) -> None:
        plan_memory = getattr(self.agent, "plan_memory", None)
        if plan_memory is None:
            return
        try:
            plan_memory.record_success(goal_sig)
        except Exception as exc:
            log.warning(
                "[BDI:%s] No se pudo registrar éxito de '%s' en plan memory: %s",
                self.agent.npc_id,
                goal_sig,
                exc,
            )

    def _record_plan_memory_unverified(self, goal_sig: str) -> None:
        plan_memory = getattr(self.agent, "plan_memory", None)
        if plan_memory is None:
            return
        try:
            plan_memory.record_unverified(goal_sig)
        except Exception as exc:
            log.warning(
                "[BDI:%s] No se pudo registrar cierre unverified de '%s': %s",
                self.agent.npc_id, goal_sig, exc,
            )

    def _record_plan_memory_failure(self, goal_sig: str) -> None:
        plan_memory = getattr(self.agent, "plan_memory", None)
        if plan_memory is None:
            return
        try:
            plan_memory.record_failure(goal_sig)
        except Exception as exc:
            log.warning(
                "[BDI:%s] No se pudo registrar fallo de '%s' en plan memory: %s",
                self.agent.npc_id,
                goal_sig,
                exc,
            )

    # --- goal injection -----------------------------------------------------

    def _inject_goal(self, goal: Any) -> None:
        """Inyecta el goal activo en asp_agent. Lanza AslError si ningún plan aplica."""
        call_args: list = getattr(goal, "call_args", []) or []
        args = tuple(
            agentspeak.Literal(str(a), (), frozenset()) if isinstance(a, str) else a
            for a in call_args
        )
        term = agentspeak.Literal(goal.sig, args, frozenset())
        base_intention = asp_runtime.Intention()
        self._asp.call(
            agentspeak.Trigger.addition,
            agentspeak.GoalType.achievement,
            term,
            base_intention,
        )

    # --- arbitraje de goals: aparcar/restaurar intención (Fase 14) -----------
    #
    # `agentspeak.runtime.Agent.step()` YA soporta varias pilas de intención
    # concurrentes (`self._asp.intentions` es un `deque` de `intention_stack`,
    # y step() salta automáticamente cualquiera bloqueada en un waiter no
    # resuelto — ver DOC/PLAN_EJECUCION/FASE_14_ARBITRAJE_LLM.md §1). Lo único
    # que faltaba era que el envoltorio de este bucle (`run()`) dejara de asumir
    # una única `agent.intention`. Park/restore NO introduce concurrencia real
    # (nunca hay más de UNA pila viva en `self._asp.intentions` a la vez): es
    # una preempción cooperativa de un único goal bloqueado por otro, no un
    # scheduler general. Por eso los ~8 `self._asp.intentions.clear()`
    # existentes en el resto del bucle siguen siendo correctos sin tocarlos.

    def _park_intention(self, goal: Any) -> None:
        """Saca la pila de intención activa (si la hay) de `self._asp.intentions`
        y la guarda en el propio `goal`. Llamado SOLO desde el punto de
        preempción (`_maybe_arbitrate`) justo antes de cambiar `agent.intention`
        a otro goal — el motor de agentspeak queda vacío, listo para que el
        run() del siguiente tick inyecte el goal entrante desde cero."""
        goal._parked_stack = list(self._asp.intentions)
        goal._parked_at = time.time()
        self._asp.intentions.clear()

    def _restore_intention(self, goal: Any) -> bool:
        """Si `goal` tiene una pila aparcada, la reinserta en `self._asp.intentions`
        y desplaza el deadline de cualquier waiter activo en su frame superior
        por el tiempo que estuvo aparcada (ver Waiter.shift_deadline — sin esto
        el waiter expiraría de inmediato al volver). Devuelve True si había algo
        que restaurar. Debe llamarse ANTES de que run() compruebe
        `if not self._asp.intentions: self._inject_goal(...)`, para que esa
        inyección se salte y el plan continúe donde lo dejó."""
        stack = getattr(goal, "_parked_stack", None)
        if not stack:
            return False
        parked_at = getattr(goal, "_parked_at", None) or time.time()
        elapsed = time.time() - parked_at
        for intention_stack in stack:
            if intention_stack:
                top = intention_stack[-1]
                waiter = getattr(top, "waiter", None)
                if waiter is not None and hasattr(waiter, "shift_deadline"):
                    waiter.shift_deadline(elapsed)
            self._asp.intentions.append(intention_stack)
        goal._parked_stack = None
        goal._parked_at = None
        _trace(
            "goal_resumed",
            npc_id=self.agent.npc_id,
            goal=goal.sig,
            parked_s=round(elapsed, 3),
        )
        return True

    # --- punto de preempción (Fase 14) ---------------------------------------

    async def _maybe_arbitrate(self, goal: Any) -> bool:
        """Punto de preempción. Devuelve True si cambió `agent.intention` (el
        caller de `run()` debe hacer return de inmediato; el siguiente tick
        procesa el goal nuevo por el camino normal — no hace falta código
        especial, `_goal_key`→`plan_graph`→inyectar ya sabe tratarlo). False
        (no-op) salvo que TODAS estas condiciones se cumplan, de la más barata
        a la más cara — así el caso normal (E1-E5, sin conflicto) sale por la
        primera condición sin tocar nada más:

          1. `goal_arbitration_enabled` y `coordination_enabled`.
          2. cooldown respetado (`arbitration_cooldown_s`).
          3. `arbitration_max_switches` no agotado.
          4. hay ≥1 goal candidato distinto del activo en `agent.goals`.
          5. la intención activa está bloqueada en un `PeerDoneWaiter`
             (esperando el CIERRE del goal de un peer vía `.await_peer`) — NO
             un `ActionResultWaiter` de Unity (el NPC está a mitad de una
             acción real, no preemptible) ni un `PeerReplyWaiter` de
             query/agree (espera corta, no el punto muerto que motiva esta
             fase — ver FASE_14_ARBITRAJE_LLM.md §1).
        """
        from config import settings

        if not settings.goal_arbitration_enabled or not settings.coordination_enabled:
            return False

        now = time.time()
        if now - self._last_arbitration_ts < settings.arbitration_cooldown_s:
            return False

        candidates = [g for g in self.agent.goals if not self._same_goal(g, goal)]
        if not candidates:
            return False

        # Fase 17w: el tope evita el ping-pong entre goals DISTINTOS. Volver al mismo
        # goal al que ya se cambió (el mismo encargo tras un replan, o su reintento)
        # no lo gasta: en CO6/ATOM un solo encargo con 3 replans agotó los 4 cambios
        # y, al llegar el reintento, miller y baker se esperaron mutuamente 800 s.
        last_key = getattr(self, "_last_switch_key", None)
        at_cap = self._arbitration_switch_count >= settings.arbitration_max_switches
        if at_cap and not any(self._goal_key(c) == last_key for c in candidates):
            return False

        if not self._asp.intentions:
            return False
        intention_stack = self._asp.intentions[0]
        if not intention_stack:
            return False
        top = intention_stack[-1]
        waiter = getattr(top, "waiter", None)
        if not isinstance(waiter, PeerDoneWaiter):
            return False

        waiting_on_npc = getattr(waiter, "_npc_id", None)
        if not waiting_on_npc:
            return False
        self._last_arbitration_ts = now

        chosen, source, reason = await self._decide_arbitration(goal, waiting_on_npc, candidates)
        waited_s = round(now - getattr(waiter, "_t0", now), 3)
        _trace(
            "arbitration_decision",
            npc_id=self.agent.npc_id,
            from_goal=goal.sig,
            to_goal=chosen.sig if chosen else None,
            decision="switch" if chosen else "wait",
            source=source,
            reason=reason,
            waiting_on=waiting_on_npc,
            waited_s=waited_s,
        )
        tl = TraceLogger.get()
        if tl is not None:
            m = tl.metrics.npc(self.agent.npc_id)
            if source == "LLM":
                m.arbitrations_llm += 1
            elif source == "RULE_FALLBACK":
                m.arbitrations_fallback += 1
            else:
                m.arbitrations_rule += 1

        if chosen is None:
            return False

        chosen_key = self._goal_key(chosen)
        if at_cap and chosen_key != last_key:
            return False

        self._park_intention(goal)
        self.agent.intention = chosen
        if chosen_key != last_key:
            self._arbitration_switch_count += 1
        self._last_switch_key = chosen_key
        if tl is not None:
            tl.metrics.npc(self.agent.npc_id).goal_switches += 1
        log.info(
            "[BDI:%s] Arbitraje (%s): '%s' -> '%s' (esperando a %s, %.1fs) — %s",
            self.agent.npc_id, source, goal.sig, chosen.sig, waiting_on_npc, waited_s, reason,
        )
        return True

    async def _decide_arbitration(
        self, goal: Any, waiting_on_npc: str, candidates: list[Any],
    ) -> tuple[Any | None, str, str]:
        """Despacha según `goal_arbitration_mode`. Modo "rule": SIEMPRE la regla
        determinista. Modo "llm": consulta la tarea LLM `arbitrate`; si falla,
        da timeout, o el resultado no valida (`_validate_arbitrate` — incluye
        rechazar un `goal_sig` que no esté entre los candidatos ofrecidos, ver
        política de no-fabricación del proyecto), cae a la regla SIEMPRE — el
        LLM es la política, nunca el mecanismo de seguridad (§2.1 del plan)."""
        from config import settings

        if settings.goal_arbitration_mode == "rule":
            chosen = self._arbitrate_by_rule(waiting_on_npc, candidates)
            return chosen, "RULE", ("cycle_detected" if chosen else "no_cycle")

        t0 = time.time()
        try:
            chosen, reason = await self._arbitrate_via_llm(goal, waiting_on_npc, candidates)
            return chosen, "LLM", reason
        except Exception as exc:
            log.warning(
                "[BDI:%s] Error en arbitrate (LLM) -> fallback a regla: %s",
                self.agent.npc_id, exc,
            )
            chosen = self._arbitrate_by_rule(waiting_on_npc, candidates)
            return chosen, "RULE_FALLBACK", f"llm_error:{exc}"
        finally:
            tl = TraceLogger.get()
            if tl is not None:
                tl.metrics.npc(self.agent.npc_id).arbitration_llm_s += time.time() - t0

    def _arbitrate_by_rule(self, waiting_on_npc: str, candidates: list[Any]) -> Any | None:
        """Regla determinista (T5) — red de seguridad SIEMPRE activa como
        fallback del modo "llm", y único mecanismo en modo "rule". Detecta un
        ciclo de espera de longitud 2: si estoy esperando a `waiting_on_npc` y
        tengo un goal pendiente que ESE MISMO peer me pidió
        (`peer_requester_jid` normalizado == waiting_on_npc), cambio a él — es
        justo lo que `waiting_on_npc` necesita de mí para poder, a su vez,
        resolver lo que yo le pedí. Sin el cambio ninguno de los dos avanza
        (ver EVALUACION_RESULTADOS.md §5, causa raíz de E6 real sin arbitraje).
        Determinista, decidible localmente, sin LLM."""
        from npc.behaviours.peer_coord import _npc_id_from_jid

        for g in candidates:
            requester = getattr(g, "peer_requester_jid", None)
            if requester and _npc_id_from_jid(requester) == waiting_on_npc:
                return g
        return None

    async def _arbitrate_via_llm(
        self, goal: Any, waiting_on_npc: str, candidates: list[Any],
    ) -> tuple[Any | None, str]:
        """Consulta la tarea LLM `arbitrate` (T4): le da al modelo el goal
        activo (a quién espera, desde hace cuánto) y los candidatos
        (incluyendo, si aplica, quién los pidió), y decide `wait` o `switch`.
        Lanza si `run_llm_task` falla o si el `goal_sig` devuelto no está entre
        los candidatos ofrecidos (el caller ya sabe caer a la regla)."""
        from npc.behaviours.peer_coord import _npc_id_from_jid

        agent = self.agent
        intention_stack = self._asp.intentions[0]
        waiter = intention_stack[-1].waiter
        waited_s = round(time.time() - getattr(waiter, "_t0", time.time()), 1)

        payload = {
            "task": "arbitrate",
            "npc_id": agent.npc_id,
            "current_goal": {
                "sig": goal.sig,
                "condition": goal.success_condition,
                "waiting_on": waiting_on_npc,
                "waited_s": waited_s,
            },
            "candidates": [
                {
                    "sig": g.sig,
                    "condition": g.success_condition,
                    "goal_source": getattr(g, "goal_source", None),
                    "requested_by": (
                        _npc_id_from_jid(g.peer_requester_jid)
                        if getattr(g, "peer_requester_jid", None) else None
                    ),
                }
                for g in candidates
            ],
            "beliefs": agent.beliefs.snapshot(),
        }
        result = await agent.run_llm_task(payload)
        decision = result.get("decision")
        reason = result.get("reason", "") or ""
        if decision == "switch":
            goal_sig = result.get("goal_sig")
            for g in candidates:
                if g.sig == goal_sig:
                    return g, reason or "llm_switch"
            # No debería pasar: _validate_arbitrate exige que goal_sig esté
            # entre los ofrecidos. Defensivo — tratar como error del LLM.
            raise ValueError(f"arbitrate devolvió goal_sig no ofrecido: {goal_sig!r}")
        return None, reason or "llm_wait"

    # --- error handling -----------------------------------------------------

    async def _handle_asp_error(
        self, goal: Any, node: GoalNode, exc: agentspeak.AslError
    ) -> None:
        """Clasifica y maneja errores de agentspeak provenientes de step()."""
        err = str(exc)
        self._asp.intentions.clear()

        if "no applicable plan" in err:
            await self._handle_no_applicable_variant(goal)
            return

        if "unity_action_failed:" in err:
            error_code = err.split("unity_action_failed:", 1)[1].strip()
        elif "invalid_action:" in err:
            error_code = "invalid_action"
        else:
            error_code = "plan_failure"

        log.warning("[BDI:%s] Error ASP en '%s': %s", self.agent.npc_id, goal.sig, error_code)
        await self._handle_action_failure(goal, {
            "status": "Failure",
            "errorCode": error_code,
            "errorMessage": err,
        })

    # --- guard evaluation (success_condition) --------------------------------

    def _eval_guard(self, guard: str) -> bool:
        ok, _ = _guard_evaluator.eval_guard(
            guard=guard,
            param_names=[],
            call_args=[],
            snapshot=self.agent.beliefs.snapshot(),
        )
        return ok

    # --- intention selection (LLM prioritization) ----------------------------

    async def _select_intention(self) -> None:
        agent = self.agent
        unranked = [g for g in agent.goals if g.priority is None]

        if unranked and len(agent.goals) > 1:
            try:
                ranked = await agent.run_llm_task({
                    "task": "prioritize",
                    "goals": [{"sig": g.sig} for g in agent.goals],
                    "beliefs": agent.beliefs.snapshot(),
                })
                for item in ranked:
                    for g in agent.goals:
                        if g.sig == item.get("sig"):
                            g.priority = item.get("score", 0.0)
            except Exception as exc:
                log.warning(f"[BDI:{agent.npc_id}] Error en prioritize: {exc}")
                for g in unranked:
                    g.priority = 0.0

        ready = sorted(
            agent.goals,
            key=lambda g: g.priority or 0.0,
            reverse=True,
        )
        agent.intention = ready[0] if ready else None
        if agent.intention is not None:
            # Fase 14: si el goal elegido tiene una pila aparcada (fue
            # preemptado por otro goal y ahora vuelve a tener turno),
            # restaurarla en vez de re-inyectar desde cero.
            self._restore_intention(agent.intention)
            _trace(
                "goal_selected",
                npc_id=agent.npc_id,
                goal=agent.intention.sig,
                priority=agent.intention.priority,
            )
        await agent.push_status()

    # --- plan synthesis (LLM) -----------------------------------------------

    async def _try_canonical_family_plan(self, goal: Any, node: GoalNode) -> bool:
        """Fase 6.5 (T4): intenta planificar un goal de la familia `has_item` con la
        familia paramétrica DETERMINISTA (`llm.family_plan`), sin tocar el LLM.

        Solo actúa bajo `canonical_reuse_enabled` y si el `sig` del goal es la
        familia. Si el mundo no aporta ninguna variante (sin recetas ni spawns),
        devuelve False y el flujo cae al pipeline LLM normal. Devuelve True si cargó
        la familia (el run loop la compilará en asp_agent vía _load_plans_into_asp).
        """
        from config import settings
        from llm.family_plan import build_has_item_family, FAMILY_SIG

        # Paradigma generativo (familia determinista), OFF por defecto. NO se ata a
        # canonical_reuse_enabled: ese flag es para el match de memoria episódica,
        # y la familia determinista preemptaría el aprendizaje (no habría plan que
        # persistir). Requiere además naming canónico (call_args/param_names).
        if not settings.canonical_family_plan or not settings.canonical_reuse_enabled:
            return False
        if goal.sig != FAMILY_SIG:
            return False

        agent = self.agent
        profile = agent.profile
        recipes = [r.to_dict() for r in profile.recipes] if profile else []
        spawns = [s.to_dict() for s in profile.item_spawns] if profile else []
        family = build_has_item_family(
            recipes, spawns,
            coordination_enabled=settings.coordination_enabled,
            peer_request_timeout_s=settings.peer_request_timeout_s,
        )
        if not family:
            return False

        node.variants = [
            PlanVariant(guard=v.guard, steps=[], full_asl=v.asl, source=v.source)
            for v in family
        ]
        node.description = f"familia paramétrica {FAMILY_SIG}"
        node.status = NodeStatus.READY
        agent.phase = "executing"
        await agent.push_status()
        log.info(
            "[BDI:%s] Plan de FAMILIA determinista (0 LLM) para '%s' %s — %d variantes",
            agent.npc_id, goal.sig, goal.call_args, len(node.variants),
        )
        _trace(
            "plan_reuse_canonical",
            npc_id=agent.npc_id, goal=goal.sig, call_args=list(goal.call_args),
            variants=len(node.variants), source="CODE", llm_calls=0,
        )
        # plan_ready para paridad con el camino LLM (tooling/observabilidad).
        _trace(
            "plan_ready",
            npc_id=agent.npc_id, goal=goal.sig, elapsed_s=0.0,
            variants=len(node.variants),
            guard=node.variants[0].guard if node.variants else "true",
        )
        return True

    def _ensure_has_item_family_loaded(self) -> None:
        """Fase 12: carga la familia `achieve_has_item` (bajo clave fija, sin
        binding — es compartida, igual que en `_try_canonical_family_plan`)
        en `self._asp` si aún no lo está. Idempotente. Independiente de
        `canonical_reuse_enabled`/`canonical_family_plan`: la delegación
        necesita la familia como mecanismo de resolución de sub-goals sea
        cual sea el paradigma de memoria del agente."""
        from config import settings
        from llm.family_plan import FAMILY_SIG as _HAS_ITEM_SIG, build_has_item_family

        if self._is_plan_loaded(_HAS_ITEM_SIG):
            return
        agent = self.agent
        profile = agent.profile
        recipes = [r.to_dict() for r in profile.recipes] if profile else []
        spawns = [s.to_dict() for s in profile.item_spawns] if profile else []
        family = build_has_item_family(
            recipes, spawns,
            coordination_enabled=settings.coordination_enabled,
            peer_request_timeout_s=settings.peer_request_timeout_s,
        )
        if not family:
            return
        family_node = GoalNode(sig=_HAS_ITEM_SIG, status=NodeStatus.READY)
        family_node.variants = [
            PlanVariant(guard=v.guard, steps=[], full_asl=v.asl, source=v.source)
            for v in family
        ]
        self._load_plans_into_asp(family_node, _HAS_ITEM_SIG)
        log.info(
            "[BDI:%s] Familia achieve_has_item cargada bajo demanda (entrega a peer) — %d variantes",
            agent.npc_id, len(family_node.variants),
        )

    async def _try_peer_delivery_plan(self, goal: Any, node: GoalNode) -> bool:
        """Fase 12: goal `achieve_deliver_to_peer` — adoptado al ACEPTAR una
        petición de otro NPC (`PeerCoordBehaviour._handle_request` →
        `_adopt_goal_from_trigger`). Plan determinista (`llm.peer_delivery_plan`),
        sin LLM — análogo a `_try_canonical_family_plan` pero para este sig
        concreto, y sin depender de `canonical_reuse_enabled`/
        `canonical_family_plan` (la delegación no es un paradigma de reuso de
        memoria; es la ÚNICA forma en que este goal puede planificarse).
        """
        from config import settings
        from llm.peer_delivery_plan import build_deliver_to_peer_variants, SIG as _DELIVERY_SIG

        if not settings.coordination_enabled:
            return False
        if goal.sig != _DELIVERY_SIG:
            return False
        if getattr(settings, "coordination_planner", "family") == "llm":
            # Fase 17: la entrega la planifica el pipeline (step1b: conseguir el
            # item + peldaño de entrega de andamiaje), no el plan determinista.
            return False

        agent = self.agent
        # La variante "not has_item" de la entrega invoca !achieve_has_item
        # como sub-goal para reunir el ingrediente que falta. Esa familia
        # normalmente se carga en _try_canonical_family_plan, pero SOLO
        # cuando el agente tiene un goal top-level de esa familia — un NPC
        # que solo entrega bajo demanda de un peer (nunca tiene su propio
        # goal has_item) jamás pasaría por ahí. Cargarla aquí, bajo demanda,
        # cierra ese hueco (idempotente: no-op si ya está cargada).
        self._ensure_has_item_family_loaded()

        variants = build_deliver_to_peer_variants()
        node.variants = [
            PlanVariant(guard=v.guard, steps=[], full_asl=v.asl, source=v.source)
            for v in variants
        ]
        node.description = "entrega a peer (Fase 12)"
        node.status = NodeStatus.READY
        agent.phase = "executing"
        await agent.push_status()
        log.info(
            "[BDI:%s] Plan de ENTREGA a peer (0 LLM) para '%s' %s — %d variantes",
            agent.npc_id, goal.sig, goal.call_args, len(node.variants),
        )
        _trace(
            "plan_reuse_canonical",
            npc_id=agent.npc_id, goal=goal.sig, call_args=list(goal.call_args),
            variants=len(node.variants), source="CODE", llm_calls=0,
        )
        _trace(
            "plan_ready",
            npc_id=agent.npc_id, goal=goal.sig, elapsed_s=0.0,
            variants=len(node.variants),
            guard=node.variants[0].guard if node.variants else "true",
        )
        return True

    async def _request_plan(self, goal: Any) -> None:
        """Solicita al LLM un plan, construye PlanVariants y las carga en asp_agent."""
        from llm.pipeline.pipeline_runner import _steps_to_asl_body
        from llm.pipeline.step2_guards import build_guard_expression
        from config import settings

        agent = self.agent
        goal_key = self._goal_key(goal)
        # 0.C1 — Si ya existe el GoalNode (replan), reutilizarlo para conservar
        # failure_count/failure_history/success_count en vez de pisarlos con uno nuevo.
        # Indexado por sig+call_args (goal_key) para no mezclar bindings de familia.
        if agent.plan_graph.has_node(goal_key):
            node: GoalNode = agent.plan_graph.nodes[goal_key]["data"]
            node.status = NodeStatus.GENERATING
        else:
            node = GoalNode(sig=goal.sig, status=NodeStatus.GENERATING)
            agent.plan_graph.add_node(goal_key, data=node)
        # Fase 6.5 (reuso canónico): si el goal trae param_names (familia canónica),
        # propágalos al nodo para que las cabezas de las variantes sean paramétricas
        # con la aridad correcta (`+!achieve_has_item(Item, Qty)`), de modo que casen
        # con el goal despachado `achieve_has_item(bread, 1)` (call_args). Sin esto la
        # aridad no cuadra y ninguna variante aplica (no_applicable_variant_fuse).
        goal_params = getattr(goal, "param_names", []) or []
        if goal_params:
            node.param_names = list(goal_params)
        # Registrar el binding concreto del plan que se va a generar (identidad por
        # sig+call_args para no confundir bindings de la misma familia al reusar).
        goal_call_args = getattr(goal, "call_args", []) or []
        if goal_call_args:
            node.call_args = list(goal_call_args)

        # Fase 6.5 (T4): REUSO canónico — si el goal es de la familia has_item y el
        # mundo cubre su estructura (spawns/recetas), planifica con la familia
        # paramétrica DETERMINISTA (cero LLM) en vez de llamar a Ollama. Es la
        # contribución de reuso: una sola familia sirve a todos los bindings.
        if await self._try_canonical_family_plan(goal, node):
            return
        # Fase 12: goal `achieve_deliver_to_peer` (adoptado al aceptar una
        # petición de otro NPC) — plan determinista, sin LLM.
        if await self._try_peer_delivery_plan(goal, node):
            return

        log.info(f"[BDI:{agent.npc_id}] Solicitando plan para '{goal.sig}'")

        _trace("plan_requested", npc_id=agent.npc_id, goal=goal.sig)
        agent.phase = "planning"
        await agent.push_status()
        tl = TraceLogger.get()
        if tl is not None:
            tl.metrics.npc(agent.npc_id).goals_started += 1
        t0 = time.time()

        try:
            from protocol.messages import derive_entity_catalog

            _plan_payload: dict = {
                "task": "generate_plan",
                "_timeout": settings.plan_timeout_s,
                "npc_id": agent.npc_id,
                "goal_sig": goal.sig,
                "npc_statement": goal_statement(goal),
                "use_refinement": settings.use_refinement,
                # Fase 16: ablación de sub-planes (move_to_and_pickup/craft_item).
                "builtin_subplans": bool(getattr(settings, "builtin_subplans_enabled", True)),
                # Fase 17: coordinación planificada por el LLM (en vez de la familia
                # determinista) y NPCs conocidos para los peldaños de ayuda.
                "coordination": bool(
                    getattr(settings, "coordination_enabled", False)
                    and getattr(settings, "coordination_planner", "family") == "llm"
                ),
                "peers": [list(row) for row in agent.beliefs.snapshot().get("peer", [])],
                "success_condition": goal.success_condition,
                "profile": agent.profile.to_dict() if agent.profile else {},
                "beliefs": agent.beliefs.snapshot(),
                "known_goals": self._reusable_known_goals(),
                "entity_catalog": derive_entity_catalog(agent.profile) if agent.profile else {},
            }
            # Fase 16: contratos de capacidad aislados por sesión. Por defecto el
            # pipeline los persiste en src/plans/contracts/ (versionado) y un
            # sub-plan creado por el LLM en una sesión se cargaría en la siguiente
            # — contaminación entre runs de una batería. Con el flag se leen y
            # escriben en el directorio de la sesión (arrancan vacíos, igual que
            # el fichero versionado).
            if getattr(settings, "isolate_capability_contracts", False) and tl is not None:
                _plan_payload["capability_contracts_path"] = str(
                    tl.session_dir / "capability_contracts.json"
                )
            # T6: inyectar replan_hint cuando es un reintento
            if goal.replan_count > 0 and goal.replan_hint:
                _plan_payload["replan_hint"] = goal.replan_hint
                _plan_payload["replan_count"] = goal.replan_count
                log.info(
                    f"[BDI:{agent.npc_id}] Replan #{goal.replan_count} para '{goal.sig}' "
                    f"— hint: {goal.replan_hint[:80]}"
                )
            result = await agent.run_llm_task(_plan_payload)

            if not isinstance(result, dict):
                raise ValueError(f"PipelineResult inesperado: {type(result)}")

            result_sig = result.get("sig")
            if isinstance(result_sig, str) and result_sig and result_sig != goal.sig:
                raise ValueError(
                    f"PipelineResult sig mismatch: expected '{goal.sig}', got '{result_sig}'"
                )

            variants_payload = result.get("variants")
            variants: list[PlanVariant] = []
            if isinstance(variants_payload, list) and variants_payload:
                for item in variants_payload:
                    if not isinstance(item, dict):
                        continue
                    raw_steps = item.get("steps", [])
                    step_body = raw_steps
                    if raw_steps and not isinstance(raw_steps[0], str):
                        step_body = _steps_to_asl_body(raw_steps)
                    variants.append(
                        PlanVariant(
                            guard=item.get("guard", "true"),
                            steps=step_body,
                            full_asl=item.get("asl", ""),
                        )
                    )
            elif "steps" in result:
                guard_expr = build_guard_expression(
                    result.get("facts", []),
                    result.get("guards", []),
                )
                main_variant = PlanVariant(
                    guard=guard_expr,
                    steps=_steps_to_asl_body(result.get("steps", [])),
                    full_asl=result.get("main_asl", ""),
                )
                variants = [main_variant]

                for cont in result.get("contingency_plans", []):
                    cont_guard = cont.get("guard_expression", "true")
                    variants.append(PlanVariant(
                        guard=f"not ({cont_guard})",
                        steps=_steps_to_asl_body(cont.get("steps", [])),
                        full_asl=cont.get("asl", ""),
                    ))
            else:
                raise ValueError("PipelineResult sin variants ni steps")

            if not variants:
                raise ValueError("PipelineResult sin variantes ejecutables")

            node.variants = variants
            node.description = result.get("description", goal.sig)
            node.status = NodeStatus.READY
            agent.phase = "executing"
            await agent.push_status()
            elapsed = round(time.time() - t0, 3)
            log.info(
                f"[BDI:{agent.npc_id}] Plan listo para '{goal.sig}' — "
                f"{len(variants)} variante(s), guard principal: '{variants[0].guard}'"
            )
            _trace(
                "plan_ready",
                npc_id=agent.npc_id,
                goal=goal.sig,
                elapsed_s=elapsed,
                variants=len(variants),
                guard=variants[0].guard,
            )
            # Fase 4: contar plan generado por LLM (vs reusado de memoria).
            if tl is not None:
                tl.metrics.npc(agent.npc_id).plans_from_llm += 1

            # Fase 17c: los sub-goals que inventó el LLM traen su plan (Paso 6).
            loaded_subplans = self._load_llm_subplans(result, goal_key)

            dag_payload = (
                result.get("dag", {}) if isinstance(result.get("dag"), dict) else {}
            )
            dag_edges = (
                dag_payload.get("edges", []) if isinstance(dag_payload, dict) else []
            )
            for edge in dag_edges:
                if not isinstance(edge, list) or len(edge) != 2:
                    continue
                parent_sig, child_sig = edge
                if not isinstance(parent_sig, str) or not isinstance(child_sig, str):
                    continue
                if not agent.plan_graph.has_node(parent_sig):
                    agent.plan_graph.add_node(
                        parent_sig,
                        data=GoalNode(sig=parent_sig, status=NodeStatus.PENDING),
                    )
                if not agent.plan_graph.has_node(child_sig):
                    agent.plan_graph.add_node(
                        child_sig,
                        data=GoalNode(sig=child_sig, status=NodeStatus.PENDING),
                    )
                agent.plan_graph.add_edge(parent_sig, child_sig)

            for sub_item in result.get("subgoals_to_expand", []):
                if isinstance(sub_item, dict):
                    sub_sig = sub_item.get("sig", "")
                    sub_desc = sub_item.get("description", "")
                else:
                    sub_sig = sub_item
                    sub_desc = ""
                if not isinstance(sub_sig, str) or not sub_sig:
                    continue
                if sub_sig in loaded_subplans:
                    continue  # ya cargado: el plan padre lo llama inline
                if not agent.plan_graph.has_node(sub_sig):
                    sub_node = GoalNode(
                        sig=sub_sig, status=NodeStatus.PENDING, description=sub_desc
                    )
                    agent.plan_graph.add_node(sub_sig, data=sub_node)
                if not agent.plan_graph.has_edge(goal_key, sub_sig):
                    agent.plan_graph.add_edge(goal_key, sub_sig)
                if not any(g.sig == sub_sig for g in agent.goals):
                    from npc.agent import Goal

                    agent.goals.append(Goal(sig=sub_sig, parent=goal.sig))
                    log.debug(f"[BDI:{agent.npc_id}] Sub-goal encolado: '{sub_sig}'")

            self._persist_generated_plan(node)

            # Cargar variantes inmediatamente en asp_agent
            self._load_plans_into_asp(node)

        except Exception as exc:
            log.error(f"[BDI:{agent.npc_id}] Error generando plan para '{goal.sig}': {exc}")
            _trace("plan_failed", npc_id=agent.npc_id, goal=goal.sig, error=str(exc))
            # Fase 17r: con success_condition, un plan que no se pudo generar (p.ej.
            # un peldaño sin pasos válidos por JSON inválido del LLM) gasta un intento
            # del presupuesto de replan en vez de tumbar el goal. Piloto 17q (CO5/ATOM):
            # el goal murió en el replan 1/3 y la harina llegó 22 s después.
            if goal.success_condition:
                await self._trigger_replan_or_fail(
                    goal, node, reason="plan_failed", detail=str(exc)[:200],
                )
                return
            node.status = NodeStatus.FAILED
            tl2 = TraceLogger.get()
            if tl2 is not None:
                tl2.metrics.npc(agent.npc_id).goals_failed += 1

    def _load_llm_subplans(self, result: dict, parent_key: str) -> set[str]:
        """Fase 17c: carga en el motor ASL los sub-planes que el propio pipeline
        generó (Paso 6, `sub_results`) para los sub-goals que inventó el LLM.

        Antes se descartaban: el sub-goal se encolaba como goal de nivel superior
        y el plan padre, al llamarlo, fallaba con "no applicable plan" (tanda ATOM
        2026-09-15: 18 de 32 sesiones). Ahora el padre los encuentra cargados.
        Si ninguna variante del sub-plan aplica, su catch-all pide replanificar
        el goal padre. Recursivo. Devuelve los sigs cargados.
        """
        from llm.pipeline.pipeline_runner import _steps_to_asl_body

        loaded: set[str] = set()
        subs = result.get("sub_results") if isinstance(result, dict) else None
        if not isinstance(subs, dict):
            return loaded
        agent = self.agent
        for sub_sig, sub in subs.items():
            if not isinstance(sub_sig, str) or not sub_sig or not isinstance(sub, dict):
                continue
            variants: list[PlanVariant] = []
            for item in sub.get("variants") or []:
                if not isinstance(item, dict):
                    continue
                raw_steps = item.get("steps", [])
                step_body = raw_steps
                if raw_steps and not isinstance(raw_steps[0], str):
                    step_body = _steps_to_asl_body(raw_steps)
                variants.append(PlanVariant(
                    guard=item.get("guard", "true"),
                    steps=step_body,
                    full_asl=item.get("asl", ""),
                    source=item.get("source", "LLM"),
                ))
            if not variants:
                continue
            sub_node = GoalNode(sig=sub_sig, status=NodeStatus.READY)
            sub_node.variants = variants
            sub_node.description = sub.get("description", sub_sig)
            agent.plan_graph.add_node(sub_sig, data=sub_node)
            if not agent.plan_graph.has_edge(parent_key, sub_sig):
                agent.plan_graph.add_edge(parent_key, sub_sig)
            self._unload_plan(sub_sig)
            self._load_plans_into_asp(sub_node, sub_sig)
            loaded.add(sub_sig)
            _trace(
                "subplan_loaded", npc_id=agent.npc_id, parent=parent_key,
                subgoal=sub_sig, variants=len(variants), source="LLM",
            )
            loaded |= self._load_llm_subplans(sub, sub_sig)
        return loaded

    # --- goal lifecycle -----------------------------------------------------

    async def _on_variant_completed(self, goal: Any, node: GoalNode) -> None:
        """Una variante (rama) del plan terminó su cuerpo. Decide:
          - sin success_condition (legacy) → cierre como antes.
          - success_condition cumplida → cerrar el goal (éxito).
          - no cumplida → la escalera debe AVANZAR: no replanificamos; el siguiente
            tick re-inyecta el goal y dispara la siguiente variante aplicable. Se
            aplica una guarda anti-bucle por si la rama no progresa el estado.
        """
        if not goal.success_condition:
            await self._complete_goal(goal)
            return
        if self._eval_guard(goal.success_condition):
            await self._complete_goal(goal)
            return
        await self._note_ladder_progress(goal, node)

    async def _note_ladder_progress(self, goal: Any, node: GoalNode) -> None:
        """Guarda anti-bucle de la escalera. Si una rama casa y se ejecuta pero las
        beliefs NO cambian _MAX_LADDER_STUCK veces seguidas, fuerza un replan (en
        vez de re-inyectar la misma rama indefinidamente)."""
        sig = repr(sorted(self.agent.beliefs.snapshot().items()))
        if sig == getattr(goal, "_last_ladder_sig", None):
            goal._ladder_stuck = getattr(goal, "_ladder_stuck", 0) + 1
        else:
            goal._ladder_stuck = 0
            goal._last_ladder_sig = sig
        goal.reinject_count = getattr(goal, "reinject_count", 0) + 1
        _trace(
            "ladder_advance",
            npc_id=self.agent.npc_id, goal=goal.sig,
            reinject=goal.reinject_count, stuck=goal._ladder_stuck,
        )
        if goal._ladder_stuck >= _MAX_LADDER_STUCK:
            log.warning(
                "[BDI:%s] Escalera de '%s' sin progreso tras %d re-inyecciones → replan",
                self.agent.npc_id, goal.sig, goal._ladder_stuck,
            )
            goal._ladder_stuck = 0
            await self._trigger_replan_or_fail(goal, node, reason="ladder_stuck")

    async def _trigger_replan_or_fail(
        self, goal: Any, node: "GoalNode | None", *, reason: str, detail: str = ""
    ) -> None:
        """El plan cacheado no puede avanzar (ninguna variante aplica o la escalera
        no progresa). Replanifica con el LLM dentro del presupuesto (3); si se
        agota, marca el goal FAILED. Respeta el budget para no replanificar infinito."""
        agent = self.agent
        tl = TraceLogger.get()
        replan_budget = 3
        self._asp.intentions.clear()
        if goal.replan_count < replan_budget:
            goal.replan_count += 1
            goal.replan_hint = (
                f"The cached plan could not reach {goal.success_condition} "
                f"(reason: {reason}). Replan attempt {goal.replan_count}/{replan_budget}."
                + (f" Last failure: {detail}." if detail else "")
            )
            log.warning(
                "[BDI:%s] Goal '%s' — replan %d/%d (%s)",
                agent.npc_id, goal.sig, goal.replan_count, replan_budget, reason,
            )
            _trace(
                "goal_replan_required",
                npc_id=agent.npc_id, goal=goal.sig,
                success_condition=goal.success_condition,
                replan_count=goal.replan_count, replan_hint=goal.replan_hint,
                reason=reason,
            )
            if tl is not None:
                tl.metrics.npc(agent.npc_id).goals_replan_attempted += 1
            if node is not None:
                node.status = NodeStatus.NEEDS_REPLAN
            self._reset_attempt_counts(goal.sig)
            agent.intention = None
        else:
            log.error(
                "[BDI:%s] Goal '%s' FAILED tras %d replans — '%s' nunca cumplida (%s)",
                agent.npc_id, goal.sig, replan_budget, goal.success_condition, reason,
            )
            _trace(
                "goal_failed_after_replans",
                npc_id=agent.npc_id, goal=goal.sig,
                success_condition=goal.success_condition,
                replan_count=goal.replan_count, reason=reason,
            )
            if tl is not None:
                tl.metrics.npc(agent.npc_id).record_goal(GoalRecord(
                    sig=goal.sig, success_condition=goal.success_condition,
                    belief_met=False, replan_count=goal.replan_count,
                    final_status="failed",
                ))
            if node is not None:
                node.status = NodeStatus.FAILED
            self._reset_attempt_counts(goal.sig)
            self._notify_peer_on_failure(goal, reason=reason)
            agent.goals = [g for g in agent.goals if not self._same_goal(g, goal)]
            agent.intention = None
            self._record_plan_memory_failure(self._goal_key(goal))

    async def _complete_goal(self, goal: Any) -> None:
        """
        Cierra un goal con semántica belief-driven (T3 audit).

        Tres ramas:
          - goal_belief_met is True  → completado real; actualizar métricas.
          - goal_belief_met is False → replan si budget disponible; FAILED si agotado.
          - goal_belief_met is None  → sin success_condition; cierre legacy con warning.
        """
        agent = self.agent
        goal_key = self._goal_key(goal)

        # --- Evaluar success_condition ANTES de tocar contadores ---
        goal_belief_met: bool | None = None
        if goal.success_condition:
            try:
                goal_belief_met = self._eval_guard(goal.success_condition)
            except Exception:
                goal_belief_met = None

        intent_match: bool | None = None
        if goal.expected_condition:
            try:
                intent_match = self._eval_guard(goal.expected_condition)
            except Exception:
                intent_match = None

        tl = TraceLogger.get()

        # --- Rama: creencia cumplida ---
        if goal_belief_met is True:
            log.info(f"[BDI:{agent.npc_id}] Goal completado (belief met): '{goal.sig}'")
            _trace(
                "goal_completed",
                npc_id=agent.npc_id,
                goal=goal.sig,
                derived_condition=goal.success_condition,
                expected_condition=goal.expected_condition,
                goal_belief_met=True,
                intent_match=intent_match,
            )
            if tl is not None:
                npc_m = tl.metrics.npc(agent.npc_id)
                npc_m.record_goal(GoalRecord(
                    sig=goal.sig,
                    success_condition=goal.success_condition,
                    belief_met=True,
                    replan_count=goal.replan_count,
                    final_status="completed",
                ))
            agent.goals = [g for g in agent.goals if not self._same_goal(g, goal)]
            agent.intention = None
            agent.completed_goal_count += 1
            if agent.plan_graph.has_node(goal_key):
                node: GoalNode = agent.plan_graph.nodes[goal_key]["data"]
                node.success_count += 1
            self._reset_attempt_counts(goal.sig)
            self._record_plan_memory_success(self._goal_key(goal))
            self._unload_plan(goal_key)
            return

        # --- Rama: creencia NO cumplida ---
        if goal_belief_met is False:
            replan_budget = 3
            if goal.replan_count < replan_budget:
                goal.replan_count += 1
                goal.replan_hint = (
                    f"Previous plan completed but {goal.success_condition} was not achieved. "
                    f"Replan attempt {goal.replan_count}/{replan_budget}."
                )
                log.warning(
                    f"[BDI:{agent.npc_id}] Goal '{goal.sig}' — belief not met "
                    f"(replan {goal.replan_count}/{replan_budget})"
                )
                _trace(
                    "goal_replan_required",
                    npc_id=agent.npc_id,
                    goal=goal.sig,
                    success_condition=goal.success_condition,
                    replan_count=goal.replan_count,
                    replan_hint=goal.replan_hint,
                )
                if tl is not None:
                    tl.metrics.npc(agent.npc_id).goals_replan_attempted += 1
                # Marcar nodo para replanificar en el siguiente tick
                if agent.plan_graph.has_node(goal_key):
                    node: GoalNode = agent.plan_graph.nodes[goal_key]["data"]
                    node.status = NodeStatus.NEEDS_REPLAN
                # Reset de contadores/exhausted: el replan parte de cero (0.B2)
                self._reset_attempt_counts(goal.sig)
                # Reset intención para que el BDI vuelva a entrar en _tick_goal
                agent.intention = None
                self._asp.intentions.clear()
                return
            else:
                # Budget agotado → FAILED
                log.error(
                    f"[BDI:{agent.npc_id}] Goal '{goal.sig}' FAILED after {replan_budget} replans "
                    f"— success_condition '{goal.success_condition}' never met"
                )
                _trace(
                    "goal_failed_after_replans",
                    npc_id=agent.npc_id,
                    goal=goal.sig,
                    success_condition=goal.success_condition,
                    replan_count=goal.replan_count,
                )
                if tl is not None:
                    npc_m = tl.metrics.npc(agent.npc_id)
                    npc_m.record_goal(GoalRecord(
                        sig=goal.sig,
                        success_condition=goal.success_condition,
                        belief_met=False,
                        replan_count=goal.replan_count,
                        final_status="failed",
                    ))
                if agent.plan_graph.has_node(goal_key):
                    node: GoalNode = agent.plan_graph.nodes[goal_key]["data"]
                    node.status = NodeStatus.FAILED
                self._reset_attempt_counts(goal.sig)
                self._notify_peer_on_failure(goal, reason="belief_not_met")
                agent.goals = [g for g in agent.goals if not self._same_goal(g, goal)]
                agent.intention = None
                self._asp.intentions.clear()
                self._unload_plan(goal_key)
                return

        # --- Rama: sin success_condition (legacy) ---
        log.info(
            f"[BDI:{agent.npc_id}] Goal completado (unverified — sin success_condition): '{goal.sig}'"
        )
        log.warning(
            f"[BDI:{agent.npc_id}] Goal '{goal.sig}' cerrado sin verificar creencia objetivo. "
            "Añadir success_condition al NPCProfile para habilitar semantica belief-driven."
        )
        _trace(
            "goal_completed_unverified",
            npc_id=agent.npc_id,
            goal=goal.sig,
            expected_condition=goal.expected_condition,
            intent_match=intent_match,
            warning="no_success_condition",
        )
        if tl is not None:
            npc_m = tl.metrics.npc(agent.npc_id)
            npc_m.record_goal(GoalRecord(
                sig=goal.sig,
                success_condition=None,
                belief_met=None,
                replan_count=goal.replan_count,
                final_status="unverified",
            ))
        agent.goals = [g for g in agent.goals if not self._same_goal(g, goal)]
        agent.intention = None
        agent.completed_goal_count += 1
        if agent.plan_graph.has_node(goal_key):
            node: GoalNode = agent.plan_graph.nodes[goal_key]["data"]
            node.success_count += 1
        self._reset_attempt_counts(goal.sig)
        self._unload_plan(goal_key)
        # 0.C5 — cierre unverified: NO promover el plan (no hay verificación de
        # la creencia objetivo). Se registra aparte como uses_unverified.
        self._record_plan_memory_unverified(self._goal_key(goal))

    def _note_observational_failure(self, goal_sig: str | None, action_type: str) -> None:
        """0.B2 — Incrementa el contador de fallos de una acción observacional y,
        al alcanzar el attempt_budget del contrato, assertea exhausted(action, budget).

        Observacional = el contrato declara `may_observe` (ExploreArea, Search).
        """
        from protocol.action_semantics import CONTRACT_REGISTRY

        if not goal_sig or not action_type:
            return
        contract = CONTRACT_REGISTRY.get(action_type)
        if contract is None or not contract.may_observe:
            return

        key = (goal_sig, action_type)
        count = self._attempt_counts.get(key, 0) + 1
        self._attempt_counts[key] = count

        if count >= contract.attempt_budget:
            self.agent.beliefs.apply_exhausted(action_type, contract.attempt_budget)
            log.info(
                "[BDI:%s] %s agotado en '%s' tras %d intentos → exhausted(%s, %d)",
                self.agent.npc_id, action_type, goal_sig, count,
                action_type, contract.attempt_budget,
            )
            _trace(
                "action_exhausted",
                npc_id=self.agent.npc_id,
                goal=goal_sig,
                action=action_type,
                budget=contract.attempt_budget,
                attempts=count,
            )

    def _reset_attempt_counts(self, goal_sig: str) -> None:
        """Resetea los contadores de intentos del goal y limpia sus beliefs
        exhausted. Se llama al completar o replanificar el goal (0.B2)."""
        stale_actions = {
            action for (gsig, action) in self._attempt_counts if gsig == goal_sig
        }
        self._attempt_counts = {
            k: v for k, v in self._attempt_counts.items() if k[0] != goal_sig
        }
        for action in stale_actions:
            self.agent.beliefs.clear_exhausted(action)

    async def _handle_action_failure(self, goal: Any, result: dict) -> None:
        agent = self.agent
        goal_key = self._goal_key(goal)
        reason = result.get("errorCode") or result.get("errorMessage") or "unknown"
        log.warning(f"[BDI:{agent.npc_id}] Acción fallida en '{goal.sig}': {reason}")

        if agent.plan_graph.has_node(goal_key):
            node: GoalNode = agent.plan_graph.nodes[goal_key]["data"]
            node.failure_count += 1
            node.failure_history.append(reason)

            if node.failure_count >= 3 and goal.success_condition:
                # Fase 17e: con success_condition, el limite de fallos de accion pide
                # replan al LLM (mismo presupuesto que ladder_stuck) con el error de
                # Unity en el hint. Antes el goal pasaba a FAILED sin que el LLM viera
                # nunca el fallo ni las beliefs observadas (tanda 2 ATOM: A3 8/8).
                log.warning(
                    f"[BDI:{agent.npc_id}] Goal '{goal.sig}' alcanzó límite de fallos → replan"
                )
                node.failure_count = 0
                self._record_plan_memory_failure(goal_key)
                await self._trigger_replan_or_fail(
                    goal, node, reason=f"action_failed:{reason}",
                    detail=str(result.get("errorMessage") or reason),
                )
                return
            if node.failure_count >= 3:
                log.error(f"[BDI:{agent.npc_id}] Goal '{goal.sig}' alcanzó límite de fallos")
                node.status = NodeStatus.FAILED
                agent.intention = None
                self._asp.intentions.clear()
                self._unload_plan(goal_key)
                self._reset_attempt_counts(goal.sig)
                self._record_plan_memory_failure(self._goal_key(goal))

    async def _handle_no_applicable_variant(self, goal: Any) -> None:
        """Fuse de seguridad para evitar loops cuando ninguna variante aplica."""
        agent = self.agent
        hits = int(getattr(goal, "no_variant_hits", 0)) + 1
        goal.no_variant_hits = hits

        if hits == 1 or hits % 10 == 0:
            log.warning(
                f"[BDI:{agent.npc_id}] Sin variante aplicable para '{goal.sig}' "
                f"(hits={hits}/{_MAX_NO_VARIANT_HITS})"
            )

        if hits < _MAX_NO_VARIANT_HITS:
            await asyncio.sleep(0.2)
            return

        reason = f"no_applicable_variant_fuse({hits})"
        log.error(
            f"[BDI:{agent.npc_id}] Goal '{goal.sig}' abortado por fusible: "
            f"ninguna variante aplicable tras {hits} ciclos"
        )
        _trace("no_applicable_variant_fuse", npc_id=agent.npc_id, goal=goal.sig, hits=hits)

        goal_key = self._goal_key(goal)
        if agent.plan_graph.has_node(goal_key):
            node: GoalNode = agent.plan_graph.nodes[goal_key]["data"]
            node.failure_count += 1
            node.failure_history.append(reason)
            node.status = NodeStatus.FAILED
            self._reset_attempt_counts(goal.sig)
            self._record_plan_memory_failure(self._goal_key(goal))

        agent.intention = None
