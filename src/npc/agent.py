from __future__ import annotations

import asyncio
import collections
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine
import uuid

import spade
import networkx as nx
from spade.template import Template
from spade_bdi.bdi import BDIAgent

from npc.beliefs import BeliefStore
from npc.plan_graph import GoalNode, NodeStatus
from npc.trigger_registry import TriggerRegistry
from utils.plan_memory import PlanMemory, build_plan_memory
from protocol.messages import NPCProfilePayload

log = logging.getLogger(__name__)


class Goal:
    """Un goal activo del NPC."""

    def __init__(self, sig: str, priority: float | None = None, parent: str | None = None,
                 call_args: list | None = None):
        self.sig = sig
        self.priority = priority
        self.parent = parent                        # goal que generó este sub-goal
        self.call_args: list = call_args or []      # args del callsite, e.g. ["wheat", 1]
        # Fase 6.5: nombres de variable de la CABEZA paramétrica, paralelos a
        # call_args (e.g. ["Item", "Qty"]). Vacío salvo reuso canónico; fija la
        # aridad de la cabeza del plan para que case con el goal despachado.
        self.param_names: list[str] = []
        self.current_step: int = 0                  # paso actual dentro de la variante seleccionada
        self.current_variant_guard: str | None = None  # guard de la variante en curso
        # Condición de éxito derivada por el LLM desde el texto NL del goal
        self.success_condition: str | None = None
        # Condición de éxito esperada por el diseñador (de goal_conditions en NPCProfile)
        self.expected_condition: str | None = None
        # Replanificación: intentos realizados y pista para el siguiente intento
        self.replan_count: int = 0
        self.replan_hint: str | None = None
        # Escalera de variantes: nº de re-inyecciones para avanzar la escalera
        # (observabilidad) + guarda anti-bucle (estado interno en bdi).
        self.reinject_count: int = 0
        # --- Fase 12: coordinación NPC↔NPC -----------------------------------
        # Origen del goal: "designer" (NPCProfile) | "trigger" (adopt_goal
        # reactivo, Fase 9) | "peer" (request delegado de otro NPC, Fase 12).
        self.goal_source: str = "designer"
        # Solo con goal_source="peer": JID de quien pidió este goal (para
        # notificarle inform-done/failure al cerrar) y profundidad de la
        # cadena de delegación heredada (para que si ESTE goal a su vez
        # delega, envíe depth+1 — anti-bucle, ver settings.peer_max_depth).
        self.peer_requester_jid: str | None = None
        self.peer_request_depth: int = 0
        # sig que el REQUESTER original está esperando en su .await_peer (para
        # construir el inform-done/failure con el goal_sig que él reconoce).
        self.peer_origin_sig: str | None = None


class NPCAgent(BDIAgent):
    """
    Agente SPADE que representa un NPC.
    Tiene dos behaviours permanentes (UnityEventBehaviour + BDIBehaviour)
    y puede arrancar LLMBehaviour on-demand.

    Hereda de `spade_bdi.bdi.BDIAgent` (uso real de la librería, no
    decorativo: `BDIBehaviour` — ver `behaviours/bdi.py` — hereda a su vez de
    `BDIAgent.BDIBehaviour` y sus helpers heredados `get_belief`/`set_belief`/
    `get_beliefs`/`print_beliefs` operan sobre el MISMO `agentspeak.Agent`
    que usa nuestra lógica, no sobre un objeto de fachada aparte).

    `__init__` NO delega en `BDIAgent.__init__` porque este hace dos cosas
    que no encajan con este proyecto:
      1. Auto-añade su propio `BDIBehaviour` DURANTE `__init__` (antes de que
         `setup()` prepare `plan_graph`/`beliefs`) — el orden que ya
         garantizamos hoy (`setup()` primero, comportamientos después, ver
         `Agent._async_start`) es el que queremos conservar sin arriesgarlo.
      2. Carga un `.asl` FIJO de un fichero al arrancar. Este sistema no
         tiene un plan estático por NPC: los planes se sintetizan por LLM (o
         por la familia determinista) GOAL A GOAL en tiempo de ejecución
         (Fase 3/6.5/12/14) y se inyectan sobre el `agentspeak.Agent` ya
         vivo — por eso `BDIBehaviour.on_start()` sigue construyendo el
         agente de agentspeak de forma perezosa con una fuente vacía, como
         ya hacía, y expone ese MISMO objeto bajo los nombres que
         `BDIAgent` espera (`bdi_env`/`bdi_actions`/`bdi_agent`) para que el
         resto de la clase base (heredada, no reescrita) siga siendo
         funcional de verdad.
    El resto de `BDIAgent` (pause_bdi/resume_bdi/add_behaviour/set_asl/
    _load_asl/add_custom_actions) se hereda sin tocar.

    Estado compartido accesible desde todos los behaviours:
      inbox           — mensajes entrantes de Unity
      beliefs         — BeliefStore (predicados canónicos)
      goals           — lista de Goals activos
      intention       — Goal actualmente en ejecución
      plan_graph      — DiGraph con GoalNodes generados
      profile         — NPCProfilePayload recibido de Unity
      trigger_registry— TriggerRegistry con reglas reactivas
      paused          — True cuando Unity está desconectado
    """

    def __init__(
        self,
        jid: str,
        password: str,
        npc_id: str,
        send_to_unity: Callable[[dict], Coroutine],
        llm_planning_jid: str,
        plan_memory_root: "Path | None" = None,
    ):
        # spade.agent.Agent.__init__ directo (no BDIAgent.__init__) -- ver
        # docstring de la clase para el porqué. Replicamos a mano el estado
        # que BDIAgent.__init__ deja preparado, sin su auto-add de
        # comportamiento ni su carga eager de fichero.
        spade.agent.Agent.__init__(self, jid, password)
        self.asl_file = None
        self.bdi_enabled = False
        self.bdi_intention_buffer: collections.deque = collections.deque()
        self.bdi = None
        self.bdi_agent = None
        self.bdi_env = None
        self.bdi_actions = None

        self.npc_id = npc_id
        self.send_to_unity = send_to_unity
        self.llm_jid = llm_planning_jid

        # Estado compartido
        self.inbox: asyncio.Queue[dict] = asyncio.Queue()
        self.beliefs: BeliefStore = BeliefStore()
        self.goals: list[Goal] = []
        self.intention: Goal | None = None
        self.plan_graph: nx.DiGraph = nx.DiGraph()
        self.trigger_registry: TriggerRegistry = TriggerRegistry()
        self.profile: NPCProfilePayload | None = None
        self.paused: bool = False

        # Resultados de acciones Unity: cmd_id → ActionResult dict
        # Compartido con BDIBehaviour (ActionResultWaiter) y UnityEventBehaviour
        self.action_results: dict[str, dict] = {}

        # Comandos enviados a Unity: cmd_id → {"actionType", "args"}.
        # Lo escribe BDIBehaviour al enviar y lo lee UnityEventBehaviour para
        # correlacionar resultados con sus args originales (p.ej. la receta de un
        # Craft cuando hay que actualizar el inventario provisionalmente — C2).
        self.sent_commands: dict[str, dict] = {}

        # Fase 17t: encargos de peers que acabaron en fallo, por clave de binding
        # (`achieve_deliver_to_peer__npc_baker_flour_1` → nº de fallos).
        self.failed_peer_deliveries: dict[str, int] = {}

        # Fase 12: respuestas de coordinación (inform/agree/refuse) a algo que
        # ESTE NPC preguntó/pidió — conversation_id → PeerMessage. Lo escribe
        # PeerCoordBehaviour al recibir la respuesta; lo consume el Waiter de
        # la acción ASL correspondiente (.ask_peer/.request_peer, en bdi.py).
        # No confundir con peer_done/peer_failed (esos van a BeliefStore, no
        # aquí — son notificaciones asíncronas sin Waiter esperando).
        self.peer_results: dict[str, Any] = {}

        # Plan memory (Fase 4): el run root (compartido entre NPCs) lo resuelve
        # main.py según settings (on/off, nuevo/reuse/dir) y lo pasa el registry.
        # Si plan_memory_root es None → memoria desactivada (no-op).
        from config import settings as _settings
        self.plan_memory = build_plan_memory(npc_id, _settings, plan_memory_root)

        # Contador de goals completados (para reflexión futura — Fase 4)
        self.completed_goal_count: int = 0

        # Fase de razonamiento actual (para el cuadro de información de Unity):
        # idle | parsing_goals | planning | executing | ready.
        self.phase: str = "idle"

        # Referencia a la task de bootstrap de goals (para cancelarla al parar)
        self._bootstrap_task: asyncio.Task | None = None
        self._profile_fingerprint: str | None = None
        self._bootstrapped_profile_fingerprint: str | None = None
        # True una vez que el NPC ha tenido goals activos (tras bootstrap). Sirve
        # para detectar "trabajo terminado" (is_work_done) sin confundir el estado
        # inicial sin goals con el final de haberlos resuelto todos.
        self._had_goals: bool = False

    def is_work_done(self) -> bool:
        """True si el NPC ya tuvo goals y los ha resuelto todos (completados o
        fallados sin repair) y no tiene intención en curso. Lo usa el watchdog de
        apagado por inactividad (settings.shutdown_when_idle)."""
        return self._had_goals and not self.goals and self.intention is None

    def is_idle_for_shutdown(self) -> bool:
        """Fase 17: variante multi-NPC de is_work_done. Un NPC cuyo perfil no trae
        goals (solo ayuda a otros) está resuelto en cuanto está ocioso; uno que sí
        los trae debe haberlos tenido y resuelto (no confundir 'aún parseando' con
        'terminado')."""
        if self.goals or self.intention is not None:
            return False
        if self.profile is None:
            return False
        if getattr(self.profile, "goals_nl", None):
            return self._had_goals
        return True

    async def setup(self) -> None:
        from npc.behaviours.unity_events import UnityEventBehaviour
        from npc.behaviours.bdi import BDIBehaviour
        from npc.builtin_loader import load_builtin_plans

        log.info(f"[NPC:{self.npc_id}] Arrancando behaviours")

        # Cargar triggers built-in (ruta absoluta relativa a este fichero).
        # Fase 13: builtin_triggers_enabled (default True, sin cambio de
        # comportamiento) permite desactivarlos para la batería experimental
        # -- el trigger demostrativo de inventory.asl (wheat>=2 -> adopta
        # achieve_bake_bread) contamina experimentos de un solo goal aislado
        # (E2/E3/E4: cualquiera que acumule 2+ wheat dispara un SEGUNDO goal
        # no pedido, inflando n_act/goals_started más allá de lo que el
        # experimento mide). Confirmado en el piloto de E2, 2026-08-11.
        from config import settings as _settings
        _plans_dir = Path(__file__).resolve().parent.parent / "plans"
        _triggers_dir = _plans_dir / "triggers"
        if getattr(_settings, "builtin_triggers_enabled", True):
            self.trigger_registry.load_from_path(_triggers_dir)
        self.beliefs.npc_id = self.npc_id
        self.beliefs.set_trigger_registry(self.trigger_registry)
        self.beliefs.set_goal_adopter(self._adopt_goal_from_trigger)

        # Cargar planes built-in (goals reutilizables por el pipeline LLM).
        # Fase 16: con builtin_subplans_enabled=False se omiten los sub-planes
        # macro move_to_and_pickup/craft_item (ablación) y se traza qué se cargó.
        from llm.pipeline.builtins import builtin_file_exclusions
        from utils.trace_logger import trace as _trace_builtins
        _builtin_dir = _plans_dir / "builtin"
        _subplans_on = bool(getattr(_settings, "builtin_subplans_enabled", True))
        _coordination_on = bool(getattr(_settings, "coordination_enabled", False))
        _sigs = load_builtin_plans(
            self.plan_graph, _builtin_dir,
            exclude_files=builtin_file_exclusions(_subplans_on, _coordination_on),
        )
        if _sigs:
            log.info(f"[NPC:{self.npc_id}] Built-in plans: {_sigs}")
        _trace_builtins(
            "builtin_plans_loaded", npc_id=self.npc_id,
            sigs=list(_sigs), subplans_enabled=_subplans_on,
        )
        if not _subplans_on and (
            getattr(_settings, "canonical_family_plan", False)
            or (_coordination_on and getattr(_settings, "coordination_planner", "family") != "llm")
        ):
            log.warning(
                f"[NPC:{self.npc_id}] builtin_subplans_enabled=False con "
                "canonical_family_plan/coordination_enabled: sus planes deterministas "
                "invocan move_to_and_pickup/craft_item y fallarán"
            )

        # Fase 4: cargar planes APROBADOS de plan memory como GoalNodes READY
        # (from_memory=True), sin llamar al LLM. Visible y trazado.
        self._load_memory_plans()

        self.plan_memory.rebuild_bundle_from_graph(self.plan_graph)

        self.add_behaviour(UnityEventBehaviour())
        self.add_behaviour(BDIBehaviour())

        # Fase 12: canal de coordinación NPC↔NPC, solo si está activado.
        # Con el flag off, cero behaviours nuevos — comportamiento idéntico
        # a la Fase 11.
        from config import settings as _settings
        if getattr(_settings, "coordination_enabled", False):
            from npc.behaviours.peer_coord import PeerCoordBehaviour
            self.add_behaviour(PeerCoordBehaviour())
            log.info(f"[NPC:{self.npc_id}] PeerCoordBehaviour activo (coordination_enabled)")

    def _load_memory_plans(self) -> None:
        """Carga los planes approved de plan memory al plan_graph (Fase 4).

        Cada plan aprobado se convierte en un GoalNode READY con from_memory=True.
        Emite traza plan_memory_load y log INFO (regla de no-silencio).
        """
        from utils.trace_logger import trace as _trace
        from config import settings as _settings

        if not getattr(self.plan_memory, "enabled", False):
            return
        records = self.plan_memory.load_all_approved()
        # Modo REUSE explícito: además de los approved, reusar los pending con
        # evidencia de éxito (1 sesión = 1 uso, nunca llega al umbral de promoción).
        # Es lo que materializa "aprender en una sesión y reusar en la siguiente".
        reuse_mode = bool(getattr(_settings, "plan_memory_reuse", False))
        if reuse_mode:
            approved_sigs = {r.get("goal_sig") for r in records}
            for rec in self.plan_memory.load_all_reusable_pending():
                if rec.get("goal_sig") not in approved_sigs:
                    records.append(rec)
        from llm.canonical import goal_identity_key
        loaded: list[dict] = []
        for record in records:
            # `functor` = sig real de familia (achieve_has_item) para el node.sig/ASL.
            # Ficheros viejos no lo traen → cae a goal_sig (que entonces ERA el sig).
            functor = record.get("functor") or record.get("goal_sig", "")
            if not functor:
                continue
            # El nodo se indexa por sig+call_args (clave de identidad) igual que en
            # el ciclo BDI, para que el goal canónico (achieve_has_item con su
            # binding) encuentre su plan reusado y no colisione con otro binding.
            node_key = goal_identity_key(functor, record.get("call_args", []) or [])
            if self.plan_graph.has_node(node_key):
                continue  # builtins/otros tienen prioridad; no pisar
            node = self.plan_memory.build_goalnode(functor, record)
            node.from_memory = True
            self.plan_graph.add_node(node_key, data=node)
            loaded.append({
                "sig": functor,
                "success_rate": record.get("success_rate"),
                "tier": record.get("status", "pending"),
            })

        if loaded:
            log.info(
                "[MEMORY:%s] %d planes aprobados cargados: %s",
                self.npc_id, len(loaded), [p["sig"] for p in loaded],
            )
        _trace(
            "plan_memory_load",
            npc_id=self.npc_id,
            run_dir=str(self.plan_memory.root_dir),
            count=len(loaded),
            plans=loaded,
        )

    def _adopt_goal_from_trigger(
        self,
        sig: str,
        condition: str | None = None,
        *,
        origin: str = "trigger",
        call_args: list | None = None,
        peer_requester_jid: str | None = None,
        peer_request_depth: int = 0,
        peer_origin_sig: str | None = None,
    ) -> None:
        """Adopta un Goal nuevo desde un trigger reactivo (adopt_goal en body)
        o desde una delegación de otro NPC (Fase 12, `origin="peer"`).

        `condition` (Fase 9): success_condition del goal REACTIVO → se cierra por
        creencia verificada, no "unverified". Si el reuso canónico está activo,
        hay condición Y NO se pasó `call_args` explícito, el sig/call_args/
        param_names se derivan de la condición (igual que los goals de arranque)
        para que el goal reactivo entre en la misma familia/identidad y pueda
        reusar memoria.

        `call_args` explícito (Fase 12 — `PeerCoordBehaviour` al aceptar un
        `request`): el llamante YA conoce los argumentos exactos (p.ej.
        `achieve_deliver_to_peer` con `[requester_id, item, qty]`) — se usan
        tal cual, SIN pasar por la canonicalización desde condición (esa lógica
        es específica de la familia `has_item` derivada de NL, no aplica aquí).

        Idempotente por identidad (sig + call_args): no re-encola un goal ya
        activo. Ejecutado de forma síncrona durante una actualización de
        beliefs (trigger) o desde `PeerCoordBehaviour` (peer request).
        """
        new_goal = Goal(sig=sig)
        new_goal.goal_source = origin
        new_goal.peer_requester_jid = peer_requester_jid
        new_goal.peer_request_depth = peer_request_depth
        new_goal.peer_origin_sig = peer_origin_sig

        if call_args is not None:
            new_goal.call_args = list(call_args)
            if condition:
                new_goal.success_condition = condition
                new_goal.expected_condition = condition
        elif condition:
            new_goal.success_condition = condition
            new_goal.expected_condition = condition
            from config import settings as _settings
            if getattr(_settings, "canonical_reuse_enabled", False):
                from llm.canonical import (
                    canonical_key, derive_family_sig,
                    call_args_from_condition, param_names_from_condition,
                )
                from protocol.messages import derive_entity_catalog
                cat = derive_entity_catalog(self.profile) if self.profile else {}
                schema, _ = canonical_key(condition, cat)
                new_goal.sig = derive_family_sig(schema)
                new_goal.call_args = call_args_from_condition(condition)
                new_goal.param_names = param_names_from_condition(condition, cat)

        def _same(g: Goal) -> bool:
            return g.sig == new_goal.sig and (
                list(getattr(g, "call_args", []) or []) == list(new_goal.call_args or [])
            )

        if any(_same(g) for g in self.goals) or (
            self.intention is not None and _same(self.intention)
        ):
            log.debug(f"[NPC:{self.npc_id}] adopt_goal({new_goal.sig}) ignorado — ya activo")
            return
        # Fase 17s: una petición NUEVA de un peer no hereda el nodo FAILED de un
        # encargo anterior con el mismo binding. Antes el goal se cerraba en 0,1 s
        # sin planificar y el peticionario repetía hasta too_many_requests (piloto
        # 17r, CO5 y CO6). El goal nuevo arranca con su propio presupuesto de replan.
        if origin == "peer":
            self._reset_failed_plan_node(new_goal)
        self.goals.append(new_goal)
        log.info(
            f"[NPC:{self.npc_id}] Goal adoptado por trigger: {new_goal.sig} "
            f"{new_goal.call_args or ''} (cond={condition})"
        )

    def _reset_failed_plan_node(self, goal: Goal) -> None:
        from llm.canonical import goal_identity_key
        from npc.plan_graph import NodeStatus
        key = goal_identity_key(goal.sig, list(goal.call_args or []))
        if not self.plan_graph.has_node(key):
            return
        node = self.plan_graph.nodes[key].get("data")
        if node is None or node.status != NodeStatus.FAILED:
            return
        node.status = NodeStatus.NEEDS_REPLAN
        node.failure_count = 0
        log.info(f"[NPC:{self.npc_id}] {key}: petición nueva tras un encargo fallido — plan nuevo")

    async def teardown(self) -> None:
        if self._bootstrap_task is not None and not self._bootstrap_task.done():
            self._bootstrap_task.cancel()
            log.debug(f"[NPC:{self.npc_id}] Bootstrap task cancelada en teardown")

    async def push_status(self) -> None:
        """Envía a Unity la lista de goals + cuál está activo + la fase de
        razonamiento (cuadro de información, PlanDebugUI). Best-effort: nunca
        rompe el ciclo BDI si el envío falla."""
        intention_sig = self.intention.sig if self.intention is not None else None
        try:
            await self.send_to_unity({
                "type": "GoalsUpdate",
                "npcId": self.npc_id,
                "phase": self.phase,
                "goals": [
                    {
                        "sig": g.sig,
                        "condition": g.success_condition or "",
                        "active": g.sig == intention_sig,
                    }
                    for g in self.goals
                ],
            })
        except Exception as exc:
            log.debug(f"[NPC:{self.npc_id}] push_status falló: {exc}")

    async def run_llm_task(self, task: dict) -> Any:
        """
        Lanza un LLMBehaviour on-demand y espera su resultado.
        Bloqueante para el NPC, pero puntual.
        """
        from config import settings
        from npc.llm_behaviour import LLMBehaviour

        # Pipeline timeout: el pipeline puede hacer ~10 llamadas LLM; dar margen suficiente.
        # Respetar el _timeout que ya venga del llamante (p.ej. bdi._request_plan);
        # solo fijar el default si no se especificó.
        pipeline_timeout = float(settings.llm_timeout) * 10
        task = dict(task)
        task.setdefault("_timeout", pipeline_timeout)

        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        thread_id = str(uuid.uuid4())
        b = LLMBehaviour(
            task=task,
            result_future=future,
            llm_jid=self.llm_jid,
            thread_id=thread_id,
        )
        self.add_behaviour(b, Template(thread=thread_id))
        return await future
