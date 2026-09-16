from __future__ import annotations

import asyncio
import logging
import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.server import TCPServer
    from gateway.registry import NPCRegistry

log = logging.getLogger(__name__)


class GracefulShutdown:
    """
    Registra handlers de señales y coordina el cierre ordenado.
    Funciona tanto con Ctrl+C (SIGINT) como con exit() en consola interactiva.
    """

    def __init__(self, registry: "NPCRegistry", tcp_server: "TCPServer | None" = None):
        self.registry = registry
        self.tcp_server = tcp_server
        self._shutdown_event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def install(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop = loop
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._trigger)
            except (NotImplementedError, RuntimeError):
                # Windows no soporta add_signal_handler (lanza NotImplementedError).
                # 0.C3 — fallback: registrar el handler clásico con signal.signal,
                # que reentra al loop de forma thread-safe para disparar el cierre.
                try:
                    signal.signal(sig, self._signal_fallback)
                except (ValueError, OSError):
                    # ValueError: no estamos en el hilo principal;
                    # OSError: la señal no existe en esta plataforma.
                    pass

    def _signal_fallback(self, signum: int, frame: object) -> None:
        """Handler clásico (Windows): re-entra al event loop de forma segura."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._trigger)

    def _trigger(self) -> None:
        if self._shutdown_event.is_set():
            return
        log.info("[SHUTDOWN] Señal recibida — iniciando cierre ordenado")
        self._shutdown_event.set()

    def is_triggered(self) -> bool:
        return self._shutdown_event.is_set()

    async def wait_and_shutdown(self) -> None:
        await self._shutdown_event.wait()
        await self._run()

    async def _run(self) -> None:
        log.info("[SHUTDOWN] Notificando a Unity si está conectado...")
        # No se envía mensaje de control: el cliente Unity actual no reconoce
        # "PythonDisconnecting" y genera warning. Se notifica por cierre TCP.

        log.info("[SHUTDOWN] Deteniendo NPCAgents...")
        await self.registry.stop_all()

        log.info("[SHUTDOWN] Cerrando servidor TCP...")
        if self.tcp_server:
            await self.tcp_server.stop()

        log.info("[SHUTDOWN] Listo. Saliendo.")
        loop = asyncio.get_running_loop()
        loop.stop()


async def session_timeout_watchdog(
    registry: "NPCRegistry",
    shutdown: GracefulShutdown,
    max_s: float,
) -> None:
    """Fase 16: cierre ORDENADO al vencer la duración máxima de sesión.

    Antes, una sesión de batería que no terminaba la mataba el lanzador
    (Stop-Process -Force): sin metrics.json y con los goals abiertos fuera del
    denominador de éxito. Aquí, al vencer `max_s`, cada goal aún abierto se
    registra como `final_status="timeout"` (belief_met=False), se traza
    `shutdown_timeout` y se dispara el cierre normal (que escribe las métricas).
    """
    from utils.trace_logger import GoalRecord, TraceLogger, trace as _trace

    await asyncio.sleep(max_s)
    if shutdown.is_triggered():
        return
    open_goals = registry.open_goals()
    tl = TraceLogger.get()
    if tl is not None:
        for npc_id, goal in open_goals:
            tl.metrics.npc(npc_id).record_goal(GoalRecord(
                sig=goal.sig,
                success_condition=getattr(goal, "success_condition", None),
                belief_met=False,
                replan_count=int(getattr(goal, "replan_count", 0) or 0),
                final_status="timeout",
            ))
    log.warning(
        "[SHUTDOWN] Duración máxima de sesión (%.0f s) alcanzada — %d goal(s) abiertos",
        max_s, len(open_goals),
    )
    _trace(
        "shutdown_timeout",
        max_s=max_s,
        open_goals=[
            {"npc": npc_id, "goal": goal.sig,
             "success_condition": getattr(goal, "success_condition", None)}
            for npc_id, goal in open_goals
        ],
    )
    shutdown._trigger()
