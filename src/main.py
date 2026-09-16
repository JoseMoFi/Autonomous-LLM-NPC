from __future__ import annotations

"""Punto de entrada V2 del sistema NPC BDI + LLM.

Arranca:
  1. Servidor XMPP embebido (pyjabber vía spade.run)
  2. LLMPlanningAgent (XMPP JID: llm_planning@localhost)
  3. Servidor TCP en :7777 (acepta conexiones de Unity)
  4. GracefulShutdown (Ctrl+C → cierre ordenado)
"""

import asyncio
import contextlib
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Awaitable, Callable

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from gateway.server import TCPServer
from gateway.router import MessageRouter
from gateway.registry import NPCRegistry
from gateway.shutdown import GracefulShutdown

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
# El proveedor LLM vive ahora en `llm.providers` (adaptador de spade-llm).
# Se construye con `build_provider(settings)` en _run_with_embedded_spade.


async def _idle_watchdog(
    registry: "NPCRegistry",
    shutdown: "GracefulShutdown",
    *,
    interval: float = 1.5,
    confirmations: int = 3,
) -> None:
    """Dispara el cierre ordenado cuando TODOS los NPCs han resuelto su trabajo
    (goals completados o fallados sin repair). Activo solo si
    settings.shutdown_when_idle. Requiere `confirmations` lecturas consecutivas
    para no cortar en transiciones puntuales entre goals.
    """
    from utils.trace_logger import trace as _trace

    idle_count = 0
    while not shutdown.is_triggered():
        await asyncio.sleep(interval)
        if registry.work_done():
            idle_count += 1
            if idle_count >= confirmations:
                log.info("[MAIN] Todos los goals resueltos — apagado por inactividad")
                _trace("shutdown_idle", reason="all_goals_resolved")
                shutdown._trigger()
                return
        else:
            idle_count = 0


# ---------------------------------------------------------------------------
# Parches de compatibilidad pyjabber (heredados de V1 — probados en producción)
# ---------------------------------------------------------------------------

def _patch_pyjabber_xml_parser_resilience() -> None:
    """
    pyjabber en Windows lanza dos tipos de errores XML que no son fatales:
      1. Exception() vacía desde XMLParser.startElementNS
      2. SAXParseException desde XMLProtocol.data_received al recibir datos
         no-XML durante el handshake interno SPADE↔pyjabber.
    Ambos se suprimen (log a DEBUG) para no contaminar la consola.
    """
    enabled = os.getenv("PYJABBER_SOFT_XML_ERRORS", "1").strip().lower() not in {
        "0", "false", "no"
    }
    if not enabled:
        return

    # ── Patch 1: XMLParser.startElementNS ───────────────────────────────
    try:
        pyjabber_xml = __import__("pyjabber.network.XMLParser", fromlist=["XMLParser"])
        xml_cls = getattr(pyjabber_xml, "XMLParser", None)
        if xml_cls is not None:
            original = getattr(xml_cls, "startElementNS", None)
            if callable(original) and not getattr(original, "_v2_patched", False):
                def _safe_start(self, pair, qname, attrs):
                    try:
                        return original(self, pair, qname, attrs)
                    except BaseException as exc:
                        if type(exc) is Exception and not getattr(exc, "args", ()):
                            log.debug("[pyjabber] XMLParser Exception() suprimida")
                            return None
                        raise
                setattr(_safe_start, "_v2_patched", True)
                setattr(xml_cls, "startElementNS", _safe_start)
    except Exception:
        pass

    # ── Patch 2: XMLProtocol.data_received ──────────────────────────────
    # Suprime SAXParseException / ExpatError que llegan por datos no-XML
    # durante el handshake interno SPADE ↔ pyjabber.
    try:
        import xml.sax._exceptions as _sax_exc
        pyjabber_proto = __import__(
            "pyjabber.network.XMLProtocol", fromlist=["XMLProtocol"]
        )
        proto_cls = getattr(pyjabber_proto, "XMLProtocol", None)
        if proto_cls is not None:
            orig_data = getattr(proto_cls, "data_received", None)
            if callable(orig_data) and not getattr(orig_data, "_v2_patched", False):
                def _safe_data_received(self, data: bytes) -> None:
                    try:
                        return orig_data(self, data)
                    except (_sax_exc.SAXParseException, Exception) as exc:
                        log.debug("[pyjabber] XMLProtocol.data_received suprimido: %s", exc)
                setattr(_safe_data_received, "_v2_patched", True)
                setattr(proto_cls, "data_received", _safe_data_received)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Coroutine principal
# ---------------------------------------------------------------------------

async def _run() -> None:
    from llm.planning_agent import LLMPlanningAgent
    from llm.providers import build_provider

    xmpp_host = settings.xmpp_host
    llm_jid = f"llm_planning@{xmpp_host}"

    # Fase 4: resolver el run root de plan memory una vez (compartido por todos
    # los NPCs). Según settings: desactivada / nuevo run / reuse (dir o último).
    from utils.plan_memory import resolve_run_root
    from utils.trace_logger import trace as _trace_main
    # _PROJECT_ROOT es src/ (main.py vive en src/); la memoria va en la raíz del
    # repo, igual que la legacy plans/memory/ y el .gitignore.
    _pm_base = _PROJECT_ROOT.parent / "plans" / "memory"
    plan_memory_root, plan_memory_mode = resolve_run_root(settings, _pm_base)
    log.info(
        "[MEMORY] plan memory: mode=%s root=%s (enabled=%s reuse=%s)",
        plan_memory_mode,
        plan_memory_root if plan_memory_root is not None else "-",
        settings.plan_memory_enabled, settings.plan_memory_reuse,
    )
    _trace_main(
        "plan_memory_run",
        mode=plan_memory_mode,
        run_dir=str(plan_memory_root) if plan_memory_root is not None else None,
    )

    registry = NPCRegistry(llm_planning_jid=llm_jid, plan_memory_root=plan_memory_root)
    router = MessageRouter(registry)
    tcp_server = TCPServer(
        host=settings.unity_host,
        port=settings.unity_port,
        router=router,
        registry=registry,
    )

    shutdown = GracefulShutdown(registry, tcp_server)
    shutdown.install()

    # Watchdog opcional: apaga el sistema cuando todos los NPCs resuelven sus
    # goals (settings.shutdown_when_idle). Pensado para pruebas/eval.
    idle_task: asyncio.Task | None = None
    if settings.shutdown_when_idle:
        idle_task = asyncio.create_task(_idle_watchdog(registry, shutdown))
        log.info("[MAIN] shutdown_when_idle activo — el sistema se cerrará al resolver todos los goals")

    # Fase 16: duración máxima de sesión con cierre ORDENADO (métricas escritas,
    # goals abiertos registrados como timeout). 0 = desactivado (uso normal).
    timeout_task: asyncio.Task | None = None
    if settings.session_max_s and settings.session_max_s > 0:
        from gateway.shutdown import session_timeout_watchdog
        timeout_task = asyncio.create_task(
            session_timeout_watchdog(registry, shutdown, settings.session_max_s)
        )
        log.info("[MAIN] session_max_s=%.0f — cierre ordenado al vencer", settings.session_max_s)

    # Levanta el TCP primero para que Unity pueda conectar de inmediato,
    # aunque el arranque de XMPP/LLM tarde unos segundos.
    tcp_task = asyncio.create_task(tcp_server.start())
    shutdown_task = asyncio.create_task(shutdown.wait_and_shutdown())

    llm_agent = LLMPlanningAgent(
        jid=llm_jid,
        password="llm_planning",
        llm_provider=build_provider(settings),
    )
    llm_started = False
    llm_start_task = asyncio.create_task(llm_agent.start(auto_register=True))

    done, _pending = await asyncio.wait(
        {tcp_task, llm_start_task, shutdown_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    if tcp_task in done:
        await tcp_task
        raise RuntimeError("[MAIN] El servidor TCP terminó inesperadamente durante el arranque")

    if llm_start_task in done:
        await llm_start_task
        llm_started = True
        log.info("[MAIN] LLMPlanningAgent arrancado — JID: %s", llm_jid)
        log.info(
            "[MAIN] LLM: %s @ %s (model=%s)",
            settings.llm_provider,
            settings.llm_base_url,
            settings.llm_model,
        )

        done, _pending = await asyncio.wait(
            {tcp_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if tcp_task in done:
            await tcp_task
            raise RuntimeError("[MAIN] El servidor TCP terminó inesperadamente en ejecución")

        await shutdown_task
    else:
        # Ctrl+C durante el arranque: evita quedar bloqueados en start().
        llm_start_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await llm_start_task

    if idle_task is not None and not idle_task.done():
        idle_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await idle_task
    if timeout_task is not None and not timeout_task.done():
        timeout_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await timeout_task

    results = await asyncio.gather(tcp_task, return_exceptions=True)
    for _r in results:
        if isinstance(_r, BaseException):
            log.error("[MAIN] Tarea finalizada con excepción: %r", _r)

    if llm_started:
        with contextlib.suppress(Exception):
            await llm_agent.stop()
    log.info("[MAIN] Sistema detenido")

    from utils.trace_logger import TraceLogger
    tl = TraceLogger.get()
    if tl is not None:
        tl.finalize()
        log.info("[MAIN] Sesión de log finalizada → %s", tl.session_dir)


async def _wait_embedded_xmpp_ready(server_task: asyncio.Task, ready_event: asyncio.Event) -> None:
    ready_task = asyncio.create_task(ready_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {server_task, ready_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if server_task in done:
            await server_task
            raise RuntimeError("[MAIN] El servidor XMPP embebido terminó antes de quedar listo")
        await ready_task
    finally:
        if not ready_task.done():
            ready_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ready_task


def _current_git_sha() -> str | None:
    """SHA corto del commit actual (Fase 13 — reproducibilidad de la batería
    experimental: cada sesión queda ligada al código exacto que la generó).
    None si no es un repo git o `git` no está disponible — nunca lanza."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        sha = result.stdout.strip()
        return sha or None
    except Exception:
        return None


def _is_address_in_use_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, OSError):
            if getattr(current, "errno", None) in {48, 98}:
                return True
            if getattr(current, "winerror", None) == 10048:
                return True
        current = current.__cause__ or current.__context__
    return False


def _assert_port_available(host: str, port: int, label: str) -> None:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    bind_target = (host, port, 0, 0) if family == socket.AF_INET6 else (host, port)

    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind(bind_target)
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"{label} no disponible en {host}:{port}. Ya hay otra instancia o el puerto está ocupado.",
        ) from exc


def _assert_startup_ports_available() -> None:
    _assert_port_available("127.0.0.1", 5222, "XMPP embebido")
    _assert_port_available(settings.unity_host, int(settings.unity_port), "Servidor TCP Unity")


def _run_with_embedded_spade(
    main_factory: Callable[[], Awaitable[None]],
    exception_handler,
) -> None:
    import loguru
    from pyjabber.server import Server
    from pyjabber.server_parameters import Parameters
    from spade.container import Container

    container = Container()
    loop = container.loop
    loop.set_exception_handler(exception_handler)

    server_task: asyncio.Task | None = None

    try:
        _assert_startup_ports_available()
        loguru.logger.remove()
        server_instance = Server(
            Parameters(host="localhost", database_in_memory=True)
        )
        server_task = loop.create_task(server_instance.start())
        loop.run_until_complete(_wait_embedded_xmpp_ready(server_task, server_instance.ready))
        log.info("[MAIN] XMPP embebido listo en localhost:5222")
        loop.run_until_complete(main_factory())
    finally:
        with contextlib.suppress(Exception):
            container.stop_agents()

        if server_task is not None:
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(server_task)

        pending_tasks = [
            task for task in asyncio.all_tasks(loop=loop)
            if not task.done()
        ]
        for task in pending_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(task)

        with contextlib.suppress(Exception):
            loop.run_until_complete(loop.shutdown_asyncgens())
        with contextlib.suppress(Exception):
            loop.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.getLevelName(settings.log_level),
        format="%(asctime)s [%(levelname)8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Fijar nivel INFO en el StreamHandler de consola explícitamente, de modo que
    # cuando TraceLogger baje el root a DEBUG (para session.log), la consola no se
    # inunde con mensajes de debug.
    for _h in logging.getLogger().handlers:
        if isinstance(_h, logging.StreamHandler) and not isinstance(_h, logging.FileHandler):
            _h.setLevel(logging.getLevelName(settings.log_level))

    noisy_logger_levels = {
        "spade.behaviour": logging.INFO,
        "spade.Agent": logging.INFO,
        "slixmpp": logging.INFO,
        "slixmpp.xmlstream.xmlstream": logging.INFO,
        "slixmpp.plugins.xep_0199.ping": logging.INFO,
    }
    for _name, _level in noisy_logger_levels.items():
        _logger = logging.getLogger(_name)
        if _logger.level == logging.NOTSET or _logger.level < _level:
            _logger.setLevel(_level)

    # Con spade.run() no siempre podemos inyectar nuestro exception handler
    # en el loop interno. Filtramos el log de asyncio conocido/no fatal de
    # pyjabber para evitar tracebacks ruidosos en consola.
    class _AsyncioPyjabberFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.name != "asyncio":
                return True
            msg = record.getMessage()
            exc_text = ""
            if record.exc_info and len(record.exc_info) >= 2 and record.exc_info[1] is not None:
                exc_text = str(record.exc_info[1])
            text = f"{msg} {exc_text}"
            if "Fatal error on transport" in text and "XMLProtocol" in text:
                return False
            if "SAXParseException" in text or "ExpatError" in text:
                return False
            return True

    logging.getLogger("asyncio").addFilter(_AsyncioPyjabberFilter())

    from utils.trace_logger import TraceLogger
    _logs_root = Path(__file__).resolve().parent.parent / "logs"
    tl = TraceLogger.init_session(logs_root=_logs_root)
    tl.trace(
        "session_start",
        model=settings.llm_model,
        provider=settings.llm_provider,
        host=settings.unity_host,
        port=settings.unity_port,
        log_level=settings.log_level,
        # Fase 13 — etiquetado de sesión (todo opcional; ausente en uso normal).
        experiment_id=settings.experiment_id or None,
        config_label=settings.config_label or None,
        n_opt=settings.experiment_n_opt,
        npcs=settings.experiment_npcs,
        git_sha=_current_git_sha(),
        # Fase 16 — ablación de sub-planes y rigor de la batería.
        builtin_subplans=settings.builtin_subplans_enabled,
        capability_contracts_isolated=settings.isolate_capability_contracts,
        session_max_s=settings.session_max_s or None,
        coordination_planner=(
            settings.coordination_planner if settings.coordination_enabled else None
        ),
    )
    log.info("[MAIN] Trazas de sesión → %s", tl.session_dir)

    _patch_pyjabber_xml_parser_resilience()

    # Silenciar los "Fatal error on transport" de asyncio causados por
    # el XML mal formado de pyjabber en Windows (inofensivos).
    def _asyncio_exception_handler(loop, context):
        exc = context.get("exception")
        msg = str(context.get("message", ""))
        exc_str = str(exc) if exc else ""
        if "XMLProtocol" in str(context.get("protocol", "")) or \
           "SAXParseException" in exc_str or \
           "ExpatError" in exc_str or \
           ("Fatal error on transport" in msg and "XMLProtocol" in str(context)):
            log.debug("[pyjabber] Transport error suprimido: %s", exc_str or msg)
            return
        loop.default_exception_handler(context)

    try:
        log.info("[MAIN] Arrancando contenedor SPADE con XMPP embebido")
        _run_with_embedded_spade(_run, _asyncio_exception_handler)

    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        exc = sys.exc_info()[1]
        if exc is not None and _is_address_in_use_error(exc):
            log.error(
                "[MAIN] No se pudo arrancar el servidor embebido. "
                "Comprueba si ya hay otra instancia ejecutándose o si los puertos 5222/7777 están ocupados."
            )
        log.exception("[MAIN] Error fatal")
        sys.exit(1)
    finally:
        # Garantizar que metrics.json se escribe aunque loop.stop() haya
        # abortado _run() antes de llegar al finalize() dentro del coroutine.
        from utils.trace_logger import TraceLogger as _TL
        _tl = _TL.get()
        if _tl is not None:
            _tl.finalize()
            log.info("[MAIN] Sesión de log finalizada (finally) → %s", _tl.session_dir)


if __name__ == "__main__":
    main()
