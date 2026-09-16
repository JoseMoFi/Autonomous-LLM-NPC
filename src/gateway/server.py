from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys

from utils.trace_logger import unity_log as _unity_log

log = logging.getLogger(__name__)


class TCPServer:
    """
    Python es el servidor TCP. Escucha en :7777.
    Unity se conecta como cliente. Si Unity se desconecta y vuelve
    a conectar, la nueva conexión es aceptada automáticamente.
    """

    def __init__(self, host: str, port: int, router: "MessageRouter", registry: "NPCRegistry"):
        self.host = host
        self.port = port
        self.router = router
        self.registry = registry
        self._server: asyncio.AbstractServer | None = None
        self._current_writer: asyncio.StreamWriter | None = None
        self._running = False
        self._wait_dots_index = 0
        self._waiting_task: asyncio.Task | None = None
        self._wait_line_visible = False
        self._interactive_stdout = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    async def start(self) -> None:
        self._running = True
        self._waiting_task = asyncio.create_task(self._waiting_unity_loop())
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        log.info(f"[TCP] Escuchando en {self.host}:{self.port}")
        try:
            async with self._server:
                await self._server.serve_forever()
        finally:
            self._running = False
            if self._waiting_task is not None:
                self._waiting_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._waiting_task

    async def _waiting_unity_loop(self) -> None:
        dots = (".", "..", "...", "....")
        while self._running:
            if self._current_writer is None:
                suffix = dots[self._wait_dots_index % len(dots)]
                self._wait_dots_index += 1
                self._render_wait_status(suffix)
            else:
                self._clear_wait_status()
            await asyncio.sleep(0.5)
        self._clear_wait_status(newline=True)

    def _render_wait_status(self, suffix: str) -> None:
        if not self._interactive_stdout:
            return
        message = f"[TCP] Esperando conectar a Unity{suffix}   "
        sys.stdout.write("\r" + message)
        sys.stdout.flush()
        self._wait_line_visible = True

    def _clear_wait_status(self, newline: bool = False) -> None:
        if not self._interactive_stdout or not self._wait_line_visible:
            return
        sys.stdout.write("\r" + (" " * 80) + "\r")
        if newline:
            sys.stdout.write("\n")
        sys.stdout.flush()
        self._wait_line_visible = False

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        addr = writer.get_extra_info("peername")
        self._clear_wait_status()
        log.info(f"[TCP] Unity conectado desde {addr}")
        _unity_log(direction="event", note=f"connected from {addr}")
        # 0.C6 — Si ya hay una conexión activa, cerrarla explícitamente antes de
        # aceptar la nueva. Evita dos clientes Unity simultáneos pisándose el
        # _current_writer (comportamiento indefinido).
        old_writer = self._current_writer
        if old_writer is not None and not old_writer.is_closing():
            log.warning("[TCP] Nueva conexión Unity reemplaza a la anterior — cerrando la antigua")
            _unity_log(direction="event", note="replaced previous connection")
            old_writer.close()
            with contextlib.suppress(Exception):
                await old_writer.wait_closed()
        self._current_writer = writer
        self.registry.resume_all()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break  # EOF — Unity desconectó
                try:
                    # utf-8-sig strips UTF-8 BOM if Unity adds it
                    raw_line = line.decode("utf-8-sig").strip()
                    _unity_log(direction="in_raw", raw=raw_line)
                    msg = json.loads(raw_line)
                    await self.router.route(msg, writer)
                except json.JSONDecodeError as exc:
                    log.warning(f"[TCP] Mensaje malformado ignorado: {exc}")
                    _unity_log(direction="in_malformed", raw=line.decode("utf-8-sig", errors="replace").strip(), note=str(exc))
        finally:
            writer.close()
            # 0.C6 — solo limpiar estado global si esta conexión sigue siendo la
            # activa. Si ya fue reemplazada por una nueva (reconexión), no pisar
            # el _current_writer nuevo ni re-pausar a los NPCs que ya se reanudaron.
            if self._current_writer is writer:
                log.warning("[TCP] Unity desconectado — pausando NPCAgents")
                _unity_log(direction="event", note="disconnected")
                self.registry.pause_all()
                self._current_writer = None
            # Servidor sigue escuchando: aceptará la próxima reconexión de Unity

    async def stop(self) -> None:
        self._running = False
        self._clear_wait_status(newline=True)

        # Cerrar la conexion activa con Unity para que el cliente detecte
        # desconexion inmediatamente durante el shutdown.
        if self._current_writer and not self._current_writer.is_closing():
            self._current_writer.close()
            with contextlib.suppress(Exception):
                await self._current_writer.wait_closed()
        self._current_writer = None

        if self._server:
            self._server.close()
            await self._server.wait_closed()

    @property
    def current_writer(self) -> asyncio.StreamWriter | None:
        return self._current_writer
