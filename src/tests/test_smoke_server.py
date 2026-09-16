"""Smoke tests — TCPServer (0.C6: reemplazo de conexión)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.server import TCPServer


@pytest.mark.asyncio
async def test_second_connection_closes_first() -> None:
    """Una nueva conexión Unity cierra explícitamente la anterior (0.C6)."""
    registry = MagicMock()
    router = MagicMock()
    server = TCPServer("127.0.0.1", 7777, router, registry)

    old_writer = MagicMock()
    old_writer.is_closing.return_value = False
    old_writer.wait_closed = AsyncMock()
    old_writer.get_extra_info.return_value = ("127.0.0.1", 1111)
    server._current_writer = old_writer

    new_reader = MagicMock()
    new_reader.readline = AsyncMock(return_value=b"")  # EOF inmediato
    new_writer = MagicMock()
    new_writer.get_extra_info.return_value = ("127.0.0.1", 2222)

    await server._handle_client(new_reader, new_writer)

    # La conexión antigua se cerró al llegar la nueva.
    old_writer.close.assert_called()
    old_writer.wait_closed.assert_awaited()
    # Tras el EOF de la nueva, el estado global queda limpio.
    assert server._current_writer is None
    registry.resume_all.assert_called()
