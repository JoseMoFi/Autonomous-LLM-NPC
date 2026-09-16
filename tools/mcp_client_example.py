from __future__ import annotations

"""mcp_client_example.py — spade-llm como CLIENTE MCP del servidor de mundo (Fase 6, stretch).

Demuestra la interoperabilidad: spade-llm 0.3.0 trae soporte MCP nativo
(`spade_llm.mcp`). Aquí registramos el servidor `tools/mcp_world_server.py` como
servidor MCP por STDIO y dejamos que spade-llm DESCUBRA y EJECUTE sus tools a
través de su abstracción `LLMTool` — el mismo mecanismo que usaría un `LLMAgent`
para resolver una pregunta en lenguaje natural sobre el estado del mundo.

Este ejemplo NO requiere Ollama: ejercita el descubrimiento + ejecución de tools
(la capa de interop). Para conectarlo a un `LLMAgent` real (que razone en NL con
estas tools) basta con pasar `tools=get_all_mcp_tools([cfg])` al construir el
agente con el provider de spade-llm (ver DOC/MCP_SERVER.md, sección stretch).

Uso:
    .venv\\Scripts\\python.exe tools/mcp_client_example.py
"""

import asyncio
import json
import sys
from pathlib import Path

from spade_llm.mcp import StdioServerConfig, get_mcp_server_tools

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = sys.executable  # el intérprete actual (idealmente el venv 3.12)
_SERVER = str(_PROJECT_ROOT / "tools" / "mcp_world_server.py")


def _server_config() -> StdioServerConfig:
    return StdioServerConfig(
        name="world",
        command=_PYTHON,
        args=[_SERVER],
        cache_tools=True,
        read_timeout_seconds=30.0,
    )


def _unwrap(result: object) -> object:
    """spade-llm devuelve {'type':'text','text': <json>} o un str JSON. Normaliza a dict."""
    if isinstance(result, dict) and result.get("type") == "text":
        result = result.get("text", "")
    if isinstance(result, str):
        return json.loads(result)
    return result


async def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # consola Windows con acentos limpios
    except Exception:
        pass
    cfg = _server_config()
    tools = await get_mcp_server_tools(cfg)

    print(f"spade-llm descubrió {len(tools)} tools del servidor MCP de mundo:")
    for t in tools:
        print(f"  - {t.name}")

    # Ejecutar un par de tools a través de la abstracción LLMTool de spade-llm.
    by_name = {t.name: t for t in tools}

    catalog = by_name["world_action_catalog"]
    cat = _unwrap(await catalog.execute())
    print(f"\n[world_action_catalog] {cat['count']} acciones: "
          f"{[a['name'] for a in cat['actions']]}")

    validate = by_name["world_validate_plan"]
    val = _unwrap(await validate.execute(asl="+!achieve_x : true <- .teleport(1, 2)."))
    print(f"[world_validate_plan] ok={val['ok']} errors={val['errors']}")

    print("\nInterop spade-llm <-> servidor MCP: OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
