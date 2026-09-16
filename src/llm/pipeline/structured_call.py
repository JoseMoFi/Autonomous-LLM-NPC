from __future__ import annotations

"""Helper para invocar el llm_call del pipeline con structured output opcional.

Los pasos del pipeline reciben un `llm_call(user, system) -> str`. En producción
ese callable (definido en planning_agent) acepta además `schema=` para forzar la
salida a un esquema Pydantic (spade-llm). Los mocks de test son de 2 args.

`call_llm` detecta por inspección si el callable admite `schema` y lo usa; si no
(tests), cae a texto plano. NO se captura TypeError (no se enmascaran errores
reales del propio callable): la decisión es por firma.
"""

import inspect
from typing import Any, Awaitable, Callable


def _accepts_schema(fn: Callable[..., Awaitable[str]]) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
    if "schema" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


async def call_llm(
    llm_call: Callable[..., Awaitable[str]],
    prompt: tuple[str, str],
    schema: type[Any] | None = None,
) -> str:
    """Llama a llm_call(user, system) con `schema=` si el callable lo soporta."""
    user, system = prompt
    if schema is not None and _accepts_schema(llm_call):
        return await llm_call(user, system, schema=schema)
    return await llm_call(user, system)
