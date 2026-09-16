from __future__ import annotations

"""Paso 0 del pipeline BDI: NL → nombre BDI + descripción detallada.

Convierte un texto en lenguaje natural (goal de Unity o descripción de
un sub-plan encolado) en:
  - sig:         nombre BDI en snake_case  (achieve_verb_object)
  - description: descripción detallada en inglés de qué quiere lograr el NPC

La description se usa como contexto rico en el paso 1 para derivar
las success_conditions y en sub-planes encolados para no arrancar en vacío.
"""

import logging
from typing import Callable, Awaitable

log = logging.getLogger(__name__)


async def run_step0(
    goal_nl: str,
    llm_call: Callable[[str, str], Awaitable[str]],
    *,
    npc_profile: dict | None = None,
) -> dict:
    """
    Convierte un texto NL en nombre + descripción BDI.

    Args:
        goal_nl:     Texto NL del goal (cualquier idioma) o descripción de sub-plan.
        llm_call:    Función async (user_prompt, system_prompt) → raw_text.
        npc_profile: Perfil del NPC (opcional, aporta contexto de rol/personalidad).

    Returns:
        {"sig": "achieve_verb_object", "description": "..."}
    """
    from llm.prompts.planning import build_prompt
    from llm.parser import parse_llm_response
    from llm.validator import validate_goal_name
    from llm.schemas import Step0Response
    from llm.pipeline.structured_call import call_llm

    payload = {
        "task": "step0_name",
        "goal_nl": goal_nl,
        "npc_profile": npc_profile or {},
    }

    for attempt in range(2):
        # Fase 3: structured output (la FORMA la garantiza Step0Response). Si el
        # llm_call no acepta `schema` (mocks de test de 2 args), call_llm cae a
        # texto plano automáticamente.
        raw = await call_llm(llm_call, build_prompt(payload), Step0Response)
        result = parse_llm_response(raw, "step0_name")

        if not isinstance(result, dict):
            log.warning(f"[STEP0] Intento {attempt+1}: respuesta no es dict")
            continue

        sig = result.get("sig", "")
        description = result.get("description", "")

        if validate_goal_name(sig):
            log.warning(f"[STEP0] Intento {attempt+1}: nombre inválido: {sig!r}")
            payload["_retry_errors"] = [f"Invalid goal name: {sig!r}. Must be achieve_verb_object in snake_case."]
            continue

        if not description or len(description) < 10:
            log.warning(f"[STEP0] Intento {attempt+1}: descripción vacía")
            payload["_retry_errors"] = ["description is missing or too short. Provide a full sentence."]
            continue

        return {"sig": sig, "description": description}

    raise ValueError(f"step0_name failed after 2 attempts for: {goal_nl!r}")
