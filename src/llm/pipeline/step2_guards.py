from __future__ import annotations

"""Paso 2 del pipeline BDI: descripción NL del problema por variante.

Para cada condición negada (not success_condition), el LLM describe en
lenguaje natural:
  - Qué estado del mundo causa que esa condición no esté satisfecha.
  - Qué facts conocidos son relevantes (si los hay en las creencias actuales).

Esta descripción se usa como contexto en step3 para guiar la generación
de steps coherentes con el estado del mundo real.
"""

import logging
from typing import Callable, Awaitable

log = logging.getLogger(__name__)


async def run_step2(
    sig: str,
    description: str,
    neg_guard: str,
    llm_call: Callable[[str, str], Awaitable[str]],
    *,
    beliefs: dict | None = None,
    entity_catalog: dict | None = None,
    facts: list[str] | None = None,
    guards: list[str] | None = None,
    bound_variables: list[str] | None = None,
) -> dict:
    """
    Describe en NL qué causa que una condición no esté satisfecha.

    Args:
        sig:          Nombre BDI del goal.
        description:  Descripción detallada del goal (del paso 0).
        neg_guard:    Condición negada que se intenta resolver.
                      Ej: "not has_item(wheat, 1)"
        llm_call:     Función async (user_prompt, system_prompt) → raw_text.
        beliefs:      Creencias actuales del NPC (para contextualizar).
        entity_catalog: Vocabulario de entidades válidas.

    Returns:
        {
          "problem_nl": "The NPC does not have wheat. It spawns in the farmland zone.",
          "known_facts": ["item_spawn(wheat, farmland)"]  # facts inferidos del contexto
        }
    """
    from llm.prompts.planning import build_prompt
    from llm.parser import parse_llm_response
    from llm.schemas import Step2Response
    from llm.pipeline.structured_call import call_llm

    payload = {
        "task": "step2_problem",
        "sig": sig,
        "description": description,
        "neg_guard": neg_guard,
        "beliefs": beliefs or {},
        "entity_catalog": entity_catalog or {},
        "facts": facts or [],
        "guards": guards or [],
        "bound_variables": bound_variables or [],
    }

    for attempt in range(2):
        raw = await call_llm(llm_call, build_prompt(payload), Step2Response)
        result = parse_llm_response(raw, "step2_problem")

        if not isinstance(result, dict):
            log.warning(f"[STEP2:{sig}] Intento {attempt+1}: respuesta no es dict")
            continue

        problem_nl = result.get("problem_nl", "")
        if not problem_nl or len(problem_nl) < 5:
            log.warning(f"[STEP2:{sig}] Intento {attempt+1}: problem_nl vacío")
            payload["_retry_errors"] = ["problem_nl must be a non-empty description string."]
            continue

        result.setdefault("known_facts", [])
        log.info(f"[STEP2:{sig}] {neg_guard!r} → {problem_nl[:60]}...")
        return result

    # Fallback: contexto mínimo para no bloquear el pipeline
    log.warning(f"[STEP2:{sig}] Fallback vacío para: {neg_guard}")
    return {"problem_nl": f"The condition {neg_guard} is not satisfied.", "known_facts": []}


# ---------------------------------------------------------------------------
# Helpers heredados (usados por pipeline_runner y otros módulos)
# ---------------------------------------------------------------------------

def build_guard_expression(facts: list[dict], guards: list[dict]) -> str:
    """
    Compila la lista de facts + guards a una expresión ASL.
    Mantenido por compatibilidad con compile_plan_to_asl y tests.
    """
    parts: list[str] = []
    for f in facts:
        functor = f["functor"]
        args = ", ".join(str(a) for a in f.get("args", []))
        parts.append(f"{functor}({args})" if args else functor)
    for g in guards:
        parts.append(g["expr"])
    return " & ".join(parts) if parts else "true"


def extract_bound_variables(
    facts: list[dict], guards: list[dict]
) -> list[str]:
    """Extrae variables (args en mayúscula) ligadas por facts o guards."""
    variables: list[str] = []
    for f in facts:
        for arg in f.get("args", []):
            if isinstance(arg, str) and arg[0].isupper() and arg not in variables:
                variables.append(arg)
    return variables
