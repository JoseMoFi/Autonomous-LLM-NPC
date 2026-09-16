from __future__ import annotations

"""Reparación selectiva de steps fallidos (step3).

Para los steps que no superaron la validación semántica, envía al LLM
un prompt con el contexto completo del goal, las variables disponibles
y los errores concretos de cada step fallido.

El LLM devuelve solo las correcciones (índice + step corregido), que
se re-inyectan en sus posiciones originales. Los steps correctos no
se tocan nunca.

Flujo:
  1. run_step3 obtiene steps del LLM.
  2. validate_steps detecta hard errors.
  3. repair_failing_steps:
       a. Construye prompt con errores contextualizados.
       b. Llama LLM UNA sola vez (batch repair sin bucle infinito).
       c. Re-inyecta fixes válidos, descarta los inválidos.
       d. Retorna lista combinada: steps originales + correcciones aplicadas.
"""

import json
import logging
from typing import Callable, Awaitable

from llm.pipeline.step3_validator import (
    StepError,
    build_fact_var_map,
    validate_steps,
)

log = logging.getLogger(__name__)


async def repair_failing_steps(
    goal_name: str,
    steps: list[dict],
    step_errors: list[StepError],
    facts: list[dict],
    guards: list[dict],
    bound_vars: set[str],
    existing_subgoals: list[str],
    llm_call: Callable[[str, str], Awaitable[str]],
    *,
    reasoning: dict | None = None,
) -> list[dict]:
    """
    Intenta reparar los steps fallidos enviando sus errores al LLM.

    Args:
        goal_name:       Nombre del goal para contexto.
        steps:           Lista completa de steps original (puede contener válidos e inválidos).
        step_errors:     Lista de StepError (solo los que tienen errors != []).
        facts:           Facts del paso 2 (para contexto de variables).
        guards:          Guards del paso 2.
        bound_vars:      Variables ligadas por facts/guards.
        existing_subgoals: Sub-goals disponibles.
        llm_call:        Función async (user_prompt, system_prompt) → raw_text.
        reasoning:       Resultado de step0 (contexto NL adicional, opcional).

    Returns:
        Lista de steps con los fallos reparados en su posición original.
        Si la reparación falla o produce resultados inválidos, devuelve
        los steps originales sin modificar.
    """
    from llm.prompts.planning import build_prompt
    from llm.parser import parse_llm_response

    fact_var_map = build_fact_var_map(facts)

    failing_items = [
        {
            "index": err.step_index,
            "step": err.step,
            "errors": err.errors,
        }
        for err in step_errors
        if err.has_errors
    ]

    if not failing_items:
        return steps

    payload = {
        "task": "step3_repair",
        "goal_name": goal_name,
        "all_steps": steps,
        "failing_steps": failing_items,
        "bound_vars": sorted(bound_vars),
        "fact_var_map": fact_var_map,
        "facts": facts,
        "guards": guards,
        "existing_subgoals": existing_subgoals,
        "reasoning": reasoning,
    }

    raw = await llm_call(*build_prompt(payload))
    result = parse_llm_response(raw, "step3_repair")

    if not isinstance(result, dict):
        log.warning(f"[REPAIR:{goal_name}] Respuesta no parseable, usando steps originales")
        return steps

    fixes = result.get("fixes", [])
    if not isinstance(fixes, list) or not fixes:
        log.warning(f"[REPAIR:{goal_name}] Sin fixes en respuesta, usando steps originales")
        return steps

    # Aplicar fixes válidos sobre una copia de los steps originales
    repaired = list(steps)
    n_applied = 0
    valid_indices = {item["index"] for item in failing_items}

    for fix in fixes:
        if not isinstance(fix, dict):
            continue
        idx = fix.get("index")
        new_step = fix.get("step")
        if not isinstance(idx, int) or idx not in valid_indices:
            log.debug(f"[REPAIR:{goal_name}] Fix con índice inválido {idx!r} ignorado")
            continue
        if not isinstance(new_step, dict) or "type" not in new_step or "name" not in new_step:
            log.debug(f"[REPAIR:{goal_name}] Fix malformado en índice {idx} ignorado: {new_step!r}")
            continue
        if 0 <= idx < len(repaired):
            repaired[idx] = new_step
            n_applied += 1

    if n_applied == 0:
        log.warning(f"[REPAIR:{goal_name}] Ningún fix aplicable, usando steps originales")
        return steps

    # Post-validación: si los steps reparados tienen más errores que los originales,
    # preferir los originales para no empeorar el resultado.
    original_error_count = sum(len(e.errors) for e in step_errors)
    repaired_errors = validate_steps(repaired, facts, guards, bound_vars)
    repaired_error_count = sum(len(e.errors) for e in repaired_errors)

    if repaired_error_count >= original_error_count:
        log.warning(
            f"[REPAIR:{goal_name}] Reparación no mejoró el resultado "
            f"({original_error_count} → {repaired_error_count} errores), revirtiendo"
        )
        return steps

    log.info(
        f"[REPAIR:{goal_name}] {n_applied}/{len(failing_items)} steps reparados. "
        f"Errores: {original_error_count} → {repaired_error_count}"
    )
    return repaired
