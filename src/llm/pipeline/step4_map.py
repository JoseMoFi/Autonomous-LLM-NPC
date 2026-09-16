from __future__ import annotations

"""Paso 4 del pipeline BDI: mapeo y validación de steps al catálogo.

Dado el conjunto de steps generado en el paso 3, este paso:
  1. Valida semánticamente cada step contra el catálogo de acciones y los
     bounds variables disponibles.
  2. Si hay errores semánticos, invoca al LLM (repair) para corregirlos.
  3. Devuelve los steps corregidos, garantizando que toda acción es una
     entrada válida del catálogo y todo sub-goal tiene nombre correcto.

Se separa intencionalmente de step3 para que la generación (step3) sea
libre, y la validación/mapeo al catálogo sea un paso explícito y auditable.
"""

import logging
from typing import Callable, Awaitable

log = logging.getLogger(__name__)


async def run_step4(
    goal_name: str,
    steps: list[dict],
    existing_subgoals: list[str],
    llm_call: Callable[[str, str], Awaitable[str]],
    *,
    facts: list[dict] | None = None,
    guards: list[dict] | None = None,
    bound_vars: set[str] | None = None,
    atomic_only: bool = False,
) -> list[dict]:
    """
    Mapea y valida los steps del paso 3 contra el catálogo de acciones.

    Para cada step con type='action':
      - El name debe ser una entrada de PRIMITIVE_ACTIONS.
      - Los args deben ser válidos (tipos, aridad, variables ligadas).
    Para steps type='subgoal':
      - El name debe seguir la convención achieve_verb_object o existir en
        los planes disponibles.

    Si se detectan errores de mapeo, se invoca un LLM (repair) para corregirlos.

    Args:
        goal_name:         Nombre del goal en curso.
        steps:             Steps del paso 3 (pueden tener errores de mapeo).
        existing_subgoals: Plans ya disponibles (reusables como sub-goals).
        llm_call:          Función async (user_prompt, system_prompt) → raw_text.
        facts:             Facts activos (para validación de variables).
        guards:            Guards activos (para validación de variables).
        bound_vars:        Variables ligadas disponibles (uppercase).

    Returns:
        Lista de steps corregidos y validados.
    """
    from llm.pipeline.step3_validator import validate_steps
    from llm.pipeline.step3_repair import repair_failing_steps

    _facts = facts or []
    _guards = guards or []
    _bound_vars = bound_vars or set()

    # Subgoals son válidos en cualquier modo — solo validamos steps de tipo "action".
    action_steps = [s for s in steps if isinstance(s, dict) and s.get("type") == "action"]
    step_errors = validate_steps(action_steps, _facts, _guards, _bound_vars)
    hard_errors = [e for e in step_errors if e.has_errors]

    if not hard_errors:
        log.debug(f"[STEP4:{goal_name}] Sin errores de mapeo — {len(steps)} steps válidos")
        return steps

    # Hay errores en acciones primitivas: reparar el plan completo (incluyendo subgoals
    # como contexto) para que el LLM mantenga la estructura correcta.
    log.info(
        f"[STEP4:{goal_name}] {len(hard_errors)} step(s) con errores de mapeo — "
        "reparando contra catálogo"
    )
    repaired = await repair_failing_steps(
        goal_name=goal_name,
        steps=steps,
        step_errors=hard_errors,
        facts=_facts,
        guards=_guards,
        bound_vars=_bound_vars,
        existing_subgoals=existing_subgoals,
        llm_call=llm_call,
        reasoning=None,
    )

    log.info(f"[STEP4:{goal_name}] Reparación completada — {len(repaired)} steps")
    return repaired
