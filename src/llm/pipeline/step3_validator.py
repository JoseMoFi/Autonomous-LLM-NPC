from __future__ import annotations

"""Validador semántico de los pasos generados en step3.

Checks implementados:
  H1 - Arg count: número de args coincide con la signatura de la acción primitiva.
  H2 - Unbound vars: args uppercase no declarados en bound_vars → error bloqueante.
  W1 - Variable reuse: arg uppercase que existe en bound_vars pero no coincide
       con ninguna variable del mismo "rol semántico" en facts → advertencia
       (no bloquea, se incluye como contexto en el repair prompt).

Cada check devuelve una lista de strings de error/warning.
`validate_steps` agrega todo en `StepError` por índice.
"""

import re
from dataclasses import dataclass, field

# ACTION_SIGNATURES se deriva automáticamente de ACTION_ALLOWLIST en catalogs.py
from llm.catalogs import ACTION_SIGNATURES  # noqa: F401  (re-used in this module)


# ---------------------------------------------------------------------------
# Tipo de resultado por step
# ---------------------------------------------------------------------------

@dataclass
class StepError:
    """Errores y advertencias encontrados en un step concreto."""
    step_index: int
    step: dict
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_fact_var_map(facts: list[dict]) -> dict[str, str]:
    """
    Construye un mapa variable → descripción de origen para mensajes de error.

    Ejemplo: {"N": "has_item arg[1] (qty)", "X": "zone_center arg[1]", ...}
    """
    var_map: dict[str, str] = {}
    for fact in facts:
        functor = fact.get("functor", "?")
        for i, arg in enumerate(fact.get("args", [])):
            if isinstance(arg, str) and arg and arg[0].isupper():
                # Evitar sobreescritura si ya hay una fuente anterior
                if arg not in var_map:
                    var_map[arg] = f"{functor} arg[{i}]"
    return var_map


def _is_variable(arg: str) -> bool:
    """True si el arg sigue la convención Prolog/ASL: empieza por mayúscula.

    EXCEPTION: PascalCase identifiers that contain underscore followed by a
    lowercase letter (e.g. Bread_recipe, Bakeri_point) are treated as constants
    — they are recipe IDs / delivery point IDs in this codebase, not ASL variables.
    Pure single-word mixed-case args (X, Y, N, ItemId, ZoneTag, Have) ARE variables.
    """
    if not arg or not arg[0].isupper():
        return False
    if not re.match(r'^[A-Z][A-Za-z0-9_]*$', arg):
        return False
    # PascalCase with underscore-lowercase → recipe/delivery constant, not a variable
    if re.search(r'_[a-z]', arg):
        return False
    return True


# ---------------------------------------------------------------------------
# Checks individuales
# ---------------------------------------------------------------------------

def check_arg_count(step: dict) -> list[str]:
    """
    H1 — Verifica que el número de args coincide con la signatura de la acción.
    Solo aplica a steps de tipo "action" con nombre en ACTION_SIGNATURES.
    Subgoals no tienen signatura fija y se omiten.
    """
    if step.get("type") != "action":
        return []

    name = step.get("name", "")
    sig = ACTION_SIGNATURES.get(name.upper())
    if sig is None:
        return []  # Acción desconocida: no podemos validar

    min_args, max_args, arg_names = sig
    n_args = len(step.get("args", []))
    if not (min_args <= n_args <= max_args):
        expected = (
            f"{min_args}" if min_args == max_args else f"{min_args}-{max_args}"
        )
        return [
            f"Arg count: {name}({', '.join(arg_names)}) "
            f"expects {expected} arg(s), got {n_args}."
        ]
    return []


def check_unbound_vars(
    step: dict,
    bound_vars: set[str],
    fact_var_map: dict[str, str],
) -> list[str]:
    """
    H2 — Detecta variables uppercase en los args que no están en bound_vars.

    Cuando un arg es una variable no ligada, incluye en el error qué variables
    SÍ están disponibles y de dónde vienen, para que el LLM pueda corregirlo.
    """
    errors: list[str] = []
    for arg in step.get("args", []):
        if not isinstance(arg, str):
            continue
        if not _is_variable(arg):
            continue
        if arg in bound_vars:
            continue
        # Variable uppercase no ligada
        available = ", ".join(
            f"{v} (from {origin})" for v, origin in fact_var_map.items()
        ) or "(none)"
        errors.append(
            f"Unbound variable '{arg}' — not declared in bound_vars. "
            f"Available: {available}."
        )
    return errors


def check_variable_reuse(
    step: dict,
    bound_vars: set[str],
    fact_var_map: dict[str, str],
) -> list[str]:
    """
    W1 — Advertencia: un arg uppercase está en bound_vars pero el step usa
    otro nombre uppercase que no está. Sugiere la variable canónica.

    Nota: es una advertencia, no bloquea la ejecución.
    """
    warnings: list[str] = []
    step_vars = {
        arg for arg in step.get("args", [])
        if isinstance(arg, str) and _is_variable(arg) and arg in bound_vars
    }
    orphan_vars = {
        arg for arg in step.get("args", [])
        if isinstance(arg, str) and _is_variable(arg) and arg not in bound_vars
    }
    # Si hay vars ligadas usadas, el step es coherente → no advertimos
    if step_vars:
        return []
    # Si no hay ninguna var ligada y tampoco orphans, es un paso de constantes → OK
    if not orphan_vars:
        return []
    # Si solo hay orphans: ya están cubiertas por H2 (check_unbound_vars),
    # en este check solo queremos el caso de "mixing coherente".
    return warnings  # actualmente vacío; reservado para extensiones futuras


# ---------------------------------------------------------------------------
# W2 (T5, Fase 6.5) — variante de recolección sin acotar por estado del mundo
# ---------------------------------------------------------------------------

_GATHER_SUBGOALS = {"move_to_and_pickup"}
_GATHER_ACTIONS = {"pickup", "search", "moveto"}


def gather_variant_underconstrained(guard: str, steps: list[dict]) -> bool:
    """W2 (T5): True si la variante RECOLECTA (move_to_and_pickup / PickUp /
    Search) pero su guard NO la acota por estado del mundo ni por el item — rama
    demasiado permisiva: podría disparar para un item que en realidad se craftea.

    Se considera ACOTADA si el guard referencia `item_spawn`, `recipe_output`, o
    un `has_item` (la partición por ausencia de item que usa step1b la cumple). Es
    solo un AVISO defensivo (no bloquea): las variantes ground por binding son
    seguras, pero una rama de gather con guard `true`/vacío sí es sospechosa.
    """
    has_gather = any(
        isinstance(s, dict) and (
            (s.get("type") == "subgoal" and str(s.get("name", "")).lower() in _GATHER_SUBGOALS)
            or (s.get("type") == "action" and str(s.get("name", "")).lower() in _GATHER_ACTIONS)
        )
        for s in steps
    )
    if not has_gather:
        return False
    g = (guard or "").lower()
    constrained = ("item_spawn" in g) or ("recipe_output" in g) or ("has_item" in g)
    return not constrained


# ---------------------------------------------------------------------------
# Validador principal
# ---------------------------------------------------------------------------

def validate_steps(
    steps: list[dict],
    facts: list[dict],
    guards: list[dict],
    bound_vars: set[str],
) -> list[StepError]:
    """
    Valida todos los steps y devuelve los errores agrupados por índice.

    Solo devuelve StepError para steps que tienen al menos un error o warning.
    Steps limpios no aparecen en la lista.
    """
    fact_var_map = build_fact_var_map(facts)
    results: list[StepError] = []

    for i, step in enumerate(steps):
        errors: list[str] = []
        warnings: list[str] = []

        errors.extend(check_arg_count(step))
        errors.extend(check_unbound_vars(step, bound_vars, fact_var_map))
        warnings.extend(check_variable_reuse(step, bound_vars, fact_var_map))

        if errors or warnings:
            results.append(StepError(
                step_index=i,
                step=step,
                errors=errors,
                warnings=warnings,
            ))

    return results
