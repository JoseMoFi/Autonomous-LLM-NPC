from __future__ import annotations

"""Paso 1 del pipeline BDI: success_conditions + variante .done.

Dado el nombre BDI y la descripción detallada del goal (del paso 0),
deriva las condiciones de éxito en formato ASL y genera la variante
'.done' (guard satisfecho → true.) que se emite siempre como primera
variante del plan.
"""

import logging
import re
from typing import Callable, Awaitable

log = logging.getLogger(__name__)

_GUARD_OPERATORS = (">=", "<=", "==", "!=", ">", "<")


def _split_conjunction(expression: str) -> list[str]:
    """Split the flat conjunctions emitted by the prompt into fragments."""
    if not expression:
        return []
    return [part.strip() for part in expression.split("&") if part.strip()]


def _is_guard_fragment(fragment: str) -> bool:
    return any(op in fragment for op in _GUARD_OPERATORS)


def _extract_bound_variables(facts: list[str]) -> list[str]:
    bound: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        for variable in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", fact):
            if variable not in seen:
                seen.add(variable)
                bound.append(variable)
    return bound


def _normalize_success_variant(entry: dict) -> dict:
    facts = [f.strip() for f in entry.get("facts", []) if isinstance(f, str) and f.strip()]
    guards = [g.strip() for g in entry.get("guards", []) if isinstance(g, str) and g.strip()]
    done_fragment = entry.get("done_fragment", "")
    if not done_fragment:
        done_fragment = " & ".join(facts + guards)
    done_fragment = done_fragment.strip()
    return {
        "facts": facts,
        "guards": guards,
        "done_fragment": done_fragment,
        "bound_variables": _extract_bound_variables(facts),
    }


def _derive_success_model_from_conditions(conditions: list[str]) -> list[dict]:
    success_model: list[dict] = []
    for condition in conditions:
        fragments = _split_conjunction(condition)
        facts = [fragment for fragment in fragments if not _is_guard_fragment(fragment)]
        guards = [fragment for fragment in fragments if _is_guard_fragment(fragment)]
        success_model.append(
            {
                "facts": facts,
                "guards": guards,
                "done_fragment": condition,
                "bound_variables": _extract_bound_variables(facts),
            }
        )
    return success_model


async def run_step1(
    sig: str,
    description: str,
    llm_call: Callable[[str, str], Awaitable[str]],
    *,
    entity_catalog: dict | None = None,
) -> dict:
    """
    Deriva el success_model estructurado para un goal.

    Args:
        sig:          Nombre BDI del goal (snake_case).
        description:  Descripción detallada del goal (del paso 0).
        llm_call:     Función async (user_prompt, system_prompt) → raw_text.
        entity_catalog: Vocabulario de entidades válidas.

    Returns:
                {
                    "success_model": [
                        {
                            "facts": ["has_item(wheat, N)"],
                            "guards": ["N >= 1"],
                            "done_fragment": "has_item(wheat, N) & N >= 1",
                            "bound_variables": ["N"],
                        }
                    ],
                    "success_conditions": ["has_item(wheat, N) & N >= 1"],
                    "done_guard": "has_item(wheat, N) & N >= 1",
                    "done_asl": "+!achieve_... : has_item(wheat, N) & N >= 1 <- true."
                }
    """
    from llm.prompts.planning import build_prompt
    from llm.parser import parse_llm_response
    from llm.schemas import Step1Response
    from llm.pipeline.structured_call import call_llm

    payload = {
        "task": "step1_success",
        "sig": sig,
        "description": description,
        "entity_catalog": entity_catalog or {},
    }

    for attempt in range(2):
        raw = await call_llm(llm_call, build_prompt(payload), Step1Response)
        result = parse_llm_response(raw, "step1_success")

        if not isinstance(result, dict):
            log.warning(f"[STEP1:{sig}] Intento {attempt+1}: respuesta no es dict")
            continue

        raw_model = result.get("success_model", [])
        success_model: list[dict]
        if isinstance(raw_model, list) and raw_model:
            success_model = [
                _normalize_success_variant(entry)
                for entry in raw_model
                if isinstance(entry, dict)
            ]
            success_model = [entry for entry in success_model if entry.get("done_fragment")]
        else:
            conditions = result.get("success_conditions", [])
            if not isinstance(conditions, list):
                conditions = []
            conditions = [c.strip() for c in conditions if isinstance(c, str) and c.strip()]
            success_model = _derive_success_model_from_conditions(conditions)

        if not success_model:
            log.warning(f"[STEP1:{sig}] Intento {attempt+1}: success_model vacío")
            payload["_retry_errors"] = [
                "success_model must be a non-empty list of variants with facts, guards, and done_fragment."
            ]
            continue

        conditions = [entry["done_fragment"] for entry in success_model]
        done_guard = result.get("done_guard", "").strip() or " & ".join(conditions)
        done_asl = f"+!{sig} : {done_guard} <- true."

        log.info(f"[STEP1:{sig}] success_conditions={conditions}")
        return {
            "success_model": success_model,
            "success_conditions": conditions,
            "done_guard": done_guard,
            "done_asl": done_asl,
        }

    raise ValueError(f"step1_success failed after 2 attempts for goal: {sig!r}")


def build_negated_guard(condition: str) -> str:
    """
    Niega un predicado ASL simple o una conjunción para usarlo como guard
    de la variante de ejecución.

    Ejemplos:
        "has_item(wheat, 1)"             → "not has_item(wheat, 1)"
        "has_item(wheat,1) & at_loc(b)"  → "not (has_item(wheat,1) & at_loc(b))"
    """
    condition = condition.strip()
    if " & " in condition:
        return f"not ({condition})"
    return f"not {condition}"
