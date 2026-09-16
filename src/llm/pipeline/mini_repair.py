from __future__ import annotations

"""mini_repair.py — Fase 6.5: mini-repairs dirigidos al LLM en vez de fabricar.

Política de no-fabricación: el código NO inventa lo que el LLM no puso. Cuando un
step del plan llega incompleto, se pregunta al LLM por esa necesidad concreta; si
no la resuelve, el plan FALLA (ruidoso y trazado), nunca se rellena con un valor
por defecto. Todo lo que se decide aquí queda en la traza con `source`:
  - "LLM"  → el modelo aportó la información que faltaba (mini-repair).
  - "CODE" → solo cuando el LLM autoriza explícitamente (p.ej. quitar pasos).

Sustituye dos parches silenciosos previos:
  #3a  `move_to_and_pickup` sin qty → antes `qty=1` por defecto.   Ahora: mini-repair.
  #5b  pasos tras un builtin terminal → antes truncado por heurística. Ahora:
       se MANTIENEN salvo que el LLM confirme que sobran (sin truncado silencioso).
"""

import logging
from typing import Awaitable, Callable

from llm.parser import parse_llm_response
from llm.pipeline.builtins import TERMINAL_SUBPLANS
from utils.trace_logger import plan_transform as _plan_transform

log = logging.getLogger(__name__)

LLMCall = Callable[[str, str], Awaitable[str]]

# Aridad y nombres de argumento de los sub-planes builtin parametrizados.
_BUILTIN_ARGS: dict[str, list[str]] = {
    "move_to_and_pickup": ["itemId", "qty"],
    "craft_item": ["recipeId", "qty"],
    "achieve_explore_zone": ["zoneTag"],
    # Fase 17: macros de coordinación (solo activos con coordinación y sub-planes).
    "obtain_from_peer": ["peerId", "itemId", "qty"],
    "collect_from_peer": ["peerId", "itemId", "qty"],
}

# Acciones PRIMITIVAS (crudas) que cada builtin terminal ya ejecuta internamente.
# Solo cuando TODOS los pasos colgantes tras el terminal son acciones de este
# conjunto tiene sentido el caso de redundancia (p.ej. un PickUp/MoveTo tras
# move_to_and_pickup, un Craft tras craft_item) → ahí preguntamos al LLM. Si cuelga
# un subgoal u otra acción no subsumida (p.ej. un Drop), se conserva SIN gastar una
# llamada LLM: no es el patrón de redundancia. Nombres en minúsculas para comparar.
_TERMINAL_SUBSUMES: dict[str, frozenset[str]] = {
    "move_to_and_pickup": frozenset({"moveto", "pickup"}),
    "craft_item": frozenset({"moveto", "craft"}),
    # Solo garantiza zone_center/knows_zone (ExploreArea si la zona no se conoce): no
    # va a ningún item ni lo busca. Con moveto/search aquí se preguntaba al LLM, que
    # "confirmaba" que un Search posterior sobraba, y el peldaño quedaba sin acciones
    # (tanda corta, A1 ATOM).
    "achieve_explore_zone": frozenset({"explorearea"}),
    "obtain_from_peer": frozenset({"moveto", "pickup"}),
    "collect_from_peer": frozenset({"moveto", "pickup"}),
}


class MiniRepairError(ValueError):
    """Un hueco del plan no se pudo reparar preguntando al LLM → el plan falla."""


def _clean_args(step: dict) -> list:
    return [a for a in (step.get("args") or []) if a is not None]


async def repair_plan_gaps(
    steps: list[dict],
    sig: str,
    llm_call: LLMCall,
    *,
    entity_catalog: dict | None = None,
    npc_id: str | None = None,
    terminal_subplans: frozenset[str] | None = None,
) -> list[dict]:
    """Repara huecos del plan vía mini-repairs. Devuelve la lista de steps.

    Lanza `MiniRepairError` si un sub-goal con args faltantes no se puede
    completar preguntando al LLM (fallo ruidoso, nunca default silencioso).

    `terminal_subplans`: sub-planes terminales activos (Fase 16: sin los
    ablacionados). None = TERMINAL_SUBPLANS. Un nombre fuera de ese conjunto es
    un sub-goal más (del LLM), no un builtin: no se le aplican estos repairs.
    """
    terminal = TERMINAL_SUBPLANS if terminal_subplans is None else terminal_subplans
    await _repair_missing_subgoal_args(steps, sig, llm_call, entity_catalog, npc_id, terminal)
    steps = await _resolve_trailing_steps(steps, sig, llm_call, npc_id, terminal)
    return steps


# ---------------------------------------------------------------------------
# #3a — sub-goal con args faltantes → preguntar al LLM (no default)
# ---------------------------------------------------------------------------

async def _repair_missing_subgoal_args(
    steps: list[dict],
    sig: str,
    llm_call: LLMCall,
    entity_catalog: dict | None,
    npc_id: str | None,
    terminal: frozenset[str] = TERMINAL_SUBPLANS,
) -> None:
    for step in steps:
        if step.get("type") != "subgoal":
            continue
        name = step.get("name", "")
        spec = _BUILTIN_ARGS.get(name) if name in terminal else None
        if spec is None:
            continue
        args = _clean_args(step)
        if len(args) >= len(spec):
            continue  # completo

        new_args = await _ask_missing_args(name, spec, args, sig, llm_call, entity_catalog)
        if new_args is None or len(new_args) != len(spec):
            raise MiniRepairError(
                f"[{sig}] sub-goal '{name}' necesita {len(spec)} args {spec}, "
                f"el LLM dio {args} y el mini-repair no lo resolvió — plan abortado "
                f"(no se inventa un valor por defecto)."
            )
        _plan_transform(
            "mini_repair_subgoal_args", "LLM",
            npc_id=npc_id, before=args, after=new_args,
            reason=f"{name} requería {spec}",
        )
        step["args"] = new_args


async def _ask_missing_args(
    name: str,
    spec: list[str],
    given: list,
    sig: str,
    llm_call: LLMCall,
    entity_catalog: dict | None,
) -> list | None:
    cat = entity_catalog or {}
    catalog_lines = []
    for key, label in (
        ("item_ids", "items"), ("zone_ids", "zones"),
        ("recipe_ids", "recipes"), ("delivery_tags", "delivery points"),
    ):
        vals = cat.get(key)
        if vals:
            catalog_lines.append(f"{label}: {', '.join(vals)}")
    system = (
        "You complete a sub-goal call that is missing arguments. "
        "Answer ONLY with a JSON object {\"args\": [...]} listing ALL arguments "
        "in order. Do not invent entities outside the catalog."
    )
    user = (
        f"Goal: {sig}\n"
        f"Sub-goal '{name}' takes arguments (in order): {spec}.\n"
        f"It currently has: {given}.\n"
        + ("World catalog:\n" + "\n".join(catalog_lines) + "\n" if catalog_lines else "")
        + f"Return the full argument list for {name} as JSON: "
        f'{{"args": [{", ".join(spec)}]}}.'
    )
    try:
        raw = await llm_call(user, system)
    except Exception as exc:
        log.warning("[MINI_REPAIR:%s] llm_call falló para '%s': %s", sig, name, exc)
        return None
    data = parse_llm_response(raw, "mini_repair_args")
    args = data.get("args") if isinstance(data, dict) else None
    if not isinstance(args, list) or not args:
        return None
    return args


# ---------------------------------------------------------------------------
# #5b — pasos tras un builtin terminal → preguntar (no truncar en silencio)
# ---------------------------------------------------------------------------

async def _resolve_trailing_steps(
    steps: list[dict],
    sig: str,
    llm_call: LLMCall,
    npc_id: str | None,
    terminal: frozenset[str] = TERMINAL_SUBPLANS,
) -> list[dict]:
    term_idx = next(
        (i for i, s in enumerate(steps)
         if s.get("type") == "subgoal" and s.get("name") in terminal),
        None,
    )
    if term_idx is None or term_idx + 1 >= len(steps):
        return steps  # no hay pasos colgando tras un builtin terminal

    terminal = steps[term_idx].get("name", "")
    trailing = steps[term_idx + 1:]

    # Acotar al caso REAL de redundancia: solo preguntamos al LLM si TODOS los pasos
    # colgantes son acciones crudas cuyo primitivo ya está subsumido por el builtin
    # terminal. En cualquier otro caso (un subgoal, o una acción no subsumida como
    # Drop) los conservamos sin gastar una llamada LLM — nunca se borra contenido del
    # LLM por heurística, y se evita la llamada extra por plan en el caso común.
    subsumed = _TERMINAL_SUBSUMES.get(terminal, frozenset())
    is_redundancy_pattern = all(
        s.get("type") != "subgoal"
        and str(s.get("name", "")).lower() in subsumed
        for s in trailing
    )
    if not is_redundancy_pattern:
        _plan_transform(
            "mini_repair_trailing_kept", "CODE",
            npc_id=npc_id, before=_names(steps), after=_names(steps),
            reason=f"pasos tras '{terminal}' conservados (no encajan en el patrón "
                   f"de redundancia; sin llamada LLM)",
        )
        return steps

    redundant = await _ask_trailing_redundant(sig, terminal, trailing, llm_call)
    if redundant is True:
        # El LLM autoriza quitarlos → la decisión es del LLM, se traza.
        _plan_transform(
            "mini_repair_trailing_removed", "LLM",
            npc_id=npc_id, before=_names(steps), after=_names(steps[: term_idx + 1]),
            reason=f"el LLM confirmó que los pasos tras '{terminal}' sobran",
        )
        return steps[: term_idx + 1]
    # Por defecto se MANTIENEN: no se borra contenido del LLM por heurística.
    _plan_transform(
        "mini_repair_trailing_kept", "CODE",
        npc_id=npc_id, before=_names(steps), after=_names(steps),
        reason=f"pasos tras '{terminal}' conservados (sin truncado silencioso)",
    )
    return steps


async def _ask_trailing_redundant(
    sig: str, terminal: str, trailing: list[dict], llm_call: LLMCall
) -> bool | None:
    system = (
        "You judge whether trailing plan steps are redundant. The named sub-goal "
        "is self-contained (it handles navigation and pick-up/craft internally). "
        "Answer ONLY with JSON {\"redundant\": true|false}."
    )
    user = (
        f"Goal: {sig}\n"
        f"Sub-goal '{terminal}' is self-contained.\n"
        f"Steps that come AFTER it: {_names(trailing)}\n"
        "Are those trailing steps redundant given that the sub-goal already "
        'completes its outcome? Reply {"redundant": true} or {"redundant": false}.'
    )
    try:
        raw = await llm_call(user, system)
    except Exception as exc:
        log.warning("[MINI_REPAIR:%s] llm_call (trailing) falló: %s", sig, exc)
        return None
    data = parse_llm_response(raw, "mini_repair_trailing")
    if isinstance(data, dict) and isinstance(data.get("redundant"), bool):
        return data["redundant"]
    return None


def _names(steps: list[dict]) -> list[str]:
    return [str(s.get("name", "?")) for s in steps if isinstance(s, dict)]
