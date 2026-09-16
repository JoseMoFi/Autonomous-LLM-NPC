from __future__ import annotations

"""Paso 3 del pipeline BDI: Pasos del plan con detección de sub-planes."""

import logging
import re
from typing import Callable, Awaitable

log = logging.getLogger(__name__)


_HAS_ITEM_RE = re.compile(r"^\s*has_item\(\s*([A-Za-z_][\w]*)\s*,")


def regathered_satisfied_items(
    steps: list[dict], already_satisfied: list[str] | None, unsatisfied_condition: str | None,
) -> list[str]:
    """Fase 17t: items con `has_item` ya garantizado por el guard de la rama que el
    plan vuelve a buscar o recoger (Search/PickUp).

    Piloto 17s (CO5/ATOM): la rama "tengo el trigo y estoy en bakeri" del miller
    decía explorar farmland → Search/PickUp trigo → volver → Craft. Con el trigo ya
    en la mano el PickUp fallaba (no quedaba), la escalera volvía a esa rama y el
    LLM (determinista) devolvía la MISMA respuesta en los 8 replans: bucle. El
    prompt ya dice "Already satisfied — DO NOT generate steps for these"; aquí se
    comprueba y se pide corregir (no se fabrica nada). Si el item aparece en la
    condición que falla (p.ej. hace falta MÁS cantidad) no se aplica.
    """
    held = []
    for atom in already_satisfied or []:
        m = _HAS_ITEM_RE.match(str(atom))
        if m:
            held.append(m.group(1).lower())
    failing = (unsatisfied_condition or "").lower()
    held = [item for item in held if not re.search(rf"\b{re.escape(item)}\b", failing)]
    if not held:
        return []
    found: list[str] = []
    for step in steps:
        if step.get("type") != "action" or str(step.get("name", "")).lower() not in ("pickup", "search"):
            continue
        args = step.get("args")
        if isinstance(args, dict):
            first = args.get("itemId")
        elif isinstance(args, list) and args:
            first = args[0]
        else:
            first = None
        item = str(first or "").lower()
        if item in held and item not in found:
            found.append(item)
    return found


_PEER_CONDITION_ACTIONS = (
    ("peer_item_available", "await_peer"),
    ("peer_promised", "request_peer"),
)


def missing_peer_action(steps: list[dict], unsatisfied_condition: str | None) -> tuple[str, str] | None:
    """Fase 17v: (predicado, acción) si el peldaño del protocolo no usa la única acción
    que garantiza su condición (regla 13 del prompt: peer_promised → request_peer,
    peer_item_available → await_peer). None si cumple o no es un peldaño peer_*.

    Piloto 17u (CO5): la rama de esperar del baker escribió request_peer +
    achieve_explore_zone; reenviaba la petición, la escalera se atascaba y el goal
    se replanificaba antes de que llegara la harina. Con el error concreto el LLM
    lo corrige (4/4 reproduciendo el prompt). No se fabrica el paso.
    """
    condition = unsatisfied_condition or ""
    for predicate, action in _PEER_CONDITION_ACTIONS:
        if predicate not in condition:
            continue
        wanted = action.replace("_", "")
        names = {
            str(step.get("name", "")).lower().lstrip(".").replace("_", "")
            for step in steps if step.get("type") == "action"
        }
        return None if wanted in names else (predicate, action)
    return None


_RECIPE_INPUT_RE = re.compile(r"^\s*recipe_input\(\s*([\w]+)\s*,\s*([\w]+)\s*,")
_RECIPE_BELIEF_RE = re.compile(r"^\s*recipe\(\s*([\w]+)\s*,\s*[\w]+\s*,\s*([\w]+)\s*,")


def wrong_craft_ingredients(steps: list[dict], known_facts: list[str] | None) -> list[tuple[str, str, list[str]]]:
    """Fase 17x: Craft cuyo itemId no es un ingrediente de su receta, según los hechos
    del prompt (recipe_input/4 de step1b o recipe/4 de las creencias). Devuelve
    [(item, receta, ingredientes)]; recetas desconocidas no se comprueban.

    Fallo más repetido de ATOM en la tanda coop (17w): Craft(bread,
    Bread_from_flour_recipe) 12 veces en 8 replans (Unity: "No matching recipe");
    el LLM responde casi igual al mismo prompt, así que replanificar no lo arreglaba.
    Se pide corregirlo con el error concreto; no se cambia el argumento.
    """
    inputs: dict[str, list[str]] = {}
    for fact in known_facts or []:
        m = _RECIPE_INPUT_RE.match(str(fact)) or _RECIPE_BELIEF_RE.match(str(fact))
        if m:
            items = inputs.setdefault(m.group(1).lower(), [])
            if m.group(2).lower() not in items:
                items.append(m.group(2).lower())
    wrong: list[tuple[str, str, list[str]]] = []
    for step in steps:
        if step.get("type") != "action" or str(step.get("name", "")).lower().lstrip(".") != "craft":
            continue
        args = step.get("args")
        if isinstance(args, dict):
            item, recipe = args.get("itemId"), args.get("targetId")
        elif isinstance(args, list) and len(args) >= 2:
            item, recipe = args[0], args[1]
        else:
            continue
        expected = inputs.get(str(recipe).lower())
        if expected and str(item).lower() not in expected:
            wrong.append((str(item), str(recipe), expected))
    return wrong


def is_same_goal_sig(candidate: str, goal_sig: str) -> bool:
    """Fase 17r: `candidate` es el propio goal, también con binding (`sig__args`).

    known_goals trae las claves del plan_graph (sig+call_args, p.ej.
    `achieve_deliver_to_peer__npc_miller_bread_1`); solo se filtraba el sig exacto,
    así que el receptor de una petición veía su propio goal como sub-goal
    reutilizable, el LLM lo usaba y el plan se llamaba a sí mismo (piloto 17q, CO6).
    """
    return candidate == goal_sig or candidate.startswith(f"{goal_sig}__")


async def run_step3(
    sig: str,
    description: str,
    neg_guard: str,
    problem_nl: str,
    known_facts: list[str],
    existing_subgoals: list[str],
    llm_call: Callable[[str, str], Awaitable[str]],
    *,
    entity_catalog: dict | None = None,
    belief_gap_hints: list[str] | None = None,
    goal_name: str = "",
    npc_statement: str = "",
    facts: list[dict] | None = None,
    guards: list[dict] | None = None,
    bound_variables: list[str] | None = None,
    reasoning: dict | None = None,
    atomic_only: bool = False,
    replan_hint: str | None = None,
    already_satisfied: list[str] | None = None,
    unsatisfied_condition: str | None = None,
    builtin_subplans: bool = True,
    coordination: bool = False,
    peers: list | None = None,
) -> list[dict]:
    """Genera los pasos para resolver una variante del goal."""
    from llm.prompts.planning import build_prompt
    from llm.parser import parse_llm_response

    _goal_name = sig or goal_name
    _facts = facts or []
    _guards = guards or []
    _bound_vars: set[str] = set(bound_variables or [])

    payload = {
        "task": "step3_steps",
        "goal_name": _goal_name,
        "npc_statement": description,
        "neg_guard": neg_guard,
        "problem_nl": problem_nl,
        "known_facts": known_facts,
        "facts": _facts,
        "guards": _guards,
        "bound_variables": list(dict.fromkeys(bound_variables or [])),
        "existing_subgoals": existing_subgoals,
        "entity_catalog": entity_catalog or {},
        "belief_gap_hints": belief_gap_hints or [],
        "atomic_only": atomic_only,
        "replan_hint": replan_hint or "",
        "already_satisfied": already_satisfied or [],
        "unsatisfied_condition": unsatisfied_condition or "",
        "builtin_subplans": builtin_subplans,
        "coordination": coordination,
        "peers": [list(p) for p in (peers or [])],
    }

    steps: list[dict] = []
    for attempt in range(2):
        _retry_errs = payload.pop("_retry_errors", None)
        # NOTA (Fase 3): step3 NO usa structured output. El esquema Step3Response
        # tiene `args: list[str|int|float]` (un anyOf en la gramática) que hace la
        # generación restringida de Ollama lentísima/poco fiable para algunos
        # variants (visto: una llamada colgada ~5 min → plan_failed). step3 (y el
        # repair/mapeo de step4) se quedan en la ruta texto+parse+repair, que es la
        # contribución central del TFM. El esquema queda como documentación/validación.
        raw = await llm_call(*build_prompt(payload, retry_errors=_retry_errs))
        result = parse_llm_response(raw, "step3_steps")

        if not isinstance(result, dict):
            log.warning(f"[STEP3:{_goal_name}] Intento {attempt+1}: respuesta no es dict")
            continue

        candidate = result.get("steps", [])
        if not candidate:
            log.warning(f"[STEP3:{_goal_name}] Intento {attempt+1}: steps vacio")
            payload["_retry_errors"] = ["steps must be a non-empty list."]
            continue

        if not all(isinstance(s, dict) and "type" in s and "name" in s for s in candidate):
            log.warning(f"[STEP3:{_goal_name}] Intento {attempt+1}: steps con formato invalido")
            continue

        # Rechazar sub-goals que referencian el propio goal (recursión directa).
        self_referential = [
            s["name"] for s in candidate
            if s.get("type") == "subgoal"
            and isinstance(s.get("name"), str)
            and is_same_goal_sig(s["name"], _goal_name)
        ]
        if self_referential:
            log.warning(
                f"[STEP3:{_goal_name}] Intento {attempt+1}: sub-goal auto-referencial "
                f"detectado — causaría recursión infinita"
            )
            payload["_retry_errors"] = [
                f"CRITICAL: the plan contains a sub-goal '{_goal_name}' which is the same "
                f"as the current goal being planned. This causes infinite recursion. "
                f"Use only sub-goals from the provided list that are DIFFERENT from '{_goal_name}'."
            ]
            continue

        missing = missing_peer_action(candidate, unsatisfied_condition)
        if missing:
            predicate, action = missing
            log.warning(
                f"[STEP3:{_goal_name}] Intento {attempt+1}: la condición {predicate} "
                f"no se resuelve sin {action}"
            )
            payload["_retry_errors"] = [
                f"CRITICAL: the failing condition requires {predicate}, which only {action} "
                f"guarantees (see Completeness rule 13): the plan must use {action}."
            ]
            continue

        wrong_crafts = wrong_craft_ingredients(candidate, known_facts)
        if wrong_crafts:
            log.warning(
                f"[STEP3:{_goal_name}] Intento {attempt+1}: Craft con itemId que no es "
                f"ingrediente de la receta: {wrong_crafts}"
            )
            payload["_retry_errors"] = [
                f"CRITICAL: in Craft(itemId, targetId, qty?) itemId must be the INGREDIENT that "
                f"{recipe} consumes ({', '.join(expected)}), not '{item}'."
                for item, recipe, expected in wrong_crafts
            ]
            continue

        regathered = regathered_satisfied_items(candidate, already_satisfied, unsatisfied_condition)
        if regathered:
            log.warning(
                f"[STEP3:{_goal_name}] Intento {attempt+1}: vuelve a recolectar {regathered} "
                f"con has_item ya garantizado por el guard"
            )
            payload["_retry_errors"] = [
                f"CRITICAL: the NPC already has {', '.join(regathered)} in this branch (see "
                f"'Already satisfied'). Do NOT Search or PickUp {', '.join(regathered)} again: "
                "use what it already holds and resolve only the failing condition."
            ]
            continue

        # Rechazar plans donde TODOS los steps son subgoals sin ninguna accion primitiva,
        # EXCEPTO si el último step es un sub-plan TERMINAL (encapsula primitivas y
        # garantiza su salida). Fuente única: llm.pipeline.builtins.TERMINAL_SUBPLANS.
        # Fase 16: sin sub-planes macro, solo cuentan los terminales activos.
        from llm.pipeline.builtins import active_terminal_subplans
        if all(s.get("type") == "subgoal" for s in candidate):
            last_name = candidate[-1].get("name", "") if candidate else ""
            if last_name not in active_terminal_subplans(builtin_subplans, coordination):
                log.warning(
                    f"[STEP3:{_goal_name}] Intento {attempt+1}: plan solo tiene subgoals "
                    f"sin acciones primitivas — reintentando"
                )
                payload["_retry_errors"] = [
                    "All steps are sub-goals with no primitive action. "
                    "You MUST include at least one primitive action: "
                    "MoveTo, Search, PickUp, ExploreArea, Craft, Drop, or Wait. "
                    "Sub-goals alone create infinite loops — ground the plan with real actions."
                ]
                continue
            log.debug(
                "[STEP3:%s] Plan all-subgoal accepted — terminal builtin '%s'",
                _goal_name, last_name,
            )

        steps = candidate
        break

    if not steps:
        raise ValueError(f"step3_steps failed after 2 attempts for goal: {_goal_name!r}")

    # Validacion semantica y reparacion contra el catalogo se realiza
    # en el paso 4 (step4_map.run_step4).
    return steps
