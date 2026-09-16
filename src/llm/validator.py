from __future__ import annotations

import re

from llm.catalogs import PRIMITIVE_ACTIONS, BELIEF_CATALOG


def validate_plan_asl(asl_text: str, known_goals: set[str]) -> list[str]:
    """
    Valida un texto ASL compilado.
    Devuelve lista de errores (vacía = OK).
    """
    errors: list[str] = []

    # Sub-goals en el body deben existir en known_goals.
    # 0.D6 — el regex admite dígitos en el sig (p.ej. !achieve_get_wheat2).
    for sg in re.findall(r'!([a-z][a-z0-9_]*)', asl_text):
        if sg not in known_goals and f"achieve_{sg}" not in known_goals:
            errors.append(f"Sub-goal '!{sg}' no existe en known_goals ni en built-ins")

    # Acciones deben estar en PRIMITIVE_ACTIONS
    unity_names = {a.lower() for a in PRIMITIVE_ACTIONS}
    for a in re.findall(r'\.([a-z_]+)\(', asl_text):
        if a not in unity_names:
            errors.append(f"Acción '.{a}()' no está en PRIMITIVE_ACTIONS")

    # Beliefs en el guard deben estar en BELIEF_CATALOG
    catalog_names = {k.split("(")[0] for k in BELIEF_CATALOG}
    guard_part = asl_text.split("<-")[0] if "<-" in asl_text else ""
    for b in re.findall(r'([a-z_]+)\([^)]+\)', guard_part):
        if b not in catalog_names and b not in ("not", ):
            errors.append(f"Belief '{b}' no está en BELIEF_CATALOG — el guard puede no disparar")

    # 0.C2 — Warning si el body añade/retracta beliefs (+belief/-belief). El
    # estado persistente del NPC vive en BeliefStore y _sync_beliefs reconstruye
    # las beliefs ASL cada tick; cualquier +/- belief que haga un plan se pierde.
    body_part = asl_text.split("<-", 1)[1] if "<-" in asl_text else ""
    for op, functor in re.findall(r'(?:^|;|\s)([+-])([a-z_][a-z0-9_]*)\(', body_part):
        errors.append(
            f"WARNING: el body usa '{op}{functor}(...)' (belief add/remove): el "
            "estado ASL se reconstruye desde BeliefStore cada tick y se perderá. "
            "No usar +/-belief para estado persistente (ver _sync_beliefs)."
        )

    return errors


_VALID_GOAL_PREFIXES = (
    "achieve_", "get_", "find_", "craft_", "deliver_", "flee_", "explore_",
)


def validate_goal_name(name: str) -> list[str]:
    """
    Valida que el nombre del goal siga la convención <prefijo>_<sustantivo>.
    Prefijos válidos: achieve_, get_, find_, craft_, deliver_, flee_, explore_
    """
    errors: list[str] = []
    if not any(re.match(rf'^{p}[a-z][a-z_]*$', name) for p in _VALID_GOAL_PREFIXES):
        errors.append(
            f"'{name}' no sigue el patrón <prefijo>_<sustantivo>. "
            "Prefijos válidos: achieve_, get_, find_, craft_, deliver_, flee_, explore_. "
            "Usa snake_case inglés, p.ej.: achieve_deliver_bread, get_wheat, craft_bread"
        )
    return errors


def validate_plan_json(plan_json: dict, known_goals: set[str]) -> list[str]:
    """
    Valida el JSON de un plan antes de compilarlo a ASL (Ronda A / Paso 3).
    """
    errors: list[str] = []
    sig = plan_json.get("sig", "")
    guard = plan_json.get("guard", "")
    body = plan_json.get("body", [])

    errors.extend(validate_goal_name(sig))

    if not guard:
        errors.append("El plan no tiene 'guard'. Añade al menos 'true' como guard trivial.")

    if not body:
        errors.append("El plan no tiene 'body'. Añade al menos un paso.")

    for step in body:
        if not isinstance(step, str):
            errors.append(f"Paso inválido (no es string): {step!r}")
            continue
        if step.startswith("!"):
            sg = step[1:].split("(")[0].strip()
            if sg not in known_goals:
                errors.append(f"Sub-goal '{step}' no existe en known_goals: {sorted(known_goals)}")
        elif step.startswith("."):
            action = step[1:].split("(")[0].lower()
            unity_names = {a.lower() for a in PRIMITIVE_ACTIONS}
            if action not in unity_names:
                errors.append(f"Acción '{step}' no está en PRIMITIVE_ACTIONS")
        else:
            errors.append(f"Paso '{step}' debe empezar con '!' (sub-goal) o '.' (acción)")

    return errors


def validate_guards_json(guards_response: dict) -> list[str]:
    """
    Valida la respuesta del Paso 2 del pipeline (facts + guards).
    """
    errors: list[str] = []
    facts = guards_response.get("facts", [])
    guards = guards_response.get("guards", [])
    catalog_names = {k.split("(")[0] for k in BELIEF_CATALOG}

    for fact in facts:
        functor = fact.get("functor", "")
        if functor not in catalog_names:
            errors.append(f"Functor '{functor}' no está en BELIEF_CATALOG")

    bound_vars: set[str] = set()
    for fact in facts:
        for arg in fact.get("args", []):
            if isinstance(arg, str) and arg[0].isupper():
                bound_vars.add(arg)

    for guard in guards:
        expr = guard.get("expr", "")
        # Extraer variables usadas en la expresión
        used_vars = re.findall(r'\b([A-Z][A-Za-z0-9_]*)\b', expr)
        for var in used_vars:
            if var not in bound_vars:
                errors.append(
                    f"Variable '{var}' usada en guard '{expr}' no fue ligada por ningún fact anterior"
                )

    return errors


def validate_parse_goals_json(goals_response: object, n_goals: int | None = None) -> list[str]:
    """Validate parse_goals output shape and success_condition syntax.

    Si se pasa n_goals (número de goals_nl de origen), se valida que cada
    source_index presente esté dentro de rango [0, n_goals). La ausencia de
    source_index se tolera (back-compat: el bootstrap cae al índice posicional).
    """
    errors: list[str] = []
    catalog_names = {k.split("(")[0] for k in BELIEF_CATALOG}

    if not isinstance(goals_response, list) or not goals_response:
        return ["parse_goals response must be a non-empty JSON array"]

    placeholder_tokens = (
        "belief(",
        "condition_met",
        "zone_tag",
        "inventory_alpha",
        "status(flag)",
        "item_alpha",
    )

    for idx, goal in enumerate(goals_response):
        if not isinstance(goal, dict):
            errors.append(f"goals[{idx}] must be an object")
            continue

        sig = goal.get("sig", "")
        if not isinstance(sig, str) or not sig.strip():
            errors.append(f"goals[{idx}].sig must be non-empty string")
        else:
            errors.extend(validate_goal_name(sig.strip()))

        if not isinstance(goal.get("priority"), (int, float)):
            errors.append(f"goals[{idx}].priority must be numeric")

        # source_index (0.B3): opcional pero, si está, debe ser int en rango.
        if "source_index" in goal:
            si = goal.get("source_index")
            if isinstance(si, bool) or not isinstance(si, int):
                errors.append(f"goals[{idx}].source_index must be an integer")
            elif n_goals is not None and not (0 <= si < n_goals):
                errors.append(
                    f"goals[{idx}].source_index {si} out of range [0, {n_goals})"
                )

        reason = goal.get("reason", "")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"goals[{idx}].reason must be non-empty string")

        cond = goal.get("success_condition", "")
        if not isinstance(cond, str) or not cond.strip():
            errors.append(f"goals[{idx}].success_condition must be non-empty string")
            continue

        cond = cond.strip()
        low = cond.lower()
        if any(token in low for token in placeholder_tokens):
            errors.append(
                f"goals[{idx}].success_condition uses placeholder tokens: {cond!r}"
            )

        # Adjacent predicates without ASL conjunction, e.g. "a(x) b(y)"
        if re.search(r"\)\s+[a-z_][a-z0-9_]*\(", cond) and " & " not in cond:
            errors.append(
                f"goals[{idx}].success_condition must join predicates with ' & ': {cond!r}"
            )

        # Validate predicate functors belong to known belief catalog.
        atoms = re.findall(r"([a-z_][a-z0-9_]*)\([^)]*\)", cond)
        for atom in atoms:
            if atom not in catalog_names:
                errors.append(
                    f"goals[{idx}].success_condition uses unknown belief predicate '{atom}'"
                )

    return errors
