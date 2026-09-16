from __future__ import annotations

"""canonical.py — canonicalización de condiciones de éxito (Fase 6.5).

Convierte una `success_condition` ground en (esquema, bindings) para que dos
goals con la misma ESTRUCTURA reutilicen la misma familia de plan:

    "has_item(wheat, 2)"  → ("has_item(Item, Qty)", {"Item": "wheat", "Qty": 2})
    "has_item(bread, 1)"  → ("has_item(Item, Qty)", {"Item": "bread", "Qty": 1})

El esquema (clave de FAMILIA) es idéntico para ambos; los bindings son los
parámetros del binding concreto. De aquí se deriva un `sig` de familia estable
(`achieve_has_item`) que NO depende de cómo frasee el LLM el objetivo, lo que
deduplica goals y estabiliza el naming entre sesiones.

Módulo PURO (solo `re`): sin LLM, sin estado global. Es la base de la Fase 6.5
y del árbol por familia de la Fase 10.
"""

import re

# functor en lowercase; opcional prefijo `not `; args entre paréntesis.
_PRED_RE = re.compile(r"^\s*(not\s+)?([a-z_][a-z0-9_]*)\s*\(([^)]*)\)\s*$")
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_VAR_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")


def _is_variable(token: str) -> bool:
    """True si el token ya es una variable ASL (inicial mayúscula), no una constante."""
    return bool(_VAR_RE.match(token))


def _coerce(value: str):
    """Constante string → tipo Python (int/float/str) preservando el valor."""
    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value):
        return float(value)
    return value


def _type_base(value: str, entity_catalog: dict | None) -> str:
    """Nombre base de variable según el tipo de la constante."""
    if _INT_RE.match(value) or _FLOAT_RE.match(value):
        return "Qty"
    cat = entity_catalog or {}
    if value in cat.get("item_ids", ()):
        return "Item"
    if value in cat.get("zone_ids", ()):
        return "Zone"
    if value in cat.get("recipe_ids", ()):
        return "Recipe"
    if value in cat.get("delivery_tags", ()):
        return "Delivery"
    return "Const"


def _split_conjuncts(condition: str) -> list[str]:
    return [c.strip() for c in condition.split("&") if c.strip()]


def _canon_clause(
    clause: str,
    entity_catalog: dict | None,
    value_to_var: dict[str, str],
    base_counts: dict[str, int],
    bindings: dict[str, object],
) -> str:
    """Canonicaliza una cláusula. Las constantes iguales comparten variable
    (captura los joins, p.ej. el mismo `wheat` en recipe y en has_item)."""
    m = _PRED_RE.match(clause)
    if not m:
        # No es un predicado parseable (p.ej. un guard numérico "N >= 2"):
        # se deja verbatim — no aporta a la clave de familia.
        return clause.strip()

    neg = m.group(1) or ""
    functor = m.group(2)
    raw_args = [a.strip() for a in m.group(3).split(",") if a.strip()]

    slots: list[str] = []
    for arg in raw_args:
        if _is_variable(arg):
            slots.append(arg)  # ya es variable libre → se mantiene
            continue
        var = value_to_var.get(arg)
        if var is None:
            base = _type_base(arg, entity_catalog)
            n = base_counts.get(base, 0) + 1
            base_counts[base] = n
            var = base if n == 1 else f"{base}{n}"
            value_to_var[arg] = var
            bindings[var] = _coerce(arg)
        slots.append(var)

    return f"{neg}{functor}({', '.join(slots)})"


def canonical_key(
    condition: str,
    entity_catalog: dict | None = None,
) -> tuple[str, dict[str, object]]:
    """Devuelve (esquema, bindings) de una success_condition.

    El esquema sustituye cada constante por una variable tipada; constantes
    iguales comparten variable. Los bindings mapean cada variable a su valor
    concreto (int/float/str). Soporta conjunciones (`&`).

    >>> canonical_key("has_item(wheat, 2)")
    ('has_item(Item, Qty)', {'Item': 'wheat', 'Qty': 2})
    """
    value_to_var: dict[str, str] = {}
    base_counts: dict[str, int] = {}
    bindings: dict[str, object] = {}
    schema_parts = [
        _canon_clause(clause, entity_catalog, value_to_var, base_counts, bindings)
        for clause in _split_conjuncts(condition)
    ]
    return " & ".join(schema_parts), bindings


def derive_family_sig(condition_or_schema: str) -> str:
    """Deriva el `sig` de familia desde el predicado (NO del NL).

    Usa el functor de la primera cláusula: `has_item(...)` → `achieve_has_item`.
    Determinista y general (sin conjugaciones ad-hoc), de modo que la misma
    estructura siempre da el mismo sig.
    """
    first = _split_conjuncts(condition_or_schema)
    if not first:
        return "achieve_unknown"
    m = _PRED_RE.match(first[0])
    if not m:
        return "achieve_unknown"
    return f"achieve_{m.group(2)}"


def canonicalize_goals(goals: list, entity_catalog: dict | None = None) -> list:
    """Renombra cada goal a su familia canónica y deduplica por (familia, bindings).

    `goals` es una lista de objetos con atributos `sig`, `success_condition` y
    `call_args` (los `Goal` del NPC). Muta cada goal con sig de familia + call_args
    y descarta duplicados exactos (misma familia + mismos bindings). Devuelve la
    lista resultante. Los goals sin success_condition se dejan intactos.

    Nota (Fase 6.5): el cableado runtime completo del reuso (cabezas ASL
    paramétricas, identidad goal por sig+bindings en el ciclo de vida del BDI,
    migración de PlanMemory) requiere validación end-to-end y va tras el flag
    `canonical_reuse_enabled`. Esta función es la parte pura y testeable.
    """
    out: list = []
    seen: set = set()
    for goal in goals:
        sc = getattr(goal, "success_condition", None)
        if sc:
            schema, _ = canonical_key(sc, entity_catalog)
            goal.sig = derive_family_sig(schema)
            goal.call_args = call_args_from_condition(sc)
            # param_names paralelos: fijan la aridad de la cabeza paramétrica para
            # que `+!achieve_has_item(Item, Qty)` case con el goal despachado.
            goal.param_names = param_names_from_condition(sc, entity_catalog)
            key = (goal.sig, tuple(goal.call_args))
            if key in seen:
                continue
            seen.add(key)
        out.append(goal)
    return out


def goal_identity_key(sig: str, call_args: list | None) -> str:
    """Clave de identidad de un goal/plan que incluye el BINDING.

    `achieve_has_item` + `[bread, 1]` → `"achieve_has_item__bread_1"`. Sin
    call_args → el `sig` tal cual. Se usa para indexar el `plan_graph` (y la
    memoria) por (familia, binding) y así permitir que dos goals de la misma
    familia con bindings distintos (bread y wheat) COEXISTAN como nodos separados
    en vez de pisarse bajo el mismo `sig` de familia.

    >>> goal_identity_key("achieve_has_item", ["bread", 1])
    'achieve_has_item__bread_1'
    >>> goal_identity_key("achieve_explore_zone", [])
    'achieve_explore_zone'
    """
    if not call_args:
        return sig
    return sig + "__" + "_".join(str(a) for a in call_args)


def family_capabilities(schema: str, contract_registry) -> list[str]:
    """Acciones cuyo `guarantees_on_success` casa con el functor del esquema.

    Pre-chequeo de capability (T3): para `has_item(Item, Qty)` devuelve las
    acciones que producen `has_item` (PickUp, Craft, …). Si hay alguna, el goal
    es candidato a reuso por familia antes de llamar al LLM.
    """
    parts = _split_conjuncts(schema)
    if not parts:
        return []
    m = _PRED_RE.match(parts[0])
    if not m:
        return []
    functor = m.group(2)
    matches: list[str] = []
    for name in contract_registry.all_names():
        contract = contract_registry.get(name)
        if contract is None:
            continue
        if any(spec.functor == functor for spec in contract.guarantees_on_success):
            matches.append(name)
    return sorted(matches)


def call_args_from_condition(condition: str) -> list:
    """Args ground ordenados de la primera cláusula, para `Goal.call_args`.

    >>> call_args_from_condition("has_item(wheat, 2)")
    ['wheat', 2]

    Las variables libres (inicial mayúscula) se omiten — no son parámetros del
    binding. Devuelve [] si no hay predicado parseable.
    """
    first = _split_conjuncts(condition)
    if not first:
        return []
    m = _PRED_RE.match(first[0])
    if not m:
        return []
    args = [a.strip() for a in m.group(3).split(",") if a.strip()]
    return [_coerce(a) for a in args if not _is_variable(a)]


def param_names_from_condition(
    condition: str, entity_catalog: dict | None = None
) -> list[str]:
    """Nombres de variable tipados PARALELOS a `call_args_from_condition`.

    Para `has_item(wheat, 2)` devuelve `['Item', 'Qty']` (mismas variables que
    usaría `canonical_key`), en el mismo orden y con la misma longitud que los
    `call_args`. Son los `param_names` de la CABEZA del plan paramétrico
    (`+!achieve_has_item(Item, Qty) : ... <- ...`), de modo que la aridad de la
    cabeza case con la del goal despachado (`achieve_has_item(wheat, 2)`).

    >>> param_names_from_condition("has_item(wheat, 2)")
    ['Item', 'Qty']
    """
    first = _split_conjuncts(condition)
    if not first:
        return []
    m = _PRED_RE.match(first[0])
    if not m:
        return []
    args = [a.strip() for a in m.group(3).split(",") if a.strip()]
    value_to_var: dict[str, str] = {}
    base_counts: dict[str, int] = {}
    names: list[str] = []
    for arg in args:
        if _is_variable(arg):
            continue  # paralelo a call_args_from_condition (omite variables libres)
        var = value_to_var.get(arg)
        if var is None:
            base = _type_base(arg, entity_catalog)
            n = base_counts.get(base, 0) + 1
            base_counts[base] = n
            var = base if n == 1 else f"{base}{n}"
            value_to_var[arg] = var
        names.append(var)
    return names
