from __future__ import annotations

"""plan_simulator.py - Simulador determinista de efectos de plan sobre creencias."""

import re
from dataclasses import dataclass, field
from typing import Any

from protocol.action_semantics import CONTRACT_REGISTRY

# Contratos mínimos para sub-goals builtin: mapea sig -> lista de predicados
# que garantiza al terminar, usando los nombres de parámetro en posición.
# Formato: (functor, [pos_arg_index_or_literal, ...])
_BUILTIN_SUBGOAL_GUARANTEES: dict[str, list[tuple[str, list]]] = {
    "move_to_and_pickup": [("has_item", [0, 1])],   # has_item(ItemId, N)
    "achieve_explore_zone": [("knows_zone", [0])],   # knows_zone(ZoneTag)
    # craft_item(RecipeId, Qty): guarantees has_item for the recipe output.
    # ItemToCraft is opaque without belief data, so we add a wildcard entry.
    "craft_item": [],  # no static guarantee — success inferred at runtime
}


# Fase 17: macros de coordinación (solo con coordinación y sub-planes activos).
_COORDINATION_SUBGOAL_GUARANTEES: dict[str, list[tuple[str, list]]] = {
    "obtain_from_peer": [("has_item", [1, 2])],   # obtain_from_peer(Peer, ItemId, Qty)
    "collect_from_peer": [("has_item", [1, 2])],  # collect_from_peer(Peer, ItemId, Qty)
}


def active_subgoal_guarantees(
    builtin_subplans: bool = True, coordination: bool = False,
) -> dict[str, list[tuple[str, list]]]:
    """Garantías de sub-goals builtin activas (Fase 16: sin los ablacionados).

    Sin move_to_and_pickup/craft_item, un sub-goal con ese nombre sería del LLM
    y el simulador no debe acreditarle una garantía que ya no existe.
    """
    if builtin_subplans:
        if coordination:
            return {**_BUILTIN_SUBGOAL_GUARANTEES, **_COORDINATION_SUBGOAL_GUARANTEES}
        return _BUILTIN_SUBGOAL_GUARANTEES
    from llm.pipeline.builtins import ABLATABLE_SUBPLANS
    return {
        sig: spec for sig, spec in _BUILTIN_SUBGOAL_GUARANTEES.items()
        if sig not in ABLATABLE_SUBPLANS
    }


Beliefs = dict[str, set[tuple[str, ...]]]


@dataclass
class PlanReachability:
    """Resultado del chequeo de alcanzabilidad de un plan."""
    reachable: bool
    simulated_beliefs: Beliefs
    missing_predicates: list[str] = field(default_factory=list)
    hint: str = ""


def _bind_args(spec_args: list[str], action_args: dict[str, Any]) -> tuple[str, ...]:
    """
    Liga los placeholders de un BeliefSpec con los args concretos del step.
    Args que empiezan por mayuscula son variables; en minuscula son constantes.
    Variables sin binding → se dejan como '?'.
    """
    result: list[str] = []
    for sa in spec_args:
        if sa and sa[0].isupper():
            found = False
            for k, v in action_args.items():
                if k.lower() == sa.lower():
                    result.append(str(v))
                    found = True
                    break
            if not found:
                result.append("?")
        else:
            result.append(sa)
    return tuple(result)


def _matches_invalidate(
    stored: tuple[str, ...],
    spec_args: list[str],
    action_args: dict[str, Any],
) -> bool:
    """True si la tupla almacenada casa con la spec de invalidacion (wildcards uppercase)."""
    if len(stored) != len(spec_args):
        return False
    for stored_val, sa in zip(stored, spec_args):
        if sa and sa[0].isupper():
            continue  # wildcard
        bound = None
        for k, v in action_args.items():
            if k.lower() == sa.lower():
                bound = str(v)
                break
        compare_with = bound if bound is not None else sa
        if stored_val != compare_with:
            return False
    return True


def simulate_plan(
    steps: list[dict[str, Any]],
    initial_beliefs: Beliefs,
    *,
    contracts=None,
    subgoal_guarantees: dict[str, list[tuple[str, list]]] | None = None,
) -> Beliefs:
    """
    Simula los efectos de un plan sobre las creencias iniciales.
    Aplica guarantees_on_success e invalidates de cada contrato.
    may_observe se ignora (no garantizado en simulacion).
    """
    if contracts is None:
        contracts = CONTRACT_REGISTRY
    guarantees = _BUILTIN_SUBGOAL_GUARANTEES if subgoal_guarantees is None else subgoal_guarantees

    # El snapshot real de BeliefStore mapea cada functor a una LISTA de tuplas,
    # pero el simulador opera con sets (usa .add y comprehensions). Normalizar a
    # set en la entrada evita "'list' object has no attribute 'add'" cuando un
    # functor garantizado (p.ej. knows_zone) ya existe en las beliefs iniciales.
    # Cada fila se coacciona a tupla (hashable) — vale para tuplas y para listas.
    beliefs: Beliefs = {
        k: {tuple(row) for row in rows}
        for k, rows in initial_beliefs.items()
    }

    for step in steps:
        step_type = step.get("type")

        if step_type == "subgoal":
            subgoal_name = step.get("name", "")
            raw_args = step.get("args", [])
            args_list = list(raw_args) if isinstance(raw_args, list) else []
            for functor, param_positions in guarantees.get(subgoal_name, []):
                bound: list[str] = []
                for p in param_positions:
                    if isinstance(p, int):
                        bound.append(str(args_list[p]) if p < len(args_list) else "?")
                    else:
                        bound.append(str(p))
                beliefs.setdefault(functor, set()).add(tuple(bound))
            continue

        if step_type != "action":
            continue

        action_name = step.get("name", "")
        raw_args = step.get("args", [])

        if isinstance(raw_args, dict):
            # Copia: los alias de abajo no deben escribirse en el step del LLM.
            action_args: dict[str, Any] = dict(raw_args)
        else:
            from protocol.action_contract import ACTION_ALLOWLIST
            asl_keys = ACTION_ALLOWLIST.get(action_name, {}).get("asl_args", [])
            action_args = {k: v for k, v in zip(asl_keys, raw_args)}

        # Craft guarantee uses "ItemToCraft"/"N" but asl_args are "itemId"/"qty".
        # Add aliases so _bind_args can resolve the semantic variable names.
        if action_name == "Craft":
            it = action_args.get("itemId")
            if it is not None:
                action_args["ItemToCraft"] = str(it)
            n = action_args.get("qty")
            if n is not None:
                action_args["N"] = str(n)

        # PickUp garantiza has_item(ItemId, N) pero no lleva cantidad: N = unidades
        # acumuladas del item en la simulación (inventario inicial + PickUps previos).
        # Sin esto N quedaba '?' y ningún plan con PickUp primitivo alcanzaba
        # has_item(item, 1): el repair de completitud descartaba todo PickUp que
        # añadía el LLM (sesgo contra ATOM, tanda 2 de la Fase 16).
        if action_name == "PickUp":
            it = action_args.get("itemId")
            if it is not None:
                held = [
                    int(t[1]) for t in beliefs.get("has_item", set())
                    if len(t) == 2 and t[0] == str(it) and str(t[1]).isdigit()
                ]
                action_args["N"] = str((max(held) if held else 0) + 1)

        contract = contracts.get(action_name)
        if contract is None:
            continue

        # Aplicar invalidates
        for spec in contract.invalidates:
            if spec.functor not in beliefs:
                continue
            if not spec.args:
                beliefs.pop(spec.functor, None)
            else:
                beliefs[spec.functor] = {
                    t for t in beliefs[spec.functor]
                    if not _matches_invalidate(t, spec.args, action_args)
                }
                if not beliefs[spec.functor]:
                    del beliefs[spec.functor]

        # Aplicar guarantees_on_success
        for spec in contract.guarantees_on_success:
            bound = _bind_args(spec.args, action_args)
            if spec.functor not in beliefs:
                beliefs[spec.functor] = set()
            beliefs[spec.functor].add(bound)

    return beliefs


_PRED_RE = re.compile(r"(?P<functor>[a-z_][a-z0-9_]*)\((?P<args>[^)]*)\)")
_NUM_GUARD_RE = re.compile(
    r"(?P<var>[A-Z][A-Za-z0-9_]*)\s*(?P<op>>=|<=|>|<|==)\s*(?P<val>\d+(?:\.\d+)?)"
)


def _parse_condition(condition: str) -> list[str]:
    return [c.strip() for c in condition.split("&") if c.strip()]


def _eval_clause(clause: str, beliefs: Beliefs) -> tuple[bool, str | None]:
    """Evalua una clausula ASL simple contra las creencias."""
    if _NUM_GUARD_RE.fullmatch(clause):
        return False, clause  # guard sin contexto de functor

    pm = _PRED_RE.fullmatch(clause)
    if not pm:
        return False, clause

    functor = pm.group("functor")
    raw_args = [a.strip() for a in pm.group("args").split(",") if a.strip()]

    if functor not in beliefs:
        return False, clause

    stored_tuples = beliefs[functor]

    if not raw_args:
        return bool(stored_tuples), None if stored_tuples else clause

    for tup in stored_tuples:
        if len(tup) != len(raw_args):
            continue
        match = True
        for stored_val, ca in zip(tup, raw_args):
            if ca and ca[0].isupper():
                continue  # variable existencial
            if stored_val != ca:
                match = False
                break
        if match:
            return True, None

    return False, clause


def verify_success_condition(beliefs: Beliefs, condition: str | None) -> bool:
    """True si condition (o todas sus conjunciones) se cumplen en beliefs."""
    if not condition:
        return True
    for clause in _parse_condition(condition):
        ok, _ = _eval_clause(clause, beliefs)
        if not ok:
            return False
    return True


def _missing_predicates(beliefs: Beliefs, condition: str) -> list[str]:
    return [
        desc for clause in _parse_condition(condition)
        for ok, desc in [_eval_clause(clause, beliefs)]
        if not ok and desc
    ]


def check_plan_reachability(
    steps: list[dict[str, Any]],
    initial_beliefs: Beliefs,
    success_condition: str | None,
    *,
    contracts=None,
    subgoal_guarantees: dict[str, list[tuple[str, list]]] | None = None,
) -> PlanReachability:
    """
    Comprueba si el plan alcanza success_condition partiendo de initial_beliefs.

    Devuelve PlanReachability con reachable, simulated_beliefs, missing_predicates, hint.
    """
    sim_beliefs = simulate_plan(
        steps, initial_beliefs, contracts=contracts, subgoal_guarantees=subgoal_guarantees,
    )

    if not success_condition:
        return PlanReachability(reachable=True, simulated_beliefs=sim_beliefs)

    reachable = verify_success_condition(sim_beliefs, success_condition)
    missing = [] if reachable else _missing_predicates(sim_beliefs, success_condition)

    hint = ""
    if not reachable:
        hint = (
            f"The plan does not produce the goal condition '{success_condition}'. "
            f"Missing: {', '.join(missing)}. "
            "Add actions whose guarantees_on_success cover the missing predicates "
            "(e.g. PickUp guarantees has_item; Craft guarantees has_item; "
            "MoveTo guarantees current_position)."
        )

    return PlanReachability(
        reachable=reachable,
        simulated_beliefs=sim_beliefs,
        missing_predicates=missing,
        hint=hint,
    )
