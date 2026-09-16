"""Evaluador de guards AgentSpeak usando el interprete nativo (agentspeak lib).

Reemplaza el codigo Python artesanal (regex + dicts) de bdi.py:
  - _split_conjunction        -> parsing nativo del parser agentspeak
  - _eval_guard_with_bindings -> context.execute() del interprete
  - _eval_guard_atom          -> TermQuery / NotQuery / BinaryExpr del runtime
  - _NUM_OPS                  -> evaluacion numerica nativa de agentspeak
  - seed_bindings             -> unificacion del head del plan con el call

Semantica: la unificacion de variables, la evaluacion del guard, y el binding
de valores a nombres de variable los hace el interprete agentspeak, NO Python.

Uso:
    evaluator = GuardEvaluator()
    ok, bindings = evaluator.eval_guard(
        guard       = "not has_item(wheat, N) & item_spawn(wheat, SpawnZone) & zone_center(SpawnZone, ZX, ZY)",
        param_names = ["ItemId", "N"],
        call_args   = ["wheat", 1],
        snapshot    = agent.beliefs.snapshot(),
    )
    # ok=True, bindings={"ItemId": "wheat", "SpawnZone": "farmland", "ZX": 9.0, "ZY": -6.0}
    # (N no se bindea porque esta solo en "not has_item" --- NAF no liga vars)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import agentspeak
import agentspeak.stdlib  # noqa: F401 - garantiza que agentspeak.stdlib.actions sea accesible
from agentspeak import runtime as asp_runtime
from agentspeak.runtime import (
    BuildInstructionsVisitor,
    BuildQueryVisitor,
    BuildTermVisitor,
    Instruction,
    Plan,
    TrueQuery,
    noop,
)

log = logging.getLogger(__name__)

_VAR_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")
_EVAL_SIG = "guard_eval_internal"
_EMPTY_ACTIONS = agentspeak.Actions()


def compile_plan_with_actions(plan_text: str, actions: "agentspeak.Actions") -> "Plan | None":
    """Compila un plan ASL completo (guard + body) usando el registro de acciones dado.

    A diferencia de _compile_guard_plan, esta función compila el body usando las
    acciones reales (p.ej. .moveto, .pickup) en lugar de _EMPTY_ACTIONS.
    Esto es necesario para planes ejecutables en agentspeak.runtime.Agent.

    Returns:
        El Plan compilado, o None si hubo un error de parseo/compilación.
    """
    from agentspeak import lexer as _lexer, parser as _parser

    _log = agentspeak.Log(logging.getLogger("agentspeak"))
    try:
        tokens = list(_lexer.tokenize(
            agentspeak.StringSource("<plan>", plan_text), _log, 1
        ))
    except Exception as exc:
        log.debug("[compile_plan] tokenize error: %s  plan=%s", exc, plan_text[:80])
        return None

    if not tokens:
        return None

    first_tok = tokens.pop(0)
    try:
        _tok, ast_plan = _parser.parse_plan(first_tok, iter(tokens), _log)
    except Exception as exc:
        log.debug("[compile_plan] parse_plan error: %s  plan=%s", exc, plan_text[:80])
        return None

    variables: dict = {}
    try:
        head = ast_plan.event.head.accept(BuildTermVisitor(variables))
        context: Any = (
            ast_plan.context.accept(BuildQueryVisitor(variables, actions, _log))
            if ast_plan.context
            else TrueQuery()
        )
        body_instr = Instruction(noop)
        if ast_plan.body:
            ast_plan.body.accept(
                BuildInstructionsVisitor(variables, actions, body_instr, _log)
            )
        return Plan(
            ast_plan.event.trigger,
            ast_plan.event.goal_type,
            head,
            context,
            body_instr,
            ast_plan.body,
            ast_plan.annotation,
        )
    except Exception as exc:
        log.debug("[compile_plan] compile error: %s  plan=%s", exc, plan_text[:80])
        return None


def _to_asp_val(v: Any) -> Any:
    if isinstance(v, str):
        return agentspeak.Literal(v, ())
    return v


def _from_asp_val(v: Any) -> Any:
    if isinstance(v, agentspeak.Literal) and not v.args:
        return v.functor
    if isinstance(v, (int, float, str, bool)):
        return v
    return v


def _extract_uppercase_vars(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _VAR_RE.finditer(text):
        seen[m.group(1)] = None
    return list(seen.keys())


def _compile_guard_plan(plan_text: str) -> tuple["Plan | None", dict[str, Any]]:
    """Compila un plan ASL y devuelve (Plan, variables_dict).

    variables_dict mapea {nombre_var_str: Var_obj} que se usa con
    agentspeak.deref(var_obj, scope) para leer bindings tras call().
    """
    from agentspeak import lexer as _lexer, parser as _parser

    _log = agentspeak.Log(logging.getLogger("agentspeak"))
    try:
        tokens = list(_lexer.tokenize(
            agentspeak.StringSource("<guard_plan>", plan_text), _log, 1
        ))
    except Exception as exc:
        log.debug("[GuardEval] tokenize error: %s  plan: %s", exc, plan_text)
        return None, {}

    if not tokens:
        return None, {}

    first_tok = tokens.pop(0)
    try:
        _tok, ast_plan = _parser.parse_plan(first_tok, iter(tokens), _log)
    except Exception as exc:
        log.debug("[GuardEval] parse_plan error: %s  plan: %s", exc, plan_text)
        return None, {}

    variables: dict[str, Any] = {}
    try:
        head = ast_plan.event.head.accept(BuildTermVisitor(variables))
        context: Any = (
            ast_plan.context.accept(BuildQueryVisitor(variables, _EMPTY_ACTIONS, _log))
            if ast_plan.context
            else TrueQuery()
        )
        body_instr = Instruction(noop)
        body_instr.f = noop
        if ast_plan.body:
            ast_plan.body.accept(
                BuildInstructionsVisitor(variables, _EMPTY_ACTIONS, body_instr, _log)
            )
        plan = Plan(
            ast_plan.event.trigger,
            ast_plan.event.goal_type,
            head,
            context,
            body_instr,
            ast_plan.body,
            ast_plan.annotation,
        )
    except Exception as exc:
        log.debug("[GuardEval] plan compilation error: %s  plan: %s", exc, plan_text)
        return None, {}

    return plan, variables


class GuardEvaluator:
    """Evalua guards AgentSpeak usando el interprete nativo (agentspeak.runtime).

    Para cada llamada a eval_guard:
      1. Crea un agente agentspeak temporal con las beliefs del NPC
      2. Compila el plan: +!guard_eval_internal(P1,P2,...) : <guard> <- true.
         preservando el dict variables {nombre_var: Var_obj}
      3. Llama al goal con los args reales -> agentspeak unifica P1=wheat, etc.
      4. Si call() tiene exito, el guard se cumplio; las vars estan en scope
      5. Lee scope usando variables dict: agentspeak.deref(Var_obj, scope)
         -> eso es lo que hacia el interprete, no Python con regex
    """

    def __init__(self) -> None:
        self._env = asp_runtime.Environment()

    def eval_guard(
        self,
        guard: str,
        param_names: list[str],
        call_args: list[Any],
        snapshot: dict[str, list[tuple]],
    ) -> tuple[bool, dict[str, Any]]:
        """Evalua el guard de una variante de plan usando agentspeak real.

        Args:
            guard:        String del guard ASL.
            param_names:  Parametros formales del plan, e.g. ["ItemId", "N"].
            call_args:    Argumentos del callsite,    e.g. ["wheat", 1].
            snapshot:     Beliefs actuales: dict pred -> [(arg1, arg2, ...)]

        Returns:
            (True,  bindings)  si el guard se satisface.
            (False, {})        si el guard no se satisface o hay error.
        """
        stripped = guard.strip() if guard else ""
        if not stripped or stripped == "true":
            return True, {p: v for p, v in zip(param_names, call_args)}
        try:
            return self._run_eval(stripped, param_names, call_args, snapshot)
        except Exception as exc:
            log.debug("[GuardEval] error inesperado: %s", exc, exc_info=True)
            return False, {}

    def _run_eval(
        self,
        guard: str,
        param_names: list[str],
        call_args: list[Any],
        snapshot: dict[str, list[tuple]],
    ) -> tuple[bool, dict[str, Any]]:
        # 1. Agente temporal vacio
        asp_agent = asp_runtime.Agent(self._env, f"geval{id(self)}")

        # 2. Inyectar beliefs
        for pred, tuples in snapshot.items():
            for tup in tuples:
                args = tuple(_to_asp_val(v) for v in tup)
                term = agentspeak.Literal(pred, args, frozenset())
                asp_agent.beliefs[(pred, len(args))].add(term)

        # 3. Todas las variables (params primero, luego extras del guard)
        param_set = set(param_names)
        guard_vars = _extract_uppercase_vars(guard)
        all_vars: list[str] = list(param_names) + [
            v for v in guard_vars if v not in param_set
        ]

        # 4. Plan ASL con body trivial (true no requiere grounding de vars)
        if all_vars:
            head_args_str = ", ".join(all_vars)
            plan_text = f"+!{_EVAL_SIG}({head_args_str}) : {guard} <- true."
        else:
            plan_text = f"+!{_EVAL_SIG} : {guard} <- true."

        # 5. Compilar plan -> obtener variables dict
        plan, variables = _compile_guard_plan(plan_text)
        if plan is None:
            return False, {}
        asp_agent.add_plan(plan)

        # 6. Preparar call: param_names pre-bound, resto wildcard
        param_map = dict(zip(param_names, call_args))
        if all_vars:
            call_args_asp = tuple(
                _to_asp_val(param_map[v]) if v in param_map else agentspeak.Var()
                for v in all_vars
            )
            call_term = agentspeak.Literal(_EVAL_SIG, call_args_asp)
        else:
            call_term = agentspeak.Literal(_EVAL_SIG, ())

        # 7. Llamar al goal -> agentspeak evalua guard y liga variables
        calling_int = asp_runtime.Intention()
        try:
            ok = asp_agent.call(
                agentspeak.Trigger.addition,
                agentspeak.GoalType.achievement,
                call_term,
                calling_int,
            )
        except agentspeak.AslError:
            return False, {}

        if not ok:
            return False, {}

        # 8. Leer bindings del scope usando variables dict
        if not asp_agent.intentions:
            return False, {}
        intention_stack = asp_agent.intentions[-1]
        if not intention_stack:
            return False, {}
        intention = intention_stack[-1]
        scope = intention.scope

        bindings: dict[str, Any] = {}
        for var_name, var_obj in variables.items():
            val = agentspeak.deref(var_obj, scope)
            if not isinstance(val, agentspeak.Var):
                bindings[var_name] = _from_asp_val(val)

        return True, bindings
