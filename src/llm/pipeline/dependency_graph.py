from __future__ import annotations

"""
dependency_graph.py — derivacion determinista de variantes desde contratos semanticos.

Iter 5: modulo nuevo, sin conexion con el pipeline runner todavia.
Iter 7: exhaustion branches explicitas para beliefs observacionales.

Dado un chain de DependencyNode (beliefs ordenados por dependencia), deriva
las variantes positiva/negativa para cada belief adquirible de forma explicita
y trazable.

Terminologia:
  required  — belief que siempre se conoce (del perfil/mundo); nunca se niega.
  provided  — belief producido de forma garantizada por una accion.
  observed  — belief observacional; puede aparecer o no tras la accion.

El goal es el ultimo nodo de la cadena; siempre negado en variantes no-done.

Ejemplo canonico (trigo):
  chain = [item_spawn(wheat,Z), zone_center(Z,ZX,ZY), item_at(wheat,WX,WY), has_item(wheat,N)]
  derive_variants(chain) sin contracts -> 3 variantes (Iter 5, backward compat):
    not has_item(wheat,N) & item_spawn(wheat,Z) & not zone_center(Z,ZX,ZY)
    not has_item(wheat,N) & item_spawn(wheat,Z) & zone_center(Z,ZX,ZY) & not item_at(wheat,WX,WY)
    not has_item(wheat,N) & item_spawn(wheat,Z) & zone_center(Z,ZX,ZY) & item_at(wheat,WX,WY)

  derive_variants(chain, contracts=CONTRACT_REGISTRY) con contracts -> 5 variantes (Iter 7):
    las 3 anteriores + 2 exhaustion branches:
    ... & not zone_center(Z,ZX,ZY) & exhausted(ExploreArea, 2)
    ... & zone_center(Z,ZX,ZY) & not item_at(wheat,WX,WY) & exhausted(Search, 3)
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protocol.action_semantics import ContractRegistry


# ---------------------------------------------------------------------------
# DependencyNode
# ---------------------------------------------------------------------------

@dataclass
class DependencyNode:
    """Nodo de la cadena de dependencias.

    Atributos:
        functor:  nombre del predicado en lowercase (ej. "zone_center").
        args:     argumentos en orden (ej. ["Z", "ZX", "ZY"]).
        state:    "required" | "provided" | "observed"
                  required  -> siempre conocido; nunca se niega.
                  provided  -> producido con certeza por una accion.
                  observed  -> observacional; puede o no aparecer.
        producer: nombre de la accion que produce/observa este belief (PascalCase).
                  Vacio si el belief viene del perfil o del mundo directamente.
    """

    functor: str
    args: list[str] = field(default_factory=list)
    state: str = "required"
    producer: str = ""

    VALID_STATES: frozenset[str] = field(
        default_factory=lambda: frozenset({"required", "provided", "observed"}),
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if not self.functor:
            raise ValueError("DependencyNode.functor no puede ser vacio")
        self.functor = self.functor.lower()
        if self.state not in ("required", "provided", "observed"):
            raise ValueError(
                f"DependencyNode.state invalido: '{self.state}'. "
                "Valores permitidos: required, provided, observed"
            )

    def belief_str(self) -> str:
        """Representacion canonica del belief: functor(arg1, arg2, ...)."""
        if self.args:
            return f"{self.functor}({', '.join(self.args)})"
        return self.functor


# ---------------------------------------------------------------------------
# Variant
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    """Una variante derivada del grafo de dependencias.

    Atributos:
        parts:              lista de literales del guard.
        guard:              conjuncion ASL: " & ".join(parts).
        negated_belief:     belief negado en esta variante (vacio en is_final).
        producer_action:    accion que resuelve el belief negado.
        is_final:           True si todos los pre-goal beliefs son positivos.
        attempt_budget:     (Iter 7) max intentos para la accion observacional.
        on_exhaustion:      (Iter 7) politica al agotar el presupuesto.
                            Valores: "goal_failed" | "replan_goal" | "degrade_goal".
        is_exhaustion_branch: (Iter 7) True si esta variante representa el agotamiento
                              del presupuesto sin haber observado el belief.
    """

    parts: list[str]
    guard: str
    negated_belief: str
    producer_action: str
    is_final: bool = False
    # Iter 7: metadatos de presupuesto observacional
    attempt_budget: int = 1
    on_exhaustion: str = "goal_failed"
    is_exhaustion_branch: bool = False

    @classmethod
    def build(
        cls,
        parts: list[str],
        negated_belief: str = "",
        producer_action: str = "",
        is_final: bool = False,
        attempt_budget: int = 1,
        on_exhaustion: str = "goal_failed",
        is_exhaustion_branch: bool = False,
    ) -> "Variant":
        return cls(
            parts=parts,
            guard=" & ".join(parts),
            negated_belief=negated_belief,
            producer_action=producer_action,
            is_final=is_final,
            attempt_budget=attempt_budget,
            on_exhaustion=on_exhaustion,
            is_exhaustion_branch=is_exhaustion_branch,
        )


# ---------------------------------------------------------------------------
# parse_step_breakdown
# ---------------------------------------------------------------------------

def parse_step_breakdown(breakdown: list[dict]) -> list[DependencyNode]:
    """Convierte la lista de pasos anotados del LLM en DependencyNodes.

    Cada elemento de 'breakdown' debe tener al menos 'functor'.
    Campos opcionales: 'args' (list[str]), 'state' (str), 'producer' (str).
    """
    nodes: list[DependencyNode] = []
    for step in breakdown:
        if not isinstance(step, dict):
            continue
        functor = str(step.get("functor", "")).strip().lower()
        if not functor:
            continue
        raw_args = step.get("args", [])
        args = [str(a).strip() for a in raw_args] if isinstance(raw_args, list) else []
        raw_state = str(step.get("state", "required")).strip()
        state = raw_state if raw_state in ("required", "provided", "observed") else "required"
        producer = str(step.get("producer", "")).strip()
        nodes.append(DependencyNode(functor=functor, args=args, state=state, producer=producer))
    return nodes


# ---------------------------------------------------------------------------
# _parse_belief_string
# ---------------------------------------------------------------------------

_BELIEF_RE = re.compile(r'^(\w+)\s*\(([^)]*)\)$')


def _parse_belief_string(s: str, state: str = "provided", producer: str = "") -> DependencyNode:
    """Parsea una cadena como 'has_item(wheat, N)' en un DependencyNode."""
    s = s.strip()
    m = _BELIEF_RE.match(s)
    if m:
        functor = m.group(1).lower()
        args = [a.strip() for a in m.group(2).split(',') if a.strip()]
    else:
        functor = s.lower()
        args = []
    return DependencyNode(functor=functor, args=args, state=state, producer=producer)


# ---------------------------------------------------------------------------
# build_dependency_graph
# ---------------------------------------------------------------------------

def build_dependency_graph(
    success_condition: str,
    breakdown: list[DependencyNode],
    contracts: "ContractRegistry | None" = None,
) -> list[DependencyNode]:
    """Construye la cadena ordenada de dependencias.

    Retorna: list[DependencyNode] donde el ultimo elemento es el goal.
    Valida producers contra ContractRegistry si se pasa uno.

    Raises:
        ValueError: si algun producer no esta en el registry.
    """
    if contracts is not None:
        bad = [
            n.producer
            for n in breakdown
            if n.producer and contracts.get(n.producer) is None
        ]
        if bad:
            raise ValueError(
                "Producers no registrados en ContractRegistry: "
                + ", ".join(sorted(set(bad)))
            )

    goal_node = _parse_belief_string(success_condition, state="provided", producer="")
    return list(breakdown) + [goal_node]


# ---------------------------------------------------------------------------
# derive_variants
# ---------------------------------------------------------------------------

def derive_variants(
    chain: list[DependencyNode],
    contracts: "ContractRegistry | None" = None,
) -> list[Variant]:
    """Deriva variantes positiva/negativa desde la cadena de dependencias.

    El ultimo nodo del chain es el goal (siempre negado en las variantes).

    Comportamiento sin contracts (Iter 5, backward compat):
      Para cada belief adquirible en posicion k:
        - Todos los beliefs anteriores a k: positivos.
        - Belief k: negado.
        - Goal: negado.
      Variante final (is_final=True):
        - Todos los beliefs no-goal: positivos.
        - Goal: negado.

    Comportamiento con contracts (Iter 7):
      Ademas de las variantes anteriores, para cada belief observacional
      (state="observed") emite una variante de agotamiento:
        - Misma guard que la variante normal + "exhausted(Producer, budget)".
        - is_exhaustion_branch=True.
        - attempt_budget y on_exhaustion tomados del contrato del producer.
      Las variantes de agotamiento se insertan justo despues de la variante
      normal correspondiente para mantener el orden logico.

    La variante normal de un belief observacional recibe attempt_budget y
    on_exhaustion del contrato si contracts esta disponible.
    """
    if not chain:
        return []

    goal = chain[-1]
    pre_chain = chain[:-1]
    goal_neg = f"not {goal.belief_str()}"

    variants: list[Variant] = []

    acquirable_indices = [
        i for i, n in enumerate(pre_chain) if n.state != "required"
    ]

    for idx in acquirable_indices:
        node = pre_chain[idx]
        before_parts = [n.belief_str() for n in pre_chain[:idx]]
        current_neg = f"not {node.belief_str()}"
        parts = [goal_neg] + before_parts + [current_neg]

        # Metadatos del contrato del producer (si disponibles)
        budget = 1
        exhaustion_policy = "goal_failed"
        if contracts is not None and node.producer:
            contract = contracts.get(node.producer)
            if contract is not None:
                budget = contract.attempt_budget
                exhaustion_policy = contract.on_exhaustion

        # Variante normal
        variants.append(Variant.build(
            parts=parts,
            negated_belief=node.belief_str(),
            producer_action=node.producer,
            is_final=False,
            attempt_budget=budget,
            on_exhaustion=exhaustion_policy,
            is_exhaustion_branch=False,
        ))

        # Variante de agotamiento (solo si contracts disponible y belief observacional)
        if contracts is not None and node.state == "observed":
            exhaustion_literal = f"exhausted({node.producer}, {budget})"
            exh_parts = [goal_neg] + before_parts + [current_neg, exhaustion_literal]
            variants.append(Variant.build(
                parts=exh_parts,
                negated_belief=node.belief_str(),
                producer_action=node.producer,
                is_final=False,
                attempt_budget=budget,
                on_exhaustion=exhaustion_policy,
                is_exhaustion_branch=True,
            ))

    # Variante final: todos los pre-goal positivos
    final_parts = [goal_neg] + [n.belief_str() for n in pre_chain]
    variants.append(Variant.build(
        parts=final_parts,
        negated_belief="",
        producer_action=goal.producer,
        is_final=True,
    ))

    return variants


# ---------------------------------------------------------------------------
# exhaustion_policy_asl  (Iter 7)
# ---------------------------------------------------------------------------

def exhaustion_policy_asl(goal_sig: str, variant: Variant) -> str:
    """Genera el fragmento ASL para una variante de agotamiento de presupuesto.

    Politicas implementadas:
      goal_failed   -> body: .fail  (fallo explicito del goal)
      replan_goal   -> stub, body: .print("replan_goal: not implemented")
      degrade_goal  -> stub, body: .print("degrade_goal: not implemented")

    El guard de la variante de agotamiento ya incluye exhausted(Producer, budget),
    por lo que es distinto del guard de la variante normal.
    """
    guard = variant.guard
    policy = variant.on_exhaustion

    if policy == "goal_failed":
        body = ".fail"
    elif policy == "replan_goal":
        # Stub: politica de replanificacion no implementada en Iter 7.
        body = f'.print("replan_goal stub: {goal_sig} exhausted {variant.producer_action}")'
    elif policy == "degrade_goal":
        # Stub: politica de degradacion no implementada en Iter 7.
        body = f'.print("degrade_goal stub: {goal_sig} exhausted {variant.producer_action}")'
    else:
        body = ".fail"

    return f"+!{goal_sig} : {guard} <- {body}."
