from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeStatus(str, Enum):
    PENDING      = "pending"
    GENERATING   = "generating"
    READY        = "ready"
    FAILED       = "failed"
    NEEDS_REPLAN = "needs_replan"   # plan ejecutado pero success_condition no cumplida


@dataclass
class PlanVariant:
    """Una variante de un plan: un guard + lista de pasos ASL."""
    guard: str            # condición ASL evaluada por el runtime BDI
    steps: list[str]      # ["!achieve_craft_bread", ".moveto(X, Y)"]
    full_asl: str = ""    # texto ASL compilado completo de esta variante
    # Fase 6.5: autoría del cuerpo de la variante — "LLM" (lo generó el modelo)
    # o "CODE" (andamiaje: .done, ramas de agotamiento, guards deterministas).
    source: str = "LLM"


@dataclass
class GoalNode:
    """Nodo del grafo de planes. Representa un goal y sus variantes de plan."""
    sig: str
    variants: list[PlanVariant] = field(default_factory=list)
    is_primitive: bool = False   # acción Unity directa — no descomponer
    is_builtin: bool = False     # viene de plans/builtin/
    from_memory: bool = False    # Fase 4: cargado de plan memory (approved) al arranque
    param_names: list[str] = field(default_factory=list)  # parámetros formales del plan, e.g. ["ItemId", "N"]
    # Fase 6.5: binding concreto de este plan (e.g. ["bread", 1]). Bajo reuso
    # canónico el `sig` es de FAMILIA (achieve_has_item) y se comparte entre items;
    # call_args distingue el binding para no reusar el plan de un item con otro.
    call_args: list = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    success_count: int = 0
    failure_count: int = 0
    failure_history: list[str] = field(default_factory=list)
    description: str = ""        # npc_statement del Paso 1 del pipeline


@dataclass
class DependencyEdge:
    """Arista en el grafo de dependencias entre goals."""
    via_guard: str    # variante del padre que crea esta dependencia
    step_index: int   # posición en los steps de esa variante
    optional: bool    # True si otras variantes del padre no necesitan este hijo
