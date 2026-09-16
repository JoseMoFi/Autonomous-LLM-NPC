from __future__ import annotations

"""Paso 4 del pipeline BDI: Clasificación de acciones (sin LLM).

Separa los pasos en primitivos (Unity los ejecuta directamente) y
sub-goals que hay que expandir. Incluye la lógica de encolado con
detección de ciclos en el DAG.
"""

import collections

import networkx as nx  # type: ignore

from protocol.action_contract import PRIMITIVE_ACTIONS
from llm.catalogs import PEER_ACTION_NAMES


def classify_steps(steps: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Clasifica pasos en primitivos y sub-goals a expandir.

    Un step con type=='action' y name en PRIMITIVE_ACTIONS → primitivo.
    Cualquier otro step → sub-goal a expandir (se devuelve el dict completo
    para preservar 'description' y 'replaces_steps' del paso 3).

    Returns:
        (pasos_primitivos, subgoal_dicts_a_expandir)
    """
    primitives: list[dict] = []
    to_expand: list[dict] = []

    for step in steps:
        step_type = step.get("type", "")
        step_name = step.get("name", "")
        # Fase 17: las acciones de coordinación también son primitivas (no se expanden).
        if step_type == "action" and (step_name in PRIMITIVE_ACTIONS or step_name in PEER_ACTION_NAMES):
            primitives.append(step)
        else:
            # Preserve full step dict so callers can access description/replaces_steps
            to_expand.append(step)

    return primitives, to_expand


def enqueue_if_needed(
    child: str,
    parent: str,
    dag: nx.DiGraph,
    plan_library: dict,
    queue: collections.deque,
) -> bool:
    """
    Añade child a la cola de expansión si es necesario.

    Orden de comprobación:
    1. ¿Está en plan_library?    → solo arista DAG, no encolar.
    2. ¿Está en el DAG?          → solo arista DAG (ya pendiente/en proceso).
    3. ¿La arista crearía ciclo? → fallback (retorna False).
    4. En otro caso              → añadir nodo pending + arista + encolar.

    Returns:
        False si se detectó un ciclo; True en cualquier otro caso.
    """
    if not child or child == parent:
        return False

    if child in plan_library:
        dag.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(parent, child)
            return False
        return True

    if child in dag.nodes:
        dag.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(parent, child)
            return False
        return True

    # Goal nuevo
    dag.add_node(child, status="pending")
    dag.add_edge(parent, child)
    if not nx.is_directed_acyclic_graph(dag):
        dag.remove_edge(parent, child)
        dag.remove_node(child)
        return False

    queue.append(child)
    return True
