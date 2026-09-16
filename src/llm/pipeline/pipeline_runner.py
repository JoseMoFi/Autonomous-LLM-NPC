from __future__ import annotations

"""Orquestador del pipeline BDI completo para un único goal.

Diseño v3 — 7 pasos (0-6):
  Paso 0 — NL → sig + descripción detallada
    Paso 1 — sig + descripción → success_model estructurado + variante .done

  !! Inicio bucle — una variante por cada (guard+fact) enlazados por variables:
    Paso 2 — Describe en NL el problema que causa que la condición falle
    Paso 3 — Genera steps para resolver ese problema (NL + catálogo como guía)
    Paso 4 — Mapea/valida los steps al catálogo de acciones disponibles (LLM repair si hay errores)
    Paso 5 — Clasifica steps en primitivos vs sub-goals; encola nuevos sub-planes + grafo DAG
  !! Fin bucle

  Paso 6 — Para cada sub-plan nuevo no diseñado: describe en NL y ejecuta pipeline
            desde paso 1 recursivamente (hasta MAX_EXPAND_DEPTH)

Produce un PipelineResult con:
  - Variantes ASL (done + una por cada condición negada)
    - Success model estructurado (facts + guards por variante)
  - Sub-goals expandidos recursivamente en sub_results
  - Lista de sub-goals a expandir (si recursión está desactivada o llega al límite)
    - Snapshot serializable del DAG de dependencias
"""

import collections
import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import networkx as nx

from llm.pipeline.step0_reasoning import run_step0
from llm.pipeline.step1_name import run_step1, build_negated_guard
from llm.pipeline.step1b_decompose import build_need_variants, VariantSpec as _VariantSpec
from llm.pipeline.step2_guards import run_step2
from llm.pipeline.step3_steps import is_same_goal_sig, run_step3
from llm.pipeline.step3_validator import gather_variant_underconstrained
from llm.pipeline.step4_map import run_step4
from llm.pipeline.step4_classify import classify_steps, enqueue_if_needed
from llm.pipeline.step5_refine import run_step5_refine
from llm.pipeline.plan_simulator import check_plan_reachability, active_subgoal_guarantees
from llm.pipeline.builtins import ABLATABLE_SUBPLANS, active_terminal_subplans
from llm.catalogs import PEER_ACTION_NAMES
from llm.prompts.planning import build_prompt as _build_prompt
from llm.pipeline.capability_contracts import load_contracts, save_contracts, upsert_created_plan_contracts
from llm.pipeline.dependency_graph import exhaustion_policy_asl, Variant as DepVariant
from protocol.action_semantics import CONTRACT_REGISTRY as _ACTION_CONTRACT_REGISTRY
from llm.parser import compile_plan_to_asl
from utils.trace_logger import trace as _trace, plan_transform as _plan_transform

_MAX_EXPAND_DEPTH = 2  # máximo de niveles de recursión en paso 6

log = logging.getLogger(__name__)

# Builtin plans always available to the LLM regardless of what the current goal has generated.
# Seeded into plan_library at pipeline start so they appear in the "Available sub-goals" list.
_BUILTIN_PLANS: dict[str, str] = {
    "move_to_and_pickup": "find where the requested item can be obtained, navigate there, and pick up the required quantity (args: itemId, qty)",
    "craft_item": "navigate to the recipe's crafting zone and craft the specified quantity of the output item (args: recipeId, qty)",
    "achieve_explore_zone": "ensure that the target zone center is known; explore if not yet discovered (args: zoneTag)",
}

# Fase 17: sub-planes macro de coordinación (solo con coordinación y sub-planes activos).
_COORDINATION_PLANS: dict[str, str] = {
    "obtain_from_peer": "ask another NPC that can make the item to produce it, wait for its delivery and pick it up (args: peerId, itemId, qty)",
    "collect_from_peer": "go to where another NPC dropped the item and pick it up (args: peerId, itemId, qty)",
}


# ---------------------------------------------------------------------------
# Tipos de resultado
# ---------------------------------------------------------------------------

@dataclass
class PlanVariant:
    """Una variante ASL del plan (guard + steps + asl compilado)."""
    guard: str        # expresión ASL del guard (ej. "not has_item(item_alpha,1)")
    steps: list[dict]
    asl: str
    facts: list[str] = field(default_factory=list)
    guards: list[str] = field(default_factory=list)
    bound_variables: list[str] = field(default_factory=list)
    # Fase 6.5: autoría del cuerpo — "LLM" o "CODE" (andamiaje: .done, agotamiento).
    source: str = "LLM"

    def to_dict(self) -> dict:
        return {
            "guard": self.guard,
            "steps": self.steps,
            "asl": self.asl,
            "facts": self.facts,
            "guards": self.guards,
            "bound_variables": self.bound_variables,
            "source": self.source,
        }


@dataclass
class SubgoalEntry:
    """Sub-plan nuevo a expandir con su contexto."""
    sig: str
    description: str   # descripción NL heredada del step3 que lo generó

    def to_dict(self) -> dict:
        return {"sig": self.sig, "description": self.description}


@dataclass
class PipelineResult:
    """Resultado completo del pipeline para un goal."""
    sig: str
    description: str
    success_conditions: list[str]
    variants: list[PlanVariant]          # [done_variant, variant_cond1, variant_cond2, ...]
    success_model: list[dict] = field(default_factory=list)
    subgoals_to_expand: list[SubgoalEntry] = field(default_factory=list)
    # Sub-pipelines expandidos recursivamente en paso 6 (sig → PipelineResult)
    sub_results: dict[str, "PipelineResult"] = field(default_factory=dict)
    dag_nodes: list[dict] = field(default_factory=list)
    dag_edges: list[list[str]] = field(default_factory=list)

    # Compatibilidad hacia atrás con código que usa main_asl, facts, guards, steps
    @property
    def main_asl(self) -> str:
        """ASL completo: todas las variantes concatenadas."""
        return "\n".join(v.asl for v in self.variants)

    @property
    def facts(self) -> list[dict]:
        return []

    @property
    def guards(self) -> list[dict]:
        return []

    @property
    def steps(self) -> list[dict]:
        """Steps de la primera variante de ejecución (no .done)."""
        exec_variants = self.variants[1:]
        return exec_variants[0].steps if exec_variants else []

    @property
    def contingency_plans(self) -> list:
        return []

    def to_dict(self) -> dict:
        return {
            "sig": self.sig,
            "description": self.description,
            "success_conditions": self.success_conditions,
            "success_model": self.success_model,
            "variants": [v.to_dict() for v in self.variants],
            "subgoals_to_expand": [s.to_dict() for s in self.subgoals_to_expand],
            "sub_results": {k: v.to_dict() for k, v in self.sub_results.items()},
            "dag": {
                "nodes": self.dag_nodes,
                "edges": self.dag_edges,
            },
            # Compat
            "main_asl": self.main_asl,
        }


# Mantener ContingencyPlan como stub para no romper imports que lo usen
@dataclass
class ContingencyPlan:
    fact: dict
    guard_expression: str
    steps: list[dict]
    asl: str

    def to_dict(self) -> dict:
        return {"fact": self.fact, "guard_expression": self.guard_expression,
                "steps": self.steps, "asl": self.asl}


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

async def run_full_pipeline(
    goal_sig: str,
    npc_statement: str,
    existing_goals: list[str],
    llm_call: Callable[[str, str], Awaitable[str]],
    *,
    description: str | None = None,
    # Si se pasa, step1 se omite y se usa directamente esta condición
    success_condition: str | None = None,
    # Compat: ignorado pero aceptado para no romper callers existentes
    use_reasoning: bool = True,
    use_refinement: bool = False,
    capability_contracts_path: str | None = None,
    entity_catalog: dict | None = None,
    beliefs: dict | None = None,
    # Datos crudos de recipe del NPCProfile para step1b (escalera de variantes)
    recipes: list[dict] | None = None,
    item_spawns: list[dict] | None = None,
    # T6: pista de replanificacion cuando el BDI re-solicita tras belief no cumplida
    replan_hint: str | None = None,
    # Fase 16 — ablación: False = sin los sub-planes macro move_to_and_pickup /
    # craft_item (ni en la lista de sub-goals, ni en prompts, ni en repairs).
    builtin_subplans: bool = True,
    # Fase 17 — coordinación planificada por el LLM: acciones de peer en el
    # prompt, peldaños de step1b para lo que solo puede dar otro NPC, y
    # `peers` = [(npc_id, role), ...] conocidos.
    coordination: bool = False,
    peers: list | None = None,
    # Control de recursión (paso 6)
    _depth: int = 0,
    _dag: nx.DiGraph | None = None,
    _known_plans: dict[str, str] | None = None,
    _capability_contracts: dict | None = None,
    _active_stack: tuple[str, ...] = (),
) -> PipelineResult:
    """
    Ejecuta el pipeline completo 7 pasos (0-6) para un goal.

    Paso 0 — NL → sig + descripción detallada  (omitido si description ya se pasa)
    Paso 1 — sig + desc → success_model estructurado + variante .done
    Bucle por cada variante (facts+guards) que puede fallar:
      Paso 2 — NL: qué causa que la condición no esté satisfecha
      Paso 3 — NL+catálogo: steps para resolver el problema
      Paso 4 — LLM: mapear/validar steps al catálogo (repair si hay errores)
      Paso 5 — Python: clasificar primitivos vs sub-goals; encolar + DAG
    Paso 6 — Recursión: nuevos sub-planes expandidos hasta _MAX_EXPAND_DEPTH
    """
    dag: nx.DiGraph = _dag if _dag is not None else nx.DiGraph()
    if not dag.has_node(goal_sig):
        dag.add_node(goal_sig, status="main" if _depth == 0 else "pending")

    plan_library: dict[str, str] = dict(_known_plans or {})
    for known_sig in existing_goals:
        plan_library.setdefault(known_sig, known_sig)
    # Builtin plans are always available — seed so LLM sees them as reusable sub-goals.
    # Fase 16 (ablación): sin los sub-planes macro si builtin_subplans=False.
    for _bsig, _bdesc in _BUILTIN_PLANS.items():
        if not builtin_subplans and _bsig in ABLATABLE_SUBPLANS:
            continue
        plan_library.setdefault(_bsig, _bdesc)
    if coordination and builtin_subplans:
        for _csig, _cdesc in _COORDINATION_PLANS.items():
            plan_library.setdefault(_csig, _cdesc)

    active_stack = _active_stack + (goal_sig,)

    capability_contracts: dict = _capability_contracts if isinstance(_capability_contracts, dict) else {}
    if use_refinement and not capability_contracts:
        capability_contracts = load_contracts(capability_contracts_path)

    # ── Paso 0: NL → sig + descripción ────────────────────────────────────
    sig = goal_sig
    if not description:
        try:
            step0_result = await run_step0(
                npc_statement, llm_call,
                npc_profile=None,
            )
            description = step0_result.get("description", npc_statement)
        except Exception as exc:
            log.warning(f"[PIPELINE:{goal_sig}] step0 falló, usando npc_statement como desc: {exc}")
            description = npc_statement

    plan_library.setdefault(sig, description or sig)
    if dag.has_node(sig):
        dag.nodes[sig]["status"] = "main" if _depth == 0 else "generating"

    # ── Paso 1: sig + desc → success_model + .done variant ────────────────
    # Si se pasa success_condition (de parse_goals) se bypasea el LLM:
    # la condicion de exito ya fue decidida y no debe volver a derivarse.
    if success_condition:
        from llm.pipeline.step1_name import _derive_success_model_from_conditions
        _sc_list = [success_condition]
        _sm = _derive_success_model_from_conditions(_sc_list)
        step1_result = {
            "success_model": _sm,
            "success_conditions": _sc_list,
            "done_guard": success_condition,
            "done_asl": f"+!{sig} : {success_condition} <- true.",
        }
        log.info(
            "[PIPELINE:%s] step1 fijado por parse_goals — success_condition=%r",
            sig,
            success_condition,
        )
    else:
        try:
            step1_result = await run_step1(sig, description, llm_call, entity_catalog=entity_catalog)
        except Exception as exc:
            log.warning(f"[PIPELINE:{sig}] step1 fallo: {exc}")
            step1_result = {
                "success_model": [],
                "success_conditions": [],
                "done_guard": "true",
                "done_asl": f"+!{sig} : true <- true.",
            }

    success_model: list[dict] = _dedupe_success_model(step1_result.get("success_model", []))
    success_conditions: list[str] = step1_result.get("success_conditions", [])
    done_guard: str = step1_result.get("done_guard", "true")
    done_asl: str = step1_result.get("done_asl", f"+!{sig} : true <- true.")

    variants: list[PlanVariant] = []
    # .done variant siempre la primera — andamiaje (cuerpo trivial `true`), no LLM.
    variants.append(PlanVariant(guard=done_guard, steps=[], asl=done_asl, source="CODE"))

    new_subgoals: dict[str, str] = {}  # name → description

    # ── Paso 1b: escalera de variantes desde recipe catalog ───────────────
    # Solo cuando success_condition viene de parse_goals y hay datos de recipe.
    # Genera N VariantSpec (escalera profundidad→surface) en lugar de iterar
    # success_model con una sola entrada.
    _need_variants: list | None = None
    # Fase 17f: sin sub-planes, un item recolectable también tiene escalera
    # (precondiciones de PickUp) aunque el NPC no tenga recetas.
    if success_condition and _depth == 0 and (recipes or (not builtin_subplans and item_spawns)):
        _need_variants = build_need_variants(
            success_condition,
            recipes,
            item_spawns or [],
            builtin_subplans=builtin_subplans,
            coordination=coordination,
            peers=peers,
        ) or None

    # ── Bucle por cada variante ────────────────────────────────────────────
    # Si step1b generó variantes, se itera sobre ellas; si no, sobre success_model.
    _variant_iter = _need_variants if _need_variants else success_model
    for variant_spec in _variant_iter:
        # Fase 17s: una clave con binding (`sig__args`, clave del plan_graph de otro
        # goal activo) no es un sub-goal invocable (piloto 17r, CO6: el miller metía
        # su encargo de harina en el plan del pan → plan_failure).
        existing_subgoals_list = [
            s for s in plan_library.keys()
            if not is_same_goal_sig(s, sig) and "__" not in s
        ]

        # Distinguir entre VariantSpec de step1b y dict de success_model
        already_satisfied: list[str] = []
        unsatisfied_condition: str = ""
        if isinstance(variant_spec, _VariantSpec):
            # Variante generada por step1b: guard y problem_nl ya son deterministas
            neg_guard = variant_spec.variant_guard
            problem_nl = variant_spec.problem_nl
            known_facts = variant_spec.known_facts
            if not builtin_subplans:
                # Fase 16 (ablación): move_to_and_pickup/craft_item leían en runtime
                # item_at/zone_center/at_zone. Sin ellos, el LLM recibe esas mismas
                # creencias al planificar cada peldaño — paridad de información.
                known_facts = list(dict.fromkeys(
                    list(variant_spec.known_facts)
                    + rung_state_facts(
                        _build_known_facts_from_beliefs(beliefs or {}),
                        variant_spec.already_satisfied,
                    )
                ))
            already_satisfied = variant_spec.already_satisfied
            unsatisfied_condition = variant_spec.unsatisfied_condition
            condition = neg_guard  # usado solo para logging
            guards_expr: list[str] = []
            bound_variables: list[str] = list(variant_spec.bound_variables)
            # Fase 17i: los literales positivos del guard que ligan esas variables
            # (item_at(wheat, X, Y), peer_item_available(P, …)) se muestran como Facts.
            # Antes step3 veía "Facts: (none)" y los nombres X, Y sin significado
            # (tanda 3 ATOM: 0/160 respuestas usaron MoveTo(X, Y)).
            facts_expr: list[str] = _bound_guard_facts(neg_guard, bound_variables)
            facts: list[dict] = [_parse_fact_expression(expr) for expr in facts_expr]
            guards: list[dict] = []
        else:
            # Variante del success_model original (comportamiento previo)
            condition = variant_spec.get("done_fragment", "true")
            facts_expr = [f for f in variant_spec.get("facts", []) if isinstance(f, str) and f]
            guards_expr = [g for g in variant_spec.get("guards", []) if isinstance(g, str) and g]
            bound_variables = [v for v in variant_spec.get("bound_variables", []) if isinstance(v, str) and v]
            neg_guard = build_negated_guard(condition)
            facts = [_parse_fact_expression(expr) for expr in facts_expr]
            guards = [{"expr": expr} for expr in guards_expr]

            # Paso 2: problema NL + known_facts
            if success_condition:
                problem_nl = f"The condition '{condition}' is not currently satisfied."
                known_facts = _build_known_facts_from_beliefs(beliefs or {})
                log.info(
                    "[PIPELINE:%s] step2 determinista — known_facts=%d porque success_condition ya viene de parse_goals",
                    sig,
                    len(known_facts),
                )
            else:
                try:
                    step2_result = await run_step2(
                        sig, description, neg_guard, llm_call,
                        beliefs=beliefs, entity_catalog=entity_catalog,
                        facts=facts_expr,
                        guards=guards_expr,
                        bound_variables=bound_variables,
                    )
                    problem_nl = step2_result.get("problem_nl", f"The condition '{condition}' is not met.")
                    known_facts = step2_result.get("known_facts", [])
                except Exception as exc:
                    log.warning(f"[PIPELINE:{sig}] step2 falló para cond '{condition}': {exc}")
                    problem_nl = f"The condition '{condition}' is not met."
                    known_facts = []

        # Fase 17: peldaño con cuerpo FIJO de andamiaje (la entrega al peer tras
        # aceptar un request: .drop + .deliver_to_peer). Es el compromiso del
        # protocolo, no una decisión de plan: no pasa por el LLM y se traza CODE.
        _fixed_steps = (
            list(getattr(variant_spec, "fixed_steps", None) or [])
            if isinstance(variant_spec, _VariantSpec) else []
        )
        if _fixed_steps:
            steps = [dict(s) for s in _fixed_steps]
            _plan_transform(
                "scaffold_fixed_variant", "CODE",
                reason=f"peldaño de andamiaje de '{sig}' (sin LLM)",
                guard=neg_guard, after=_summarize_steps(steps),
            )
            variants.append(PlanVariant(
                guard=neg_guard,
                steps=steps,
                asl=compile_plan_to_asl({
                    "sig": sig, "guard": neg_guard, "body": _steps_to_asl_body(steps),
                }),
                source="CODE",
            ))
            continue

        # Paso 3: steps para resolver este problema
        try:
            steps = await run_step3(
                sig, description, neg_guard, problem_nl, known_facts,
                existing_subgoals_list, llm_call, entity_catalog=entity_catalog,
                facts=facts,
                guards=guards,
                bound_variables=bound_variables,
                atomic_only=use_refinement,
                replan_hint=replan_hint,
                already_satisfied=already_satisfied,
                unsatisfied_condition=unsatisfied_condition,
                builtin_subplans=builtin_subplans,
                coordination=coordination,
                peers=peers,
            )
        except Exception as exc:
            log.warning("[PIPELINE:%s] step3 fallo para cond %r: %s", sig, condition, exc)
            steps = []

        # Args como objeto → posicionales antes de validar y reparar.
        _normalize_step_args(steps)

        # craft_item_strip_arg es específico del sub-plan craft_item: sin él
        # (Fase 16) un `craft_item` del LLM es un sub-goal suyo y no se toca.
        if builtin_subplans:
            _repair_subgoal_args(steps)
        _repair_craft_args(steps)
        _normalize_step_types(steps, set(plan_library.keys()))

        # Fase 6.5: mini-repairs dirigidos al LLM en vez de fabricar (sustituyen
        # el qty=1 por defecto y el truncado silencioso). Si un hueco no se puede
        # reparar preguntando al LLM, MiniRepairError aborta el plan (ruidoso).
        if steps:
            from llm.pipeline.mini_repair import repair_plan_gaps
            steps = await repair_plan_gaps(
                steps, sig, llm_call, entity_catalog=entity_catalog,
                terminal_subplans=active_terminal_subplans(builtin_subplans, coordination),
            )

        if steps:
            log.info(
                "[PIPELINE:%s] step3 draft para %r: %s",
                sig,
                condition,
                _summarize_steps(steps),
            )

        # Paso 4: mapear/validar steps al catálogo de acciones disponibles
        try:
            steps = await run_step4(
                sig, steps, existing_subgoals_list, llm_call,
                facts=facts,
                guards=guards,
                bound_vars=set(bound_variables),
                atomic_only=use_refinement,
            )
        except Exception as exc:
            log.warning(f"[PIPELINE:{sig}] step4 falló para cond '{condition}': {exc}")

        # Paso 5 (V4 draft, opcional): refinamiento por necesidades/capacidades.
        # Se omite para variantes step1b (_VariantSpec): el guard ya encoda las
        # pre-condiciones de forma determinista (e.g. at_zone en happy-path), por
        # lo que step5 introduciría subgoals redundantes o erróneos.
        if use_refinement and steps and not isinstance(variant_spec, _VariantSpec):
            refine_result = await run_step5_refine(
                sig,
                steps,
                existing_subgoals_list,
                llm_call=llm_call,
                entity_catalog=entity_catalog,
                capability_contracts=capability_contracts,
                beliefs=beliefs or {},
            )
            steps = refine_result.steps
            if refine_result.created_plans:
                if upsert_created_plan_contracts(capability_contracts, refine_result.created_plans):
                    log.info(
                        "[PIPELINE:%s] Updated capability contracts with %d plan(s)",
                        sig,
                        len(refine_result.created_plans),
                    )
                for created in refine_result.created_plans:
                    created_sig = str(created.get("sig", "")).strip()
                    if not created_sig:
                        continue
                    created_desc = str(created.get("description", "")).strip()
                    plan_library.setdefault(created_sig, created_desc or created_sig)
            if refine_result.resolved_needs:
                log.info(
                    "[PIPELINE:%s] step5_refine resolvió %d need(s)",
                    sig,
                    len(refine_result.resolved_needs),
                )
            if refine_result.unresolved_needs:
                log.debug(
                    "[PIPELINE:%s] step5_refine dejó %d need(s) sin resolver",
                    sig,
                    len(refine_result.unresolved_needs),
                )

        # Reachability check: solo cuando use_refinement=True, ya que es ahí donde
        # step5_refine puede introducir sub-goals cuyos efectos necesitan verificarse.
        # Con use_refinement=False los planes son atómicos simples; el simulador no
        # puede resolver variables de receta (ItemToCraft) ni args simbólicos.
        # Para variantes step1b los pasos son precomputados desde el catálogo — no hay
        # necesidad de verificar porque son correctos por construcción.
        _reach_cond = success_condition if success_condition else condition
        if use_refinement and steps and _reach_cond and not isinstance(variant_spec, _VariantSpec):
            _beliefs_snap = beliefs or {}
            _reach = check_plan_reachability(
                steps, _beliefs_snap, _reach_cond,
                subgoal_guarantees=active_subgoal_guarantees(builtin_subplans, coordination),
            )
            if not _reach.reachable:
                log.warning(
                    "[PIPELINE:%s] plan unreachable — missing %s, requesting repair",
                    sig,
                    _reach.missing_predicates,
                )
                try:
                    _repair_payload = {
                        "task":               "step3_repair_completeness",
                        "goal_sig":           sig,
                        "success_condition":  _reach_cond,
                        "current_steps":      steps,
                        "missing_predicates": _reach.missing_predicates,
                        "hint":               _reach.hint,
                        "entity_catalog":     entity_catalog,
                    }
                    _ru, _rs = _build_prompt(_repair_payload)
                    _repair_raw = await llm_call(_ru, _rs)
                    import json as _json
                    _repaired = _json.loads(_repair_raw)
                    _new_steps = _repaired.get("steps", steps)
                    # Verificar si el repair solucionó el problema.
                    # Debe evaluarse contra la MISMA condición del primer check
                    # (_reach_cond = success_condition si existe), no contra `condition`,
                    # que puede ser el done_fragment u otra cosa en variantes step1b.
                    _reach2 = check_plan_reachability(
                        _new_steps, _beliefs_snap, _reach_cond,
                        subgoal_guarantees=active_subgoal_guarantees(builtin_subplans, coordination),
                    )
                    if _reach2.reachable:
                        steps = _new_steps
                        log.info("[PIPELINE:%s] repair_completeness solved reachability", sig)
                    else:
                        log.warning(
                            "[PIPELINE:%s] repair_completeness did not solve — keeping original plan",
                            sig,
                        )
                except Exception as _exc:
                    log.warning("[PIPELINE:%s] repair_completeness failed: %s", sig, _exc)

        # Fail-fast: no aceptar variantes de ejecución sin steps.
        # Evita planes vacíos que dejan al BDI en bucles de "sin variante aplicable".
        if not steps:
            raise ValueError(
                f"[PIPELINE:{sig}] No executable steps for condition '{condition}' "
                f"(neg_guard='{neg_guard}')"
            )

        # Forma final antes de compilar ASL: step4/step5/repair pueden haber
        # devuelto args como objeto, y los ids del mundo en mayúscula serían variables.
        _normalize_step_args(steps)
        _lowercase_asl_constants(steps)

        # Paso 5: clasificar primitivos vs sub-goals + encolar nuevos + grafo DAG
        _queue: collections.deque = collections.deque()
        _, to_expand = classify_steps(steps)
        for sub_step in to_expand:
            name = sub_step.get("name", "")
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            # description viene del paso 3 (puede incluir replaces_steps)
            desc = sub_step.get("description", "")
            if name in active_stack:
                log.warning(
                    "[PIPELINE:%s] ciclo detectado: %s -> %s; sub-goal descartado para rectificación",
                    sig,
                    sig,
                    name,
                )
                continue
            enqueued = enqueue_if_needed(name, sig, dag, plan_library, _queue)
            if not enqueued:
                log.warning(
                    "[PIPELINE:%s] sub-goal '%s' descartado por ciclo o duplicado inválido",
                    sig,
                    name,
                )
                continue
            plan_library.setdefault(name, desc or name)
            if name not in existing_goals and name not in new_subgoals:
                new_subgoals[name] = desc

        # Compilar variante ASL para esta condición
        variant_asl = compile_plan_to_asl({
            "sig": sig,
            "guard": neg_guard,
            "body": _steps_to_asl_body(steps),
        })
        variants.append(
            PlanVariant(
                guard=neg_guard,
                steps=steps,
                asl=variant_asl,
                facts=facts_expr,
                guards=guards_expr,
                bound_variables=bound_variables,
            )
        )

        # W2 (T5, Fase 6.5): aviso defensivo si una rama de recolección no se acota
        # por estado del mundo (item_spawn/recipe_output) ni por has_item — riesgo
        # de rama demasiado permisiva. No bloquea; se traza como CODE (no-silencio).
        if gather_variant_underconstrained(neg_guard, steps):
            log.warning(
                "[PIPELINE:%s] Variante de recolección con guard permisivo '%s' "
                "(no referencia item_spawn/recipe_output/has_item)", sig, neg_guard,
            )
            _plan_transform(
                "variant_underconstrained_gather", "CODE",
                reason=f"rama de gather de '{sig}' sin acotar por estado del mundo",
                guard=neg_guard,
            )

        # Iter 7: emitir variantes de agotamiento para acciones observacionales.
        exhaustion_variants = _emit_exhaustion_branches(
            sig, neg_guard, steps, dag, _ACTION_CONTRACT_REGISTRY,
        )
        variants.extend(exhaustion_variants)

    subgoals_to_expand = [SubgoalEntry(sig=n, description=d) for n, d in new_subgoals.items()]

    # ── Paso 6: expansión recursiva de sub-planes nuevos ─────────────────
    sub_results: dict[str, PipelineResult] = {}
    if _depth < _MAX_EXPAND_DEPTH and subgoals_to_expand:
        all_known = list(plan_library.keys())
        for subgoal in subgoals_to_expand:
            log.info(
                f"[PIPELINE:{sig}] Paso 6 — expandiendo sub-plan '{subgoal.sig}' "
                f"(depth={_depth+1})"
            )
            try:
                sub_result = await run_full_pipeline(
                    goal_sig=subgoal.sig,
                    npc_statement=subgoal.description or subgoal.sig,
                    existing_goals=all_known,
                    llm_call=llm_call,
                    description=subgoal.description or None,
                    use_refinement=use_refinement,
                    entity_catalog=entity_catalog,
                    beliefs=beliefs,
                    builtin_subplans=builtin_subplans,
                    coordination=coordination,
                    peers=peers,
                    _depth=_depth + 1,
                    _dag=dag,
                    _known_plans=plan_library,
                    _capability_contracts=capability_contracts,
                    _active_stack=active_stack,
                )
                sub_results[subgoal.sig] = sub_result
                # Añadir a plan_library para que sub-sub-planes puedan reusarlo
                plan_library[subgoal.sig] = subgoal.description or subgoal.sig
                if subgoal.sig not in all_known:
                    all_known.append(subgoal.sig)
            except Exception as exc:
                log.warning(
                    f"[PIPELINE:{sig}] Paso 6 falló para '{subgoal.sig}': {exc}"
                )

    log.info(
        f"[PIPELINE:{sig}] Completado (depth={_depth}) — "
        f"{len(success_model)} variantes de éxito, "
        f"{len(variants)} variantes ASL, "
        f"{len(subgoals_to_expand)} sub-goals "
        f"({len(sub_results)} expandidos en paso 6)"
    )

    dag_nodes = [
        {"id": str(node), **attrs}
        for node, attrs in dag.nodes(data=True)
    ]
    dag_edges = [[str(source), str(target)] for source, target in dag.edges()]

    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError(f"[PIPELINE:{sig}] DAG validation failed: cycle detected")

    if use_refinement and _depth == 0:
        save_contracts(capability_contracts, capability_contracts_path)

    return PipelineResult(
        sig=sig,
        description=description,
        success_conditions=success_conditions,
        variants=variants,
        success_model=success_model,
        subgoals_to_expand=subgoals_to_expand,
        sub_results=sub_results,
        dag_nodes=dag_nodes,
        dag_edges=dag_edges,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _emit_exhaustion_branches(
    sig: str,
    neg_guard: str,
    steps: list[dict],
    dag: "nx.DiGraph",
    contracts: object | None,
) -> list["PlanVariant"]:
    """Iter 7 — para cada step con may_observe en su contrato semántico, emite
    una variante de agotamiento de presupuesto.

    Políticas:
      goal_failed  → plan variant con body .fail + nodo goal_failed en el DAG.
      replan_goal  → stub: log + nodo replan en DAG (no implementado en Iter 7).
      degrade_goal → stub: log + nodo degrade en DAG (no implementado en Iter 7).

    Retorna lista de PlanVariant adicionales (puede estar vacía).
    """
    if contracts is None:
        return []

    extra: list[PlanVariant] = []
    for step in steps:
        action_name = step.get("name", "")
        if not isinstance(action_name, str) or not action_name.strip():
            continue
        action_name = action_name.strip()
        contract = contracts.get(action_name)
        if contract is None or not getattr(contract, "may_observe", None):
            continue

        budget: int = getattr(contract, "attempt_budget", 1)
        policy: str = getattr(contract, "on_exhaustion", "goal_failed")

        # Guard de agotamiento = guard normal + exhausted(Action, budget)
        exhaustion_literal = f"exhausted({action_name}, {budget})"
        exh_guard = f"{neg_guard} & {exhaustion_literal}"

        # Construir DepVariant para reutilizar exhaustion_policy_asl
        dep_variant = DepVariant.build(
            parts=neg_guard.split(" & ") + [exhaustion_literal],
            negated_belief="",
            producer_action=action_name,
            is_final=False,
            attempt_budget=budget,
            on_exhaustion=policy,
            is_exhaustion_branch=True,
        )
        exh_asl = exhaustion_policy_asl(sig, dep_variant)

        if policy == "goal_failed":
            gf_node = f"goal_failed_{sig}"
            if not dag.has_node(gf_node):
                dag.add_node(gf_node, status="goal_failed")
            if not dag.has_edge(sig, gf_node):
                dag.add_edge(sig, gf_node)
            log.info(
                "[PIPELINE:%s] exhaustion branch \'%s\' budget=%d → goal_failed DAG node \'%s\'",
                sig, action_name, budget, gf_node,
            )
        elif policy == "replan_goal":
            rp_node = f"replan_{sig}"
            if not dag.has_node(rp_node):
                dag.add_node(rp_node, status="replan_stub")
            if not dag.has_edge(sig, rp_node):
                dag.add_edge(sig, rp_node)
            log.info(
                "[PIPELINE:%s] exhaustion branch \'%s\' budget=%d → replan_goal stub",
                sig, action_name, budget,
            )
        elif policy == "degrade_goal":
            dg_node = f"degrade_{sig}"
            if not dag.has_node(dg_node):
                dag.add_node(dg_node, status="degrade_stub")
            if not dag.has_edge(sig, dg_node):
                dag.add_edge(sig, dg_node)
            log.info(
                "[PIPELINE:%s] exhaustion branch \'%s\' budget=%d → degrade_goal stub",
                sig, action_name, budget,
            )

        # Andamiaje generado por código (.fail / .print stub), no por el LLM.
        extra.append(PlanVariant(guard=exh_guard, steps=[], asl=exh_asl, source="CODE"))
        if policy in ("replan_goal", "degrade_goal"):
            _trace(
                "policy_stub",
                goal=sig,
                action=action_name,
                policy=policy,
                budget=budget,
                note="política declarada pero no implementada (stub .print)",
            )

    return extra



def _signature_arg_names(step: dict) -> list[str]:
    """Nombres de argumento, en orden, de la firma conocida del step (acción de
    Unity, acción de coordinación o sub-plan builtin). [] si no hay firma."""
    from protocol.action_contract import ACTION_ALLOWLIST
    from llm.catalogs import PEER_ACTION_SPECS
    from llm.pipeline.mini_repair import _BUILTIN_ARGS

    name = str(step.get("name", ""))
    if step.get("type") == "subgoal":
        return list(_BUILTIN_ARGS.get(name, []))
    for action, spec in ACTION_ALLOWLIST.items():
        if action.lower() == name.lower():
            return list(spec.get("asl_args", []))
    for action, spec in PEER_ACTION_SPECS.items():
        if action == name:
            return list(spec.get("asl_args", []))
    return []


def _normalize_step_args(steps: list[dict]) -> None:
    """Corrección CLARA de forma (trazada CODE): args como objeto → posicionales.

    El LLM a veces da los args como objeto (`{"X": 9, "Y": -6}`) o como lista de
    objetos (`[{"itemId": "wheat"}]`). Compilados tal cual, el ASL recibía el
    texto de un dict de Python y la rama no compilaba (tanda ATOM 2026-09-15:
    todas las sesiones de A3). Se pasan a posicionales en el orden de la firma
    conocida; si las claves no casan con ninguna firma, en el orden en que las
    escribió el LLM. No se añade ni se inventa ningún valor. Los marcadores
    `{"var": ...}` / `{"value": ...}` que ya entiende `_steps_to_asl_body` no se tocan.
    Muta steps in-place.
    """
    for step in steps:
        if not isinstance(step, dict):
            continue
        args = step.get("args")
        if isinstance(args, dict):
            parts = [args]
        elif isinstance(args, list) and args and all(isinstance(a, dict) for a in args):
            if all(set(a) <= {"var", "value"} for a in args):
                continue
            parts = list(args)
        else:
            continue
        mapping: dict = {}
        for part in parts:
            mapping.update(part)
        order = _signature_arg_names(step)
        by_lower = {str(k).lower(): v for k, v in mapping.items()}
        if order and all(str(k).lower() in {o.lower() for o in order} for k in mapping):
            values = [by_lower[o.lower()] for o in order if o.lower() in by_lower]
            how = "orden de la firma"
        else:
            values = list(mapping.values())
            how = "orden escrito por el LLM"
        step["args"] = values
        _plan_transform(
            "args_object_to_positional", "CODE", before=args, after=values,
            reason=f"'{step.get('name')}': args como objeto → posicionales ({how})",
        )


def _lowercase_asl_constants(steps: list[dict]) -> None:
    """Corrección CLARA de sintaxis ASL (trazada CODE).

    En AgentSpeak un identificador que empieza por mayúscula es una VARIABLE.
    Los ids del mundo usados como constantes (`Bread_recipe`, `Bakeri_point`)
    compilaban como variables sin ligar y la acción fallaba con `term not ground`
    (p.ej. `.craft(wheat, Bread_recipe)`). Se pasan a minúscula: las creencias ya
    los guardan así y `_send_and_wait` restaura el case original de la receta al
    hablar con Unity. Las variables de verdad (X, Y, P, Qty…) no se tocan.
    Solo en ACCIONES (que groundean sus args al ejecutarse): en una llamada a
    sub-goal el identificador en mayúscula unifica sin error, así que el plan con
    sub-planes queda idéntico al de siempre. Muta steps in-place.
    """
    from llm.pipeline.step3_validator import _is_variable

    for step in steps:
        if not isinstance(step, dict) or step.get("type") == "subgoal":
            continue
        args = step.get("args")
        if not isinstance(args, list):
            continue
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg[:1].isupper() and not _is_variable(arg):
                args[i] = arg.lower()
                _plan_transform(
                    "asl_constant_lowercase", "CODE", before=arg, after=args[i],
                    reason=f"'{arg}' compilaría como variable ASL sin ligar",
                )


def _peer_action_alias(name: object) -> str | None:
    """Fase 17: nombre canónico de una acción de coordinación (AskPeer, ask-peer…
    → ask_peer), o None si no lo es."""
    if not isinstance(name, str):
        return None
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    for canon in PEER_ACTION_NAMES:
        if key == canon.replace("_", ""):
            return canon
    return None


def _normalize_step_types(steps: list[dict], known_subgoals: set[str]) -> None:
    """Normalize LLM type field mistakes before step4 validation.

    Correcciones CLARAS (trazadas como CODE):
    1. "sub-goal" (con guion) → "subgoal"
    2. type="action" para un nombre de sub-plan conocido → "subgoal"

    Fase 6.5: el truncado de los pasos tras un builtin terminal ya NO se hace
    aquí (era una heurística silenciosa que borraba contenido del LLM) — se
    delega al mini-repair, que pregunta al LLM si esos pasos sobran.
    Muta steps in-place.
    """
    for step in steps:
        step_type = step.get("type", "")
        name = step.get("name", "")
        # Fase 17: acciones de coordinación — nombre canónico y tipo "action".
        canon = _peer_action_alias(name)
        if canon is not None:
            if name != canon:
                step["name"] = canon
                _plan_transform(
                    "peer_action_rename", "CODE", before=name, after=canon,
                    reason="nombre canónico de acción de coordinación",
                )
            if step_type != "action":
                step["type"] = "action"
                _plan_transform(
                    "peer_action_type", "CODE", before=step_type, after="action",
                    reason=f"'{canon}' es una acción de coordinación, no un sub-goal",
                )
            continue
        if step_type == "sub-goal":
            step["type"] = "subgoal"
            _plan_transform(
                "type_normalize", "CODE", before="sub-goal", after="subgoal",
                reason=f"step '{name}'",
            )
        elif step_type == "action" and name in known_subgoals:
            step["type"] = "subgoal"
            _plan_transform(
                "type_action_to_subgoal", "CODE", before="action", after="subgoal",
                reason=f"'{name}' es un sub-plan conocido",
            )

    # Fase 6.5: el truncado de los pasos tras un builtin terminal (#5b) ya NO se
    # hace por heurística silenciosa — se delega al mini-repair, que pregunta al
    # LLM si esos pasos sobran. Aquí solo se normaliza el tipo.


def _repair_craft_args(steps: list[dict]) -> None:
    """Fix Craft(recipeId, itemId, qty?) → Craft(itemId, recipeId, qty?).

    The LLM sometimes emits args in wrong order: recipe first, ingredient second.
    Recipe IDs in this system are PascalCase with underscore (Bread_recipe);
    item IDs are lowercase (wheat, bread).
    If args[0] looks like a recipe ID and args[1] looks like an item ID, swap them.
    Mutates steps in-place.
    """
    import re as _re
    for step in steps:
        if step.get("type") != "action" or step.get("name") != "Craft":
            continue
        args = step.get("args") or []
        if len(args) < 2:
            continue
        a0, a1 = str(args[0]), str(args[1])
        # Recipe ID heuristic: PascalCase with underscore-lowercase (Bread_recipe)
        is_recipe = lambda s: bool(_re.search(r'_[a-z]', s)) and s[0].isupper()  # noqa
        is_item = lambda s: s.islower() or (s and s[0].islower())  # noqa
        if is_recipe(a0) and is_item(a1):
            new_args = [a1, a0] + list(args[2:])
            _plan_transform(
                "craft_reorder", "CODE", before=[a0, a1], after=[a1, a0],
                reason="Craft(itemId, recipeId): orden de args invertido",
            )
            step["args"] = new_args


def _repair_subgoal_args(steps: list[dict]) -> None:
    """Corrección CLARA (#3b): quita un arg evidentemente sobrante de craft_item.

    craft_item: la firma es (recipeId, qty) — 2 args. El LLM a veces genera
    (itemId, recipeId, qty) → se quita el itemId inicial. Es una corrección
    inequívoca (no añade información) → se mantiene, pero TRAZADA como CODE.

    Fase 6.5: el qty AUSENTE de move_to_and_pickup ya NO se rellena con `1` por
    defecto — eso era fabricación silenciosa. Se delega al mini-repair (#3a), que
    pregunta al LLM la cantidad real o falla ruidoso. Muta steps in-place.
    """
    for step in steps:
        if step.get("type") != "subgoal":
            continue
        name = step.get("name", "")
        args = [a for a in (step.get("args") or []) if a is not None]
        if name == "craft_item" and len(args) == 3:
            new_args = [args[1], args[2]]
            _plan_transform(
                "craft_item_strip_arg", "CODE", before=args, after=new_args,
                reason="craft_item(recipeId, qty): itemId inicial sobrante",
            )
            step["args"] = new_args


_NPC_STATE_FUNCTORS = ("current_position", "at_zone", "has_item")
# Fase 17v: estado del protocolo en el snapshot (resultado de ask/request/await
# anteriores). Los peldaños que dependen de él lo ligan por su guard; en los demás
# confundía al LLM (piloto 17u, CO5: con peer_can_make(npc_miller, flour) en Known
# facts la rama de recoger la harina entregada escribía "recolectar trigo" 3/3;
# sin él, MoveTo(X, Y) + PickUp(flour) 3/3). `peer(Npc, Rol)` (directorio) se queda.
_PEER_STATE_FUNCTORS = (
    "peer_can_make", "peer_has_item", "peer_knows_zone", "peer_busy", "peer_promised",
    "peer_refused", "peer_done", "peer_failed", "peer_item_available",
)


def rung_state_facts(snapshot_facts: list[str], already_satisfied: list[str] | tuple) -> list[str]:
    """Fase 17u: hechos del snapshot para el prompt de UN peldaño de la escalera.

    El peldaño se planifica ANTES de ejecutarse, para el estado que fija su guard
    (p.ej. has_item(wheat, 2) & at_zone(bakeri)), pero Known facts traía el estado
    del NPC al planificar (current_position(0, 0), at_zone(farmland), sin trigo).
    El prompt se contradecía con "Already satisfied" y el LLM seguía los hechos:
    volvía a recolectar (piloto 17t). Se quitan los hechos de estado del propio NPC
    (posición, zona, inventario) y se añaden los que garantiza el guard; el resto
    (zonas, recetas, item_at…) no cambia.
    """
    kept = [
        fact for fact in snapshot_facts
        if not any(
            fact.startswith(f"{functor}(")
            for functor in _NPC_STATE_FUNCTORS + _PEER_STATE_FUNCTORS
        )
    ]
    guaranteed = [
        atom for atom in already_satisfied or ()
        if any(str(atom).startswith(f"{functor}(") for functor in _NPC_STATE_FUNCTORS)
    ]
    return guaranteed + kept


def _build_known_facts_from_beliefs(beliefs: dict) -> list[str]:
    """Extrae los facts relevantes del snapshot de beliefs en formato ASL string.

    Se usan en el prompt de step3 como contexto determinístico — sin necesidad de
    preguntar al LLM qué facts son relevantes (step2).
    """
    facts: list[str] = []
    for functor, rows in beliefs.items():
        for row in rows:
            if row:
                args_str = ", ".join(str(a) for a in row)
                facts.append(f"{functor}({args_str})")
    return facts



def _bound_guard_facts(guard: str, bound_variables: list[str]) -> list[str]:
    """Fase 17i: literales POSITIVOS del guard que contienen alguna variable ligada.

    Son los que dan valor a esas variables cuando la rama se ejecuta. Sin
    variables ligadas devuelve [] (el prompt queda como antes).
    """
    if not bound_variables:
        return []
    bound = set(bound_variables)
    out: list[str] = []
    for part in (guard or "").split(" & "):
        literal = part.strip()
        if not literal or literal.startswith("not ") or "(" not in literal or not literal.endswith(")"):
            continue
        args = {a.strip() for a in literal[literal.index("(") + 1:-1].split(",")}
        if bound & args:
            out.append(literal)
    return out


def _parse_fact_expression(expression: str) -> dict:
    """Convert a simple ASL fact string into the fact dict shape used elsewhere."""
    expr = expression.strip()
    if not expr.endswith(")") or "(" not in expr:
        return {"functor": expr, "args": []}
    functor, raw_args = expr.split("(", 1)
    args = [arg.strip() for arg in raw_args[:-1].split(",") if arg.strip()]
    return {"functor": functor.strip(), "args": args}



def _dedupe_success_model(success_model: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for entry in success_model:
        if not isinstance(entry, dict):
            continue
        fragment = str(entry.get("done_fragment", "")).strip()
        if not fragment or fragment in seen:
            continue
        seen.add(fragment)
        deduped.append(entry)
    return deduped


def _steps_to_asl_body(steps: list[dict]) -> list[str]:
    """Convierte una lista de step-dicts a strings ASL para el cuerpo del plan.

    Acepta tanto el formato canónico {"type":"action","name":"MoveTo"} como
    variantes que algunos modelos devuelven: {"type":"primitive","action":"MoveTo"}.
    """
    body: list[str] = []
    for step in steps:
        step_type = step.get("type", "")
        # Normalise type: "primitive" → "action"
        if step_type == "primitive":
            step_type = "action"
        # Accept both "name" and "action" keys for the action name
        name = step.get("name") or step.get("action", "")
        args = step.get("args", [])
        # Un arg puede ser un string, un número o un dict {"var": "N"} (variable ligada)
        def _arg_to_str(a: object) -> str:
            if isinstance(a, dict):
                return str(a.get("var", a.get("value", str(a))))
            return str(a)
        args_str = ", ".join(_arg_to_str(a) for a in args)
        if step_type == "subgoal":
            term = f"!{name}({args_str})" if args_str else f"!{name}"
        else:
            term = f".{name.lower()}({args_str})" if args_str else f".{name.lower()}"
        body.append(term)
    return body



def _summarize_steps(steps: list[dict]) -> str:
    """Resumen compacto de una lista de steps para logging."""
    parts: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type", "?"))
        name = str(step.get("name", step.get("action", "?")))
        args = step.get("args", []) or []
        args_str = ", ".join(str(a) for a in args)
        tag = "!" if step_type == "subgoal" else "."
        parts.append(f"{tag}{name}({args_str})" if args_str else f"{tag}{name}")
    return " -> ".join(parts) if parts else "(empty)"
