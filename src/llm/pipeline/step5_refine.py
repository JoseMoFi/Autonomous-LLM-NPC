from __future__ import annotations

"""Paso 5 (V5 contratos): refinamiento por necesidades y capacidades.

Iter 6: _detect_unmet_needs reescrito para iterar sobre contract.requires
usando CONTRACT_REGISTRY en lugar de heuristicas hardcoded por nombre de accion.
_build_capability_index reescrito con _BUILTIN_CAPABILITIES (datos, no ifs).

Cambios respecto a V4:
- Import CONTRACT_REGISTRY desde action_semantics.
- _META_FUNCTORS: predicados de meta-tipado que se saltan en requires.
- _BUILTIN_CAPABILITIES: dict con capacidades conocidas (move_to_and_pickup,
  achieve_explore_zone) declaradas como datos, no como if-en-codigo.
- _build_capability_index: usa _BUILTIN_CAPABILITIES + contratos persistidos.
- _require_to_need: deriva un Need desde un BeliefSpec de requires.
- _observe_to_need: deriva un Need desde un BeliefSpec de may_observe.
- _detect_unmet_needs: itera sobre contract.requires y contract.may_observe.
  Conserva legacy para MoveTo(zone_tag_string).
- _infer_ingredient_from_beliefs: extrae ingrediente/qty desde beliefs de receta.
- _inventory_ok: check puntual de has_item en beliefs.
- _build_subgoal_step_from_need: mantiene casos especificos para back-compat
  mas fallback generico para capacidades declaradas en datos.
"""

from dataclasses import dataclass, field
import logging
from typing import Callable, Awaitable

from protocol.action_semantics import CONTRACT_REGISTRY, BeliefSpec

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de meta-predicados y capacidades integradas
# ---------------------------------------------------------------------------

# Functores de meta-tipado en requires: no representan beliefs reales.
_META_FUNCTORS: frozenset[str] = frozenset({"binding", "constant"})

# Need kinds que NO deben delegarse al LLM si no hay capability conocida.
# Para estos, la solución correcta es un builtin plan (e.g. craft_item);
# si no está disponible, se registra como unresolved pero no se genera subplan LLM.
_NO_LLM_FALLBACK_KINDS: frozenset[str] = frozenset({"craft_at_zone"})

# Capacidades integradas conocidas: declaradas como datos, no como if-en-codigo.
# Estructura: sig -> {provides: [{kind, constraints}], param_names: [...]}
_BUILTIN_CAPABILITIES: dict[str, dict] = {
    "move_to_and_pickup": {
        "provides": [{"kind": "inventory_at_least", "constraints": {}}],
        "param_names": ["item", "qty"],
    },
    "achieve_explore_zone": {
        "provides": [{"kind": "know_zone", "constraints": {}}],
        "param_names": ["zone"],
    },
    "craft_item": {
        "provides": [{"kind": "craft_at_zone", "constraints": {}}],
        "param_names": ["recipe_id", "qty"],
    },
}


# ---------------------------------------------------------------------------
# Dataclasses publicos
# ---------------------------------------------------------------------------

@dataclass
class Need:
    kind: str
    consumer_step_index: int
    source_step: dict
    payload: dict[str, object] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class Capability:
    kind: str
    plan_sig: str
    param_names: list[str]
    constraints: dict[str, object] = field(default_factory=dict)


@dataclass
class RefinementResult:
    steps: list[dict]
    resolved_needs: list[Need] = field(default_factory=list)
    unresolved_needs: list[Need] = field(default_factory=list)
    created_plans: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _to_need_key(need: Need) -> tuple:
    payload_items = tuple(sorted((str(k), str(v)) for k, v in need.payload.items()))
    return (need.kind, payload_items)


def _infer_ingredient_from_beliefs(recipe_id: str, beliefs: dict) -> tuple[str | None, int]:
    """Busca (ingrediente, qty) en beliefs de receta para el recipe_id dado.

    Formato esperado de belief:
      recipe: [[recipeId, zone, inputItemId, qty], ...]
    """
    rows = beliefs.get("recipe", [])
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        rid = str(row[0]).strip().lower()
        if rid == str(recipe_id).strip().lower():
            ingredient = str(row[2]).strip()
            try:
                qty = max(1, int(row[3]))
            except (TypeError, ValueError):
                qty = 1
            return ingredient, qty
    return None, 1


def _infer_recipe_zone(recipe_id: str, beliefs: dict) -> str | None:
    """Extrae la zona de crafteo para recipe_id desde beliefs de receta.

    Formato esperado: recipe: [[recipeId, zone, inputItemId, qty], ...]
    """
    rows = beliefs.get("recipe", [])
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        rid = str(row[0]).strip().lower()
        if rid == str(recipe_id).strip().lower():
            zone = str(row[1]).strip()
            return zone if zone else None
    return None


def _inventory_ok(item_id: str, qty_required: int, beliefs: dict) -> bool:
    """True si has_item(item_id) >= qty_required en beliefs actuales."""
    rows = beliefs.get("has_item", [])
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        if str(row[0]) == item_id:
            try:
                return int(row[1]) >= qty_required
            except (TypeError, ValueError):
                return False
    return False


# ---------------------------------------------------------------------------
# Construccion de capacidades
# ---------------------------------------------------------------------------

def _build_capability_index(
    existing_subgoals: list[str],
    capability_contracts: dict | None = None,
) -> list[Capability]:
    """Construye el indice de capacidades disponibles.

    Fuentes (en orden):
    1. _BUILTIN_CAPABILITIES: capacidades integradas declaradas como datos.
    2. capability_contracts: contratos persistidos en disco (dict JSON).
    """
    caps: list[Capability] = []
    known = {s.strip() for s in existing_subgoals if isinstance(s, str)}

    # 1. Capacidades integradas (datos, no ifs).
    for sig, spec in _BUILTIN_CAPABILITIES.items():
        if sig not in known:
            continue
        param_names = [str(p) for p in spec.get("param_names", []) if isinstance(p, str)]
        for prov in spec.get("provides", []):
            if not isinstance(prov, dict):
                continue
            kind = str(prov.get("kind", "")).strip()
            constraints = prov.get("constraints", {})
            if not kind:
                continue
            if not isinstance(constraints, dict):
                constraints = {}
            caps.append(Capability(
                kind=kind,
                plan_sig=sig,
                param_names=param_names,
                constraints={str(k): v for k, v in constraints.items()},
            ))

    # 2. Contratos persistidos en disco.
    contracts = capability_contracts if isinstance(capability_contracts, dict) else {}
    by_sig = contracts.get("contracts", {})
    if isinstance(by_sig, dict):
        for sig, spec in by_sig.items():
            if not isinstance(sig, str) or not sig.strip():
                continue
            if not isinstance(spec, dict):
                continue
            provides = spec.get("provides", [])
            param_names = spec.get("param_names", [])
            if not isinstance(provides, list):
                continue
            if not isinstance(param_names, list):
                param_names = []
            for prov in provides:
                if not isinstance(prov, dict):
                    continue
                kind = str(prov.get("kind", "")).strip()
                constraints = prov.get("constraints", {})
                if not kind:
                    continue
                if not isinstance(constraints, dict):
                    constraints = {}
                caps.append(Capability(
                    kind=kind,
                    plan_sig=sig,
                    param_names=[str(n) for n in param_names if isinstance(n, str)],
                    constraints={str(k): v for k, v in constraints.items()},
                ))

    return caps


# ---------------------------------------------------------------------------
# Deteccion de necesidades — funciones auxiliares
# ---------------------------------------------------------------------------

def _require_to_need(
    req: BeliefSpec,
    action_name: str,
    step_args: list,
    beliefs: dict,
    step_idx: int,
    step: dict,
) -> Need | None:
    """Convierte un BeliefSpec de requires en un Need si no esta satisfecho.

    Solo genera necesidades para functores que tienen capacidades conocidas.
    Los meta-predicados (binding, constant) deben filtrarse antes de llamar.

    Mapeos functor -> need kind implementados:
    - has_item  -> inventory_at_least  (Craft: via recipe lookup; Drop: inventario directo)
    - item_at   -> inventory_at_least  (PickUp: requiere navegacion hasta la celda del item)
    """
    functor = req.functor
    args = step_args if isinstance(step_args, list) else []

    if functor == "has_item":
        if action_name == "Craft":
            # Craft(ItemId, TargetId, ...) -> ItemId=ingrediente, TargetId=recipeId. Ingrediente real via beliefs de receta.
            recipe_id = str(args[1]) if len(args) >= 2 else ""
            if not recipe_id:
                return None
            ingredient, qty = _infer_ingredient_from_beliefs(recipe_id, beliefs)
            if not ingredient:
                return None
            if _inventory_ok(ingredient, qty, beliefs):
                return None
            return Need(
                kind="inventory_at_least",
                consumer_step_index=step_idx,
                source_step=step,
                payload={"item": ingredient, "qty": qty},
                rationale=(
                    f"Craft({args}) requires qty >= {qty} of {ingredient}, "
                    "insufficient in current inventory"
                ),
            )

        if action_name == "Drop":
            # Drop(ItemId) requiere has_item(ItemId, N) con N >= 1.
            item_id = str(args[0]).strip() if args else ""
            if not item_id:
                return None
            if _inventory_ok(item_id, 1, beliefs):
                return None
            return Need(
                kind="inventory_at_least",
                consumer_step_index=step_idx,
                source_step=step,
                payload={"item": item_id, "qty": 1},
                rationale=f"Drop({item_id}) requires at least 1 unit in inventory",
            )

    if functor == "item_at":
        if action_name == "PickUp":
            # PickUp(ItemId) requiere item_at(ItemId, X, Y): el item debe estar en el suelo
            # en la celda exacta del NPC. La capability move_to_and_pickup resuelve esto
            # navegando hasta el item y recogiendolo.
            item_id = str(args[0]).strip() if args else ""
            if not item_id:
                return None
            return Need(
                kind="inventory_at_least",
                consumer_step_index=step_idx,
                source_step=step,
                payload={"item": item_id, "qty": 1},
                rationale=(
                    f"PickUp({item_id}) requires item_at({item_id}, X, Y): "
                    "the NPC must stand on the item's tile"
                ),
            )

    if functor == "at_zone":
        if action_name == "Craft":
            # Craft requiere que el NPC esté físicamente en la zona de crafteo.
            # craft_item(RecipeId, Qty) encapsula navegar a la zona + craftear.
            recipe_id = str(args[1]) if len(args) >= 2 else ""
            if not recipe_id:
                return None
            # Verificar si at_zone ya está satisfecho para la zona de la receta.
            zone_tag = _infer_recipe_zone(recipe_id, beliefs)
            if zone_tag:
                at_zone_rows = beliefs.get("at_zone", []) if isinstance(beliefs, dict) else []
                already_there = any(
                    isinstance(r, (list, tuple)) and r and str(r[0]) == zone_tag
                    for r in at_zone_rows
                )
                if already_there:
                    return None
            return Need(
                kind="craft_at_zone",
                consumer_step_index=step_idx,
                source_step=step,
                payload={"recipe_id": recipe_id, "qty": 1},
                rationale=(
                    f"Craft({args}) requires at_zone({zone_tag or '?'}): "
                    "the NPC must be inside the crafting zone"
                ),
            )

    # Otros functores (recipe, recipe_output, current_position, zone_center...):
    # no se traducen a needs accionables en esta iteracion.
    return None


def _observe_to_need(
    obs: BeliefSpec,
    action_name: str,
    step_args: list,
    beliefs: dict,
    step_idx: int,
    step: dict,
) -> Need | None:
    """Genera un Need desde un BeliefSpec de may_observe.

    Si la accion puede descubrir zone_center y aun no se conoce,
    se emite un need know_zone (semantica: la accion ES la exploracion).
    """
    args = step_args if isinstance(step_args, list) else []

    if obs.functor == "zone_center" and args:
        # Solo para zone tags de tipo string (no coordenadas numericas).
        raw = args[0]
        if not isinstance(raw, str):
            return None
        zone = raw.strip()
        if not zone:
            return None
        # Check si ya se conoce la zona.
        rows = beliefs.get("zone_center", []) if isinstance(beliefs, dict) else []
        known = any(isinstance(r, (list, tuple)) and r and str(r[0]) == zone for r in rows)
        if known:
            return None
        return Need(
            kind="know_zone",
            consumer_step_index=step_idx,
            source_step=step,
            payload={"zone": zone},
            rationale=(
                f"{action_name}({zone}) is an explicit attempt to establish "
                f"zone_center({zone}, X, Y)"
            ),
        )

    return None


# ---------------------------------------------------------------------------
# Deteccion de necesidades — punto de entrada
# ---------------------------------------------------------------------------

def _detect_unmet_needs(steps: list[dict], beliefs: dict) -> list[Need]:
    """Detecta necesidades abiertas iterando sobre los contratos de cada step.

    Fuentes de necesidades:
    - contract.requires: beliefs que el step necesita para ejecutarse.
    - contract.may_observe: beliefs que el step puede descubrir (genera know_zone
      si la accion es de tipo exploratorio y la zona no se conoce aun).
    - Legacy: MoveTo(zone_tag_string) sin contrato de zona.
    """
    needs: list[Need] = []
    bmap = beliefs if isinstance(beliefs, dict) else {}

    for idx, step in enumerate(steps):
        if step.get("type") != "action":
            continue
        name = str(step.get("name", "")).strip()
        args = step.get("args", [])

        # --- Legacy: MoveTo con zone tag string ---
        # MoveTo(X, Y) numerico tiene contrato; MoveTo("zone") es un patron legacy
        # que el pipeline a veces genera y que requiere know_zone.
        if (
            name == "MoveTo"
            and isinstance(args, list)
            and len(args) == 1
            and isinstance(args[0], str)
        ):
            zone = args[0]
            known = any(
                isinstance(row, (list, tuple)) and row and str(row[0]) == zone
                for row in bmap.get("zone_center", [])
            )
            if not known:
                needs.append(Need(
                    kind="know_zone",
                    consumer_step_index=idx,
                    source_step=step,
                    payload={"zone": zone},
                    rationale=f"MoveTo({zone}) requires known zone center",
                ))
            continue  # El contrato de MoveTo es para X, Y numericos.

        # --- Deteccion basada en contrato ---
        contract = CONTRACT_REGISTRY.get(name)
        if contract is None:
            continue

        # requires: beliefs que el step necesita.
        for req in contract.requires:
            if req.functor in _META_FUNCTORS:
                continue
            need = _require_to_need(req, name, args, bmap, idx, step)
            if need is not None:
                needs.append(need)

        # may_observe: beliefs que el step puede descubrir.
        # Si la accion es exploratoria y el belief objetivo no se conoce,
        # emitir un need (la accion misma es el mecanismo de descubrimiento).
        for obs in contract.may_observe:
            need = _observe_to_need(obs, name, args, bmap, idx, step)
            if need is not None:
                needs.append(need)

    return needs


# ---------------------------------------------------------------------------
# Resolucion de necesidades
# ---------------------------------------------------------------------------

def _find_capability_for_need(need: Need, capabilities: list[Capability]) -> Capability | None:
    for cap in capabilities:
        if cap.kind != need.kind:
            continue
        # Constraints son matches exactos sobre el subconjunto declarado por la capability.
        if any(need.payload.get(k) != v for k, v in cap.constraints.items()):
            continue
        return cap
    return None


def _build_subgoal_step_from_need(need: Need, cap: Capability) -> dict | None:
    if cap.plan_sig == "move_to_and_pickup" and need.kind == "inventory_at_least":
        item = str(need.payload.get("item", "")).strip()
        qty = int(need.payload.get("qty", 1))
        if not item:
            return None
        return {
            "type": "subgoal",
            "name": "move_to_and_pickup",
            "args": [item, max(1, qty)],
            "description": f"Obtain at least {max(1, qty)} units of {item}.",
        }

    if cap.plan_sig == "achieve_explore_zone" and need.kind == "know_zone":
        zone = str(need.payload.get("zone", "")).strip()
        if not zone:
            return None
        return {
            "type": "subgoal",
            "name": "achieve_explore_zone",
            "args": [zone],
            "description": f"Ensure the center coordinates of zone {zone} are known.",
        }

    if cap.plan_sig == "craft_item" and need.kind == "craft_at_zone":
        recipe_id = str(need.payload.get("recipe_id", "")).strip()
        qty = int(need.payload.get("qty", 1))
        if not recipe_id:
            return None
        return {
            "type": "subgoal",
            "name": "craft_item",
            "args": [recipe_id, max(1, qty)],
            "description": f"Navigate to crafting zone and craft {max(1, qty)} unit(s) using {recipe_id}.",
        }

    # Fallback generico: construir subgoal desde param_names y payload.
    args: list[object] = []
    for pname in cap.param_names:
        if pname in need.payload:
            args.append(need.payload[pname])
    if not args and cap.param_names:
        return None
    return {
        "type": "subgoal",
        "name": cap.plan_sig,
        "args": args,
        "description": f"Resolve need {need.kind} using {cap.plan_sig}.",
    }


# ---------------------------------------------------------------------------
# Generacion LLM de subgoal
# ---------------------------------------------------------------------------

async def _generate_subgoal_with_llm(
    goal_name: str,
    need: Need,
    existing_subgoals: list[str],
    llm_call: Callable[[str, str], Awaitable[str]],
    *,
    entity_catalog: dict | None = None,
) -> tuple[dict | None, dict | None]:
    """Genera un sub-goal para una necesidad sin capability via LLM.

    Returns:
      (subgoal_step, created_plan_meta) or (None, None) on failure.
    """
    from llm.prompts.planning import build_prompt
    from llm.parser import parse_llm_response

    payload = {
        "task": "step5_need_plan",
        "goal_name": goal_name,
        "need_kind": need.kind,
        "need_payload": need.payload,
        "need_rationale": need.rationale,
        "existing_subgoals": existing_subgoals,
        "entity_catalog": entity_catalog or {},
    }

    raw = await llm_call(*build_prompt(payload))
    parsed = parse_llm_response(raw, "step5_need_plan")
    if not isinstance(parsed, dict):
        return None, None

    subgoal = parsed.get("subgoal", {})
    if not isinstance(subgoal, dict):
        return None, None

    sig = str(subgoal.get("sig", "")).strip()
    if not sig:
        return None, None
    if sig == goal_name:
        return None, None
    if sig in set(existing_subgoals):
        return None, None

    args = subgoal.get("args", [])
    if not isinstance(args, list):
        args = []
    description = str(subgoal.get("description", "")).strip() or f"Resolve need {need.kind}."

    step = {
        "type": "subgoal",
        "name": sig,
        "args": args,
        "description": description,
    }

    created_meta = {
        "sig": sig,
        "description": description,
        "need_kind": need.kind,
        "need_payload": dict(need.payload),
    }
    return step, created_meta


# ---------------------------------------------------------------------------
# Orquestador de refinamiento
# ---------------------------------------------------------------------------

async def run_step5_refine(
    goal_name: str,
    steps: list[dict],
    existing_subgoals: list[str],
    *,
    llm_call: Callable[[str, str], Awaitable[str]] | None = None,
    entity_catalog: dict | None = None,
    capability_contracts: dict | None = None,
    beliefs: dict | None = None,
    max_iterations: int = 2,
) -> RefinementResult:
    """Refina steps validados insertando subplanes reutilizables para needs abiertas.

    Politica: reuse-first. Solo invoca LLM si no hay capability conocida.
    """
    if not steps:
        return RefinementResult(steps=[])

    local_steps = list(steps)
    resolved: list[Need] = []
    unresolved: list[Need] = []
    created_plans: list[dict] = []
    seen_need_keys: set[tuple] = set()

    capabilities = _build_capability_index(
        existing_subgoals,
        capability_contracts=capability_contracts,
    )

    belief_snapshot = beliefs or {}

    for _ in range(max_iterations):
        needs = _detect_unmet_needs(local_steps, belief_snapshot)
        if not needs:
            break

        progress = False
        insertion_offset = 0

        for need in needs:
            key = _to_need_key(need)
            if key in seen_need_keys:
                continue
            seen_need_keys.add(key)

            cap = _find_capability_for_need(need, capabilities)
            if cap is None:
                # Need kinds que requieren un builtin plan específico — no delegar al LLM.
                if need.kind in _NO_LLM_FALLBACK_KINDS:
                    unresolved.append(need)
                    continue
                # Reuse-first: si no hay capability conocida, generar con LLM.
                if llm_call is None:
                    unresolved.append(need)
                    continue
                generated_step, created_meta = await _generate_subgoal_with_llm(
                    goal_name,
                    need,
                    existing_subgoals,
                    llm_call,
                    entity_catalog=entity_catalog,
                )
                if generated_step is None:
                    unresolved.append(need)
                    continue

                insert_at = max(0, min(len(local_steps), need.consumer_step_index + insertion_offset))
                local_steps.insert(insert_at, generated_step)
                insertion_offset += 1
                resolved.append(need)
                progress = True
                if created_meta is not None:
                    created_meta.setdefault("provides", [
                        {
                            "kind": need.kind,
                            "constraints": dict(need.payload),
                        }
                    ])
                    created_meta.setdefault("param_names", [f"arg{i}" for i, _ in enumerate(generated_step.get("args", []))])
                    created_plans.append(created_meta)
                    created_sig = str(created_meta.get("sig", "")).strip()
                    if created_sig:
                        existing_subgoals.append(created_sig)
                        capabilities.extend(
                            _build_capability_index(
                                [created_sig],
                                capability_contracts={
                                    "contracts": {
                                        created_sig: {
                                            "provides": created_meta.get("provides", []),
                                            "param_names": created_meta.get("param_names", []),
                                        }
                                    }
                                },
                            )
                        )
                log.info(
                    "[STEP5:%s] Need %s resolved by NEW subplan %s before step %d",
                    goal_name,
                    need.kind,
                    generated_step.get("name", "?"),
                    need.consumer_step_index,
                )
                continue

            subgoal = _build_subgoal_step_from_need(need, cap)
            if subgoal is None:
                unresolved.append(need)
                continue

            actual_idx = need.consumer_step_index + insertion_offset
            actual_idx = max(0, min(len(local_steps) - 1, actual_idx))

            # ExploreArea(zone): reemplazar la exploracion atomica con el subgoal reutilizable.
            if need.source_step.get("name") == "ExploreArea":
                local_steps[actual_idx : actual_idx + 1] = [subgoal]
                resolved.append(need)
                progress = True
                log.info(
                    "[STEP5:%s] Need %s resolved by %s — replaced ExploreArea at step %d",
                    goal_name,
                    need.kind,
                    cap.plan_sig,
                    actual_idx,
                )
                continue

            # PickUp: reemplazar la secuencia [MoveTo?, Search?, PickUp] con el subgoal.
            if need.source_step.get("name") == "PickUp":
                item_id = str(need.payload.get("item", "")).strip()
                start_idx = actual_idx
                while start_idx > 0:
                    prev = local_steps[start_idx - 1]
                    prev_name = prev.get("name", "")
                    if prev_name == "Search" and prev.get("args") == [item_id]:
                        start_idx -= 1
                    elif prev_name == "MoveTo":
                        start_idx -= 1
                    else:
                        break

                n_replaced = actual_idx - start_idx + 1
                local_steps[start_idx : actual_idx + 1] = [subgoal]
                insertion_offset += 1 - n_replaced
                resolved.append(need)
                progress = True
                log.info(
                    "[STEP5:%s] Need %s resolved by %s — replaced %d atomic step(s) at [%d..%d]",
                    goal_name,
                    need.kind,
                    cap.plan_sig,
                    n_replaced,
                    start_idx,
                    actual_idx,
                )
                continue

            # craft_at_zone: reemplazar el Craft primitivo con craft_item subgoal.
            # craft_item encapsula navegar a la zona + craftear; dejar .Craft además
            # causaría que el NPC intente craftear dos veces (la segunda sin ingredientes).
            if need.kind == "craft_at_zone" and need.source_step.get("type") == "action":
                local_steps[actual_idx : actual_idx + 1] = [subgoal]
                resolved.append(need)
                progress = True
                log.info(
                    "[STEP5:%s] Need %s resolved by %s — replaced Craft at step %d",
                    goal_name,
                    need.kind,
                    cap.plan_sig,
                    actual_idx,
                )
                continue

            # Default: insertar subgoal antes del step consumidor.
            insert_at = max(0, min(len(local_steps), actual_idx))

            if insert_at > 0 and local_steps[insert_at - 1] == subgoal:
                resolved.append(need)
                continue

            local_steps.insert(insert_at, subgoal)
            insertion_offset += 1
            resolved.append(need)
            progress = True

            log.info(
                "[STEP5:%s] Need %s resolved by %s before step %d",
                goal_name,
                need.kind,
                cap.plan_sig,
                need.consumer_step_index,
            )

        if not progress:
            break

    return RefinementResult(
        steps=local_steps,
        resolved_needs=resolved,
        unresolved_needs=unresolved,
        created_plans=created_plans,
    )
