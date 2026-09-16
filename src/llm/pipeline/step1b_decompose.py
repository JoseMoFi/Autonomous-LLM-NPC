from __future__ import annotations

"""Paso 1b: descompone success_condition en variantes de escalera.

Dado success_condition = "has_item(bread, 1)" y los recipes del NPC, construye
una lista de VariantSpec en orden de escalera (dependencia más profunda primero):

  Ej. para bread (recipe: wheat×2 @ bakeri):

    1. not has_item(bread,1) & not has_item(wheat,2)
       → "NPC doesn't have wheat (2 needed). Must gather it."

    2. not has_item(bread,1) & has_item(wheat,2) & not at_zone(bakeri)
       → "NPC has wheat but is not at bakeri. Must move there."

    3. not has_item(bread,1) & has_item(wheat,2) & at_zone(bakeri)
       → "NPC has all ingredients at the right zone. Ready to craft."

Sólo aplica a goals de la forma has_item(X, N) donde existe una recipe que
produce X. Para cualquier otro goal devuelve [] y el pipeline cae al
comportamiento de variante única.

Fase 17 — coordinación planificada por el LLM (`coordination=True`):
  - Un item que hace falta (el propio objetivo o un ingrediente) y que el NPC no
    puede recolectar (sin spawn) ni fabricar (sin receta propia) se parte en DOS
    peldaños: conseguir que otro NPC lo produzca y lo entregue
    (`not peer_item_available(_, Item, _, _, _)`) y recogerlo donde lo dejó
    (`peer_item_available(P, Item, Q, X, Y)`, con P/Q/X/Y ligadas para el cuerpo).
    La NECESIDAD de ayuda la detecta este andamiaje, igual que detecta los
    ingredientes; CÓMO pedirla y recogerla lo planifica el LLM.
  - `delivered_to_peer(R, Item, Qty)` (goal que un NPC adopta al aceptar un
    request): la escalera para conseguir `Item` con el prefijo
    `not delivered_to_peer(R, Item, Qty)` y un peldaño final de ENTREGA con cuerpo
    fijo (`.drop` + `.deliver_to_peer`, andamiaje CODE: es el compromiso del
    protocolo, no una decisión de plan).
Sin coordinación y con has_item(X, N) la salida es idéntica a la de antes.

Fase 17f — sin sub-planes escritos a mano (`builtin_subplans=False`, brazo ATOM):
  recolectar un item con spawn (el objetivo, un ingrediente o lo que se entrega)
  se desgrana en los `requires` del contrato de PickUp (`item_at(ItemId, X, Y)`,
  `current_position(X, Y)`), una precondición por peldaño y en su orden: item no
  observado → observado pero el NPC no está en su casilla → en la casilla. X/Y
  los liga el guard y el cuerpo puede usarlos. Es la estructura que en SUB aporta
  `move_to_and_pickup.asl` (reglas A/B); los cuerpos siguen siendo del LLM. Con
  sub-planes la salida no cambia.
"""

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_HAS_ITEM_RE = re.compile(r"^has_item\(\s*(\w+)\s*,\s*(\d+)\s*\)$")
_DELIVERED_RE = re.compile(r"^delivered_to_peer\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*\)$")


@dataclass
class VariantSpec:
    variant_guard: str          # guard completo para esta variante
    problem_nl: str             # descripción NL del problema a resolver
    known_facts: list[str] = field(default_factory=list)
    # Condiciones positivas ya satisfechas en este guard (p.ej. ["has_item(wheat, 2)"]).
    # Se pasan a step3 para que el LLM NO genere pasos que ya están garantizados.
    already_satisfied: list[str] = field(default_factory=list)
    # La única condición negada que ESTA variante debe resolver (átomo concreto).
    # Si se pasa a step3, se usa como "Failing condition" en lugar del guard completo,
    # evitando que la regla de completitud dispare para otras condiciones del guard.
    unsatisfied_condition: str = ""
    # Fase 17: variables que el guard liga y que el cuerpo puede usar (P, Q, X, Y
    # de peer_item_available), y cuerpo FIJO de andamiaje (no pasa por el LLM).
    bound_variables: list[str] = field(default_factory=list)
    fixed_steps: list[dict] = field(default_factory=list)


def build_need_variants(
    success_condition: str,
    recipes: list[dict],
    item_spawns: list[dict],
    *,
    builtin_subplans: bool = True,
    coordination: bool = False,
    peers: list | None = None,
) -> list[VariantSpec]:
    """Devuelve la escalera de VariantSpec o [] si no aplica el patrón."""
    condition = success_condition.strip()
    peer_rows = _normalize_peers(peers) if coordination else []

    m_deliver = _DELIVERED_RE.match(condition)
    if m_deliver:
        if not coordination:
            return []
        return _delivery_ladder(
            m_deliver.group(1).lower(), m_deliver.group(2).lower(), int(m_deliver.group(3)),
            recipes, item_spawns, builtin_subplans=builtin_subplans, peers=peer_rows,
        )

    m = _HAS_ITEM_RE.match(condition)
    if not m:
        return []

    target_item = m.group(1).lower()
    target_qty = int(m.group(2))

    recipe = _find_recipe_for_item(target_item, target_qty, recipes)
    if recipe is None:
        # Fase 17: el objetivo mismo solo puede darlo otro NPC → pedir + recoger.
        if peer_rows and not _build_spawn_map(item_spawns).get(target_item) \
                and not _has_recipe_for(target_item, recipes):
            variants = _peer_variants(
                target_item, target_qty, [], peer_rows, builtin_subplans=builtin_subplans,
                self_note=_self_sufficiency_note(target_item, recipes, item_spawns),
            )
            log.info(
                "[STEP1B] '%s' → %d variantes (item solo disponible vía otro NPC)",
                success_condition, len(variants),
            )
            return variants
        spawn_zones = _build_spawn_map(item_spawns).get(target_item, [])
        if not builtin_subplans and spawn_zones:
            variants = _pickup_precondition_rungs(
                target_item, target_qty, [f"not has_item({target_item}, {target_qty})"], spawn_zones,
            )
            log.info(
                "[STEP1B] '%s' → %d variantes (precondiciones de PickUp, sin sub-planes)",
                success_condition, len(variants),
            )
            return variants
        log.debug("[STEP1B] No hay recipe que produzca '%s' — variante única", target_item)
        return []

    return _recipe_ladder(
        success_condition, target_item, target_qty, recipe, recipes, item_spawns,
        prefix_parts=[], builtin_subplans=builtin_subplans, peers=peer_rows,
    )


def _recipe_ladder(
    success_condition: str,
    target_item: str,
    target_qty: int,
    recipe: dict,
    recipes: list[dict],
    item_spawns: list[dict],
    *,
    prefix_parts: list[str],
    builtin_subplans: bool,
    peers: list[tuple[str, str]],
) -> list[VariantSpec]:
    """Escalera de receta: ingredientes → zona → craftear. `prefix_parts` va
    delante de cada guard (Fase 17: `not delivered_to_peer(...)` en entregas)."""
    recipe_id: str = recipe["recipeId"]
    craft_zone: str = recipe["zone"]
    inputs: list[dict] = recipe.get("inputs", [])
    # Fase 17k: unidades de UNA hornada. Si no llegan al objetivo (A5: 3 panes con
    # una receta de 1), la escalera se re-inyecta y se craftea varias veces.
    out_qty = _output_qty(recipe, target_item)
    repeat_note = (
        f" One craft of {recipe_id} produces {out_qty} {target_item}; the goal needs {target_qty}, "
        f"so this ladder runs again (gather, move, craft) until the NPC has {target_qty}."
        if 0 < out_qty < target_qty else ""
    )

    not_has_target = f"not has_item({target_item}, {target_qty})"
    neg_goal = " & ".join(list(prefix_parts) + [not_has_target])

    variants: list[VariantSpec] = []
    spawn_map = _build_spawn_map(item_spawns)

    # ── Escalera: una variante por cada ingrediente faltante ─────────────────
    # Construimos la "cadena de contexto positivo" de los ingredientes ya listos.
    # Para cada ingrediente i, la variante es:
    #   not goal & has(ing_0) & … & has(ing_{i-1}) & not has(ing_i)
    # seguida por la variante "happy path" al final.

    positive_so_far: list[str] = []   # ingredientes ya satisfechos en guardia

    for inp in inputs:
        item_id = inp.get("itemId", "").lower()
        qty = int(inp.get("qty", 1))
        if not item_id:
            continue

        has_pred = f"has_item({item_id}, {qty})"
        not_has_pred = f"not has_item({item_id}, {qty})"
        spawn_zones = spawn_map.get(item_id, [])

        # Fase 17: ingrediente que solo puede dar otro NPC → pedir + recoger.
        if peers and not spawn_zones and not _has_recipe_for(item_id, recipes):
            variants.extend(_peer_variants(
                item_id, qty, [neg_goal] + positive_so_far, peers,
                purpose=f" to craft {target_item}", already_satisfied=positive_so_far,
                builtin_subplans=builtin_subplans,
                self_note=_self_sufficiency_note(item_id, recipes, item_spawns),
            ))
            positive_so_far.append(has_pred)
            continue

        # Guardia: neg_goal + ingredientes previos listos + este ingrediente faltante
        parts = [neg_goal] + positive_so_far + [not_has_pred]

        # Fase 17f: sin sub-planes, recolectar se desgrana en las precondiciones de PickUp.
        if not builtin_subplans and spawn_zones:
            variants.extend(_pickup_precondition_rungs(
                item_id, qty, parts, spawn_zones,
                purpose=f" to craft {target_item}", already_satisfied=positive_so_far,
            ))
            positive_so_far.append(has_pred)
            continue

        variant_guard = " & ".join(parts)

        spawn_hint = (
            f"It can be gathered at {spawn_zones[0]}." if spawn_zones else
            "Its spawn location is unknown."
        )
        problem_nl = (
            f"The NPC does not have {item_id} (needs {qty}). {spawn_hint} "
            f"Gather {qty} {item_id} before crafting {target_item}."
        )
        known_facts_for_variant = [f"item_spawn({item_id}, {z})" for z in spawn_zones]

        variants.append(VariantSpec(
            variant_guard=variant_guard,
            problem_nl=problem_nl,
            known_facts=known_facts_for_variant,
            already_satisfied=list(positive_so_far),
            unsatisfied_condition=not_has_pred,
        ))

        positive_so_far.append(has_pred)

    # ── Variante de zona: todos los ingredientes disponibles, zona incorrecta ─
    all_ingredients_positive = " & ".join(positive_so_far)
    not_at_zone = f"not at_zone({craft_zone})"

    if positive_so_far:
        zone_guard_parts = [neg_goal] + positive_so_far + [not_at_zone]
    else:
        zone_guard_parts = [neg_goal, not_at_zone]

    if builtin_subplans:
        zone_problem_nl = (
            f"The NPC has all required ingredients but is not at {craft_zone}. "
            f"Use !craft_item({recipe_id}, {target_qty}) as the ONLY step — "
            f"it handles zone discovery, navigation, and crafting internally. "
            f"Do NOT add MoveTo, ExploreArea, achieve_explore_zone, or raw Craft actions."
        )
    else:
        # Fase 16 (ablación de sub-planes): sin craft_item el problema se describe
        # sin prescribir acciones — decide el LLM cómo llegar a la zona y craftear.
        zone_problem_nl = (
            f"The NPC has all required ingredients but is not at {craft_zone}. "
            f"It must be inside {craft_zone} before it can craft {target_item} with {recipe_id}."
        )
    variants.append(VariantSpec(
        variant_guard=" & ".join(zone_guard_parts),
        problem_nl=zone_problem_nl,
        known_facts=[f"recipe_output({recipe_id}, {craft_zone}, {target_item}, {out_qty})"],
        already_satisfied=list(positive_so_far),
        unsatisfied_condition=not_at_zone,
    ))

    # ── Happy path: ingredientes + zona correcta → craftear ──────────────────
    at_zone = f"at_zone({craft_zone})"
    if positive_so_far:
        hp_parts = [neg_goal] + positive_so_far + [at_zone]
    else:
        hp_parts = [neg_goal, at_zone]

    variants.append(VariantSpec(
        variant_guard=" & ".join(hp_parts),
        problem_nl=(
            f"The NPC has all ingredients ({all_ingredients_positive or 'none required'}) "
            f"and is at {craft_zone}. Use {recipe_id} to craft {target_item}."
            + repeat_note
        ),
        known_facts=[
            f"recipe_output({recipe_id}, {craft_zone}, {target_item}, {out_qty})",
        ] + [
            f"recipe_input({recipe_id}, {inp['itemId'].lower()}, {inp['qty']})"
            for inp in inputs if inp.get("itemId")
        ],
        already_satisfied=list(positive_so_far) + [f"at_zone({craft_zone})"],
        unsatisfied_condition=not_has_target,
    ))

    log.info(
        "[STEP1B] '%s' → %d variantes en escalera (recipe=%s, zone=%s)",
        success_condition, len(variants), recipe_id, craft_zone,
    )
    return variants


def _delivery_ladder(
    requester: str,
    item: str,
    qty: int,
    recipes: list[dict],
    item_spawns: list[dict],
    *,
    builtin_subplans: bool,
    peers: list[tuple[str, str]],
) -> list[VariantSpec]:
    """Fase 17: escalera de `delivered_to_peer(R, Item, Qty)` — conseguir el item
    (receta, recolección u otro NPC) y un peldaño final de entrega de andamiaje."""
    condition = f"delivered_to_peer({requester}, {item}, {qty})"
    prefix = f"not {condition}"
    not_has = f"not has_item({item}, {qty})"
    spawn_map = _build_spawn_map(item_spawns)

    recipe = _find_recipe_for_item(item, qty, recipes)
    if recipe is not None:
        obtain = _recipe_ladder(
            condition, item, qty, recipe, recipes, item_spawns,
            prefix_parts=[prefix], builtin_subplans=builtin_subplans, peers=peers,
        )
    elif spawn_map.get(item) and not builtin_subplans:
        obtain = _pickup_precondition_rungs(
            item, qty, [prefix, not_has], spawn_map[item],
            purpose=f" to hand it over to {requester}",
        )
    elif spawn_map.get(item):
        zones = spawn_map[item]
        obtain = [VariantSpec(
            variant_guard=f"{prefix} & {not_has}",
            problem_nl=(
                f"The NPC does not have {item} (needs {qty}). It can be gathered at {zones[0]}. "
                f"Gather {qty} {item} to hand it over to {requester}."
            ),
            known_facts=[f"item_spawn({item}, {z})" for z in zones],
            unsatisfied_condition=not_has,
        )]
    elif peers:
        obtain = _peer_variants(
            item, qty, [prefix], peers, builtin_subplans=builtin_subplans,
            self_note=_self_sufficiency_note(item, recipes, item_spawns),
        )
    else:
        return []

    handover = VariantSpec(
        variant_guard=f"{prefix} & has_item({item}, {qty})",
        problem_nl=f"The NPC has the {qty} {item} requested by {requester} and hands it over.",
        already_satisfied=[f"has_item({item}, {qty})"],
        unsatisfied_condition=prefix,
        fixed_steps=[
            {"type": "action", "name": "Drop", "args": [item, qty]},
            {"type": "action", "name": "deliver_to_peer", "args": [requester, item, qty]},
        ],
    )
    log.info("[STEP1B] '%s' → %d variantes (entrega a otro NPC)", condition, len(obtain) + 1)
    return obtain + [handover]


def _peer_variants(
    item: str,
    qty: int,
    base_parts: list[str],
    peers: list[tuple[str, str]],
    *,
    purpose: str = "",
    already_satisfied: list[str] | tuple = (),
    builtin_subplans: bool = True,
    self_note: str = "",
) -> list[VariantSpec]:
    """Fase 17: peldaños para un item que solo puede dar otro NPC (pedir + recoger).

    Fase 17o (sin sub-planes, ATOM): el protocolo se desgrana en peldaños, del
    estado más avanzado al menos, como las precondiciones de PickUp (17f):
    recoger (entregado) → esperar (prometido) → reintentar (la petición falló) →
    pedir (nadie lo ha prometido). En SUB esa secuencia la aporta
    obtain_from_peer (pedir → esperar → recoger en un solo cuerpo) y los peldaños
    no cambian.
    """
    not_has = f"not has_item({item}, {qty})"
    peer_ids = ", ".join(pid for pid, _role in peers)
    peer_facts = [f"peer({pid}, {role})" for pid, role in peers]
    request = VariantSpec(
        variant_guard=" & ".join(
            list(base_parts) + [not_has, f"not peer_item_available(_, {item}, _, _, _)"]
        ),
        problem_nl=(
            f"The NPC needs {qty} {item}{purpose} but cannot obtain it by itself: {item} does not "
            f"spawn anywhere and the NPC knows no recipe that produces it. Other NPCs it can ask: "
            f"{peer_ids}. This branch must get one of them to produce the {item} and deliver it; "
            f"picking it up once delivered is handled by a separate branch."
        ),
        known_facts=list(peer_facts),
        already_satisfied=list(already_satisfied),
        unsatisfied_condition=not_has,
    )
    collect = VariantSpec(
        variant_guard=" & ".join(
            list(base_parts) + [not_has, f"peer_item_available(P, {item}, Q, X, Y)"]
        ),
        problem_nl=(
            f"Another NPC (P) has delivered the {item}: Q units were dropped on the ground at "
            f"coordinates (X, Y). The NPC must pick up the {item} there."
        ),
        known_facts=list(peer_facts),
        already_satisfied=list(already_satisfied),
        unsatisfied_condition=not_has,
        bound_variables=["P", "Q", "X", "Y"],
    )
    if builtin_subplans:
        return [request, collect]

    # Fase 17o: en ATOM el peldaño de pedir mostraba `not has_item(item)` como
    # condición y el LLM se ponía a recolectar (piloto CO5/ATOM); y como su guard
    # seguía siendo cierto tras el agree, se re-inyectaba y volvía a pedir (CO6).
    no_delivery = f"not peer_item_available(_, {item}, _, _, _)"
    wait = VariantSpec(
        variant_guard=" & ".join(list(base_parts) + [
            not_has, no_delivery, "peer_promised(P, G)", "not peer_failed(P, G, _)",
        ]),
        problem_nl=(
            f"Another NPC (P) agreed to produce the {item}{purpose} (request G) and has not "
            f"delivered it yet. peer_item_available(P, {item}, Q, X, Y) only becomes true when P "
            f"reports request G as done.{self_note}"
        ),
        known_facts=list(peer_facts),
        already_satisfied=list(already_satisfied),
        unsatisfied_condition=f"not peer_item_available(P, {item}, _, _, _)",
        bound_variables=["P", "G"],
    )
    retry = VariantSpec(
        variant_guard=" & ".join(list(base_parts) + [
            not_has, no_delivery, "peer_promised(P, G)", "peer_failed(P, G, R)",
        ]),
        problem_nl=(
            f"The NPC needs {qty} {item}{purpose}. Its request G to P failed (reason R), so nobody "
            f"is producing the {item} now. Other NPCs it can ask: {peer_ids}. This branch only has "
            f"to get one of them to agree again; waiting for the delivery and picking it up are "
            f"handled by separate branches.{self_note}"
        ),
        known_facts=list(peer_facts),
        already_satisfied=list(already_satisfied),
        unsatisfied_condition="not peer_promised(_, _)",
        bound_variables=["P", "G", "R"],
    )
    ask = VariantSpec(
        variant_guard=" & ".join(list(base_parts) + [not_has, no_delivery, "not peer_promised(_, _)"]),
        problem_nl=(
            f"The NPC needs {qty} {item}{purpose} but cannot obtain it by itself: {item} does not "
            f"spawn anywhere and the NPC knows no recipe that produces it. Other NPCs it can ask: "
            f"{peer_ids}. This branch only has to get one of them to agree to produce and deliver "
            f"the {item}; waiting for the delivery and picking it up are handled by separate branches."
            f"{self_note}"
        ),
        known_facts=list(peer_facts),
        already_satisfied=list(already_satisfied),
        unsatisfied_condition="not peer_promised(_, _)",
    )
    return [collect, wait, retry, ask]


def _self_sufficiency_note(item: str, recipes: list[dict], item_spawns: list[dict]) -> str:
    """Fase 17p (ATOM): hechos del estado para los peldaños de pedir/esperar/reintentar.

    En el piloto de la Fase 17o el baker, en el peldaño de pedir harina, se ponía a
    recolectar trigo (asocia harina con trigo) aunque no tiene ninguna receta que lo
    use. Se le dice qué puede conseguir por sí mismo y que nada de eso es `item`,
    sin nombrar acciones: el cuerpo lo sigue decidiendo el LLM.
    """
    gatherable = sorted(_build_spawn_map(item_spawns))
    produced = sorted({
        str(output.get("itemId", "")).lower()
        for recipe in recipes for output in recipe.get("outputs", [])
        if output.get("itemId")
    })
    sources = f"Gathering {', '.join(gatherable)} or crafting" if gatherable else "Crafting"
    recipes_text = (
        f"none of its recipes produces {item} (they produce {', '.join(produced)})"
        if produced else "it knows no recipes"
    )
    return f" {sources} cannot give this NPC {item}: {recipes_text}. Only another NPC can provide the {item}."


def _pickup_precondition_rungs(
    item: str,
    qty: int,
    base_parts: list[str],
    spawn_zones: list[str],
    *,
    purpose: str = "",
    already_satisfied: list[str] | tuple = (),
) -> list[VariantSpec]:
    """Fase 17f: peldaños para recolectar `item`, derivados de los `requires`
    del contrato de PickUp (protocol.action_semantics) y en su orden.

    Con requires = [item_at(ItemId, X, Y), current_position(X, Y)] devuelve, en
    este orden (del estado más avanzado al menos):
      1. base & item_at(wheat, X, Y) & current_position(X, Y)      (en la casilla)
      2. base & item_at(wheat, X, Y) & not current_position(X, Y)  (visto, lejos)
      3. base & not item_at(wheat, _, _)                           (sin observar)

    El orden importa: agentspeak elige el PRIMER plan aplicable y, con varios
    ejemplares observados, 1 y 2 aplican a la vez (en la casilla de uno, 2 liga
    X/Y a otro). Con 2 antes que 1 el NPC oscilaba entre casillas sin recoger
    nunca (test_integration_atom_gather_ladder). Es el mismo orden que
    move_to_and_pickup.asl (regla A antes que B).

    Cada peldaño resuelve UNA precondición y el de la casilla el has_item. Las variables
    que ligan los literales positivos del guard pasan al cuerpo (bound_variables);
    bajo `not`, una variable aún sin ligar se escribe `_`. El problema se describe
    con el contrato y el estado, sin prescribir acciones: el cuerpo lo escribe el LLM.
    """
    from protocol.action_semantics import CONTRACT_REGISTRY

    contract = CONTRACT_REGISTRY.get("PickUp")
    requires = [
        spec for spec in (contract.requires if contract else [])
        if spec.functor not in ("binding", "constant")
    ]
    if not requires:
        return []

    def _args(spec, bound: set[str] | None) -> str:
        out = []
        for arg in spec.args:
            if arg == "ItemId":
                out.append(item)
            elif bound is not None and arg not in bound:
                out.append("_")
            else:
                out.append(arg)
        return ", ".join(out)

    all_requires = ", ".join(f"{s.functor}({_args(s, None)})" for s in requires)
    intro = f"The NPC needs {qty} {item}{purpose}. PickUp({item}) requires {all_requires}."
    # Fase 17l: el LLM buscaba el trigo en el centro de la PANADERÍA (la zona de
    # crafteo del goal) y la escalera se atascaba sin recoger nada (tanda corta 2, A3/A5).
    spawn_hint = (
        f" {item} spawns at {spawn_zones[0]}: the NPC must be inside {spawn_zones[0]} "
        f"to observe it, not at a crafting zone."
        if spawn_zones else ""
    )
    spawn_facts = [f"item_spawn({item}, {z})" for z in spawn_zones]

    rungs: list[VariantSpec] = []
    positives: list[str] = []
    bound: list[str] = []
    for spec in requires:
        negated = f"not {spec.functor}({_args(spec, set(bound))})"
        if positives:
            state = (
                f" In this branch {' and '.join(positives)} hold (with {', '.join(bound)} "
                f"bound by the guard), but {negated[4:]} does not."
            )
        else:
            state = f" In this branch {negated[4:]} does not hold.{spawn_hint}"
        rungs.append(VariantSpec(
            variant_guard=" & ".join(list(base_parts) + positives + [negated]),
            problem_nl=intro + state,
            known_facts=list(spawn_facts),
            already_satisfied=list(already_satisfied),
            unsatisfied_condition=negated,
            bound_variables=list(bound),
        ))
        positives.append(f"{spec.functor}({_args(spec, None)})")
        bound.extend(a for a in spec.args if a != "ItemId" and a[:1].isupper() and a not in bound)

    rungs.append(VariantSpec(
        variant_guard=" & ".join(list(base_parts) + positives),
        problem_nl=(
            f"{intro} In this branch all of them hold (with {', '.join(bound)} bound by the guard)."
        ),
        known_facts=list(spawn_facts),
        already_satisfied=list(already_satisfied),
        unsatisfied_condition=f"not has_item({item}, {qty})",
        bound_variables=list(bound),
    ))
    rungs.reverse()
    return rungs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_recipe_for_item(
    item_id: str,
    min_qty: int,
    recipes: list[dict],
) -> dict | None:
    """Receta que produce item_id: la primera cuya hornada da >= min_qty y, si
    ninguna llega, la primera que lo produce. Fase 17k: antes devolvía None con
    has_item(bread, 3) y una receta de 1 pan, y el goal se planificaba sin escalera;
    ahora se craftea varias veces (la escalera se re-inyecta hasta el objetivo)."""
    producing = [recipe for recipe in recipes if _output_qty(recipe, item_id) > 0]
    for recipe in producing:
        if _output_qty(recipe, item_id) >= min_qty:
            return recipe
    return producing[0] if producing else None


def _output_qty(recipe: dict, item_id: str) -> int:
    """Unidades de item_id que produce UNA ejecución de la receta."""
    return sum(
        int(output.get("qty", 0))
        for output in recipe.get("outputs", [])
        if output.get("itemId", "").lower() == item_id
    )


def _has_recipe_for(item_id: str, recipes: list[dict]) -> bool:
    """True si alguna receta del NPC produce item_id (cualquier cantidad)."""
    return any(
        output.get("itemId", "").lower() == item_id
        for recipe in recipes
        for output in recipe.get("outputs", [])
    )


def _normalize_peers(peers: list | None) -> list[tuple[str, str]]:
    """[(npc_id, role), ...] a partir de filas del belief peer/2 o de ids sueltos."""
    rows: list[tuple[str, str]] = []
    for peer in peers or []:
        if isinstance(peer, str):
            rows.append((peer.lower(), "unknown"))
        elif isinstance(peer, (list, tuple)) and peer:
            role = str(peer[1]).lower() if len(peer) > 1 else "unknown"
            rows.append((str(peer[0]).lower(), role))
    return rows


def _build_spawn_map(item_spawns: list[dict]) -> dict[str, list[str]]:
    """itemId → [zone, ...] desde item_spawns del perfil."""
    spawn_map: dict[str, list[str]] = {}
    for spawn in item_spawns:
        item_id = spawn.get("itemId", "").lower()
        # "none" es el centinela de Unity para items que NO se recolectan del mundo
        # (flour, bread: craft-only, targetCount 0). unity_events ya lo ignora al crear
        # item_spawn; aquí se tomaba como zona real, flour parecía recolectable y no se
        # generaban los peldaños de pedirlo a otro NPC (tanda corta, A5).
        zones = [
            z for z in spawn.get("zones", [])
            if str(z).strip() and str(z).strip().lower() != "none"
        ]
        if item_id and zones:
            spawn_map.setdefault(item_id, [])
            for z in zones:
                if z not in spawn_map[item_id]:
                    spawn_map[item_id].append(z)
    return spawn_map
