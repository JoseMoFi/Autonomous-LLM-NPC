from __future__ import annotations

"""family_plan.py — generador determinista de la familia paramétrica
`achieve_has_item(Item, Qty)` (Fase 6.5, T4).

A diferencia de `step1b_decompose` (que produce una escalera GROUND por binding
a través del pipeline LLM), este módulo construye UN plan paramétrico que sirve a
TODOS los bindings de la familia `has_item`, enrutado por **estado del mundo**:

    +!achieve_has_item(Item, Qty) : has_item(Item, Qty) <- true.                       // done
    +!achieve_has_item(Item, Qty) : not has_item(Item, Qty) & item_spawn(Item, _)      // gather (genérico)
        <- !move_to_and_pickup(Item, Qty).
    +!achieve_has_item(Item, Qty) : not has_item(Item, Qty) & recipe_output(R,_,Item,_) // craft (por receta)
        <- !achieve_has_item(in1, q1); ...; !craft_item(R, Qty).

Claves del diseño:
  - **Enrutado por guard**: un item spawneable cae por `gather`; un item con receta
    cae por `craft` (el guard `recipe_output(R,_,Item,_)` solo casa cuando `Item` es
    el output de esa receta). Conjuntos completos y disjuntos por estado del mundo.
  - **Reuso recursivo**: la rama `craft` reúne cada ingrediente con
    `!achieve_has_item(Input, IQty)`, que vuelve a entrar en la familia (gather o
    craft según el ingrediente). Así `bread` reúne `wheat` recursivamente.
  - **Delega en builtins** ya paramétricos (`move_to_and_pickup`, `craft_item`).
  - **Determinista y sin LLM**: es andamiaje simbólico derivado del MUNDO (como los
    builtins escritos a mano), no contenido del LLM → `source = "CODE"`. Reusar esta
    familia para un goal `has_item` cubierto por el mundo = planificación con CERO
    llamadas a Ollama (la contribución de reuso de la Fase 6.5).

Módulo PURO (sin estado, sin LLM): testeable por compilación ASL y enrutado de
guards. El cableado al runtime (usarla en `_request_plan` bajo el flag canónico)
y la validación e2e van aparte.

Caveat: asume recetas ACÍCLICAS (el grafo de ingredientes no se cierra sobre sí
mismo). Un ciclo de recetas produciría recursión infinita en la rama craft.
"""

from dataclasses import dataclass

FAMILY_SIG = "achieve_has_item"
_HEAD = f"+!{FAMILY_SIG}(Item, Qty)"


@dataclass
class FamilyVariant:
    """Una variante de la familia paramétrica. `source` siempre CODE (andamiaje
    determinista, no contenido del LLM)."""
    guard: str
    asl: str
    source: str = "CODE"


def build_has_item_family(
    recipes: list[dict] | None,
    item_spawns: list[dict] | None = None,
    *,
    coordination_enabled: bool = False,
    peer_request_timeout_s: float = 120.0,
) -> list[FamilyVariant]:
    """Construye las variantes paramétricas de `achieve_has_item(Item, Qty)`.

    Args:
        recipes: lista de recetas del perfil/mundo, cada una
            `{"recipeId": str, "inputs": [{"itemId": str, "qty": int}], ...}`.
        item_spawns: lista de spawns `{"itemId": str, "zones": [...]}`. Solo se usa
            para decidir si emitir la rama genérica de gather (que en runtime se
            enruta por el belief `item_spawn`); su contenido concreto no entra en
            el plan paramétrico.
        coordination_enabled: Fase 12 — si True, añade la variante `delegate`
            (último recurso: ni spawn ni receta propia, pero hay un peer
            conocido → pedírselo). Guardada por `peer(P, _)`: sin peers en el
            mundo, la variante nunca casa (coste cero fuera de la Fase 12).
        peer_request_timeout_s: se inlinea como literal numérico en el cuerpo
            de `.await_peer` (no hay forma de pasar settings a una acción ASL
            sin una variable de cabeza extra; más simple inlinearlo al
            construir la familia).

    Returns:
        Lista de `FamilyVariant` en orden: done → gather → craft (una por
        receta) → delegate (si coordination_enabled).
    """
    variants: list[FamilyVariant] = []

    # 0. done — ya se tiene la cantidad pedida.
    done_guard = "has_item(Item, Qty)"
    variants.append(FamilyVariant(
        guard=done_guard,
        asl=f"{_HEAD} : {done_guard} <- true.",
    ))

    # 1. gather — genérico, enrutado por el belief item_spawn(Item, _) en runtime.
    #    Solo se emite si el mundo tiene algún spawn (si no, ningún item es
    #    recolectable y la rama nunca casaría; se omite para no cargar ruido).
    if item_spawns:
        gather_guard = "not has_item(Item, Qty) & item_spawn(Item, _)"
        variants.append(FamilyVariant(
            guard=gather_guard,
            asl=f"{_HEAD} : {gather_guard} <- !move_to_and_pickup(Item, Qty).",
        ))

    # 2..N. craft — una rama por receta, guardada por su recipe_output.
    for recipe in recipes or []:
        rid = str(recipe.get("recipeId", "")).strip().lower()
        if not rid:
            continue
        inputs = recipe.get("inputs", []) or []
        ensure_steps = [
            f"!{FAMILY_SIG}({str(inp['itemId']).strip().lower()}, {int(inp['qty'])})"
            for inp in inputs
            if inp.get("itemId")
        ]
        craft_guard = f"not has_item(Item, Qty) & recipe_output({rid}, _, Item, _)"
        body = "; ".join(ensure_steps + [f"!craft_item({rid}, Qty)"])
        variants.append(FamilyVariant(
            guard=craft_guard,
            asl=f"{_HEAD} : {craft_guard} <-\n    {body}.",
        ))

    # 3. delegate — último recurso (Fase 12): ni recolectable ni crafteable
    #    por MÍ, pero conozco a un peer → pedírselo. Guardas negativas sobre
    #    LOS MISMOS predicados que gather/craft evalúan (item_spawn,
    #    recipe_output) garantizan que esta rama SOLO casa cuando las otras
    #    dos son imposibles — conjunto disjunto, mismo principio que gather
    #    vs craft (ver cabecera del módulo).
    if coordination_enabled:
        delegate_guard = (
            "not has_item(Item, Qty) & not item_spawn(Item, _) "
            "& not recipe_output(_, _, Item, _) & peer(P, _)"
        )
        delegate_body = (
            f".ask_peer(P, can_make, Item);\n"
            f"    .request_peer(P, achieve_has_item, Item, Qty);\n"
            f"    .await_peer(P, achieve_has_item, {peer_request_timeout_s});\n"
            f"    !collect_from_peer(P, Item, Qty)"
        )
        variants.append(FamilyVariant(
            guard=delegate_guard,
            asl=f"{_HEAD} : {delegate_guard} <-\n    {delegate_body}.",
        ))

    return variants
