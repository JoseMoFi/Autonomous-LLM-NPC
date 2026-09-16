from __future__ import annotations

# PRIMITIVE_ACTIONS y ACTION_ALLOWLIST son la FUENTE DE VERDAD para acciones.
# Este módulo los re-exporta y deriva catálogos de texto usados en prompts.
from protocol.action_contract import PRIMITIVE_ACTIONS, ACTION_ALLOWLIST

# Nombres en ASL (lowercase con punto) → tipo Unity (PascalCase)
ASL_TO_UNITY: dict[str, str] = {
    "moveto":       "MoveTo",
    "explorearea":  "ExploreArea",
    "pickup":       "PickUp",
    "craft":        "Craft",
    "drop":         "Drop",
    "wait":         "Wait",
}

# ---------------------------------------------------------------------------
# Signaturas para el validador semántico step3_validator.
# Derivadas de ACTION_ALLOWLIST.asl_args para evitar duplicación.
#
# Formato: nombre_upper → (min_args, max_args, [nombres_de_arg])
# min_args = len(required) pero usando asl_args como referencia posicional.
# ---------------------------------------------------------------------------
def _build_action_signatures() -> dict[str, tuple[int, int, list[str]]]:
    sigs: dict[str, tuple[int, int, list[str]]] = {}
    for name, spec in ACTION_ALLOWLIST.items():
        asl_args: list[str] = spec.get("asl_args", [])
        required: list[str] = spec.get("required", [])
        # min = número de required que aparecen en asl_args
        min_args = sum(1 for a in asl_args if a in required)
        max_args = len(asl_args)
        sigs[name.upper()] = (min_args, max_args, asl_args)
    return sigs

ACTION_SIGNATURES: dict[str, tuple[int, int, list[str]]] = _build_action_signatures()

# ---------------------------------------------------------------------------
# Catálogo de beliefs. Fuente de verdad para LLM y validadores.
#
# Formato: "predicado(Arg1, Arg2, ...)" → descripción
# Los args en Uppercase son variables numéricas (pueden usarse en guards).
# Los args en lowercase son constantes (nombres de items, zonas, etc.).
# ---------------------------------------------------------------------------
BELIEF_CATALOG: dict[str, str] = {
    "current_position(X, Y)":          "NPC current position on the map. X, Y are integers (grid coordinates). Updated on MoveTo/ExploreArea success or at RegisterNPC.",
    "has_item(ItemId, N)":             "NPC has N units of ItemId in inventory. N is an integer.",
    "zone_center(Tag, X, Y)":          "Zone Tag discovered; center at coordinates X, Y.",
    "knows_zone(Tag)":                 "Zone Tag has been discovered (zone_center is known). Use as simple boolean guard.",
    "at_zone(ZoneTag)":               "NPC is currently inside zone ZoneTag (set by Unity on zone entry, removed on exit). Required precondition for Craft.",
    "item_at(ItemId, X, Y)":          "Item ItemId visible at coordinates X, Y (from Search; expires after ~70 ticks).",
    "delivery_point(Tag, X, Y)":       "Delivery point Tag at coordinates X, Y.",
    "recipe(RecipeId, Zone, In, Out)":  "Recipe RecipeId available in Zone. One predicate per ingredient: In=ingredient item, Out=result item.",
    "recipe_output(RecipeId, Zone, ItemId, Qty)": "Output when crafting RecipeId in Zone: Qty units of ItemId.",
    "item_spawn(ItemId, ZoneTag)":      "Item ItemId can appear in zone ZoneTag.",
}

# Which Unity message type populates each belief predicate.
# Used for documentation and tracing.
BELIEF_SOURCES: dict[str, str] = {
    "current_position": "RegisterNPC.pos (initial seed via gateway/router.py) / ActionResult.payload.current_position (MoveTo or ExploreArea Success). NEVER from the outgoing command.",
    "has_item":       "InventoryUpdate  (sent by Unity after PickUp / Drop / Craft result)",
    "zone_center":    "ZoneDiscovery / NPCProfile / ActionResult (Search: kind=Zone)",
    "knows_zone":     "Derived automatically from zone_center (set together)",
    "at_zone":        "ZoneEntry (sent by Unity when NPC enters a crafting zone; removed on ZoneExit). Prerequisite for Craft.",
    "item_at":        "ActionResult     (Search / WanderSearch payload hits where kind=Item; TTL=70 ticks)",
    "delivery_point": "NPCProfile       (sent once at agent startup: profile.delivery_points)",
    "recipe":         "NPCProfile       (sent once at agent startup: profile.recipes[].inputs)",
    "recipe_output":  "NPCProfile       (sent once at agent startup: profile.recipes[].outputs)",
    "item_spawn":     "NPCProfile       (sent once at agent startup: profile.item_spawns)",
}

# ---------------------------------------------------------------------------
# Leyenda de beliefs para los prompts del pipeline (PIPELINE_PROMPTS Paso 2).
# Se inyecta literalmente en los prompts de guards.
# ---------------------------------------------------------------------------
BELIEFS_LEGEND: str = """\
FACTS — exist or not in the belief base; use them with constant or variable args; no operator:
  current_position(X, Y)        X,Y: integer grid coordinates of the NPC. Updated on MoveTo/ExploreArea success. Always exactly one entry.
  zone_center(Tag, X, Y)        Tag: zone identifier (constant, e.g. farmland); X,Y: coordinates (variables)
  knows_zone(Tag)               Tag: zone identifier (constant). True when zone_center is known. Use for simple guards.
  at_zone(ZoneTag)              ZoneTag: zone identifier (constant). True when the NPC is physically inside that zone. Set by Unity on zone entry; removed on exit. REQUIRED guard before Craft.
  item_at(ItemId, X, Y)         ItemId: item type (constant); X,Y: coordinates (variables). Set by Search; expires after ~70 ticks.
  delivery_point(Tag, X, Y)     Tag: delivery point name (constant); X,Y: coordinates (variables)
  recipe(RecipeId, Zone, In, Out)  RecipeId: recipe name (constant); Zone: zone tag (constant); In,Out: item info. One predicate per ingredient.
  recipe_output(RecipeId, Zone, ItemId, Qty)  Output of craft: RecipeId in Zone produces Qty units of ItemId.
  item_spawn(ItemId, ZoneTag)   ItemId: item type (constant); ZoneTag: zone name (constant)

NUMERIC BELIEFS — bind a variable that can then be compared in a guard:
  has_item(ItemId, N)           ItemId: item type (constant); N: integer count in inventory

NOTE: There are no boolean beliefs. Binary states are modeled as facts
(present or absent). Do not use true/false values.
"""

# ---------------------------------------------------------------------------
# Catálogo de acciones para prompts — DERIVADO de ACTION_ALLOWLIST.
# No editar manualmente: añadir/cambiar acciones en action_contract.py.
#


def _build_actions_catalog() -> str:
    """Genera el catálogo de acciones para prompts a partir de ACTION_ALLOWLIST."""
    from protocol.action_contract import ACTION_ALLOWLIST
    lines = []
    for name, spec in ACTION_ALLOWLIST.items():
        synopsis = spec.get("synopsis", name)
        desc = " ".join(str(spec.get("description", "")).split())
        lines.append(f"{synopsis:<38}-> {desc}")
    return "\n".join(lines) + "\n"


ACTIONS_CATALOG: str = _build_actions_catalog()


# ---------------------------------------------------------------------------
# Fase 17 — acciones de coordinación NPC↔NPC para el pipeline LLM. NO son
# acciones de Unity (no están en ACTION_ALLOWLIST ni en PRIMITIVE_ACTIONS): las
# registra bdi._register_peer_actions y hablan por XMPP con otro NPC. Solo se
# muestran al LLM con coordinación planificada por el LLM.
# ---------------------------------------------------------------------------
PEER_ACTION_SPECS: dict[str, dict] = {
    "ask_peer": {
        "synopsis": "ask_peer(npcId, can_make, itemId)",
        "description": (
            "Ask another NPC whether it can make itemId (it checks its own recipes). "
            "Does not move the NPC."
        ),
        "asl_args": ["npcId", "pred", "itemId"],
    },
    "request_peer": {
        "synopsis": "request_peer(npcId, goalSig, itemId, qty)",
        "description": (
            "Ask another NPC to produce qty units of itemId for this NPC; waits for its agree/refuse. "
            "If it agrees, it will drop the items on the ground and report when done. goalSig is a "
            "label you choose (e.g. achieve_has_item): use the SAME label in await_peer. "
            "Effect when it agrees: peer_promised(NpcId, GoalSig)."
        ),
        "asl_args": ["npcId", "goalSig", "itemId", "qty"],
    },
    "await_peer": {
        "synopsis": "await_peer(npcId, goalSig, timeoutSeconds)",
        "description": (
            "Wait until that NPC reports the requested goal as done (or failed). Delegated work can "
            "take several minutes (e.g. 300). Effect on success: peer_item_available(NpcId, ItemId, "
            "Qty, X, Y), where the items were dropped. Does not pick them up."
        ),
        "asl_args": ["npcId", "goalSig", "timeoutSeconds"],
    },
}
PEER_ACTION_NAMES: frozenset[str] = frozenset(PEER_ACTION_SPECS)


def _build_peer_actions_catalog() -> str:
    lines = []
    for spec in PEER_ACTION_SPECS.values():
        desc = " ".join(str(spec["description"]).split())
        lines.append(f"{spec['synopsis']:<38}-> {desc}")
    return "\n".join(lines) + "\n"


PEER_ACTIONS_CATALOG: str = _build_peer_actions_catalog()

# Firmas para el validador de step3/step4 (aridad exacta).
ACTION_SIGNATURES.update({
    name.upper(): (len(spec["asl_args"]), len(spec["asl_args"]), list(spec["asl_args"]))
    for name, spec in PEER_ACTION_SPECS.items()
})
