from __future__ import annotations

from typing import Any


class ActionContractError(ValueError):
    pass


# Acciones primitivas que Unity ejecuta directamente.
# Estos nombres (PascalCase) son los que van en ActionCommand.actionType.
# En ASL se usan como .lower_case(args), ej: .moveto(X, Y).
#
# FUENTE DE VERDAD para:
#   - El validador de protcolo (validate_action)
#   - El catálogo de acciones generado para prompts LLM (catalogs.py lo importa)
#   - El validador semántico step3_validator.py (deriva ACTION_SIGNATURES desde aquí)
PRIMITIVE_ACTIONS: set[str] = {
    "MoveTo",
    "ExploreArea",
    "Search",
    "PickUp",
    "Craft",
    "Drop",
    "Wait",
}

# Cada entrada documenta la acción para validators, prompts y DOC.
# Campos:
#   description  — texto en inglés para el LLM (inyectado en prompts)
#   synopsis     — firma compacta para el ACTIONS_CATALOG en prompts
#   required     — args obligatorios en ActionCommand.args
#   optional     — args opcionales en ActionCommand.args
#   asl_args     — nombres de posición en ASL .action(arg0, arg1, ...)
#                  El orden importa: coincide con min_args → max_args para el validator
ACTION_ALLOWLIST: dict[str, dict[str, Any]] = {
    "MoveTo": {
        "description": (
            "Move NPC to map coordinates X, Y. "
            "Effect: sets current_position(X, Y)."
        ),
        "synopsis":    "MoveTo(X, Y)",
        "required":    ["x", "y"],
        "optional":    [],
        "asl_args":    ["x", "y"],
    },
    "ExploreArea": {
        "description": (
            "Roam the map to discover zones or items. "
            "Use this only when a zone_center belief for zoneTag is not yet known. "
            "If zone_center(zoneTag, X, Y) is already known, prefer MoveTo(X, Y) and then Search(itemId) instead of ExploreArea(zoneTag). "
            "Optional zoneTag restricts exploration to one zone type. "
            "Range is controlled by Unity and must NOT be sent. "
            "Effect: may observe zone_center and item_at; updates current_position."
        ),
        "synopsis":    "ExploreArea(zoneTag?)",
        "required":    [],
        "optional":    ["zoneTag", "maxAttempts"],
        "asl_args":    ["zoneTag"],
    },
    "Search": {
        "description": (
            "Search the current zone for a specific item type. "
            "Use after MoveTo to a known spawn zone when item_at belief is not yet set. "
            "On success Unity sets the item_at belief. "
            "Effect: may observe item_at(ItemId, X, Y); does NOT pick up; does NOT move the NPC."
        ),
        "synopsis":    "Search(itemId)",
        "required":    ["itemId"],
        "optional":    [],
        "asl_args":    ["itemId"],
    },
    "PickUp": {
        "description": (
            "Pick up itemId from the NPC's current position. "
            "Effect: places ItemId into inventory (has_item); requires the NPC to be on the item's tile."
        ),
        "synopsis":    "PickUp(itemId)",
        "required":    [],
        "optional":    ["itemId", "id", "instanceId"],
        "asl_args":    ["itemId"],
    },
    "Craft": {
        "description": (
            "Craft an item using a recipe. "
            "itemId = main ingredient, targetId = recipeId (e.g. bread_recipe). "
            "qty is optional and counts units of the INGREDIENT itemId to consume (NOT units of the output): "
            "it must equal the recipe's input quantity, so omit it to use the recipe's amount. "
            "PRECONDITION: the NPC must be inside the recipe's zone — at_zone(RecipeZone) must be true. "
            "Use MoveTo(ZX, ZY) first (from zone_center(RecipeZone, ZX, ZY)) if at_zone is not yet set. "
            # Fase 17u: "produces ItemToCraft" hacía leer itemId como el producto
            # (Craft(flour, Flour_recipe) 3/3 → Craft(wheat, …) 3/3 con este texto).
            "Effect: produces the recipe's OUTPUT item in inventory (has_item); consumes the ingredient itemId."
        ),
        "synopsis":    "Craft(itemId, targetId, qty?)",
        "required":    ["itemId", "targetId"],
        "optional":    ["qty"],
        "asl_args":    ["itemId", "targetId", "qty"],
    },
    "Drop": {
        "description": (
            "Drop qty units of itemId from inventory, optionally at a delivery targetId. "
            "Effect: removes one unit from inventory and creates item_at at the NPC's current position."
        ),
        "synopsis":    "Drop(itemId, qty, targetId?)",
        "required":    ["itemId"],
        "optional":    ["instanceId", "id", "qty", "targetId"],
        "asl_args":    ["itemId", "qty", "targetId"],
    },
    "Wait": {
        "description": (
            "Idle for the given number of ticks without doing anything. "
            "Effect: none on beliefs; passes ticks."
        ),
        "synopsis":    "Wait(ticks)",
        "required":    ["ticks"],
        "optional":    [],
        "asl_args":    ["ticks"],
    },
}


# Campos que deben ser numéricos (int o float) en los ActionCommand enviados a Unity.
# Un string como 'ZX' o '9' indica una variable sin resolver o un tipo incorrecto.
_NUMERIC_ARGS: dict[str, set[str]] = {
    "MoveTo":      {"x", "y"},
    "Wait":        {"ticks"},
}


def validate_action(action_type: str, args: dict[str, Any]) -> None:
    if action_type not in ACTION_ALLOWLIST:
        raise ActionContractError(
            f"Action '{action_type}' not in allowlist. "
            f"Allowed: {sorted(ACTION_ALLOWLIST)}"
        )
    spec = ACTION_ALLOWLIST[action_type]
    for field in spec["required"]:
        if field not in args:
            raise ActionContractError(
                f"Action '{action_type}' missing required arg '{field}'."
            )
    # Validar tipos numericos: evitar que variables sin resolver lleguen a Unity.
    for field in _NUMERIC_ARGS.get(action_type, set()):
        val = args.get(field)
        if val is not None and not isinstance(val, (int, float)):
            raise ActionContractError(
                f"Action '{action_type}' arg '{field}' must be numeric, got {type(val).__name__!r}: {val!r}. "
                f"Possible unresolved variable or wrong belief type."
            )
