from __future__ import annotations

"""builtins.py — fuente ÚNICA de los sub-planes builtin "terminales".

Un sub-plan **terminal** encapsula sus propias acciones primitivas y GARANTIZA
su salida, por lo que un plan puede terminar en él aunque no contenga una acción
primitiva explícita (no es un "plan vacío" ni un bucle de subgoals).

Antes esta lista estaba **hardcodeada y duplicada** en `step3_steps.py`,
`mini_repair.py` y un docstring de `pipeline_runner.py`, y NO incluía
`achieve_explore_zone` → un plan legítimo `[achieve_explore_zone(zona)]` se
rechazaba por "sin acción primitiva" y el goal de explorar fallaba. Centralizado
aquí (parche pequeño, 2026-06-15).

GENERALIZACIÓN PENDIENTE (fix grande, ver
`DOC/Diseno/SUBPLANES_TERMINALES_Y_GARANTIAS_2026-06-15.md`): derivar este
conjunto de los **contratos de capability** (`guarantees_on_success` /
`_BUILTIN_SUBGOAL_GUARANTEES`) en vez de listarlo a mano, de modo que cualquier
sub-plan nuevo que garantice su salida se acepte como terminal automáticamente,
sin tocar este fichero.
"""

# Sub-planes builtin que pueden cerrar un plan (ser el último step) por garantizar
# su salida con primitivas internas. Coincide hoy con los builtins de
# `plans/builtin/*.asl` (move_to_and_pickup, craft_item, achieve_explore_zone).
TERMINAL_SUBPLANS: frozenset[str] = frozenset({
    "move_to_and_pickup",    # garantiza has_item(ItemId, N)
    "craft_item",            # garantiza has_item(ItemToCraft, N) (output de receta)
    "achieve_explore_zone",  # garantiza knows_zone(ZoneTag) / zone_center
})

# Fase 16 — ablación de sub-planes. Sub-planes "macro" escritos a mano que
# resuelven adquisición y crafteo por dentro (leen item_at/zone_center/at_zone
# en runtime). Con settings.builtin_subplans_enabled=False no se cargan ni se
# ofrecen al LLM. achieve_explore_zone se conserva: zone_center es global desde
# el arranque, así que no resuelve nada de la tarea (y `.explorearea` se omite
# sola si la zona ya se conoce).
ABLATABLE_SUBPLANS: frozenset[str] = frozenset({"move_to_and_pickup", "craft_item"})
# Ficheros de plans/builtin/ que los definen (incluye helpers internos como
# craft_item_check, que viven en el mismo fichero).
ABLATABLE_BUILTIN_FILES: frozenset[str] = frozenset({"move_to_and_pickup.asl", "craft_item.asl"})


def terminal_subplans(builtin_subplans: bool = True) -> frozenset[str]:
    """Sub-planes terminales activos: sin los ablacionados si están desactivados."""
    if builtin_subplans:
        return TERMINAL_SUBPLANS
    return TERMINAL_SUBPLANS - ABLATABLE_SUBPLANS


# Fase 17 — sub-planes macro de COORDINACIÓN escritos a mano: pedir un item a un
# peer (ask + request + await + recoger) y recoger lo que un peer dejó en el
# suelo. Solo se ofrecen con coordinación y sub-planes activos; en ATOM el LLM
# compone .ask_peer/.request_peer/.await_peer y la recogida con MoveTo/PickUp.
COORDINATION_SUBPLANS: frozenset[str] = frozenset({"obtain_from_peer", "collect_from_peer"})
COORDINATION_BUILTIN_FILES: frozenset[str] = frozenset({"obtain_from_peer.asl", "collect_from_peer.asl"})


def active_terminal_subplans(builtin_subplans: bool = True, coordination: bool = False) -> frozenset[str]:
    """terminal_subplans() más los macros de coordinación cuando se ofrecen."""
    base = terminal_subplans(builtin_subplans)
    if coordination and builtin_subplans:
        return base | COORDINATION_SUBPLANS
    return base


def builtin_file_exclusions(builtin_subplans: bool = True, coordination: bool = False) -> frozenset[str]:
    """Ficheros de plans/builtin/ que NO se cargan.

    - obtain_from_peer.asl (Fase 17) solo tiene sentido con coordinación; sin
      ella no se carga, y los prompts de un solo NPC quedan como antes.
    - Sin sub-planes (ATOM): fuera move_to_and_pickup/craft_item y, con
      coordinación, también los macros de coordinación. Sin coordinación,
      collect_from_peer se sigue cargando como hasta ahora (nadie lo usa).
    """
    excluded: set[str] = set()
    if not coordination:
        excluded.add("obtain_from_peer.asl")
    if not builtin_subplans:
        excluded |= ABLATABLE_BUILTIN_FILES
        if coordination:
            excluded |= COORDINATION_BUILTIN_FILES
    return frozenset(excluded)
