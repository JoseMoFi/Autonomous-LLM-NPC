from __future__ import annotations

"""peer_delivery_plan.py — plan determinista para `achieve_deliver_to_peer`
(Fase 12).

Análogo a `family_plan.py` (Fase 6.5 T4): andamiaje `source=CODE`, sin LLM,
para el goal TOP-LEVEL que un NPC adopta al ACEPTAR una petición de otro
(`request` → `agree` → `_adopt_goal_from_trigger("achieve_deliver_to_peer",
...)`, ver `PeerCoordBehaviour._handle_request`):

    +!achieve_deliver_to_peer(Requester, Item, Qty) : has_item(Item, Qty)
        <- .drop(Item, Qty); .deliver_to_peer(Requester, Item, Qty).
    +!achieve_deliver_to_peer(Requester, Item, Qty) : not has_item(Item, Qty)
        <- !achieve_has_item(Item, Qty); !achieve_deliver_to_peer(Requester, Item, Qty).

La segunda variante reentra en la familia `achieve_has_item` (gather/craft/
delegate — incluida su propia posible sub-delegación, acotada por
`peer_max_depth`) para conseguir el ingrediente, y luego se re-invoca a sí
misma — misma técnica recursiva que `family_plan.py` usa para reunir
ingredientes de una receta.

Por qué NO es un builtin de `plans/builtin/*.asl`: ese mecanismo
(`builtin_loader.py`) registra el GoalNode bajo el `sig` PLANO (sin binding),
pero `achieve_deliver_to_peer` se adopta como goal de NIVEL SUPERIOR con
`call_args=[requester, item, qty]` — el `plan_graph` lo indexa por
`sig+call_args` (identidad por binding, Fase 6.5), así que necesita el MISMO
patrón que `family_plan.py`: generado bajo demanda en `bdi._request_plan`
(`_try_peer_delivery_plan`), no cargado al arranque.
"""

from dataclasses import dataclass

SIG = "achieve_deliver_to_peer"
_HEAD = f"+!{SIG}(Requester, Item, Qty)"


@dataclass
class DeliveryVariant:
    guard: str
    asl: str
    source: str = "CODE"


def build_deliver_to_peer_variants() -> list[DeliveryVariant]:
    """Construye las 2 variantes de `achieve_deliver_to_peer`. Sin parámetros:
    a diferencia de `build_has_item_family`, no depende de recetas/spawns del
    mundo — reusa la familia `achieve_has_item` ya cargada para conseguir el
    ingrediente.

    `.deliver_to_peer` va INMEDIATAMENTE después de `.drop` (misma posición
    que el drop — necesaria para decirle al peticionario dónde recoger). El
    paso aparte del punto de entrega (hallazgo de la validación e2e real,
    Fase 13, sesión E5 2026-08-11: el que entrega se queda quieto y bloquea
    el pathfinding del peticionario) lo dispara la PROPIA acción
    `.deliver_to_peer` como comando de Unity independiente, fuera del ciclo
    de la intención — no puede ir aquí como un `.moveto` más en el body: si
    la creencia `delivered_to_peer` se fijara ANTES del movimiento, el
    short-circuit de `run()` completaría el goal y cancelaría las
    intenciones (incluida la del `.moveto`) antes de que el NPC se apartara
    de verdad; y si se fijara DESPUÉS, `.deliver_to_peer` ya no estaría en la
    posición del drop para reportarla correctamente. Ver bdi.py."""
    have_guard = "has_item(Item, Qty)"
    need_guard = "not has_item(Item, Qty)"
    return [
        DeliveryVariant(
            guard=have_guard,
            asl=(
                f"{_HEAD} : {have_guard} <-\n"
                f"    .drop(Item, Qty);\n"
                f"    .deliver_to_peer(Requester, Item, Qty)."
            ),
        ),
        DeliveryVariant(
            guard=need_guard,
            asl=(
                f"{_HEAD} : {need_guard} <-\n"
                f"    !achieve_has_item(Item, Qty);\n"
                f"    !{SIG}(Requester, Item, Qty)."
            ),
        ),
    ]
