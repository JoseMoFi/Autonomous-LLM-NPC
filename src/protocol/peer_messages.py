from __future__ import annotations

"""peer_messages.py — esquema de mensajes de coordinación NPC↔NPC (Fase 12).

Subconjunto FIPA de 7 performativas, transportadas por XMPP directo
(`Message.metadata={"protocol": "npc-coord"}`), cuerpo JSON. Módulo PURO
(sin I/O, sin SPADE): parseo y validación, para que `PeerCoordBehaviour`
(src/npc/behaviours/peer_coord.py) nunca reviente con un mensaje malformado
ni con una performativa/predicado fuera de la allowlist — regla de
no-silencio: todo rechazo queda trazado con su motivo, nunca una excepción
no controlada que tumbe el behaviour.

Interacciones (ver PLAN_EJECUCION/FASE_12_COORDINACION_MULTIAGENTE.md §D2):

  Consulta:  query-if {pred, args}       -> inform {pred, args, value}
  Petición:  request {goal_sig, condition, depth}
                                          -> agree {goal_sig}
                                           | refuse {goal_sig, reason}
             (más tarde, asíncrono, sin más round-trip)
                                          -> inform-done {goal_sig, item?, qty?, x?, y?}
                                           | failure {goal_sig, reason}

`inform-done` puede llevar la entrega física en el mismo mensaje (item/qty/x/y
— Nivel 1 del intercambio: el emisor ya hizo `.drop` en su posición actual y
avisa dónde recogerlo). Es una simplificación deliberada respecto al diseño
inicial (que separaba un `inform-item` aparte): un solo mensaje, un solo punto
de fallo menos.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Metadata SPADE que identifica el canal de coordinación (no interfiere con
# el canal thread-based que ya usa el NPC para hablar con el planificador).
PROTOCOL_METADATA = {"protocol": "npc-coord"}

PERFORMATIVES = {
    "query-if", "inform", "request", "agree", "refuse", "inform-done", "failure",
}

# Predicados consultables vía `query-if` — allowlist explícita, nada más es
# respondible (mismo principio que la allowlist de acciones de trigger body).
QUERYABLE_PREDICATES = {"can_make", "has_item", "knows_zone", "busy"}


class PeerMessageError(ValueError):
    """Mensaje malformado o fuera de protocolo. Nunca se propaga como excepción
    no controlada — quien la capture debe loguear+trazar y responder `refuse`
    (o simplemente descartar, para performativas sin respuesta esperada)."""


@dataclass
class PeerMessage:
    performative: str
    conversation_id: str
    depth: int = 0
    # query-if / inform
    pred: str | None = None
    args: list[Any] = field(default_factory=list)
    value: Any = None
    # request / agree / refuse / inform-done / failure
    goal_sig: str | None = None
    condition: str | None = None
    reason: str | None = None
    # inform-done — entrega física opcional (Nivel 1)
    item: str | None = None
    qty: int | None = None
    x: float | None = None
    y: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "performative": self.performative,
            "conversation_id": self.conversation_id,
            "depth": self.depth,
        }
        for field_name in (
            "pred", "args", "value", "goal_sig", "condition", "reason",
            "item", "qty", "x", "y",
        ):
            v = getattr(self, field_name)
            if v is not None and v != []:
                d[field_name] = v
        return d


def parse_peer_message(payload: dict) -> PeerMessage:
    """Valida y construye un PeerMessage desde un dict ya deserializado (el
    llamante hace el `json.loads`; aquí solo se valida la FORMA).

    Lanza PeerMessageError con un motivo legible si el mensaje no es válido.
    Nunca lanza otra excepción (KeyError/TypeError se normalizan aquí).
    """
    if not isinstance(payload, dict):
        raise PeerMessageError("payload no es un objeto JSON")

    performative = payload.get("performative")
    if performative not in PERFORMATIVES:
        raise PeerMessageError(f"performativa desconocida: {performative!r}")

    conversation_id = payload.get("conversation_id")
    if not conversation_id or not isinstance(conversation_id, str):
        raise PeerMessageError("conversation_id ausente o inválido")

    try:
        depth = int(payload.get("depth", 0))
    except (TypeError, ValueError):
        raise PeerMessageError("depth no es entero") from None

    msg = PeerMessage(performative=performative, conversation_id=conversation_id, depth=depth)

    if performative == "query-if":
        pred = payload.get("pred")
        if pred not in QUERYABLE_PREDICATES:
            raise PeerMessageError(f"predicado fuera de la allowlist: {pred!r}")
        msg.pred = pred
        args = payload.get("args", [])
        msg.args = list(args) if isinstance(args, list) else [args]

    elif performative == "inform":
        msg.pred = payload.get("pred")
        args = payload.get("args", [])
        msg.args = list(args) if isinstance(args, list) else [args]
        msg.value = payload.get("value")

    elif performative == "request":
        goal_sig = payload.get("goal_sig")
        condition = payload.get("condition")
        if not goal_sig or not isinstance(goal_sig, str):
            raise PeerMessageError("request sin goal_sig")
        if not condition or not isinstance(condition, str):
            raise PeerMessageError("request sin condition")
        msg.goal_sig = goal_sig
        msg.condition = condition

    elif performative == "agree":
        msg.goal_sig = payload.get("goal_sig")

    elif performative == "refuse":
        msg.goal_sig = payload.get("goal_sig")
        msg.reason = payload.get("reason", "unspecified")

    elif performative == "inform-done":
        msg.goal_sig = payload.get("goal_sig")
        msg.item = payload.get("item")
        msg.qty = payload.get("qty")
        msg.x = payload.get("x")
        msg.y = payload.get("y")

    elif performative == "failure":
        msg.goal_sig = payload.get("goal_sig")
        msg.reason = payload.get("reason", "unspecified")

    return msg


def try_parse_peer_message(payload: Any) -> tuple[PeerMessage | None, str | None]:
    """Variante no-lanzante de `parse_peer_message`: devuelve (msg, None) si es
    válido, o (None, motivo) si no. Para el llamante que solo quiere loguear y
    seguir (PeerCoordBehaviour.run — un mensaje malformado NUNCA debe tumbar
    el behaviour cíclico)."""
    try:
        return parse_peer_message(payload), None
    except PeerMessageError as exc:
        return None, str(exc)
