from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Callable

from utils.trace_logger import trace as _trace

if TYPE_CHECKING:
    from npc.trigger_registry import TriggerRegistry

log = logging.getLogger(__name__)


class BeliefStore:
    """
    Puente canónico entre mensajes Unity y el runtime ASL.

    Los nombres de predicados aquí deben coincidir 1:1 con los del
    BELIEF_CATALOG que se inyecta en los prompts LLM.

    Estructura interna: dict[predicate_name → set[tuple[args...]]]
    """

    ITEM_AT_TTL: int = 70  # ticks de vida de item_at

    def __init__(self) -> None:
        self._facts: dict[str, set[tuple]] = {}
        self._recent_additions: list[str] = []
        self._trigger_registry: "TriggerRegistry | None" = None
        # Callback para adopt_goal(sig) desde un trigger body. Lo inyecta el
        # NPCAgent en setup() (añade un Goal a agent.goals si no existe).
        self._goal_adopter: "Callable[[str], None] | None" = None
        # npc_id para trazas de triggers (lo fija el agente).
        self.npc_id: str = "?"
        # TTL para item_at: (item_id, x, y) → tick_expiry
        self._item_at_ttl: dict[tuple, int] = {}
        # Bounds de cada zona conocida (del NPCProfile / ZoneDiscovery).
        # tag → (x_min, x_max, y_min, y_max). Usados para derivar at_zone por
        # posición cuando Unity no emite ZoneEntry (fallback del hallazgo C1).
        self._zone_bounds: dict[str, tuple[int, int, int, int]] = {}

    def set_trigger_registry(self, registry: "TriggerRegistry") -> None:
        self._trigger_registry = registry

    def set_goal_adopter(self, adopter: "Callable[[str], None]") -> None:
        """Inyecta el callback que un trigger usa para adoptar un goal nuevo."""
        self._goal_adopter = adopter

    # ------------------------------------------------------------------
    # Unity event handlers
    # ------------------------------------------------------------------

    def apply_zone_discovery(self, tag: str, x: float, y: float) -> None:
        """Llega desde ZoneDiscovery de Unity o NPCProfile.
        Establece zone_center y knows_zone (derivada automáticamente).
        Upsert: elimina entradas anteriores del mismo tag para evitar duplicados
        con tipos mezclados (int/float) que romperían guards de agentspeak.
        """
        t = tag.lower()
        # Upsert: eliminar cualquier entrada previa para este tag
        existing = self._facts.get("zone_center", set())
        self._facts["zone_center"] = {f for f in existing if f[0] != t}
        # Coercionar a int para garantizar tipo consistente
        self._set("zone_center", (t, int(x), int(y)))
        self._set("knows_zone", (t,))
        self._recent_additions.append("zone_center")
        self._fire_triggers("zone_center")

    def apply_inventory_update(self, items: list[dict]) -> None:
        """Llega desde InventoryUpdate de Unity — reemplaza todo el inventario."""
        self._clear_predicate("has_item")
        for item in items:
            item_id = item["itemId"].lower()
            qty = item["qty"]
            self._set("has_item", (item_id, qty))
        self._recent_additions.append("has_item")
        self._fire_triggers("has_item")

    def apply_has_item(self, item_id: str, qty: int) -> None:
        """Upsert de un item concreto en el inventario.
        Elimina cualquier tupla previa has_item(item_id, *) y escribe la nueva.
        qty == 0 elimina el item del inventario.
        """
        item_id_l = item_id.lower()
        facts = self._facts.get("has_item", set())
        to_remove = {f for f in facts if f[0] == item_id_l}
        facts -= to_remove
        if qty > 0:
            self._set("has_item", (item_id_l, qty))
        self._recent_additions.append("has_item")
        self._fire_triggers("has_item")

    def apply_drop_item(self, item_id: str, qty_dropped: int) -> None:
        """Reduce la cantidad de un item tras un Drop exitoso.
        Si el resultado es <= 0, elimina el belief.
        """
        item_id_l = item_id.lower()
        results = self.query("has_item", item_id_l, None)
        current_qty = results[0][1] if results else 0
        new_qty = max(0, current_qty - qty_dropped)
        self.apply_has_item(item_id_l, new_qty)

    def apply_delivery_point(self, tag: str, x: float, y: float) -> None:
        """Registra un punto de entrega. Normalmente viene del NPCProfile."""
        self._set("delivery_point", (tag.lower(), x, y))

    def apply_recipe(self, recipe_id: str, zone: str, inputs: list[dict], outputs: list[dict]) -> None:
        """Registra una receta. Viene del NPCProfile."""
        for inp in inputs:
            self._set("recipe", (recipe_id.lower(), zone.lower(),
                                 inp["itemId"].lower(), inp["qty"]))
        for out in outputs:
            self._set("recipe_output", (recipe_id.lower(), zone.lower(),
                                        out["itemId"].lower(), out["qty"]))

    def apply_item_spawn(self, item_id: str, zone_tag: str) -> None:
        """Registra que un item puede aparecer en una zona."""
        self._set("item_spawn", (item_id.lower(), zone_tag.lower()))

    def apply_peer(self, npc_id: str, role: str) -> None:
        """Registra la existencia de otro NPC del sistema (peer/2).

        Fase 11 (T6): lo siembra `NPCRegistry.announce_peers` cuando un NPC
        nuevo se registra — SOLO existencia y rol, nada de capacidades ni
        inventario (eso se pregunta en la Fase 12; un agente no es omnisciente
        sobre otro). Upsert por npc_id (el rol puede llegar como "unknown" al
        registrarse y actualizarse después). Sin trigger_registry o sin la
        Fase 12 activa, este belief no dispara nada — coste cero.
        """
        npc_id_l = npc_id.lower()
        facts = self._facts.get("peer", set())
        to_remove = {f for f in facts if f[0] == npc_id_l}
        facts -= to_remove
        self._facts["peer"] = facts
        self._set("peer", (npc_id_l, (role or "unknown").lower()))
        self._recent_additions.append("peer")
        self._fire_triggers("peer")

    # ------------------------------------------------------------------
    # Fase 12 — coordinación NPC↔NPC: creencias que el EMISOR escribe con
    # la respuesta de un peer. SIEMPRE prefijadas `peer_*`/`peer_item_*`,
    # nunca se mezclan con las creencias propias del NPC (p.ej. `has_item`
    # es MI inventario; `peer_has_item` es lo que un peer me dijo que tiene).
    # Todas son upsert por (npc_id[, goal_sig]) — sin TTL: las escribe
    # PeerCoordBehaviour al recibir una respuesta y las consume una acción
    # ASL (.ask_peer/.await_peer) o una variante de la familia canónica.
    # ------------------------------------------------------------------

    def _upsert_peer_fact(self, predicate: str, key_prefix: tuple, args: tuple) -> None:
        """Upsert genérico: elimina cualquier hecho previo cuyo prefijo de
        argumentos coincida con `key_prefix` (p.ej. (npc_id,) o (npc_id,goal_sig))
        antes de añadir `args`. Evita acumular respuestas viejas del mismo peer."""
        n = len(key_prefix)
        facts = self._facts.get(predicate, set())
        facts = {f for f in facts if f[:n] != key_prefix}
        self._facts[predicate] = facts
        self._set(predicate, args)

    def apply_peer_can_make(self, npc_id: str, item_id: str, value: bool) -> None:
        """Respuesta a `.ask_peer(Npc, can_make, Item)` — peer_can_make(Npc, Item)
        solo se assertea si value es True (ausencia = "no sé" o "no puede", no se
        fabrica un peer_cannot_make; el guard del delegate variant solo necesita
        el caso positivo)."""
        npc_id_l, item_l = npc_id.lower(), item_id.lower()
        facts = self._facts.get("peer_can_make", set())
        facts = {f for f in facts if f != (npc_id_l, item_l)}
        self._facts["peer_can_make"] = facts
        if value:
            self._set("peer_can_make", (npc_id_l, item_l))
        self._recent_additions.append("peer_can_make")

    def apply_peer_has_item(self, npc_id: str, item_id: str, qty: int) -> None:
        """Respuesta a `.ask_peer(Npc, has_item, Item)` — cantidad que el peer
        reportó tener (0 si no tiene)."""
        npc_id_l, item_l = npc_id.lower(), item_id.lower()
        self._upsert_peer_fact(
            "peer_has_item", (npc_id_l, item_l), (npc_id_l, item_l, int(qty))
        )
        self._recent_additions.append("peer_has_item")

    def apply_peer_knows_zone(self, npc_id: str, zone_tag: str, value: bool) -> None:
        npc_id_l, zone_l = npc_id.lower(), zone_tag.lower()
        facts = self._facts.get("peer_knows_zone", set())
        facts = {f for f in facts if f != (npc_id_l, zone_l)}
        self._facts["peer_knows_zone"] = facts
        if value:
            self._set("peer_knows_zone", (npc_id_l, zone_l))
        self._recent_additions.append("peer_knows_zone")

    def apply_peer_busy(self, npc_id: str, value: bool) -> None:
        npc_id_l = npc_id.lower()
        facts = self._facts.get("peer_busy", set())
        facts = {f for f in facts if f != (npc_id_l,)}
        self._facts["peer_busy"] = facts
        if value:
            self._set("peer_busy", (npc_id_l,))
        self._recent_additions.append("peer_busy")

    def apply_peer_promised(self, npc_id: str, goal_sig: str) -> None:
        """El peer respondió `agree` a un `.request_peer`."""
        npc_id_l = npc_id.lower()
        self._upsert_peer_fact(
            "peer_promised", (npc_id_l, goal_sig), (npc_id_l, goal_sig)
        )
        self._recent_additions.append("peer_promised")

    def apply_peer_refused(self, npc_id: str, goal_sig: str, reason: str) -> None:
        """El peer respondió `refuse` a un `.request_peer`.

        Fase 17x: el rechazo retira la promesa anterior del mismo peer y goal_sig.
        Sin esto, tras un `already_failed` la rama de esperar (peer_promised & not
        peer_failed) seguía aplicando y `.await_peer` esperaba 300 s a un encargo
        que nadie hacía (piloto de la tanda coop 17w, CO6/ATOM: 700 s perdidos)."""
        npc_id_l = npc_id.lower()
        self._upsert_peer_fact(
            "peer_refused", (npc_id_l, goal_sig), (npc_id_l, goal_sig, reason)
        )
        promised = self._facts.get("peer_promised")
        if promised:
            self._facts["peer_promised"] = {f for f in promised if f[:2] != (npc_id_l, goal_sig)}
        self._recent_additions.append("peer_refused")

    def apply_peer_done(
        self, npc_id: str, goal_sig: str,
        item_id: str | None = None, qty: int | None = None,
        x: float | None = None, y: float | None = None,
    ) -> None:
        """El peer completó el goal delegado (`inform-done`). Si trae entrega
        física (item/qty/x/y — Nivel 1, drop+aviso en un solo mensaje), también
        assertea `peer_item_available(Npc, Item, Qty, X, Y)` para que
        `!collect_from_peer` sepa dónde recoger."""
        npc_id_l = npc_id.lower()
        self._upsert_peer_fact("peer_done", (npc_id_l, goal_sig), (npc_id_l, goal_sig))
        self._recent_additions.append("peer_done")
        if item_id is not None and qty is not None and x is not None and y is not None:
            item_l = item_id.lower()
            self._upsert_peer_fact(
                "peer_item_available", (npc_id_l, item_l),
                (npc_id_l, item_l, int(qty), x, y),
            )
            self._recent_additions.append("peer_item_available")

    def apply_peer_failed(self, npc_id: str, goal_sig: str, reason: str) -> None:
        """El goal delegado en el peer falló (`failure`) — nunca se cierra
        `.await_peer` en silencio: se assertea `peer_failed` (que la acción lee
        para fallar de forma trazada, no colgarse hasta el timeout)."""
        npc_id_l = npc_id.lower()
        self._upsert_peer_fact(
            "peer_failed", (npc_id_l, goal_sig), (npc_id_l, goal_sig, reason)
        )
        self._recent_additions.append("peer_failed")

    def clear_peer_request(self, npc_id: str, goal_sig: str) -> None:
        """Fase 17o: al enviar una petición nueva se olvida el FALLO de la anterior
        al mismo peer y goal_sig (peer_failed/peer_refused). Sin esto,
        PeerDoneWaiter leía un peer_failed viejo y el .await_peer de la petición
        nueva fallaba al instante: CO6/ATOM pedía en bucle hasta que el receptor
        rechazaba por too_many_requests. peer_done se conserva: una entrega ya
        hecha sigue siendo válida, y SUB/DET (que al retomar una intención aparcada
        vuelven a enviar la misma petición) no cambian de comportamiento."""
        key = (npc_id.lower(), goal_sig)
        for predicate in ("peer_failed", "peer_refused"):
            facts = self._facts.get(predicate)
            if facts:
                self._facts[predicate] = {f for f in facts if f[:2] != key}

    def clear_peer_item_available(self, npc_id: str, item_id: str) -> None:
        """Limpia `peer_item_available` tras recogerlo con éxito (evita que
        `!collect_from_peer` reintente sobre una entrega ya consumida)."""
        npc_id_l, item_l = npc_id.lower(), item_id.lower()
        facts = self._facts.get("peer_item_available")
        if facts:
            self._facts["peer_item_available"] = {
                f for f in facts if not (f[0] == npc_id_l and f[1] == item_l)
            }

    def apply_exhausted(self, action_type: str, budget: int) -> None:
        """Assertea exhausted(ActionType, Budget) tras agotar los intentos de una
        acción observacional (0.B2). Lo consulta el guard de las variantes de
        agotamiento generadas por el pipeline (p.ej. exhausted(Search, 3))."""
        self._set("exhausted", (action_type, int(budget)))
        self._recent_additions.append("exhausted")
        self._fire_triggers("exhausted")

    def clear_exhausted(self, action_type: str | None = None) -> None:
        """Limpia los beliefs exhausted. Si action_type es None, limpia todos;
        si se da, solo los de esa acción. Se llama al completar/replanificar un goal."""
        if action_type is None:
            self._clear_predicate("exhausted")
            return
        facts = self._facts.get("exhausted")
        if facts:
            self._facts["exhausted"] = {f for f in facts if f[0] != action_type}
            if not self._facts["exhausted"]:
                self._facts.pop("exhausted", None)

    def apply_zone_bounds(self, tag: str, x_min: int, x_max: int,
                          y_min: int, y_max: int) -> None:
        """Registra los límites espaciales de una zona (del NPCProfile).

        Se usan para derivar at_zone por posición — ver zone_containing().
        """
        self._zone_bounds[tag.lower()] = (int(x_min), int(x_max),
                                          int(y_min), int(y_max))

    def zone_containing(self, x: int, y: int) -> str | None:
        """Devuelve el tag de la zona cuyos bounds contienen (x, y), o None.

        Fallback para derivar at_zone cuando Unity no emite ZoneEntry (C1).
        Si varias zonas se solapan, devuelve la primera coincidencia.
        """
        for tag, (x_min, x_max, y_min, y_max) in self._zone_bounds.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return tag
        return None

    def apply_current_position(self, x: int, y: int) -> None:
        """Upsert de current_position(X, Y). Elimina cualquier entrada previa.

        Siempre coerciona a int para garantizar tipo consistente y evitar mezcla
        int/float que romperia guards de agentspeak.

        Fuentes autorizadas:
          - RegisterNPC.pos  (via gateway/router.py al registrar el NPC)
          - ActionResult.payload.current_position  (MoveTo o ExploreArea Success)

        NUNCA llamar desde el comando enviado; solo desde confirmacion de Unity.
        """
        self._clear_predicate("current_position")
        self._set("current_position", (int(x), int(y)))
        self._recent_additions.append("current_position")
        self._fire_triggers("current_position")

    def apply_at_zone(self, zone_tag: str) -> None:
        """El NPC ha entrado en una zona (emitido por Unity ZoneEntry.entered=true).
        Reemplaza cualquier at_zone previo — el NPC solo puede estar en una zona a la vez.
        """
        tag = zone_tag.lower()
        self._clear_predicate("at_zone")
        self._set("at_zone", (tag,))
        self._recent_additions.append("at_zone")
        self._fire_triggers("at_zone")

    def remove_at_zone(self, zone_tag: str) -> None:
        """El NPC ha salido de una zona (emitido por Unity ZoneEntry.entered=false)."""
        tag = zone_tag.lower()
        facts = self._facts.get("at_zone", set())
        facts.discard((tag,))
        if not facts:
            self._facts.pop("at_zone", None)

    def apply_item_at(self, item_id: str, x: float, y: float, current_tick: int) -> None:
        """Registra la posición de un item descubierto por Search/WanderSearch.
        TTL = ITEM_AT_TTL ticks desde current_tick.
        Si ya existe el mismo item en distinta posición, se añade (puede haber
        varios ejemplares del mismo item en el mapa).
        """
        key = (item_id.lower(), x, y)
        self._set("item_at", key)
        self._item_at_ttl[key] = current_tick + self.ITEM_AT_TTL
        self._recent_additions.append("item_at")

    def expire_item_at(self, current_tick: int) -> None:
        """Purga los item_at cuyo TTL haya expirado. Llamar en cada ActionResult."""
        expired = [k for k, exp in self._item_at_ttl.items() if current_tick >= exp]
        if expired:
            for key in expired:
                self._item_at_ttl.pop(key, None)
                facts = self._facts.get("item_at")
                if facts:
                    facts.discard(key)
            log.debug(f"[BELIEFS] item_at expirados: {len(expired)}")

    def clear_item_at(self, item_id: str) -> None:
        """Elimina todas las posiciones conocidas de un item (p.ej. tras PickUp exitoso)."""
        item_id_l = item_id.lower()
        to_remove = [k for k in self._item_at_ttl if k[0] == item_id_l]
        for key in to_remove:
            self._item_at_ttl.pop(key, None)
            facts = self._facts.get("item_at")
            if facts:
                facts.discard(key)
        if to_remove:
            log.debug(f"[BELIEFS] item_at limpiados para '{item_id_l}': {len(to_remove)}")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query(self, predicate: str, *args) -> list[tuple]:
        """
        Devuelve todos los hechos que coinciden con el predicado.
        Args son valores concretos o None para wildcard.
        """
        results = []
        for fact in self._facts.get(predicate, set()):
            if len(fact) < len(args):
                continue
            match = all(
                a is None or fact[i] == a
                for i, a in enumerate(args)
            )
            if match:
                results.append(fact)
        return results

    def has(self, predicate: str, *args) -> bool:
        return bool(self.query(predicate, *args))

    def snapshot(self) -> dict[str, list[tuple]]:
        return {k: list(v) for k, v in self._facts.items()}

    def recent_additions(self) -> list[str]:
        result, self._recent_additions = self._recent_additions[:], []
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set(self, predicate: str, args: tuple) -> None:
        self._facts.setdefault(predicate, set()).add(args)

    def _clear_predicate(self, predicate: str) -> None:
        self._facts.pop(predicate, None)

    def _fire_triggers(self, belief_key: str) -> None:
        if self._trigger_registry:
            for rule in self._trigger_registry.evaluate(belief_key, self):
                log.debug(f"[BELIEFS] Trigger disparado: {rule.sig}")
                _trace("trigger_fired", npc_id=self.npc_id,
                       sig=rule.sig, belief_key=belief_key)
                self._execute_trigger_body(rule.body)

    # Acciones de body soportadas en triggers reactivos.
    _LOG_RE = re.compile(r"^\.log\(\s*['\"](?P<msg>.*)['\"]\s*\)$")
    # adopt_goal(sig) | adopt_goal(sig, 'success_condition'). La condición
    # (Fase 9) hace que el goal REACTIVO sea belief-verificado en vez de cerrarse
    # unverified — cierra el gap de "reactividad cableada pero solo-log".
    _ADOPT_RE = re.compile(
        r"^adopt_goal\(\s*(?P<sig>[a-z][a-z0-9_]*)\s*"
        r"(?:,\s*'(?P<cond>[^']*)')?\s*\)$"
    )

    def _execute_trigger_body(self, body: list[str]) -> None:
        """Ejecuta las acciones del body de un trigger.

        Acciones soportadas (MVP de la reactividad — Fase 0):
          - .log('mensaje')   → log INFO real.
          - adopt_goal(sig)    → adopta un Goal nuevo vía el callback inyectado.
        Cualquier otra acción genera un warning (no-silencio).
        """
        for action in body:
            act = action.strip()
            m_log = self._LOG_RE.match(act)
            if m_log:
                log.info("[TRIGGER:%s] %s", self.npc_id, m_log.group("msg"))
                continue
            m_adopt = self._ADOPT_RE.match(act)
            if m_adopt:
                sig = m_adopt.group("sig")
                cond = m_adopt.group("cond")  # None si no se dio condición
                if self._goal_adopter is not None:
                    self._goal_adopter(sig, cond)
                    log.info("[TRIGGER:%s] adopt_goal(%s%s)", self.npc_id, sig,
                             f", '{cond}'" if cond else "")
                    _trace("trigger_adopt_goal", npc_id=self.npc_id, sig=sig,
                           success_condition=cond)
                else:
                    log.warning(
                        "[TRIGGER:%s] adopt_goal(%s) ignorado — sin goal_adopter",
                        self.npc_id, sig,
                    )
                continue
            log.warning("[TRIGGER:%s] trigger action not supported: %s",
                        self.npc_id, act)
