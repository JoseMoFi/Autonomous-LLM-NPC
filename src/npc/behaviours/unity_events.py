from __future__ import annotations

import asyncio
import json
import logging

import spade

from utils.trace_logger import trace as _trace

log = logging.getLogger(__name__)


class UnityEventBehaviour(spade.behaviour.CyclicBehaviour):
    """
    Consume agent.inbox y traduce mensajes Unity a beliefs.
    Es el único punto donde entran datos del mundo exterior al NPC.
    """

    async def run(self) -> None:
        if self.agent.paused:
            await asyncio.sleep(0.1)
            return

        try:
            msg = self.agent.inbox.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.05)
            return

        await self._dispatch(msg)

    async def _dispatch(self, msg: dict) -> None:
        msg_type = msg.get("type", "")

        match msg_type:
            case "NPCProfile":
                await self._handle_npc_profile(msg)

            case "ZoneDiscovery":
                self._handle_zone_discovery(msg)

            case "InventoryUpdate":
                self._handle_inventory_update(msg)

            case "ActionResult":
                self._handle_action_result(msg)

            case "ZoneEntry":
                self._handle_zone_entry(msg)

            case "WanderRetry":
                self._handle_wander_retry(msg)

            case _:
                log.debug(f"[UNITY_EV:{self.agent.npc_id}] Tipo ignorado: {msg_type}")

    async def _handle_npc_profile(self, msg: dict) -> None:
        from protocol.messages import NPCProfilePayload

        log.info(f"[UNITY_EV:{self.agent.npc_id}] NPCProfile recibido")
        raw_profile = msg.get("profile") or {}
        profile_fingerprint = _profile_fingerprint(raw_profile)
        same_profile = profile_fingerprint == getattr(
            self.agent, "_profile_fingerprint", None
        )

        try:
            self.agent.profile = NPCProfilePayload.from_dict(raw_profile)
        except Exception as exc:
            log.error(f"[UNITY_EV:{self.agent.npc_id}] Error parseando NPCProfile: {exc}")
            return

        self.agent._profile_fingerprint = profile_fingerprint

        profile = self.agent.profile

        from protocol.messages import derive_entity_catalog
        catalog = derive_entity_catalog(profile)
        log.info(
            "[UNITY_EV:%s] Entity catalog — zones: %s | items: %s | delivery: %s | recipes: %s",
            self.agent.npc_id,
            catalog["zone_ids"],
            catalog["item_ids"],
            catalog["delivery_tags"],
            catalog["recipe_ids"],
        )
        _trace(
            "npc_profile",
            npc_id=self.agent.npc_id,
            goals_nl=profile.goals_nl,
            zones=len(profile.zones),
            recipes=len(profile.recipes),
            delivery_points=len(profile.delivery_points),
        )
        for zone in profile.zones:
            self.agent.beliefs.apply_zone_discovery(zone.tag, zone.center.x, zone.center.y)
            self.agent.beliefs.apply_zone_bounds(
                zone.tag,
                zone.bounds.x_min, zone.bounds.x_max,
                zone.bounds.y_min, zone.bounds.y_max,
            )
        for name, pos in profile.delivery_points.items():
            self.agent.beliefs.apply_delivery_point(name, pos.x, pos.y)
        for recipe in profile.recipes:
            self.agent.beliefs.apply_recipe(
                recipe.recipeId, recipe.zone,
                [{"itemId": i.itemId, "qty": i.qty} for i in recipe.inputs],
                [{"itemId": o.itemId, "qty": o.qty} for o in recipe.outputs],
            )
        for spawn in profile.item_spawns:
            for zone_tag in spawn.zones:
                # "none" is a sentinel meaning the item cannot be gathered from the world.
                # Storing item_spawn(x, none) would make move_to_and_pickup call ExploreArea("none").
                if zone_tag.lower() != "none":
                    self.agent.beliefs.apply_item_spawn(spawn.itemId, zone_tag)

        # Acknowledge the profile so Unity can proceed with item spawning
        import uuid
        from protocol.messages import NPCProfileAck
        ack = NPCProfileAck(
            type="NPCProfileAck",
            msg_id=str(uuid.uuid4()),
            npc_id=self.agent.npc_id,
            status="ok",
        )
        await self.agent.send_to_unity(ack.to_dict())
        log.info(f"[UNITY_EV:{self.agent.npc_id}] NPCProfileAck enviado")

        if same_profile:
            bootstrap_task = getattr(self.agent, "_bootstrap_task", None)
            if bootstrap_task is not None and not bootstrap_task.done():
                log.info(
                    "[UNITY_EV:%s] NPCProfile duplicado — parse_goals ya en curso, se omite relanzar",
                    self.agent.npc_id,
                )
                return
            if (
                getattr(self.agent, "_bootstrapped_profile_fingerprint", None)
                == profile_fingerprint
            ):
                log.info(
                    "[UNITY_EV:%s] NPCProfile duplicado — goals ya inicializados, se omite relanzar",
                    self.agent.npc_id,
                )
                return

        bootstrap_task = getattr(self.agent, "_bootstrap_task", None)
        if bootstrap_task is not None and not bootstrap_task.done():
            bootstrap_task.cancel()
            log.info(
                "[UNITY_EV:%s] NPCProfile actualizado — cancelando bootstrap anterior",
                self.agent.npc_id,
            )

        # Arrancar cadena de goals vía LLM (parse_goals)
        # Guardar referencia para poder cancelar la task si el agente para antes
        self.agent._bootstrapped_profile_fingerprint = None
        self.agent._bootstrap_task = asyncio.create_task(
            self._bootstrap_goals(profile_fingerprint)
        )

    async def _bootstrap_goals(self, profile_fingerprint: str) -> None:
        from npc.agent import Goal

        if not self.agent.profile or not self.agent.profile.goals_nl:
            log.warning(f"[UNITY_EV:{self.agent.npc_id}] NPCProfile sin goals_nl")
            return

        log.info(f"[UNITY_EV:{self.agent.npc_id}] Arrancando parse_goals para "
                 f"{len(self.agent.profile.goals_nl)} goals")
        self.agent.phase = "parsing_goals"
        try:
            result = await self.agent.run_llm_task({
                "task": "parse_goals",
                "goals_nl": self.agent.profile.goals_nl,
                "profile": self.agent.profile.to_dict(),
            })
            # goal_conditions del perfil: lista paralela a goals_nl definida por el diseñador.
            # Se empareja por source_index (0.B3): el LLM puede reordenar/añadir goals,
            # así que el índice posicional no es fiable. Fallback posicional con warning
            # si el modelo no devolvió source_index.
            goal_conditions = self.agent.profile.goal_conditions if self.agent.profile else []
            goals = []
            for i, g in enumerate(result):
                goal = Goal(sig=g["sig"], priority=g.get("priority"))
                raw_sc = g.get("success_condition") or None
                goal.success_condition = raw_sc
                src_idx = g.get("source_index")
                if isinstance(src_idx, int) and not isinstance(src_idx, bool):
                    if 0 <= src_idx < len(goal_conditions):
                        goal.expected_condition = goal_conditions[src_idx]
                    else:
                        goal.expected_condition = None
                else:
                    log.warning(
                        "[UNITY_EV:%s] goal '%s' sin source_index — emparejando "
                        "expected_condition por posición (%d)",
                        self.agent.npc_id, g.get("sig", "?"), i,
                    )
                    goal.expected_condition = goal_conditions[i] if i < len(goal_conditions) else None

                # La condición del diseñador (goals.json) es la VERDAD reproducible:
                # si existe, se usa DIRECTAMENTE como success_condition en vez de la
                # derivada por el LLM. Esto evita que el modelo renombre entidades en
                # la derivación (p.ej. quarry -> quarry_zone) y deje el goal sin cerrar,
                # y hace las baterías reproducibles. NO es fabricación: es ground-truth
                # humano explícito; se traza el origen cuando difiere de la del LLM.
                if goal.expected_condition:
                    if goal.expected_condition != goal.success_condition:
                        _trace(
                            "success_condition_from_designer",
                            npc_id=self.agent.npc_id,
                            goal=goal.sig,
                            llm_derived=goal.success_condition,
                            designer_condition=goal.expected_condition,
                        )
                    goal.success_condition = goal.expected_condition
                goals.append(goal)

            # Fase 6.5 (tras flag canonical_reuse_enabled, default off): naming
            # canónico + dedup por familia. Con el flag off no se ejecuta → el
            # comportamiento por defecto es idéntico al anterior.
            from config import settings as _settings
            if _settings.canonical_reuse_enabled and goals:
                from llm.canonical import canonicalize_goals
                from protocol.messages import derive_entity_catalog
                _cat = derive_entity_catalog(self.agent.profile) if self.agent.profile else {}
                _before = [g.sig for g in goals]
                goals = canonicalize_goals(goals, _cat)
                _trace(
                    "goals_canonicalized",
                    npc_id=self.agent.npc_id,
                    before=_before,
                    after=[{"sig": g.sig, "call_args": g.call_args} for g in goals],
                )

            self.agent.goals = goals
            if goals:
                self.agent._had_goals = True
            self.agent._bootstrapped_profile_fingerprint = profile_fingerprint
            log.info(f"[UNITY_EV:{self.agent.npc_id}] Goals iniciales: {[g.sig for g in goals]}")
            _trace(
                "goals_parsed",
                npc_id=self.agent.npc_id,
                goals=[g.sig for g in goals],
                success_conditions={g.sig: g.success_condition for g in goals},
                expected_conditions={g.sig: g.expected_condition for g in goals},
            )
            self.agent.phase = "ready"
            await self.agent.push_status()
        except asyncio.CancelledError:
            log.debug(f"[UNITY_EV:{self.agent.npc_id}] Bootstrap goals cancelado")
            return
        except Exception as exc:
            log.error(f"[UNITY_EV:{self.agent.npc_id}] Error en parse_goals: {exc}")

    def _handle_zone_discovery(self, msg: dict) -> None:
        for zone in msg.get("zones", []):
            tag = zone.get("tag", "").lower()
            center = zone.get("center", {})
            x, y = center.get("x", 0), center.get("y", 0)
            if tag:
                self.agent.beliefs.apply_zone_discovery(tag, x, y)
                log.debug(f"[UNITY_EV:{self.agent.npc_id}] Zona descubierta: {tag} ({x},{y})")
                _trace(
                    "belief_updated",
                    npc_id=self.agent.npc_id,
                    predicate="zone_center",
                    args=[tag, x, y],
                )

    def _handle_inventory_update(self, msg: dict) -> None:
        # 0.C7 — InventoryUpdate es la FUENTE AUTORITATIVA del inventario
        # (replace-all): reemplaza todo el estado has_item. Los deltas que
        # aplican PickUp/Drop/Craft son PROVISIONALES y quedan corregidos por el
        # siguiente InventoryUpdate. Si Unity no emite InventoryUpdate periódico
        # tras esas acciones, los deltas provisionales son el único estado.
        items = msg.get("items", [])
        self.agent.beliefs.apply_inventory_update(items)
        log.debug(f"[UNITY_EV:{self.agent.npc_id}] Inventario actualizado: {items}")
        for item in items:
            if isinstance(item, dict):
                _trace(
                    "belief_updated",
                    npc_id=self.agent.npc_id,
                    predicate="has_item",
                    args=[item.get("itemId", "?"), item.get("qty", 0)],
                )

    def _handle_action_result(self, msg: dict) -> None:
        cmd_id = msg.get("commandId", "")
        action_type = msg.get("actionType", "")
        status = msg.get("status", "")
        current_tick = msg.get("endedTick", 0)

        # Purgar item_at expirados en cada tick recibido
        self.agent.beliefs.expire_item_at(current_tick)

        # Actualizar current_position desde payload si la accion mueve al NPC.
        # Solo MoveTo y ExploreArea emiten current_position en su ActionResult.
        # Search, PickUp, Craft, Drop y Wait NO la devuelven.
        if status == "Success" and action_type in ("MoveTo", "ExploreArea"):
            payload = msg.get("payload") or {}
            cp = payload.get("current_position") or {}
            if isinstance(cp, dict) and "x" in cp and "y" in cp:
                px, py = int(cp["x"]), int(cp["y"])
                self.agent.beliefs.apply_current_position(px, py)
                log.debug(
                    "[UNITY_EV:%s] current_position(%d, %d)",
                    self.agent.npc_id, px, py,
                )
                _trace(
                    "belief_updated",
                    npc_id=self.agent.npc_id,
                    predicate="current_position",
                    args=[px, py],
                )
                # Fallback C1: derivar at_zone por posición cuando Unity no emite
                # ZoneEntry. Si la posición cae dentro de los bounds de una zona y
                # aún no estamos marcados en ella, assertamos at_zone(zone).
                derived_zone = self.agent.beliefs.zone_containing(px, py)
                if derived_zone and not self.agent.beliefs.has("at_zone", derived_zone):
                    self.agent.beliefs.apply_at_zone(derived_zone)
                    log.debug(
                        "[UNITY_EV:%s] +at_zone(%s) derivado por posición",
                        self.agent.npc_id, derived_zone,
                    )
                    _trace(
                        "belief_updated",
                        npc_id=self.agent.npc_id,
                        predicate="at_zone",
                        args=[derived_zone],
                        source="derived_position",
                    )

        # Extraer percepciones del payload de Search / WanderSearch / ExploreArea
        if status == "Success" and action_type in ("Search", "WanderSearch", "ExploreArea"):
            self._apply_search_hits(msg.get("payload") or {}, current_tick)

        # Limpiar posiciones conocidas del item recogido y actualizar inventario
        if status == "Success" and action_type == "PickUp":
            payload = msg.get("payload") or {}
            item_id = payload.get("itemId", "")
            qty = int(payload.get("qty", 1))
            if item_id:
                self.agent.beliefs.clear_item_at(item_id)
                # Unity sends delta qty per PickUp, not total. Accumulate.
                current = self.agent.beliefs.query("has_item", item_id.lower(), None)
                current_qty = current[0][1] if current else 0
                self.agent.beliefs.apply_has_item(item_id, current_qty + qty)
                log.debug(
                    f"[UNITY_EV:{self.agent.npc_id}] PickUp: has_item({item_id}, {qty})"
                )
                _trace(
                    "belief_updated",
                    npc_id=self.agent.npc_id,
                    predicate="has_item",
                    args=[item_id, qty],
                )

        # Actualizar inventario tras Craft exitoso (provisional desde la receta).
        # Fuente autoritativa real: InventoryUpdate (0.C7); si Unity lo emite tras
        # el Craft, sobreescribe estos deltas. Si NO lo emite, esto evita que
        # craft_item_check recurse infinito al no cumplirse nunca has_item(out, N).
        if status == "Success" and action_type == "Craft":
            self._apply_craft_inventory(msg, cmd_id)

        # Actualizar inventario tras Drop exitoso
        if status == "Success" and action_type == "Drop":
            payload = msg.get("payload") or {}
            item_id = payload.get("itemId", "")
            qty_dropped = int(payload.get("qty", 1))
            if item_id:
                self.agent.beliefs.apply_drop_item(item_id, qty_dropped)
                log.debug(
                    f"[UNITY_EV:{self.agent.npc_id}] Drop: -{qty_dropped} {item_id}"
                )

        # Escribir resultado en el dict compartido con BDIBehaviour
        self.agent.action_results[cmd_id] = msg
        log.debug(f"[UNITY_EV:{self.agent.npc_id}] ActionResult registrado: {cmd_id}")

        # Liberar el comando recordado una vez el resultado es terminal.
        if status in ("Success", "Failure"):
            self.agent.sent_commands.pop(cmd_id, None)

    def _apply_craft_inventory(self, msg: dict, cmd_id: str) -> None:
        """Actualiza el inventario tras un Craft Success usando la receta.

        Acumula los outputs y descuenta los inputs consumidos. Resuelve la receta
        por `targetId`/`recipeId` (del payload o del comando enviado); si no, por
        el item de salida. Es provisional: un InventoryUpdate posterior la corrige.
        """
        beliefs = self.agent.beliefs
        payload = msg.get("payload") or {}
        cmd = self.agent.sent_commands.get(cmd_id) or {}
        cmd_args = cmd.get("args", {}) if isinstance(cmd, dict) else {}

        recipe_id = str(
            payload.get("targetId") or payload.get("recipeId")
            or cmd_args.get("targetId") or ""
        ).lower()
        out_rows = beliefs.query("recipe_output", recipe_id, None, None, None) if recipe_id else []
        if not out_rows:
            # Fallback: el itemId del payload/comando es el item de salida.
            out_item = str(payload.get("itemId") or cmd_args.get("itemId") or "").lower()
            if out_item:
                out_rows = beliefs.query("recipe_output", None, None, out_item, None)
                if out_rows and not recipe_id:
                    recipe_id = out_rows[0][0]
        if not out_rows:
            log.warning(
                "[UNITY_EV:%s] Craft Success sin receta resoluble (cmd=%s) — inventario no actualizado",
                self.agent.npc_id, cmd_id,
            )
            return

        # Fase 17l: manda lo que Unity dice haber consumido y producido. Antes se
        # multiplicaba la receta por el `qty` del comando, que en Unity son unidades del
        # INGREDIENTE de UNA hornada: con Craft(wheat, Bread_recipe, 2) se apuntaban
        # +2 panes y -4 trigos, has_item(bread, 1) no casaba nunca y el NPC crafteaba
        # sin parar (tanda corta 2, A4 ATOM).
        def _sum_item_qty(rows: object) -> dict:
            totals: dict[str, int] = {}
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                item = str(row.get("itemId") or "").lower()
                if not item:
                    continue
                try:
                    totals[item] = totals.get(item, 0) + int(row.get("qty", 1))
                except (TypeError, ValueError):
                    totals[item] = totals.get(item, 0) + 1
            return totals

        produced_map = _sum_item_qty(payload.get("produced"))
        consumed_map = _sum_item_qty(payload.get("consumed"))
        in_rows = beliefs.query("recipe", recipe_id, None, None, None)

        # Sin listas en el payload: hornadas = qty / unidades de ingrediente por hornada.
        cmd_qty = payload.get("qty") or cmd_args.get("qty")
        batches = 1
        if cmd_qty and in_rows:
            try:
                per_batch = int(in_rows[0][3])
                batches = max(1, int(cmd_qty) // per_batch) if per_batch > 0 else 1
            except (TypeError, ValueError):
                batches = 1

        # Producir outputs (acumular sobre el inventario actual).
        for _rid, _zone, out_item, out_qty in out_rows:
            produced = produced_map.get(out_item, int(out_qty) * batches)
            current = beliefs.query("has_item", out_item, None)
            current_qty = current[0][1] if current else 0
            beliefs.apply_has_item(out_item, current_qty + produced)
            log.debug(
                "[UNITY_EV:%s] Craft: +%d %s (provisional)",
                self.agent.npc_id, produced, out_item,
            )
            _trace(
                "belief_updated", npc_id=self.agent.npc_id, predicate="has_item",
                args=[out_item, produced], source="craft_output",
            )

        # Consumir inputs: lo que diga Unity y, si no viene, la receta por hornada.
        for _rid, _zone, in_item, in_qty in in_rows:
            consumed = consumed_map.get(in_item, int(in_qty) * batches)
            beliefs.apply_drop_item(in_item, consumed)
            log.debug(
                "[UNITY_EV:%s] Craft: -%d %s (provisional)",
                self.agent.npc_id, consumed, in_item,
            )
            _trace(
                "belief_updated", npc_id=self.agent.npc_id, predicate="has_item",
                args=[in_item, -consumed], source="craft_input",
            )

    def _apply_search_hits(self, payload: dict, current_tick: int) -> None:
        """Parsea el payload de Search/WanderSearch/ExploreArea y actualiza beliefs."""
        hits: dict = payload.get("hits") or {}
        for name, hit in hits.items():
            if not isinstance(hit, dict):
                continue
            kind = hit.get("kind", "")
            x = hit.get("x", 0)
            y = hit.get("y", 0)
            if kind == "Item":
                # Key del dict es el itemId; campo "itemId" dentro del hit como fallback
                item_id = (hit.get("itemId") or name).lower()
                if item_id:
                    self.agent.beliefs.apply_item_at(item_id, x, y, current_tick)
                    log.debug(f"[UNITY_EV:{self.agent.npc_id}] item_at({item_id},{x},{y})")
            elif kind == "Zone":
                # Key del dict es el zoneTag; campo "tag" dentro del hit como fallback
                tag = (hit.get("tag") or name).lower()
                if tag:
                    self.agent.beliefs.apply_zone_discovery(tag, int(x), int(y))
                    log.debug(f"[UNITY_EV:{self.agent.npc_id}] zone_center({tag},{x},{y})")

    def _handle_zone_entry(self, msg: dict) -> None:
        zone_tag = msg.get("zone_tag", "").lower()
        entered = msg.get("entered", True)
        if not zone_tag:
            return
        if entered:
            self.agent.beliefs.apply_at_zone(zone_tag)
            log.debug(f"[UNITY_EV:{self.agent.npc_id}] +at_zone({zone_tag})")
            _trace("belief_updated", npc_id=self.agent.npc_id,
                   predicate="at_zone", args=[zone_tag])
        else:
            self.agent.beliefs.remove_at_zone(zone_tag)
            log.debug(f"[UNITY_EV:{self.agent.npc_id}] -at_zone({zone_tag})")

    def _handle_wander_retry(self, msg: dict) -> None:
        """Unity pide reintentar la búsqueda — responder con WanderContinue y seguir esperando el ActionResult.

        Unity envía WanderRetry cuando un ExploreArea necesita orientación para continuar o parar.
        La respuesta correcta es WanderContinue con action='continue'. El Future pendiente NO se resuelve
        aquí; la resolución definitiva llega con el ActionResult posterior.
        """
        cmd_id = msg.get("commandId", "")
        npc_id = self.agent.npc_id
        attempt = msg.get("attempt", 0)
        max_attempts = msg.get("maxAttempts", 10)

        # Decidir si continuar o detener la exploración
        continue_action = "continue" if attempt < max_attempts else "stop"

        async def _send_wander_continue() -> None:
            await self.agent.send_to_unity({
                "type": "WanderContinue",
                "commandId": cmd_id,
                "npcId": npc_id,
                "action": continue_action,
            })

        asyncio.create_task(_send_wander_continue())
        log.debug(
            f"[UNITY_EV:{npc_id}] WanderRetry recv cmd={cmd_id} attempt={attempt}/{max_attempts}"
            f" → WanderContinue({continue_action})"
        )


def _profile_fingerprint(profile_payload: dict) -> str:
    return json.dumps(profile_payload, sort_keys=True, ensure_ascii=False)
