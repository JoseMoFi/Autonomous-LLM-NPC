from __future__ import annotations

import asyncio
import json
import logging
import uuid

from utils.trace_logger import trace as _trace
from utils.trace_logger import unity_log as _unity_log
from utils.trace_logger import TraceLogger

log = logging.getLogger(__name__)

# Mensaje types that carry npc_id and go directly to a specific NPCAgent inbox.
_NPC_MESSAGES = {"NPCProfile", "ActionResult", "ZoneDiscovery", "InventoryUpdate", "WanderRetry", "ZoneEntry"}

# Message types handled at the gateway level (no npc_id routing needed).
_GATEWAY_MESSAGES = {"RegisterNPC", "Ping"}


class MessageRouter:
    """
    Enruta mensajes Unity -> NPCAgent correcto.
    No tiene logica de negocio: solo despacha por npc_id y tipo.
    """

    def __init__(self, registry: "NPCRegistry"):  # type: ignore[name-defined]
        self.registry = registry

    async def route(self, msg: dict, unity_writer: asyncio.StreamWriter) -> None:
        msg_type = msg.get("type", "")
        npc_hint = msg.get("npc_id") or msg.get("npcId") or None
        log.info("[UNITY_IN] type=%s npc=%s", msg_type or "?", npc_hint or "-")
        _trace("unity_in", npc_id=npc_hint, msg_type=msg_type, payload=msg)
        _unity_log(direction="in", npc_id=npc_hint, msg_type=msg_type, payload=msg)

        if msg_type == "RegisterNPC":
            npc_id = msg.get("npc_id", "")
            if not npc_id:
                log.warning("[ROUTER] RegisterNPC sin npc_id -- ignorado")
                return
            agent = await self.registry.register(npc_id, unity_writer)
            # Sembrar current_position desde RegisterNPC.pos (fuente autoritativa inicial).
            # Nunca desde el comando enviado; solo desde confirmacion de Unity.
            pos = msg.get("pos") or {}
            if isinstance(pos, dict) and "x" in pos and "y" in pos:
                agent.beliefs.apply_current_position(int(pos["x"]), int(pos["y"]))
                log.debug(
                    "[ROUTER] current_position sembrada desde RegisterNPC: (%s, %s)",
                    pos["x"], pos["y"],
                )
            # Confirmar a Unity con AssignAgent (compatibilidad con el cliente actual).
            jid = f"{npc_id}@localhost"
            reply_v1 = json.dumps(
                {
                    "type": "AssignAgent",
                    "npc_id": npc_id,
                    "jid": jid,
                    "agent_id": jid,
                    "agentId": jid,
                    "status": "ready",
                }
            )
            unity_writer.write(reply_v1.encode() + b"\n")
            await unity_writer.drain()
            log.info("[UNITY_OUT] type=AssignAgent npc=%s", npc_id)
            _trace("unity_out", npc_id=npc_id, msg_type="AssignAgent")
            _unity_log(direction="out", npc_id=npc_id, msg_type="AssignAgent", payload={
                "type": "AssignAgent",
                "npc_id": npc_id,
                "jid": jid,
                "agent_id": jid,
                "agentId": jid,
                "status": "ready",
            })

            tl = TraceLogger.get()
            if tl is not None:
                unity_log_cfg = {
                    "type": "UnityLogConfig",
                    "msg_id": uuid.uuid4().hex,
                    "directory": str(tl.session_dir),
                    "fileName": "unity.log",
                    "append": True,
                }
                unity_writer.write(json.dumps(unity_log_cfg).encode() + b"\n")
                await unity_writer.drain()
                log.info("[UNITY_OUT] type=UnityLogConfig npc=%s dir=%s", npc_id, tl.session_dir)
                _trace("unity_out", npc_id=npc_id, msg_type="UnityLogConfig", payload=unity_log_cfg)
                _unity_log(direction="out", npc_id=npc_id, msg_type="UnityLogConfig", payload=unity_log_cfg)
            return

        if msg_type == "Ping":
            msg_id = msg.get("msg_id", "")
            reply = json.dumps({"type": "Pong", "msg_id": msg_id})
            unity_writer.write(reply.encode() + b"\n")
            await unity_writer.drain()
            log.info("[UNITY_OUT] type=Pong npc=-")
            _trace("unity_out", msg_type="Pong", msg_id=msg_id)
            _unity_log(direction="out", msg_type="Pong", payload={"type": "Pong", "msg_id": msg_id})
            return

        if msg_type in _NPC_MESSAGES:
            npc_id = msg.get("npc_id") or msg.get("npcId", "")
            if not npc_id:
                log.warning(f"[ROUTER] Mensaje '{msg_type}' sin npc_id -- ignorado")
                return
            agent = self.registry.get(npc_id)
            if agent is None:
                log.warning(f"[ROUTER] NPC '{npc_id}' no registrado -- ignorando '{msg_type}'")
                return
            await agent.inbox.put(msg)
            return

        log.debug(f"[ROUTER] Tipo desconocido '{msg_type}' -- ignorado")
