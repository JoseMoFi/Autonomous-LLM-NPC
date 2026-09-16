from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from utils.trace_logger import trace as _trace
from utils.trace_logger import unity_log as _unity_log

if TYPE_CHECKING:
    from npc.agent import NPCAgent

log = logging.getLogger(__name__)


class NPCRegistry:
    """
    Mantiene el directorio de NPCAgents registrados.
    Reutiliza agentes existentes si Unity reconecta con el mismo npc_id.
    """

    def __init__(self, llm_planning_jid: str, plan_memory_root=None):
        self._agents: dict[str, "NPCAgent"] = {}
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._llm_planning_jid = llm_planning_jid
        # Run root de plan memory (resuelto una vez en main.py). None = desactivada.
        self._plan_memory_root = plan_memory_root

    async def register(self, npc_id: str, unity_writer: asyncio.StreamWriter) -> "NPCAgent":
        """Registra un NPC declarado por Unity. Reutiliza si ya existe."""
        from npc.agent import NPCAgent

        self._writers[npc_id] = unity_writer

        if npc_id not in self._agents:
            agent = NPCAgent(
                jid=f"{npc_id}@localhost",
                password="npc",
                npc_id=npc_id,
                send_to_unity=self._make_sender(npc_id),
                llm_planning_jid=self._llm_planning_jid,
                plan_memory_root=self._plan_memory_root,
            )
            self._agents[npc_id] = agent
            asyncio.create_task(agent.start())
            log.info(f"[REGISTRY] NPCAgent '{npc_id}' levantado")
            self.announce_peers(npc_id)
        else:
            # Reconexión: actualizar writer sin reiniciar el agente
            self._agents[npc_id].send_to_unity = self._make_sender(npc_id)
            log.info(f"[REGISTRY] NPCAgent '{npc_id}' reconectado (writer actualizado)")

        return self._agents[npc_id]

    def get(self, npc_id: str) -> "NPCAgent | None":
        return self._agents.get(npc_id)

    def announce_peers(self, new_npc_id: str) -> None:
        """Fase 11 (T6): siembra `peer/2` cruzado entre el NPC recién
        registrado y todos los demás ya registrados. Solo existencia + rol
        (o "unknown" si el NPCProfile aún no ha llegado). Semilla del
        directorio de agentes que usará la coordinación (Fase 12); hoy no
        dispara ninguna decisión BDI — el belief no está en ningún guard.
        """
        new_agent = self._agents.get(new_npc_id)
        if new_agent is None:
            return
        others = [(oid, oa) for oid, oa in self._agents.items() if oid != new_npc_id]
        if not others:
            return

        def _role(agent: "NPCAgent") -> str:
            profile = getattr(agent, "profile", None)
            return getattr(profile, "role", None) or "unknown"

        new_role = _role(new_agent)
        for other_id, other_agent in others:
            other_agent.beliefs.apply_peer(new_npc_id, new_role)
            new_agent.beliefs.apply_peer(other_id, _role(other_agent))

        peer_ids = [oid for oid, _ in others]
        log.info(f"[REGISTRY] peer_directory: '{new_npc_id}' <-> {peer_ids}")
        _trace("peer_directory", npc_id=new_npc_id, peers=peer_ids)
        for other_id, _other_agent in others:
            _trace("peer_directory", npc_id=other_id, peers=[new_npc_id])

    def all_npc_ids(self) -> list[str]:
        return list(self._agents.keys())

    def open_goals(self) -> list[tuple[str, object]]:
        """(npc_id, goal) de cada goal aún sin resolver. Lo usa el watchdog de
        duración máxima de sesión para registrarlos como timeout (Fase 16)."""
        return [
            (npc_id, goal)
            for npc_id, agent in self._agents.items()
            for goal in list(getattr(agent, "goals", None) or [])
        ]

    def work_done(self) -> bool:
        """True si hay al menos un NPC registrado y TODOS han resuelto su trabajo
        (is_work_done). Lo consulta el watchdog de apagado por inactividad."""
        if not self._agents:
            return False
        agents = list(self._agents.values())
        # Fase 17 (multi-NPC): basta con que ALGÚN NPC tuviera goals; los que no
        # traen goals propios (p.ej. el que solo ayuda) cuentan como resueltos
        # cuando están ociosos. Con un solo NPC equivale a la regla anterior.
        # Objetos que solo exponen is_work_done (regla previa) se evalúan con ella.

        def _idle(agent) -> bool:
            check = getattr(agent, "is_idle_for_shutdown", None)
            return bool(check()) if callable(check) else bool(agent.is_work_done())

        def _had_goals(agent) -> bool:
            if hasattr(agent, "_had_goals"):
                return bool(agent._had_goals)
            return bool(agent.is_work_done())

        return any(_had_goals(a) for a in agents) and all(_idle(a) for a in agents)

    def _make_sender(self, npc_id: str):
        async def send(msg: dict) -> None:
            writer = self._writers.get(npc_id)
            if writer and not writer.is_closing():
                writer.write(json.dumps(msg).encode() + b"\n")
                await writer.drain()
                msg_type = msg.get("type", "") if isinstance(msg, dict) else ""
                log.info("[UNITY_OUT] type=%s npc=%s", msg_type or "?", npc_id)
                _trace("unity_out", npc_id=npc_id, msg_type=msg_type, payload=msg)
                _unity_log(direction="out", npc_id=npc_id, msg_type=msg_type, payload=msg)
            else:
                log.warning("[UNITY_OUT] drop npc=%s (writer cerrado)", npc_id)
                _trace("unity_out_dropped", npc_id=npc_id)
                _unity_log(direction="out_dropped", npc_id=npc_id, note="writer_closed")
        return send

    async def stop_all(self) -> None:
        for npc_id, agent in self._agents.items():
            log.info(f"[REGISTRY] Deteniendo {npc_id}...")
            await agent.stop()
        self._agents.clear()
        self._writers.clear()

    def pause_all(self) -> None:
        for agent in self._agents.values():
            agent.paused = True

    def resume_all(self) -> None:
        for agent in self._agents.values():
            agent.paused = False

    async def handle_query_npc(self, npc_id: str) -> dict:
        """Devuelve el JID del NPC si está registrado (para coordinación NPC↔NPC).

        unused — reservado para coordinación NPC↔NPC (Fase 7). No lo llama nadie aún.
        """
        agent = self._agents.get(npc_id)
        jid = f"{npc_id}@localhost" if agent else None
        return {"type": "QueryNPCResult", "npc_id": npc_id, "jid": jid}
