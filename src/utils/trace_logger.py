from __future__ import annotations

"""Sistema de trazabilidad y métricas de sesión para el agente NPC BDI+LLM.

Genera tres ficheros en logs/sessions/YYYYMMDD/HHMMSS/:
  session.log    — log de texto idéntico al stdout (a través de FileHandler)
  trace.jsonl    — eventos estructurados (JSONL compacto, un registro por línea)
  prompts.jsonl  — prompts completos enviados al LLM + respuestas recibidas
    unity_tcp.log  — tráfico Unity<->Python y eventos de conexión/desconexión
  metrics.json   — métricas de la sesión escritas al finalizar

Uso básico:
    from utils.trace_logger import TraceLogger, trace

    # Al arrancar el sistema
    tl = TraceLogger.init_session(logs_root="logs")

    # En cualquier módulo (no-op si no hay sesión activa → tests seguros)
    trace("goal_started", npc_id="npc_001", goal="achieve_harvest_wheat")

    # Para registrar prompts LLM completos
    tl.log_prompt(npc_id="npc_001", goal="achieve_harvest_wheat",
                  step="step3_steps", prompt=user_p, system=sys_p,
                  response=raw, latency_s=2.1)

    # Para actualizar métricas
    tl.metrics.npc("npc_001").llm_calls += 1

    # Al finalizar (Ctrl+C / shutdown)
    tl.finalize()
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Registro detallado por goal
# ---------------------------------------------------------------------------

@dataclass
class GoalRecord:
    """Captura el resultado observable de un goal al cerrarse."""
    sig: str
    success_condition: str | None
    belief_met: bool | None          # True/False si había success_condition; None si no
    replan_count: int
    final_status: str                # "completed" | "failed" | "unverified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sig": self.sig,
            "success_condition": self.success_condition,
            "belief_met": self.belief_met,
            "replan_count": self.replan_count,
            "final_status": self.final_status,
        }


# ---------------------------------------------------------------------------
# Métricas por NPC
# ---------------------------------------------------------------------------

@dataclass
class NPCMetrics:
    """Contadores acumulados para un NPC durante la sesión."""
    npc_id: str
    goals_started: int = 0
    goals_completed: int = 0
    goals_failed: int = 0
    # --- Métricas de creencias (T4) ---
    goals_belief_met: int = 0          # goals donde success_condition se cumplió al cerrar
    goals_belief_missing: int = 0      # goals cerrados con success_condition NO cumplida
    goals_replan_attempted: int = 0    # replanificaciones totales intentadas
    goals_replan_success: int = 0      # replans que terminaron con creencia cumplida
    goals_unverified: int = 0          # goals sin success_condition (cierre a ciegas)
    # --- Acciones y LLM ---
    actions_sent: int = 0
    actions_ok: int = 0
    actions_failed: int = 0
    llm_calls: int = 0
    llm_total_s: float = 0.0
    repairs_attempted: int = 0
    plan_steps_total: int = 0
    # --- Cola de planificación compartida (Fase 11 T5) ---
    # Tiempo esperando a que el LLMPlanningAgent (compartido entre NPCs)
    # atienda la tarea, ANTES de que empiece a trabajar en ella. Separa
    # "esperando cola" de "trabajando" en sesiones multi-agente.
    planning_wait_s: float = 0.0
    # --- Coordinación NPC↔NPC (Fase 12) ---
    peer_msgs_sent: int = 0
    peer_msgs_received: int = 0
    peer_requests_sent: int = 0
    peer_requests_accepted: int = 0
    peer_requests_refused: int = 0
    peer_requests_completed: int = 0
    peer_requests_failed: int = 0
    items_given: int = 0
    items_received: int = 0
    peer_wait_s: float = 0.0            # tiempo bloqueado en .await_peer
    # --- Fase 14: arbitraje de goals (preempción) ---
    arbitrations_llm: int = 0           # decisiones arbitrate resueltas por el LLM
    arbitrations_rule: int = 0          # decisiones resueltas por la regla determinista (modo "rule")
    arbitrations_fallback: int = 0      # decisiones LLM que fallaron/no validaron -> cayeron a la regla
    goal_switches: int = 0              # nº de veces que _maybe_arbitrate cambió agent.intention
    arbitration_llm_s: float = 0.0      # tiempo total en llamadas LLM de arbitrate (subconjunto de llm_total_s)
    # --- Plan memory (Fase 4) ---
    plans_from_memory: int = 0          # goals cuyo plan vino de memoria approved
    plans_from_llm: int = 0             # goals cuyo plan se generó con el LLM
    # --- Detalle por goal ---
    goals_detail: list[GoalRecord] = field(default_factory=list)

    def record_goal(self, record: GoalRecord) -> None:
        """Añade un GoalRecord al detalle y actualiza los contadores correspondientes."""
        self.goals_detail.append(record)
        if record.final_status == "unverified":
            self.goals_unverified += 1
        elif record.belief_met is True:
            self.goals_belief_met += 1
            self.goals_completed += 1
        else:
            self.goals_belief_missing += 1
            if record.final_status == "failed":
                self.goals_failed += 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Serializar goals_detail manualmente (asdict ya lo hace, pero queremos to_dict)
        d["goals_detail"] = [r.to_dict() for r in self.goals_detail]
        # Métricas derivadas
        if self.actions_sent:
            d["action_success_rate"] = round(self.actions_ok / self.actions_sent, 3)
        if self.goals_started:
            d["goal_success_rate"] = round(self.goals_completed / self.goals_started, 3)
            d["belief_success_rate"] = round(self.goals_belief_met / self.goals_started, 3)
        if self.llm_calls:
            d["llm_avg_latency_s"] = round(self.llm_total_s / self.llm_calls, 3)
        return d


@dataclass
class SessionMetrics:
    """Contenedor de métricas de la sesión completa."""
    session_id: str
    started_at: float = field(default_factory=time.time)
    npcs: dict[str, NPCMetrics] = field(default_factory=dict)

    def npc(self, npc_id: str) -> NPCMetrics:
        """Devuelve (o crea) los contadores del NPC indicado."""
        if npc_id not in self.npcs:
            self.npcs[npc_id] = NPCMetrics(npc_id=npc_id)
        return self.npcs[npc_id]

    def to_dict(self, ended_at: float | None = None) -> dict[str, Any]:
        t = ended_at or time.time()
        return {
            "session_id": self.session_id,
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(),
            "duration_s": round(t - self.started_at, 3),
            "npcs": {k: v.to_dict() for k, v in self.npcs.items()},
        }


# ---------------------------------------------------------------------------
# TraceLogger — singleton por sesión
# ---------------------------------------------------------------------------

class TraceLogger:
    """
    Registra eventos estructurados en JSONL y gestiona las métricas de sesión.

    Uso:
        tl = TraceLogger.init_session("logs")
        tl.trace("action_sent", npc_id="npc_001", action="MoveTo", args={...})
        tl.log_prompt(npc_id="npc_001", goal="...", step="step3_steps",
                      prompt="...", system="...", response="...", latency_s=2.1)
        tl.finalize()
    """

    _instance: "TraceLogger | None" = None

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.session_id = f"{session_dir.parent.name}/{session_dir.name}"
        self._trace_fh = (session_dir / "trace.jsonl").open("a", encoding="utf-8")
        self._prompt_fh = (session_dir / "prompts.jsonl").open("a", encoding="utf-8")
        self._unity_fh = (session_dir / "unity_tcp.log").open("a", encoding="utf-8")
        self.metrics = SessionMetrics(session_id=self.session_id)
        self._file_handler: logging.FileHandler | None = None

    @classmethod
    def init_session(cls, logs_root: str | Path = "logs") -> "TraceLogger":
        """
        Crea una nueva sesión de log.
        Si ya existe una sesión activa, la devuelve sin crear otra.
        """
        if cls._instance is not None:
            return cls._instance

        now = datetime.now()
        session_dir = (
            Path(logs_root) / "sessions"
            / now.strftime("%Y%m%d")
            / now.strftime("%H%M%S")
        )
        session_dir.mkdir(parents=True, exist_ok=True)

        inst = cls(session_dir)

        # Añadir FileHandler al root logger para volcar texto a session.log
        # El FileHandler usa DEBUG para capturar también los prompts LLM completos.
        # La consola (StreamHandler configurado en main.basicConfig) sigue en INFO.
        fh = logging.FileHandler(session_dir / "session.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        root = logging.getLogger()
        # Bajar el nivel del root para que DEBUG pase (los handlers filtran individualmente)
        if root.level > logging.DEBUG:
            root.setLevel(logging.DEBUG)
        root.addHandler(fh)
        inst._file_handler = fh

        cls._instance = inst
        return inst

    @classmethod
    def get(cls) -> "TraceLogger | None":
        """Devuelve la instancia activa, o None si no hay sesión."""
        return cls._instance

    # ------------------------------------------------------------------
    # Registro de eventos
    # ------------------------------------------------------------------

    def trace(self, event: str, *, npc_id: str | None = None, **kw: Any) -> None:
        """
        Escribe un evento en trace.jsonl.

        Formato: {"t": <unix_ts>, "ev": "<nombre>", ["npc": "<id>"], ...campos extra}
        """
        rec: dict[str, Any] = {"t": round(time.time(), 3), "ev": event}
        if npc_id:
            rec["npc"] = npc_id
        rec.update(kw)
        self._trace_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._trace_fh.flush()

    def log_prompt(
        self,
        *,
        npc_id: str | None = None,
        goal: str | None = None,
        step: str,
        prompt: str,
        system: str,
        response: str,
        latency_s: float,
    ) -> None:
        """
        Escribe en prompts.jsonl el prompt completo enviado al LLM y su respuesta.
        Estos registros son grandes; se guardan en fichero separado de trace.jsonl.
        """
        rec: dict[str, Any] = {
            "t": round(time.time(), 3),
            "step": step,
            "latency_s": latency_s,
        }
        if npc_id:
            rec["npc"] = npc_id
        if goal:
            rec["goal"] = goal
        rec["system"] = system
        rec["prompt"] = prompt
        rec["response"] = response
        self._prompt_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._prompt_fh.flush()

        # Resumen compacto en trace.jsonl para filtrado rápido de eventos.
        self.trace(
            "llm_prompt_logged",
            npc_id=npc_id,
            goal=goal,
            step=step,
            latency_s=latency_s,
            prompt_chars=len(prompt),
            response_chars=len(response),
        )

        # Vista legible en session.log (previews, no contenido completo).
        _prompt_preview = prompt.replace("\n", " ").strip()[:240]
        _response_preview = response.replace("\n", " ").strip()[:240]
        logging.getLogger("trace.prompt").info(
            "[LLM_LOG] npc=%s goal=%s step=%s latency=%.3fs prompt_chars=%d response_chars=%d",
            npc_id or "-",
            goal or "-",
            step,
            latency_s,
            len(prompt),
            len(response),
        )
        logging.getLogger("trace.prompt").info("[LLM_LOG] prompt: %s", _prompt_preview)
        logging.getLogger("trace.prompt").info("[LLM_LOG] response: %s", _response_preview)

    def log_unity(
        self,
        *,
        direction: str,
        npc_id: str | None = None,
        msg_type: str | None = None,
        payload: Any | None = None,
        raw: str | None = None,
        note: str | None = None,
    ) -> None:
        """Write Unity-side traffic/events to unity_tcp.log as JSON lines."""
        rec: dict[str, Any] = {
            "t": round(time.time(), 3),
            "dir": direction,
        }
        if npc_id:
            rec["npc"] = npc_id
        if msg_type:
            rec["msg_type"] = msg_type
        if payload is not None:
            rec["payload"] = payload
        if raw is not None:
            rec["raw"] = raw
        if note:
            rec["note"] = note
        self._unity_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._unity_fh.flush()

    # ------------------------------------------------------------------
    # Cierre de sesión
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """
        Escribe metrics.json, cierra los ficheros y resetea el singleton.
        Llamar una sola vez al apagar el sistema.
        """
        ended = time.time()
        self.trace("session_end", duration_s=round(ended - self.metrics.started_at, 3))

        metrics_path = self.session_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(self.metrics.to_dict(ended_at=ended), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._trace_fh.close()
        self._prompt_fh.close()
        self._unity_fh.close()

        if self._file_handler:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()

        TraceLogger._instance = None


# ---------------------------------------------------------------------------
# Helpers de módulo
# ---------------------------------------------------------------------------

def trace(event: str, *, npc_id: str | None = None, **kw: Any) -> None:
    """
    Helper de módulo: escribe un evento si hay sesión activa.
    No-op en tests (cuando no se inicializa TraceLogger).
    """
    tl = TraceLogger.get()
    if tl is not None:
        tl.trace(event, npc_id=npc_id, **kw)


def plan_transform(
    kind: str,
    source: str,
    *,
    npc_id: str | None = None,
    before: Any = None,
    after: Any = None,
    reason: str = "",
    **kw: Any,
) -> None:
    """Fase 6.5: traza una transformación de un plan con su AUTORÍA.

    `source` debe ser "LLM" (el modelo produjo/reparó el contenido) o "CODE"
    (corrección clara o andamiaje generado por el orquestador). `kind` describe
    la transformación (p.ej. "craft_reorder", "type_normalize", "scaffold_done",
    "mini_repair_qty"). No-op si no hay sesión activa (tests).
    """
    trace(
        "plan_transform",
        npc_id=npc_id,
        kind=kind,
        source=source,
        before=before,
        after=after,
        reason=reason,
        **kw,
    )


def unity_log(
    *,
    direction: str,
    npc_id: str | None = None,
    msg_type: str | None = None,
    payload: Any | None = None,
    raw: str | None = None,
    note: str | None = None,
) -> None:
    """Module helper: writes to unity.log when a session is active."""
    tl = TraceLogger.get()
    if tl is not None:
        tl.log_unity(
            direction=direction,
            npc_id=npc_id,
            msg_type=msg_type,
            payload=payload,
            raw=raw,
            note=note,
        )
