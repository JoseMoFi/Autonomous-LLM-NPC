from __future__ import annotations

"""mcp_world_server.py — Servidor MCP de mundo y observabilidad (Fase 6, opción A).

Expone, en modo SOLO LECTURA, el conocimiento del sistema multiagente BDI+LLM a
cualquier cliente MCP (Claude Code/Desktop, un LLMAgent de spade-llm, …):

  Tools:
    - get_action_catalog()                  catálogo de acciones primitivas + contratos
    - get_capability_contracts()            src/plans/contracts/capability_contracts.json
    - get_beliefs(npc_id, session?)         snapshot de beliefs reconstruido del trace
    - get_plan_memory(npc_id)               planes aprendidos con sus stats
    - validate_asl(asl, known_goals?)       envuelve el validador del proyecto
    - list_sessions()                       sesiones bajo logs/sessions/
    - get_session_summary(session?)         métricas agregadas de una sesión
    - get_trace_events(session?, ...)       eventos del trace.jsonl (filtrables)
    - read_session_artifact(session, name)  contenido de trace/prompts/*.asl/*.dag.json

  Resources (navegables):
    - world://catalog/actions               igual que get_action_catalog()
    - world://contracts/capabilities        igual que get_capability_contracts()
    - world://sessions                      igual que list_sessions()

NINGÚN tool muta el estado del juego ni del repositorio. El servidor NO se conecta
al runtime vivo: lee artefactos persistidos (catálogos en código, JSON de memoria,
trazas de sesión).

Uso como proceso MCP (STDIO):
    .venv\\Scripts\\python.exe tools/mcp_world_server.py

Registro en Claude Code:
    claude mcp add world -- <python> <ruta>/tools/mcp_world_server.py
(ver DOC/MCP_SERVER.md).
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths e imports del proyecto (para poder ejecutar el servidor standalone).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
_TOOLS = _PROJECT_ROOT / "tools"
for _p in (_SRC, _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from protocol.action_contract import ACTION_ALLOWLIST  # noqa: E402
from protocol.action_semantics import CONTRACT_REGISTRY  # noqa: E402
from llm.validator import validate_plan_asl  # noqa: E402

import analyze_sessions as _A  # noqa: E402  (tools/ está en sys.path)


# Directorios por defecto (parametrizables en cada función para los tests).
DEFAULT_SESSIONS_ROOT = _PROJECT_ROOT / "logs" / "sessions"
DEFAULT_MEMORY_ROOT = _PROJECT_ROOT / "plans" / "memory"
CAPABILITY_CONTRACTS_PATH = _SRC / "plans" / "contracts" / "capability_contracts.json"

# Artefactos legibles dentro de un directorio de sesión.
_ARTIFACT_SUFFIXES = (".jsonl", ".asl", ".json", ".dag.json")


# ===========================================================================
# Lógica de negocio (funciones puras, testeables sin transporte MCP)
# ===========================================================================

def _belief_spec_to_dict(spec: Any) -> dict[str, Any]:
    return {"functor": spec.functor, "args": list(spec.args), "note": spec.note}


def _contract_to_dict(contract: Any) -> dict[str, Any]:
    return {
        "attempt_budget": contract.attempt_budget,
        "on_exhaustion": contract.on_exhaustion,
        "requires": [_belief_spec_to_dict(b) for b in contract.requires],
        "guarantees_on_success": [_belief_spec_to_dict(b) for b in contract.guarantees_on_success],
        "may_observe": [_belief_spec_to_dict(b) for b in contract.may_observe],
        "invalidates": [_belief_spec_to_dict(b) for b in contract.invalidates],
        "consumes": [_belief_spec_to_dict(b) for b in contract.consumes],
        "binds": list(contract.binds),
        "notes": contract.notes,
    }


def get_action_catalog() -> dict[str, Any]:
    """Catálogo de acciones primitivas con su firma y su contrato semántico."""
    actions: list[dict[str, Any]] = []
    for name, spec in ACTION_ALLOWLIST.items():
        contract = CONTRACT_REGISTRY.get(name)
        actions.append({
            "name": name,
            "synopsis": spec.get("synopsis", name),
            "description": " ".join(str(spec.get("description", "")).split()),
            "required": list(spec.get("required", [])),
            "optional": list(spec.get("optional", [])),
            "asl_args": list(spec.get("asl_args", [])),
            "contract": _contract_to_dict(contract) if contract is not None else None,
        })
    return {"count": len(actions), "actions": actions}


def get_capability_contracts(path: str | Path | None = None) -> dict[str, Any]:
    """Lee src/plans/contracts/capability_contracts.json (solo lectura)."""
    p = Path(path) if path else CAPABILITY_CONTRACTS_PATH
    if not p.exists():
        return {"contracts": {}, "version": None, "_note": f"no existe: {p}"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"contracts": {}, "version": None, "_error": f"JSON inválido en {p}: {exc}"}


def reconstruct_beliefs(events: list[dict], npc_id: str | None = None) -> dict[str, Any]:
    """Reconstruye un snapshot de beliefs replayando eventos `belief_updated`.

    Reglas (documentadas porque el trace solo registra altas/actualizaciones):
      - has_item: cantidades. Las fuentes craft_output/craft_input son deltas
        (+producido / -consumido); el resto (InventoryUpdate, PickUp) son
        absolutas.
      - current_position: último valor.
      - zone_center: último valor por tag.
      - at_zone: presencia (las salidas de zona NO se trazan → puede quedar stale).
      - cualquier otro predicado: lista de tuplas de args distintas (última vista).
    """
    has_item: dict[str, float] = {}
    position: list[Any] | None = None
    zone_center: dict[str, list[Any]] = {}
    at_zone: set[str] = set()
    other: dict[str, dict[tuple, list[Any]]] = defaultdict(dict)

    for ev in events:
        if ev.get("ev") != "belief_updated":
            continue
        if npc_id is not None and ev.get("npc") != npc_id:
            continue
        pred = ev.get("predicate")
        args = ev.get("args", []) or []
        source = ev.get("source")

        if pred == "has_item" and len(args) >= 2:
            item, qty = args[0], args[1]
            if source in ("craft_output", "craft_input"):
                has_item[item] = has_item.get(item, 0) + qty
            else:
                has_item[item] = qty
        elif pred == "current_position":
            position = list(args)
        elif pred == "zone_center" and args:
            zone_center[args[0]] = list(args[1:])
        elif pred == "at_zone" and args:
            at_zone.add(args[0])
        elif pred:
            other[pred][tuple(args)] = list(args)

    snapshot: dict[str, Any] = {
        "has_item": dict(has_item),
        "current_position": position,
        "zone_center": zone_center,
        "at_zone": sorted(at_zone),
    }
    for pred, tuples in other.items():
        snapshot[pred] = list(tuples.values())
    return snapshot


def get_beliefs(
    npc_id: str,
    session: str | None = None,
    sessions_root: str | Path | None = None,
) -> dict[str, Any]:
    """Snapshot de beliefs de un NPC reconstruido del trace de una sesión.

    Si `session` es None usa la sesión más reciente. NO se conecta al runtime.
    """
    session_dir = _resolve_session(session, sessions_root)
    events = _A._read_jsonl(session_dir / "trace.jsonl")
    snapshot = reconstruct_beliefs(events, npc_id)
    return {
        "npc_id": npc_id,
        "session": _session_id(session_dir, sessions_root),
        "beliefs": snapshot,
        "_note": (
            "Reconstruido del trace.jsonl (eventos belief_updated). Las "
            "retracciones que Unity no traza (p.ej. salida de zona) pueden "
            "quedar reflejadas como stale. No es el estado vivo del runtime."
        ),
    }


def get_plan_memory(npc_id: str, memory_root: str | Path | None = None) -> dict[str, Any]:
    """Planes aprendidos de un NPC con sus stats (pending + approved).

    Busca tanto en `<root>/<npc>/` como en `<root>/runs/<ts>/<npc>/` (memoria por
    ejecución de la Fase 4).
    """
    base = Path(memory_root) if memory_root else DEFAULT_MEMORY_ROOT
    npc_dirs: list[Path] = []
    if (base / npc_id).is_dir():
        npc_dirs.append(base / npc_id)
    runs = base / "runs"
    if runs.is_dir():
        for run in sorted(runs.iterdir()):
            if (run / npc_id).is_dir():
                npc_dirs.append(run / npc_id)

    fields = (
        "goal_sig", "status", "uses_success", "uses_error", "uses_unverified",
        "success_rate", "description", "guard", "created_at", "last_used_at",
        "promoted_at",
    )
    plans: list[dict[str, Any]] = []
    for npc_dir in npc_dirs:
        for status in ("approved", "pending"):
            sub = npc_dir / status
            if not sub.is_dir():
                continue
            for jp in sorted(sub.glob("*.json")):
                try:
                    rec = json.loads(jp.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                entry = {k: rec.get(k) for k in fields if k in rec}
                try:
                    entry["source"] = str(jp.relative_to(base))
                except ValueError:
                    entry["source"] = str(jp)
                plans.append(entry)
    return {"npc_id": npc_id, "count": len(plans), "memory_root": str(base), "plans": plans}


def validate_asl(asl: str, known_goals: Optional[list[str]] = None) -> dict[str, Any]:
    """Valida un texto ASL con el validador del proyecto (`validate_plan_asl`).

    Si no se pasan `known_goals`, se derivan de las cabeceras `+!sig` del propio
    ASL (para validar un plan autocontenido). Separa errores duros de warnings.
    """
    if known_goals is None:
        goals = set(re.findall(r"\+!([a-z][a-z0-9_]*)", asl))
    else:
        goals = set(known_goals)
    all_msgs = validate_plan_asl(asl, goals)
    errors = [m for m in all_msgs if not m.startswith("WARNING")]
    warnings = [m for m in all_msgs if m.startswith("WARNING")]
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "known_goals": sorted(goals),
    }


# ---------------------------------------------------------------------------
# Observabilidad de sesiones (reusa tools/analyze_sessions.py)
# ---------------------------------------------------------------------------

def _session_id(session_dir: Path, sessions_root: str | Path | None) -> str:
    root = Path(sessions_root) if sessions_root else DEFAULT_SESSIONS_ROOT
    try:
        return str(session_dir.relative_to(root))
    except ValueError:
        return session_dir.name


def _resolve_session(session: str | None, sessions_root: str | Path | None) -> Path:
    root = Path(sessions_root) if sessions_root else DEFAULT_SESSIONS_ROOT
    if session:
        cand = root / session
        if cand.exists():
            return cand
        p = Path(session)
        if p.exists():
            return p
        raise FileNotFoundError(f"sesión no encontrada: {session} (root={root})")
    dirs = _A.find_sessions(root)
    if not dirs:
        raise FileNotFoundError(f"no hay sesiones con trace.jsonl bajo {root}")
    return dirs[-1]


def list_sessions(sessions_root: str | Path | None = None) -> dict[str, Any]:
    """Lista las sesiones (dirs con trace.jsonl) bajo logs/sessions/."""
    root = Path(sessions_root) if sessions_root else DEFAULT_SESSIONS_ROOT
    if not root.exists():
        return {"root": str(root), "count": 0, "sessions": []}
    dirs = _A.find_sessions(root)
    sessions = [{"session": _session_id(d, sessions_root), "path": str(d)} for d in dirs]
    return {"root": str(root), "count": len(sessions), "sessions": sessions}


def get_session_summary(
    session: str | None = None,
    sessions_root: str | Path | None = None,
) -> dict[str, Any]:
    """Métricas agregadas de una sesión (planning/pipeline/ejecución)."""
    session_dir = _resolve_session(session, sessions_root)
    return _A.summarize_session(session_dir)


def get_trace_events(
    session: str | None = None,
    ev_filter: str | list[str] | None = None,
    npc: str | None = None,
    limit: int = 500,
    sessions_root: str | Path | None = None,
) -> dict[str, Any]:
    """Eventos del trace.jsonl de una sesión, filtrables por tipo (`ev`) y npc.

    `limit` recorta a los últimos N eventos (0 = sin límite).
    """
    session_dir = _resolve_session(session, sessions_root)
    events = _A._read_jsonl(session_dir / "trace.jsonl")

    if ev_filter:
        wanted = {ev_filter} if isinstance(ev_filter, str) else set(ev_filter)
        events = [e for e in events if e.get("ev") in wanted]
    if npc:
        # Conserva eventos del npc pedido y los globales (sin clave "npc").
        events = [e for e in events if e.get("npc") == npc or "npc" not in e]

    total = len(events)
    if limit and total > limit:
        events = events[-limit:]
    return {
        "session": _session_id(session_dir, sessions_root),
        "total": total,
        "returned": len(events),
        "events": events,
    }


def read_session_artifact(
    session: str,
    name: str,
    sessions_root: str | Path | None = None,
) -> dict[str, Any]:
    """Devuelve el contenido textual de un artefacto dentro del dir de sesión.

    Solo se permiten ficheros con sufijo conocido (trace/prompts/*.asl/*.dag.json)
    y que resuelvan DENTRO del directorio de la sesión (sin path traversal).
    """
    session_dir = _resolve_session(session, sessions_root)
    target = (session_dir / name).resolve()
    # Anti path-traversal: el fichero debe estar dentro de la sesión.
    if session_dir.resolve() not in target.parents and target != session_dir.resolve():
        raise ValueError(f"ruta fuera de la sesión: {name}")
    if not any(name.endswith(suf) for suf in _ARTIFACT_SUFFIXES):
        raise ValueError(f"sufijo no permitido: {name} (permitidos: {_ARTIFACT_SUFFIXES})")
    if not target.is_file():
        raise FileNotFoundError(f"no existe: {name} en {session_dir}")
    return {
        "session": _session_id(session_dir, sessions_root),
        "name": name,
        "content": target.read_text(encoding="utf-8"),
    }


# ===========================================================================
# Servidor MCP (transporte) — envoltorios finos sobre la lógica de negocio
# ===========================================================================

def build_server() -> Any:
    """Construye el servidor FastMCP. Import perezoso para no exigir `mcp` en tests."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("world-observability")

    # --- Tools ---
    @mcp.tool()
    def action_catalog() -> dict:
        """Catálogo de acciones primitivas (firma + contrato semántico)."""
        return get_action_catalog()

    @mcp.tool()
    def capability_contracts() -> dict:
        """Contratos de capacidad declarados (capability_contracts.json)."""
        return get_capability_contracts()

    @mcp.tool()
    def beliefs(npc_id: str, session: str | None = None) -> dict:
        """Snapshot de beliefs de un NPC reconstruido del trace de una sesión."""
        return get_beliefs(npc_id, session=session)

    @mcp.tool()
    def plan_memory(npc_id: str) -> dict:
        """Planes aprendidos de un NPC (pending + approved) con sus stats."""
        return get_plan_memory(npc_id)

    @mcp.tool()
    def validate_plan(asl: str, known_goals: list[str] | None = None) -> dict:
        """Valida un texto ASL con el validador del pipeline."""
        return validate_asl(asl, known_goals)

    @mcp.tool()
    def sessions() -> dict:
        """Lista las sesiones disponibles bajo logs/sessions/."""
        return list_sessions()

    @mcp.tool()
    def session_summary(session: str | None = None) -> dict:
        """Métricas agregadas de una sesión (la última si no se indica)."""
        return get_session_summary(session)

    @mcp.tool()
    def trace_events(
        session: str | None = None,
        ev_filter: list[str] | None = None,
        npc: str | None = None,
        limit: int = 500,
    ) -> dict:
        """Eventos del trace.jsonl de una sesión, filtrables por tipo y npc."""
        return get_trace_events(session, ev_filter=ev_filter, npc=npc, limit=limit)

    @mcp.tool()
    def session_artifact(session: str, name: str) -> dict:
        """Contenido de un artefacto (trace/prompts/*.asl/*.dag.json) de la sesión."""
        return read_session_artifact(session, name)

    # --- Resources navegables ---
    @mcp.resource("world://catalog/actions")
    def _res_actions() -> str:
        return json.dumps(get_action_catalog(), ensure_ascii=False, indent=2)

    @mcp.resource("world://contracts/capabilities")
    def _res_contracts() -> str:
        return json.dumps(get_capability_contracts(), ensure_ascii=False, indent=2)

    @mcp.resource("world://sessions")
    def _res_sessions() -> str:
        return json.dumps(list_sessions(), ensure_ascii=False, indent=2)

    return mcp


def main() -> int:
    server = build_server()
    server.run()  # STDIO por defecto
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
