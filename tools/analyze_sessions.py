from __future__ import annotations

"""analyze_sessions.py — Análisis de sesiones para la evaluación del TFM (Fase 5).

Lee las sesiones bajo logs/sessions/ (trace.jsonl + metrics.json) y produce
tablas agregadas de planning / pipeline / ejecución, en markdown y/o CSV.

Uso:
    python tools/analyze_sessions.py logs/sessions
    python tools/analyze_sessions.py logs/sessions --csv out/
    python tools/analyze_sessions.py logs/sessions --md > out/eval.md

No toca el runtime: solo explota las trazas que el sistema ya genera.
"""

import argparse
import csv as _csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def find_sessions(root: Path) -> list[Path]:
    """Devuelve los directorios de sesión (los que tienen trace.jsonl)."""
    if (root / "trace.jsonl").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("trace.jsonl"))


def _read_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Resumen por sesión
# ---------------------------------------------------------------------------

def summarize_session(session_dir: Path) -> dict[str, Any]:
    events = _read_jsonl(session_dir / "trace.jsonl")
    metrics = _read_json(session_dir / "metrics.json")

    # --- Agregados de metrics.json (suma sobre NPCs) ---
    npcs_dict = metrics.get("npcs") or {}
    npcs = npcs_dict.values()
    def _sum(key: str) -> float:
        return sum(float(n.get(key, 0) or 0) for n in npcs)

    goals_started = _sum("goals_started")
    goals_completed = _sum("goals_completed")
    goals_belief_met = _sum("goals_belief_met")
    plans_from_memory = _sum("plans_from_memory")
    plans_from_llm = _sum("plans_from_llm")
    llm_calls_m = _sum("llm_calls")
    llm_total_s = _sum("llm_total_s")
    actions_sent = _sum("actions_sent")
    actions_ok = _sum("actions_ok")
    actions_failed = _sum("actions_failed")
    planning_wait_s = _sum("planning_wait_s")
    # --- Coordinación NPC↔NPC (Fase 12) ---
    peer_requests_sent = _sum("peer_requests_sent")
    peer_requests_accepted = _sum("peer_requests_accepted")
    peer_requests_refused = _sum("peer_requests_refused")
    peer_requests_completed = _sum("peer_requests_completed")
    peer_requests_failed = _sum("peer_requests_failed")
    items_given = _sum("items_given")
    items_received = _sum("items_received")
    peer_wait_s = _sum("peer_wait_s")
    # --- Arbitraje de goals / preempción (Fase 14) ---
    goal_switches = _sum("goal_switches")
    arbitrations_llm = _sum("arbitrations_llm")
    arbitrations_rule = _sum("arbitrations_rule")
    arbitrations_fallback = _sum("arbitrations_fallback")
    arbitration_llm_s = _sum("arbitration_llm_s")

    # --- Desglose por NPC (Fase 11 — sesiones multi-agente) ---
    # Cada entrada trae los contadores propios del NPC tal cual metrics.json,
    # más session_id para poder identificarlo tras agregar varias sesiones.
    npc_count = len(npcs_dict)
    by_npc: dict[str, dict[str, Any]] = {}
    for npc_id, n in npcs_dict.items():
        by_npc[npc_id] = {
            "session_id": metrics.get("session_id") or session_dir.name,
            "goals_started": n.get("goals_started", 0),
            "goals_completed": n.get("goals_completed", 0),
            "goals_belief_met": n.get("goals_belief_met", 0),
            "llm_calls": n.get("llm_calls", 0),
            "llm_total_s": n.get("llm_total_s", 0.0),
            "planning_wait_s": n.get("planning_wait_s", 0.0),
            "actions_sent": n.get("actions_sent", 0),
            "actions_ok": n.get("actions_ok", 0),
            "actions_failed": n.get("actions_failed", 0),
            "peer_requests_sent": n.get("peer_requests_sent", 0),
            "peer_requests_accepted": n.get("peer_requests_accepted", 0),
            "peer_requests_refused": n.get("peer_requests_refused", 0),
            "peer_requests_completed": n.get("peer_requests_completed", 0),
            "peer_requests_failed": n.get("peer_requests_failed", 0),
            "items_given": n.get("items_given", 0),
            "items_received": n.get("items_received", 0),
            "peer_wait_s": n.get("peer_wait_s", 0.0),
            "goal_switches": n.get("goal_switches", 0),
            "arbitrations_llm": n.get("arbitrations_llm", 0),
            "arbitrations_rule": n.get("arbitrations_rule", 0),
            "arbitrations_fallback": n.get("arbitrations_fallback", 0),
            "arbitration_llm_s": n.get("arbitration_llm_s", 0.0),
        }

    # --- Detalle desde trace.jsonl ---
    by_ev: dict[str, int] = defaultdict(int)
    llm_latency_by_step: dict[str, list[float]] = defaultdict(list)
    structured_calls: dict[str, int] = defaultdict(int)
    action_by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "fail": 0})
    action_latencies: list[float] = []
    intent_matches: list[bool] = []
    belief_mets: list[bool] = []
    plan_failed_reasons: list[str] = []

    for ev in events:
        name = ev.get("ev", "")
        by_ev[name] += 1
        if name == "llm_call":
            step = ev.get("step", "?")
            lat = ev.get("latency_s")
            if isinstance(lat, (int, float)):
                llm_latency_by_step[step].append(float(lat))
            if ev.get("structured"):
                structured_calls[ev["structured"]] += 1
        elif name == "action_result":
            act = ev.get("action", "?")
            status = ev.get("status", "")
            if status == "Success":
                action_by_type[act]["ok"] += 1
            else:
                action_by_type[act]["fail"] += 1
            lat = ev.get("latency_s")
            if isinstance(lat, (int, float)):
                action_latencies.append(float(lat))
        elif name == "goal_completed":
            if isinstance(ev.get("intent_match"), bool):
                intent_matches.append(ev["intent_match"])
            if isinstance(ev.get("goal_belief_met"), bool):
                belief_mets.append(ev["goal_belief_met"])
        elif name == "plan_failed":
            plan_failed_reasons.append(str(ev.get("error", ""))[:120])

    llm_calls_trace = by_ev.get("llm_call", 0)

    return {
        "session_id": metrics.get("session_id") or session_dir.name,
        "duration_s": metrics.get("duration_s"),
        "npc_count": npc_count,
        "by_npc": by_npc,
        "goals_started": goals_started,
        "goals_completed": goals_completed,
        "goals_belief_met": goals_belief_met,
        "intent_match_n": sum(1 for m in intent_matches if m),
        "intent_match_total": len(intent_matches),
        "plans_from_memory": plans_from_memory,
        "plans_from_llm": plans_from_llm,
        "llm_calls": llm_calls_m or llm_calls_trace,
        "llm_total_s": llm_total_s,
        "planning_wait_s": planning_wait_s,
        "actions_sent": actions_sent,
        "actions_ok": actions_ok,
        "actions_failed": actions_failed,
        "peer_requests_sent": peer_requests_sent,
        "peer_requests_accepted": peer_requests_accepted,
        "peer_requests_refused": peer_requests_refused,
        "peer_requests_completed": peer_requests_completed,
        "peer_requests_failed": peer_requests_failed,
        "items_given": items_given,
        "items_received": items_received,
        "peer_wait_s": peer_wait_s,
        "goal_switches": goal_switches,
        "arbitrations_llm": arbitrations_llm,
        "arbitrations_rule": arbitrations_rule,
        "arbitrations_fallback": arbitrations_fallback,
        "arbitration_llm_s": arbitration_llm_s,
        "fuse_activations": by_ev.get("no_applicable_variant_fuse", 0),
        "plan_failed": by_ev.get("plan_failed", 0),
        "plan_failed_reasons": plan_failed_reasons,
        "replan_required": by_ev.get("goal_replan_required", 0),
        "plan_memory_reuse": by_ev.get("plan_memory_reuse", 0),
        "llm_latency_by_step": {s: round(statistics.mean(v), 2) for s, v in llm_latency_by_step.items()},
        "llm_count_by_step": {s: len(v) for s, v in llm_latency_by_step.items()},
        "structured_calls": dict(structured_calls),
        "action_by_type": {a: dict(c) for a, c in action_by_type.items()},
        "action_avg_latency_s": round(statistics.mean(action_latencies), 2) if action_latencies else None,
    }


# ---------------------------------------------------------------------------
# Agregado
# ---------------------------------------------------------------------------

def _pct(n: float, total: float) -> str:
    return f"{(100 * n / total):.0f}%" if total else "—"


def aggregate(summaries: list[dict]) -> dict[str, Any]:
    agg: dict[str, float] = defaultdict(float)
    keys = [
        "goals_started", "goals_completed", "goals_belief_met", "intent_match_n",
        "intent_match_total", "plans_from_memory", "plans_from_llm", "llm_calls",
        "llm_total_s", "planning_wait_s", "actions_sent", "actions_ok", "actions_failed",
        "fuse_activations", "plan_failed", "replan_required", "plan_memory_reuse",
        "peer_requests_sent", "peer_requests_accepted", "peer_requests_refused",
        "peer_requests_completed", "peer_requests_failed", "items_given",
        "items_received", "peer_wait_s",
    ]
    for s in summaries:
        for k in keys:
            agg[k] += float(s.get(k, 0) or 0)
    agg["sessions"] = len(summaries)
    return dict(agg)


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------

def to_markdown(summaries: list[dict], agg: dict) -> str:
    lines: list[str] = []
    lines.append("# Análisis de sesiones\n")
    lines.append(f"Sesiones analizadas: **{agg.get('sessions', 0)}**\n")

    # --- Planning ---
    lines.append("## Planning\n")
    lines.append("| Sesión | NPCs | Goals | Completados | belief_met | intent_match | Plan LLM/Mem | Replans |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['session_id']} | {int(s.get('npc_count', 0))} | {int(s['goals_started'])} | "
            f"{int(s['goals_completed'])} ({_pct(s['goals_completed'], s['goals_started'])}) | "
            f"{int(s['goals_belief_met'])} | "
            f"{s['intent_match_n']}/{s['intent_match_total']} | "
            f"{int(s['plans_from_llm'])}/{int(s['plans_from_memory'])} | "
            f"{int(s['replan_required'])} |"
        )
    lines.append(
        f"| **TOTAL** | — | {int(agg['goals_started'])} | "
        f"{int(agg['goals_completed'])} ({_pct(agg['goals_completed'], agg['goals_started'])}) | "
        f"{int(agg['goals_belief_met'])} | "
        f"{int(agg['intent_match_n'])}/{int(agg['intent_match_total'])} | "
        f"{int(agg['plans_from_llm'])}/{int(agg['plans_from_memory'])} | "
        f"{int(agg['replan_required'])} |\n"
    )

    # --- Pipeline ---
    lines.append("## Pipeline (LLM)\n")
    lines.append(
        f"- Llamadas LLM totales: **{int(agg['llm_calls'])}** · "
        f"tiempo LLM total: **{agg['llm_total_s']:.0f}s** · "
        f"media/llamada: **{(agg['llm_total_s'] / agg['llm_calls']):.1f}s**"
        if agg["llm_calls"] else "- Sin llamadas LLM."
    )
    lines.append(f"- pipelines fallidos: **{int(agg['plan_failed'])}** · "
                 f"fusibles no-applicable-variant: **{int(agg['fuse_activations'])}**\n")

    # Latencia media por paso (agregando los por-sesión)
    step_lat: dict[str, list[float]] = defaultdict(list)
    step_cnt: dict[str, int] = defaultdict(int)
    structured_total: dict[str, int] = defaultdict(int)
    for s in summaries:
        for st, lat in s["llm_latency_by_step"].items():
            step_lat[st].append(lat)
        for st, c in s["llm_count_by_step"].items():
            step_cnt[st] += c
        for sc, c in s["structured_calls"].items():
            structured_total[sc] += c
    if step_lat:
        lines.append("| Paso | Llamadas | Latencia media (s) |")
        lines.append("|---|---|---|")
        for st in sorted(step_lat):
            lines.append(f"| {st} | {step_cnt[st]} | {statistics.mean(step_lat[st]):.1f} |")
        lines.append("")
    if structured_total:
        lines.append("Structured outputs (Fase 3): " +
                      ", ".join(f"{k}×{v}" for k, v in sorted(structured_total.items())) + "\n")

    # --- Por NPC (Fase 11 — solo se muestra si hay sesiones multi-agente) ---
    multi_agent = any(s.get("npc_count", 0) > 1 for s in summaries)
    if multi_agent:
        lines.append("## Por NPC (multi-agente)\n")
        lines.append(
            "| Sesión | NPC | Goals | Completados | belief_met | LLM calls | "
            "LLM total (s) | espera cola (s) | Acciones OK/Fail |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for s in summaries:
            for npc_id, n in sorted(s.get("by_npc", {}).items()):
                lines.append(
                    f"| {n['session_id']} | {npc_id} | {int(n['goals_started'])} | "
                    f"{int(n['goals_completed'])} | {int(n['goals_belief_met'])} | "
                    f"{int(n['llm_calls'])} | {float(n['llm_total_s']):.1f} | "
                    f"{float(n['planning_wait_s']):.1f} | "
                    f"{int(n['actions_ok'])}/{int(n['actions_failed'])} |"
                )
        lines.append("")
        if agg.get("llm_calls"):
            lines.append(
                f"- Espera de cola total (todas las sesiones): "
                f"**{agg.get('planning_wait_s', 0.0):.1f}s** de "
                f"{agg['llm_total_s']:.1f}s de trabajo LLM.\n"
            )

    # --- Coordinación (Fase 12 — solo se muestra si hay actividad) ---
    coord_active = agg.get("peer_requests_sent", 0) or agg.get("peer_requests_accepted", 0)
    if coord_active:
        lines.append("## Coordinación NPC↔NPC\n")
        lines.append(
            "| Sesión | NPC | Pedidas | Aceptadas | Rechazadas | Completadas | "
            "Fallidas | Items dados/recibidos | espera peer (s) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for s in summaries:
            for npc_id, n in sorted(s.get("by_npc", {}).items()):
                if not (n.get("peer_requests_sent") or n.get("peer_requests_accepted")):
                    continue
                lines.append(
                    f"| {n['session_id']} | {npc_id} | "
                    f"{int(n['peer_requests_sent'])} | {int(n['peer_requests_accepted'])} | "
                    f"{int(n['peer_requests_refused'])} | {int(n['peer_requests_completed'])} | "
                    f"{int(n['peer_requests_failed'])} | "
                    f"{int(n['items_given'])}/{int(n['items_received'])} | "
                    f"{float(n['peer_wait_s']):.1f} |"
                )
        lines.append(
            f"\n- Total: **{int(agg['peer_requests_sent'])}** peticiones enviadas, "
            f"**{int(agg['peer_requests_accepted'])}** aceptadas, "
            f"**{int(agg['peer_requests_refused'])}** rechazadas, "
            f"**{int(agg['peer_requests_completed'])}** completadas, "
            f"**{int(agg['peer_requests_failed'])}** fallidas.\n"
        )

    # --- Ejecución ---
    lines.append("## Ejecución (acciones)\n")
    act_agg: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "fail": 0})
    for s in summaries:
        for a, c in s["action_by_type"].items():
            act_agg[a]["ok"] += c.get("ok", 0)
            act_agg[a]["fail"] += c.get("fail", 0)
    lines.append("| Acción | OK | Fallos |")
    lines.append("|---|---|---|")
    for a in sorted(act_agg):
        lines.append(f"| {a} | {act_agg[a]['ok']} | {act_agg[a]['fail']} |")
    lines.append("")

    return "\n".join(lines)


def write_csv(summaries: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "session_id", "duration_s", "npc_count", "goals_started", "goals_completed",
        "goals_belief_met", "intent_match_n", "intent_match_total",
        "plans_from_llm", "plans_from_memory", "llm_calls", "llm_total_s",
        "planning_wait_s", "actions_sent", "actions_ok", "actions_failed",
        "fuse_activations", "plan_failed", "replan_required", "plan_memory_reuse",
        "peer_requests_sent", "peer_requests_accepted", "peer_requests_refused",
        "peer_requests_completed", "peer_requests_failed", "items_given",
        "items_received", "peer_wait_s",
    ]
    with (out_dir / "sessions.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)


def analyze(root: Path) -> tuple[list[dict], dict]:
    summaries = [summarize_session(d) for d in find_sessions(root)]
    return summaries, aggregate(summaries)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # markdown con acentos limpio
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Análisis de sesiones (Fase 5).")
    parser.add_argument("sessions", type=Path, help="logs/sessions o un dir de sesión")
    parser.add_argument("--csv", type=Path, default=None, help="dir de salida para CSV")
    parser.add_argument("--md", action="store_true", help="emitir markdown (default)")
    args = parser.parse_args(argv)

    if not args.sessions.exists():
        print(f"No existe: {args.sessions}", file=sys.stderr)
        return 2

    summaries, agg = analyze(args.sessions)
    if not summaries:
        print("No se encontraron sesiones con trace.jsonl.", file=sys.stderr)
        return 1

    if args.csv:
        write_csv(summaries, args.csv)
        print(f"CSV escrito en {args.csv / 'sessions.csv'}", file=sys.stderr)

    print(to_markdown(summaries, agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
