from __future__ import annotations

"""analyze_experiments.py — Análisis de la batería experimental (Fase 13).

Agrega sesiones etiquetadas (session_start.experiment_id/config_label, ver
Fase 13 T2) por experiment_id × config_label: eficiencia de ejecución
(η_exec = n_opt/n_act), acciones redundantes clasificadas, coste de
planificación y coste temporal. Reusa `summarize_session()` de
`analyze_sessions.py` — NO duplica la lectura de trazas, solo añade lo
específico de la batería encima.

Las sesiones SIN `experiment_id` (uso normal, fuera de la batería) se
ignoran silenciosamente al agregar — no rompen nada.

Uso:
    python tools/analyze_experiments.py logs/sessions --out DOC/Evaluacion
    python tools/analyze_experiments.py logs/sessions --csv DOC/Evaluacion
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import analyze_sessions as _base  # noqa: E402


# ---------------------------------------------------------------------------
# Estadística ligera (mediana + IQR, sin numpy)
# ---------------------------------------------------------------------------

def _median_iqr(values: list[float]) -> tuple[float | None, float | None, float | None]:
    """(mediana, Q1, Q3). None si no hay datos suficientes."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None, None, None
    med = statistics.median(vals)
    if len(vals) >= 4:
        half = len(vals) // 2
        q1 = statistics.median(vals[:half])
        q3 = statistics.median(vals[-half:])
    else:
        q1 = q3 = None
    return round(med, 3), (round(q1, 3) if q1 is not None else None), (round(q3, 3) if q3 is not None else None)


# ---------------------------------------------------------------------------
# Acciones redundantes ("dar vueltas") — heurísticas documentadas, no exactas
# ---------------------------------------------------------------------------

def count_redundant_actions(events: list[dict]) -> dict[str, dict[str, int]]:
    """Clasifica acciones redundantes por NPC a partir de la traza cronológica:

      - redundant_moveto: MoveTo(x,y) con (x,y) == current_position ya conocida.
      - redundant_search: Search(item) repetido para el mismo item sin haber
        cambiado de posición/zona desde el Search anterior de ese item.
      - redundant_explorearea: ExploreArea(zone) con esa zona ya en
        zone_center (no debería hacer falta, ver hallazgo Fase 11: zone_center
        es global desde el NPCProfile).

    Heurísticas basadas en `belief_updated`/`action_sent` de trace.jsonl — no
    pretenden ser exactas al 100%, son una señal reproducible y documentada.
    """
    per_npc: dict[str, dict[str, int]] = defaultdict(
        lambda: {"redundant_moveto": 0, "redundant_search": 0, "redundant_explorearea": 0}
    )
    position: dict[str, tuple] = {}
    current_zone: dict[str, str] = {}
    known_zones: dict[str, set] = defaultdict(set)
    searched_since_move: dict[str, set] = defaultdict(set)  # npc -> {item_id}

    for ev in events:
        npc = ev.get("npc")
        if not npc:
            continue
        name = ev.get("ev")

        if name == "belief_updated":
            pred = ev.get("predicate")
            args = ev.get("args") or []
            if pred == "current_position" and len(args) == 2:
                new_pos = (args[0], args[1])
                if position.get(npc) != new_pos:
                    searched_since_move[npc] = set()
                position[npc] = new_pos
            elif pred == "zone_center" and len(args) == 3:
                known_zones[npc].add(args[0])
            elif pred == "at_zone" and len(args) == 1:
                if current_zone.get(npc) != args[0]:
                    searched_since_move[npc] = set()
                current_zone[npc] = args[0]

        elif name == "action_sent":
            action = ev.get("action")
            args = ev.get("args") or {}
            if action == "MoveTo":
                target = (args.get("x"), args.get("y"))
                if npc in position and position[npc] == target:
                    per_npc[npc]["redundant_moveto"] += 1
            elif action == "Search":
                item_id = args.get("itemId")
                if item_id is not None:
                    if item_id in searched_since_move[npc]:
                        per_npc[npc]["redundant_search"] += 1
                    else:
                        searched_since_move[npc].add(item_id)
            elif action == "ExploreArea":
                zone_tag = args.get("zoneTag")
                if zone_tag and zone_tag in known_zones[npc]:
                    per_npc[npc]["redundant_explorearea"] += 1

    return dict(per_npc)


# ---------------------------------------------------------------------------
# Resumen por sesión (extiende summarize_session con lo propio de la batería)
# ---------------------------------------------------------------------------

def summarize_experiment_session(session_dir: Path) -> dict[str, Any] | None:
    """None si la sesión no tiene `experiment_id` en `session_start` (no es
    parte de la batería — se ignora silenciosamente al agregar, no es un error)."""
    events = _base._read_jsonl(session_dir / "trace.jsonl")
    session_start = next((e for e in events if e.get("ev") == "session_start"), {})
    experiment_id = session_start.get("experiment_id")
    if not experiment_id:
        return None

    base = _base.summarize_session(session_dir)
    n_opt = session_start.get("n_opt")
    n_act = base.get("actions_sent", 0) or 0
    eta_exec = (n_opt / n_act) if (n_opt and n_act) else None
    detour = (n_act - n_opt) if (n_opt is not None) else None

    goals_unverified = sum(1 for e in events if e.get("ev") == "goal_completed_unverified")

    redundant = count_redundant_actions(events)
    redundant_total = sum(sum(d.values()) for d in redundant.values())

    # goals_started (metrics.json) cuenta PETICIONES DE PLAN, no goals
    # distintos -- se incrementa una vez por replan, no solo al abrir el goal
    # (bdi.py:_request_plan). Para la tasa de éxito de la batería, el
    # denominador correcto es el nº de goals DISTINTOS realmente cerrados:
    # len(goals_detail) por NPC. Se reporta `plan_requests` aparte para
    # visibilizar el coste de replanificación sin contaminar el % de éxito.
    metrics = _base._read_json(session_dir / "metrics.json")
    npcs_m = (metrics.get("npcs") or {}).values()
    goals_distinct = sum(len(n.get("goals_detail", [])) for n in npcs_m)
    goals_distinct_belief_met = sum(
        1 for n in npcs_m for g in n.get("goals_detail", []) if g.get("belief_met") is True
    )
    plan_requests = base.get("goals_started", 0)

    return {
        **base,
        "experiment_id": experiment_id,
        "config_label": session_start.get("config_label"),
        "git_sha": session_start.get("git_sha"),
        "n_opt": n_opt,
        "n_act": n_act,
        "eta_exec": round(eta_exec, 3) if eta_exec is not None else None,
        "detour": detour,
        "goals_unverified": goals_unverified,
        "redundant_by_npc": redundant,
        "redundant_total": redundant_total,
        "plan_requests": plan_requests,
        "goals_distinct": goals_distinct,
        "goals_distinct_belief_met": goals_distinct_belief_met,
        "goal_switches": base.get("goal_switches", 0),
        "arbitrations_llm": base.get("arbitrations_llm", 0),
        "arbitrations_rule": base.get("arbitrations_rule", 0),
        "arbitrations_fallback": base.get("arbitrations_fallback", 0),
        "arbitration_llm_s": base.get("arbitration_llm_s", 0.0),
    }


def collect(root: Path) -> list[dict]:
    out: list[dict] = []
    for session_dir in _base.find_sessions(root):
        s = summarize_experiment_session(session_dir)
        if s is not None:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Agregado por (experiment_id, config_label)
# ---------------------------------------------------------------------------

def aggregate_by_experiment_config(summaries: list[dict]) -> dict[str, dict]:
    """Clave: '<experiment_id>/<config_label>'."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        key = f"{s['experiment_id']}/{s.get('config_label') or '?'}"
        groups[key].append(s)

    result: dict[str, dict] = {}
    for key, items in sorted(groups.items()):
        eta_med, eta_q1, eta_q3 = _median_iqr([s["eta_exec"] for s in items])
        llm_med, _, _ = _median_iqr([s.get("llm_calls", 0) for s in items])
        dur_med, dur_q1, dur_q3 = _median_iqr([s.get("duration_s") for s in items])
        red_med, _, _ = _median_iqr([s.get("redundant_total", 0) for s in items])
        n_act_med, _, _ = _median_iqr([s.get("n_act") for s in items])

        # Éxito: sobre goals DISTINTOS (goals_distinct), no sobre peticiones de
        # plan (goals_started/plan_requests incluye un +1 por cada replan --
        # ver summarize_experiment_session). plan_requests se reporta aparte
        # como coste de replanificación, no como denominador de éxito.
        goals_distinct = sum(s.get("goals_distinct", 0) for s in items)
        goals_distinct_belief_met = sum(s.get("goals_distinct_belief_met", 0) for s in items)
        plan_requests = sum(s.get("plan_requests", 0) for s in items)
        goals_unverified = sum(s.get("goals_unverified", 0) for s in items)
        plans_from_memory = sum(s.get("plans_from_memory", 0) for s in items)
        plans_from_llm = sum(s.get("plans_from_llm", 0) for s in items)
        goal_switches = sum(s.get("goal_switches", 0) for s in items)
        arbitrations_llm = sum(s.get("arbitrations_llm", 0) for s in items)
        arbitrations_rule = sum(s.get("arbitrations_rule", 0) for s in items)
        arbitrations_fallback = sum(s.get("arbitrations_fallback", 0) for s in items)

        result[key] = {
            "experiment_id": items[0]["experiment_id"],
            "config_label": items[0].get("config_label") or "?",
            "n": len(items),
            "n_opt": items[0].get("n_opt"),
            "eta_exec_median": eta_med, "eta_exec_q1": eta_q1, "eta_exec_q3": eta_q3,
            "n_act_median": n_act_med,
            "goal_success_rate": round(goals_distinct_belief_met / goals_distinct, 3) if goals_distinct else None,
            "goals_started": goals_distinct,
            "goals_belief_met": goals_distinct_belief_met,
            "plan_requests": plan_requests,
            "goals_unverified": goals_unverified,
            "plans_from_memory": plans_from_memory,
            "plans_from_llm": plans_from_llm,
            "llm_calls_median": llm_med,
            "duration_s_median": dur_med, "duration_s_q1": dur_q1, "duration_s_q3": dur_q3,
            "redundant_actions_median": red_med,
            "goal_switches": goal_switches,
            "arbitrations_llm": arbitrations_llm,
            "arbitrations_rule": arbitrations_rule,
            "arbitrations_fallback": arbitrations_fallback,
        }
    return result


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------

def to_markdown(summaries: list[dict], agg: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("# Batería experimental — resultados agregados\n")
    lines.append(f"Sesiones de batería analizadas: **{len(summaries)}**\n")

    if not agg:
        lines.append("_Sin sesiones etiquetadas (`experiment_id`) — nada que agregar._\n")
        return "\n".join(lines)

    lines.append("## Tabla maestra (mediana + IQR)\n")
    lines.append(
        "| Exp/Config | N | éxito (belief, sobre goals distintos) | unverified | η_exec (med.) | "
        "n_act (med.) | acciones redundantes (med.) | llm_calls (med.) | peticiones de plan | "
        "switches (LLM/regla/fallback) | t_sesión (med. s) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, a in agg.items():
        eta_str = f"{a['eta_exec_median']}" if a["eta_exec_median"] is not None else "—"
        if a["eta_exec_q1"] is not None and a["eta_exec_q3"] is not None:
            eta_str += f" [{a['eta_exec_q1']}, {a['eta_exec_q3']}]"
        success = _base._pct(a["goals_belief_met"], a["goals_started"])
        switches = a.get("goal_switches", 0)
        switches_str = (
            f"{switches} ({a.get('arbitrations_llm',0)}/{a.get('arbitrations_rule',0)}/{a.get('arbitrations_fallback',0)})"
            if switches else "0"
        )
        lines.append(
            f"| {key} | {a['n']} | {a['goals_belief_met']}/{a['goals_started']} ({success}) | "
            f"{a['goals_unverified']} | {eta_str} | {a['n_act_median']} | "
            f"{a['redundant_actions_median']} | {a['llm_calls_median']} | {a['plan_requests']} | "
            f"{switches_str} | {a['duration_s_median']} |"
        )
    lines.append("")
    lines.append(
        "_'éxito' cuenta **goals distintos** cerrados (no peticiones de plan: "
        "una replanificación no es un goal nuevo). 'peticiones de plan' agrega "
        "cuántas veces se llamó a `_request_plan` en total (incluye replans) — "
        "es el coste real de planificación, no el nº de goals. 'switches' "
        "(Fase 14) es el total de cambios de intención por arbitraje, con el "
        "desglose LLM/regla/fallback-a-regla entre paréntesis._\n"
    )

    lines.append("## Nota de honestidad\n")
    lines.append(
        "- `n_opt` proviene de `session_start` (lo fija `run_experiment.ps1` desde "
        "el manifiesto) — si falta, `η_exec` sale vacío para esa sesión.\n"
        "- Configuraciones con `N < 5` son **indicativas, no concluyentes** "
        "(ver PROTOCOLO_EXPERIMENTOS.md §5).\n"
        "- `C1`/`X1` son deterministas (oráculo) — repórtense aparte, nunca "
        "mezclados con las medias de M0/M1/M2.\n"
    )
    return "\n".join(lines)


def write_csv(summaries: list[dict], agg: dict[str, dict], out_dir: Path) -> None:
    import csv as _csv
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "experiment_id", "config_label", "session_id", "git_sha", "n_opt", "n_act",
        "eta_exec", "detour", "goals_distinct", "goals_distinct_belief_met", "plan_requests",
        "goals_unverified", "llm_calls", "llm_total_s", "duration_s", "redundant_total",
        "goal_switches", "arbitrations_llm", "arbitrations_rule", "arbitrations_fallback",
        "arbitration_llm_s",
    ]
    with (out_dir / "experiments_sessions.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    agg_cols = [
        "experiment_id", "config_label", "n", "n_opt", "eta_exec_median", "eta_exec_q1",
        "eta_exec_q3", "n_act_median", "goal_success_rate", "goals_started",
        "goals_belief_met", "plan_requests", "goals_unverified", "llm_calls_median",
        "duration_s_median", "redundant_actions_median",
        "goal_switches", "arbitrations_llm", "arbitrations_rule", "arbitrations_fallback",
    ]
    with (out_dir / "experiments_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=agg_cols, extrasaction="ignore")
        w.writeheader()
        for a in agg.values():
            w.writerow(a)


def analyze(root: Path) -> tuple[list[dict], dict[str, dict]]:
    summaries = collect(root)
    return summaries, aggregate_by_experiment_config(summaries)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Análisis de la batería experimental (Fase 13).")
    parser.add_argument("sessions", type=Path, help="logs/sessions o un dir de sesión")
    parser.add_argument("--out", type=Path, default=None, help="dir de salida para el markdown")
    parser.add_argument("--csv", type=Path, default=None, help="dir de salida para CSV")
    args = parser.parse_args(argv)

    if not args.sessions.exists():
        print(f"No existe: {args.sessions}", file=sys.stderr)
        return 2

    summaries, agg = analyze(args.sessions)
    md = to_markdown(summaries, agg)

    if args.csv:
        write_csv(summaries, agg, args.csv)
        print(f"CSV escrito en {args.csv}", file=sys.stderr)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        out_path = args.out / "BATERIA_RESULTADOS.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"Markdown escrito en {out_path}", file=sys.stderr)
    else:
        print(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
