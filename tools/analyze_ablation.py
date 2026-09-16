from __future__ import annotations

"""analyze_ablation.py — Fase 16: ablación de sub-planes escritos a mano.

Compara, sobre los MISMOS escenarios (manifiestos de la suite, A1..A4), la
config SUB (con move_to_and_pickup/craft_item) contra ATOM (sin ellos: el LLM
compone el plan con acciones primitivas). Todo lo demás es idéntico.

Diferencias con analyze_experiments.py, a propósito:
  - El éxito se mide contra los goals DEL MANIFIESTO leyendo `goal_completed`
    (goal_belief_met=true) en la traza, no contra metrics.json: una sesión que
    no cierra ningún goal (timeout, kill) cuenta como fracaso en vez de
    desaparecer del denominador.
  - Clasifica la causa de cada fracaso (causa principal + señales).
  - Comprobación de manipulación: en ATOM los sub-planes no deben haberse
    cargado (evento `builtin_plans_loaded`) y `session_start.builtin_subplans`
    debe valer false.
  - Estadística en Python puro (sin scipy): IC de Wilson, Fisher exacto para el
    éxito y Mann-Whitney U (aprox. normal con corrección de empates y de
    continuidad) para las variables continuas. Con N=8 por brazo la potencia es
    baja: los p-valores se reportan, no se venden como concluyentes.

Uso:
    python tools/analyze_ablation.py logs/sessions --suite ablation_builtins --out out/ablation_builtins
    python tools/analyze_ablation.py logs/sessions --since-epoch 1789000000
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _ROOT / "tools" / "experiments"

# Espejo de llm.pipeline.builtins.ABLATABLE_SUBPLANS (tools/ no importa src/).
ABLATED_SUBPLANS: tuple[str, ...] = ("move_to_and_pickup", "craft_item")

_COST_METRICS: tuple[str, ...] = ("llm_calls", "n_act", "duration_s")


# ---------------------------------------------------------------------------
# Estadística
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    """Intervalo de Wilson al 95% para una proporción k/n."""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """p-valor bilateral del test exacto de Fisher para [[a, b], [c, d]]
    (filas = brazos, columnas = éxito / fracaso)."""
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    if n == 0:
        return 1.0
    total = comb(n, col1)

    def prob(x: int) -> float:
        return comb(row1, x) * comb(row2, col1 - x) / total

    p_obs = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    p = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-7))
    return min(1.0, p)


def mann_whitney_u(x: Iterable[float | None], y: Iterable[float | None]) -> tuple[float, float] | None:
    """(U, p bilateral) con aproximación normal, corrección de empates y de
    continuidad. None si algún grupo está vacío."""
    xs = [float(v) for v in x if v is not None]
    ys = [float(v) for v in y if v is not None]
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return None
    combined = sorted([(v, 0) for v in xs] + [(v, 1) for v in ys], key=lambda t: t[0])
    ranks = [0.0] * len(combined)
    tie_term = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for idx in range(i, j + 1):
            ranks[idx] = avg_rank
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    r1 = sum(r for r, (_v, group) in zip(ranks, combined) if group == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return u, 1.0
    z = max(0.0, (abs(u1 - n1 * n2 / 2) - 0.5) / math.sqrt(var))
    return u, min(1.0, math.erfc(z / math.sqrt(2)))


def median_iqr(values: Iterable[float | None]) -> tuple[float | None, float | None, float | None]:
    """(mediana, Q1, Q3); cuartiles solo con N >= 4 (mismo criterio que analyze_experiments)."""
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None, None, None
    med = round(statistics.median(vals), 3)
    if len(vals) < 4:
        return med, None, None
    half = len(vals) // 2
    return med, round(statistics.median(vals[:half]), 3), round(statistics.median(vals[-half:]), 3)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_suite(suite: str | Path) -> dict:
    path = Path(suite)
    if not path.exists():
        path = _EXPERIMENTS_DIR / "suites" / f"{suite}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifests(experiment_ids: Iterable[str]) -> dict[str, dict]:
    return {
        exp_id: json.loads((_EXPERIMENTS_DIR / f"{exp_id}.json").read_text(encoding="utf-8"))
        for exp_id in experiment_ids
    }


def _read_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def find_session_dirs(root: Path) -> list[Path]:
    if (root / "trace.jsonl").exists():
        return [root]
    return sorted({p.parent for p in root.rglob("trace.jsonl")})


def _norm_condition(cond: Any) -> str:
    return re.sub(r"\s+", "", str(cond or "")).lower()


# ---------------------------------------------------------------------------
# Resumen por sesión
# ---------------------------------------------------------------------------

def summarize_session(
    session_dir: Path,
    manifests: dict[str, dict],
    configs: Iterable[str],
    *,
    expected_flags: dict[str, bool] | None = None,
    since_epoch: float | None = None,
    ablated_subplans: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """None si la sesión no pertenece a la suite (experimento/config ajenos o
    anterior a `since_epoch`)."""
    events = _read_jsonl(session_dir / "trace.jsonl")
    start = next((e for e in events if e.get("ev") == "session_start"), None)
    if start is None:
        return None
    exp_id, cfg = start.get("experiment_id"), start.get("config_label")
    if exp_id not in manifests or cfg not in set(configs):
        return None
    t0 = float(start.get("t", 0.0) or 0.0)
    if since_epoch is not None and t0 < since_epoch:
        return None

    expected = [
        (str(npc), _norm_condition(goal.get("condition")))
        for npc, goals in (manifests[exp_id].get("goals") or {}).items()
        for goal in goals
    ]

    counts: Counter = Counter()
    met_at: dict[tuple[str, str], float] = {}
    llm_s = 0.0
    actions_failed = 0
    failed_reasons: list[str] = []
    goal_failed_reasons: list[str] = []
    closed_without_belief = 0
    transforms: Counter = Counter()
    registered: set[str] = set()
    loaded_sigs: set[str] = set()
    saw_loaded_event = False
    last_t = t0
    max_depth = 0
    n_act_by_npc: Counter = Counter()

    for e in events:
        ev = str(e.get("ev", ""))
        counts[ev] += 1
        if isinstance(e.get("t"), (int, float)):
            last_t = max(last_t, float(e["t"]))
        if ev == "goal_completed":
            if e.get("goal_belief_met") is True:
                cond = e.get("expected_condition") or e.get("derived_condition")
                met_at.setdefault((str(e.get("npc")), _norm_condition(cond)), float(e.get("t", t0)))
            else:
                closed_without_belief += 1
        elif ev == "unity_in" and e.get("msg_type") == "RegisterNPC":
            registered.add(str(e.get("npc")))
        elif ev == "llm_call":
            if isinstance(e.get("latency_s"), (int, float)):
                llm_s += float(e["latency_s"])
        elif ev == "action_result" and e.get("status") != "Success":
            actions_failed += 1
        elif ev == "goal_failed_after_replans":
            failed_reasons.append(str(e.get("reason") or "?"))
        elif ev == "goal_failed":
            goal_failed_reasons.append(str(e.get("reason") or "?"))
        elif ev == "plan_transform":
            transforms[str(e.get("source") or "?")] += 1
        elif ev == "builtin_plans_loaded":
            saw_loaded_event = True
            loaded_sigs.update(str(s) for s in (e.get("sigs") or []))
        elif ev == "peer_request_sent":
            if isinstance(e.get("depth"), (int, float)):
                max_depth = max(max_depth, int(e["depth"]))
        if ev == "action_sent" and e.get("npc"):
            n_act_by_npc[str(e["npc"])] += 1

    goals_met = sum(1 for key in expected if key in met_at)
    success = bool(expected) and goals_met == len(expected)
    t_success = round(max(met_at[key] for key in expected) - t0, 3) if success else None

    if counts["shutdown_idle"]:
        end_reason = "idle"
    elif counts["shutdown_timeout"]:
        end_reason = "timeout"
    elif not counts["session_end"]:
        end_reason = "killed"
    else:
        end_reason = "other"
    session_end = next((e for e in reversed(events) if e.get("ev") == "session_end"), None)
    duration = session_end.get("duration_s") if session_end else round(last_t - t0, 3)

    pipeline_errors = counts["pipeline_error"] + counts["plan_failed"]
    signals: list[str] = [f"failed_after_replans:{r}" for r in failed_reasons]
    signals += [f"failed:{r}" for r in goal_failed_reasons]
    if counts["no_applicable_variant_fuse"]:
        signals.append("no_applicable_variant")
    if closed_without_belief:
        signals.append("closed_without_belief")
    if pipeline_errors:
        signals.append("pipeline_error")
    if actions_failed:
        signals.append("action_failures")
    if end_reason in ("timeout", "killed"):
        signals.append("timeout")

    # Infraestructura: un NPC con goal en el manifiesto nunca se registró desde Unity
    # (build o spawn equivocados). La sesión no mide el brazo: se excluye del análisis.
    missing_npcs = sorted({npc for npc, _cond in expected} - registered)
    infra_error = bool(missing_npcs)
    if infra_error:
        signals.insert(0, "infra:npc_not_registered")

    primary: str | None = None
    if not success:
        if infra_error:
            primary = "infra:npc_not_registered"
        elif failed_reasons:
            primary = f"failed_after_replans:{failed_reasons[-1]}"
        elif goal_failed_reasons:
            primary = f"failed:{goal_failed_reasons[-1]}"
        elif counts["no_applicable_variant_fuse"]:
            primary = "no_applicable_variant"
        elif closed_without_belief:
            primary = "closed_without_belief"
        elif end_reason in ("timeout", "killed"):
            primary = "timeout"
        elif pipeline_errors:
            primary = "pipeline_error"
        else:
            primary = "unknown"

    # Comprobación de manipulación: flag de la sesión y sub-planes cargados.
    # La lista la fija el manifiesto si la declara (A5: también los macros de
    # coordinación, que en A1–A4 se cargan sin usarse); si no, la de la suite.
    manifest_ablated = manifests[exp_id].get("ablated_subplans")
    if manifest_ablated is not None:
        ablated = tuple(manifest_ablated)
    else:
        ablated = tuple(ablated_subplans) if ablated_subplans is not None else ABLATED_SUBPLANS
    ablated_loaded = sorted(loaded_sigs & set(ablated))
    flag = start.get("builtin_subplans")
    expected_flag = (expected_flags or {}).get(cfg)
    if not saw_loaded_event or flag is None or expected_flag is None:
        manipulation_ok = None
    else:
        manipulation_ok = (flag is expected_flag) and (bool(ablated_loaded) is expected_flag)

    # Llamadas a esos nombres en los planes guardados de la sesión. En SUB mide
    # el uso de los builtins; en ATOM son sub-goals inventados por el LLM.
    plan_calls = 0
    for asl in session_dir.glob("*.asl"):
        try:
            text = asl.read_text(encoding="utf-8")
        except OSError:
            continue
        plan_calls += sum(text.count(f"!{name}(") for name in ablated)

    return {
        "session_dir": str(session_dir),
        "infra_error": infra_error,
        "missing_npcs": missing_npcs,
        "experiment_id": exp_id,
        "config_label": cfg,
        "git_sha": start.get("git_sha"),
        "builtin_subplans": flag,
        "goals_expected": len(expected),
        "goals_met": goals_met,
        "success": success,
        "first_plan_success": success and counts["goal_replan_required"] == 0,
        "t_success": t_success,
        "duration_s": duration,
        "end_reason": end_reason,
        "llm_calls": counts["llm_call"],
        "llm_s": round(llm_s, 3),
        "plan_requests": counts["plan_requested"],
        "replans": counts["goal_replan_required"],
        "ladder_advance": counts["ladder_advance"],
        "n_act": counts["action_sent"],
        "actions_failed": actions_failed,
        "transforms_code": transforms.get("CODE", 0),
        "transforms_llm": transforms.get("LLM", 0),
        "plan_calls_to_ablated": plan_calls,
        "ablated_loaded": ablated_loaded,
        "manipulation_ok": manipulation_ok,
        "primary_cause": primary,
        "signals": signals,
        # Fase 17 — coordinación entre NPCs.
        "subplans_loaded": counts["subplan_loaded"],
        "peer_requests_sent": counts["peer_request_sent"],
        "peer_requests_accepted": counts["peer_request_accepted"],
        "peer_requests_refused": counts["peer_request_refused"],
        "peer_transfers": counts["peer_transfer"],
        "peer_wait_timeouts": counts["peer_wait_timeout"],
        "arbitrations": counts["arbitration_decision"],
        "max_peer_depth": max_depth,
        "n_act_by_npc": dict(n_act_by_npc),
    }


# ---------------------------------------------------------------------------
# Agregado y comparación
# ---------------------------------------------------------------------------

def aggregate(summaries: list[dict]) -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in summaries:
        groups[(s["experiment_id"], s["config_label"])].append(s)

    result: dict[tuple[str, str], dict] = {}
    for key, items in sorted(groups.items()):
        n = len(items)
        k = sum(1 for s in items if s["success"])
        lo, hi = wilson_ci(k, n)
        result[key] = {
            "experiment_id": key[0],
            "config_label": key[1],
            "n": n,
            "successes": k,
            "success_rate": round(k / n, 3) if n else None,
            "success_ci_low": round(lo, 3) if lo is not None else None,
            "success_ci_high": round(hi, 3) if hi is not None else None,
            "first_plan_successes": sum(1 for s in items if s["first_plan_success"]),
            "goals_met": sum(s["goals_met"] for s in items),
            "goals_expected": sum(s["goals_expected"] for s in items),
            "llm_calls": median_iqr(s["llm_calls"] for s in items),
            "llm_s": median_iqr(s["llm_s"] for s in items),
            "n_act": median_iqr(s["n_act"] for s in items),
            "replans": median_iqr(s["replans"] for s in items),
            "duration_s": median_iqr(s["duration_s"] for s in items),
            "t_success": median_iqr(s["t_success"] for s in items if s["success"]),
            "causes": Counter(s["primary_cause"] for s in items if not s["success"]),
            "end_reasons": Counter(s["end_reason"] for s in items),
            "manipulation_bad": sum(1 for s in items if s["manipulation_ok"] is False),
            "manipulation_unknown": sum(1 for s in items if s["manipulation_ok"] is None),
            "plan_calls_to_ablated": sum(s["plan_calls_to_ablated"] for s in items),
            "peer_requests_sent": median_iqr(s["peer_requests_sent"] for s in items),
            "peer_transfers": median_iqr(s["peer_transfers"] for s in items),
            "sessions_with_transfer": sum(1 for s in items if s["peer_transfers"] > 0),
            "peer_requests_refused": sum(s["peer_requests_refused"] for s in items),
            "peer_wait_timeouts": sum(s["peer_wait_timeouts"] for s in items),
            "arbitrations": sum(s["arbitrations"] for s in items),
            "max_peer_depth": max((s["max_peer_depth"] for s in items), default=0),
            "shas": sorted({str(s["git_sha"]) for s in items if s.get("git_sha")}),
        }
    return result


def compare_arms(summaries: list[dict], experiments: Iterable[str], configs: Iterable[str]) -> list[dict]:
    """Comparaciones por pares de configs (en el orden de la suite), por
    experimento y agregadas. Con dos configs es una sola comparación por fila."""
    configs = list(configs)
    pairs = [(configs[i], other) for i in range(len(configs)) for other in configs[i + 1:]]
    rows: list[dict] = []
    for exp, (base, treatment) in [(e, p) for e in list(experiments) + ["(todos)"] for p in pairs]:
        selected = summaries if exp == "(todos)" else [s for s in summaries if s["experiment_id"] == exp]
        a = [s for s in selected if s["config_label"] == base]
        b = [s for s in selected if s["config_label"] == treatment]
        if not a or not b:
            continue
        ka = sum(1 for s in a if s["success"])
        kb = sum(1 for s in b if s["success"])
        row: dict[str, Any] = {
            "experiment_id": exp,
            "base": base,
            "treatment": treatment,
            "n_base": len(a),
            "n_treatment": len(b),
            "success_base": ka,
            "success_treatment": kb,
            "fisher_p": round(fisher_exact_two_sided(ka, len(a) - ka, kb, len(b) - kb), 4),
        }
        for metric in _COST_METRICS:
            mw = mann_whitney_u([s[metric] for s in a], [s[metric] for s in b])
            row[f"mw_p_{metric}"] = round(mw[1], 4) if mw else None
        rows.append(row)
    return rows


def analyze(
    root: Path,
    suite: dict,
    manifests: dict[str, dict],
    *,
    since_epoch: float | None = None,
) -> tuple[list[dict], dict[tuple[str, str], dict], list[dict]]:
    configs = tuple(suite.get("configs") or ())
    summaries: list[dict] = []
    for session_dir in find_session_dirs(root):
        s = summarize_session(
            session_dir, manifests, configs,
            expected_flags=suite.get("config_builtin_subplans"),
            since_epoch=since_epoch,
            ablated_subplans=suite.get("ablated_subplans"),
        )
        if s is not None:
            summaries.append(s)
    # Las sesiones con error de infraestructura se listan, pero no cuentan en éxito ni coste.
    valid = [s for s in summaries if not s.get("infra_error")]
    return summaries, aggregate(valid), compare_arms(valid, suite.get("experiments") or [], configs)


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------

def _fmt_miq(stat: tuple[float | None, float | None, float | None]) -> str:
    med, q1, q3 = stat
    if med is None:
        return "—"
    text = f"{med:g}"
    if q1 is not None and q3 is not None:
        text += f" [{q1:g}, {q3:g}]"
    return text


def _fmt_p(p: float | None) -> str:
    return "—" if p is None else f"{p:.3f}"


def to_markdown(
    summaries: list[dict],
    agg: dict[tuple[str, str], dict],
    comparisons: list[dict],
    suite: dict,
) -> str:
    configs = list(suite.get("configs") or [])
    lines: list[str] = [f"# Ablación de sub-planes — resultados (`{suite.get('id', '?')}`)", ""]
    shas = sorted({str(s["git_sha"]) for s in summaries if s.get("git_sha")})
    lines.append(f"Sesiones analizadas: **{len(summaries)}** · commit(s): {', '.join(shas) or '—'}")
    infra = [s for s in summaries if s.get("infra_error")]
    if infra:
        lines.append("")
        lines.append(
            f"> ⚠️ {len(infra)} sesión(es) EXCLUIDAS por error de infraestructura (el NPC con goal no se "
            "registró desde Unity): " + ", ".join(
                f"{s['experiment_id']}/{s['config_label']} ({Path(s['session_dir']).name}, "
                f"falta {', '.join(s['missing_npcs'])})"
                for s in infra
            )
        )
    if len(shas) > 1:
        lines.append("")
        lines.append("> ⚠️ Hay sesiones de más de un commit: los brazos no son estrictamente comparables.")
    lines.append("")
    if not summaries:
        lines.append("_Sin sesiones de la suite._")
        return "\n".join(lines) + "\n"

    lines += [
        "## 1. Comprobación de manipulación",
        "",
        "| Exp/Config | N | manipulación incorrecta | no verificable | llamadas a move_to_and_pickup/craft_item en planes |",
        "|---|---|---|---|---|",
    ]
    for a in agg.values():
        lines.append(
            f"| {a['experiment_id']}/{a['config_label']} | {a['n']} | {a['manipulation_bad']} | "
            f"{a['manipulation_unknown']} | {a['plan_calls_to_ablated']} |"
        )
    lines += [
        "",
        "_Correcto = en ATOM no se cargaron los sub-planes y la sesión arrancó con "
        "`builtin_subplans=false` (y al revés en SUB). En ATOM, una llamada a esos "
        "nombres es un sub-goal inventado por el LLM (el pipeline lo expande con el "
        "propio LLM), no el builtin._",
        "",
        "## 2. Éxito (todos los goals del manifiesto verificados en creencias)",
        "",
    ]
    header = "| Exp | " + " | ".join(f"{c}: éxito [IC95]" for c in configs)
    header += " | " + " | ".join(f"{c}: sin replan" for c in configs) + " |"
    lines += [header, "|" + "---|" * (2 * len(configs) + 1)]
    for exp in list(suite.get("experiments") or []):
        cells_success, cells_first = [], []
        for cfg in configs:
            a = agg.get((exp, cfg))
            if a is None:
                cells_success.append("—")
                cells_first.append("—")
                continue
            cells_success.append(
                f"{a['successes']}/{a['n']} [{a['success_ci_low']:.2f}, {a['success_ci_high']:.2f}]"
            )
            cells_first.append(f"{a['first_plan_successes']}/{a['n']}")
        lines.append(f"| {exp} | " + " | ".join(cells_success) + " | " + " | ".join(cells_first) + " |")
    lines += [
        "",
        "**Comparaciones por pares**: Fisher exacto (éxito) y Mann-Whitney U (p bilateral, todas las sesiones de cada brazo).",
        "",
        "| Exp | par | éxito | Fisher p | MW llm_calls | MW n_act | MW duración |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c['experiment_id']} | {c['base']} vs {c['treatment']} | "
            f"{c['success_base']}/{c['n_base']} vs {c['success_treatment']}/{c['n_treatment']} | "
            f"{_fmt_p(c['fisher_p'])} | {_fmt_p(c.get('mw_p_llm_calls'))} | "
            f"{_fmt_p(c.get('mw_p_n_act'))} | {_fmt_p(c.get('mw_p_duration_s'))} |"
        )

    lines += [
        "",
        "## 3. Coste (mediana [Q1, Q3])",
        "",
        "| Exp/Config | N | llm_calls | t LLM (s) | n_act | replans | t hasta éxito (s) | duración sesión (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in agg.values():
        lines.append(
            f"| {a['experiment_id']}/{a['config_label']} | {a['n']} | {_fmt_miq(a['llm_calls'])} | "
            f"{_fmt_miq(a['llm_s'])} | {_fmt_miq(a['n_act'])} | {_fmt_miq(a['replans'])} | "
            f"{_fmt_miq(a['t_success'])} | {_fmt_miq(a['duration_s'])} |"
        )
    if any(s["peer_requests_sent"] or s["peer_transfers"] for s in summaries):
        lines += [
            "",
            "## Coordinación entre NPCs",
            "",
            "| Exp/Config | N | sesiones con entrega | peticiones (med.) | entregas (med.) | rechazos | "
            "timeouts de espera | arbitrajes | profundidad máx. |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for a in agg.values():
            lines.append(
                f"| {a['experiment_id']}/{a['config_label']} | {a['n']} | {a['sessions_with_transfer']} | "
                f"{_fmt_miq(a['peer_requests_sent'])} | {_fmt_miq(a['peer_transfers'])} | "
                f"{a['peer_requests_refused']} | {a['peer_wait_timeouts']} | {a['arbitrations']} | "
                f"{a['max_peer_depth']} |"
            )

    lines += [
        "",
        "## 4. Fracasos: causa principal y final de sesión",
        "",
        "| Exp/Config | fracasos | causas principales | final de sesión |",
        "|---|---|---|---|",
    ]
    for a in agg.values():
        causes = ", ".join(f"{cause}: {n}" for cause, n in a["causes"].most_common()) or "—"
        ends = ", ".join(f"{end}: {n}" for end, n in a["end_reasons"].most_common())
        lines.append(f"| {a['experiment_id']}/{a['config_label']} | {a['n'] - a['successes']} | {causes} | {ends} |")

    lines += [
        "",
        "## Notas de lectura",
        "",
        "- **Éxito**: la sesión cierra TODOS los goals del manifiesto con `goal_belief_met=true`. "
        "Una sesión cortada por tiempo cuenta como fracaso (no se descarta).",
        "- **Sin replan**: éxito sin ningún `goal_replan_required` (el primer plan bastó).",
        "- **IC95** de Wilson. **Fisher** exacto bilateral sobre éxito/fracaso. **Mann-Whitney** "
        "con aproximación normal (corrección de empates y de continuidad). Con N≈8 por brazo "
        "la potencia es baja: un p alto NO demuestra igualdad.",
        "- La fila **(todos)** mezcla experimentos de dificultad distinta: descriptiva, "
        "no sustituye a la comparación por experimento.",
        "- **Causa principal** (por prioridad): `failed_after_replans:<motivo>` > `failed:<motivo>` > "
        "`no_applicable_variant` > `closed_without_belief` > `timeout` > `pipeline_error` > `unknown`. "
        "Las señales secundarias por sesión están en el CSV.",
        "",
    ]
    return "\n".join(lines)


def _flatten(row: dict) -> dict:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, tuple) and len(value) == 3:
            out[f"{key}_median"], out[f"{key}_q1"], out[f"{key}_q3"] = value
        elif isinstance(value, Counter):
            out[key] = json.dumps(dict(value), ensure_ascii=False)
        elif isinstance(value, list):
            out[key] = ";".join(str(v) for v in value)
        else:
            out[key] = value
    return out


def write_csv(summaries: list[dict], agg: dict, comparisons: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("ablation_sessions.csv", [_flatten(s) for s in summaries]),
        ("ablation_aggregate.csv", [_flatten(a) for a in agg.values()]),
        ("ablation_comparisons.csv", [_flatten(c) for c in comparisons]),
    ):
        if not rows:
            continue
        fields: list[str] = []
        for row in rows:
            fields.extend(k for k in row if k not in fields)
        with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Análisis de la ablación de sub-planes (Fase 16).")
    parser.add_argument("sessions", type=Path, help="logs/sessions o un directorio de sesión")
    parser.add_argument("--suite", default="ablation_builtins", help="id de suite o ruta a su JSON")
    parser.add_argument("--out", type=Path, default=None, help="directorio de salida (markdown + CSV)")
    parser.add_argument("--csv", type=Path, default=None, help="directorio para los CSV (por defecto = --out)")
    parser.add_argument("--since-epoch", type=float, default=None,
                        help="solo sesiones con session_start posterior (epoch s)")
    args = parser.parse_args(argv)

    if not args.sessions.exists():
        print(f"No existe: {args.sessions}", file=sys.stderr)
        return 2

    suite = load_suite(args.suite)
    manifests = load_manifests(suite.get("experiments") or [])
    summaries, agg, comparisons = analyze(args.sessions, suite, manifests, since_epoch=args.since_epoch)
    md = to_markdown(summaries, agg, comparisons, suite)

    csv_dir = args.csv or args.out
    if csv_dir:
        write_csv(summaries, agg, comparisons, csv_dir)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        out_path = args.out / "ABLACION_SUBPLANES_RESULTADOS.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"Markdown escrito en {out_path}", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
