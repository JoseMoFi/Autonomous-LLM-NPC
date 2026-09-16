"""Análisis de la batería de memoria de planes (suite `memory_reuse`).

Cada par comparte directorio de memoria (`plan_memory_run.run_dir` en la traza):
la 1.ª sesión (`<brazo>_M1`) planifica con el LLM y guarda el plan; la 2.ª
(`<brazo>_M2`) lo carga y lo reusa. Se compara, por experimento y brazo:

  - éxito y si la 2.ª sesión reusó de verdad (`plan_memory_reuse`);
  - velocidad: tiempo hasta el éxito, duración, llamadas y tiempo de LLM,
    peticiones de plan, replans y acciones;
  - diferencias PAREADAS 2.ª − 1.ª con Wilcoxon de rangos con signo (exacto
    hasta 20 pares).

Reutiliza el resumen por sesión de `analyze_ablation.py`.

Uso:
  python tools/analyze_memory.py logs/sessions --suite memory_reuse --out out/memory_reuse
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_ablation as AA  # noqa: E402

SPEED_METRICS = ("t_success", "duration_s", "llm_calls", "llm_s", "plan_requests", "replans", "n_act")
PAIRED_METRICS = ("t_success", "duration_s", "llm_calls", "llm_s", "plan_requests")


# ---------------------------------------------------------------------------
# Estadística
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(diffs: Iterable[float | None]) -> tuple[float, float] | None:
    """(W+, p bilateral) de Wilcoxon sobre diferencias pareadas. Se descartan los
    ceros; rangos medios en empates. Exacto (enumeración) hasta 20 pares no nulos;
    aproximación normal con corrección de empates por encima. None sin pares."""
    values = [float(d) for d in diffs if d is not None and float(d) != 0.0]
    n = len(values)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, values) if v > 0)
    if n <= 20:
        total = 1 << n
        le = ge = 0
        for signs in product((0, 1), repeat=n):
            w = sum(r for r, s in zip(ranks, signs) if s)
            if w <= w_plus + 1e-9:
                le += 1
            if w >= w_plus - 1e-9:
                ge += 1
        p = min(1.0, 2.0 * min(le, ge) / total)
        return w_plus, p
    mean = n * (n + 1) / 4.0
    ties: dict[float, int] = defaultdict(int)
    for r in ranks:
        ties[r] += 1
    var = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t ** 3 - t for t in ties.values()) / 48.0
    if var <= 0:
        return w_plus, 1.0
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var)
    p = math.erfc(max(z, 0.0) / math.sqrt(2.0))
    return w_plus, min(1.0, p)


# ---------------------------------------------------------------------------
# Sesiones y pares
# ---------------------------------------------------------------------------

def suite_configs(suite: dict) -> list[str]:
    return [f"{arm}_{p}" for arm in suite.get("arms") or () for p in suite.get("passes") or ()]


def _memory_signals(session_dir: Path) -> dict[str, Any]:
    run_dir, loaded, reused = None, 0, 0
    for e in AA._read_jsonl(session_dir / "trace.jsonl"):
        ev = e.get("ev")
        if ev == "plan_memory_run":
            run_dir = e.get("run_dir")
        elif ev == "plan_memory_load":
            loaded += int(e.get("count") or 0)
        elif ev == "plan_memory_reuse":
            reused += 1
    return {"memory_run_dir": run_dir, "memory_plans_loaded": loaded, "memory_reuses": reused}


def summarize(root: Path, suite: dict, manifests: dict[str, dict], *, since_epoch: float | None = None) -> list[dict]:
    configs = suite_configs(suite)
    summaries: list[dict] = []
    for session_dir in AA.find_session_dirs(root):
        s = AA.summarize_session(
            session_dir, manifests, configs,
            expected_flags=suite.get("config_builtin_subplans"),
            since_epoch=since_epoch,
        )
        if s is None:
            continue
        arm, _, pass_ = str(s["config_label"]).rpartition("_")
        s.update({"arm": arm, "pass": pass_})
        s.update(_memory_signals(session_dir))
        summaries.append(s)
    return summaries


def build_pairs(summaries: list[dict]) -> list[dict]:
    """Empareja M1 y M2 por directorio de memoria (y experimento/brazo). Si un
    directorio tiene varias sesiones de la misma pasada, se toma la última."""
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for s in sorted(summaries, key=lambda x: x["session_dir"]):
        if not s.get("memory_run_dir") or s.get("infra_error"):
            continue
        key = (s["experiment_id"], s["arm"], str(s["memory_run_dir"]))
        groups[key][s["pass"]] = s
    pairs = []
    for (exp, arm, run_dir), by_pass in sorted(groups.items()):
        if "M1" in by_pass and "M2" in by_pass:
            pairs.append({"experiment_id": exp, "arm": arm, "run_dir": run_dir,
                          "first": by_pass["M1"], "second": by_pass["M2"]})
    return pairs


def aggregate(summaries: list[dict]) -> dict[tuple[str, str, str], dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for s in summaries:
        if not s.get("infra_error"):
            groups[(s["experiment_id"], s["arm"], s["pass"])].append(s)
    out = {}
    for key, rows in groups.items():
        entry = {
            "n": len(rows),
            "success": sum(1 for r in rows if r["success"]),
            "first_plan_success": sum(1 for r in rows if r["first_plan_success"]),
            "memory_reused": sum(1 for r in rows if r.get("memory_reuses")),
        }
        for metric in SPEED_METRICS:
            values = [r[metric] for r in rows if r.get(metric) is not None]
            entry[metric] = AA.median_iqr(values)
        out[key] = entry
    return out


def compare_pairs(pairs: list[dict]) -> list[dict]:
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for pair in pairs:
        by_cell[(pair["experiment_id"], pair["arm"])].append(pair)
    rows = []
    for (exp, arm), cell in sorted(by_cell.items()):
        both_ok = [p for p in cell if p["first"]["success"] and p["second"]["success"]]
        row: dict[str, Any] = {
            "experiment_id": exp, "arm": arm, "pairs": len(cell), "pairs_both_success": len(both_ok),
            "second_reused": sum(1 for p in cell if p["second"].get("memory_reuses")),
        }
        for metric in PAIRED_METRICS:
            source = both_ok if metric == "t_success" else cell
            diffs = [
                p["second"][metric] - p["first"][metric]
                for p in source
                if p["first"].get(metric) is not None and p["second"].get(metric) is not None
            ]
            ratios = [
                p["second"][metric] / p["first"][metric]
                for p in source
                if p["first"].get(metric) and p["second"].get(metric) is not None
            ]
            row[f"{metric}_delta"] = AA.median_iqr(diffs)
            row[f"{metric}_ratio"] = AA.median_iqr(ratios)
            test = wilcoxon_signed_rank(diffs)
            row[f"{metric}_p"] = test[1] if test else None
            row[f"{metric}_n"] = len(diffs)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------

def _num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.{digits}f}"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _miq(stat: tuple, digits: int = 1) -> str:
    med, q1, q3 = stat
    if med is None:
        return "—"
    return f"{_num(med, digits)} [{_num(q1, digits)}, {_num(q3, digits)}]"


def to_markdown(summaries: list[dict], agg: dict, pairs: list[dict], comparisons: list[dict], suite: dict) -> str:
    shas = sorted({str(s.get("git_sha")) for s in summaries if s.get("git_sha")})
    lines = [
        f"# Memoria de planes — resultados (`{suite.get('id', '?')}`)",
        "",
        f"Sesiones analizadas: **{len(summaries)}** · pares completos: **{len(pairs)}** · commit(s): {', '.join(shas) or '—'}",
        "",
        "## 1. Éxito y reuso",
        "",
        "| Exp | Brazo | 1.ª: éxito | 2.ª: éxito | 1.ª: sin replan | 2.ª: sin replan | 2.ª reusó la memoria |",
        "|---|---|---|---|---|---|---|",
    ]
    cells = sorted({(k[0], k[1]) for k in agg})
    for exp, arm in cells:
        a, b = agg.get((exp, arm, "M1"), {}), agg.get((exp, arm, "M2"), {})
        lines.append(
            f"| {exp} | {arm} | {a.get('success', 0)}/{a.get('n', 0)} | {b.get('success', 0)}/{b.get('n', 0)} "
            f"| {a.get('first_plan_success', 0)}/{a.get('n', 0)} | {b.get('first_plan_success', 0)}/{b.get('n', 0)} "
            f"| {b.get('memory_reused', 0)}/{b.get('n', 0)} |"
        )
    lines += [
        "",
        "_La 2.ª sesión solo puede reusar si la 1.ª tuvo éxito: la memoria guarda el plan con evidencia de éxito "
        "(`uses_success ≥ 1`, sin errores). \"Reusó\" = al menos un `plan_memory_reuse` en la traza._",
        "",
        "## 2. Velocidad por sesión (mediana [Q1, Q3])",
        "",
        "| Exp | Brazo | Sesión | N | t hasta éxito (s) | duración (s) | llamadas LLM | t LLM (s) | peticiones de plan | replans | acciones |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for exp, arm in cells:
        for pass_, label in (("M1", "1.ª"), ("M2", "2.ª")):
            e = agg.get((exp, arm, pass_))
            if not e:
                continue
            lines.append(
                f"| {exp} | {arm} | {label} | {e['n']} | {_miq(e['t_success'])} | {_miq(e['duration_s'])} "
                f"| {_miq(e['llm_calls'], 0)} | {_miq(e['llm_s'])} | {_miq(e['plan_requests'], 0)} "
                f"| {_miq(e['replans'], 0)} | {_miq(e['n_act'], 0)} |"
            )
    lines += [
        "",
        "## 3. Diferencias pareadas (2.ª − 1.ª, mismo directorio de memoria)",
        "",
        "| Exp | Brazo | pares | t hasta éxito: Δ s (×) · p | duración: Δ s (×) · p | llamadas LLM: Δ (×) · p | t LLM: Δ s (×) · p | peticiones de plan: Δ · p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        def cell(metric: str, ratio: bool = True) -> str:
            med = c[f"{metric}_delta"][0]
            if med is None:
                return "—"
            text = _num(med)
            if ratio and c[f"{metric}_ratio"][0] is not None:
                text += f" (×{_num(c[f'{metric}_ratio'][0], 2)})"
            p = c[f"{metric}_p"]
            return text + (f" · {p:.3f}" if p is not None else " · —") + f" [n={c[f'{metric}_n']}]"
        lines.append(
            f"| {c['experiment_id']} | {c['arm']} | {c['pairs']} (ambas con éxito: {c['pairs_both_success']}) "
            f"| {cell('t_success')} | {cell('duration_s')} | {cell('llm_calls')} | {cell('llm_s')} "
            f"| {cell('plan_requests', ratio=False)} |"
        )
    lines += [
        "",
        "## 4. Pares",
        "",
        "| Exp | Brazo | directorio | 1.ª: éxito · t éxito · LLM | 2.ª: éxito · t éxito · LLM · reuso |",
        "|---|---|---|---|---|",
    ]
    for p in pairs:
        a, b = p["first"], p["second"]
        lines.append(
            f"| {p['experiment_id']} | {p['arm']} | `{Path(p['run_dir']).name}` "
            f"| {'✅' if a['success'] else '❌'} · {_num(a['t_success'])} · {a['llm_calls']} "
            f"| {'✅' if b['success'] else '❌'} · {_num(b['t_success'])} · {b['llm_calls']} · {b.get('memory_reuses', 0)} |"
        )
    lines += [
        "",
        "## Notas de lectura",
        "",
        "- **Δ** = mediana de las diferencias pareadas 2.ª − 1.ª (negativo = la 2.ª es más rápida o gasta menos); "
        "**×** = mediana del cociente 2.ª / 1.ª. El tiempo hasta éxito solo se compara en pares con éxito en ambas.",
        "- **p**: Wilcoxon de rangos con signo, bilateral, exacto hasta 20 pares (se descartan diferencias nulas). "
        "Con pocos pares la potencia es muy baja: un p alto no demuestra que no haya efecto.",
        "- La 2.ª sesión sigue llamando al LLM para interpretar el goal (`parse_goals`) y, si el plan reusado "
        "falla, para replanificar.",
        "- Los sub-planes que inventa el LLM (sub-goals propios) no se guardan en memoria: si el plan reusado los "
        "llama, la 2.ª sesión tendrá que replanificar.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(summaries: list[dict], pairs: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["experiment_id", "arm", "pass", "config_label", "session_dir", "git_sha", "success",
              "first_plan_success", "memory_run_dir", "memory_plans_loaded", "memory_reuses",
              *SPEED_METRICS, "primary_cause"]
    with open(out_dir / "memory_sessions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
    with open(out_dir / "memory_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment_id", "arm", "run_dir", "first_success", "second_success", "second_reuses",
                         *[f"first_{m}" for m in PAIRED_METRICS], *[f"second_{m}" for m in PAIRED_METRICS]])
        for p in pairs:
            writer.writerow([
                p["experiment_id"], p["arm"], p["run_dir"], p["first"]["success"], p["second"]["success"],
                p["second"].get("memory_reuses", 0),
                *[p["first"].get(m) for m in PAIRED_METRICS], *[p["second"].get(m) for m in PAIRED_METRICS],
            ])


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Análisis de la batería de memoria de planes.")
    parser.add_argument("sessions", type=Path, help="logs/sessions o un directorio de sesión")
    parser.add_argument("--suite", default="memory_reuse", help="id de suite o ruta a su JSON")
    parser.add_argument("--out", type=Path, default=None, help="directorio de salida (markdown + CSV)")
    parser.add_argument("--since-epoch", type=float, default=None,
                        help="solo sesiones con session_start posterior (epoch s)")
    args = parser.parse_args(argv)
    if not args.sessions.exists():
        print(f"No existe: {args.sessions}", file=sys.stderr)
        return 2

    suite = AA.load_suite(args.suite)
    manifests = AA.load_manifests(suite.get("experiments") or [])
    summaries = summarize(args.sessions, suite, manifests, since_epoch=args.since_epoch)
    pairs = build_pairs(summaries)
    md = to_markdown(summaries, aggregate(summaries), pairs, compare_pairs(pairs), suite)
    if args.out:
        write_csv(summaries, pairs, args.out)
        out_path = args.out / "MEMORIA_RESULTADOS.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"Markdown escrito en {out_path}", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
