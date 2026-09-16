from __future__ import annotations

"""ablation_monolithic.py — Ablación "prompt único" vs pipeline por etapas (5.3).

Argumento empírico central del TFM: ¿por qué un pipeline de contratos por etapas
y no un único prompt que pida el plan entero? Esta herramienta compara, para un
conjunto de goals, la tasa de planes ESTRUCTURALMENTE VÁLIDOS (mismos validadores)
generados por:
  - MONOLÍTICO: un solo prompt que pide {sig, success_condition, steps}.
  - PIPELINE:   run_full_pipeline (pasos 0→5, con repair).

NO toca el flujo por defecto: es solo herramienta de evaluación.

Uso:
    python tools/ablation_monolithic.py            # usa el set de goals por defecto
    python tools/ablation_monolithic.py --runs 3   # N intentos por goal
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.parser import parse_llm_response  # noqa: E402
from llm.catalogs import PRIMITIVE_ACTIONS  # noqa: E402
from llm.pipeline.step3_validator import validate_steps  # noqa: E402


# Escenario por defecto: el mundo de pan del proyecto (farmland/bakeri).
_DEFAULT_CATALOG = {
    "zone_ids": ["farmland", "bakeri"],
    "item_ids": ["wheat", "bread"],
    "delivery_tags": [],
    "recipe_ids": ["bread_recipe"],
}
_DEFAULT_BELIEFS = {"recipe": [["bread_recipe", "bakeri", "wheat", 2]],
                    "item_spawn": [["wheat", "farmland"]]}
_DEFAULT_GOALS = [
    "Collect one unit of wheat",
    "Fabricate one bread",
    "Explore the farmland zone",
]
_KNOWN_GOALS = {"move_to_and_pickup", "craft_item", "achieve_explore_zone"}


def build_monolithic_prompt(goal_nl: str, catalog: dict) -> tuple[str, str]:
    actions = ", ".join(sorted(PRIMITIVE_ACTIONS))
    system = (
        "You are a BDI planner. Given an objective, output the COMPLETE plan in a "
        "single JSON object. No markdown, no explanation."
    )
    user = f"""\
Objective: "{goal_nl}"

World catalog:
  zones: {catalog.get('zone_ids')}
  items: {catalog.get('item_ids')}
  recipes: {catalog.get('recipe_ids')}

Available primitive actions: {actions}
Available sub-goals: {sorted(_KNOWN_GOALS)}

Output JSON with the full plan in one shot:
{{
  "sig": "achieve_verb_object",
  "success_condition": "predicate(args)",
  "steps": [
    {{"type": "action", "name": "MoveTo", "args": [9, -6]}},
    {{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 1]}}
  ]
}}
Use only the actions/sub-goals listed. Args use real entities from the catalog.
"""
    return user, system


def validate_monolithic(result: object, known_goals: set[str]) -> tuple[bool, list[str]]:
    """True si el plan monolítico es estructuralmente válido con los MISMOS
    criterios que el pipeline: steps no vacíos, cada step con type/name, acción
    conocida o sub-goal conocido, y validate_steps sin errores."""
    errors: list[str] = []
    if not isinstance(result, dict):
        return False, ["respuesta no es un objeto JSON"]
    steps = result.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return False, ["steps vacío o ausente"]

    prim = {a.lower() for a in PRIMITIVE_ACTIONS}
    for i, s in enumerate(steps):
        if not isinstance(s, dict) or "type" not in s or "name" not in s:
            errors.append(f"step[{i}] sin type/name")
            continue
        name = str(s.get("name", ""))
        if s.get("type") == "action" and name.lower() not in prim:
            errors.append(f"step[{i}] acción desconocida: {name}")
        if s.get("type") == "subgoal" and name not in known_goals:
            errors.append(f"step[{i}] sub-goal desconocido: {name}")

    step_errors = validate_steps(steps, facts=[], guards=[], bound_vars=set())
    for se in step_errors:
        if se.has_errors():
            errors.extend(f"step[{se.step_index}] {e}" for e in se.errors)

    return (len(errors) == 0), errors


async def _run_monolithic(provider, goal_nl: str, catalog: dict) -> tuple[bool, list[str]]:
    user, system = build_monolithic_prompt(goal_nl, catalog)
    raw = await provider.complete(user, system)
    result = parse_llm_response(raw, "monolithic")
    return validate_monolithic(result, _KNOWN_GOALS)


async def _run_pipeline(provider, goal_nl: str, catalog: dict, beliefs: dict) -> tuple[bool, list[str]]:
    from llm.pipeline.pipeline_runner import run_full_pipeline

    async def _llm(u, s):
        return await provider.complete(u, s)

    try:
        result = await run_full_pipeline(
            goal_sig="", npc_statement=goal_nl, existing_goals=sorted(_KNOWN_GOALS),
            llm_call=_llm, use_refinement=True, entity_catalog=catalog,
            beliefs=beliefs,
        )
        ok = bool(getattr(result, "steps", None)) or bool(getattr(result, "variants", None))
        return ok, [] if ok else ["pipeline sin steps"]
    except Exception as exc:
        return False, [f"{type(exc).__name__}: {exc}"]


async def run_ablation(runs: int = 1) -> dict:
    from config import settings
    from llm.providers import build_provider

    provider = build_provider(settings)
    rows = []
    for goal in _DEFAULT_GOALS:
        mono_ok = pipe_ok = 0
        for _ in range(runs):
            m_ok, _ = await _run_monolithic(provider, goal, _DEFAULT_CATALOG)
            p_ok, _ = await _run_pipeline(provider, goal, _DEFAULT_CATALOG, _DEFAULT_BELIEFS)
            mono_ok += int(m_ok)
            pipe_ok += int(p_ok)
        rows.append({"goal": goal, "runs": runs, "mono_valid": mono_ok, "pipeline_valid": pipe_ok})
    return {"model": settings.llm_model, "rows": rows}


def to_markdown(report: dict) -> str:
    lines = [f"# Ablación monolítico vs pipeline (modelo: {report['model']})\n"]
    lines.append("| Goal | Runs | Monolítico válido | Pipeline válido |")
    lines.append("|---|---|---|---|")
    tm = tp = tr = 0
    for r in report["rows"]:
        lines.append(f"| {r['goal']} | {r['runs']} | {r['mono_valid']}/{r['runs']} | {r['pipeline_valid']}/{r['runs']} |")
        tm += r["mono_valid"]; tp += r["pipeline_valid"]; tr += r["runs"]
    lines.append(f"| **TOTAL** | {tr} | {tm}/{tr} | {tp}/{tr} |\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ablación monolítico vs pipeline (5.3).")
    parser.add_argument("--runs", type=int, default=1, help="intentos por goal")
    parser.add_argument("--json", action="store_true", help="salida JSON cruda")
    args = parser.parse_args(argv)

    report = asyncio.run(run_ablation(runs=args.runs))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
