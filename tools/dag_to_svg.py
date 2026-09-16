"""dag_to_svg.py — dibuja el grafo de decisión que el pipeline guarda por goal.

Lee un snapshot `<sig>.dag.json` (lo escribe `llm.planning_agent._save_plan_dag`)
y produce un SVG con layout por niveles topológicos.

Uso:
    python tools/dag_to_svg.py <dag.json> <out.svg> [success_condition]
    python tools/dag_to_svg.py --batch <sessions_root> <out_root>

En modo --batch recorre `<sessions_root>/<fecha>/<hora>/*.dag.json` y escribe
`<out_root>/<fecha>/<hora>/<sig>.svg`, intentando recuperar la success_condition
del `trace.jsonl` de la misma sesión (best-effort).

Nota: hoy la "sesión" se identifica por fecha/hora del log. En el futuro convendrá
un generador de IDs trackeables; este script ya respeta esa estructura.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Descripciones de los sub-planes builtin (para anotar los nodos).
DESC = {
    "move_to_and_pickup": "localiza, navega y recoge el item (itemId, qty)",
    "craft_item": "navega a la zona de receta y craftea (recipeId, qty)",
    "achieve_explore_zone": "asegura conocer el centro de la zona (zoneTag)",
}
# Estilo por status del nodo del DAG (los pone el pipeline_runner).
STATUS_STYLE = {
    "main":         ("#eff6ff", "#2563eb", "GOAL principal"),
    "generating":   ("#eef2ff", "#6366f1", "sub-goal"),
    "pending":      ("#eef2ff", "#6366f1", "sub-goal"),
    "goal_failed":  ("#fef2f2", "#dc2626", "rama goal_failed"),
    "replan_stub":  ("#fffbeb", "#f59e0b", "rama replan"),
    "degrade_stub": ("#fffbeb", "#f59e0b", "rama degrade"),
    None:           ("#eef2ff", "#6366f1", "sub-plan (builtin)"),
}


def _layered(nodes: list[dict], edges: list[list[str]]) -> dict[str, int]:
    ids = [n["id"] for n in nodes]
    incoming = {i: 0 for i in ids}
    children: dict[str, list[str]] = {i: [] for i in ids}
    for a, b in edges:
        if a in children and b in incoming:
            children[a].append(b)
            incoming[b] += 1
    level = {i: 0 for i in ids}
    frontier = [i for i in ids if incoming[i] == 0]
    seen = set(frontier)
    while frontier:
        nxt: list[str] = []
        for u in frontier:
            for v in children[u]:
                if level[v] < level[u] + 1:
                    level[v] = level[u] + 1
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return level


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(dag_path: Path, out_svg: Path, success_condition: str = "") -> tuple[int, int]:
    data = json.loads(dag_path.read_text(encoding="utf-8"))
    nodes, edges = data["nodes"], data["edges"]
    level = _layered(nodes, edges)
    by_level: dict[int, list[dict]] = {}
    for n in nodes:
        by_level.setdefault(level[n["id"]], []).append(n)

    BW, BH, HGAP, VGAP, TOP, LEFT = 250, 70, 50, 150, 150, 60
    maxrow = max(len(v) for v in by_level.values())
    W = LEFT * 2 + maxrow * BW + (maxrow - 1) * HGAP
    W = max(W, LEFT + 18 * len(data["sig"]) + 360)  # ancho mínimo para el título
    H = TOP + (max(by_level) + 1) * (BH + VGAP)
    pos: dict[str, tuple[float, float]] = {}
    for lv, ns in by_level.items():
        row_w = len(ns) * BW + (len(ns) - 1) * HGAP
        x0 = (W - row_w) / 2
        for k, n in enumerate(ns):
            pos[n["id"]] = (x0 + k * (BW + HGAP), TOP + lv * (BH + VGAP))

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Segoe UI, Arial, sans-serif">',
        '<defs><marker id="ar" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker></defs>',
        f'<text x="{LEFT}" y="46" font-size="24" font-weight="700" fill="#0f172a">'
        f'Grafo de decisión generado — {_esc(data["sig"])}</text>',
        f'<text x="{LEFT}" y="74" font-size="14" fill="#64748b">'
        f'success_condition: {_esc(success_condition) or "(n/d)"}   ·   sesión: '
        f'{_esc(dag_path.parent.parent.name)}/{_esc(dag_path.parent.name)}</text>',
        f'<text x="{LEFT}" y="98" font-size="12" fill="#94a3b8">'
        f'nodos={len(nodes)} · aristas={len(edges)} · fuente: {_esc(dag_path.name)}</text>',
    ]
    for a, b in edges:
        if a not in pos or b not in pos:
            continue
        ax, ay = pos[a]
        bx, by = pos[b]
        x1, y1 = ax + BW / 2, ay + BH
        x2, y2 = bx + BW / 2, by
        my = (y1 + y2) / 2
        p.append(f'<path d="M{x1},{y1} C{x1},{my} {x2},{my} {x2},{y2-4}" '
                 f'stroke="#475569" stroke-width="1.8" fill="none" marker-end="url(#ar)"/>')
    for n in nodes:
        nid = n["id"]
        fill, stroke, role = STATUS_STYLE.get(n.get("status"), STATUS_STYLE[None])
        x, y = pos[nid]
        p.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="10" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        p.append(f'<text x="{x+14}" y="{y+22}" font-size="11" font-weight="700" '
                 f'fill="{stroke}">{role}</text>')
        p.append(f'<text x="{x+14}" y="{y+42}" font-size="14" font-weight="700" '
                 f'fill="#0f172a">{_esc(nid)}</text>')
        d = DESC.get(nid, "")
        if d:
            p.append(f'<text x="{x+14}" y="{y+60}" font-size="10.5" fill="#475569">{_esc(d)}</text>')
    p.append("</svg>")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(p), encoding="utf-8")
    return W, H


def _success_condition_from_trace(session_dir: Path, sig: str) -> str:
    """Best-effort: busca la success_condition del goal en el trace.jsonl de la sesión."""
    trace = session_dir / "trace.jsonl"
    if not trace.exists():
        return ""
    try:
        for line in trace.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("goal") != sig and obj.get("sig") != sig:
                continue
            for key in ("derived_condition", "success_condition"):
                val = obj.get(key)
                if isinstance(val, str) and val:
                    return val
    except OSError:
        return ""
    return ""


def batch(sessions_root: Path, out_root: Path) -> int:
    dags = sorted(sessions_root.glob("*/*/*.dag.json"))
    if not dags:
        print(f"No se encontraron *.dag.json bajo {sessions_root}")
        return 1
    for dag in dags:
        session_dir = dag.parent
        date_d, hour_d = session_dir.parent.name, session_dir.name
        sig = json.loads(dag.read_text(encoding="utf-8"))["sig"]
        sc = _success_condition_from_trace(session_dir, sig)
        out_svg = out_root / date_d / hour_d / f"{sig}.svg"
        w, h = render(dag, out_svg, sc)
        print(f"OK  {date_d}/{hour_d}/{sig}.svg  ({w}x{h})  sc={sc or '(n/d)'}")
    print(f"\nTotal: {len(dags)} grafos -> {out_root}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "--batch":
        return batch(Path(argv[1]), Path(argv[2]))
    if len(argv) < 2:
        print(__doc__)
        return 1
    sc = argv[2] if len(argv) > 2 else ""
    w, h = render(Path(argv[0]), Path(argv[1]), sc)
    print(f"OK -> {argv[1]}  ({w}x{h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
