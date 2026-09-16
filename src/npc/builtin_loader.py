from __future__ import annotations

"""Loads built-in ASL plans from plans/builtin/*.asl into the plan_graph.

Each file may define one or more variants of the same goal (same +!sig name,
different guards).  All variants are collected and registered as a READY
GoalNode with is_builtin=True, so the LLM pipeline can reference them as
already-available sub-goals.

ASL plan format expected:
  +!goal_sig(params, ...) : guard_expr <- body_step1; body_step2; ... .

Comments (//) are stripped before parsing.
"""

import logging
import re
from pathlib import Path

import networkx as nx

from npc.plan_graph import GoalNode, NodeStatus, PlanVariant

log = logging.getLogger(__name__)


# Regex: captures goal sig (with optional args), guard (optional), body
_PLAN_RE = re.compile(
    r"\+!(\w[\w,\s()]*?)"        # +!sig(params)  — group 1
    r"\s*(?::\s*(.+?))?"         # : guard        — group 2 (optional)
    r"\s*<-\s*"                  # <-
    r"(.+?)(?:\.\s*(?=\+!|$))",  # body           — group 3
    re.DOTALL,
)


def _strip_comments(text: str) -> str:
    return "\n".join(
        ln for ln in text.splitlines() if not ln.strip().startswith("//")
    )


def _parse_body(raw: str) -> list[str]:
    return [s.strip().rstrip(";. ") for s in re.split(r";\s*", raw) if s.strip()]


def _base_sig(sig_with_args: str) -> str:
    """'move_to_and_pickup(ItemId, N)' → 'move_to_and_pickup'"""
    return sig_with_args.split("(")[0].strip()


def _param_names(sig_with_args: str) -> list[str]:
    """'move_to_and_pickup(ItemId, N)' → ['ItemId', 'N']"""
    m = re.search(r'\(([^)]+)\)', sig_with_args)
    if not m:
        return []
    return [p.strip() for p in m.group(1).split(",") if p.strip()]


def load_builtin_plans(
    dag: nx.DiGraph,
    plans_dir: str | Path,
    *,
    exclude_files: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Parse all .asl files in plans_dir and register GoalNodes in *dag*.

    `exclude_files`: nombres de fichero a omitir (Fase 16, ablación de sub-planes).

    Returns list of goal sigs successfully loaded.
    """
    p = Path(plans_dir)
    if not p.exists():
        log.debug("[BUILTIN] Directory '%s' not found — skipping", plans_dir)
        return []

    loaded: list[str] = []
    for asl_file in sorted(p.glob("*.asl")):
        if exclude_files and asl_file.name in exclude_files:
            log.info("[BUILTIN] '%s' omitido (ablación de sub-planes)", asl_file.name)
            continue
        sigs = _load_file(asl_file, dag)
        loaded.extend(sigs)
    return loaded


def _load_file(path: Path, dag: nx.DiGraph) -> list[str]:
    text = _strip_comments(path.read_text(encoding="utf-8"))

    # Collect variants grouped by base sig
    variants_by_sig: dict[str, list[PlanVariant]] = {}

    # Also track param names (same for all variants of same sig)
    params_by_sig: dict[str, list[str]] = {}

    for m in _PLAN_RE.finditer(text):
        sig_raw, guard_raw, body_raw = m.group(1), m.group(2), m.group(3)
        sig = _base_sig(sig_raw)
        guard = (guard_raw or "true").strip()
        steps = _parse_body(body_raw)
        full_asl = m.group(0).strip()
        variants_by_sig.setdefault(sig, []).append(
            PlanVariant(guard=guard, steps=steps, full_asl=full_asl)
        )
        if sig not in params_by_sig:
            params_by_sig[sig] = _param_names(sig_raw)

    registered: list[str] = []
    for sig, variants in variants_by_sig.items():
        if dag.has_node(sig):
            node: GoalNode = dag.nodes[sig]["data"]
            # Merge variants if node already exists (shouldn't conflict in practice)
            node.variants.extend(variants)
            node.is_builtin = True
            node.status = NodeStatus.READY
            if not node.param_names:
                node.param_names = params_by_sig.get(sig, [])
        else:
            node = GoalNode(
                sig=sig,
                variants=variants,
                is_builtin=True,
                param_names=params_by_sig.get(sig, []),
                status=NodeStatus.READY,
            )
            dag.add_node(sig, data=node)
        registered.append(sig)
        asl_code = "\n".join(v.full_asl for v in variants)
        log.info(
            "[BUILTIN] Loaded '%s' (%d variant(s)) from %s:\n%s",
            sig, len(variants), path.name, asl_code,
        )

    return registered
