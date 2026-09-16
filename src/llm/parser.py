from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)


def _repair_json(text: str) -> str:
    """Fix common LLM JSON mistakes before strict parsing.

    Known patterns:
    - Stray quote after a number before ] , or }: e.g. 1"] → 1]
      LLM sometimes emits the opening of a string literal it then forgets to close.
    """
    return re.sub(r'(\d)"(\s*[\],}])', r'\1\2', text)


def parse_llm_response(raw: str, task: str) -> dict:
    """
    Extrae el primer bloque JSON válido de la respuesta cruda del LLM.
    Tolerante a markdown code fences y texto extra alrededor del JSON.
    """
    # Quitar code fences ```json ... ``` o ``` ... ```
    cleaned = re.sub(r'```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace('```', '').strip()

    # Intentar parsear directamente, luego con reparación
    for candidate_text in (cleaned, _repair_json(cleaned)):
        try:
            return json.loads(candidate_text)
        except json.JSONDecodeError:
            pass

    # Buscar el primer bloque {...} o [...], con y sin reparación
    for pattern in (r'\{.*\}', r'\[.*\]'):
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            for candidate_text in (m.group(), _repair_json(m.group())):
                try:
                    return json.loads(candidate_text)
                except json.JSONDecodeError:
                    pass

    log.warning(f"[PARSER] No se pudo extraer JSON de respuesta para '{task}': {raw[:200]!r}")
    return {}


def extract_asl_from_response(raw: str) -> str:
    """
    Extrae texto ASL de la respuesta cruda del LLM.
    Busca bloques ```prolog o ```asl, o devuelve el texto limpio si no hay fences.
    """
    m = re.search(r'```(?:prolog|asl|agentspeak)?\s*(.*?)```', raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return raw.strip()


def compile_plan_to_asl(plan_json: dict) -> str:
    """
    Compila un JSON de plan al texto ASL correspondiente.
    Formato: +!sig : guard <- step1; step2; ... .

    Soporta múltiples variantes en 'variants' o una sola en 'guard'+'body'.
    """
    sig = plan_json.get("sig", "unknown_goal")
    variants = plan_json.get("variants")

    if not variants:
        # Variante única
        guard = plan_json.get("guard", "true")
        body = plan_json.get("body", [])
        fallback = plan_json.get("fallback")
        lines = [_format_variant(sig, guard, body)]
        if fallback:
            lines.append(_format_variant(sig, f"not ({guard})", [fallback]))
        return "\n\n".join(lines)

    # Múltiples variantes
    parts = []
    for v in variants:
        guard = v.get("guard", "true")
        body = v.get("body", [])
        parts.append(_format_variant(sig, guard, body))
    return "\n\n".join(parts)


def _format_variant(sig: str, guard: str, body: list[str]) -> str:
    if not body:
        # Fase 6.5: NO se fabrica un cuerpo `.fail` que el LLM no puso. El pipeline
        # ya falla ruidoso ante steps vacíos antes de llegar aquí; si se llegara,
        # es un bug → visible, no un plan inventado.
        raise ValueError(
            f"_format_variant: variante '{sig}' sin cuerpo (guard='{guard}'). "
            "El cuerpo debe venir del LLM; no se fabrica '.fail'."
        )
    steps = ";\n    ".join(body) + "."
    return f"+!{sig} : {guard} <-\n    {steps}"
