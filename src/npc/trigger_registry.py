from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from npc.beliefs import BeliefStore

log = logging.getLogger(__name__)


@dataclass
class TriggerRule:
    sig: str            # "+has_item"
    guard: str          # "N >= 2 & .pending_goal(achieve_craft_bread)"
    body: list[str]     # [".log('wheat disponible')"]
    full_asl: str
    source: str         # "builtin" (Fase 1) | "llm_reflexivo" (Fase 4)


class TriggerRegistry:
    """
    Catálogo de reglas reactivas ASL.
    Fase 1: solo reglas estáticas cargadas desde planes/triggers/*.asl al arranque.
    Fase 4: también se añaden reglas generadas por el LLM (source="llm_reflexivo").
    """

    def __init__(self) -> None:
        self._rules: list[TriggerRule] = []

    def load_from_path(self, path: str | Path) -> None:
        """Carga todos los .asl del directorio indicado."""
        p = Path(path)
        if not p.exists():
            log.debug(f"[TRIGGERS] Directorio '{path}' no existe — sin triggers built-in")
            return
        for asl_file in sorted(p.glob("*.asl")):
            self._load_file(asl_file)

    def _load_file(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        for rule in self._parse_asl_triggers(text, source="builtin"):
            self._rules.append(rule)
        log.debug(f"[TRIGGERS] Cargados {path.name}")

    def _parse_asl_triggers(self, text: str, source: str) -> list[TriggerRule]:
        """
        Parser mínimo de triggers ASL del tipo:
          +belief(args) : guard <- body.
        """
        rules: list[TriggerRule] = []
        # Eliminar comentarios de línea
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("//")]
        clean = " ".join(lines)
        # Separar por punto final de plan
        for plan_text in re.split(r'\.\s*(?=\+)', clean):
            plan_text = plan_text.strip()
            if not plan_text:
                continue
            m = re.match(
                r'^\+(!?)(\w[\w(,\s)]*?)\s*(?::\s*(.+?))?\s*<-\s*(.+)$',
                plan_text, re.DOTALL
            )
            if not m:
                continue
            is_goal, trigger_sig, guard, body_raw = m.groups()
            if is_goal:
                continue  # es un plan de goal, no un trigger
            body_actions = [s.strip().rstrip(";. ") for s in re.split(r';\s*', body_raw) if s.strip()]
            rules.append(TriggerRule(
                sig=f"+{trigger_sig.split('(')[0].strip()}",
                guard=guard.strip() if guard else "true",
                body=body_actions,
                full_asl=plan_text,
                source=source,
            ))
        return rules

    def add(self, rule: TriggerRule) -> None:
        self._rules.append(rule)

    @staticmethod
    def _head_args(full_asl: str) -> list[str]:
        """Lista CRUDA de args del head de un trigger, en orden y con posición.

        `+has_item(wheat, N) : ...` → ["wheat", "N"]. Conserva constantes,
        variables y wildcards `_` para poder emparejar por POSICIÓN contra cada
        tupla (las constantes restringen, las variables se ligan a su posición real).
        """
        m = re.match(r'^\s*\+!?\w+\(([^)]*)\)', full_asl)
        if not m:
            return []
        return [raw.strip() for raw in m.group(1).split(",") if raw.strip()]

    def evaluate(self, belief_key: str, beliefs: "BeliefStore") -> list[TriggerRule]:
        """Devuelve las reglas cuyo trigger casa con belief_key Y cuyo guard se
        satisface contra el estado actual de beliefs.

        El guard se evalúa con GuardEvaluator (intérprete agentspeak): para cada
        tupla actual del predicado se intenta ligar las variables del head y
        comprobar el guard. La regla dispara si ALGUNA tupla lo satisface (o si
        el guard es trivial).
        """
        from npc.asp_guard import GuardEvaluator

        candidates = [r for r in self._rules if r.sig == f"+{belief_key}"]
        if not candidates:
            return []

        snapshot = beliefs.snapshot()
        evaluator = GuardEvaluator()
        fired: list[TriggerRule] = []

        for rule in candidates:
            guard = (rule.guard or "true").strip()
            if guard == "true":
                fired.append(rule)
                continue

            head_args = self._head_args(rule.full_asl)
            tuples = snapshot.get(belief_key, [])
            satisfied = False

            # Sin head paramétrico o sin tuplas: evaluar el guard tal cual.
            if not head_args or not tuples:
                ok, _ = evaluator.eval_guard(guard, [], [], snapshot)
                satisfied = ok
            else:
                for tup in tuples:
                    if len(tup) < len(head_args):
                        continue
                    # Emparejar por POSICIÓN: las constantes del head deben casar
                    # el valor de la tupla; las variables se ligan a su valor real.
                    params: list[str] = []
                    call_args: list = []
                    matched = True
                    for i, ha in enumerate(head_args):
                        if ha == "_":
                            continue
                        if ha[0].isupper():            # variable → ligar por posición
                            params.append(ha)
                            call_args.append(tup[i])
                        elif str(tup[i]).lower() != ha.lower():  # constante → debe casar
                            matched = False
                            break
                    if not matched:
                        continue
                    ok, _ = evaluator.eval_guard(guard, params, call_args, snapshot)
                    if ok:
                        satisfied = True
                        break

            if satisfied:
                fired.append(rule)

        return fired

    def has_trigger_for(self, belief_key: str) -> bool:
        return any(r.sig == f"+{belief_key}" for r in self._rules)

    def list_sigs(self) -> list[str]:
        return [r.sig for r in self._rules]

    def remove_by_source(self, source: str) -> None:
        """Fase 4: elimina triggers de una fuente (ej: rollback de llm_reflexivo)."""
        self._rules = [r for r in self._rules if r.source != source]
