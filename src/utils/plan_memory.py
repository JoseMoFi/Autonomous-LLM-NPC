from __future__ import annotations

"""Plan memory — caché persistente de planes por NPC.

Estructura en disco:
    plans/memory/<npc_id>/pending/<goal_sig>.json   — métricas y metadata pending
    plans/memory/<npc_id>/approved/<goal_sig>.json  — métricas y metadata approved
    plans/memory/<npc_id>/asl/<goal_sig>.asl        — todas las variantes ASL de ese goal
    plans/memory/<npc_id>/_bundle.asl               — bundle con todos los planes

Criterio de promoción:
    uses_success >= promote_threshold  AND  success_rate >= promote_min_rate

Esquema del registro JSON:
    goal_sig        — nombre del goal (snake_case)
    npc_id          — NPC propietario del plan
    description     — npc_statement del plan (para inspección humana)
    asl_final       — texto ASL compilado del plan principal
    guard           — guard expression de la variante principal
    variants        — lista de [{guard, steps, full_asl}] para reconstruir GoalNode
    param_names     — parámetros formales del plan (si existen)
    uses_success    — veces que el goal se completó con éxito
    uses_error      — veces que el goal falló definitivamente
    success_rate    — uses_success / (uses_success + uses_error)  [0–1]
    status          — "pending" | "approved"
    created_at      — ISO8601
    last_updated_at — ISO8601
    last_used_at    — ISO8601 | null
    promoted_at     — ISO8601 (solo cuando status=="approved")
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npc.plan_graph import GoalNode, PlanVariant

log = logging.getLogger(__name__)

_PENDING  = "pending"
_APPROVED = "approved"


class PlanMemory:
    """
    Caché de planes en dos niveles para un NPC.

    Tier 1 (pending): el plan se guarda tras la primera generación LLM y se
    va acumulando feedback de éxito/error.

    Tier 2 (approved): cuando se alcanzan los umbrales, el plan se promueve.
    NOTA (0.D4): la carga AUTOMÁTICA del tier approved (que BDIBehaviour use el
    plan aprobado sin volver a llamar al LLM) aún NO está implementada — llegará
    en la Fase 4. Hoy la promoción solo marca el estado en disco.
    """

    def __init__(
        self,
        npc_id: str,
        memory_root: str | Path,
        promote_threshold: int = 3,
        promote_min_rate: float = 0.75,
        *,
        enabled: bool = True,
    ) -> None:
        self.npc_id = npc_id
        self.promote_threshold = promote_threshold
        self.promote_min_rate  = promote_min_rate
        # enabled=False (Fase 4): memoria desactivada. No crea directorios y todos
        # los métodos de lectura/escritura son no-ops (LLM puro, sin persistencia).
        self.enabled = enabled

        self._npc_dir = Path(memory_root) / npc_id
        self._pending_dir  = self._npc_dir / "pending"
        self._approved_dir = self._npc_dir / "approved"
        self._asl_dir = self._npc_dir / "asl"
        self._bundle_path = self._npc_dir / "_bundle.asl"
        if enabled:
            self._pending_dir.mkdir(parents=True, exist_ok=True)
            self._approved_dir.mkdir(parents=True, exist_ok=True)
            self._asl_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root_dir(self) -> Path:
        """Directorio raíz de este NPC en el run actual (para trazas/logs)."""
        return self._npc_dir

    @property
    def bundle_path(self) -> Path:
        return self._bundle_path

    def goal_asl_path(self, goal_sig: str) -> Path:
        return self._asl_dir / f"{goal_sig}.asl"

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------

    def load_approved(self, goal_sig: str) -> dict | None:
        """Devuelve el registro si está aprobado, None si no."""
        if not self.enabled:
            return None
        return self._read(self._approved_dir / f"{goal_sig}.json")

    def load_all_approved(self) -> list[dict]:
        """Devuelve todos los registros approved del NPC (Fase 4: carga al arranque).

        Vacío si la memoria está desactivada o no hay aprobados.
        """
        if not self.enabled or not self._approved_dir.exists():
            return []
        records: list[dict] = []
        for path in sorted(self._approved_dir.glob("*.json")):
            rec = self._read(path)
            if rec is not None:
                records.append(rec)
        return records

    def load_pending(self, goal_sig: str) -> dict | None:
        """Devuelve el registro pending si existe (puede no estar aprobado aún)."""
        return self._read(self._pending_dir / f"{goal_sig}.json")

    def load_all_reusable_pending(self) -> list[dict]:
        """Pending con EVIDENCIA de éxito (uses_success>=1, sin errores, rate>=umbral).

        Para el modo REUSE EXPLÍCITO: el usuario apunta a un run concreto y quiere
        reutilizar lo aprendido aunque no haya alcanzado el umbral de promoción a
        approved (pruebas cortas: una sesión = un uso, nunca llega a 2). La carga
        automática al arranque sigue siendo solo-approved (conservadora); esto es
        adicional y solo se activa bajo plan_memory_reuse (decisión del autor:
        aprender en una sesión y reusar en la siguiente)."""
        if not self.enabled or not self._pending_dir.exists():
            return []
        out: list[dict] = []
        for path in sorted(self._pending_dir.glob("*.json")):
            record = self._read(path)
            if not record:
                continue
            if (
                record.get("uses_success", 0) >= 1
                and record.get("uses_error", 0) == 0
                and (record.get("success_rate") or 0.0) >= self.promote_min_rate
            ):
                out.append(record)
        return out

    def is_approved(self, goal_sig: str) -> bool:
        return (self._approved_dir / f"{goal_sig}.json").exists()

    # ------------------------------------------------------------------
    # Almacenamiento (tras generación LLM)
    # ------------------------------------------------------------------

    def store(
        self,
        goal_sig: str,
        variants: list[dict],
        main_asl: str,
        *,
        description: str = "",
        param_names: list[str] | None = None,
        call_args: list | None = None,
        functor: str | None = None,
    ) -> None:
        """
        Guarda o actualiza el plan en el almacén pending.

        Si ya existe un registro con las MISMAS variantes, conserva los contadores
        uses_success / uses_error y solo actualiza metadatos. Si las variantes
        cambian (el LLM replanificó), es otro plan: los contadores vuelven a cero.
        Antes el plan nuevo heredaba el error del anterior (el límite de fallos de
        acción anota el error y luego replanifica) y la memoria nunca lo reusaba,
        aunque fuera el que cumplió el goal (batería memory_reuse, A5/SUB: 0/2).

        Args:
            goal_sig:    nombre del goal.
            variants:    lista de dicts {guard, steps, full_asl}.
            main_asl:    texto ASL del plan principal compilado.
            description: npc_statement (opcional, solo para inspección).
        """
        if not self.enabled:
            return
        now = _now()
        existing = self.load_pending(goal_sig)

        guard = variants[0]["guard"] if variants else "true"

        if existing is not None:
            if existing.get("variants") != variants:
                existing["uses_success"] = 0
                existing["uses_error"] = 0
                existing["success_rate"] = None
                existing["last_used_at"] = None
                existing["status"] = _PENDING
                existing["plan_revision"] = int(existing.get("plan_revision", 0) or 0) + 1
            existing["asl_final"] = main_asl
            existing["guard"]     = guard
            existing["variants"]  = variants
            existing["param_names"] = param_names or existing.get("param_names", [])
            existing["call_args"] = call_args if call_args is not None else existing.get("call_args", [])
            existing["functor"] = functor or existing.get("functor", goal_sig)
            if description:
                existing["description"] = description
            existing["last_updated_at"] = now
            record = existing
        else:
            record = {
                "goal_sig":       goal_sig,
                "npc_id":         self.npc_id,
                "description":    description,
                "asl_final":      main_asl,
                "guard":          guard,
                "variants":       variants,
                "param_names":    param_names or [],
                "call_args":      call_args or [],
                "functor":        functor or goal_sig,
                "uses_success":   0,
                "uses_error":     0,
                "success_rate":   None,
                "status":         _PENDING,
                "created_at":     now,
                "last_updated_at":now,
                "last_used_at":   None,
            }

        self._write(self._pending_dir / f"{goal_sig}.json", record)
        self._write_goal_asl(goal_sig, variants, param_names=param_names or [])
        self._rebuild_bundle_from_disk()
        log.debug("[PLAN_MEM:%s] Guardado pending: %s", self.npc_id, goal_sig)

    # ------------------------------------------------------------------
    # Registro de resultados de ejecución
    # ------------------------------------------------------------------

    def record_success(self, goal_sig: str) -> bool:
        """
        Incrementa uses_success.
        Devuelve True si acaba de promoverse a aprobado.
        """
        if not self.enabled:
            return False
        record, path = self._load_record_with_path(goal_sig)
        if record is None or path is None:
            return False

        record["uses_success"] += 1
        record["last_used_at"] = _now()
        _recalc_rate(record)
        self._write(path, record)

        if path.parent == self._pending_dir and self._meets_criteria(record):
            self._promote(goal_sig, record)
            return True
        return False

    def record_failure(self, goal_sig: str) -> None:
        """Incrementa uses_error (goal fallido definitivamente)."""
        if not self.enabled:
            return
        record, path = self._load_record_with_path(goal_sig)
        if record is None or path is None:
            return

        record["uses_error"] += 1
        record["last_used_at"] = _now()
        _recalc_rate(record)
        self._write(path, record)

    def record_unverified(self, goal_sig: str) -> None:
        """0.C5 — Cierre sin success_condition: el goal se completó por la rama
        legacy y NO hay verificación real de la creencia objetivo. Se cuenta
        aparte (uses_unverified) para NO promover el plan a aprobado sin
        evidencia. No toca uses_success ni success_rate."""
        if not self.enabled:
            return
        record, path = self._load_record_with_path(goal_sig)
        if record is None or path is None:
            return

        record["uses_unverified"] = record.get("uses_unverified", 0) + 1
        record["last_used_at"] = _now()
        self._write(path, record)

    # ------------------------------------------------------------------
    # Reconstrucción de GoalNode desde registro
    # ------------------------------------------------------------------

    def build_goalnode(self, goal_sig: str, record: dict) -> "GoalNode":
        """
        Reconstruye un GoalNode con estado READY a partir de un registro
        aprobado o pending, sin llamar al LLM.
        """
        from npc.plan_graph import GoalNode, NodeStatus, PlanVariant

        variants = [
            PlanVariant(
                guard=v.get("guard", "true"),
                steps=v.get("steps", []),
                full_asl=v.get("full_asl", ""),
            )
            for v in record.get("variants", [])
        ]
        node = GoalNode(
            sig=goal_sig,
            status=NodeStatus.READY,
            description=record.get("description", ""),
            param_names=record.get("param_names", []),
            call_args=record.get("call_args", []),
        )
        node.variants = variants
        node.success_count = record.get("uses_success", 0)
        node.failure_count = record.get("uses_error", 0)
        return node

    def rebuild_bundle_from_graph(self, plan_graph: Any) -> None:
        """Escribe _bundle.asl con la unión de disco + plan_graph actual.

        Los planes activos del grafo sobrescriben al texto persistido en disco para
        el mismo goal_sig. Así el bundle refleja el estado más reciente en memoria
        sin perder goals persistidos de sesiones anteriores.
        """
        if not self.enabled:
            return
        plans_by_sig = self._load_disk_goal_texts()

        if plan_graph is not None:
            for _sig, attrs in plan_graph.nodes(data=True):
                node = attrs.get("data") if isinstance(attrs, dict) else None
                if node is None:
                    continue
                text = self._render_goalnode_asl(node)
                if text:
                    plans_by_sig[node.sig] = text

        self._write_bundle(plans_by_sig)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _meets_criteria(self, record: dict) -> bool:
        rate = record.get("success_rate") or 0.0
        return (
            record["uses_success"] >= self.promote_threshold
            and rate >= self.promote_min_rate
        )

    def _load_record_with_path(self, goal_sig: str) -> tuple[dict | None, Path | None]:
        pending_path = self._pending_dir / f"{goal_sig}.json"
        approved_path = self._approved_dir / f"{goal_sig}.json"

        record = self._read(pending_path)
        if record is not None:
            return record, pending_path

        record = self._read(approved_path)
        if record is not None:
            return record, approved_path

        return None, None

    def _promote(self, goal_sig: str, record: dict) -> None:
        record["status"]      = _APPROVED
        record["promoted_at"] = _now()
        self._write(self._approved_dir / f"{goal_sig}.json", record)
        log.info(
            "[PLAN_MEM:%s] Plan APROBADO '%s'  ✓%d errores=%d  tasa=%.0f%%",
            self.npc_id, goal_sig,
            record["uses_success"], record["uses_error"],
            (record.get("success_rate") or 0) * 100,
        )

    def _read(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("[PLAN_MEM:%s] Error leyendo %s: %s", self.npc_id, path.name, exc)
            return None

    def _write(self, path: Path, record: dict) -> None:
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_goal_asl(
        self,
        goal_sig: str,
        variants: list[dict],
        *,
        param_names: list[str],
    ) -> None:
        text = self._render_variants_asl(goal_sig, variants, param_names=param_names)
        if not text:
            return
        self.goal_asl_path(goal_sig).write_text(text + "\n", encoding="utf-8")

    def _rebuild_bundle_from_disk(self) -> None:
        self._write_bundle(self._load_disk_goal_texts())

    def _load_disk_goal_texts(self) -> dict[str, str]:
        plans_by_sig: dict[str, str] = {}
        for path in sorted(self._asl_dir.glob("*.asl")):
            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                log.warning("[PLAN_MEM:%s] Error leyendo %s: %s", self.npc_id, path.name, exc)
                continue
            if text:
                plans_by_sig[path.stem] = text
        return plans_by_sig

    def _write_bundle(self, plans_by_sig: dict[str, str]) -> None:
        parts: list[str] = []
        for goal_sig in sorted(plans_by_sig):
            text = plans_by_sig[goal_sig].strip()
            if not text:
                continue
            parts.append(f"// goal: {goal_sig}\n{text}")
        payload = "\n\n".join(parts)
        self._bundle_path.write_text((payload + "\n") if payload else "", encoding="utf-8")

    def _render_goalnode_asl(self, node: "GoalNode") -> str:
        if not getattr(node, "variants", None):
            return ""

        variants = [
            {
                "guard": variant.guard,
                "steps": list(variant.steps),
                "full_asl": variant.full_asl,
            }
            for variant in node.variants
        ]
        return self._render_variants_asl(
            node.sig,
            variants,
            param_names=list(getattr(node, "param_names", []) or []),
        )

    def _render_variants_asl(
        self,
        goal_sig: str,
        variants: list[dict],
        *,
        param_names: list[str],
    ) -> str:
        rendered: list[str] = []
        for variant in variants:
            full_asl = str(variant.get("full_asl", "") or "").strip()
            if full_asl:
                rendered.append(full_asl)
                continue

            guard = str(variant.get("guard", "true") or "true").strip()
            steps = variant.get("steps", []) or []
            body = ";\n    ".join(str(step).strip() for step in steps if str(step).strip()) or "true"
            head_args = f"({', '.join(param_names)})" if param_names else ""
            rendered.append(f"+!{goal_sig}{head_args} : {guard} <-\n    {body}.")

        return "\n\n".join(rendered)


# ------------------------------------------------------------------
# Helpers de módulo
# ------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _recalc_rate(record: dict) -> None:
    total = record["uses_success"] + record["uses_error"]
    record["success_rate"] = round(record["uses_success"] / total, 3) if total else None


# ------------------------------------------------------------------
# Memoria por ejecución: resolución del run root (añadido del autor)
# ------------------------------------------------------------------

def _latest_run(runs_dir: Path) -> Path | None:
    """Devuelve el run más reciente, o None.

    Los runs se nombran con timestamp (%Y%m%d_%H%M%S[_%f]); el orden lexicográfico
    coincide con el cronológico, así que ordenamos por nombre (determinista, a
    diferencia de mtime).
    """
    if not runs_dir.exists():
        return None
    candidates = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name)


def _new_run(runs_dir: Path) -> Path:
    """Crea y devuelve un nuevo run con id de timestamp."""
    run = runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    # Evitar colisión si se crean dos en el mismo segundo.
    if run.exists():
        run = runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=True)
    return run


def resolve_run_root(settings: Any, base_dir: str | Path) -> tuple[Path | None, str]:
    """Decide el directorio raíz del run de plan memory según settings.

    Devuelve (root, mode) donde mode ∈ {"disabled", "new", "reuse"} para trazas.
      - plan_memory_enabled=False        → (None, "disabled")
      - plan_memory_reuse=False          → nuevo runs/<ts>/            ("new")
      - reuse + plan_memory_dir          → ese dir (o nuevo si no existe) ("reuse")
      - reuse + sin dir                  → el último run (o nuevo si no hay) ("reuse")
    """
    if not getattr(settings, "plan_memory_enabled", True):
        return None, "disabled"

    runs_dir = Path(base_dir) / "runs"

    if not getattr(settings, "plan_memory_reuse", False):
        return _new_run(runs_dir), "new"

    explicit = str(getattr(settings, "plan_memory_dir", "") or "").strip()
    if explicit:
        root = Path(explicit)
        if root.exists():
            return root, "reuse"
        log.warning("[PLAN_MEM] plan_memory_dir '%s' no existe — creando run nuevo", explicit)
        return _new_run(runs_dir), "new"

    latest = _latest_run(runs_dir)
    if latest is not None:
        return latest, "reuse"
    log.info("[PLAN_MEM] reuse pedido pero no hay runs previos — creando run nuevo")
    return _new_run(runs_dir), "new"


def build_plan_memory(npc_id: str, settings: Any, run_root: Path | None) -> "PlanMemory":
    """Crea la PlanMemory de un NPC para el run resuelto. Si run_root es None
    (memoria desactivada), devuelve una PlanMemory no-op."""
    enabled = run_root is not None
    root = run_root if run_root is not None else (Path(".") / "plans" / "memory" / "_disabled")
    return PlanMemory(
        npc_id=npc_id,
        memory_root=root,
        promote_threshold=int(getattr(settings, "plan_memory_min_uses", 2)),
        promote_min_rate=float(getattr(settings, "plan_memory_approval_rate", 0.5)),
        enabled=enabled,
    )
