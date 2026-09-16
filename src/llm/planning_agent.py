from __future__ import annotations

import asyncio
import json
import logging

import spade

from llm.prompts.planning import build_prompt
from llm.parser import parse_llm_response
from llm.validator import validate_parse_goals_json
from llm.pipeline.pipeline_runner import run_full_pipeline
from llm.schemas import ParseGoalsResponse

log = logging.getLogger(__name__)

# Fase 3: tareas (no-pipeline) que usan structured output. La FORMA la garantiza
# el esquema Pydantic; la validación semántica (_validate) sigue corriendo.
_SCHEMA_BY_TASK = {
    "parse_goals": ParseGoalsResponse,
}


def _structured_to_result(task_type: str, instance):
    """Convierte la instancia Pydantic al shape que esperan los consumidores
    (downstream sigue recibiendo lo mismo que con el parseo manual)."""
    if isinstance(instance, ParseGoalsResponse):
        return [g.model_dump() for g in instance.goals]
    return instance.model_dump()


class LLMPlanningAgent(spade.agent.Agent):
    """
    Agente SPADE que atiende tareas de planning de cualquier NPCAgent.
    Recibe mensajes XMPP con {"task": ..., ...} y responde con
    {"ok": bool, "result": ..., "errors": [...]}.

    Delega las llamadas LLM en el provider inyectado desde main.py: el
    `SpadeLLMProviderAdapter` de `llm.providers`, que envuelve spade-llm
    (LiteLLM) y expone `complete(prompt, system)`. Ver self.llm_provider / llm_call.
    """

    def __init__(self, jid: str, password: str, llm_provider):
        super().__init__(jid, password)
        self.llm_provider = llm_provider

    async def setup(self) -> None:
        self.add_behaviour(PlanningRequestBehaviour())
        log.info("[LLM_AGENT] LLMPlanningAgent listo")

    async def llm_call(self, prompt: str, system: str = "") -> str:
        """Delega al proveedor LLM configurado."""
        return await self.llm_provider.complete(prompt=prompt, system=system)

    async def llm_call_traced(
        self,
        prompt: str,
        system: str = "",
        *,
        npc_id: str | None = None,
        goal: str | None = None,
        step: str = "unknown",
    ) -> str:
        """
        Como llm_call pero registra el prompt, la respuesta y la latencia en:
          - trace.jsonl  (evento compacto "llm_call")
          - prompts.jsonl (prompt + respuesta completos)
        """
        import time
        from utils.trace_logger import TraceLogger

        t0 = time.time()
        raw = await self.llm_provider.complete(prompt=prompt, system=system)
        latency = round(time.time() - t0, 3)

        tl = TraceLogger.get()
        if tl is not None:
            tl.trace(
                "llm_call",
                npc_id=npc_id,
                goal=goal,
                step=step,
                latency_s=latency,
                prompt_chars=len(prompt),
                response_chars=len(raw),
            )
            tl.log_prompt(
                npc_id=npc_id,
                goal=goal,
                step=step,
                prompt=prompt,
                system=system,
                response=raw,
                latency_s=latency,
            )
            if npc_id:
                m = tl.metrics.npc(npc_id)
                m.llm_calls += 1
                m.llm_total_s += latency

        log.debug(
            "[LLM:%s:%s] step=%s latency=%.2fs prompt_chars=%d",
            npc_id or "-", goal or "-", step, latency, len(prompt),
        )
        log.debug(
            "[LLM_PROMPT:%s:%s]\n--- SYSTEM ---\n%s\n--- USER ---\n%s\n--- RESPONSE ---\n%s",
            npc_id or "-", step,
            system or "(none)",
            prompt,
            raw,
        )
        return raw

    async def llm_call_structured_traced(
        self,
        prompt: str,
        system: str,
        schema,
        *,
        npc_id: str | None = None,
        goal: str | None = None,
        step: str = "unknown",
    ):
        """Como llm_call_traced pero con structured output: devuelve la instancia
        Pydantic de `schema`. Traza igual (la respuesta serializada como JSON)."""
        import time
        from utils.trace_logger import TraceLogger

        t0 = time.time()
        instance = await self.llm_provider.complete_structured(
            prompt=prompt, system=system, schema=schema
        )
        latency = round(time.time() - t0, 3)
        raw = instance.model_dump_json()

        tl = TraceLogger.get()
        if tl is not None:
            tl.trace(
                "llm_call",
                npc_id=npc_id,
                goal=goal,
                step=step,
                latency_s=latency,
                prompt_chars=len(prompt),
                response_chars=len(raw),
                structured=schema.__name__,
            )
            tl.log_prompt(
                npc_id=npc_id,
                goal=goal,
                step=step,
                prompt=prompt,
                system=system,
                response=raw,
                latency_s=latency,
            )
            if npc_id:
                m = tl.metrics.npc(npc_id)
                m.llm_calls += 1
                m.llm_total_s += latency

        log.debug(
            "[LLM:%s:%s] step=%s latency=%.2fs structured=%s",
            npc_id or "-", goal or "-", step, latency, schema.__name__,
        )
        return instance


class PlanningRequestBehaviour(spade.behaviour.CyclicBehaviour):
    """Escucha y atiende tareas de planning entrantes por XMPP."""

    @staticmethod
    def _trace_queue_wait(payload: dict, task_type: str) -> None:
        """Fase 11 (T5): traza y acumula el tiempo que una tarea esperó en la
        cola del LLMPlanningAgent (compartido entre NPCs) antes de empezar a
        atenderse. `_enqueued_at` lo pone LLMBehaviour.run() al enviar; si
        falta (mensaje antiguo o de otro origen), no se traza — no se fabrica
        un tiempo de espera que no se puede medir (regla de no-silencio: la
        ausencia se documenta, no se rellena con un valor inventado)."""
        import time
        from utils.trace_logger import TraceLogger

        enqueued_at = payload.get("_enqueued_at")
        if enqueued_at is None:
            return
        try:
            wait_s = round(time.time() - float(enqueued_at), 3)
        except (TypeError, ValueError):
            return
        npc_id = payload.get("npc_id") or None

        tl = TraceLogger.get()
        if tl is not None:
            tl.trace(
                "planning_queue_wait",
                npc_id=npc_id,
                task=task_type,
                wait_s=wait_s,
            )
            if npc_id:
                tl.metrics.npc(npc_id).planning_wait_s += wait_s
        log.debug(
            "[LLM_AGENT] queue_wait npc=%s task=%s wait_s=%.3f",
            npc_id or "-", task_type, wait_s,
        )

    async def run(self) -> None:
        msg = await self.receive(timeout=30)
        if not msg:
            return

        try:
            payload: dict = json.loads(msg.body)
        except json.JSONDecodeError as exc:
            log.warning(f"[LLM_AGENT] Mensaje malformado: {exc}")
            return

        task_type = payload.get("task", "unknown")
        sender_jid = str(msg.sender)
        log.info(f"[LLM_AGENT] Tarea '{task_type}' de {sender_jid}")
        self._trace_queue_wait(payload, task_type)

        try:
            result, errors = await self._handle(payload)
        except asyncio.TimeoutError:
            log.error("[LLM_AGENT] Timeout HTTP de Ollama en tarea '%s'", task_type)
            reply = spade.message.Message(
                to=sender_jid,
                thread=msg.thread,
                body=json.dumps({"ok": False, "result": None, "errors": ["llm_timeout"]}),
            )
            await self.send(reply)
            return  # behaviour sigue vivo para la siguiente tarea
        except Exception as exc:
            log.exception("[LLM_AGENT] Error inesperado en tarea '%s'", task_type)
            reply = spade.message.Message(
                to=sender_jid,
                thread=msg.thread,
                body=json.dumps({"ok": False, "result": None, "errors": [str(exc)]}),
            )
            await self.send(reply)
            return  # behaviour sigue vivo para la siguiente tarea

        reply = spade.message.Message(
            to=sender_jid,
            thread=msg.thread,
            body=json.dumps({"ok": not errors, "result": result, "errors": errors}),
        )
        await self.send(reply)

    async def _handle(self, payload: dict) -> tuple[dict, list[str]]:
        task_type = payload.get("task", "")

        if task_type == "generate_plan":
            return await self._handle_pipeline(payload)

        npc_id: str | None = payload.get("npc_id") or None
        max_retries = 2
        errors: list[str] = []
        schema = _SCHEMA_BY_TASK.get(task_type)

        for attempt in range(max_retries):
            prompt, system = build_prompt(payload, retry_errors=errors if attempt > 0 else None)
            goal = payload.get("goal_sig") or payload.get("task", "")
            if schema is not None:
                # Fase 3: structured output — la FORMA la garantiza Pydantic; la
                # validación SEMÁNTICA (_validate) sigue corriendo y alimenta los
                # reintentos igual que antes.
                instance = await self.agent.llm_call_structured_traced(
                    prompt, system, schema, npc_id=npc_id, goal=goal, step=task_type,
                )
                result = _structured_to_result(task_type, instance)
            else:
                raw = await self.agent.llm_call_traced(
                    prompt=prompt,
                    system=system,
                    npc_id=npc_id,
                    goal=goal,
                    step=task_type,
                )
                result = parse_llm_response(raw, payload.get("task", ""))
            errors = _validate(result, payload)
            if not errors:
                return result, []
            log.warning(f"[LLM_AGENT] Intento {attempt + 1} fallido: {errors}")

        return result, errors

    async def _handle_pipeline(self, payload: dict) -> tuple[dict, list[str]]:
        """Ejecuta el pipeline completo (steps 0→2→3→4→5) para un goal."""
        import time
        from utils.trace_logger import TraceLogger

        goal_sig = payload.get("goal_sig", "")
        npc_id: str | None = payload.get("npc_id") or None
        npc_statement = payload.get("npc_statement", "") or (
            goal_sig.replace("achieve_", "").replace("_", " ").capitalize()
        )
        existing_goals: list[str] = payload.get("known_goals", [])
        entity_catalog: dict = payload.get("entity_catalog") or {}
        beliefs: dict = payload.get("beliefs") or {}
        use_refinement = bool(payload.get("use_refinement", False))
        # Fase 16: ablación de sub-planes (move_to_and_pickup/craft_item).
        builtin_subplans = bool(payload.get("builtin_subplans", True))
        # Fase 17: coordinación planificada por el LLM.
        coordination = bool(payload.get("coordination", False))
        peers = payload.get("peers") or []
        capability_contracts_path = payload.get("capability_contracts_path")
        if capability_contracts_path is not None:
            capability_contracts_path = str(capability_contracts_path)
        # success_condition ya fue decidida por parse_goals — no debe re-derivarse
        success_condition: str | None = payload.get("success_condition") or None
        # T6: replan_hint cuando el BDI re-solicita un plan tras belief no cumplida
        replan_hint: str | None = payload.get("replan_hint") or None
        # Datos crudos del profile para step1b (escalera de variantes)
        _profile: dict = payload.get("profile") or {}
        _recipes: list[dict] = _profile.get("recipes") or []
        _item_spawns: list[dict] = _profile.get("item_spawns") or []

        # step_name se actualiza en cada llamada para que prompts.jsonl refleje el paso
        _current_step: list[str] = ["pipeline"]

        async def _llm_call(user_prompt: str, system_prompt: str, *, schema=None) -> str:
            # Fase 3: si el paso pasa un esquema Pydantic, usar structured output;
            # se devuelve el JSON del modelo para que el paso lo parsee/valide igual
            # (la FORMA queda garantizada; la validación semántica del paso se mantiene).
            if schema is not None:
                instance = await self.agent.llm_call_structured_traced(
                    user_prompt, system_prompt, schema,
                    npc_id=npc_id, goal=goal_sig, step=_current_step[0],
                )
                return instance.model_dump_json()
            return await self.agent.llm_call_traced(
                user_prompt,
                system_prompt,
                npc_id=npc_id,
                goal=goal_sig,
                step=_current_step[0],
            )

        tl = TraceLogger.get()
        t0 = time.time()
        if tl is not None:
            tl.trace("pipeline_start", npc_id=npc_id, goal=goal_sig)

        try:
            # Inyectar el step actual en cada llamada mediante el hook
            # que los steps del pipeline llaman — se detecta por la firma del prompt
            result = await run_full_pipeline(
                goal_sig=goal_sig,
                npc_statement=npc_statement,
                existing_goals=existing_goals,
                llm_call=_llm_call,
                use_refinement=use_refinement,
                success_condition=success_condition,
                capability_contracts_path=capability_contracts_path,
                entity_catalog=entity_catalog,
                beliefs=beliefs,
                recipes=_recipes,
                item_spawns=_item_spawns,
                replan_hint=replan_hint,
                builtin_subplans=builtin_subplans,
                coordination=coordination,
                peers=peers,
            )
            elapsed = round(time.time() - t0, 3)
            if tl is not None:
                tl.trace(
                    "pipeline_complete",
                    npc_id=npc_id,
                    goal=goal_sig,
                    elapsed_s=elapsed,
                    steps=len(result.steps),
                    contingencies=len(result.contingency_plans),
                    subgoals=len(result.subgoals_to_expand),
                )
                if npc_id:
                    tl.metrics.npc(npc_id).plan_steps_total += len(result.steps)
                _save_plan_asl(tl.session_dir, result)
                _save_plan_dag(tl.session_dir, result)
            return result.to_dict(), []
        except Exception as exc:
            elapsed = round(time.time() - t0, 3)
            if tl is not None:
                tl.trace(
                    "pipeline_error",
                    npc_id=npc_id,
                    goal=goal_sig,
                    elapsed_s=elapsed,
                    error=str(exc),
                )
            log.error(f"[LLM_AGENT] pipeline falló para '{goal_sig}': {exc}")
            return {}, [str(exc)]


def _save_plan_asl(session_dir, result) -> None:
    """Write all plan variants (main + contingencies) to <session_dir>/<goal_sig>.asl."""
    from pathlib import Path
    parts: list[str] = []
    if result.main_asl:
        guard_summary = " & ".join(
            f"{f['functor']}({', '.join(str(a) for a in f.get('args', []))})"
            for f in result.facts
        ) or "true"
        parts.append(f"// Happy path — guard: {guard_summary}\n{result.main_asl}")
    for cp in result.contingency_plans:
        parts.append(f"// Contingency — not ({cp.guard_expression})\n{cp.asl}")
    if parts:
        asl_path = Path(session_dir) / f"{result.sig}.asl"
        asl_path.write_text("\n\n".join(parts), encoding="utf-8")
        log.info("[LLM_AGENT] Plan ASL guardado \u2192 %s", asl_path.name)


def _save_plan_dag(session_dir, result) -> None:
    """Persist DAG snapshot for a generated plan in JSON format."""
    from pathlib import Path

    payload = {
        "sig": result.sig,
        "nodes": result.dag_nodes,
        "edges": result.dag_edges,
    }
    dag_path = Path(session_dir) / f"{result.sig}.dag.json"
    dag_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("[LLM_AGENT] DAG guardado \u2192 %s", dag_path.name)


def _validate(result: dict, payload: dict) -> list[str]:
    task = payload.get("task", "")

    if task == "parse_goals":
        goals_nl = payload.get("goals_nl", [])
        n_goals = len(goals_nl) if isinstance(goals_nl, list) else None
        return validate_parse_goals_json(result, n_goals=n_goals)
    if task == "prioritize":
        return _validate_prioritize(result)
    if task == "arbitrate":
        return _validate_arbitrate(result, payload)
    return []


def _validate_prioritize(result: object) -> list[str]:
    """0.C8 — Validación mínima del resultado de prioritize: lista de dicts con
    `sig` (str no vacío) y `score` (numérico). La Fase 3 lo migrará a Pydantic."""
    if not isinstance(result, list) or not result:
        return ["prioritize response must be a non-empty JSON array"]
    errors: list[str] = []
    for idx, item in enumerate(result):
        if not isinstance(item, dict):
            errors.append(f"prioritize[{idx}] must be an object")
            continue
        sig = item.get("sig")
        if not isinstance(sig, str) or not sig.strip():
            errors.append(f"prioritize[{idx}].sig must be a non-empty string")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            errors.append(f"prioritize[{idx}].score must be numeric")
    return errors


def _validate_arbitrate(result: object, payload: dict) -> list[str]:
    """Fase 14 — validación de `arbitrate`: `decision` debe ser "wait" o
    "switch"; si "switch", `goal_sig` debe ser un string no vacío Y estar
    entre los `candidates` que el payload le ofreció al LLM — política de
    no-fabricación del proyecto: si el modelo inventa un sig que no le dimos,
    es un error de validación (dispara reintento y, si se agota, el caller en
    bdi.py cae a la regla determinista), nunca se ejecuta un goal fabricado."""
    if not isinstance(result, dict):
        return ["arbitrate response must be a JSON object"]
    decision = result.get("decision")
    if decision not in ("wait", "switch"):
        return ["arbitrate.decision must be 'wait' or 'switch'"]
    if decision == "wait":
        return []
    goal_sig = result.get("goal_sig")
    if not isinstance(goal_sig, str) or not goal_sig.strip():
        return ["arbitrate.goal_sig must be a non-empty string when decision='switch'"]
    offered = {c.get("sig") for c in payload.get("candidates", []) if isinstance(c, dict)}
    if goal_sig not in offered:
        return [f"arbitrate.goal_sig {goal_sig!r} is not among the offered candidates"]
    return []
