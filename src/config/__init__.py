from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Settings:
    _instance: "Settings | None" = None

    def __init__(self, data: dict[str, Any]):
        self.unity_host: str = data["unity_host"]
        self.unity_port: int = data["unity_port"]
        self.xmpp_host: str = data["xmpp_host"]
        self.xmpp_port: int = data["xmpp_port"]
        self.llm_provider: str = data["llm_provider"]
        self.llm_model: str = data["llm_model"]
        self.llm_base_url: str = data["llm_base_url"]
        self.llm_temperature: float = data["llm_temperature"]
        # Modo "thinking" del modelo (qwen3 y similares). Tri-estado:
        #   False -> se envía think=False a Ollama (sin razonamiento; RÁPIDO). Default.
        #   True  -> se envía think=True (razonamiento explícito; LENTO).
        #   None  -> no se envía el parámetro (modelos sin thinking: qwen2.5/llama).
        # Solo aplica al proveedor ollama. Motivo: qwen3:8b en modo thinking llegó a
        # dar timeout incluso en parse_goals (~5 min); desactivarlo lo deja en ~3 s.
        self.llm_think: bool | None = data.get("llm_think", False)
        self.llm_timeout: int = data["llm_timeout"]
        self.llm_max_retries: int = data["llm_max_retries"]
        # Timeout (s) para una solicitud completa de plan al pipeline LLM,
        # fijado por bdi._request_plan. Antes era el literal 600.
        self.plan_timeout_s: int = data.get("plan_timeout_s", 600)
        # Timeout (s) por defecto que aplica ActionResultWaiter antes de abortar
        # una acción Unity sin respuesta. ExploreArea usa el suyo, más largo,
        # porque puede encadenar WanderRetry (hasta 10 intentos).
        self.action_timeout_s: float = float(data.get("action_timeout_s", 60))
        self.explore_timeout_s: float = float(data.get("explore_timeout_s", 180))
        self.gemini_api_key: str = data.get("gemini_api_key", "")
        self.reflect_every_n_goals: int = data.get("reflect_every_n_goals", 5)
        # --- Plan memory (Fase 4) ---------------------------------------------
        # Reuso de planes aprendidos entre sesiones + memoria por ejecución.
        # enabled: master on/off (off = ni carga ni persiste, LLM puro).
        # reuse:   reusar un run existente (True) vs crear uno nuevo cada vez (False).
        # dir:     run concreto a reusar; si reuse y vacío → el último run.
        # approval_rate / min_uses: política de promoción pending→approved.
        self.plan_memory_enabled: bool = bool(data.get("plan_memory_enabled", True))
        self.plan_memory_reuse: bool = bool(data.get("plan_memory_reuse", False))
        self.plan_memory_dir: str = str(data.get("plan_memory_dir", "") or "")
        self.plan_memory_approval_rate: float = float(data.get("plan_memory_approval_rate", 0.5))
        self.plan_memory_min_uses: int = int(data.get("plan_memory_min_uses", 2))
        # Refinamiento V4 (reuse-first): si False, el pipeline genera planes
        # atómicos sin descomposición en sub-goals (config B de la ablación, Fase 5).
        self.use_refinement: bool = bool(data.get("use_refinement", True))
        # --- Fase 6.5: reuso canónico de planes -------------------------------
        # canonical_reuse_enabled: canonicaliza el `sig` desde la ESTRUCTURA de la
        # success_condition (no del NL). Su valor real es estabilizar el naming
        # entre sesiones → habilita el match de PLAN MEMORY cross-sesión (el `sig`
        # inestable del LLM lo rompía). Default OFF (no-regresión). El reuso en sí
        # lo aporta plan_memory (Fase 4): la clave canónica solo hace el matching
        # posible. Env NPC_CANONICAL_REUSE.
        self.canonical_reuse_enabled: bool = bool(data.get("canonical_reuse_enabled", False))
        # canonical_family_plan: genera UNA familia paramétrica DETERMINISTA para
        # goals has_item (src/llm/family_plan.py) saltando el LLM en la PROPIA
        # sesión. Es un paradigma DISTINTO al de memoria episódica (aquí el agente
        # no "aprende" nada que persistir): sirve para sesiones largas/generativas,
        # no para el flujo aprender→persistir→reusar entre sesiones. Default OFF
        # para no preemptar el aprendizaje. Env NPC_CANONICAL_FAMILY.
        self.canonical_family_plan: bool = bool(data.get("canonical_family_plan", False))
        # --- Fase 12: coordinación NPC↔NPC (consulta, delegación, intercambio) ---
        # coordination_enabled: master on/off. Off (default) = NPCAgent no añade
        # PeerCoordBehaviour, bdi.py no registra .ask_peer/.request_peer/.await_peer/
        # .deliver_to_peer, y family_plan no emite la variante `delegate` — cero
        # cambios de comportamiento respecto a la Fase 11. Env NPC_COORDINATION.
        self.coordination_enabled: bool = bool(data.get("coordination_enabled", False))
        # Fase 17: quién planifica bajo coordinación. "family" (default, Fases
        # 12-14): familia determinista + plan de entrega, sin LLM. "llm": el
        # pipeline LLM planifica el trabajo de cada NPC y cómo pedir ayuda
        # (peldaños de step1b para lo que solo puede dar otro NPC); la entrega
        # (.drop + aviso) sigue siendo andamiaje CODE. Env NPC_COORDINATION_PLANNER.
        self.coordination_planner: str = str(data.get("coordination_planner", "family") or "family")
        # Timeout (s) esperando agree/refuse + posterior inform-done/failure de un
        # `request_peer`. Cubre TODO el ciclo de vida de la delegación (el receptor
        # puede tardar en resolver su propio goal).
        self.peer_request_timeout_s: float = float(data.get("peer_request_timeout_s", 120))
        # Timeout (s) de una consulta query-if/inform (mucho más corta: solo espera
        # una lectura de creencias del receptor, no un goal completo).
        self.peer_query_timeout_s: float = float(data.get("peer_query_timeout_s", 15))
        # Límite anti-spam de peticiones ACEPTADAS que un NPC atiende por sesión.
        self.max_peer_requests: int = int(data.get("max_peer_requests", 5))
        # Fase 17t: intentos fallidos de un mismo encargo (peticionario + item + qty)
        # tras los que el receptor lo rechaza (`already_failed`). 2 = un reintento.
        self.peer_max_failed_attempts: int = int(data.get("peer_max_failed_attempts", 2))
        # Profundidad máxima de una cadena de delegación (anti-bucle A→B→A→B...).
        # Default 2: cubre el caso de cooperación bidireccional (E6 — A pide a B,
        # B necesita un ingrediente que solo tiene A, B se lo pide a A) sin permitir
        # cadenas más largas. depth=1 bastaría para delegación de un solo salto (E5).
        self.peer_max_depth: int = int(data.get("peer_max_depth", 2))
        # --- Fase 13: etiquetado de sesión para la batería experimental ---
        # Puramente informativos (van a session_start en la traza); no
        # cambian ningún comportamiento del sistema. Los pone
        # tools/run_experiment.ps1 vía entorno; vacíos en uso normal.
        self.experiment_id: str = str(data.get("experiment_id", "") or "")
        self.config_label: str = str(data.get("config_label", "") or "")
        self.experiment_n_opt: float | None = data.get("experiment_n_opt")
        self.experiment_npcs: int | None = data.get("experiment_npcs")
        # Triggers built-in (src/plans/triggers/*.asl, Fase 9). Default True
        # (sin cambio de comportamiento normal). La batería experimental
        # (Fase 13) los desactiva: el trigger demostrativo de inventory.asl
        # (wheat>=2 -> adopta achieve_bake_bread) contamina experimentos de
        # un solo goal aislado con un segundo goal no pedido. Env
        # NPC_BUILTIN_TRIGGERS.
        self.builtin_triggers_enabled: bool = bool(data.get("builtin_triggers_enabled", True))
        # --- Fase 16: ablación de sub-planes escritos a mano -------------------
        # builtin_subplans_enabled=False: move_to_and_pickup y craft_item NO se
        # cargan ni se ofrecen al LLM (lista de sub-goals, prompts, step1b, step3,
        # mini-repair, simulador) — el LLM compone el plan con acciones
        # primitivas. Default True (sin cambio). Env NPC_BUILTIN_SUBPLANS.
        self.builtin_subplans_enabled: bool = bool(data.get("builtin_subplans_enabled", True))
        # isolate_capability_contracts: lee/escribe los contratos de capacidad en
        # el directorio de la sesión en vez de src/plans/contracts/ (versionado),
        # para que un sub-plan creado por el LLM en un run no contamine el
        # siguiente. Default False. Env NPC_ISOLATE_CONTRACTS.
        self.isolate_capability_contracts: bool = bool(data.get("isolate_capability_contracts", False))
        # session_max_s: >0 = cierre ORDENADO al vencer (metrics.json escrito,
        # goals abiertos registrados como timeout) en vez de que el lanzador mate
        # el proceso. 0 = sin límite (uso normal). Env NPC_SESSION_MAX_S.
        self.session_max_s: float = float(data.get("session_max_s", 0) or 0)
        self.log_level: str = data.get("log_level", "INFO")
        # Si True, el sistema se apaga solo cuando TODOS los NPCs han resuelto sus
        # goals (completados o fallados sin repair). Útil para pruebas/eval: evita
        # esperar a un timeout. Por defecto False (uso interactivo). El env
        # NPC_SHUTDOWN_WHEN_IDLE lo fuerza (lo usa tools/test_session.ps1).
        self.shutdown_when_idle: bool = bool(data.get("shutdown_when_idle", False))
        # --- Fase 14: arbitraje de goals (preempción, desbloqueo de E6) ---------
        # goal_arbitration_enabled: master on/off de TODA la preempción. Off
        # (default) = comportamiento idéntico a Fase 11-13 (una intención por NPC,
        # sin excepciones). Solo tiene efecto si coordination_enabled también está
        # activo (la condición de disparo requiere un peer waiter bloqueando). Env
        # NPC_GOAL_ARBITRATION.
        self.goal_arbitration_enabled: bool = bool(data.get("goal_arbitration_enabled", False))
        # goal_arbitration_mode: "llm" consulta la tarea LLM `arbitrate` (con
        # fallback determinista si falla/timeout/no valida); "rule" usa SIEMPRE
        # la regla determinista de ciclo de espera de longitud 2, sin LLM. Env
        # NPC_ARBITRATION_MODE.
        self.goal_arbitration_mode: str = str(data.get("goal_arbitration_mode", "llm") or "llm")
        # Tope duro de cambios de goal (goal_switches) por sesión y NPC —
        # anti-thrashing. Env NPC_ARBITRATION_MAX_SWITCHES.
        self.arbitration_max_switches: int = int(data.get("arbitration_max_switches", 4))
        # Mínimo de segundos entre dos consultas de arbitraje del mismo NPC (evita
        # consultar en cada tick del bucle mientras el goal sigue bloqueado). Env
        # NPC_ARBITRATION_COOLDOWN.
        self.arbitration_cooldown_s: float = float(data.get("arbitration_cooldown_s", 5.0))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        if cls._instance is not None:
            return cls._instance
        if path is None:
            path = Path(__file__).parent / "settings.json"
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # settings.local.json (NO versionado, mismo directorio): sobreescribe
        # claves del settings.json versionado. Sitio para secretos y overrides
        # locales (p.ej. gemini_api_key) sin riesgo de subirlos a GitHub.
        local_path = path.with_name("settings.local.json")
        if local_path.exists():
            with open(local_path, encoding="utf-8") as f:
                local = json.load(f)
            if isinstance(local, dict):
                data.update(local)

        # Permite override temporal del modelo desde start_server.cmd sin tocar settings.json.
        env_model = os.getenv("NPC_LLM_MODEL_OVERRIDE", "").strip()
        if env_model:
            data["llm_model"] = env_model

        # Overrides por variable de entorno para claves sensibles.
        env_gemini = os.getenv("NPC_GEMINI_API_KEY", "").strip()
        if env_gemini:
            data["gemini_api_key"] = env_gemini

        # Forzar apagado por inactividad desde el entorno (pruebas/eval).
        env_idle = os.getenv("NPC_SHUTDOWN_WHEN_IDLE", "").strip().lower()
        if env_idle in {"1", "true", "yes"}:
            data["shutdown_when_idle"] = True

        # Overrides de plan memory por entorno (pruebas/eval/CLI).
        env_pm_enabled = os.getenv("NPC_PLAN_MEMORY_ENABLED", "").strip().lower()
        if env_pm_enabled in {"0", "false", "no"}:
            data["plan_memory_enabled"] = False
        elif env_pm_enabled in {"1", "true", "yes"}:
            data["plan_memory_enabled"] = True
        env_pm_reuse = os.getenv("NPC_PLAN_MEMORY_REUSE", "").strip().lower()
        if env_pm_reuse in {"0", "false", "no"}:
            data["plan_memory_reuse"] = False
        elif env_pm_reuse in {"1", "true", "yes"}:
            data["plan_memory_reuse"] = True
        env_pm_dir = os.getenv("NPC_PLAN_MEMORY_DIR", "").strip()
        if env_pm_dir:
            data["plan_memory_dir"] = env_pm_dir
            data["plan_memory_reuse"] = True  # pasar un dir implica reuse

        # Toggle de refinamiento desde el entorno (matriz de experimentos, Fase 5).
        env_refine = os.getenv("NPC_USE_REFINEMENT", "").strip().lower()
        if env_refine in {"0", "false", "no"}:
            data["use_refinement"] = False
        elif env_refine in {"1", "true", "yes"}:
            data["use_refinement"] = True

        # Toggle del modo thinking desde el entorno (pruebas/ablación de velocidad).
        env_think = os.getenv("NPC_LLM_THINK", "").strip().lower()
        if env_think in {"0", "false", "no", "off"}:
            data["llm_think"] = False
        elif env_think in {"1", "true", "yes", "on"}:
            data["llm_think"] = True
        elif env_think in {"none", "null", "auto"}:
            data["llm_think"] = None

        # Toggle de reuso canónico desde el entorno (Fase 6.5).
        env_canon = os.getenv("NPC_CANONICAL_REUSE", "").strip().lower()
        if env_canon in {"0", "false", "no"}:
            data["canonical_reuse_enabled"] = False
        elif env_canon in {"1", "true", "yes"}:
            data["canonical_reuse_enabled"] = True

        # Toggle de la familia paramétrica determinista (paradigma generativo).
        env_family = os.getenv("NPC_CANONICAL_FAMILY", "").strip().lower()
        if env_family in {"0", "false", "no"}:
            data["canonical_family_plan"] = False
        elif env_family in {"1", "true", "yes"}:
            data["canonical_family_plan"] = True

        # Toggle de coordinación NPC↔NPC (Fase 12).
        env_coord = os.getenv("NPC_COORDINATION", "").strip().lower()
        if env_coord in {"0", "false", "no"}:
            data["coordination_enabled"] = False
        elif env_coord in {"1", "true", "yes"}:
            data["coordination_enabled"] = True
        env_planner = os.getenv("NPC_COORDINATION_PLANNER", "").strip().lower()
        if env_planner in {"family", "llm"}:
            data["coordination_planner"] = env_planner

        # Etiquetado de sesión (Fase 13 — batería experimental).
        env_exp_id = os.getenv("NPC_EXPERIMENT_ID", "").strip()
        if env_exp_id:
            data["experiment_id"] = env_exp_id
        env_cfg_label = os.getenv("NPC_CONFIG_LABEL", "").strip()
        if env_cfg_label:
            data["config_label"] = env_cfg_label
        env_n_opt = os.getenv("NPC_EXPERIMENT_N_OPT", "").strip()
        if env_n_opt:
            try:
                data["experiment_n_opt"] = float(env_n_opt)
            except ValueError:
                pass
        env_exp_npcs = os.getenv("NPC_EXPERIMENT_NPCS", "").strip()
        if env_exp_npcs:
            try:
                data["experiment_npcs"] = int(env_exp_npcs)
            except ValueError:
                pass
        env_triggers = os.getenv("NPC_BUILTIN_TRIGGERS", "").strip().lower()
        if env_triggers in {"0", "false", "no"}:
            data["builtin_triggers_enabled"] = False
        elif env_triggers in {"1", "true", "yes"}:
            data["builtin_triggers_enabled"] = True

        # Fase 16: ablación de sub-planes, aislamiento de contratos y duración máxima.
        env_subplans = os.getenv("NPC_BUILTIN_SUBPLANS", "").strip().lower()
        if env_subplans in {"0", "false", "no"}:
            data["builtin_subplans_enabled"] = False
        elif env_subplans in {"1", "true", "yes"}:
            data["builtin_subplans_enabled"] = True
        env_isolate = os.getenv("NPC_ISOLATE_CONTRACTS", "").strip().lower()
        if env_isolate in {"0", "false", "no"}:
            data["isolate_capability_contracts"] = False
        elif env_isolate in {"1", "true", "yes"}:
            data["isolate_capability_contracts"] = True
        env_max_s = os.getenv("NPC_SESSION_MAX_S", "").strip()
        if env_max_s:
            try:
                data["session_max_s"] = float(env_max_s)
            except ValueError:
                pass

        # Toggle de arbitraje de goals (Fase 14).
        env_arbitration = os.getenv("NPC_GOAL_ARBITRATION", "").strip().lower()
        if env_arbitration in {"0", "false", "no"}:
            data["goal_arbitration_enabled"] = False
        elif env_arbitration in {"1", "true", "yes"}:
            data["goal_arbitration_enabled"] = True
        env_arb_mode = os.getenv("NPC_ARBITRATION_MODE", "").strip().lower()
        if env_arb_mode in {"llm", "rule"}:
            data["goal_arbitration_mode"] = env_arb_mode
        env_arb_max = os.getenv("NPC_ARBITRATION_MAX_SWITCHES", "").strip()
        if env_arb_max:
            try:
                data["arbitration_max_switches"] = int(env_arb_max)
            except ValueError:
                pass
        env_arb_cooldown = os.getenv("NPC_ARBITRATION_COOLDOWN", "").strip()
        if env_arb_cooldown:
            try:
                data["arbitration_cooldown_s"] = float(env_arb_cooldown)
            except ValueError:
                pass

        cls._instance = cls(data)
        return cls._instance


settings = Settings.load()
