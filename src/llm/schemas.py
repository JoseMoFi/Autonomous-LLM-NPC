from __future__ import annotations

"""schemas.py — Modelos Pydantic de las respuestas del LLM por paso del pipeline.

Fase 3: structured outputs vía spade-llm. Estos modelos definen la FORMA de cada
respuesta (campos, tipos) y se pasan como `output_schema` a spade-llm para forzar
JSON válido por esquema. La validación SEMÁNTICA (catálogo de acciones, recipes,
zonas, repair) NO vive aquí — se queda en validator.py / step3_validator.py /
step4_map.py (contribución del TFM).

Migración incremental: se añade un modelo por paso a medida que se migra.
"""

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# parse_goals  (prompt: llm/prompts/planning.py::_build_parse_goals)
# ---------------------------------------------------------------------------

class GoalItem(BaseModel):
    """Un goal derivado de un objetivo en lenguaje natural.

    `source_index` apunta al índice [i] del objetivo de origen (0.B3).
    `success_condition` es una condición ASL (validada semánticamente aparte).
    """
    model_config = ConfigDict(extra="forbid")

    sig: str
    source_index: int
    priority: float
    reason: str
    success_condition: str


class ParseGoalsResponse(BaseModel):
    """Respuesta de `parse_goals`: la lista de goals.

    Se envuelve en un objeto (`goals`) porque el structured output de spade-llm
    requiere un BaseModel de nivel superior (no un array suelto).
    """
    model_config = ConfigDict(extra="forbid")

    goals: list[GoalItem]


# ---------------------------------------------------------------------------
# step0_name  (prompt: _build_step0_name) — NL → sig + descripción
# ---------------------------------------------------------------------------

class Step0Response(BaseModel):
    """Respuesta de step0: nombre BDI + descripción detallada del goal.

    La validación del patrón del `sig` (achieve_verb_object) es semántica y se
    queda en `validate_goal_name` (no en el esquema).
    """
    model_config = ConfigDict(extra="forbid")

    sig: str
    description: str


# ---------------------------------------------------------------------------
# step2_problem  (prompt: _build_step2) — describe el problema + facts conocidos
# ---------------------------------------------------------------------------

class Step2Response(BaseModel):
    """Respuesta de step2: descripción NL del problema + facts inferidos.

    `known_facts` es opcional (el paso hace setdefault a []). La validación de
    longitud de `problem_nl` se queda en run_step2.
    """
    model_config = ConfigDict(extra="forbid")

    problem_nl: str
    known_facts: list[str] = []


# ---------------------------------------------------------------------------
# step1_success  (prompt: _build_step1) — success_model + done_guard
# ---------------------------------------------------------------------------

class SuccessVariant(BaseModel):
    """Una variante del success_model. `bound_variables` NO se pide al LLM (lo
    computa run_step1 desde facts). `done_fragment` puede omitirse: run_step1 lo
    deriva de facts+guards."""
    model_config = ConfigDict(extra="forbid")

    facts: list[str] = []
    guards: list[str] = []
    done_fragment: str = ""


class Step1Response(BaseModel):
    """Respuesta de step1. El LLM produce success_model (o success_conditions
    como alternativa) y done_guard. `done_asl` lo computa run_step1, no el LLM.
    La validación semántica (variantes no vacías) se queda en run_step1."""
    model_config = ConfigDict(extra="forbid")

    success_model: list[SuccessVariant] = []
    success_conditions: list[str] = []
    done_guard: str = ""


# ---------------------------------------------------------------------------
# step3_steps  (prompt: _build_step3) — genera los steps del plan
# ---------------------------------------------------------------------------

class StepItem(BaseModel):
    """Un step del plan: acción primitiva o sub-goal.

    `args` es heterogéneo (str/int/float: itemId, coords, qty…). `description`
    se conserva para sub-goals (la usa step5/classify). La validación SEMÁNTICA
    (catálogo, repair de args, normalización de tipos) se hace en step4_map.
    """
    model_config = ConfigDict(extra="ignore")

    type: str
    name: str
    args: list[Union[str, int, float]] = []
    description: str = ""


class Step3Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[StepItem]
