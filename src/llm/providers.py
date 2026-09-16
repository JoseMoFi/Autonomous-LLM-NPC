from __future__ import annotations

"""providers.py — Adaptador de spade-llm a la interfaz `complete(prompt, system)`.

El resto del sistema (planning_agent, pipeline) solo conoce la interfaz
`complete(prompt, system) -> str`. Este módulo es el ÚNICO punto que habla con
`spade_llm` / LiteLLM, de modo que migrar de proveedor no toca el pipeline.

Sustituye al `_OllamaProvider` casero de `main.py`.
"""

import asyncio
import os
from typing import Any, TypeVar

from pydantic import BaseModel

from spade_llm.providers import LLMProvider
from spade_llm.context.context_manager import ContextManager
from spade_llm.context._types import create_user_message

_TModel = TypeVar("_TModel", bound=BaseModel)

# Una conversación fija por llamada (cada complete() crea un ContextManager nuevo;
# es stateless, igual que el provider casero anterior).
_CONVERSATION_ID = "planning"


class SpadeLLMProviderAdapter:
    """Envuelve `spade_llm.providers.LLMProvider` y expone `complete(prompt, system)`.

    No silencia errores: las excepciones del provider/LiteLLM se propagan para que
    `PlanningRequestBehaviour` las trate. El timeout se envuelve con
    `asyncio.wait_for` para preservar `asyncio.TimeoutError` (→ `llm_timeout`).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        temperature: float = 0.2,
        timeout: float = 300.0,
        api_key: str | None = None,
        max_tokens: int | None = None,
        num_retries: int = 0,
        extra_request_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.timeout = float(timeout)
        # extra_request_kwargs se reenvía tal cual a LiteLLM (LLMProvider los
        # mezcla en cada acompletion). Lo usamos para `think` de Ollama.
        self._provider = LLMProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=float(timeout),
            max_tokens=max_tokens,
            num_retries=num_retries,
            **(extra_request_kwargs or {}),
        )

    async def complete(self, prompt: str, system: str = "") -> str:
        """Envía system+user al modelo y devuelve el texto de la respuesta."""
        context = ContextManager(system_prompt=system or None)
        context.set_current_conversation(_CONVERSATION_ID)
        context.add_message_dict(create_user_message(prompt), _CONVERSATION_ID)

        # asyncio.wait_for → asyncio.TimeoutError (la ruta que el agente traduce a
        # 'llm_timeout'); el timeout de LiteLLM queda como backstop.
        text = await asyncio.wait_for(
            self._provider.get_response(context),
            timeout=self.timeout,
        )
        # get_response devuelve None solo si hubiera tool calls (no usamos tools).
        return text or ""

    async def complete_structured(
        self, prompt: str, system: str = "", *, schema: type[_TModel]
    ) -> _TModel:
        """Como complete() pero fuerza la salida al esquema Pydantic `schema`
        (structured output de spade-llm) y devuelve la instancia tipada.

        Garantiza la FORMA (campos/tipos). La validación semántica se hace aparte.
        """
        context = ContextManager(system_prompt=system or None)
        context.set_current_conversation(_CONVERSATION_ID)
        context.add_message_dict(create_user_message(prompt), _CONVERSATION_ID)

        resp = await asyncio.wait_for(
            self._provider.get_llm_response(context, output_schema=schema),
            timeout=self.timeout,
        )
        structured = resp.get("structured")
        if not isinstance(structured, schema):
            raise ValueError(
                f"structured output inválido: se esperaba {schema.__name__}, "
                f"se recibió {type(structured).__name__}"
            )
        return structured


def build_provider(settings: Any) -> SpadeLLMProviderAdapter:
    """Construye el adaptador desde `settings` mapeando `llm_provider` al formato
    de modelo de LiteLLM. Soporta ollama (requerido), gemini y openai; el resto
    se pasa tal cual (el modelo debe traer el prefijo de LiteLLM).
    """
    provider = (getattr(settings, "llm_provider", "ollama") or "ollama").lower()
    model = settings.llm_model
    base_url = settings.llm_base_url
    temperature = settings.llm_temperature
    timeout = settings.llm_timeout

    if provider == "ollama":
        # ollama_chat/ usa /api/chat (mismo endpoint que el provider casero).
        # `think` controla el modo razonamiento de qwen3 y similares. Se envía
        # solo si no es None (None = modelo sin thinking → no mandar el parámetro,
        # Ollama da error si se manda a un modelo que no lo soporta).
        think = getattr(settings, "llm_think", False)
        extra = {} if think is None else {"think": bool(think)}
        return SpadeLLMProviderAdapter(
            model=f"ollama_chat/{model}",
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            extra_request_kwargs=extra,
        )

    if provider == "gemini":
        return SpadeLLMProviderAdapter(
            model=f"gemini/{model}",
            api_key=getattr(settings, "gemini_api_key", "") or None,
            temperature=temperature,
            timeout=timeout,
        )

    if provider == "openai":
        return SpadeLLMProviderAdapter(
            model=model if "/" in model else f"openai/{model}",
            api_key=os.getenv("OPENAI_API_KEY") or None,
            temperature=temperature,
            timeout=timeout,
        )

    # Genérico: el modelo debe incluir el prefijo de LiteLLM (p.ej. "anthropic/...").
    return SpadeLLMProviderAdapter(
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
    )
