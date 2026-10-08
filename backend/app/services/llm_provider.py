from __future__ import annotations

from typing import Protocol
from urllib.parse import urlparse

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError

from app.config import get_llm_api_key, get_llm_base_url, get_llm_model
from app.services.context_builder import GROUNDED_SYSTEM_PROMPT, build_user_prompt
from app.services.context_builder import BuiltContext

LLM_REQUEST_TIMEOUT_SECONDS = 45.0


class LLMProviderError(Exception):
    pass


class LLMConfigurationError(LLMProviderError):
    pass


class LLMTimeoutError(LLMProviderError):
    pass


class LLMServiceError(LLMProviderError):
    pass


class LLMProvider(Protocol):
    def generate_answer(
        self, context: BuiltContext, question: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate an answer grounded in the supplied context."""


class OpenAICompatibleProvider:
    def __init__(self) -> None:
        api_key = get_llm_api_key()
        if not api_key:
            raise LLMConfigurationError("LLM service is not configured")

        base_url = get_llm_base_url()
        if base_url:
            parsed_url = urlparse(base_url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise LLMConfigurationError("LLM service configuration is invalid")

        model = get_llm_model()
        if not model.strip():
            raise LLMConfigurationError("LLM service configuration is invalid")

        try:
            self._model = model
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception:
            raise LLMConfigurationError("LLM service configuration is invalid") from None

    def generate_answer(
        self, context: BuiltContext, question: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        try:
            messages = [{"role": "system", "content": GROUNDED_SYSTEM_PROMPT}]
            if history:
                messages.append({
                    "role": "system",
                    "content": (
                        "The following prior messages are conversational context only. "
                        "Prior assistant statements are not evidence and may be mistaken. "
                        "Use only the current retrieved context for factual claims."
                    ),
                })
                messages.extend(history)
            messages.append({"role": "user", "content": build_user_prompt(question, context)})
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            )
        except APITimeoutError:
            raise LLMTimeoutError("LLM request timed out") from None
        except (APIConnectionError, APIStatusError, OpenAIError):
            raise LLMServiceError("LLM provider request failed") from None
        except Exception:
            raise LLMServiceError("LLM provider request failed") from None

        answer = response.choices[0].message.content if response.choices else None
        if not isinstance(answer, str) or not answer.strip():
            raise LLMServiceError("LLM provider returned an empty response")
        return answer.strip()


def create_llm_provider() -> LLMProvider:
    return OpenAICompatibleProvider()
