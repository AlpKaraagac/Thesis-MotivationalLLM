"""Provider-specific adapter implementations live here."""

from discoverllm.llm.adapters.http import ProviderAPIError
from discoverllm.llm.adapters.openai import (
    DEFAULT_OPENAI_BASE_URL,
    OpenAIResponsesAdapter,
    extract_openai_text,
)

__all__ = [
    "DEFAULT_OPENAI_BASE_URL",
    "OpenAIResponsesAdapter",
    "ProviderAPIError",
    "extract_openai_text",
]
