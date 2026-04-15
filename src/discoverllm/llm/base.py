"""Base adapter protocol for LLM-backed modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from discoverllm.types import ConversationMessage


@dataclass(slots=True)
class LLMRequest:
    """Normalized request payload passed to an LLM adapter."""

    messages: Sequence[ConversationMessage]
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("messages must not be empty.")
        self.temperature = float(self.temperature)
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if self.max_tokens is not None:
            self.max_tokens = int(self.max_tokens)
            if self.max_tokens <= 0:
                raise ValueError("max_tokens must be positive when provided.")


@dataclass(slots=True)
class LLMResponse:
    """Normalized LLM output with optional provider metadata."""

    text: str
    raw_output: str | dict[str, Any]
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        self.text = str(self.text).strip()
        if not self.text:
            raise ValueError("text must be a non-empty string.")
        if self.prompt_tokens is not None:
            self.prompt_tokens = int(self.prompt_tokens)
            if self.prompt_tokens < 0:
                raise ValueError("prompt_tokens must be non-negative.")
        if self.completion_tokens is not None:
            self.completion_tokens = int(self.completion_tokens)
            if self.completion_tokens < 0:
                raise ValueError("completion_tokens must be non-negative.")


class LLMAdapter(ABC):
    """Abstract interface for model-specific adapters."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response from a structured request payload."""
        raise NotImplementedError
