"""Simple constructors for the OpenAI-backed runtime."""

from __future__ import annotations

import os

from discoverllm.config import DEFAULT_CONFIG
from discoverllm.generation import AssistantCandidateGenerator, ConversationSimulator
from discoverllm.intent_tree import IntentTreeBuilder
from discoverllm.llm.adapters import DEFAULT_OPENAI_BASE_URL, OpenAIResponsesAdapter
from discoverllm.prompts import PromptLoader
from discoverllm.simulator import AssistantResponseEvaluator, UserResponseGenerator


def make_openai_builder() -> IntentTreeBuilder:
    """Build the paper-faithful tree builder with one OpenAI model."""

    api_key, base_url, timeout_seconds = _openai_settings()
    return IntentTreeBuilder(
        llm_adapter=OpenAIResponsesAdapter(
            api_key=api_key,
            model=_env("OPENAI_TREE_BUILDER_MODEL", "gpt-5.4-mini"),
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        ),
        prompt_loader=PromptLoader(),
        num_abstraction_levels=_env_int("DISCOVERLLM_NUM_ABSTRACTION_LEVELS", 3),
        stable_root_sort=True,
    )


def make_openai_simulator() -> ConversationSimulator:
    """Build the full two-candidate simulator with OpenAI everywhere."""

    prompt_loader = PromptLoader()
    api_key, base_url, timeout_seconds = _openai_settings()
    return ConversationSimulator(
        assistant_generators=(
            AssistantCandidateGenerator(
                candidate_name="assistant_a",
                llm_adapter=OpenAIResponsesAdapter(
                    api_key=api_key,
                    model=_env("OPENAI_ASSISTANT_A_MODEL", "gpt-5.4-mini"),
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                ),
                prompt_loader=prompt_loader,
            ),
            AssistantCandidateGenerator(
                candidate_name="assistant_b",
                llm_adapter=OpenAIResponsesAdapter(
                    api_key=api_key,
                    model=_env("OPENAI_ASSISTANT_B_MODEL", "gpt-4.1"),
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                ),
                prompt_loader=prompt_loader,
                # This is the paper's "synthesis assistant" branch.
                system_prompt_name="synthesis_assistant",
            ),
        ),
        evaluator=AssistantResponseEvaluator(
            llm_adapter=OpenAIResponsesAdapter(
                api_key=api_key,
                model=_env("OPENAI_EVALUATOR_MODEL", "gpt-5.4-mini"),
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            ),
            prompt_loader=prompt_loader,
        ),
        user_generator=UserResponseGenerator(
            llm_adapter=OpenAIResponsesAdapter(
                api_key=api_key,
                model=_env("OPENAI_USER_MODEL", "gpt-5.4-mini"),
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            ),
            prompt_loader=prompt_loader,
        ),
        turn_limit=_env_int("DISCOVERLLM_TURN_LIMIT", DEFAULT_CONFIG.default_turn_limit),
    )


def _openai_settings() -> tuple[str, str, float]:
    return (
        _required_env("OPENAI_API_KEY"),
        _env("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        _env_float("OPENAI_TIMEOUT_SECONDS", 120.0),
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(
            f"Required environment variable {name} is not set. "
            f"Export it before running the CLI."
        )
    return value.strip()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


__all__ = [
    "make_openai_builder",
    "make_openai_simulator",
]
