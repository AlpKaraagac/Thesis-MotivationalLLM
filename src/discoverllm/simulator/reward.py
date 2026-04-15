"""Per-turn reward computation for simulator-guided generation."""

from __future__ import annotations

from typing import Callable

from discoverllm.config import DEFAULT_CONFIG
from discoverllm.simulator.state_updater import StateUpdateResult
from discoverllm.types import TurnReward

TokenCounter = Callable[[str], int]


def count_whitespace_tokens(text: str) -> int:
    """Deterministic fallback token count until provider tokenizers are wired in."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def compute_efficiency_penalty(
    response_tokens: int,
    *,
    tau: int = DEFAULT_CONFIG.tau_sft_dpo,
    length_penalty_lambda: float = DEFAULT_CONFIG.length_penalty_lambda,
) -> float:
    """Compute the capped length penalty from the paper."""

    if not isinstance(response_tokens, int):
        raise TypeError("response_tokens must be an int.")
    if response_tokens < 0:
        raise ValueError("response_tokens must be non-negative.")
    if not isinstance(tau, int):
        raise TypeError("tau must be an int.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if not isinstance(length_penalty_lambda, (int, float)):
        raise TypeError("length_penalty_lambda must be numeric.")
    length_penalty_lambda = float(length_penalty_lambda)
    if length_penalty_lambda <= 0:
        raise ValueError("length_penalty_lambda must be positive.")

    overflow = max(0, response_tokens - tau)
    return -min(length_penalty_lambda * overflow, 1.0)


def compute_turn_reward(
    *,
    discovery_delta: int,
    assistant_message: str | None = None,
    response_tokens: int | None = None,
    tau: int = DEFAULT_CONFIG.tau_sft_dpo,
    length_penalty_lambda: float = DEFAULT_CONFIG.length_penalty_lambda,
    token_counter: TokenCounter | None = None,
) -> TurnReward:
    """Compute discovery reward plus capped efficiency penalty."""

    if not isinstance(discovery_delta, int):
        raise TypeError("discovery_delta must be an int.")
    if discovery_delta < 0:
        raise ValueError("discovery_delta must be non-negative.")

    if response_tokens is None:
        if assistant_message is None:
            raise ValueError("assistant_message is required when response_tokens is not provided.")
        if token_counter is None:
            token_counter = count_whitespace_tokens
        response_tokens = token_counter(assistant_message)

    penalty = compute_efficiency_penalty(
        response_tokens,
        tau=tau,
        length_penalty_lambda=length_penalty_lambda,
    )
    return TurnReward(
        discovery_delta=discovery_delta,
        efficiency_penalty=penalty,
        response_tokens=response_tokens,
    )


def compute_turn_reward_from_state_update(
    state_update: StateUpdateResult,
    *,
    assistant_message: str | None = None,
    response_tokens: int | None = None,
    tau: int = DEFAULT_CONFIG.tau_sft_dpo,
    length_penalty_lambda: float = DEFAULT_CONFIG.length_penalty_lambda,
    token_counter: TokenCounter | None = None,
) -> TurnReward:
    """Convenience wrapper that derives discovery delta from a state update."""

    if not isinstance(state_update, StateUpdateResult):
        raise TypeError("state_update must be a StateUpdateResult.")
    return compute_turn_reward(
        discovery_delta=state_update.discovery_delta,
        assistant_message=assistant_message,
        response_tokens=response_tokens,
        tau=tau,
        length_penalty_lambda=length_penalty_lambda,
        token_counter=token_counter,
    )
