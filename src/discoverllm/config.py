"""Small home for project-wide defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class DiscoverLLMConfig:
    """Defaults shared across the builder and simulator."""

    tangential_probability: float = 0.25
    length_penalty_lambda: float = 1e-3
    tau_sft_dpo: int = 250
    tau_grpo: int = 500
    default_turn_limit: int = 5
    default_seed: int = 0
    prompt_dir: Path = PROMPT_DIR


DEFAULT_CONFIG = DiscoverLLMConfig()
