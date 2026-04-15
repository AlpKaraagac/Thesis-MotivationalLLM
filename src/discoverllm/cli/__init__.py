"""CLI entry points for tree building and data synthesis."""

from discoverllm.cli.main import (
    build_parser,
    main,
    run_build_tree,
    run_generate_dataset,
    run_simulate_conversation,
)

__all__ = [
    "build_parser",
    "main",
    "run_build_tree",
    "run_generate_dataset",
    "run_simulate_conversation",
]
