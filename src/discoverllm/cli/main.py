"""Argparse CLI for tree building, simulation, and dataset generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from discoverllm.config import DEFAULT_CONFIG
from discoverllm.generation import (
    deserialize_artifact_spec,
    deserialize_intent_tree_initialization_package,
    llm_transcript_rows_from_build,
    llm_transcript_rows_from_simulation,
    to_jsonable,
    turn_comparison_rows_from_simulation,
    write_build_outputs,
    write_dataset_outputs,
    write_simulation_outputs,
)
from discoverllm.runtime_factories import make_openai_builder, make_openai_simulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="discoverllm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_tree = subparsers.add_parser("build-tree")
    build_tree.add_argument("--artifact-file", required=True, type=Path)
    build_tree.add_argument("--output-dir", required=True, type=Path)
    build_tree.add_argument("--seed", type=int, default=DEFAULT_CONFIG.default_seed)
    build_tree.add_argument("--latent-root-id", action="append", default=[])

    simulate = subparsers.add_parser("simulate-conversation")
    simulate.add_argument("--simulator-init-file", required=True, type=Path)
    simulate.add_argument("--output-dir", required=True, type=Path)

    generate = subparsers.add_parser("generate-dataset")
    generate.add_argument("--artifacts-file", required=True, type=Path)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--seed-base", type=int, default=DEFAULT_CONFIG.default_seed)
    generate.add_argument("--seed-step", type=int, default=1)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "build-tree":
        run_build_tree(
            artifact_file=args.artifact_file,
            output_dir=args.output_dir,
            builder=make_openai_builder(),
            seed=args.seed,
            latent_root_ids=args.latent_root_id,
        )
        return 0
    if args.command == "simulate-conversation":
        run_simulate_conversation(
            simulator_init_file=args.simulator_init_file,
            output_dir=args.output_dir,
            simulator=make_openai_simulator(),
        )
        return 0
    if args.command == "generate-dataset":
        run_generate_dataset(
            artifacts_file=args.artifacts_file,
            output_dir=args.output_dir,
            builder=make_openai_builder(),
            simulator=make_openai_simulator(),
            seed_base=args.seed_base,
            seed_step=args.seed_step,
        )
        return 0
    raise RuntimeError(f"Unhandled command {args.command!r}.")


def run_build_tree(
    *,
    artifact_file: Path,
    output_dir: Path,
    builder: Any,
    seed: int,
    latent_root_ids: Sequence[str] = (),
):
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    _require_builder(builder)
    artifact = _load_single_artifact(artifact_file)
    package = builder.build_from_artifact(
        artifact,
        seed=seed,
        latent_root_ids=list(latent_root_ids) or None,
    )
    return write_build_outputs(package, output_dir)


def run_simulate_conversation(
    *,
    simulator_init_file: Path,
    output_dir: Path,
    simulator: Any,
):
    _require_simulator(simulator)
    payload = _load_json_file(simulator_init_file)
    package = deserialize_intent_tree_initialization_package(payload)
    return write_simulation_outputs(
        simulator.simulate_conversation(package.conversation_state),
        output_dir,
    )


def run_generate_dataset(
    *,
    artifacts_file: Path,
    output_dir: Path,
    builder: Any,
    simulator: Any,
    seed_base: int = DEFAULT_CONFIG.default_seed,
    seed_step: int = 1,
):
    if not isinstance(seed_base, int) or seed_base < 0:
        raise ValueError("seed_base must be a non-negative integer.")
    if not isinstance(seed_step, int) or seed_step < 1:
        raise ValueError("seed_step must be a positive integer.")
    _require_builder(builder)
    _require_simulator(simulator)

    artifacts = _load_artifact_specs(artifacts_file)
    artifact_output_dirs: list[Path] = []
    trajectories: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    turn_comparisons: list[dict[str, Any]] = []
    raw_llm_transcripts: list[dict[str, Any]] = []

    for index, artifact in enumerate(artifacts):
        seed = seed_base + (index * seed_step)
        artifact_dir = output_dir / "artifacts" / artifact.artifact_id
        artifact_output_dirs.append(artifact_dir)

        package = builder.build_from_artifact(artifact, seed=seed)
        write_build_outputs(package, artifact_dir / "build")

        simulation_result = simulator.simulate_conversation(package.conversation_state)
        write_simulation_outputs(simulation_result, artifact_dir / "simulation")

        trajectories.append(to_jsonable(simulation_result.trajectory))
        preferences.extend(to_jsonable(item) for item in simulation_result.preferences)
        turn_comparisons.extend(turn_comparison_rows_from_simulation(simulation_result))
        raw_llm_transcripts.extend(llm_transcript_rows_from_build(package))
        raw_llm_transcripts.extend(llm_transcript_rows_from_simulation(simulation_result))

    return write_dataset_outputs(
        output_dir=output_dir,
        artifact_output_dirs=artifact_output_dirs,
        trajectories=trajectories,
        preferences=preferences,
        turn_comparisons=turn_comparisons,
        raw_llm_transcripts=raw_llm_transcripts,
    )


def _load_single_artifact(path: Path):
    artifacts = _load_artifact_specs(path)
    if len(artifacts) != 1:
        raise ValueError("artifact file must contain exactly one artifact.")
    return artifacts[0]


def _load_artifact_specs(path: Path):
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix == ".jsonl":
        artifacts = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                artifacts.append(deserialize_artifact_spec(json.loads(stripped)))
        if not artifacts:
            raise ValueError("artifacts_file JSONL must contain at least one artifact.")
        return artifacts

    payload = _load_json_file(path)
    if isinstance(payload, list):
        if not payload:
            raise ValueError("artifacts_file JSON array must contain at least one artifact.")
        return [deserialize_artifact_spec(item) for item in payload]
    return [deserialize_artifact_spec(payload)]


def _load_json_file(path: Path) -> Any:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_builder(builder: Any) -> None:
    if not callable(getattr(builder, "build_from_artifact", None)):
        raise TypeError("builder must define build_from_artifact().")


def _require_simulator(simulator: Any) -> None:
    if not callable(getattr(simulator, "simulate_conversation", None)):
        raise TypeError("simulator must define simulate_conversation().")
