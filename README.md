# Thesis-MotivationalLLM

This repository supports a thesis project on psychologically aware multi-turn dialogue with large language models.

The main research goal is to test whether explicitly modeling a user's latent psychological state together with their evolving intent can improve the quality, efficiency, and personalization of LLM conversations.

The current codebase contains a DiscoverLLM-style baseline under the Python package name `discoverllm`. That baseline gives the project an intent-tree builder, a simulator, a reward function, and a chosen/rejected data-generation loop. It is the starting point, not the final thesis system.

## Thesis Goal

The project is built around this idea:

- Current conversational systems often fail because they choose the wrong interaction strategy, not because they lack knowledge.
- The same surface request may require different behavior depending on whether the user is frustrated, uncertain, exploratory, or ready to act.
- A strong dialogue system should maintain a belief not just over what the user wants, but also over how the user is in the interaction.

The intended end state of this project is a unified framework that:

- tracks latent intent
- tracks latent psychological state
- updates a joint belief state over both
- chooses better multi-turn strategies from that joint belief

## What The Repo Currently Contains

Right now the repository mostly implements the baseline side of the project:

- intent-tree construction from an artifact
- simulated multi-turn user interaction over that tree
- assistant candidate evaluation
- deterministic state updating for intent discovery
- reward computation and dataset writing

This baseline matters because it provides:

- a controlled environment for extension work
- a reproducible comparison point
- the intent-modeling half of the final thesis system

## What Is In The Repo

- `src/discoverllm/`: main implementation
- `src/discoverllm/intent_tree/`: four-stage intent tree construction
- `src/discoverllm/simulator/`: evaluator, state updater, reward, user generation
- `src/discoverllm/generation/`: turn loop and dataset writing
- `src/discoverllm/prompts/`: markdown prompt templates used by the pipeline
- `src/discoverllm/llm/`: OpenAI adapter and LLM request/response types
- `examples/`: small artifact inputs for local runs
- `tests/`: unit and integration-style tests

## Planned Thesis Extension

The next major step is to extend the baseline from intent-only user modeling to joint user modeling.

The intended research direction is:

1. Keep DiscoverLLM-style latent intent tracking as the baseline.
2. Add an explicit psychological-state representation.
3. Maintain a joint belief state over intent and psychology.
4. Condition assistant strategy selection on that joint belief.
5. Compare the joint-state system against the baseline.

Examples of psychological dimensions the project may eventually model include:

- intent clarity
- frustration or calmness
- readiness to explore versus readiness to execute

The exact state design may evolve, but the thesis goal is fixed: planning over both user intent and user psychological state.

## Current Runtime Assumptions

- The repo currently supports the OpenAI Responses API path.
- Prompt templates live under `src/discoverllm/prompts/`.
- The CLI is available through `python -m discoverllm.cli`.
- The repo is not packaged yet, so local commands should usually be run with `PYTHONPATH=src`.

## Quick Start

1. Create and activate a Python virtual environment.
2. Install the dependencies you need for the current environment.
3. Export your OpenAI API key.

```bash
export OPENAI_API_KEY=your_key_here
```

## CLI Commands

Show the top-level CLI:

```bash
PYTHONPATH=src python -m discoverllm.cli --help
```

Build an intent tree and simulator initialization package from one artifact:

```bash
PYTHONPATH=src python -m discoverllm.cli build-tree \
  --artifact-file examples/artifact_demo.json \
  --output-dir out/build
```

Run a simulated conversation from a previously generated initialization package:

```bash
PYTHONPATH=src python -m discoverllm.cli simulate-conversation \
  --simulator-init-file out/build/simulator_init.json \
  --output-dir out/simulation
```

Generate dataset outputs for multiple artifacts:

```bash
PYTHONPATH=src python -m discoverllm.cli generate-dataset \
  --artifacts-file examples/artifacts_demo.jsonl \
  --output-dir out/dataset
```

## Important Environment Variables

Required:

- `OPENAI_API_KEY`

Optional:

- `OPENAI_BASE_URL`
- `OPENAI_TIMEOUT_SECONDS`
- `OPENAI_TREE_BUILDER_MODEL`
- `OPENAI_ASSISTANT_A_MODEL`
- `OPENAI_ASSISTANT_B_MODEL`
- `OPENAI_EVALUATOR_MODEL`
- `OPENAI_USER_MODEL`
- `DISCOVERLLM_NUM_ABSTRACTION_LEVELS`
- `DISCOVERLLM_TURN_LIMIT`

## Expected Outputs

`build-tree` writes files such as:

- `intent_tree.json`
- `simulator_init.json`
- `builder_audit.json`
- `raw_llm_transcripts.jsonl`

`simulate-conversation` writes files such as:

- `chosen_trajectory.jsonl`
- `preferences.jsonl`
- `turn_comparisons.jsonl`
- `simulation_result.json`
- `raw_llm_transcripts.jsonl`

`generate-dataset` writes aggregate JSONL files plus per-artifact build and simulation folders under the chosen output directory.

## Development Notes

- `discoverllm` should be treated as the current baseline implementation.
- New work should be framed in terms of how it helps move from baseline intent modeling to joint intent-plus-psychological-state modeling.
- The project is easiest to work on by keeping deterministic logic in plain functions and leaving prompt text in the markdown files.
- Tests already cover the core parser, builder, evaluator, simulator, reward, and CLI behavior.
- `agents/` is intended for local implementation guidance and is ignored by git.

## Suggested Workflow

1. Update or add tests first when changing behavior.
2. Keep prompt edits isolated to `src/discoverllm/prompts/`.
3. Run `PYTHONPATH=src python -m unittest discover -s tests` before closing a change.
4. Use the example artifact files to smoke test the CLI flow.
5. When adding thesis extensions, keep the baseline path reproducible so the final comparison remains valid.
