# Thesis-MotivationalLLM

This repository supports a thesis project on psychologically aware multi-turn dialogue with large
language models. The research goal is to test whether explicitly modeling a user's latent
psychological state together with their evolving intent can improve the quality, efficiency, and
personalization of LLM conversations.

## What this repo currently contains

Exploratory analysis (on the `initial-analysis` branch) built on two **published** resources. This
repo no longer vendors a local DiscoverLLM reproduction (see *Archived reproduction* below); it
consumes the official DiscoverLLM dataset directly.

- **ThoughtTrace** — real multi-turn conversations with per-turn reason/reaction annotations.
  HF dataset: [`SCAI-JHU/ThoughtTrace`](https://huggingface.co/datasets/SCAI-JHU/ThoughtTrace).
- **DiscoverLLM (ICML 2026)** — official dataset
  [`kixlab/DiscoverLLM-multiturn-preferences`](https://huggingface.co/datasets/kixlab/DiscoverLLM-multiturn-preferences)
  and fine-tuned models. Collection:
  <https://huggingface.co/collections/kixlab/discoverllm-icml-2026> · paper **arXiv:2602.03429**.

## Thesis direction

1. Keep DiscoverLLM-style latent **intent** tracking as the baseline.
2. Add an explicit **psychological-state** representation (e.g. intent clarity, frustration,
   readiness to explore vs. execute).
3. Maintain a **joint belief** over intent and psychological state.
4. Condition assistant strategy selection on that joint belief, and compare against the baseline.

## Notebooks

See [`notebooks/`](notebooks/) (and its README):

- `01_thoughttrace_eda.ipynb` — ThoughtTrace corpus EDA.
- `02_discoverllm_eda.ipynb` — official DiscoverLLM multi-turn-preferences EDA (intent-criteria
  hierarchies, preference scores, per-domain breakdowns).
- `03_morpheus_smoke_test.ipynb` — TUM **Morpheus** (Qwen, OpenAI-compatible) extraction +
  regrouping smoke test against a frozen Claude Sonnet extraction.

## Setup

```bash
pip install -r requirements.txt
```

Create a git-ignored `.env` at the repo root for the Morpheus endpoint (used by notebook 03):

```
MORPHEUS_KEY=...                # required
MORPHEUS_MODEL=Qwen/Qwen3.6-35B-A3B   # optional
MORPHEUS_BASE_URL=https://morpheus.cit.tum.de/api/   # optional
HF_TOKEN=...                    # optional (silences the HF rate-limit warning)
```

## Archived reproduction

An earlier local DiscoverLLM-style reproduction previously lived under `src/discoverllm/` (intent-tree
builder, simulator, reward, dataset generation). It has been **removed from the active branch** in
favor of the official DiscoverLLM dataset, and is preserved on the **`archive/discoverllm-reproduction`**
branch (tag `discoverllm-reproduction-archive`) for reference.
