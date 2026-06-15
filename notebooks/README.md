# Notebooks — initial dataset analysis

Exploratory, **read-only** notebooks for the `initial-analysis` branch. Install
`../requirements.txt`; they resolve paths to the repo root, so they run from any working directory.

- **01_thoughttrace_eda.ipynb** — ThoughtTrace EDA (HF `SCAI-JHU/ThoughtTrace`): users/conversations/annotations, per-user and length distributions, reaction (5) and reason (7) category breakdowns, source-LLM split, and supervision-sparsity flags.
- **02_discoverllm_eda.ipynb** — official DiscoverLLM EDA (HF `kixlab/DiscoverLLM-multiturn-preferences`, 3 domain configs): record structure, per-domain counts, preference-score distribution, candidates/turns, and the intent-criteria **hierarchy** size/depth + example trees parsed from `criteria_history`.
- **03_morpheus_smoke_test.ipynb** — Morpheus (Qwen) connectivity + extraction + an always-on **regrouping** step, compared against a frozen Claude Sonnet extraction. Frozen prompts: extraction `f1924f822892`, regrouping `33ed398829ff`.

## Inputs

| Notebook | What | Status |
|---|---|---|
| 01 | ThoughtTrace | loads from HF (no local input); optional `HF_TOKEN` |
| 02 | DiscoverLLM multi-turn preferences | loads from HF (no local input); optional `HF_TOKEN` |
| 03 | extraction prompt | **bundled** `prompts/intent_tree_extraction.md` (`f1924f822892`) |
| 03 | regrouping prompt | **bundled** `prompts/intent_tree_regrouping.md` (`33ed398829ff`) |
| 03 | frozen Sonnet extraction | **bundled** `data/sonnet_extractions/user1016_task1_conversation1.json` |
| 03 | env | `MORPHEUS_KEY` (required, via a git-ignored `.env`); `TARGET_CONV_ID` defaults to `user1016_task1_conversation1` |

Credentials load from a git-ignored `.env` at the repo root — nothing is hardcoded.

## Regrouping step

NB3 always runs a **regrouping** pass after extraction — a local analog of DiscoverLLM's
hierarchy-organization stage (see the DiscoverLLM paper, **arXiv:2602.03429**): it merges
duplicate/near-duplicate intents, fixes abstract→specific parent links, keeps independent
dimensions as separate trees, and enforces a DAG. The protocol lives in
[`prompts/intent_tree_regrouping.md`](../prompts/intent_tree_regrouping.md), hash-pinned `33ed398829ff`.
