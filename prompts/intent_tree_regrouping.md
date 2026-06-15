# Intent-Tree Regrouping Protocol (v1)

## Task Overview

You receive ONE already-extracted intent structure: a JSON object with a `trees[]` array,
where each tree has a `tree_id`, an `intent_type`, and a flat list of `nodes[]`. Each node has
`id`, `label`, `discovery_state`, `turn_of_discovery`, `parent_id` (or `null` for a tree root),
and `evidence`.

Because extraction proceeds turn-by-turn, the raw structure is often redundant or flat: the
same intent appears twice under different labels, sibling nodes are really one parent and its
specialization, a flat list should be nested, or two independent dimensions are tangled into
one tree. Your job is to **regroup** the structure into a clean intent hierarchy — the same
role DiscoverLLM's stage-3 hierarchy-organization step plays — **without inventing new intents
or dropping any real one**.

This step is always run as a post-process on extraction output. Regrouping changes STRUCTURE,
not the set of real intents.

## Input

The extraction JSON object is provided as the user message.

## Core Principles

1. **Parent–child = abstract → specific.** A is the parent of B if B is a specific instance,
   constraint, or elaboration of A, and satisfying B contributes to satisfying A.
2. **Merge duplicates and near-duplicates.** If two nodes express the same intent, merge them
   into one node:
   - keep the clearest `label`;
   - join distinct `evidence` with `" | "`;
   - keep the EARLIEST non-null `turn_of_discovery`;
   - keep the most-advanced `discovery_state` (prefer `discovered` over `emerging`; keep
     `discovered-via-rejection` only if no positive discovery exists for that intent).
   This is the one place you may combine nodes — semantically equivalent nodes SHOULD merge.
3. **Multiple children and multiple roots are allowed.** Keep genuinely independent dimensions
   as SEPARATE trees (e.g. a content/planning tree vs. a meta/format tree). Never force
   unrelated intents under one root.
4. **Preserve every real intent.** Do not invent intents unsupported by evidence, and do not
   drop any intent that has evidence.
5. **No loops (DAG).** No node is its own ancestor; every `parent_id` points to an existing
   node within the SAME tree, or is `null` for that tree's single root.
6. **Keep enums valid.** `discovery_state` ∈ {`discovered`, `emerging`, `discovered-via-rejection`}.
   Keep each tree's `intent_type` unless a split/merge of trees clearly requires changing it.

## Step-by-Step Process

1. List every node across all trees with its label and evidence.
2. Cluster nodes that mean the same thing; pick one representative per cluster; record merges.
3. For each cluster, choose its parent (the most abstract node it specializes) or `null` if it
   is a root-level dimension.
4. Split tangled trees: if one tree mixes independent dimensions, separate them into distinct
   trees with appropriate `intent_type`.
5. Re-assign ids so each node id is unique across the object and every `parent_id` resolves;
   exactly one root (`parent_id: null`) per tree.
6. Verify: DAG, no duplicate labels, every input intent represented, all enums valid.

## Output Format

Return ONLY a JSON object with this shape:

```json
{
  "trees": [
    {
      "tree_id": "string",
      "intent_type": "string",
      "nodes": [
        {
          "id": "string",
          "label": "string",
          "discovery_state": "discovered | emerging | discovered-via-rejection",
          "turn_of_discovery": 1,
          "parent_id": "string-or-null",
          "evidence": "string"
        }
      ]
    }
  ],
  "regrouping_log": [
    {
      "action": "merge | move | split | keep",
      "nodes": ["id", "..."],
      "result_id": "id",
      "justification": "string"
    }
  ],
  "regrouping_notes": "1-3 sentences on the main structural changes."
}
```

Carry over `surfacing_log`, `decision_log`, `difficulty_notes`, and `validation_check` from the
input unchanged if they are present. Return only the JSON object.
