# ThoughtTrace Intent Tree Extraction — Protocol v1.1

## Goal

Hand-extract intent trees from a stratified sample of ThoughtTrace conversations to (a) validate that DiscoverLLM-style intent hierarchies can be recovered from real-user dialogue, (b) operationalize the extraction rules before scaling to LLM-based extraction, and (c) check whether reaction labels can be meaningfully attached to specific nodes — the empirical foundation for the node-scoring contribution.

---

## Critical rules (read first)

These four rules are the most common failure modes in v1.0 extractions. Apply them with discipline:

1. **One node per distinct user-introduced piece.** If a user turn states multiple constraints in one message ("I'm going with 2 friends, for 1 week, budget $3000"), each piece is a separate child node. Do not lump them into one composite node.
2. **One surfacing-log entry per distinct assistant-introduced concept.** If the assistant's response surfaces 7 ranked factors or 12 candidate shows, log them as 7 or 12 entries — not 1. A concept is "distinct" if the user could engage with it independently of the others. Sanity check: if your surfacing log has fewer entries than the assistant has bullet points across the conversation, you are under-decomposing.
3. **Create a tone tree (intent_type `meta`) whenever the user articulates a format/style/length preference.** Triggers include: "be concise", "shorter please", "more detail", "in 1 sentence", "simpler answer", "format as a checklist", "use bullet points", "in plain language", "step by step", "summarize", "TL;DR", "in N words". Even one such utterance triggers the tone tree. Multiple utterances of the same preference do not create duplicate nodes — they strengthen the discovery state and add a turn-of-discovery for the strongest instance.
4. **Update the surfacing log retroactively, every turn.** At each new user turn, scan the existing surfacing log for entries currently marked `exposed-no-engagement`. If the user's current turn engages with any of those concepts, change that entry's outcome to `adopted-deferred` and set `outcome_updated_at_turn` to the current turn. If the assistant re-surfaces a previously-ignored concept with no engagement, update to `exposed-repeated-no-engagement`. This is required, not optional — long conversations without any retroactive updates are almost always under-analyzed.

---

## Core principles

1. **One tree per conversation by default.** Multi-tree decomposition only when the independence test passes — tone/format preferences are the reliable case where it does.
2. **Trees contain user-evidenced intents only.** Articulated, adopted, hypothetically explored, or revealed through rejection of a violating path. No annotator-imagined latent intents.
3. **Intents are extracted at the value/need level, not just the surface-topic level.** When a user says "I won't use my savings," the intent is `financial stability`, not the rejection itself.
4. **Two layers, kept structurally separate:**
   - **User intent tree** — what the user actually wants/values (evidence-based)
   - **Surfacing log** — what the assistant introduced and what happened to it
5. **Build incrementally, turn pair by turn pair**, with explicit decisions logged at each step.
6. **Discussion is the universal signal**; reasons and reactions are supplementary when present. Task expectations are validation-only, never construction input.

---

## What goes in the tree (user intent tree)

A node enters the tree when one of these is true for a given turn. The `discovery_state` field uses one of three values only — `discovered`, `emerging`, `discovered-via-rejection`. The categories below describe *how* the intent surfaced; they are not `discovery_state` values themselves.

- **Articulated** → `discovered`. The user explicitly states the intent in their message.
- **Adopted from assistant** → `discovered`. The assistant surfaced something and the user explicitly took it up (e.g., "I like the culinary road trip idea"). Do not use `adopted` as a `discovery_state` value — it is not in the enum.
- **Hypothetically explored** → `emerging`. The user explores an intent conditionally ("If I could handle volatility...").
- **Revealed through rejection** → `discovered-via-rejection`. The user rejects an assistant suggestion, and the rejection makes an underlying intent explicit ("I don't want to use my savings" → intent: financial stability).
- **Revealed in a reason** → `discovered`. A reason annotation attached to a user turn reveals an unarticulated intent (e.g., a time constraint that lived in the reason, not the message text).

A node does *not* enter the tree when:
- It was surfaced by the assistant but the user didn't engage with it (those go in the surfacing log only).
- The annotator imagines the user "could have" cared about it (no inferred latents).
- It was offered as one of several alternatives that the user didn't pick (offered options aren't tree nodes; the parent dimension may be).
- It's pure clarification or rephrasing without new intent content.
- **It's a conversation move, not an intent.** "Provide constraints", "create schedule", "confirm choice", "ask for example" describe what the user is doing in the dialogue, not what they want. The *content* they provide (the task list, the constraint, the choice they confirm) is the intent — the move is not.
- **It's an absence of preference.** "I'm unsure on style", "no particular brand", "doesn't matter to me" disclaim a preference. Do not create an `emerging` node for it. Optionally note in `difficulty_notes` if relevant downstream.

### Discovery states

- **`discovered`** — user has articulated or adopted this intent.
- **`emerging`** — user has hypothetically explored this intent but not committed (e.g., "if I retire early"). Conditional and counterfactual exploration only — not general uncertainty.
- **`discovered-via-rejection`** — intent revealed by the user rejecting a violating path the assistant introduced. Use only when there is an actual rejection. **Not** "user disclosed new context that changes the assistant's working assumption" — that is plain `discovered` (the new context is itself the intent evidence). Pair `discovered-via-rejection` with an `actively-rejected` surfacing-log entry for the same turn.

---

## The two-layer separation: tree vs. surfacing log

The tree captures user intents. The surfacing log captures assistant actions and their outcomes. Keep them structurally separate; do not put assistant-introduced concepts in the tree unless the user adopts them.

### Surfacing log states

For every concept the assistant surfaces, record one of:

- **`adopted-immediately`** — user engaged with it in the next turn; concept moves into the tree as discovered.
- **`adopted-deferred`** — user engaged with it N turns later; concept moves into the tree retroactively (relabel earlier turn's outcome from `exposed-no-engagement` to `adopted-deferred`, set `outcome_updated_at_turn`).
- **`exposed-no-engagement`** — surfaced once, no user engagement, conversation continues.
- **`exposed-repeated-no-engagement`** — surfaced two or more times with no user engagement; stronger signal of misalignment.
- **`actively-rejected`** — user explicitly pushed back on it. Often reveals an intent that goes into the tree as `discovered-via-rejection`.
- **`partially-engaged`** — user referenced it without refining or adopting (acknowledged but didn't take it forward). If the user explicitly chose one of several items the assistant listed, that chosen item is `adopted-immediately`, not `partially-engaged`.

### Decomposition rule (critical rule #2, restated)

If an assistant response surfaces N distinct concepts, the surfacing log gets N entries for that turn. Decompose:
- A ranked list of factors → one entry per factor.
- A set of candidate items (shows, restaurants, books) → one entry per candidate. The specific items the user adopts get `adopted-immediately`; the unmentioned ones get `exposed-no-engagement`.
- A multi-dimensional decision matrix → one entry per dimension.

Do not aggregate into "Comprehensive framework on X" or "Multiple options for Y" entries. That destroys downstream traceability of which specific item was adopted, ignored, or re-surfaced.

### Retroactive re-labeling (critical rule #4, restated)

At every new turn pair, before writing the new entries, walk the existing surfacing log and update outcomes that have changed:
- `exposed-no-engagement` → `adopted-deferred` if user just engaged it; set `outcome_updated_at_turn` to the current turn and add a note.
- `exposed-no-engagement` → `exposed-repeated-no-engagement` if the assistant re-surfaced it with still no engagement; set `outcome_updated_at_turn`.
- An adopted-deferred concept also gets a tree node added retroactively (record turn_of_discovery as the *current* turn, not the surfacing turn, because that is when the user evidenced it).

---

## Turn-pair protocol

For each turn pair `(u_t, r_t)`, execute these steps in order:

1. **Read in context.** Note reasons on `u_t` and reactions on `r_t` when present (reactions for awareness only — not used in tree construction).
2. **Identify candidate intents from `u_t`.** Every intent articulated, adopted, hypothetically explored, or revealed through rejection. **Decompose multi-piece turns**: if the user states multiple distinct pieces in one message, each piece is a separate candidate. Test: can you name two distinct things the user said? Then there are two candidates.
3. **Decide tree action for each candidate:** extend an existing tree (specify parent + discovery state), create new tree (only if independence test passes), merge two trees, or skip.
4. **Retroactive pass + new surfacing entries for `r_t`.** First update existing surfacing-log entries per critical rule #4. Then decompose `r_t` per critical rule #2 and log new entries with initial outcome.
5. **Log the decision.** One-line justification per decision.

**Independence test (for "create new tree"):** discovering an intent in this set does not narrow the possible space of any node in any existing tree, and vice versa. Tone/format preferences are the reliable case where the test passes; content sub-topics within the same domain almost always fail it.

---

## Root-level attribute: `intent_type`

Every tree root gets an `intent_type` from this set:

- `learning` — acquire information or form opinions on a domain ("how does X work", "what are the trade-offs between A and B")
- `planning` — design or organize a multi-part artifact or activity. **Not the default.** Use only when the user is *constructing* something with multiple components (an itinerary, a curriculum, a schedule).
- `decision-support` — reach a yes/no or pick-one choice between known alternatives ("should I move out", "which laptop do I buy")
- `information-retrieval` — get a specific piece of information ("where can I watch X", "what's the capital of Y")
- `troubleshooting` — diagnose or fix a problem ("why doesn't this work", "how do I unstick X")
- `brainstorming` — explore a possibility space without committing to a specific outcome ("ideas for X", "what could I do with Y")
- `meta` — the tone/format tree only

**Negative examples for `planning`:**
- User asks where to stream a TV show → `information-retrieval`, not planning.
- User asks "should I learn Python or R first?" → `decision-support`, not planning.
- User asks for an explanation of compound interest → `learning`, not planning.
- User asks for product recommendations → `decision-support` (if they will pick one) or `brainstorming` (if exploring).

When in doubt between two types, pick the one that better describes what success looks like for the user. If success means "I now know X" → learning or information-retrieval. If success means "I have an artifact I built" → planning. If success means "I have made a choice" → decision-support.

---

## Data inputs and their roles

| Signal | Role |
|---|---|
| Discussion (message text) | Primary input for tree structure and discovery states |
| Reasons (attached to user turns) | Supplementary; reveal unarticulated intents and refine discovery states |
| Reactions (attached to assistant turns) | Out of scope for tree construction — record for awareness only |

**Reactions stay out of tree construction.** If reactions shape the tree, and a downstream model predicts reactions from the tree, we get circularity.

---

## Worked example: hierarchy formation in a long conversation

This is a stylized example illustrating how a 6-turn financial-advice conversation produces a multi-level hierarchy rather than a flat list of nodes. Use it as a reference for what good hierarchy looks like.

**Conversation summary:**
- u1: "I have $50k saved and want to learn about asset allocation."
- r1: surfaces 7 concepts (risk tolerance, risk capacity, asset classes, model portfolios, index funds, rebalancing, diversification).
- u2: "Be concise. Should I do mostly bonds or mostly stocks?" (tone-meta + bonds-vs-stocks)
- r2: explains stocks-vs-bonds, asks about timeline.
- u3: "20 years. If I retire early, maybe 15." (timeline + conditional)
- r3: recommends 80/20 stocks/bonds, mentions index funds again.
- u4: "1 sentence — which index fund?" (tone-meta reinforced + index funds adopted)
- r4: names VTI, VOO, mentions rebalancing again.
- u5: "What about crypto?" (crypto, deferred from r1)
- r5: explains crypto risk profile.
- u6: "OK so 80% VTI, 15% bonds, 5% crypto?" (synthesis)

**Good extraction (multi-level):**

```
Tree 1 (intent_type=learning):
  n1: asset allocation
    n2: bonds-vs-stocks split (turn 2, discovered)
      n3: timeline-dependent allocation (turn 3, discovered)
      n4: early-retirement scenario (turn 3, emerging)
    n5: index fund selection (turn 4, discovered) [adopted-deferred from r1]
    n6: crypto inclusion (turn 5, discovered) [adopted-deferred from r1]
    n7: portfolio synthesis (turn 6, discovered)

Tree 2 (intent_type=meta):
  m1: concise responses (turn 2, discovered) [reinforced turn 4]
```

**What this demonstrates:**
- Sub-dimensions of one domain (bonds-vs-stocks, index funds, crypto) are *children* of the parent intent, not separate trees — they all narrow the same allocation question.
- Conditional exploration (`early-retirement scenario`) is a *child of* the dimension it conditions, not a sibling.
- Tone preference gets its own tree because it satisfies the independence test.
- Retroactive `adopted-deferred` is used for `index fund selection` and `crypto inclusion` — both were surfaced in r1 but only engaged later. The surfacing log shows `outcome_updated_at_turn` for both.

**What flat extraction would look like (bad):**

```
n1: asset allocation
  n2: bonds vs stocks (turn 2)
  n3: timeline (turn 3)
  n4: early retirement (turn 3)
  n5: index funds (turn 4)
  n6: crypto (turn 5)
  n7: portfolio (turn 6)
```

Every node directly under the root. This is a star, not a tree — and it loses the relationship that timeline-dependent allocation is a *refinement of* the bonds-vs-stocks decision, not a parallel concern. Avoid star shapes in any conversation longer than 3 user turns.

---

## Output format

Return a single JSON object with this shape:

```json
{
  "conv_id": "<string>",
  "trees": [
    {
      "tree_id": "content",
      "intent_type": "learning",
      "nodes": [
        {
          "id": "n1",
          "label": "<intent at value/need level — Sentence case noun phrase>",
          "discovery_state": "discovered | emerging | discovered-via-rejection",
          "turn_of_discovery": 1,
          "parent_id": null,
          "evidence": "<one short quote or paraphrase from the user turn>"
        }
      ]
    }
  ],
  "surfacing_log": [
    {
      "concept": "<one distinct assistant-introduced concept>",
      "introduced_at_turn": 2,
      "outcome": "adopted-immediately | adopted-deferred | exposed-no-engagement | exposed-repeated-no-engagement | actively-rejected | partially-engaged",
      "outcome_updated_at_turn": null,
      "notes": "<optional, e.g. why outcome changed>"
    }
  ],
  "decision_log": [
    {
      "turn": 1,
      "decision": "extend | new | merge | skip",
      "justification": "<one sentence>"
    }
  ],
  "difficulty_notes": "<one paragraph: ambiguous parent-child relationships, intents that didn't fit obvious dimensions, segments that resisted extraction, absence-of-preference moments>",
  "validation_check": "<one paragraph: does the extracted root structure plausibly match the user's apparent task? If task_expectation is provided, comment on the match.>"
}
```

Rules for the JSON:
- Turns are 1-indexed. A "turn pair" is one user message and the assistant response that follows. Use the user-turn index when referencing a turn pair (so the first user message is turn 1, the second is turn 2, etc.).
- Exactly one node per tree has `parent_id: null` (the root). Every other `parent_id` must reference an existing node's `id` in the same tree.
- Every node id is unique within its tree (e.g. `n1`, `n2`, ...).
- Labels are short noun phrases in Sentence case (e.g. "Bonds vs stocks split", not "bonds_vs_stocks_split" or "BONDS VS STOCKS SPLIT"). Apply consistently across all trees.
- Use `discovered-via-rejection` only when there is an actual rejection of an assistant suggestion. New context that changes the assistant's working assumption is plain `discovered`. Pair `discovered-via-rejection` with an `actively-rejected` surfacing-log entry for the same turn.
- The surfacing log must decompose multi-concept assistant responses into one entry per distinct concept (critical rule #2).
- Retroactive updates to the surfacing log are required at every new turn (critical rule #4), reflected via `outcome_updated_at_turn`.
- If you create a `meta` tree, the independence test must hold — explain why in `difficulty_notes` if non-obvious.
- Do not include intents the user did not evidence. No imagined latents. No conversation-move nodes. No absence-of-preference nodes.
- Return ONLY the JSON object. No prose before or after.
