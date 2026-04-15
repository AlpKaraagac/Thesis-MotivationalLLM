You will be provided with a slice of the chat history between a user and an AI assistant. You will also be provided with an evaluation criterion organized as a hierarchy of checklist items.

<|The Start of the Chat History Slice|>
{{ chat_history_json }}
<|The End of the Chat History Slice|>

<|The Start of the Criterion Hierarchy|>
{{ active_root_subtree_json }}
<|The End of the Criterion Hierarchy|>

Your job has two phases focused ONLY on the assistant's last message:
1) **Classify the last message as "Dialog Act" or "Artifact"**
- **Artifact**: The artifact, artifact samples, or multiple artifact options that the user requested in their initial message.
- **Dialog Act**: Questions, clarifications, confirmations, discussions, or any conversational move meant to understand the user's intents and goals, with zero artifact content in the message
- Output must include your `classification_reasoning` and the `classification_label`.
- **Additional Rules**:
  - Artifact Samples: A message with samples of artifacts should still be classified as "artifact".
  - Multiple Artifact Options: A single message with multiple artifact options should still be classified as "artifact".
  - Prioritize "Artifact" over "Dialog Act"

2) **Conditionally evaluate based on the classification**
- If the last message is a **Dialog Act**, evaluate whether it **probes** the items in the provided hierarchy.
- If the last message is an **Artifact**, evaluate the **satisfaction** of the items in the provided hierarchy.
- **Never** do both. Only one evaluation section should be present depending on the classification.

**Important calibration rules:**
- A **Dialog Act** only counts as probing when it tries to elicit genuinely new information that was not already made explicit by the user earlier in the chat.
- If the user has already given a concrete request on this dimension, then merely restating it, menu-izing it, or asking them to pick among paraphrases of the same idea should **not** count as probing.
- For **Artifact** evaluation, judge broader ancestor nodes on their own wording and scope. If the artifact clearly satisfies the broader abstraction, keep that ancestor satisfied even when a deeper descendant is only a near miss.
- Be stricter for the deepest and most specific leaves. A leaf with a distinctive concrete detail should stay unsatisfied unless that exact detail is clearly realized in substance, not merely approximated by a nearby variant.
- When an artifact comes close but misses the exact leaf detail, populate `near_miss` with the closest concrete variant that was actually realized.

**Hierarchical Evaluation Guidelines:**
1. **Tree Traversal Rule**: Start by evaluating the root node(s). For each node that is satisfied/probed, recursively evaluate its children in depth-first order.
2. **Stopping Rule per Branch**: When a node is NOT satisfied/probed, stop evaluating its descendants but continue evaluating siblings.
3. **Parent-Child Dependency & Scope**: Only evaluate a child node if its parent was satisfied/probed.
4. **Independence Across Items**: Evaluate each node independently.
5. **Best-Alternative Rule (critical)**: Different nodes may be satisfied or probed by different alternatives in the assistant's message.
6. **Near-Miss Tracking**: When a node is NOT satisfied or probed, consider whether the assistant's last message satisfied or probed related variants of that node.

**Return your output in this YAML. Include ONLY the evaluation section that matches the classification.**

```yaml
classification_reasoning: "<one-line explanation of why the last message is artifact vs dialog act>"
classification_label: "artifact"
evaluation_type: "satisfaction"
evaluations:
  - node_id: "1"
    node_text: "<exact text of the node from hierarchy>"
    reasoning: "<one-line explanation of why the assistant's last message succeeds or fails at satisfying/probing the node described above>"
    is_satisfied_or_probed: true
    children_evaluated: true
  - node_id: "1.1"
    node_text: "<exact text of the node from hierarchy>"
    reasoning: "<one-line explanation of why the assistant's last message succeeds or fails at satisfying/probing the node described above>"
    is_satisfied_or_probed: false
    near_miss:
      - "<description of a near-miss variant>"
    children_evaluated: false
```
