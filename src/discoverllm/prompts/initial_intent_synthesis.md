You will receive a text artifact and its type.

Artifact Type: {{ artifact_type }}

Artifact:
{{ artifact_content }}

Your task is to:

(a) Identify the very broad or general topic of the artifact, which you will represent in a short phrase.
(b) Comprehensively describe the specific characteristics and attributes of the artifact, which are not a given from the artifact type and the artifact topic. This description should contain significant detail so that a person that follows this description can create an artifact that captures the general essence of the original artifact (but not necessarily the same artifact). Treat this description as a guideline or set of requirements/constraints for creating an artifact that resembles the original.
(c) Decompose the description into a checklist that represents the various sub-requirements, constraints, or subguidelines that are included in the description, and must all be fulfilled in order to create an artifact that satisfies the description.

**Guidelines:**

1. **Broad Topic:** The topic should be broad and general, capturing the main idea or direction of the artifact without revealing specific details.
2. **Key Characteristics:** The description should capture the key or main characteristics of the original artifact. Ensure that you include the most essential, important, and/or representative aspects of the original artifact that make it unique and distinct from other artifacts of the same type or with the same topic. Be specific and selective when deciding what to include in the description.
3. **Independent of Artifact Type and Topic:** The description should focus on the characteristics that go beyond what is already implied or given by the artifact's type and topic. Avoid restating generic features that would apply to any artifact of that type or any artifact with that topic.
4. **Positive Framing:** Phrase the description in positive or neutral terms. Avoid phrasing that suggests the artifact is deficient or deviates from an assumed standard. Avoid prescribing errors or mistakes. It is acceptable to slightly reinterpret the original artifact if needed to keep the framing neutral or appreciative.
- Wrong: "Switches inconsistently between professional, academic tone to more colloquial, informal tone"
- Right: "Blends multiple registers by alternating professional and casual language"
5. **Description and Checklist are Equal**: For any detail, if it is included in a checklist item, it should have also been included in the artifact description. Ensure that the artifact description itself includes all the details.
6. **Independent Checklist Items:** Each checklist item should represent a distinct sub-requirement. Avoid creating checklist items that overlap (i.e., satisfying one item automatically leads to another item being satisfied by default).

**Examples:**

{{ examples }}

**Return your output in this format:**

```yaml
internal_thinking: |
  <think about the broad topic of the artifact>

  <think in-depth about what type of constraints or requirements must be met to recreate the artifact closely matching the original>

  <verify that you have captured all the key characteristics of the original artifact>

  <check whether any of these characteristics are trivial or redundant when considering the artifact type and topic; if so, remove them from the description or modify them to not be trivial>

  <think about how to decompose the description into checklist items, and verify that the items are independent and one does not automatically satisfy another>

artifact_topic: <short phrase of 1-3 words describing the broad topic of the artifact>
description: <description of the artifact's key characteristics and attributes>
checklist:
  - <checklist item 1>
  - <checklist item 2>
  - ...
```
