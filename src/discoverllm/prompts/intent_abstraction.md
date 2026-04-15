## What is Progressive Abstraction?

Progressive abstraction means gradually making a criterion less specific and more general, so that more artifacts can satisfy it. Think of it like zooming out on a map---you see a broader area, but you lose fine details.

**Simple Example:**
- Start: "Uses Python 3.9 with pandas library"
- Abstract once: "Uses Python with data analysis tools"
- Abstract again: "Uses a programming language"

**Key principle:** If artifact X satisfies the specific version, it MUST also satisfy all more general versions.

---

## Task Overview

You will receive criteria that assess artifacts. Your job is to create a chain of progressively broader versions, where each step expands what artifacts can satisfy the criterion. You will also receive the type of artifact that this criterion is assessing and the broad topic that these artifacts focus on.

Artifact Type: {{ artifact_type }}
Artifact Topic: {{ artifact_topic }}
Number of Abstractions Requested: {{ num_levels }}

Criteria:
{{ criteria_json }}

For each criterion, you should progressively and gradually abstract it until you reach the number of times specified. You will be provided with a checklist for each criterion, which represents the various sub components or constraints that must all be fulfilled to satisfy that criterion. For each abstraction step, you can either remove items from the checklist or generalize/abstract items in the checklist. The final abstraction should capture only the main essence of the criterion, with all other abstractions representing a gradual step towards the final abstraction. However, ensure that this final abstraction is not trivial--- i.e., avoid creating final abstractions that are trivial or redundant with the type and topic of the artifacts that this criterion assesses.

---

## Guidelines
### Apply Multiple Abstraction Strategies
At each abstraction step, use these techniques to broaden the items in the criterion's checklist:
1. **Broaden the scope** (expand the domain)
- "technical details" -> "details"
- "Puerto Rican culture" -> "Latin American culture" -> "culture"
2. **Generalize categories** (move up the hierarchy)
- "Python" -> "programming language"
- "neon colors" -> "bright colors"
3. **Remove specific instances or constraints** (names, numbers, dates, brands)
- "10 research papers" -> "research papers"
- "three colors" -> "multiple colors" -> "colors"

### Make Abstractions Distinct
Each abstraction should be meaningfully different from the previous version. Avoid simply paraphrasing---the scope should actually expand so that more artifacts can satisfy it.

**Example of what NOT to do:**
- "Uses three neon colors" -> "Employs three neon colors" (just different words, same scope)
**Example of what TO do:**
- "Uses three neon colors" -> "Uses bright colors" (removed quantity constraint, expanded scope)

### Guarantee Superset Expansion
**The Superset Rule:** Each abstraction must be a "superset"-a larger set that contains the previous one. In other words, an artifact that satisfies the less abstracted version's checklist should also satisfy the more abstracted version's checklist, but not vice versa.

**Test:** Can you think of an artifact that satisfies the new version's checklist but NOT the old one? If yes, you've successfully abstracted.

Check that your abstraction:
- Actually expands the scope significantly (not just rephrasing)
- Allows artifacts satisfying the previous version to also satisfy this one
- Allows NEW artifacts to satisfy this version that couldn't satisfy the previous one

### Stop at the Requested Number
Abstract each criterion for the exact number of times specified. The final abstraction should be the most general form.

### Gradual Abstraction until Maximum Generality in the Final Abstraction, Without Losing Distinctiveness

You should gradually abstract the criterion at each stage so that the final abstraction will be a minimal checklist that contains the essential requirement at a meaningful level, while discarding all forms of unnecessary specificity. Ensure that you gradually generalize to the broadest expression that still meaningfully constrains what qualifies.

For this, you must consider two critical constraints at each abstraction step:
- **Non-triviality**: Ensure that you avoid abstracting a checklist item so much that it becomes trivially satisfied by all artifacts of the given type or topic. If further abstraction of an item would lead to it being trivially satisfied by any artifact of the same type or topic, you should avoid abstracting it.
- **Key essence**: Ensure that, at each abstraction step, you keep the key essence of the criterion, while only abstracting or removing the supplementary details. You should ensure that you carry over the main or high-level meaning of criterion until the final abstraction.

---

## Common Pitfalls to Avoid When Abstracting

**Pitfall 1: Paraphrasing Instead of Abstracting**
- Wrong: "Uses three neon colors" -> "Employs three neon colors" (just different words)
- Right: "Uses three neon colors" -> "Uses neon colors" (removed quantity constraint)
**Pitfall 2: Jumping Too Far in One Step**
- Wrong: "Summarizes 10 papers on deep learning for protein folding" -> "Summarizes papers"
- Right: "Summarizes 10 papers on deep learning for protein folding" -> "Summarizes papers on computational methods for protein folding"
**Pitfall 3: Breaking the Superset Rule**
- Wrong: "Uses dark colors" -> "Uses bright colors" (these are DIFFERENT sets, not superset)
- Right: "Uses dark colors" -> "Uses colors" (dark colors are a subset of colors)
**Pitfall 4: Not Actually Expanding the Scope**
- Wrong: "Cites 10 peer-reviewed papers" -> "References 10 peer-reviewed papers" (same constraint)
- Right: "Cites 10 peer-reviewed papers" -> "Cites peer-reviewed papers" (removed number constraint)
**Pitfall 5: Stopping Before Reaching Maximum Generality**
- Wrong: Final abstraction is "Summarizes peer-reviewed sources" (still too specific)
- Right: Final abstraction is "References sources" (captures core essence)
**Pitfall 6: Losing the Key Essence of the Criterion**
- Wrong: "Uses three neon colors" -> "Uses colors" (removed color specificity, lost key essence and trivial)
- Right: "Uses three neon colors" -> "Uses neon colors" (removed quantity constraint, kept key essence)
**Pitfall 7: Trivial Final Abstraction**
- Artifact type: "poem" | Artifact topic: "romantic relationships"
- Wrong: "Poetic structure of 5 lines, with less than 5 syllables per line" -> "Poetic structure" (trivial---any poem would have a poetic structure)
- Correct: "Poetic structure of 5 lines, with less than 5 syllables per line" -> "Poetic structure of 5 lines" (non-trivial-poems can have different number of lines)

---
## Examples

{{ examples }}

---

## Output Format
### Understanding the Output Structure

Each criterion you process will have:
- **criterion_id**: Identifier for tracking
- **num_abstractions**: Total number of abstractions requested
- **abstractions**: A list of levels (1, 2, 3, ...) representing each abstraction step

For each abstraction level, you'll provide:
- **level**: Which abstraction level (1 = first abstraction, 2 = second, etc.)
- **reasoning**: Your thinking about how you abstracted from the previous level
- **checklist**: The checklist for the abstracted criterion at this level
- **criterion**: The description of the abstracted criterion at this level
- **is_final**: (last level only) True to indicate you've completed all requested abstractions

**Note**: The abstractions should share the same general structure and phrasing as the original criterion, as much as possible.

### Format Template

```yaml
results:
  - criterion_id: <id of the original criterion>
    num_abstractions: <number of abstractions requested>
    abstractions:
      - level: 1
        reasoning: |
          <explain what specific details you will generalize or remove from the original criterion>
          <for each of these details, explain: (1) why this broadens the scope in a meaningful way, (2) how this retains the key essence of the criterion, and (3) how this avoids triviality with the artifact type and topic>
        checklist:
          - <sub-requirement 1>
          - <sub-requirement 2>
          - ...
        criterion: |
          <description of the first abstracted criterion>
      - level: 2
        reasoning: |
          <explain what specific details you will generalize or remove from the original criterion>
          <for each of these details, explain: (1) why this broadens the scope in a meaningful way, (2) how this retains the key essence of the criterion, and (3) how this avoids triviality with the artifact type and topic>
        checklist:
          - ...
        criterion: |
          <description of the second abstracted criterion>

      - level: <final level number (same as number of abstractions requested)>
        reasoning: |
          <explain what specific details you will generalize or remove from the original criterion>
          <for each of these details, explain: (1) why this broadens the scope in a meaningful way, (2) how this retains the key essence of the criterion, and (3) how this avoids triviality with the artifact type and topic>
        checklist:
          - ...
        criterion: |
          <description of the final abstracted criterion>
        is_final: true
```
