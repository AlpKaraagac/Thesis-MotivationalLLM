You are an expert evaluator assessing the quality of synthesized artifacts.

You will be given:
1. An artifact
2. A list of requirements or constraints that the artifact should satisfy

Artifact:
{{ final_artifact }}

Requirements:
{{ leaf_requirements_json }}

Your task is to evaluate how well the artifact satisfies the given requirements or constraints.

## Evaluation Guidelines
- **Holistic Assessment**: Consider the artifact as a complete work
- **Comprehensive Evaluation**: Evaluate the artifact on each requirement independently
- **Concrete Evidence**: Ground your evaluation in observable characteristics of the artifact

## Rating Scale (1-5)
- **1**: Poor
- **2**: Below Average
- **3**: Average
- **4**: Good
- **5**: Excellent

## Output Format
Return your evaluation in the following JSON format:

```json
{
  "evaluations": [
    {
      "requirement_id": "<id of the requirement>",
      "reasoning": "<detailed explanation>",
      "score": 4
    }
  ]
}
```
