"""Load prompt markdown files and fill in ``{{ variables }}``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from discoverllm.config import DEFAULT_CONFIG

PLACEHOLDER_PATTERN = re.compile(r"{{\s*([a-zA-Z_]\w*)\s*}}")

class PromptNotFoundError(FileNotFoundError):
    """Raised when a required prompt template cannot be found."""


@dataclass(slots=True)
class PromptLoader:
    """Load and render markdown prompt templates from disk."""

    prompt_dir: Path = DEFAULT_CONFIG.prompt_dir

    def __post_init__(self) -> None:
        self.prompt_dir = Path(self.prompt_dir)
        if not self.prompt_dir.is_dir():
            raise FileNotFoundError(f"Prompt directory does not exist: {self.prompt_dir}")

    def list_prompt_names(self) -> list[str]:
        return sorted(path.stem for path in self.prompt_dir.glob("*.md"))

    def prompt_path(self, prompt_name: str) -> Path:
        normalized = str(prompt_name).strip()
        if not normalized:
            raise ValueError("prompt_name must be non-empty.")
        if "/" in normalized or "\\" in normalized:
            raise ValueError("prompt_name must not contain path separators.")

        path = (self.prompt_dir / f"{normalized}.md").resolve()
        prompt_dir_resolved = self.prompt_dir.resolve()
        if prompt_dir_resolved not in path.parents:
            raise ValueError("prompt_name resolved outside prompt_dir.")
        if not path.exists():
            raise PromptNotFoundError(f"Prompt template not found: {normalized}")
        return path

    def load_template(self, prompt_name: str) -> str:
        path = self.prompt_path(prompt_name)
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Prompt template is empty: {path.name}")
        return content

    def render(
        self,
        prompt_name: str,
        variables: Mapping[str, Any] | None = None,
        *,
        strict: bool = True,
    ) -> str:
        template = self.load_template(prompt_name)
        return render_template(template, variables or {}, strict=strict)


def find_placeholders(template: str) -> list[str]:
    """Return placeholder names found in a template in first-seen order."""

    found = PLACEHOLDER_PATTERN.findall(template)
    unique: list[str] = []
    seen: set[str] = set()
    for name in found:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique

def render_template(
    template: str,
    variables: Mapping[str, Any],
    *,
    strict: bool = True,
) -> str:
    """Replace placeholders in a template with values from a mapping."""

    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            missing.add(key)
            return match.group(0)
        value = variables[key]
        if value is None:
            return ""
        return str(value)

    rendered = PLACEHOLDER_PATTERN.sub(replace, template)
    if strict and missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing prompt variables: {missing_list}")
    return rendered
