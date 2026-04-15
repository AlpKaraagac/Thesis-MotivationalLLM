"""Common parser helpers for structured LLM outputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


class StructuredParseError(ValueError):
    """Raised when structured parsing fails."""


@dataclass(frozen=True, slots=True)
class StructuredPayload:
    """Normalized extracted payload and source metadata."""

    text: str
    source: str


FENCE_PATTERN = re.compile(r"```(?P<lang>[a-zA-Z0-9_-]*)\n(?P<body>.*?)```", re.DOTALL)


def extract_fenced_payload(
    raw_text: str, *, languages: Iterable[str] | None = None
) -> StructuredPayload | None:
    """Extract the first fenced block with a matching language (if provided)."""

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string.")
    allowed = {lang.lower() for lang in languages} if languages else None
    for match in FENCE_PATTERN.finditer(raw_text):
        lang = (match.group("lang") or "").strip().lower()
        if allowed is not None and lang not in allowed:
            continue
        body = match.group("body").strip()
        if body:
            source = f"fenced:{lang or 'plain'}"
            return StructuredPayload(text=body, source=source)
    return None


def extract_braced_payload(raw_text: str) -> StructuredPayload | None:
    """Extract the first balanced JSON-like object/array from text."""

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string.")
    text = raw_text.strip()
    starts = [idx for idx, char in enumerate(text) if char in "[{"]
    for start in starts:
        payload = _balanced_slice(text, start)
        if payload is not None:
            return StructuredPayload(text=payload, source="balanced-braces")
    return None


def _balanced_slice(text: str, start_index: int) -> str | None:
    opener = text[start_index]
    closer = "}" if opener == "{" else "]"
    stack: list[str] = [closer]
    in_string = False
    escape = False
    quote_char = ""

    for index in range(start_index + 1, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == quote_char:
                in_string = False
            continue

        if char in ('"', "'"):
            in_string = True
            quote_char = char
            continue

        if char == "{":
            stack.append("}")
            continue
        if char == "[":
            stack.append("]")
            continue
        if char in ("}", "]"):
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start_index : index + 1].strip()
    return None


def candidate_payload(
    raw_text: str, *, languages: Iterable[str] | None = None
) -> StructuredPayload:
    """Pick the best candidate payload: fenced block, then balanced braces, then raw text."""

    fenced = extract_fenced_payload(raw_text, languages=languages)
    if fenced is not None:
        return fenced
    braced = extract_braced_payload(raw_text)
    if braced is not None:
        return braced
    text = raw_text.strip()
    if not text:
        raise StructuredParseError("Empty model output.")
    return StructuredPayload(text=text, source="raw")
