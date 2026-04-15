"""Lightweight lexical grounding heuristics for deterministic alignment checks."""

from __future__ import annotations

import re
from typing import Iterable

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "bit",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "make",
    "maybe",
    "more",
    "of",
    "on",
    "or",
    "so",
    "some",
    "that",
    "the",
    "their",
    "there",
    "through",
    "to",
    "up",
    "with",
    "would",
    "you",
    "your",
}

_GENERIC_CRITERION_TOKENS = {
    "along",
    "clear",
    "depict",
    "describe",
    "describ",
    "different",
    "during",
    "end",
    "featur",
    "hour",
    "include",
    "includ",
    "introduc",
    "moment",
    "part",
    "place",
    "present",
    "prominent",
    "reduc",
    "route",
    "scene",
    "set",
    "setting",
    "show",
    "subtle",
    "through",
    "use",
    "visual",
}

_SYNONYM_MAP = {
    "afterdark": "night",
    "blueblack": "night",
    "canal": "water",
    "canals": "water",
    "city": "urban",
    "cities": "urban",
    "contemplative": "reflect",
    "dark": "night",
    "darkened": "night",
    "darkness": "night",
    "deepening": "night",
    "dimmer": "light",
    "dusk": "evening",
    "embankment": "water",
    "glow": "light",
    "glowed": "light",
    "glowing": "light",
    "harbor": "water",
    "harbour": "water",
    "headlights": "light",
    "illuminated": "light",
    "illumination": "light",
    "inwardness": "inward",
    "lamps": "light",
    "lanterns": "light",
    "late-day": "evening",
    "late": "evening",
    "lit": "light",
    "lights": "light",
    "misty": "mist",
    "mysterious": "mystery",
    "nightfall": "night",
    "nighttime": "night",
    "pause": "pause",
    "paused": "pause",
    "pausing": "pause",
    "peaceful": "calm",
    "quieted": "quiet",
    "quieter": "quiet",
    "quieting": "quiet",
    "reflective": "reflect",
    "reflection": "reflect",
    "reflects": "reflect",
    "river": "water",
    "rivers": "water",
    "riverside": "water",
    "river-side": "water",
    "riverbank": "water",
    "settled": "settle",
    "settling": "settle",
    "softened": "soft",
    "softening": "soft",
    "softer": "soft",
    "soundscape": "sound",
    "streetlights": "light",
    "sunset": "evening",
    "twilight": "evening",
    "walked": "walk",
    "walking": "walk",
    "walks": "walk",
    "waterfront": "water",
    "windows": "window",
}


def salient_tokens(text: str) -> list[str]:
    """Return normalized lexical cues suitable for lightweight grounding checks."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    normalized = text.replace("’", "'").replace("-", " ")
    tokens: list[str] = []
    for raw_token in _TOKEN_PATTERN.findall(normalized):
        token = _normalize_token(raw_token)
        if not token:
            continue
        if token in _STOPWORDS or token in _GENERIC_CRITERION_TOKENS:
            continue
        tokens.append(token)
    return tokens


def grounding_score(text: str, reference_texts: Iterable[str]) -> int:
    """Count the distinct salient overlaps between text and one or more references."""

    text_token_set = set(salient_tokens(text))
    if not text_token_set:
        return 0
    reference_token_set: set[str] = set()
    for value in reference_texts:
        reference_token_set.update(salient_tokens(value))
    return len(text_token_set & reference_token_set)


def overlap_tokens(text: str, reference_texts: Iterable[str]) -> list[str]:
    """Return the distinct overlapping salient tokens in deterministic order."""

    text_tokens = salient_tokens(text)
    reference_token_set: set[str] = set()
    for value in reference_texts:
        reference_token_set.update(salient_tokens(value))

    overlap: list[str] = []
    seen: set[str] = set()
    for token in text_tokens:
        if token not in reference_token_set or token in seen:
            continue
        seen.add(token)
        overlap.append(token)
    return overlap


def _normalize_token(token: str) -> str:
    normalized = token.strip("'").lower()
    if not normalized:
        return ""
    if normalized in _SYNONYM_MAP:
        return _SYNONYM_MAP[normalized]

    if normalized.endswith("ies") and len(normalized) > 4:
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("ing") and len(normalized) > 5:
        normalized = normalized[:-3]
    elif normalized.endswith("ed") and len(normalized) > 4:
        normalized = normalized[:-2]
    elif normalized.endswith("ly") and len(normalized) > 4:
        normalized = normalized[:-2]
    elif normalized.endswith("ness") and len(normalized) > 6:
        normalized = normalized[:-4]
    elif normalized.endswith("er") and len(normalized) > 4:
        normalized = normalized[:-2]
    elif normalized.endswith("s") and len(normalized) > 3:
        normalized = normalized[:-1]

    return _SYNONYM_MAP.get(normalized, normalized)
