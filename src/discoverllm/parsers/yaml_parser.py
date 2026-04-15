"""YAML parser with optional backend and deterministic fallback."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from typing import Any

from discoverllm.parsers.common import StructuredParseError, StructuredPayload, extract_fenced_payload
from discoverllm.parsers.json_parser import parse_json_with_repair


def parse_yaml_strict(raw_text: str) -> Any:
    """Parse YAML from a model output payload."""

    payload = _yaml_candidate_payload(raw_text)
    text = payload.text.strip()
    if not text:
        raise StructuredParseError("Empty YAML payload.")

    if text.startswith("{") or text.startswith("["):
        return parse_json_with_repair(text)

    if importlib.util.find_spec("yaml") is not None:
        import yaml  # type: ignore

        try:
            parsed = yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001
            raise StructuredParseError("YAML parse failed with backend parser.") from exc
        if parsed is None:
            raise StructuredParseError("YAML parser returned empty payload.")
        return _normalize_yaml_values(parsed)

    return _normalize_yaml_values(_parse_simple_yaml(text))


def parse_yaml_with_repair(raw_text: str) -> Any:
    """Parse YAML and attempt deterministic repair for malformed outputs."""

    try:
        return parse_yaml_strict(raw_text)
    except StructuredParseError:
        payload = _yaml_candidate_payload(raw_text)
        repaired = payload.text.replace("\t", "  ").strip()
        if not repaired:
            raise StructuredParseError("YAML parse failed after deterministic repair.")

        if repaired.startswith("{") or repaired.startswith("["):
            try:
                return _normalize_yaml_values(parse_json_with_repair(repaired))
            except Exception as exc:  # noqa: BLE001
                raise StructuredParseError(
                    "YAML parse failed after deterministic repair."
                ) from exc

        try:
            return _normalize_yaml_values(_parse_simple_yaml(repaired))
        except Exception as exc:  # noqa: BLE001
            raise StructuredParseError("YAML parse failed after deterministic repair.") from exc


def _yaml_candidate_payload(raw_text: str) -> StructuredPayload:
    fenced = extract_fenced_payload(raw_text, languages=("yaml", "yml", "json"))
    if fenced is not None:
        return fenced
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string.")
    text = raw_text.strip()
    if not text:
        raise StructuredParseError("Empty YAML payload.")
    return StructuredPayload(text=text, source="raw")


def _parse_simple_yaml(text: str) -> Any:
    """Minimal YAML subset parser (dict/list/scalar) for deterministic fallback."""

    lines = _prepare_lines(text)
    if not lines:
        raise StructuredParseError("No YAML content to parse.")
    value, index = _parse_block(lines, 0, 0)
    if index != len(lines):
        raise StructuredParseError("Unexpected trailing YAML content.")
    return value


def _prepare_lines(text: str) -> list[tuple[int, str]]:
    prepared: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if "\t" in raw_line:
            raise StructuredParseError("Tabs are not supported in fallback YAML parser.")
        stripped = raw_line.lstrip(" ")
        if stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(stripped)
        if indent % 2 != 0:
            raise StructuredParseError(
                "Fallback YAML parser requires indentation in multiples of two spaces."
            )
        prepared.append((indent, stripped.rstrip()))
    return prepared


def _parse_block(
    lines: list[tuple[int, str]], index: int, indent: int
) -> tuple[Any, int]:
    if index >= len(lines):
        raise StructuredParseError("Unexpected end of YAML input.")
    current_indent, current_text = lines[index]
    if current_indent < indent:
        raise StructuredParseError("Unexpected indentation drop in YAML block.")
    if current_text.startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_dict(lines, index, indent)


def _parse_dict(
    lines: list[tuple[int, str]], index: int, indent: int
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent:
            raise StructuredParseError("Invalid indentation in YAML mapping.")
        if text.startswith("- "):
            break
        if ":" not in text:
            raise StructuredParseError(f"Invalid YAML mapping entry: {text!r}")

        key, remainder = text.split(":", 1)
        key = key.strip()
        remainder = remainder.strip()
        if not key:
            raise StructuredParseError("YAML mapping key must be non-empty.")
        if key in result:
            raise StructuredParseError(f"Duplicate YAML key: {key!r}")

        if _is_block_scalar_marker(remainder):
            index += 1
            block_value, index = _parse_block_scalar(
                lines,
                index,
                current_indent + 2,
                folded=remainder.startswith(">"),
            )
            result[key] = block_value
            continue

        if remainder:
            result[key] = _parse_scalar(remainder)
            index += 1
            continue

        index += 1
        if index >= len(lines) or lines[index][0] <= current_indent:
            result[key] = None
            continue
        nested, index = _parse_block(lines, index, current_indent + 2)
        result[key] = nested
    return result, index


INLINE_MAPPING_PATTERN = re.compile(r"^[^:\[\{][^:]*:\s*.*$")


def _parse_list(
    lines: list[tuple[int, str]], index: int, indent: int
) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent:
            raise StructuredParseError("Invalid indentation in YAML sequence.")
        if not text.startswith("- "):
            break

        remainder = text[2:].strip()
        if not remainder:
            index += 1
            if index >= len(lines) or lines[index][0] <= current_indent:
                result.append(None)
                continue
            nested, index = _parse_block(lines, index, current_indent + 2)
            result.append(nested)
            continue

        if _is_block_scalar_marker(remainder):
            index += 1
            block_value, index = _parse_block_scalar(
                lines,
                index,
                current_indent + 2,
                folded=remainder.startswith(">"),
            )
            result.append(block_value)
            continue

        if INLINE_MAPPING_PATTERN.match(remainder):
            inline_mapping, index = _parse_inline_mapping_item(
                lines, index, indent, remainder
            )
            result.append(inline_mapping)
            continue

        result.append(_parse_scalar(remainder))
        index += 1
    return result, index


def _parse_inline_mapping_item(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
    inline_text: str,
) -> tuple[dict[str, Any], int]:
    key, remainder = inline_text.split(":", 1)
    key = key.strip()
    remainder = remainder.strip()
    mapping: dict[str, Any] = {}
    mapping[key] = _parse_scalar(remainder) if remainder else None
    index += 1

    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent + 2:
            break
        if current_indent != indent + 2:
            raise StructuredParseError("Invalid indentation in inline mapping continuation.")
        if text.startswith("- "):
            break
        if ":" not in text:
            raise StructuredParseError(f"Invalid YAML mapping entry: {text!r}")
        child_key, child_remainder = text.split(":", 1)
        child_key = child_key.strip()
        child_remainder = child_remainder.strip()
        if _is_block_scalar_marker(child_remainder):
            index += 1
            block_value, index = _parse_block_scalar(
                lines,
                index,
                current_indent + 2,
                folded=child_remainder.startswith(">"),
            )
            mapping[child_key] = block_value
            continue

        if child_remainder:
            mapping[child_key] = _parse_scalar(child_remainder)
            index += 1
            continue

        index += 1
        if index >= len(lines) or lines[index][0] <= current_indent:
            mapping[child_key] = None
            continue
        nested, index = _parse_block(lines, index, current_indent + 2)
        mapping[child_key] = nested

    return mapping, index


def _is_block_scalar_marker(value: str) -> bool:
    return value in {"|", "|-", "|+", ">", ">-", ">+"}


def _parse_block_scalar(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
    *,
    folded: bool,
) -> tuple[str, int]:
    parts: list[str] = []
    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent:
            break
        relative_indent = max(0, current_indent - indent)
        parts.append((" " * relative_indent) + text)
        index += 1
    if folded:
        content = " ".join(part.strip() for part in parts if part.strip())
    else:
        content = "\n".join(parts)
    return content.strip(), index


INT_PATTERN = re.compile(r"^[+-]?\d+$")
FLOAT_PATTERN = re.compile(r"^[+-]?(\d+\.\d+|\d+\.\d*|\.\d+)$")


def _parse_scalar(token: str) -> Any:
    value = token.strip()
    lower = value.lower()
    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    if lower in {"null", "~", "none"}:
        return None

    if INT_PATTERN.match(value):
        try:
            return int(value)
        except ValueError:
            pass
    if FLOAT_PATTERN.match(value):
        try:
            return float(value)
        except ValueError:
            pass

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        try:
            return ast.literal_eval(value)
        except Exception as exc:  # noqa: BLE001
            raise StructuredParseError(f"Invalid quoted YAML scalar: {value!r}") from exc

    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return parse_json_with_repair(value)

    return value


def _normalize_yaml_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_yaml_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_yaml_values(item) for item in value]
    if isinstance(value, str):
        return value.rstrip("\n")
    return value
