"""JSON-schema validator for tool inputs and outputs.

validate_schema(data, schema) — lightweight validation without jsonschema dependency.
assert_valid(tool_name, data, schema) — validates and logs, returns data unchanged.

Supports:
  • type checking: string, number, integer, boolean, object, array, null
  • nullable types: ["string", "null"]
  • required fields on objects
  • nested object / array validation
  • additionalProperties (ignored — permissive)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

try:
    from app.observability.logger import get_logger
    _log = get_logger("medorchestrator.validator")
except Exception:
    _log = logging.getLogger("medorchestrator.validator")

# ── Type map ──────────────────────────────────────────────────────────────────

_TYPE_MAP = {
    "string":  str,
    "number":  (int, float),
    "integer": int,
    "boolean": bool,
    "object":  dict,
    "array":   list,
}


# ── Core validator ────────────────────────────────────────────────────────────

def validate_schema(
    data:   Any,
    schema: Dict[str, Any],
    path:   str = "$",
) -> Tuple[bool, List[str]]:
    """Validate `data` against a simplified JSON-schema dict.

    Returns
    -------
    (is_valid: bool, errors: list[str])
    """
    errors: List[str] = []

    if not schema:
        return True, []

    schema_type = schema.get("type")

    # ── Handle nullable types: ["string", "null"] ─────────────────────────
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        if data is None:
            return True, []     # null is explicitly allowed
        # Validate against any of the non-null types
        for t in non_null:
            ok, _ = validate_schema(data, {**schema, "type": t}, path)
            if ok:
                return True, []
        errors.append(f"{path}: expected one of {schema_type}, got {type(data).__name__!r}")
        return False, errors

    # ── null type ─────────────────────────────────────────────────────────
    if schema_type == "null":
        if data is not None:
            errors.append(f"{path}: expected null, got {type(data).__name__!r}")
        return len(errors) == 0, errors

    # ── Type check ────────────────────────────────────────────────────────
    if schema_type and schema_type in _TYPE_MAP:
        expected_py = _TYPE_MAP[schema_type]
        if not isinstance(data, expected_py):
            errors.append(
                f"{path}: expected {schema_type}, got {type(data).__name__!r}"
                + (f" (value: {str(data)[:60]})" if data is not None else "")
            )
            return False, errors

    # ── Object ────────────────────────────────────────────────────────────
    if schema_type == "object" or isinstance(data, dict):
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__!r}")
            return False, errors

        # Required fields
        for req_field in schema.get("required", []):
            if req_field not in data:
                errors.append(f"{path}.{req_field}: required field missing")

        # Property validation
        props = schema.get("properties", {})
        for key, sub_schema in props.items():
            if key in data:
                _, sub_errors = validate_schema(data[key], sub_schema, f"{path}.{key}")
                errors.extend(sub_errors)

    # ── Array ─────────────────────────────────────────────────────────────
    elif schema_type == "array" or isinstance(data, list):
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__!r}")
            return False, errors

        item_schema = schema.get("items")
        if item_schema:
            # Only validate first 10 items for performance
            for i, item in enumerate(data[:10]):
                _, sub_errors = validate_schema(item, item_schema, f"{path}[{i}]")
                errors.extend(sub_errors)

    return len(errors) == 0, errors


# ── Convenience helpers ───────────────────────────────────────────────────────

def assert_valid(
    tool_name: str,
    data:      Any,
    schema:    Dict[str, Any],
    direction: str = "output",   # "input" or "output"
) -> Any:
    """Validate data and log failures.  Always returns data unchanged."""
    ok, errors = validate_schema(data, schema)
    if not ok:
        _log.warning(
            f"Tool '{tool_name}' {direction} validation failed",
            extra={"tool": tool_name, "direction": direction, "errors": errors[:5]},
        )
    return data


def coerce_to_list(value: Any, fallback: list | None = None) -> list:
    """Safely coerce a value to a list; return fallback on failure."""
    if isinstance(value, list):
        return value
    if value is None:
        return fallback if fallback is not None else []
    return [value]


def coerce_to_dict(value: Any, fallback: dict | None = None) -> dict:
    """Safely coerce a value to a dict; return fallback on failure."""
    if isinstance(value, dict):
        return value
    if value is None:
        return fallback if fallback is not None else {}
    try:
        return dict(value)
    except Exception:
        return fallback if fallback is not None else {}
