"""Validate provider-generated function calls against caller-declared schemas."""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


def validate_tool_schemas(tools: list[dict[str, Any]] | None) -> None:
    """Check caller-declared schemas and reject references that could fetch remote data."""
    for tool in tools or []:
        schema = tool["function"].get("parameters", {"type": "object"})
        Draft202012Validator.check_schema(schema)
        pending = [schema]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                reference = item.get("$ref")
                if reference is not None and (not isinstance(reference, str) or not reference.startswith("#")):
                    raise ValueError("Tool schemas may only use local JSON pointer references")
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)


def validate_tool_calls(tool_calls: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                        reject_undeclared: bool = True) -> None:
    """Raise ValueError when a generated call is undeclared or has invalid arguments."""
    if not tool_calls:
        return
    validate_tool_schemas(tools)
    declared = {
        tool["function"]["name"]: tool["function"].get("parameters", {"type": "object"})
        for tool in tools or []
    }
    for index, call in enumerate(tool_calls):
        function = call.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"Generated tool call {index} has no function object")  # noqa: TRY004
        name = function.get("name")
        if name not in declared:
            if reject_undeclared:
                raise ValueError(f"Generated tool call {index} references an undeclared function")
            continue
        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            raise ValueError(f"Generated tool call {index} arguments must be a JSON string")  # noqa: TRY004
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Generated tool call {index} arguments are invalid JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"Generated tool call {index} arguments must be a JSON object")  # noqa: TRY004
        schema = declared[name]
        try:
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(arguments)
        except SchemaError as exc:
            raise ValueError(f"Declared schema for tool {name!r} is invalid") from exc
        except ValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path)
            location = f" at {path}" if path else ""
            raise ValueError(f"Generated tool call {index} arguments violate the schema{location}") from exc
