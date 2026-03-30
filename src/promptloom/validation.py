"""Response processing and validation for LLM outputs.

Provides a pipeline for processing, validating, and correcting LLM
responses.  Includes built-in processors (JSON extraction) and validators
(JSON Schema), plus support for custom validators via dotted-path imports.

Built-in response processors:

- ``"text"`` -- no-op, returns the raw string unchanged.
- ``"json"`` -- extracts and parses JSON from the response, handling
  markdown code fences, raw JSON, and heuristic extraction.

Built-in validators:

- ``json_schema`` -- validates parsed data against a JSON Schema file.

Custom validators are Python callables with the signature::

    def my_validator(data: Any, context: dict) -> ValidationResult:
        ...

They are referenced in the YAML config by their dotted import path::

    validators:
      - type: custom
        callable: "mypackage.module.my_validator"

The ``context`` dict passed to every validator contains:

- ``task_id`` -- the task identifier.
- ``params`` -- the task's resolved parameters dict.
- ``model`` -- the LiteLLM model identifier.
- ``attempt`` -- the current attempt number (1-based).
"""

from __future__ import annotations

import functools
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Pattern to extract JSON from markdown code fences.
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a validation or processing step.

    Use the factory class methods :meth:`ok` and :meth:`fail` to create
    instances.

    :param success: Whether the validation passed.
    :param data: The (potentially transformed) data on success.
    :param error: Human-readable error message on failure.
    """

    success: bool
    data: Any = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, data: Any) -> ValidationResult:
        """Create a successful validation result.

        :param data: The validated (and optionally transformed) data.
        """
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> ValidationResult:
        """Create a failed validation result.

        :param error: Human-readable description of what went wrong.
            This string is substituted into the ``{{ERROR}}`` placeholder
            of the correction prompt template.
        """
        return cls(success=False, error=error)


# Type alias for validator callables.
ValidatorFn = Callable[[Any, Dict[str, Any]], ValidationResult]


# ---------------------------------------------------------------------------
# Built-in response processors
# ---------------------------------------------------------------------------

def extract_json(raw: str) -> Any:
    """Extract and parse JSON from an LLM response.

    Attempts extraction in order of reliability:

    1. Direct ``json.loads`` on the full (stripped) response.
    2. Extract content from markdown code fences
       (````` ```json ... ``` ````` or ````` ``` ... ``` `````).
    3. Heuristic: locate the outermost ``{…}`` or ``[…]`` and parse.

    :param raw: The raw LLM response string.
    :returns: Parsed JSON data (dict, list, or other JSON-compatible type).
    :raises ValueError: If no valid JSON could be extracted.
    """
    stripped = raw.strip()

    # 1. Direct parse.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. Code fence extraction.
    for match in _JSON_FENCE_RE.finditer(raw):
        candidate = match.group(1).strip()
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # 3. Heuristic: find outermost JSON structure.
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = stripped.find(start_char)
        if start_idx != -1:
            end_idx = stripped.rfind(end_char)
            if end_idx > start_idx:
                try:
                    return json.loads(stripped[start_idx : end_idx + 1])
                except json.JSONDecodeError:
                    continue

    preview = raw[:200] + ("…" if len(raw) > 200 else "")
    raise ValueError(f"Could not extract valid JSON from response: {preview}")


def _identity_processor(raw: str) -> str:
    """No-op processor: returns the raw string unchanged."""
    return raw


# Registry of built-in processors keyed by ``response_format`` name.
_PROCESSORS: Dict[str, Callable[[str], Any]] = {
    "text": _identity_processor,
    "json": extract_json,
}


def load_processor(response_format: str) -> Callable[[str], Any]:
    """Look up a response processor by format name.

    :param response_format: One of ``"text"`` or ``"json"``.
    :returns: A callable ``(raw: str) -> Any``.
    :raises ValueError: If the format name is not recognised.
    """
    if response_format not in _PROCESSORS:
        raise ValueError(
            f"Unknown response_format {response_format!r}. "
            f"Available: {sorted(_PROCESSORS)}"
        )
    return _PROCESSORS[response_format]


# ---------------------------------------------------------------------------
# Built-in validators
# ---------------------------------------------------------------------------

def validate_json_schema(
    data: Any,
    context: Dict[str, Any],
    *,
    schema: dict,
) -> ValidationResult:
    """Validate data against a JSON Schema.

    :param data: The data to validate (typically a parsed JSON object).
    :param context: Task context dict (unused by this validator).
    :param schema: The JSON Schema dict to validate against.
    :returns: :class:`ValidationResult` indicating success or failure.
    """
    import jsonschema

    try:
        jsonschema.validate(instance=data, schema=schema)
        return ValidationResult.ok(data)
    except jsonschema.ValidationError as exc:
        return ValidationResult.fail(
            f"JSON Schema validation error: {exc.message}"
        )


# ---------------------------------------------------------------------------
# Validator loading
# ---------------------------------------------------------------------------

def _import_callable(dotted_path: str) -> Callable:
    """Import a callable by its dotted Python path.

    :param dotted_path: Fully-qualified name, e.g.
        ``"mypackage.module.my_validator"``.
    :returns: The imported callable.
    :raises ImportError: If the module cannot be found.
    :raises AttributeError: If the attribute does not exist in the module.
    """
    module_path, sep, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(
            f"Invalid callable path {dotted_path!r}: "
            f"must be a dotted path like 'package.module.function'"
        )
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def load_validators(
    specs: List[Dict[str, Any]],
    base_dir: Path,
) -> List[ValidatorFn]:
    """Resolve validator specifications from YAML config into callables.

    Supported spec types:

    ``json_schema``
        Validates against a JSON Schema file.  Requires a ``schema`` key
        pointing to a ``.json`` file (relative to *base_dir* or absolute).

    ``custom``
        Imports a user-defined validator function.  Requires a ``callable``
        key with a dotted Python import path.

    :param specs: List of validator specification dicts from the YAML config.
    :param base_dir: Base directory for resolving relative file paths.
    :returns: List of ready-to-call validator functions.
    :raises ValueError: If a spec has an unknown ``type``.
    :raises FileNotFoundError: If a referenced schema file does not exist.
    """
    validators: List[ValidatorFn] = []

    for spec in specs:
        vtype = spec.get("type", "")

        if vtype == "json_schema":
            schema_path_str = spec.get("schema")
            if not schema_path_str:
                raise ValueError(
                    "json_schema validator requires a 'schema' key "
                    "pointing to a JSON Schema file."
                )
            schema_path = Path(schema_path_str)
            if not schema_path.is_absolute():
                schema_path = base_dir / schema_path
            if not schema_path.exists():
                raise FileNotFoundError(
                    f"JSON Schema file not found: {schema_path}"
                )
            with open(schema_path, "r", encoding="utf-8") as fh:
                schema = json.load(fh)
            validators.append(
                functools.partial(validate_json_schema, schema=schema)
            )

        elif vtype == "custom":
            dotted = spec.get("callable")
            if not dotted:
                raise ValueError(
                    "custom validator requires a 'callable' key "
                    "with a dotted Python import path."
                )
            fn = _import_callable(dotted)
            validators.append(fn)

        else:
            raise ValueError(
                f"Unknown validator type {vtype!r}. "
                f"Supported types: 'json_schema', 'custom'"
            )

    return validators


# ---------------------------------------------------------------------------
# Validator pipeline
# ---------------------------------------------------------------------------

def run_validators(
    validators: List[ValidatorFn],
    data: Any,
    context: Dict[str, Any],
) -> ValidationResult:
    """Run an ordered chain of validators on the processed response data.

    Validators execute sequentially.  Each receives the (potentially
    transformed) data from the previous validator's
    :attr:`ValidationResult.data`.  The chain **short-circuits** on the
    first failure.

    :param validators: Ordered list of validator callables.
    :param data: The processed response data to validate.
    :param context: Task context dict forwarded to each validator.
    :returns: The final :class:`ValidationResult` — either the last
        successful result or the first failure.
    """
    for validator in validators:
        result = validator(data, context)
        if not result.success:
            return result
        data = result.data  # Allow validators to transform data.
    return ValidationResult.ok(data)
