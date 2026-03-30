"""Tests for the response processing and validation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from promptloom.validation import (
    ValidationResult,
    _import_callable,
    extract_json,
    load_processor,
    load_validators,
    run_validators,
    validate_json_schema,
)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:
    """Tests for :class:`ValidationResult`."""

    def test_ok(self) -> None:
        r = ValidationResult.ok({"key": "value"})
        assert r.success is True
        assert r.data == {"key": "value"}
        assert r.error is None

    def test_fail(self) -> None:
        r = ValidationResult.fail("something went wrong")
        assert r.success is False
        assert r.error == "something went wrong"
        assert r.data is None


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

class TestExtractJson:
    """Tests for :func:`extract_json`."""

    def test_raw_json_object(self) -> None:
        raw = '{"name": "Alice", "age": 30}'
        result = extract_json(raw)
        assert result == {"name": "Alice", "age": 30}

    def test_raw_json_array(self) -> None:
        raw = '[1, 2, 3]'
        result = extract_json(raw)
        assert result == [1, 2, 3]

    def test_json_in_code_fence(self) -> None:
        raw = 'Here is the output:\n```json\n{"key": "value"}\n```\nDone.'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_json_in_plain_code_fence(self) -> None:
        raw = '```\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self) -> None:
        raw = 'The answer is: {"result": 42} and that is final.'
        result = extract_json(raw)
        assert result == {"result": 42}

    def test_json_with_leading_text(self) -> None:
        raw = "Sure! Here's the JSON:\n\n{\"items\": [\"a\", \"b\"]}"
        result = extract_json(raw)
        assert result == {"items": ["a", "b"]}

    def test_whitespace_padding(self) -> None:
        raw = '   \n  {"key": "value"}  \n  '
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not extract"):
            extract_json("This is just plain text with no JSON.")

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not extract"):
            extract_json("{invalid json content}")

    def test_nested_json(self) -> None:
        raw = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = extract_json(raw)
        assert result["outer"]["inner"] == [1, 2, 3]
        assert result["flag"] is True

    def test_multiple_code_fences_returns_first_valid(self) -> None:
        raw = (
            "```\nnot json\n```\n"
            '```json\n{"key": "value"}\n```\n'
            '```json\n{"other": "ignored"}\n```'
        )
        result = extract_json(raw)
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# Processor loading
# ---------------------------------------------------------------------------

class TestLoadProcessor:
    """Tests for :func:`load_processor`."""

    def test_text_processor(self) -> None:
        proc = load_processor("text")
        assert proc("hello") == "hello"

    def test_json_processor(self) -> None:
        proc = load_processor("json")
        result = proc('{"k": "v"}')
        assert result == {"k": "v"}

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown response_format"):
            load_processor("xml")


# ---------------------------------------------------------------------------
# JSON Schema validation
# ---------------------------------------------------------------------------

class TestValidateJsonSchema:
    """Tests for :func:`validate_json_schema`."""

    SCHEMA = {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }

    def test_valid_data(self) -> None:
        result = validate_json_schema(
            {"name": "Alice", "age": 30}, {}, schema=self.SCHEMA
        )
        assert result.success is True
        assert result.data == {"name": "Alice", "age": 30}

    def test_missing_required_field(self) -> None:
        result = validate_json_schema({"age": 30}, {}, schema=self.SCHEMA)
        assert result.success is False
        assert "name" in result.error

    def test_wrong_type(self) -> None:
        result = validate_json_schema(
            {"name": "Alice", "age": "thirty"}, {}, schema=self.SCHEMA
        )
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# Validator loading
# ---------------------------------------------------------------------------

class TestLoadValidators:
    """Tests for :func:`load_validators`."""

    def test_json_schema_validator(self, tmp_path: Path) -> None:
        schema = {"type": "object", "required": ["x"]}
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(schema), encoding="utf-8")

        specs = [{"type": "json_schema", "schema": "schema.json"}]
        validators = load_validators(specs, tmp_path)

        assert len(validators) == 1
        # Valid data passes.
        r = validators[0]({"x": 1}, {})
        assert r.success is True
        # Invalid data fails.
        r = validators[0]({}, {})
        assert r.success is False

    def test_custom_validator(self, tmp_path: Path) -> None:
        # Use a built-in callable as a test target.
        specs = [
            {
                "type": "custom",
                "callable": "promptloom.validation.extract_json",
            }
        ]
        # Should load without error (the callable exists).
        validators = load_validators(specs, tmp_path)
        assert len(validators) == 1

    def test_missing_schema_file_raises(self, tmp_path: Path) -> None:
        specs = [{"type": "json_schema", "schema": "missing.json"}]
        with pytest.raises(FileNotFoundError):
            load_validators(specs, tmp_path)

    def test_unknown_type_raises(self, tmp_path: Path) -> None:
        specs = [{"type": "xml_validator"}]
        with pytest.raises(ValueError, match="Unknown validator type"):
            load_validators(specs, tmp_path)

    def test_empty_specs(self, tmp_path: Path) -> None:
        validators = load_validators([], tmp_path)
        assert validators == []

    def test_missing_schema_key_raises(self, tmp_path: Path) -> None:
        specs = [{"type": "json_schema"}]
        with pytest.raises(ValueError, match="'schema' key"):
            load_validators(specs, tmp_path)

    def test_missing_callable_key_raises(self, tmp_path: Path) -> None:
        specs = [{"type": "custom"}]
        with pytest.raises(ValueError, match="'callable' key"):
            load_validators(specs, tmp_path)


# ---------------------------------------------------------------------------
# Validator pipeline
# ---------------------------------------------------------------------------

def _pass_validator(data: Any, context: Dict[str, Any]) -> ValidationResult:
    """Always-passing validator for testing."""
    return ValidationResult.ok(data)


def _fail_validator(data: Any, context: Dict[str, Any]) -> ValidationResult:
    """Always-failing validator for testing."""
    return ValidationResult.fail("intentional failure")


def _transform_validator(
    data: Any, context: Dict[str, Any]
) -> ValidationResult:
    """Validator that transforms data (adds a key)."""
    if isinstance(data, dict):
        data = {**data, "validated": True}
    return ValidationResult.ok(data)


class TestRunValidators:
    """Tests for :func:`run_validators`."""

    def test_all_pass(self) -> None:
        result = run_validators(
            [_pass_validator, _pass_validator],
            {"x": 1},
            {},
        )
        assert result.success is True
        assert result.data == {"x": 1}

    def test_first_failure_short_circuits(self) -> None:
        result = run_validators(
            [_fail_validator, _pass_validator],
            {"x": 1},
            {},
        )
        assert result.success is False
        assert result.error == "intentional failure"

    def test_second_failure(self) -> None:
        result = run_validators(
            [_pass_validator, _fail_validator],
            {"x": 1},
            {},
        )
        assert result.success is False

    def test_data_transformation_pipeline(self) -> None:
        result = run_validators(
            [_transform_validator, _pass_validator],
            {"x": 1},
            {},
        )
        assert result.success is True
        assert result.data == {"x": 1, "validated": True}

    def test_empty_validators(self) -> None:
        result = run_validators([], {"x": 1}, {})
        assert result.success is True
        assert result.data == {"x": 1}

    def test_context_is_forwarded(self) -> None:
        received_contexts = []

        def capture_context(
            data: Any, context: Dict[str, Any]
        ) -> ValidationResult:
            received_contexts.append(context)
            return ValidationResult.ok(data)

        ctx = {"task_id": "t1", "model": "gpt-4"}
        run_validators([capture_context], "data", ctx)
        assert len(received_contexts) == 1
        assert received_contexts[0]["task_id"] == "t1"


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------

class TestImportCallable:
    """Tests for :func:`_import_callable`."""

    def test_valid_import(self) -> None:
        fn = _import_callable("json.loads")
        assert fn is json.loads

    def test_invalid_path_raises(self) -> None:
        with pytest.raises(ImportError, match="Invalid callable path"):
            _import_callable("nodotshere")

    def test_nonexistent_module_raises(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            _import_callable("nonexistent_module_xyz.func")

    def test_nonexistent_attr_raises(self) -> None:
        with pytest.raises(AttributeError):
            _import_callable("json.nonexistent_function_xyz")
