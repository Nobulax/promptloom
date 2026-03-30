"""Tests for prompt template loading and placeholder resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptloom.prompt import (
    assemble_prompt,
    check_placeholders,
    extract_placeholders,
    load_template,
    resolve_param_value,
    resolve_params,
)


class TestExtractPlaceholders:
    """Tests for :func:`extract_placeholders`."""

    def test_basic(self) -> None:
        template = "Hello {{NAME}}, your id is {{ID}}."
        result = extract_placeholders(template)
        assert result == {"NAME", "ID"}

    def test_no_placeholders(self) -> None:
        assert extract_placeholders("plain text") == set()

    def test_duplicate_placeholders(self) -> None:
        template = "{{A}} and {{A}} again"
        assert extract_placeholders(template) == {"A"}

    def test_mixed_case(self) -> None:
        template = "{{Lower}} and {{UPPER}}"
        assert extract_placeholders(template) == {"Lower", "UPPER"}

    def test_underscores(self) -> None:
        template = "{{MY_PARAM_1}}"
        assert extract_placeholders(template) == {"MY_PARAM_1"}


class TestResolveParamValue:
    """Tests for :func:`resolve_param_value`."""

    def test_literal_string(self, tmp_path: Path) -> None:
        assert resolve_param_value("hello", tmp_path) == "hello"

    def test_file_reference(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("file content", encoding="utf-8")
        result = resolve_param_value(f"file:{f}", tmp_path)
        assert result == "file content"

    def test_file_reference_relative(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("relative content", encoding="utf-8")
        result = resolve_param_value("file:data.txt", tmp_path)
        assert result == "relative content"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_param_value("file:nonexistent.txt", tmp_path)

    def test_glob_relative(self, tmp_path: Path) -> None:
        """glob: prefix reads multiple files sorted by name."""
        (tmp_path / "a.json").write_text('{"a": 1}', encoding="utf-8")
        (tmp_path / "b.json").write_text('{"b": 2}', encoding="utf-8")
        (tmp_path / "c.txt").write_text("ignored", encoding="utf-8")
        result = resolve_param_value("glob:*.json", tmp_path)
        assert "=== File 1: a.json ===" in result
        assert "=== File 2: b.json ===" in result
        assert '{"a": 1}' in result
        assert '{"b": 2}' in result
        assert "ignored" not in result

    def test_glob_absolute(self, tmp_path: Path) -> None:
        """glob: with absolute path works."""
        (tmp_path / "x.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "y.txt").write_text("world", encoding="utf-8")
        result = resolve_param_value(f"glob:{tmp_path}/*.txt", tmp_path)
        assert "=== File 1:" in result
        assert "=== File 2:" in result
        assert "hello" in result
        assert "world" in result

    def test_glob_no_matches_raises(self, tmp_path: Path) -> None:
        """glob: with no matches raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Glob pattern matched no files"):
            resolve_param_value("glob:*.nonexistent", tmp_path)

    def test_glob_subdirectory(self, tmp_path: Path) -> None:
        """glob: works with subdirectory patterns."""
        subdir = tmp_path / "data"
        subdir.mkdir()
        (subdir / "f1.json").write_text("one", encoding="utf-8")
        (subdir / "f2.json").write_text("two", encoding="utf-8")
        result = resolve_param_value("glob:data/*.json", tmp_path)
        assert "=== File 1: f1.json ===" in result
        assert "=== File 2: f2.json ===" in result


class TestResolveParams:
    """Tests for :func:`resolve_params`."""

    def test_mixed(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("doc content", encoding="utf-8")
        params = {
            "instruction": "do something",
            "document": f"file:{f}",
        }
        resolved = resolve_params(params, tmp_path)
        assert resolved["instruction"] == "do something"
        assert resolved["document"] == "doc content"


class TestAssemblePrompt:
    """Tests for :func:`assemble_prompt`."""

    def test_basic_substitution(self) -> None:
        template = "Hello {{NAME}}, your task is {{TASK}}."
        params = {"name": "Alice", "task": "summarize"}
        result = assemble_prompt(template, params)
        assert result == "Hello Alice, your task is summarize."

    def test_case_insensitive(self) -> None:
        template = "{{VALUE}}"
        result = assemble_prompt(template, {"value": "ok"})
        assert result == "ok"

    def test_missing_param_left_unchanged(self) -> None:
        template = "{{PRESENT}} and {{MISSING}}"
        result = assemble_prompt(template, {"present": "yes"})
        assert "yes" in result
        assert "{{MISSING}}" in result

    def test_empty_params(self) -> None:
        template = "no placeholders"
        result = assemble_prompt(template, {})
        assert result == "no placeholders"


class TestCheckPlaceholders:
    """Tests for :func:`check_placeholders`."""

    def test_all_matched(self) -> None:
        template = "{{A}} and {{B}}"
        params = {"a": "1", "b": "2"}
        missing, unused = check_placeholders(template, params)
        assert missing == []
        assert unused == []

    def test_missing_placeholder(self) -> None:
        template = "{{A}} and {{B}}"
        params = {"a": "1"}
        missing, unused = check_placeholders(template, params)
        assert missing == ["B"]
        assert unused == []

    def test_unused_param(self) -> None:
        template = "{{A}}"
        params = {"a": "1", "extra": "2"}
        missing, unused = check_placeholders(template, params)
        assert missing == []
        assert unused == ["EXTRA"]

    def test_both_missing_and_unused(self) -> None:
        template = "{{NEEDED}}"
        params = {"other": "val"}
        missing, unused = check_placeholders(template, params)
        assert missing == ["NEEDED"]
        assert unused == ["OTHER"]


class TestLoadTemplate:
    """Tests for :func:`load_template`."""

    def test_load(self, tmp_path: Path) -> None:
        f = tmp_path / "template.md"
        f.write_text("# Hello\n{{NAME}}", encoding="utf-8")
        result = load_template(f)
        assert "{{NAME}}" in result

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_template(tmp_path / "missing.md")
