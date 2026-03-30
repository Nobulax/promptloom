"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptloom.config import ExperimentConfig, TaskConfig, load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_YAML = """\
defaults:
  models:
    - "openai/gpt-4o"
  prompt_template: "prompt.md"

tasks:
  - id: "t1"
    params:
      text: "hello"
"""

OVERRIDE_YAML = """\
defaults:
  models:
    - "openai/gpt-4o"
  prompt_template: "default.md"
  system_prompt: "default system"
  output_dir: "out"
  max_completion_tokens: 1000
  timeout: 60
  max_concurrency: 5
  ignore_unused_params: true

tasks:
  - id: "t1"
    params:
      x: "1"
  - id: "t2"
    params:
      x: "2"
    models:
      - "anthropic/claude-sonnet-4-20250514"
    prompt_template: "override.md"
    system_prompt: "override system"
    output_dir: "custom_out"
    max_completion_tokens: 9999
    timeout: 999
"""

VALIDATION_YAML = """\
defaults:
  models:
    - "openai/gpt-4o"
  prompt_template: "prompt.md"
  response_format: "json"
  validators:
    - type: json_schema
      schema: "schema.json"
  correction_prompt: "correction.md"
  max_corrections: 3

tasks:
  - id: "t1"
    params:
      text: "hello"
  - id: "t2"
    params:
      text: "world"
    response_format: "text"
    validators: []
    max_corrections: 0
"""


def _write_yaml(tmp_path: Path, content: str, name: str = "config.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Tests for :func:`load_config`."""

    def test_minimal_config(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(tmp_path, MINIMAL_YAML)
        config = load_config(cfg_path)

        assert config.name == "unnamed"
        assert len(config.tasks) == 1
        assert config.tasks[0].id == "t1"
        assert config.tasks[0].models == ["openai/gpt-4o"]
        assert config.tasks[0].params == {"text": "hello"}

    def test_minimal_config_defaults_for_new_fields(self, tmp_path: Path) -> None:
        """New fields get sensible defaults when not specified."""
        cfg_path = _write_yaml(tmp_path, MINIMAL_YAML)
        config = load_config(cfg_path)

        t = config.tasks[0]
        assert t.response_format == "text"
        assert t.validators == []
        assert t.correction_prompt is None
        assert t.max_corrections == 0

    def test_per_task_overrides(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(tmp_path, OVERRIDE_YAML)
        config = load_config(cfg_path)

        assert config.max_concurrency == 5
        assert config.ignore_unused_params is True

        t1 = config.tasks[0]
        assert t1.models == ["openai/gpt-4o"]
        assert t1.prompt_template == "default.md"
        assert t1.system_prompt == "default system"
        assert t1.max_completion_tokens == 1000
        assert t1.timeout == 60

        t2 = config.tasks[1]
        assert t2.models == ["anthropic/claude-sonnet-4-20250514"]
        assert t2.prompt_template == "override.md"
        assert t2.system_prompt == "override system"
        assert t2.max_completion_tokens == 9999
        assert t2.timeout == 999

    def test_validation_fields_from_defaults(self, tmp_path: Path) -> None:
        """Validation pipeline fields propagate from defaults to tasks."""
        cfg_path = _write_yaml(tmp_path, VALIDATION_YAML)
        config = load_config(cfg_path)

        t1 = config.tasks[0]
        assert t1.response_format == "json"
        assert len(t1.validators) == 1
        assert t1.validators[0]["type"] == "json_schema"
        assert t1.correction_prompt == "correction.md"
        assert t1.max_corrections == 3

    def test_validation_fields_per_task_override(self, tmp_path: Path) -> None:
        """Per-task overrides take precedence over defaults."""
        cfg_path = _write_yaml(tmp_path, VALIDATION_YAML)
        config = load_config(cfg_path)

        t2 = config.tasks[1]
        assert t2.response_format == "text"
        assert t2.validators == []
        assert t2.max_corrections == 0

    def test_all_models_deduplication(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(tmp_path, OVERRIDE_YAML)
        config = load_config(cfg_path)
        models = config.all_models
        assert len(models) == len(set(models))
        assert "openai/gpt-4o" in models
        assert "anthropic/claude-sonnet-4-20250514" in models

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_empty_config_raises(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(tmp_path, "")
        with pytest.raises(ValueError, match="empty"):
            load_config(cfg_path)

    def test_no_tasks_raises(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(tmp_path, "defaults:\n  models: ['x']\n")
        with pytest.raises(ValueError, match="at least one task"):
            load_config(cfg_path)

    def test_duplicate_id_raises(self, tmp_path: Path) -> None:
        yaml_content = """\
defaults:
  models: ["m"]
  prompt_template: "p.md"
tasks:
  - id: "dup"
    params: {}
  - id: "dup"
    params: {}
"""
        cfg_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ValueError, match="Duplicate"):
            load_config(cfg_path)

    def test_missing_id_raises(self, tmp_path: Path) -> None:
        yaml_content = """\
defaults:
  models: ["m"]
  prompt_template: "p.md"
tasks:
  - params: {}
"""
        cfg_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ValueError, match="missing a required 'id'"):
            load_config(cfg_path)

    def test_temperature_default_is_none(self, tmp_path: Path) -> None:
        """Temperature defaults to None (provider default)."""
        cfg_path = _write_yaml(tmp_path, MINIMAL_YAML)
        config = load_config(cfg_path)
        assert config.tasks[0].temperature is None

    def test_temperature_from_defaults(self, tmp_path: Path) -> None:
        yaml_content = """\
defaults:
  models: ["m"]
  prompt_template: "p.md"
  temperature: 0.7
tasks:
  - id: "t1"
    params: {}
"""
        cfg_path = _write_yaml(tmp_path, yaml_content)
        config = load_config(cfg_path)
        assert config.tasks[0].temperature == 0.7

    def test_temperature_per_task_override(self, tmp_path: Path) -> None:
        yaml_content = """\
defaults:
  models: ["m"]
  prompt_template: "p.md"
  temperature: 0.7
tasks:
  - id: "t1"
    params: {}
    temperature: 0.0
"""
        cfg_path = _write_yaml(tmp_path, yaml_content)
        config = load_config(cfg_path)
        assert config.tasks[0].temperature == 0.0

    def test_repeat_default_is_one(self, tmp_path: Path) -> None:
        """Repeat defaults to 1."""
        cfg_path = _write_yaml(tmp_path, MINIMAL_YAML)
        config = load_config(cfg_path)
        assert config.tasks[0].repeat == 1

    def test_repeat_from_defaults(self, tmp_path: Path) -> None:
        yaml_content = """\
defaults:
  models: ["m"]
  prompt_template: "p.md"
  repeat: 5
tasks:
  - id: "t1"
    params: {}
"""
        cfg_path = _write_yaml(tmp_path, yaml_content)
        config = load_config(cfg_path)
        assert config.tasks[0].repeat == 5

    def test_repeat_per_task_override(self, tmp_path: Path) -> None:
        yaml_content = """\
defaults:
  models: ["m"]
  prompt_template: "p.md"
  repeat: 5
tasks:
  - id: "t1"
    params: {}
    repeat: 3
"""
        cfg_path = _write_yaml(tmp_path, yaml_content)
        config = load_config(cfg_path)
        assert config.tasks[0].repeat == 3
