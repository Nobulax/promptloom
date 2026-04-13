"""Tests for report generation (save_report_yaml, generate_failed_yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from promptloom.config import load_config
from promptloom.report import generate_failed_yaml, save_report_yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG_YAML = """\
experiment:
  name: "Test Experiment"
  description: "A test"

defaults:
  models:
    - "openai/gpt-4o"
    - "anthropic/claude-sonnet-4-20250514"
  prompt_template: "prompt.md"
  system_prompt: "Be helpful."
  max_concurrency: 5
  response_format: json
  validators:
    - type: custom
      callable: my_pkg.validators.check
  correction_prompt: "correction.md"
  max_corrections: 2

tasks:
  - id: "task_a"
    params:
      text: "hello"
  - id: "task_b"
    params:
      text: "world"
  - id: "task_c"
    params:
      text: "foo"
    models:
      - "openai/gpt-4o"
"""

ALL_FAIL_RESULTS = {
    "tasks": [
        {
            "id": "task_a",
            "models": [
                {"model": "openai/gpt-4o", "status": "error"},
                {"model": "anthropic/claude-sonnet-4-20250514", "status": "error"},
            ],
        },
        {
            "id": "task_b",
            "models": [
                {"model": "openai/gpt-4o", "status": "error"},
                {"model": "anthropic/claude-sonnet-4-20250514", "status": "error"},
            ],
        },
        {
            "id": "task_c",
            "models": [
                {"model": "openai/gpt-4o", "status": "error"},
            ],
        },
    ],
    "failures": 5,
}

PARTIAL_FAIL_RESULTS = {
    "tasks": [
        {
            "id": "task_a",
            "models": [
                {"model": "openai/gpt-4o", "status": "success"},
                {"model": "anthropic/claude-sonnet-4-20250514", "status": "error"},
            ],
        },
        {
            "id": "task_b",
            "models": [
                {"model": "openai/gpt-4o", "status": "success"},
                {"model": "anthropic/claude-sonnet-4-20250514", "status": "success"},
            ],
        },
        {
            "id": "task_c",
            "models": [
                {"model": "openai/gpt-4o", "status": "error"},
            ],
        },
    ],
    "failures": 2,
}

ALL_SUCCESS_RESULTS = {
    "tasks": [
        {
            "id": "task_a",
            "models": [
                {"model": "openai/gpt-4o", "status": "success"},
                {"model": "anthropic/claude-sonnet-4-20250514", "status": "success"},
            ],
        },
        {
            "id": "task_b",
            "models": [
                {"model": "openai/gpt-4o", "status": "success"},
                {"model": "anthropic/claude-sonnet-4-20250514", "status": "success"},
            ],
        },
        {
            "id": "task_c",
            "models": [
                {"model": "openai/gpt-4o", "status": "success"},
            ],
        },
    ],
    "failures": 0,
}


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    """Write the test YAML and return its path."""
    p = tmp_path / "test_config.yaml"
    p.write_text(CONFIG_YAML, encoding="utf-8")
    return p


@pytest.fixture()
def config(config_path: Path):
    """Load the test config."""
    return load_config(config_path)


# ---------------------------------------------------------------------------
# generate_failed_yaml
# ---------------------------------------------------------------------------


class TestGenerateFailedYaml:
    """Tests for generate_failed_yaml."""

    def test_returns_none_when_no_failures(self, config, config_path):
        result = generate_failed_yaml(config, ALL_SUCCESS_RESULTS, config_path)
        assert result is None

    def test_output_file_path(self, config, config_path):
        result = generate_failed_yaml(config, ALL_FAIL_RESULTS, config_path)
        assert result is not None
        assert result.name == "test_config_failed.yaml"
        assert result.parent == config_path.parent

    def test_preserves_defaults_section(self, config, config_path):
        """The failed YAML should keep the original defaults (minus models)."""
        result = generate_failed_yaml(config, ALL_FAIL_RESULTS, config_path)
        failed = yaml.safe_load(result.read_text(encoding="utf-8"))

        defaults = failed["defaults"]
        assert defaults["prompt_template"] == "prompt.md"
        assert defaults["system_prompt"] == "Be helpful."
        assert defaults["max_concurrency"] == 5
        assert defaults["response_format"] == "json"
        assert defaults["correction_prompt"] == "correction.md"
        assert defaults["max_corrections"] == 2
        assert defaults["validators"] == [
            {"type": "custom", "callable": "my_pkg.validators.check"}
        ]

    def test_removes_default_models(self, config, config_path):
        """Default models should be removed since each task has explicit models."""
        result = generate_failed_yaml(config, ALL_FAIL_RESULTS, config_path)
        failed = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert "models" not in failed.get("defaults", {})

    def test_updates_experiment_metadata(self, config, config_path):
        result = generate_failed_yaml(config, ALL_FAIL_RESULTS, config_path)
        failed = yaml.safe_load(result.read_text(encoding="utf-8"))

        assert "(failed re-run)" in failed["experiment"]["name"]
        assert "test_config.yaml" in failed["experiment"]["description"]

    def test_keeps_only_failed_tasks(self, config, config_path):
        """Only tasks with failures should appear in the output."""
        result = generate_failed_yaml(config, PARTIAL_FAIL_RESULTS, config_path)
        failed = yaml.safe_load(result.read_text(encoding="utf-8"))

        task_ids = [t["id"] for t in failed["tasks"]]
        assert "task_a" in task_ids
        assert "task_b" not in task_ids  # all succeeded
        assert "task_c" in task_ids

    def test_keeps_only_failed_models(self, config, config_path):
        """Each task should list only its failed models."""
        result = generate_failed_yaml(config, PARTIAL_FAIL_RESULTS, config_path)
        failed = yaml.safe_load(result.read_text(encoding="utf-8"))

        task_a = next(t for t in failed["tasks"] if t["id"] == "task_a")
        assert task_a["models"] == ["anthropic/claude-sonnet-4-20250514"]

        task_c = next(t for t in failed["tasks"] if t["id"] == "task_c")
        assert task_c["models"] == ["openai/gpt-4o"]

    def test_preserves_per_task_overrides(self, config, config_path):
        """Per-task fields like params should be preserved as-is."""
        result = generate_failed_yaml(config, ALL_FAIL_RESULTS, config_path)
        failed = yaml.safe_load(result.read_text(encoding="utf-8"))

        task_a = next(t for t in failed["tasks"] if t["id"] == "task_a")
        assert task_a["params"] == {"text": "hello"}

        task_b = next(t for t in failed["tasks"] if t["id"] == "task_b")
        assert task_b["params"] == {"text": "world"}

    def test_no_yaml_aliases(self, config, config_path):
        """The output must not contain YAML aliases like *id001."""
        result = generate_failed_yaml(config, ALL_FAIL_RESULTS, config_path)
        content = result.read_text(encoding="utf-8")
        assert "*id" not in content
        assert "&id" not in content

    def test_roundtrip_loadable(self, config, config_path):
        """The generated failed YAML should be loadable by load_config."""
        result = generate_failed_yaml(config, ALL_FAIL_RESULTS, config_path)
        reloaded = load_config(result)

        task_ids = [t.id for t in reloaded.tasks]
        assert "task_a" in task_ids
        assert "task_b" in task_ids
        assert "task_c" in task_ids

    def test_raw_yaml_snapshot_not_mutated(self, config, config_path):
        """generate_failed_yaml should not mutate config._raw_yaml."""
        import copy

        snapshot_before = copy.deepcopy(config._raw_yaml)
        generate_failed_yaml(config, PARTIAL_FAIL_RESULTS, config_path)
        assert config._raw_yaml == snapshot_before


# ---------------------------------------------------------------------------
# save_report_yaml
# ---------------------------------------------------------------------------


class TestSaveReportYaml:
    """Tests for save_report_yaml."""

    def test_creates_report_file(self, tmp_path):
        results = {"tasks": [], "failures": 0}
        path = save_report_yaml(results, tmp_path, "my_config.yaml")
        assert path.exists()
        assert path.name.startswith("my_config_report_")
        assert path.suffix == ".yaml"

    def test_report_content(self, tmp_path):
        results = {"tasks": [{"id": "t1", "status": "ok"}], "failures": 0}
        path = save_report_yaml(results, tmp_path, "cfg.yaml")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded["failures"] == 0
        assert loaded["tasks"][0]["id"] == "t1"

    def test_no_yaml_aliases_in_report(self, tmp_path):
        """Shared objects should be inlined, not aliased."""
        shared_list = [{"key": "value"}]
        results = {
            "a": shared_list,
            "b": shared_list,
        }
        path = save_report_yaml(results, tmp_path, "test.yaml")
        content = path.read_text(encoding="utf-8")
        assert "*id" not in content
        assert "&id" not in content


# ---------------------------------------------------------------------------
# _raw_yaml on ExperimentConfig
# ---------------------------------------------------------------------------


class TestRawYamlSnapshot:
    """Tests for the _raw_yaml field on ExperimentConfig."""

    def test_raw_yaml_populated(self, config):
        assert config._raw_yaml
        assert "experiment" in config._raw_yaml
        assert "tasks" in config._raw_yaml

    def test_raw_yaml_is_independent_copy(self, config_path):
        """Modifying _raw_yaml on one config should not affect another load."""
        config1 = load_config(config_path)
        config2 = load_config(config_path)
        config1._raw_yaml["experiment"]["name"] = "MUTATED"
        assert config2._raw_yaml["experiment"]["name"] == "Test Experiment"
