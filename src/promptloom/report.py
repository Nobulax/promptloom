"""Experiment report generation.

Provides functions to save structured YAML reports after an experiment
run and to generate a YAML configuration file containing only the
failed tasks for convenient re-runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .config import ExperimentConfig


def save_report_yaml(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    config_name: str,
) -> Path:
    """Save the experiment results as a timestamped YAML report.

    :param results: The experiment results dictionary produced by the
        runner.
    :param output_dir: Directory where the report file is written.
    :param config_name: Name of the original config file, used to derive
        the report filename.
    :returns: Path to the saved report YAML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = Path(config_name).stem
    report_name = f"{stem}_report_{timestamp}.yaml"
    report_path = output_dir / report_name

    with open(report_path, "w", encoding="utf-8") as fh:
        yaml.dump(
            results,
            fh,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    return report_path


def generate_failed_yaml(
    config: ExperimentConfig,
    results: Dict[str, Any],
    config_path: Union[str, Path],
) -> Optional[Path]:
    """Generate a YAML config containing only the failed (task, model) pairs.

    The generated file can be passed directly to :func:`run_experiment`
    to re-run only the failures.

    :param config: The original experiment configuration.
    :param results: The experiment results dictionary.
    :param config_path: Path to the original config file, used to derive
        the output filename.
    :returns: Path to the generated failed-runs YAML, or ``None`` if
        there were no failures.
    """
    config_path = Path(config_path)

    # Collect failed (task_id, model) pairs.
    failed_by_task: Dict[str, List[str]] = {}
    for task_block in results.get("tasks", []):
        task_id = task_block["id"]
        for model_result in task_block.get("models", []):
            if model_result["status"] != "success":
                failed_by_task.setdefault(task_id, []).append(
                    model_result["model"]
                )

    if not failed_by_task:
        return None

    # Build a task lookup from the original config.
    task_lookup: Dict[str, Any] = {}
    for task in config.tasks:
        task_lookup[task.id] = task

    # Assemble the failed-runs config.
    failed_tasks: List[Dict[str, Any]] = []
    for task_id, models in failed_by_task.items():
        original = task_lookup.get(task_id)
        if original is None:
            continue
        entry: Dict[str, Any] = {
            "id": original.id,
            "params": dict(original.params),
            "models": models,
        }
        if original.output_dir:
            entry["output_dir"] = original.output_dir
        if original.prompt_template:
            entry["prompt_template"] = original.prompt_template
        if original.system_prompt:
            entry["system_prompt"] = original.system_prompt
        if original.timeout is not None:
            entry["timeout"] = original.timeout
        if original.max_completion_tokens:
            entry["max_completion_tokens"] = original.max_completion_tokens
        if original.response_format != "text":
            entry["response_format"] = original.response_format
        if original.validators:
            entry["validators"] = original.validators
        if original.correction_prompt:
            entry["correction_prompt"] = original.correction_prompt
        if original.max_corrections:
            entry["max_corrections"] = original.max_corrections
        failed_tasks.append(entry)

    failed_config: Dict[str, Any] = {
        "experiment": {
            "name": config.name + " (failed re-run)",
            "description": (
                f"Re-run of failed tasks from {config_path.name}"
            ),
        },
        "defaults": {
            "max_concurrency": config.max_concurrency,
            "ignore_unused_params": config.ignore_unused_params,
        },
        "tasks": failed_tasks,
    }

    failed_path = config_path.with_name(
        f"{config_path.stem}_failed{config_path.suffix}"
    )
    with open(failed_path, "w", encoding="utf-8") as fh:
        yaml.dump(
            failed_config,
            fh,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    return failed_path
