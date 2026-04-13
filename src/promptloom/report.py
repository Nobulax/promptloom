"""Experiment report generation.

Provides functions to save structured YAML reports after an experiment
run and to generate a YAML configuration file containing only the
failed tasks for convenient re-runs.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .config import ExperimentConfig


class _NoAliasDumper(yaml.SafeDumper):
    """YAML dumper that never emits anchors/aliases for repeated objects.

    The default :class:`yaml.SafeDumper` detects when the same Python
    object appears in multiple places and serialises subsequent
    occurrences as YAML aliases (``*idNNN``).  This is valid YAML but
    produces confusing output when shared objects (e.g. a common
    ``validators`` list) are repeated across tasks.
    """

    def ignore_aliases(self, data: Any) -> bool:  # noqa: D401, ARG002
        return True


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
            Dumper=_NoAliasDumper,
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

    The generated file mirrors the structure of the original config —
    preserving the ``defaults`` section and per-task overrides — but with
    tasks and models pruned to only those that failed.  Tasks where all
    models succeeded are removed entirely.

    :param config: The original experiment configuration (must carry a
        ``_raw_yaml`` snapshot from :func:`load_config`).
    :param results: The experiment results dictionary.
    :param config_path: Path to the original config file, used to derive
        the output filename.
    :returns: Path to the generated failed-runs YAML, or ``None`` if
        there were no failures.
    """
    config_path = Path(config_path)

    # -- Collect failed (task_id → [model, …]) pairs -------------------------
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

    # -- Build the failed config from a snapshot of the original YAML ---------
    raw = copy.deepcopy(config._raw_yaml)

    # Update experiment metadata.
    raw.setdefault("experiment", {})
    orig_name = raw["experiment"].get("name", "unnamed")
    raw["experiment"]["name"] = f"{orig_name} (failed re-run)"
    raw["experiment"]["description"] = (
        f"Re-run of failed tasks from {config_path.name}"
    )

    # Remove ``models`` from defaults — each task will carry its own
    # explicit list of failed models, so a default would be misleading.
    if "defaults" in raw and "models" in raw["defaults"]:
        del raw["defaults"]["models"]

    # Filter tasks: keep only those with failures, override their models.
    filtered_tasks: List[Dict[str, Any]] = []
    for task_entry in raw.get("tasks", []):
        task_id = task_entry.get("id")
        if task_id in failed_by_task:
            task_copy = dict(task_entry)
            task_copy["models"] = failed_by_task[task_id]
            filtered_tasks.append(task_copy)

    raw["tasks"] = filtered_tasks

    # -- Write ----------------------------------------------------------------
    failed_path = config_path.with_name(
        f"{config_path.stem}_failed{config_path.suffix}"
    )
    with open(failed_path, "w", encoding="utf-8") as fh:
        yaml.dump(
            raw,
            fh,
            Dumper=_NoAliasDumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    return failed_path
