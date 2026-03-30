"""YAML configuration loading and validation.

Handles parsing the experiment YAML file, merging per-task overrides
with global defaults, and producing typed configuration objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml



@dataclass
class TaskConfig:
    """Configuration for a single task in the experiment.

    :param id: Unique identifier for this task.
    :param params: Arbitrary key-value parameters to substitute into the
        prompt template.  Values prefixed with ``file:`` are resolved to
        file contents at prompt-assembly time.
    :param models: List of LiteLLM model identifiers to run this task on.
    :param prompt_template: Path to the prompt template file (relative to
        *base_dir* or absolute).
    :param system_prompt: Optional system-level prompt sent before the user
        message.  Can be a literal string or a ``file:`` reference to a
        template file (which also supports ``{{PLACEHOLDER}}`` substitution).
    :param output_dir: Directory where outputs for this task are written.
    :param max_completion_tokens: Maximum number of tokens in the LLM
        response.
    :param timeout: Timeout in seconds for each LLM API call.  ``None``
        means no timeout.
    :param response_format: Response processing format.  ``"text"`` (default)
        keeps the raw response; ``"json"`` extracts and parses JSON.
    :param validators: List of validator specification dicts.  Each dict must
        have a ``type`` key (``"json_schema"`` or ``"custom"``).  See
        :mod:`promptloom.validation` for details.
    :param correction_prompt: Path to a correction prompt template file
        (relative to *base_dir* or absolute).  Must contain a ``{{ERROR}}``
        placeholder that will be replaced with the validation error message.
        Only used when *max_corrections* > 0.
    :param max_corrections: Maximum number of correction turns when
        validation fails.  ``0`` (default) disables the correction loop.
    :param temperature: Sampling temperature for the LLM.  ``None``
        (default) uses the provider's default.
    :param repeat: Number of independent runs per (task, model) pair.
        ``1`` (default) means a single run.  Values > 1 produce
        numbered output files (e.g. ``task_model_001.json``).
    """

    id: str
    params: Dict[str, str]
    models: List[str]
    prompt_template: str
    system_prompt: Optional[str]
    output_dir: str
    max_completion_tokens: int
    timeout: Optional[int]
    response_format: str
    validators: List[Dict[str, Any]]
    correction_prompt: Optional[str]
    max_corrections: int
    temperature: Optional[float]
    repeat: int


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    :param name: Human-readable experiment name.
    :param description: Optional longer description.
    :param tasks: List of fully-resolved task configurations.
    :param max_concurrency: Maximum number of parallel LLM API calls.
    :param ignore_unused_params: If ``True``, unused parameters in a task
        (i.e. params not referenced by any placeholder in the prompt
        template) do not trigger an interactive confirmation prompt.
    :param base_dir: Directory used to resolve relative paths in the config.
    :param config_path: Path to the original YAML config file.
    """

    name: str
    description: str
    tasks: List[TaskConfig]
    max_concurrency: int
    ignore_unused_params: bool
    base_dir: Path
    config_path: Path

    @property
    def all_models(self) -> List[str]:
        """Return a deduplicated list of all models across all tasks.

        :returns: Sorted list of unique model identifiers.
        :rtype: list[str]
        """
        seen: set[str] = set()
        result: List[str] = []
        for task in self.tasks:
            for model in task.models:
                if model not in seen:
                    seen.add(model)
                    result.append(model)
        return result


def load_config(
    config_path: Union[str, Path],
    *,
    base_dir: Optional[Path] = None,
) -> ExperimentConfig:
    """Load and validate an experiment YAML configuration file.

    Global defaults are merged with per-task overrides.  Per-task values
    take precedence over defaults for every field that supports overrides.

    :param config_path: Path to the YAML configuration file.
    :param base_dir: Base directory for resolving relative paths.  Defaults
        to the parent directory of *config_path*.
    :returns: A fully-resolved :class:`ExperimentConfig`.
    :raises FileNotFoundError: If *config_path* does not exist.
    :raises ValueError: If required fields are missing or invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_dir = base_dir or config_path.parent

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    if not raw:
        raise ValueError("Config file is empty.")

    # -- Experiment metadata --------------------------------------------------
    experiment_block = raw.get("experiment", {})
    experiment_name: str = experiment_block.get("name", "unnamed")
    experiment_desc: str = experiment_block.get("description", "")

    # -- Global defaults ------------------------------------------------------
    defaults: Dict[str, Any] = raw.get("defaults", {})
    default_models: List[str] = defaults.get("models", [])
    default_prompt_template: str = defaults.get("prompt_template", "")
    default_system_prompt: Optional[str] = defaults.get("system_prompt")
    default_output_dir: str = defaults.get("output_dir", "output")
    default_max_tokens: int = defaults.get("max_completion_tokens", 64_000)
    default_timeout: Optional[int] = defaults.get("timeout")
    default_max_concurrency: int = defaults.get("max_concurrency", 10)
    default_ignore_unused: bool = defaults.get("ignore_unused_params", False)
    default_response_format: str = defaults.get("response_format", "text")
    default_validators: List[Dict[str, Any]] = defaults.get("validators", [])
    default_correction_prompt: Optional[str] = defaults.get(
        "correction_prompt"
    )
    default_max_corrections: int = defaults.get("max_corrections", 0)
    default_temperature: Optional[float] = defaults.get("temperature")
    default_repeat: int = defaults.get("repeat", 1)

    # -- Tasks ----------------------------------------------------------------
    raw_tasks: List[Dict[str, Any]] = raw.get("tasks", [])
    if not raw_tasks:
        raise ValueError("Config must define at least one task in 'tasks'.")

    tasks: List[TaskConfig] = []
    seen_ids: set[str] = set()

    for idx, entry in enumerate(raw_tasks):
        task_id = entry.get("id")
        if not task_id:
            raise ValueError(
                f"Task at index {idx} is missing a required 'id' field."
            )
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task id: '{task_id}'.")
        seen_ids.add(task_id)

        params: Dict[str, str] = entry.get("params", {})

        models = entry.get("models", default_models)
        if not models:
            raise ValueError(
                f"Task '{task_id}' has no models and no default models are set."
            )

        prompt_template = entry.get("prompt_template", default_prompt_template)
        if not prompt_template:
            raise ValueError(
                f"Task '{task_id}' has no prompt_template and no default is set."
            )

        system_prompt = entry.get("system_prompt", default_system_prompt)

        output_dir = entry.get("output_dir")
        if not output_dir:
            resolved_base = Path(default_output_dir)
            if not resolved_base.is_absolute():
                resolved_base = base_dir / resolved_base
            output_dir = str(resolved_base / task_id)
        else:
            od = Path(output_dir)
            if not od.is_absolute():
                output_dir = str(base_dir / od)

        max_tokens = entry.get("max_completion_tokens", default_max_tokens)
        timeout = entry.get("timeout", default_timeout)
        response_format = entry.get(
            "response_format", default_response_format
        )
        validators = entry.get("validators", default_validators)
        correction_prompt = entry.get(
            "correction_prompt", default_correction_prompt
        )
        max_corrections = entry.get(
            "max_corrections", default_max_corrections
        )
        temperature = entry.get("temperature", default_temperature)
        repeat = entry.get("repeat", default_repeat)

        tasks.append(
            TaskConfig(
                id=task_id,
                params=params,
                models=models,
                prompt_template=prompt_template,
                system_prompt=system_prompt,
                output_dir=output_dir,
                max_completion_tokens=max_tokens,
                timeout=timeout,
                response_format=response_format,
                validators=validators,
                correction_prompt=correction_prompt,
                max_corrections=max_corrections,
                temperature=temperature,
                repeat=repeat,
            )
        )

    return ExperimentConfig(
        name=experiment_name,
        description=experiment_desc,
        tasks=tasks,
        max_concurrency=default_max_concurrency,
        ignore_unused_params=default_ignore_unused,
        base_dir=base_dir,
        config_path=config_path,
    )
