"""A toolkit for batch LLM API calls driven by YAML configuration.

Provides a pipeline to send prompts -- assembled from templates and
arbitrary parameters -- to multiple LLM providers concurrently via
LiteLLM, with pre-flight checks, structured reporting, automatic
failed-run config generation, and an optional response validation /
correction loop.

Typical usage (CLI)::

    promptloom run experiments/config.yaml

Typical usage (Python)::

    from promptloom import load_config, run_experiment

    results = run_experiment("experiments/config.yaml")

Programmatic usage with custom validators::

    from promptloom import load_config, run_experiment_async
    from promptloom.validation import ValidationResult

    def my_validator(data, context):
        if not data.get("nodes"):
            return ValidationResult.fail("No nodes in output")
        return ValidationResult.ok(data)

    config = load_config("experiments/config.yaml")
    # validators can be added programmatically too
    results = await run_experiment_async(config)
"""

from .config import ExperimentConfig, TaskConfig, load_config
from .runner import run_experiment, run_experiment_async
from .validation import ValidationResult

__all__ = [
    "ExperimentConfig",
    "TaskConfig",
    "ValidationResult",
    "load_config",
    "run_experiment",
    "run_experiment_async",
]
