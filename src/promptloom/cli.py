"""Command-line interface for promptloom.

Provides the ``promptloom`` command with subcommands for running
experiments and performing dry-run validation.

Usage::

    promptloom run config.yaml
    promptloom run config.yaml --skip-preflight
    promptloom run config.yaml --dry-run
"""

from __future__ import annotations

from pathlib import Path

import click

from .config import load_config
from .runner import run_experiment


@click.group()
@click.version_option(package_name="promptloom")
def main() -> None:
    """promptloom -- batch LLM API calls from YAML configs."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--skip-preflight",
    is_flag=True,
    default=False,
    help="Skip all pre-flight checks (model connectivity and placeholder validation).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Load and validate the config without making any API calls.",
)
@click.option(
    "--model-timeout",
    type=int,
    default=30,
    show_default=True,
    help="Default timeout in seconds for each LLM API call.",
)
def run(
    config_path: str,
    skip_preflight: bool,
    dry_run: bool,
    model_timeout: int,
) -> None:
    """Run an experiment from a YAML configuration file.

    CONFIG_PATH is the path to the YAML experiment configuration.
    """
    if dry_run:
        _dry_run(config_path)
        return

    results = run_experiment(
        config_path,
        skip_preflight=skip_preflight,
        model_timeout=model_timeout,
    )

    if results.get("status") == "aborted":
        reason = results.get("reason", "unknown")
        raise SystemExit(f"Experiment aborted: {reason}")


def _dry_run(config_path: str) -> None:
    """Load and validate config, print summary, but do not call any APIs.

    :param config_path: Path to the YAML configuration file.
    """
    from dotenv import load_dotenv

    load_dotenv()

    config = load_config(config_path)

    print(f"Experiment: {config.name}")
    if config.description:
        print(f"Description: {config.description}")
    print(f"Tasks: {len(config.tasks)}")
    print(f"Models (unique): {len(config.all_models)}")
    print(f"Max concurrency: {config.max_concurrency}")
    print(f"Ignore unused params: {config.ignore_unused_params}")

    total_calls = sum(len(t.models) for t in config.tasks)
    print(f"Total API calls: {total_calls}")

    print("\nTasks:")
    for task in config.tasks:
        print(f"  {task.id}:")
        print(f"    Template: {task.prompt_template}")
        if task.system_prompt:
            sp_preview = task.system_prompt[:60]
            if len(task.system_prompt) > 60:
                sp_preview += "…"
            print(f"    System prompt: {sp_preview}")
        print(f"    Models: {', '.join(task.models)}")
        print(f"    Params: {list(task.params.keys())}")
        print(f"    Output: {task.output_dir}")
        if task.timeout:
            print(f"    Timeout: {task.timeout}s")
        print(f"    Max tokens: {task.max_completion_tokens}")
        if task.response_format != "text":
            print(f"    Response format: {task.response_format}")
        if task.validators:
            vtypes = [v.get("type", "?") for v in task.validators]
            print(f"    Validators: {', '.join(vtypes)}")
        if task.max_corrections > 0:
            print(f"    Max corrections: {task.max_corrections}")
            if task.correction_prompt:
                print(f"    Correction prompt: {task.correction_prompt}")

    print("\nDry run complete. No API calls were made.")
