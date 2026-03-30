"""Asynchronous LLM execution engine.

Orchestrates the full experiment lifecycle: loading configuration,
running pre-flight checks, dispatching concurrent LLM API calls via
``litellm.acompletion``, collecting results, and writing reports.

Supports an optional response-processing and validation pipeline with
a multi-turn correction loop:

1. The raw LLM response is processed (e.g. JSON extraction).
2. Processed data is run through a chain of validators.
3. On validation failure, a correction prompt (containing the error
   report) is appended to the conversation and the LLM is called again.
4. Steps 1–3 repeat up to ``max_corrections`` times.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import litellm
from dotenv import load_dotenv

from .config import ExperimentConfig, TaskConfig, load_config
from .preflight import (
    PreflightReport,
    print_preflight_report,
    run_preflight,
)
from .prompt import assemble_prompt, load_template, resolve_param_value, resolve_params
from .report import generate_failed_yaml, save_report_yaml
from .validation import (
    ValidatorFn,
    load_processor,
    load_validators,
    run_validators,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_output(
    path: Path,
    raw_content: str,
    processed: Any,
    response_format: str,
) -> None:
    """Persist task output to disk.

    For ``"json"`` format the processed (parsed) data is pretty-printed.
    For ``"text"`` format the raw LLM response is saved as-is.

    :param path: Destination file path.
    :param raw_content: The raw LLM response string.
    :param processed: The processed/validated data.
    :param response_format: ``"json"`` or ``"text"``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if response_format == "json" and isinstance(processed, (dict, list)):
        path.write_text(
            json.dumps(processed, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        path.write_text(raw_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Single (task, model) execution
# ---------------------------------------------------------------------------

async def _execute_single(
    task: TaskConfig,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    *,
    run_number: int = 1,
    base_dir: Optional[Path] = None,
    system_prompt_content: Optional[str] = None,
    processor: Callable[[str], Any],
    validators: List[ValidatorFn],
    correction_template: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a single LLM API call for one (task, model) pair.

    If validators are configured and a correction template is available,
    validation failures trigger a multi-turn correction loop (up to
    ``task.max_corrections`` additional turns).

    :param task: The task configuration.
    :param model: The LiteLLM model identifier.
    :param prompt: The fully-assembled user prompt string.
    :param semaphore: Semaphore controlling maximum concurrency.
    :param run_number: The run index (1-based) for repeat runs.
    :param base_dir: Base directory from the experiment config, passed
        through to the validator context so custom validators can
        resolve file paths.
    :param system_prompt_content: Resolved system prompt string, or ``None``.
    :param processor: Response processor callable (e.g. JSON extractor).
    :param validators: Pre-loaded list of validator callables.
    :param correction_template: Loaded correction prompt template string
        with ``{{ERROR}}`` placeholder, or ``None``.
    :returns: A dict containing the result metadata:

        - ``task_id`` -- the task identifier.
        - ``model`` -- the model identifier.
        - ``run_number`` -- the run index (1-based).
        - ``status`` -- ``"success"`` or ``"failed"``.
        - ``output_path`` -- path to the saved output (on success or
          validation failure).
        - ``error_type`` / ``error`` -- error details (on failure).
        - ``duration_s`` -- wall-clock seconds for the full execution
          (including all correction turns).
        - ``usage`` -- aggregated token usage dict across all turns.
        - ``corrections_attempted`` -- number of correction turns used.
    """
    # Derive a filesystem-safe model label.
    model_label = model.replace("/", "_").replace(":", "_")
    output_dir = Path(task.output_dir)
    ext = ".json" if task.response_format == "json" else ".txt"
    if task.repeat > 1:
        output_path = output_dir / f"{task.id}_{model_label}_{run_number:03d}{ext}"
    else:
        output_path = output_dir / f"{task.id}_{model_label}{ext}"

    # Build initial message history.
    messages: List[Dict[str, str]] = []
    if system_prompt_content:
        messages.append({"role": "system", "content": system_prompt_content})
    messages.append({"role": "user", "content": prompt})

    completion_kwargs: Dict[str, Any] = {}
    if task.timeout is not None:
        completion_kwargs["timeout"] = task.timeout
    if task.temperature is not None:
        completion_kwargs["temperature"] = task.temperature

    corrections_attempted = 0
    total_usage: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    max_attempts = 1 + task.max_corrections

    async with semaphore:
        start = time.monotonic()

        for attempt in range(max_attempts):
            # -- LLM API call -------------------------------------------------
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    max_completion_tokens=task.max_completion_tokens,
                    **completion_kwargs,
                )
            except Exception as exc:
                duration = time.monotonic() - start
                return {
                    "task_id": task.id,
                    "model": model,
                    "status": "failed",
                    "error_type": "api_error",
                    "error": str(exc),
                    "duration_s": round(duration, 2),
                    "corrections_attempted": corrections_attempted,
                    "usage": total_usage,
                }

            content = response.choices[0].message.content or ""

            # Accumulate token usage.
            if hasattr(response, "usage") and response.usage is not None:
                for key in total_usage:
                    val = getattr(response.usage, key, None)
                    if val is not None:
                        total_usage[key] += val

            # -- Response processing ------------------------------------------
            try:
                processed = processor(content)
            except Exception as exc:
                error_msg = f"Response processing error: {exc}"
                # Can we correct?
                if (
                    attempt < max_attempts - 1
                    and correction_template is not None
                ):
                    corrections_attempted += 1
                    correction = assemble_prompt(
                        correction_template, {"ERROR": error_msg}
                    )
                    messages.append(
                        {"role": "assistant", "content": content}
                    )
                    messages.append({"role": "user", "content": correction})
                    continue

                # No corrections left — fail.
                duration = time.monotonic() - start
                _save_output(
                    output_path, content, content, task.response_format
                )
                return {
                    "task_id": task.id,
                    "model": model,
                    "status": "failed",
                    "error_type": "processing_error",
                    "error": error_msg,
                    "output_path": str(output_path),
                    "duration_s": round(duration, 2),
                    "corrections_attempted": corrections_attempted,
                    "usage": total_usage,
                }

            # -- Validation ---------------------------------------------------
            if validators:
                context = {
                    "task_id": task.id,
                    "params": task.params,
                    "model": model,
                    "attempt": attempt + 1,
                    "run_number": run_number,
                    "base_dir": str(base_dir) if base_dir else None,
                }
                vresult = run_validators(validators, processed, context)

                if not vresult.success:
                    # Can we correct?
                    if (
                        attempt < max_attempts - 1
                        and correction_template is not None
                    ):
                        corrections_attempted += 1
                        correction = assemble_prompt(
                            correction_template,
                            {"ERROR": vresult.error or "Unknown error"},
                        )
                        messages.append(
                            {"role": "assistant", "content": content}
                        )
                        messages.append(
                            {"role": "user", "content": correction}
                        )
                        continue

                    # No corrections left — fail (but still save output).
                    duration = time.monotonic() - start
                    _save_output(
                        output_path, content, processed, task.response_format
                    )
                    return {
                        "task_id": task.id,
                        "model": model,
                        "status": "failed",
                        "error_type": "validation_error",
                        "error": vresult.error,
                        "output_path": str(output_path),
                        "duration_s": round(duration, 2),
                        "corrections_attempted": corrections_attempted,
                        "usage": total_usage,
                    }

                # Validators may transform data.
                processed = vresult.data

            # -- Success! -----------------------------------------------------
            duration = time.monotonic() - start
            _save_output(output_path, content, processed, task.response_format)

            return {
                "task_id": task.id,
                "model": model,
                "status": "success",
                "output_path": str(output_path),
                "duration_s": round(duration, 2),
                "corrections_attempted": corrections_attempted,
                "usage": total_usage,
            }

    # Should not be reachable, but handle gracefully.
    duration = time.monotonic() - start
    return {
        "task_id": task.id,
        "model": model,
        "status": "failed",
        "error_type": "internal_error",
        "error": "Unexpected end of execution loop",
        "duration_s": round(duration, 2),
        "corrections_attempted": corrections_attempted,
        "usage": total_usage,
    }


# ---------------------------------------------------------------------------
# Async experiment runner
# ---------------------------------------------------------------------------

async def _run_experiment_async(
    config: ExperimentConfig,
    *,
    skip_preflight: bool = False,
    model_timeout: int = 30,
) -> Dict[str, Any]:
    """Run the experiment asynchronously (internal implementation).

    :param config: The fully-resolved experiment configuration.
    :param skip_preflight: If ``True``, skip all pre-flight checks.
    :param model_timeout: Default timeout in seconds for each LLM API call.
    :returns: Experiment results dictionary.
    """
    results: Dict[str, Any] = {
        "experiment": config.name,
        "description": config.description,
        "config_file": str(config.config_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "tasks": [],
        "successes": 0,
        "failures": 0,
    }

    # -- Pre-flight checks ----------------------------------------------------
    if not skip_preflight:
        print(f"Starting experiment: {config.name}")
        print(f"Running pre-flight checks...")

        report = run_preflight(
            config,
            skip_model_check=False,
        )
        print_preflight_report(report)

        if report.has_errors:
            results["status"] = "aborted"
            results["reason"] = "preflight_check_failed"
            return results

        if report.has_warnings:
            if config.ignore_unused_params:
                print("  Continuing (ignore_unused_params is set).\n")
            else:
                try:
                    answer = input(
                        "  Continue despite warnings? [y/N] "
                    ).strip().lower()
                except EOFError:
                    answer = "n"
                if answer not in ("y", "yes"):
                    results["status"] = "aborted"
                    results["reason"] = "user_cancelled_on_warnings"
                    return results
                print()

    # -- Build prompts and dispatch -------------------------------------------
    semaphore = asyncio.Semaphore(config.max_concurrency)

    # Pre-load and cache templates, processors, validators.
    template_cache: Dict[str, str] = {}
    processor_cache: Dict[str, Callable] = {}
    validator_cache: Dict[str, List[ValidatorFn]] = {}
    correction_cache: Dict[str, str] = {}

    coroutines: List[asyncio.Task[Dict[str, Any]]] = []

    for task in config.tasks:
        # -- Prompt template --------------------------------------------------
        template_path = Path(task.prompt_template)
        if not template_path.is_absolute():
            template_path = config.base_dir / template_path
        cache_key = str(template_path)
        if cache_key not in template_cache:
            template_cache[cache_key] = load_template(template_path)

        template = template_cache[cache_key]
        resolved = resolve_params(task.params, config.base_dir)
        prompt = assemble_prompt(template, resolved)

        # -- System prompt (resolve file: prefix + placeholder substitution) --
        system_prompt_content: Optional[str] = None
        if task.system_prompt:
            raw_system = resolve_param_value(
                task.system_prompt, config.base_dir
            )
            system_prompt_content = assemble_prompt(raw_system, resolved)

        # -- Response processor -----------------------------------------------
        fmt = task.response_format
        if fmt not in processor_cache:
            processor_cache[fmt] = load_processor(fmt)
        processor = processor_cache[fmt]

        # -- Validators -------------------------------------------------------
        v_key = repr(task.validators)  # hashable cache key
        if v_key not in validator_cache:
            validator_cache[v_key] = load_validators(
                task.validators, config.base_dir
            )
        validators = validator_cache[v_key]

        # -- Correction prompt template ---------------------------------------
        correction_template: Optional[str] = None
        if task.correction_prompt and task.max_corrections > 0:
            cp_path = Path(task.correction_prompt)
            if not cp_path.is_absolute():
                cp_path = config.base_dir / cp_path
            cp_key = str(cp_path)
            if cp_key not in correction_cache:
                correction_cache[cp_key] = load_template(cp_path)
            correction_template = correction_cache[cp_key]

        # -- Dispatch coroutines per (model, run) pair ------------------------
        for model in task.models:
            for run_num in range(1, task.repeat + 1):
                coro = _execute_single(
                    task,
                    model,
                    prompt,
                    semaphore,
                    run_number=run_num,
                    base_dir=config.base_dir,
                    system_prompt_content=system_prompt_content,
                    processor=processor,
                    validators=validators,
                    correction_template=correction_template,
                )
                coroutines.append(coro)

    total = len(coroutines)
    print(f"Dispatching {total} API call(s) "
          f"(max concurrency: {config.max_concurrency})...\n")

    raw_results: List[Dict[str, Any]] = await asyncio.gather(*coroutines)

    # -- Aggregate results by task --------------------------------------------
    task_results: Dict[str, Dict[str, Any]] = {}
    for task in config.tasks:
        task_results[task.id] = {
            "id": task.id,
            "models": [],
        }

    for r in raw_results:
        tid = r["task_id"]
        task_results[tid]["models"].append(r)
        if r["status"] == "success":
            results["successes"] += 1
        else:
            results["failures"] += 1

    results["tasks"] = list(task_results.values())
    results["completed_at"] = datetime.now(timezone.utc).isoformat()

    # -- Console summary ------------------------------------------------------
    print("=" * 60)
    print(f"Experiment complete: {config.name}")
    print(
        f"Successes: {results['successes']}, "
        f"Failures: {results['failures']}"
    )
    print("=" * 60)

    for task_block in results["tasks"]:
        print(f"\n  {task_block['id']}:")
        for model_result in task_block["models"]:
            model_name = model_result["model"]
            corrections = model_result.get("corrections_attempted", 0)
            corr_info = f" [{corrections} correction(s)]" if corrections else ""
            if model_result["status"] == "success":
                print(
                    f"    [OK]   {model_name}  "
                    f"({model_result['duration_s']}s){corr_info} "
                    f"-> {model_result['output_path']}"
                )
            else:
                print(
                    f"    [FAIL] {model_name}  "
                    f"({model_result['duration_s']}s){corr_info} "
                    f"{model_result.get('error_type', 'unknown')}: "
                    f"{model_result.get('error', 'no details')}"
                )

    # -- Save reports ---------------------------------------------------------
    # Determine report output directory from the first task or base_dir.
    report_dir = config.base_dir
    if config.tasks:
        first_output = Path(config.tasks[0].output_dir)
        report_dir = first_output.parent

    report_path = save_report_yaml(results, report_dir, config.config_path.name)
    results["report_file"] = str(report_path)
    print(f"\nReport saved -> {report_path}")

    # Generate failed-runs config if needed.
    if results["failures"] > 0:
        failed_path = generate_failed_yaml(config, results, config.config_path)
        if failed_path:
            results["failed_config"] = str(failed_path)
            print(f"Failed runs config saved -> {failed_path}")

    return results


# ---------------------------------------------------------------------------
# Public synchronous entry point
# ---------------------------------------------------------------------------

def run_experiment(
    config_path: Union[str, Path],
    *,
    base_dir: Optional[Path] = None,
    skip_preflight: bool = False,
    model_timeout: int = 30,
) -> Dict[str, Any]:
    """Run an experiment from a YAML configuration file.

    This is the main entry point.  It loads the configuration, runs
    pre-flight checks, dispatches all LLM calls concurrently, and
    saves a YAML report.

    :param config_path: Path to the YAML configuration file.
    :param base_dir: Base directory for resolving relative paths.
        Defaults to the parent directory of *config_path*.
    :param skip_preflight: If ``True``, skip all pre-flight checks.
    :param model_timeout: Default timeout in seconds for each LLM API call.
    :returns: Experiment results dictionary with per-task, per-model
        status, timing, and token usage information.
    """
    # Load .env for API keys.
    load_dotenv()

    config = load_config(config_path, base_dir=base_dir)

    return asyncio.run(
        _run_experiment_async(
            config,
            skip_preflight=skip_preflight,
            model_timeout=model_timeout,
        )
    )


async def run_experiment_async(
    config: ExperimentConfig,
    *,
    skip_preflight: bool = False,
    model_timeout: int = 30,
) -> Dict[str, Any]:
    """Run an experiment from a pre-loaded configuration (async version).

    This is the programmatic async entry point for use in notebooks or
    async applications.  Unlike :func:`run_experiment`, it accepts an
    already-loaded :class:`ExperimentConfig` and does not call
    ``asyncio.run()``.

    :param config: A fully-resolved experiment configuration.
    :param skip_preflight: If ``True``, skip all pre-flight checks.
    :param model_timeout: Default timeout in seconds for each LLM API call.
    :returns: Experiment results dictionary.
    """
    return await _run_experiment_async(
        config,
        skip_preflight=skip_preflight,
        model_timeout=model_timeout,
    )
