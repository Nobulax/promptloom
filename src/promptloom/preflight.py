"""Pre-flight checks for model validation and placeholder validation.

Two independent checks are performed before an experiment run:

1. **Model validation** -- verifies that each unique model name is
   recognised by LiteLLM and that the required API keys / environment
   variables are set for the model's provider.  This check is instant,
   free, and makes **no API calls**.
2. **Placeholder validation** -- ensures that every ``{{PLACEHOLDER}}``
   in each task's prompt template has a corresponding parameter, and
   warns about unused parameters.

Both checks run to completion before an abort decision is made so that
the user sees all problems at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import litellm

from .config import ExperimentConfig
from .prompt import check_placeholders, extract_placeholders, load_template
from .validation import load_processor


# ---------------------------------------------------------------------------
# Data classes for pre-flight results
# ---------------------------------------------------------------------------

@dataclass
class ModelCheckResult:
    """Result of a single model validation check.

    :param model: The LiteLLM model identifier that was checked.
    :param ok: ``True`` if the model name is recognised and the required
        environment variables are set.
    :param error: Error message if the check failed, ``None`` otherwise.
    """

    model: str
    ok: bool
    error: Optional[str] = None


@dataclass
class PlaceholderCheckResult:
    """Result of placeholder validation for a single task.

    :param task_id: Identifier of the task that was checked.
    :param template_path: Path to the prompt template used.
    :param missing: Placeholder names in the template without a
        corresponding parameter in the task.
    :param unused: Parameter names in the task not referenced by any
        placeholder in the template.
    """

    task_id: str
    template_path: str
    missing: List[str] = field(default_factory=list)
    unused: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return ``True`` if there are missing placeholders (fatal)."""
        return len(self.missing) > 0

    @property
    def has_warnings(self) -> bool:
        """Return ``True`` if there are unused parameters (non-fatal)."""
        return len(self.unused) > 0


@dataclass
class ValidationConfigCheckResult:
    """Result of validation pipeline configuration check for a single task.

    :param task_id: Identifier of the task that was checked.
    :param errors: List of fatal configuration errors.
    :param warnings: List of non-fatal configuration warnings.
    """

    task_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return ``True`` if there are fatal errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Return ``True`` if there are non-fatal warnings."""
        return len(self.warnings) > 0


@dataclass
class PreflightReport:
    """Aggregated results of all pre-flight checks.

    :param model_results: Per-model validation check results.
    :param placeholder_results: Per-task placeholder validation results.
    :param validation_config_results: Per-task validation pipeline
        configuration check results.
    """

    model_results: List[ModelCheckResult] = field(default_factory=list)
    placeholder_results: List[PlaceholderCheckResult] = field(
        default_factory=list
    )
    validation_config_results: List[ValidationConfigCheckResult] = field(
        default_factory=list
    )

    @property
    def has_errors(self) -> bool:
        """Return ``True`` if any check produced a fatal error."""
        model_errors = any(not r.ok for r in self.model_results)
        placeholder_errors = any(
            r.has_errors for r in self.placeholder_results
        )
        validation_errors = any(
            r.has_errors for r in self.validation_config_results
        )
        return model_errors or placeholder_errors or validation_errors

    @property
    def has_warnings(self) -> bool:
        """Return ``True`` if any check produced a non-fatal warning."""
        placeholder_warnings = any(
            r.has_warnings for r in self.placeholder_results
        )
        validation_warnings = any(
            r.has_warnings for r in self.validation_config_results
        )
        return placeholder_warnings or validation_warnings

    @property
    def model_error_count(self) -> int:
        """Number of models that failed the validation check."""
        return sum(1 for r in self.model_results if not r.ok)

    @property
    def placeholder_error_count(self) -> int:
        """Number of tasks with missing placeholders."""
        return sum(1 for r in self.placeholder_results if r.has_errors)

    @property
    def validation_config_error_count(self) -> int:
        """Number of tasks with validation pipeline config errors."""
        return sum(
            1 for r in self.validation_config_results if r.has_errors
        )

    @property
    def warning_count(self) -> int:
        """Number of tasks with any warnings."""
        pw = sum(1 for r in self.placeholder_results if r.has_warnings)
        vw = sum(
            1 for r in self.validation_config_results if r.has_warnings
        )
        return pw + vw


# ---------------------------------------------------------------------------
# Model validation check
# ---------------------------------------------------------------------------

def _check_single_model(model: str) -> ModelCheckResult:
    """Validate a model name and its required environment variables.

    Uses :func:`litellm.get_model_info` to verify that the model name
    is recognised by LiteLLM, and :func:`litellm.validate_environment`
    to confirm that the required API keys / env vars are set.

    This check is **instant**, **free**, and makes **no API calls**.

    :param model: LiteLLM model identifier (e.g. ``"gemini/gemini-2.0-flash"``).
    :returns: A :class:`ModelCheckResult` indicating success or failure.
    """
    errors: List[str] = []

    # 1. Verify the model name is recognised by litellm
    try:
        litellm.get_model_info(model)
    except Exception as exc:
        errors.append(f"Unknown model: {exc}")

    # 2. Verify required API keys / env vars are set
    try:
        env_info = litellm.validate_environment(model)
        missing_keys = env_info.get("missing_keys", [])
        if missing_keys:
            errors.append(
                f"Missing environment variable(s): {', '.join(missing_keys)}"
            )
    except Exception as exc:
        errors.append(f"Environment validation error: {exc}")

    if errors:
        return ModelCheckResult(
            model=model, ok=False, error="; ".join(errors)
        )
    return ModelCheckResult(model=model, ok=True)


def check_models(models: List[str]) -> List[ModelCheckResult]:
    """Validate model names and environment for a list of models.

    :param models: List of LiteLLM model identifiers.
    :returns: List of :class:`ModelCheckResult` objects.
    """
    return [_check_single_model(model) for model in models]


# ---------------------------------------------------------------------------
# Placeholder validation
# ---------------------------------------------------------------------------

def check_all_placeholders(
    config: ExperimentConfig,
) -> List[PlaceholderCheckResult]:
    """Validate placeholder coverage for all tasks in the experiment.

    For each task, loads the associated prompt template and checks that
    every ``{{PLACEHOLDER}}`` has a matching key in the task's ``params``
    dict, and reports any params that are not used by the template.

    Templates are cached so that a shared template is only read from disk
    once.

    :param config: The fully-resolved experiment configuration.
    :returns: List of :class:`PlaceholderCheckResult` objects, one per task.
    """
    template_cache: Dict[str, str] = {}
    results: List[PlaceholderCheckResult] = []

    for task in config.tasks:
        template_path = Path(task.prompt_template)
        if not template_path.is_absolute():
            template_path = config.base_dir / template_path

        cache_key = str(template_path)
        if cache_key not in template_cache:
            template_cache[cache_key] = load_template(template_path)

        template = template_cache[cache_key]
        missing, unused = check_placeholders(template, task.params)

        results.append(
            PlaceholderCheckResult(
                task_id=task.id,
                template_path=str(template_path),
                missing=missing,
                unused=unused,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Validation pipeline configuration check
# ---------------------------------------------------------------------------

def check_validation_config(
    config: ExperimentConfig,
) -> List[ValidationConfigCheckResult]:
    """Check the validation pipeline configuration for all tasks.

    Verifies that:

    - ``response_format`` is a recognised value.
    - If ``max_corrections > 0``, a ``correction_prompt`` is specified
      and the file exists.
    - The correction prompt template contains a ``{{ERROR}}`` placeholder.
    - Validator specs reference existing schema files.

    :param config: The fully-resolved experiment configuration.
    :returns: List of :class:`ValidationConfigCheckResult`, one per task
        that has validation features configured.
    """
    results: List[ValidationConfigCheckResult] = []

    for task in config.tasks:
        errors: List[str] = []
        warnings: List[str] = []

        # Check response_format is valid.
        try:
            load_processor(task.response_format)
        except ValueError as exc:
            errors.append(str(exc))

        # Check correction loop configuration.
        if task.max_corrections > 0:
            if not task.correction_prompt:
                errors.append(
                    "max_corrections > 0 but no correction_prompt is set."
                )
            else:
                cp_path = Path(task.correction_prompt)
                if not cp_path.is_absolute():
                    cp_path = config.base_dir / cp_path
                if not cp_path.exists():
                    errors.append(
                        f"Correction prompt file not found: {cp_path}"
                    )
                else:
                    cp_content = cp_path.read_text(encoding="utf-8")
                    placeholders = extract_placeholders(cp_content)
                    if "ERROR" not in {p.upper() for p in placeholders}:
                        warnings.append(
                            f"Correction prompt '{cp_path.name}' does not "
                            f"contain a {{{{ERROR}}}} placeholder."
                        )

        if task.max_corrections > 0 and not task.validators:
            warnings.append(
                "max_corrections > 0 but no validators are configured. "
                "The correction loop will only trigger on processing errors."
            )

        # Check validator specs reference existing files.
        for spec in task.validators:
            vtype = spec.get("type", "")
            if vtype == "json_schema":
                schema_path_str = spec.get("schema")
                if not schema_path_str:
                    errors.append(
                        "json_schema validator is missing a 'schema' key."
                    )
                else:
                    sp = Path(schema_path_str)
                    if not sp.is_absolute():
                        sp = config.base_dir / sp
                    if not sp.exists():
                        errors.append(
                            f"JSON Schema file not found: {sp}"
                        )
            elif vtype == "custom":
                if not spec.get("callable"):
                    errors.append(
                        "custom validator is missing a 'callable' key."
                    )
            elif vtype:
                errors.append(
                    f"Unknown validator type: {vtype!r}"
                )

        if errors or warnings:
            results.append(
                ValidationConfigCheckResult(
                    task_id=task.id,
                    errors=errors,
                    warnings=warnings,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Full pre-flight check
# ---------------------------------------------------------------------------

def run_preflight(
    config: ExperimentConfig,
    *,
    skip_model_check: bool = False,
) -> PreflightReport:
    """Run all pre-flight checks and return an aggregated report.

    Both the model validation check and the placeholder validation
    are executed.  The report contains all results so the caller can
    decide whether to proceed.

    This function is **instant**, **free**, and makes **no API calls**.

    :param config: The experiment configuration.
    :param skip_model_check: If ``True``, skip the model validation
        check entirely.
    :returns: A :class:`PreflightReport` with all check results.
    """
    report = PreflightReport()

    # -- Check 1: model validation --------------------------------------------
    if not skip_model_check:
        models = config.all_models
        if models:
            report.model_results = check_models(models)

    # -- Check 2: placeholder validation --------------------------------------
    report.placeholder_results = check_all_placeholders(config)

    # -- Check 3: validation pipeline configuration ---------------------------
    report.validation_config_results = check_validation_config(config)

    return report


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_preflight_report(report: PreflightReport) -> None:
    """Print a human-readable summary of the pre-flight check results.

    :param report: The :class:`PreflightReport` to display.
    """
    # -- Model results --------------------------------------------------------
    if report.model_results:
        print("\nPre-check 1: Model validation")
        print("-" * 40)
        for result in report.model_results:
            if result.ok:
                print(f"  [PASS] {result.model}")
            else:
                print(f"  [FAIL] {result.model}: {result.error}")

    # -- Placeholder results --------------------------------------------------
    if report.placeholder_results:
        print("\nPre-check 2: Placeholder validation")
        print("-" * 40)
        for result in report.placeholder_results:
            prefix = f"  Task '{result.task_id}'"
            if result.has_errors:
                for name in result.missing:
                    print(
                        f"{prefix} [ERROR] Missing param for "
                        f"placeholder: {{{{{name}}}}}"
                    )
            if result.has_warnings:
                for name in result.unused:
                    print(
                        f"{prefix} [WARN]  Unused param: '{name}'"
                    )
            if not result.has_errors and not result.has_warnings:
                print(f"{prefix} [PASS]  All placeholders resolved")

    # -- Validation config results --------------------------------------------
    if report.validation_config_results:
        print("\nPre-check 3: Validation pipeline config")
        print("-" * 40)
        for result in report.validation_config_results:
            prefix = f"  Task '{result.task_id}'"
            for err in result.errors:
                print(f"{prefix} [ERROR] {err}")
            for warn in result.warnings:
                print(f"{prefix} [WARN]  {warn}")

    # -- Summary --------------------------------------------------------------
    print()
    print("=" * 50)
    total_errors = (
        report.model_error_count
        + report.placeholder_error_count
        + report.validation_config_error_count
    )
    total_warnings = report.warning_count
    print(f"  PRE-CHECK RESULTS")
    print(f"  Errors:   {total_errors}", end="")
    if report.model_error_count:
        print(f" ({report.model_error_count} model)", end="")
    if report.placeholder_error_count:
        print(f" ({report.placeholder_error_count} placeholder)", end="")
    if report.validation_config_error_count:
        print(
            f" ({report.validation_config_error_count} validation config)",
            end="",
        )
    print()
    print(f"  Warnings: {total_warnings}")
    if total_errors:
        print("  --> Run CANCELLED due to errors.")
    elif total_warnings:
        print("  --> Warnings present.")
    else:
        print("  --> All checks passed.")
    print("=" * 50)
