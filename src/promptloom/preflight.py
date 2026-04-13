"""Pre-flight checks for model validation and placeholder validation.

Two independent checks are performed before an experiment run:

1. **Model validation** -- verifies that each unique model name is
   recognised by LiteLLM (local registry) **or** available at the
   provider (remote check), and that the required API keys /
   environment variables are set.  For models not in litellm's static
   registry, a lightweight ``GET /v1/models`` (or equivalent) call is
   made to the provider to confirm availability.  This remote check
   is **free** (no tokens consumed) and **fast** (results are cached
   per provider).
2. **Placeholder validation** -- ensures that every ``{{PLACEHOLDER}}``
   in each task's prompt template has a corresponding parameter, and
   warns about unused parameters.

Both checks run to completion before an abort decision is made so that
the user sees all problems at once.
"""

from __future__ import annotations

import json as _json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

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
    :param ok: ``True`` if the model passed validation (or passed with
        warnings).  ``False`` only for fatal issues (e.g. missing API key).
    :param error: Error message if the check failed, ``None`` otherwise.
    :param warning: Warning message for non-fatal issues (e.g. model not
        in litellm's static registry but may still work at runtime).
    """

    model: str
    ok: bool
    error: Optional[str] = None
    warning: Optional[str] = None


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
        model_warnings = any(
            r.warning for r in self.model_results
        )
        placeholder_warnings = any(
            r.has_warnings for r in self.placeholder_results
        )
        validation_warnings = any(
            r.has_warnings for r in self.validation_config_results
        )
        return model_warnings or placeholder_warnings or validation_warnings

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
    def model_warning_count(self) -> int:
        """Number of models with non-fatal warnings."""
        return sum(1 for r in self.model_results if r.warning)

    @property
    def warning_count(self) -> int:
        """Total number of checks with any warnings."""
        mw = self.model_warning_count
        pw = sum(1 for r in self.placeholder_results if r.has_warnings)
        vw = sum(
            1 for r in self.validation_config_results if r.has_warnings
        )
        return mw + pw + vw


# ---------------------------------------------------------------------------
# Remote provider model-list helpers
# ---------------------------------------------------------------------------

# Module-level cache: provider key → set of model IDs.
_provider_model_cache: Dict[str, Set[str]] = {}

# Timeout (seconds) for the lightweight model-list GET request.
_PROVIDER_LIST_TIMEOUT = 10


def _fetch_openrouter_models(api_key: Optional[str] = None) -> Set[str]:
    """Fetch available model IDs from OpenRouter's ``/api/v1/models``.

    :param api_key: OpenRouter API key (reads ``OPENROUTER_API_KEY`` from
        the environment if not provided).
    :returns: A set of model IDs (e.g. ``{"deepseek/deepseek-v3.2", …}``).
    :raises Exception: On any network / parsing error.
    """
    url = "https://openrouter.ai/api/v1/models"
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=_PROVIDER_LIST_TIMEOUT) as resp:
        data = _json.loads(resp.read().decode())
    return {m["id"] for m in data.get("data", [])}


def _fetch_ollama_models(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Set[str]:
    """Fetch available model names from an Ollama server.

    Tries the ``/api/tags`` endpoint (native Ollama) first, then falls
    back to ``/v1/models`` (OpenAI-compatible shim).

    :param api_base: Ollama server base URL (reads ``OLLAMA_API_BASE``
        from the environment if not provided).
    :param api_key: Ollama API key, if the server requires one (reads
        ``OLLAMA_API_KEY`` from the environment if not provided).
    :returns: A set of model names.
    :raises Exception: On any network / parsing error.
    """
    base = (api_base or os.environ.get("OLLAMA_API_BASE", "")).rstrip("/")
    if not base:
        raise ValueError("OLLAMA_API_BASE not set")
    key = api_key or os.environ.get("OLLAMA_API_KEY", "")

    # Try native Ollama endpoint first.
    for endpoint, extract in [
        (f"{base}/api/tags", lambda d: {m["name"] for m in d.get("models", [])}),
        (f"{base}/v1/models", lambda d: {m["id"] for m in d.get("data", [])}),
    ]:
        try:
            req = urllib.request.Request(endpoint)
            if key:
                req.add_header("Authorization", f"Bearer {key}")
            with urllib.request.urlopen(
                req, timeout=_PROVIDER_LIST_TIMEOUT
            ) as resp:
                data = _json.loads(resp.read().decode())
            return extract(data)
        except (urllib.error.URLError, ValueError, KeyError):
            continue

    raise RuntimeError(f"Could not fetch model list from Ollama at {base}")


def _fetch_provider_models(provider: str) -> Set[str]:
    """Fetch and cache the model list for a known provider.

    Returns a cached result on subsequent calls for the same provider.

    :param provider: The litellm provider prefix (e.g. ``"openrouter"``
        or ``"ollama"``).
    :returns: Set of model identifiers available at the provider.
    :raises Exception: On network / parsing errors (caller should catch).
    """
    if provider in _provider_model_cache:
        return _provider_model_cache[provider]

    if provider == "openrouter":
        models = _fetch_openrouter_models()
    elif provider == "ollama":
        models = _fetch_ollama_models()
    else:
        raise ValueError(f"No model-list fetcher for provider: {provider}")

    _provider_model_cache[provider] = models
    return models


def _check_model_at_provider(model: str) -> Optional[bool]:
    """Check if a model exists at its provider via the model-list API.

    :param model: Full litellm model identifier (e.g.
        ``"openrouter/deepseek/deepseek-v3.2"``).
    :returns:
        - ``True`` if the model was confirmed available.
        - ``False`` if the provider was reachable but the model was
          **not** found.
        - ``None`` if the remote check could not be performed (unknown
          provider, network error, etc.).
    """
    # Supported providers and how to derive the provider model ID.
    if model.startswith("openrouter/"):
        provider = "openrouter"
        # Strip the "openrouter/" prefix → e.g. "deepseek/deepseek-v3.2"
        provider_model_id = model[len("openrouter/"):]
    elif model.startswith("ollama/"):
        provider = "ollama"
        provider_model_id = model[len("ollama/"):]
    else:
        return None  # Provider not supported for remote check.

    try:
        available = _fetch_provider_models(provider)
    except Exception:
        return None  # Network error — can't verify.

    return provider_model_id in available


# ---------------------------------------------------------------------------
# Model validation check
# ---------------------------------------------------------------------------

def _check_single_model(model: str) -> ModelCheckResult:
    """Validate a model name and its required environment variables.

    Uses a **two-tier** approach:

    1. **Local (instant):** :func:`litellm.get_model_info` checks the
       model against litellm's built-in registry.
    2. **Remote (lightweight):** If Tier 1 fails, queries the provider's
       model-list endpoint (e.g. ``GET /v1/models``) to confirm the
       model actually exists.  This is free (no tokens consumed) and
       fast (cached per provider).

    Additionally, :func:`litellm.validate_environment` confirms the
    required API keys / env vars are set.

    :param model: LiteLLM model identifier (e.g. ``"openrouter/deepseek/deepseek-v3.2"``).
    :returns: A :class:`ModelCheckResult` indicating success or failure.
    """
    errors: List[str] = []
    warning: Optional[str] = None

    # Tier 1: local litellm registry check.
    model_known = True
    try:
        litellm.get_model_info(model)
    except Exception:
        model_known = False

    # Environment / API key check.
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
        if not model_known:
            errors.insert(0, "Not in litellm model registry")
        return ModelCheckResult(
            model=model, ok=False, error="; ".join(errors)
        )

    if not model_known:
        # Tier 2: remote provider model-list check.
        remote_result = _check_model_at_provider(model)

        if remote_result is True:
            # Confirmed available at provider — pass.
            return ModelCheckResult(model=model, ok=True)

        if remote_result is False:
            # Provider was reachable but model not found — fail.
            return ModelCheckResult(
                model=model,
                ok=False,
                error="Model not found at provider (not in litellm "
                      "registry and not listed by provider's model API)",
            )

        # remote_result is None — could not verify (network error or
        # unsupported provider).  Downgrade to warning.
        warning = (
            "Not in litellm model registry and could not verify "
            "at provider (may still work at runtime)"
        )
        return ModelCheckResult(model=model, ok=True, warning=warning)

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

    For models not found in litellm's local registry, a lightweight
    ``GET /v1/models`` call may be made to the provider to confirm
    availability.  This remote check is **free** (no tokens consumed)
    and results are **cached** per provider.

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
            if result.ok and result.warning:
                print(f"  [WARN] {result.model}: {result.warning}")
            elif result.ok:
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
    if report.model_warning_count:
        print(f"  Warnings: {total_warnings}"
              f" ({report.model_warning_count} model)", end="")
        remaining = total_warnings - report.model_warning_count
        if remaining:
            print(f" ({remaining} other)", end="")
        print()
    else:
        print(f"  Warnings: {total_warnings}")
    if total_errors:
        print("  --> Run CANCELLED due to errors.")
    elif total_warnings:
        print("  --> Warnings present.")
    else:
        print("  --> All checks passed.")
    print("=" * 50)
