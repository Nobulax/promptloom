# AGENTS.md — Technical Overview for AI Agents

Quick-reference for AI agents working on this codebase.

## What this project does

`promptloom` is a Python toolkit that dispatches batch LLM
API calls from YAML configuration files.  It handles prompt assembly,
concurrent async execution, response processing (JSON extraction),
validation (JSON Schema + custom callables), a multi-turn correction
loop, structured reporting, and failed-run re-generation.

## Project structure

```
promptloom/
├── pyproject.toml                 # Package metadata, deps, CLI entry point
├── .env.example                   # Template for API keys
├── README.md                      # User-facing documentation
├── AGENTS.md                      # This file
│
├── src/promptloom/
│   ├── __init__.py                # Public API exports
│   ├── cli.py                     # Click CLI (promptloom run)
│   ├── config.py                  # YAML loading → TaskConfig / ExperimentConfig dataclasses
│   ├── prompt.py                  # Template loading, {{PLACEHOLDER}} substitution, file: resolution
│   ├── preflight.py               # Pre-flight checks (model, placeholder, validation config)
│   ├── runner.py                  # Async execution engine with correction loop
│   ├── validation.py              # Response processors, validators, ValidationResult
│   └── report.py                  # YAML report generation, failed-config generation
│
├── tests/
│   ├── test_config.py             # Config loading + new validation fields
│   ├── test_prompt.py             # Prompt assembly, placeholder checks
│   └── test_validation.py         # JSON extraction, validators, pipeline
│
└── examples/
    ├── config.yaml                # Basic example (text output, no validation)
    ├── config_with_validation.yaml# Example with JSON validation + correction loop
    ├── config_full.yaml           # Full reference showing ALL settings
    ├── prompt.md                  # Basic prompt template
    ├── prompt_structured.md       # Prompt template for JSON output
    ├── correction.md              # Correction prompt template with {{ERROR}}
    ├── schemas/
    │   └── summary_schema.json    # Example JSON Schema
    └── data/
        └── sample.txt             # Sample input data
```

## Key data flow

```
YAML config
  → load_config() → ExperimentConfig (list of TaskConfig)
  → run_preflight() → PreflightReport (errors/warnings)
  → _run_experiment_async():
      For each task:
        load_template() → resolve_params() → assemble_prompt()
        resolve system_prompt (file: prefix + placeholder substitution)
        load_processor() + load_validators() + load correction template

        For each model:
          → _execute_single() [async, semaphore-guarded]:
              Build messages [system?, user]
              Loop (max 1 + max_corrections attempts):
                litellm.acompletion(messages) → raw content
                processor(content) → processed data
                run_validators(processed, context) → ValidationResult
                If fail + corrections left:
                  Append assistant + correction prompt to messages
                  Continue
                Else:
                  Save output, return result dict

  → save_report_yaml() + generate_failed_yaml()
```

## Module responsibilities

### `config.py`
- `TaskConfig` dataclass: all per-task settings (prompt_template, system_prompt,
  models, response_format, validators, correction_prompt, max_corrections, etc.)
- `ExperimentConfig` dataclass: experiment metadata + list of TaskConfig + base_dir
- `load_config(path)`: parses YAML, merges `defaults` with per-task overrides

### `prompt.py`
- `load_template(path)`: reads a Markdown template file
- `extract_placeholders(template)`: finds all `{{NAME}}` placeholders
- `resolve_param_value(value, base_dir)`: handles `file:` prefix → reads file content;
  handles `glob:` prefix → reads all matching files, concatenated with `=== File N: name ===` headers
- `resolve_params(params, base_dir)`: resolves all params in a dict
- `assemble_prompt(template, params)`: substitutes `{{KEY}}` placeholders (case-insensitive)
- `check_placeholders(template, params)`: returns (missing, unused) lists

### `validation.py`
- `ValidationResult`: dataclass with `.ok(data)` / `.fail(error)` factories
- `extract_json(raw)`: robust JSON extraction (direct parse → code fences → heuristic)
- `validate_json_schema(data, context, *, schema)`: built-in JSON Schema validator
- `load_processor(format_name)`: returns processor callable ("text" or "json")
- `load_validators(specs, base_dir)`: resolves YAML validator specs to callables
  - `json_schema` type: loads schema file, returns partial of validate_json_schema
  - `custom` type: imports callable by dotted path
- `run_validators(validators, data, context)`: runs chain, short-circuits on failure

### `runner.py`
- `_execute_single()`: async coroutine for one (task, model) pair with correction loop
- `_run_experiment_async()`: orchestrates full experiment (preflight, dispatch, aggregate, report)
- `run_experiment(config_path)`: sync entry point (loads config, calls asyncio.run)
- `run_experiment_async(config)`: async entry point (accepts pre-loaded ExperimentConfig)

### `preflight.py`
- Model validation: `litellm.get_model_info()` + `litellm.validate_environment()` (no API calls)
- Placeholder validation: checks template placeholders vs task params
- Validation config: checks response_format, correction_prompt existence/{{ERROR}}, schema files, validator specs
- `PreflightReport`: aggregated results with `.has_errors` / `.has_warnings`

### `report.py`
- `save_report_yaml()`: timestamped YAML report of all results
- `generate_failed_yaml()`: YAML config with only failed (task, model) pairs for re-runs

### `cli.py`
- Click-based CLI: `promptloom run CONFIG [--dry-run] [--skip-preflight] [--model-timeout N]`

## Public Python API

```python
from promptloom import (
    load_config,           # YAML → ExperimentConfig
    run_experiment,        # sync: path → results dict
    run_experiment_async,  # async: ExperimentConfig → results dict
    ExperimentConfig,
    TaskConfig,
    ValidationResult,
)
```

## Validator callable signature

```python
def my_validator(data: Any, context: dict) -> ValidationResult:
    # context keys: task_id, params, model, attempt, run_number, base_dir
    if problem:
        return ValidationResult.fail("error description")
    return ValidationResult.ok(data)  # data may be transformed
```

## Dependencies

- `litellm` — unified LLM API (supports all major providers)
- `pyyaml` — YAML parsing
- `python-dotenv` — .env file loading
- `click` — CLI framework
- `jsonschema` (optional) — JSON Schema validation (`pip install .[validation]`)

## Testing

```bash
.venv/bin/pytest tests/ -v    # 77 tests, all pure unit tests (no API calls)
```

Tests cover: config loading (incl. new validation fields), prompt assembly,
JSON extraction, validator loading/chaining, ValidationResult, import helper.
Runner/preflight tests require mocking litellm and are not yet implemented.
