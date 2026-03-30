# promptloom

`promptloom` is a Python toolkit that turns a prompt template and a
YAML config file into a fully managed batch of LLM API calls.  You
write a Markdown prompt with `{{PLACEHOLDER}}` slots, declare your
tasks and models in YAML, and the toolkit takes care of the rest:
prompt assembly, concurrent async execution via
[LiteLLM](https://docs.litellm.ai/), structured JSON extraction,
schema & custom validation, a multi-turn correction loop for invalid
responses, and detailed YAML reporting — including automatic
re-generation configs for any failed runs.

## Features

- **YAML-driven configuration** — define experiments, tasks, models,
  parameters, and settings in a single YAML file.
- **Arbitrary prompt parameters** — use `{{PLACEHOLDER}}` syntax in
  prompt templates; parameter values come from the YAML task definition.
- **File references** — prefix parameter values with `file:` to read
  content from disk (e.g. `file:data/law.txt`).
- **System prompts** — optional system-level messages, supporting both
  literal strings and `file:` references with placeholder substitution.
- **Multi-provider support** — any model supported by
  [LiteLLM](https://docs.litellm.ai/) (OpenAI, Anthropic, Google Gemini,
  Ollama, etc.).
- **Fully asynchronous** — all API calls run concurrently via `asyncio`
  with configurable concurrency limits.
- **Response processing** — built-in JSON extraction from raw LLM
  responses (handles code fences, raw JSON, and heuristic extraction).
- **Validation pipeline** — chain validators (JSON Schema, custom Python
  functions) to check structured output for correctness.
- **Multi-turn correction loop** — when validation fails, automatically
  send the error report back to the LLM and retry (configurable number
  of correction turns).
- **Three-phase pre-flight checks** (instant, free, no API calls):
  1. *Model validation* — verifies model names and API key availability
     using LiteLLM's built-in model registry.
  2. *Placeholder validation* — ensures every `{{PLACEHOLDER}}` in each
     template has a matching parameter; warns about unused parameters.
  3. *Validation config check* — verifies response format, correction
     prompt existence, schema files, and validator specs.
- **Structured YAML reports** — timestamped report with per-task,
  per-model status, error details, timing, token usage, and correction
  attempt counts.
- **Failed-run config** — automatically generates a YAML config
  containing only the failed (task, model) pairs for convenient re-runs.
- **Programmatic API** — use from Python scripts and Jupyter notebooks
  with both sync (`run_experiment`) and async (`run_experiment_async`)
  entry points.

## Installation

```bash
pip install promptloom
```

For development (clone the repo, then install in editable mode):

```bash
git clone https://github.com/Nobulax/promptloom.git
cd promptloom
pip install -e ".[dev]"
```

## Quick start

### 1. Create a prompt template

Write a Markdown file with `{{PLACEHOLDER}}` syntax for variable parts:

```markdown
# Task

{{INSTRUCTION}}

# Document

{{DOCUMENT}}
```

### 2. Create a YAML config

```yaml
experiment:
  name: "My Experiment"

defaults:
  models:
    - "openai/gpt-4o"
  prompt_template: "prompt.md"
  system_prompt: "You are a helpful assistant."
  output_dir: "output/"
  max_completion_tokens: 4000
  timeout: 120
  max_concurrency: 5

tasks:
  - id: "task-1"
    params:
      document: "file:data/input.txt"
      instruction: "Summarize this document."
  - id: "task-2"
    params:
      document: "Inline text content."
      instruction: "Translate this to German."
    models:
      - "openai/gpt-4o"
      - "anthropic/claude-sonnet-4-20250514"
```

### 3. Set up API keys

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GEMINI_API_KEY="..."
```

### 4. Run

From the command line:

```bash
promptloom run config.yaml
promptloom run config.yaml --dry-run       # validate only, no API calls
promptloom run config.yaml --skip-preflight # skip pre-flight checks
```

Or from Python:

```python
from promptloom import run_experiment

results = run_experiment("config.yaml")
```

## Structured output with validation

For tasks that require structured JSON output, the toolkit provides a
complete processing pipeline: extract → validate → correct.

### Example config

```yaml
defaults:
  models:
    - "openai/gpt-4o"
  prompt_template: "prompt.md"
  system_prompt: "You are a helpful assistant that responds with valid JSON."
  response_format: "json"
  validators:
    - type: json_schema
      schema: "schemas/output_schema.json"
  correction_prompt: "correction.md"
  max_corrections: 3

tasks:
  - id: "generate-data"
    params:
      instruction: "Generate a structured summary."
      document: "file:data/input.txt"
```

### Correction prompt template

The correction prompt is a Markdown file with a `{{ERROR}}` placeholder:

```markdown
Your previous response was invalid. Here is the error report:

{{ERROR}}

Please correct your output. Return only the valid JSON object.
```

### Custom validators

For domain-specific checks (e.g., data integrity, allowed labels,
no duplicate IDs), write a Python function and reference it by its
import path:

```yaml
validators:
  - type: json_schema
    schema: "schemas/output.json"
  - type: custom
    callable: "mypackage.validators.check_integrity"
```

The function must have this signature:

```python
from promptloom.validation import ValidationResult

def check_integrity(data, context):
    """
    data:    the parsed response (e.g., dict from JSON)
    context: {"task_id": ..., "params": {...}, "model": ..., "attempt": ...}
    """
    errors = []
    ids = [item["id"] for item in data["items"]]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate IDs found")
    
    if errors:
        return ValidationResult.fail("\n".join(errors))
    return ValidationResult.ok(data)
```

The error string from `ValidationResult.fail()` is substituted into the
`{{ERROR}}` placeholder of the correction prompt and sent back to the LLM.

### How the correction loop works

1. LLM responds → response processor runs (e.g., JSON extraction).
2. Validators run in order.  First failure stops the chain.
3. If validation fails and corrections remain:
   - The assistant's response is appended to the conversation.
   - A correction prompt with `{{ERROR}}` filled in is appended.
   - The LLM is called again with the full conversation history.
4. Repeat up to `max_corrections` times.
5. If all corrections are exhausted, the task is marked as failed
   (but the last output is still saved for inspection).

## YAML config reference

A fully-commented reference config showing every available field is at
[`examples/config_full.yaml`](examples/config_full.yaml).

### `experiment` (optional)

| Field           | Type   | Description                      |
|-----------------|--------|----------------------------------|
| `name`          | string | Human-readable experiment name.  |
| `description`   | string | Longer description.              |

### `defaults` (optional)

Global defaults applied to all tasks unless overridden per-task.

| Field                    | Type    | Default    | Description                                         |
|--------------------------|---------|------------|-----------------------------------------------------|
| `models`                 | list    | —          | LiteLLM model identifiers.                          |
| `prompt_template`        | string  | —          | Path to the prompt template file.                   |
| `system_prompt`          | string  | `null`     | System message (literal or `file:` reference).      |
| `output_dir`             | string  | `output`   | Base output directory.                              |
| `max_completion_tokens`  | int     | `64000`    | Max tokens in LLM response.                         |
| `timeout`                | int     | `null`     | Timeout in seconds per API call.                    |
| `max_concurrency`        | int     | `10`       | Max parallel API calls.                             |
| `ignore_unused_params`   | bool    | `false`    | Auto-continue on unused-param warnings.             |
| `response_format`        | string  | `text`     | Response processor: `"text"` or `"json"`.           |
| `validators`             | list    | `[]`       | Ordered list of validator specs.                    |
| `correction_prompt`      | string  | `null`     | Path to correction prompt template (needs `{{ERROR}}`). |
| `max_corrections`        | int     | `0`        | Max correction turns on validation failure.         |

### `tasks` (required)

List of task objects.  Each task defines one prompt sent to one or more
models.  All `defaults` fields can be overridden per-task.

| Field                    | Type   | Required | Description                                              |
|--------------------------|--------|----------|----------------------------------------------------------|
| `id`                     | string | yes      | Unique task identifier.                                  |
| `params`                 | dict   | no       | Key-value pairs substituted into the prompt template.    |
| `models`                 | list   | no       | Override default models for this task.                   |
| `prompt_template`        | string | no       | Override default prompt template.                        |
| `system_prompt`          | string | no       | Override default system prompt.                          |
| `output_dir`             | string | no       | Override output directory.                               |
| `max_completion_tokens`  | int    | no       | Override max tokens.                                     |
| `timeout`                | int    | no       | Override timeout.                                        |
| `response_format`        | string | no       | Override response processor.                             |
| `validators`             | list   | no       | Override validators.                                     |
| `correction_prompt`      | string | no       | Override correction prompt template.                     |
| `max_corrections`        | int    | no       | Override max corrections.                                |

### Parameter values

Parameter values in `params` are strings by default.  To include file
contents, prefix the value with `file:`:

```yaml
params:
  document: "file:data/input.txt"     # reads file content
  instruction: "Summarize this."       # literal string
```

File paths are resolved relative to the YAML config file's directory.

## Output structure

```
output/
  task-1/
    task-1_openai_gpt-4o.txt           # text format
  task-2/
    task-2_openai_gpt-4o.json          # json format (pretty-printed)
    task-2_anthropic_claude-sonnet-4-20250514.json
  config_report_20260319_143000.yaml    # timestamped report
  config_failed.yaml                    # only if there were failures
```

## Pre-flight checks

Before dispatching API calls, three instant checks run (no API calls,
no cost):

1. **Model validation** — uses LiteLLM's built-in model registry
   (`litellm.get_model_info` and `litellm.validate_environment`) to
   verify that each model name is recognised and the required API
   keys / environment variables are set.
2. **Placeholder validation** — for each task, verifies that every
   `{{PLACEHOLDER}}` in the prompt template has a matching key in the
   task's `params`.  Missing params are fatal errors.  Unused params
   are warnings (auto-continued if `ignore_unused_params: true`).
3. **Validation config** — checks that `response_format` is valid,
   correction prompt files exist and contain `{{ERROR}}`, schema files
   exist, and validator specs are well-formed.

All checks run to completion before any abort decision, so you see all
problems at once.

## Programmatic API

```python
from promptloom import load_config, run_experiment, run_experiment_async
from promptloom.validation import ValidationResult

# Synchronous (from scripts)
results = run_experiment("config.yaml")

# Async (from notebooks or async code)
config = load_config("config.yaml")
results = await run_experiment_async(config, skip_preflight=True)
```

## License

MIT
