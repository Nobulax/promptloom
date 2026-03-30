"""Prompt template loading and placeholder resolution.

Handles reading prompt template files, extracting placeholder names,
resolving parameter values (including ``file:`` references), and
assembling the final prompt string.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Pattern matching ``{{PLACEHOLDER_NAME}}`` in prompt templates.
PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")

# Prefix indicating that a parameter value should be read from a file.
FILE_PREFIX = "file:"

# Prefix indicating that a parameter value should be read from multiple
# files matching a glob pattern.
GLOB_PREFIX = "glob:"


def load_template(
    template_path: Path,
) -> str:
    """Read a prompt template file and return its content.

    :param template_path: Absolute or pre-resolved path to the template.
    :returns: The raw template string.
    :raises FileNotFoundError: If the template file does not exist.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def extract_placeholders(template: str) -> Set[str]:
    """Extract all unique placeholder names from a template string.

    Placeholders use the ``{{NAME}}`` syntax.  The returned names
    preserve the original casing from the template.

    :param template: The prompt template string.
    :returns: Set of placeholder names found in the template.
    """
    return set(PLACEHOLDER_PATTERN.findall(template))


def resolve_param_value(
    value: str,
    base_dir: Path,
) -> str:
    """Resolve a single parameter value.

    Supports three modes:

    ``file:path``
        Reads the content of a single file.  The path is relative to
        *base_dir* unless absolute.

    ``glob:pattern``
        Reads all files matching the glob *pattern* (relative to
        *base_dir* unless the pattern is absolute).  Files are sorted
        by name, and their contents are concatenated with
        ``=== File N: filename ===`` headers.

    Otherwise the literal string is returned unchanged.

    :param value: The raw parameter value from the YAML config.
    :param base_dir: Base directory for resolving relative file paths.
    :returns: The resolved string value.
    :raises FileNotFoundError: If a ``file:`` reference points to a
        non-existent file, or if a ``glob:`` pattern matches no files.
    """
    if value.startswith(GLOB_PREFIX):
        pattern_str = value[len(GLOB_PREFIX):]
        pattern_path = Path(pattern_str)
        if pattern_path.is_absolute():
            # Absolute glob: split into anchor directory + pattern part.
            # E.g. /home/user/data/*.json → anchor=/home/user/data, pat=*.json
            anchor = pattern_path.parent
            pat = pattern_path.name
            files = sorted(anchor.glob(pat))
        else:
            files = sorted(base_dir.glob(pattern_str))
        if not files:
            raise FileNotFoundError(
                f"Glob pattern matched no files: {value} "
                f"(resolved from base_dir '{base_dir}')"
            )
        parts: list[str] = []
        for idx, fpath in enumerate(files, start=1):
            content = fpath.read_text(encoding="utf-8")
            parts.append(f"=== File {idx}: {fpath.name} ===\n{content}")
        return "\n\n".join(parts)

    if value.startswith(FILE_PREFIX):
        file_path_str = value[len(FILE_PREFIX):]
        file_path = Path(file_path_str)
        if not file_path.is_absolute():
            file_path = base_dir / file_path
        if not file_path.exists():
            raise FileNotFoundError(
                f"Parameter file not found: {file_path} "
                f"(from value '{value}')"
            )
        return file_path.read_text(encoding="utf-8")

    return value


def resolve_params(
    params: Dict[str, str],
    base_dir: Path,
) -> Dict[str, str]:
    """Resolve all parameter values in a params dict.

    Each value is processed through :func:`resolve_param_value`, which
    reads file contents for ``file:``-prefixed values and passes through
    literal strings unchanged.

    :param params: Mapping of parameter names to raw values.
    :param base_dir: Base directory for resolving relative file paths.
    :returns: Mapping of parameter names to resolved string values.
    """
    return {
        key: resolve_param_value(value, base_dir)
        for key, value in params.items()
    }


def assemble_prompt(
    template: str,
    resolved_params: Dict[str, str],
) -> str:
    """Substitute resolved parameter values into a prompt template.

    For each ``{{KEY}}`` placeholder in the template, the corresponding
    value from *resolved_params* is inserted.  Matching is
    case-insensitive: parameter keys are uppercased before lookup, and
    placeholders in the template are matched as-is.  The convention is
    that both template placeholders and YAML param keys use the same
    casing (typically ``UPPER_CASE`` in the template, ``lower_case`` or
    ``UPPER_CASE`` in the YAML).

    Placeholders without a matching parameter are left unchanged.

    :param template: The prompt template string with ``{{KEY}}``
        placeholders.
    :param resolved_params: Mapping of parameter names to resolved values.
    :returns: The assembled prompt string with placeholders substituted.
    """
    # Build a lookup keyed on uppercase names.
    upper_params: Dict[str, str] = {
        k.upper(): v for k, v in resolved_params.items()
    }

    def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
        name = match.group(1)
        return upper_params.get(name.upper(), match.group(0))

    return PLACEHOLDER_PATTERN.sub(_replace, template).strip()


def check_placeholders(
    template: str,
    params: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    """Check placeholder coverage between a template and a params dict.

    :param template: The prompt template string.
    :param params: The parameter dict for a task (keys as defined in
        YAML, not yet uppercased).
    :returns: A tuple of ``(missing, unused)`` where *missing* lists
        placeholder names present in the template but absent from
        *params*, and *unused* lists param keys defined in *params* but
        not referenced by any placeholder in the template.
    """
    placeholders = extract_placeholders(template)
    placeholder_names_upper = {p.upper() for p in placeholders}
    param_names_upper = {k.upper() for k in params}

    missing = sorted(placeholder_names_upper - param_names_upper)
    unused = sorted(param_names_upper - placeholder_names_upper)
    return missing, unused
