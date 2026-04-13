"""Tests for the preflight module.

Covers the two-tier model validation (local litellm registry +
remote provider model-list) and the data classes.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from promptloom.preflight import (
    ModelCheckResult,
    PreflightReport,
    _check_model_at_provider,
    _check_single_model,
    _fetch_provider_models,
    _provider_model_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_cache():
    """Clear the module-level provider model cache between tests."""
    _provider_model_cache.clear()


@pytest.fixture(autouse=True)
def _clean_cache():
    """Auto-clear the provider model cache before each test."""
    _clear_cache()
    yield
    _clear_cache()


def _mock_urlopen(model_ids: list[str]):
    """Return a context-manager mock for urllib.request.urlopen.

    The mock returns an OpenAI-compatible ``/v1/models`` JSON response
    containing the given *model_ids*.
    """
    body = json.dumps({
        "data": [{"id": mid} for mid in model_ids],
    }).encode()

    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# _check_model_at_provider
# ---------------------------------------------------------------------------

class TestCheckModelAtProvider:
    """Tests for _check_model_at_provider()."""

    def test_unsupported_provider_returns_none(self):
        """Unknown provider prefix → None (can't check)."""
        assert _check_model_at_provider("anthropic/claude-3") is None

    @mock.patch("promptloom.preflight.urllib.request.urlopen")
    def test_openrouter_model_found(self, mock_urlopen_fn):
        """OpenRouter model found in provider list → True."""
        mock_urlopen_fn.return_value = _mock_urlopen([
            "deepseek/deepseek-v3.2",
            "minimax/minimax-m2.7",
        ])
        assert _check_model_at_provider("openrouter/deepseek/deepseek-v3.2") is True

    @mock.patch("promptloom.preflight.urllib.request.urlopen")
    def test_openrouter_model_not_found(self, mock_urlopen_fn):
        """OpenRouter model NOT in provider list → False."""
        mock_urlopen_fn.return_value = _mock_urlopen([
            "deepseek/deepseek-v3.2",
        ])
        assert _check_model_at_provider("openrouter/nonexistent/model-99") is False

    @mock.patch("promptloom.preflight.urllib.request.urlopen")
    def test_network_error_returns_none(self, mock_urlopen_fn):
        """Network error during fetch → None (can't verify)."""
        mock_urlopen_fn.side_effect = ConnectionError("no network")
        assert _check_model_at_provider("openrouter/deepseek/deepseek-v3.2") is None

    @mock.patch("promptloom.preflight.urllib.request.urlopen")
    def test_ollama_model_found(self, mock_urlopen_fn):
        """Ollama model found in provider list → True."""
        # Simulate the /api/tags endpoint
        body = json.dumps({
            "models": [{"name": "llama3"}, {"name": "qwen3.5"}],
        }).encode()
        resp = mock.MagicMock()
        resp.read.return_value = body
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        mock_urlopen_fn.return_value = resp

        with mock.patch.dict("os.environ", {"OLLAMA_API_BASE": "http://localhost:11434"}):
            assert _check_model_at_provider("ollama/qwen3.5") is True

    @mock.patch("promptloom.preflight.urllib.request.urlopen")
    def test_ollama_model_not_found(self, mock_urlopen_fn):
        """Ollama model NOT in provider list → False."""
        body = json.dumps({
            "models": [{"name": "llama3"}],
        }).encode()
        resp = mock.MagicMock()
        resp.read.return_value = body
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        mock_urlopen_fn.return_value = resp

        with mock.patch.dict("os.environ", {"OLLAMA_API_BASE": "http://localhost:11434"}):
            assert _check_model_at_provider("ollama/nonexistent") is False


# ---------------------------------------------------------------------------
# _fetch_provider_models (caching)
# ---------------------------------------------------------------------------

class TestFetchProviderModels:
    """Tests for _fetch_provider_models() caching behavior."""

    @mock.patch("promptloom.preflight.urllib.request.urlopen")
    def test_cache_is_used(self, mock_urlopen_fn):
        """Second call for same provider should use cache, not HTTP."""
        mock_urlopen_fn.return_value = _mock_urlopen(["model/a"])

        result1 = _fetch_provider_models("openrouter")
        result2 = _fetch_provider_models("openrouter")

        assert result1 == result2 == {"model/a"}
        # urlopen should be called only once due to caching.
        assert mock_urlopen_fn.call_count == 1

    def test_unknown_provider_raises(self):
        """Unknown provider should raise ValueError."""
        with pytest.raises(ValueError, match="No model-list fetcher"):
            _fetch_provider_models("unknown_provider")


# ---------------------------------------------------------------------------
# _check_single_model (two-tier integration)
# ---------------------------------------------------------------------------

class TestCheckSingleModel:
    """Tests for _check_single_model() two-tier logic."""

    @mock.patch("promptloom.preflight.litellm.validate_environment")
    @mock.patch("promptloom.preflight.litellm.get_model_info")
    def test_known_model_passes(self, mock_info, mock_env):
        """Model in litellm registry + env ok → PASS."""
        mock_info.return_value = {"key": "value"}
        mock_env.return_value = {"missing_keys": []}

        result = _check_single_model("gemini/gemini-2.0-flash")
        assert result.ok is True
        assert result.error is None
        assert result.warning is None

    @mock.patch("promptloom.preflight.litellm.validate_environment")
    @mock.patch("promptloom.preflight.litellm.get_model_info")
    def test_missing_env_vars_fails(self, mock_info, mock_env):
        """Missing API key → FAIL regardless of model registry."""
        mock_info.return_value = {"key": "value"}
        mock_env.return_value = {"missing_keys": ["GEMINI_API_KEY"]}

        result = _check_single_model("gemini/gemini-2.0-flash")
        assert result.ok is False
        assert "GEMINI_API_KEY" in result.error

    @mock.patch("promptloom.preflight._check_model_at_provider")
    @mock.patch("promptloom.preflight.litellm.validate_environment")
    @mock.patch("promptloom.preflight.litellm.get_model_info")
    def test_unknown_model_confirmed_at_provider_passes(
        self, mock_info, mock_env, mock_remote
    ):
        """Not in litellm registry, but confirmed at provider → PASS."""
        mock_info.side_effect = Exception("not found")
        mock_env.return_value = {"missing_keys": []}
        mock_remote.return_value = True

        result = _check_single_model("openrouter/deepseek/deepseek-v3.2")
        assert result.ok is True
        assert result.error is None
        assert result.warning is None

    @mock.patch("promptloom.preflight._check_model_at_provider")
    @mock.patch("promptloom.preflight.litellm.validate_environment")
    @mock.patch("promptloom.preflight.litellm.get_model_info")
    def test_unknown_model_not_at_provider_fails(
        self, mock_info, mock_env, mock_remote
    ):
        """Not in litellm registry AND not at provider → FAIL."""
        mock_info.side_effect = Exception("not found")
        mock_env.return_value = {"missing_keys": []}
        mock_remote.return_value = False

        result = _check_single_model("openrouter/nonexistent/model-99")
        assert result.ok is False
        assert "not found at provider" in result.error.lower()

    @mock.patch("promptloom.preflight._check_model_at_provider")
    @mock.patch("promptloom.preflight.litellm.validate_environment")
    @mock.patch("promptloom.preflight.litellm.get_model_info")
    def test_unknown_model_remote_unavailable_warns(
        self, mock_info, mock_env, mock_remote
    ):
        """Not in litellm registry, remote check fails → WARN."""
        mock_info.side_effect = Exception("not found")
        mock_env.return_value = {"missing_keys": []}
        mock_remote.return_value = None  # Could not verify.

        result = _check_single_model("someprovider/some-model")
        assert result.ok is True
        assert result.warning is not None
        assert "could not verify" in result.warning.lower()


# ---------------------------------------------------------------------------
# PreflightReport aggregation
# ---------------------------------------------------------------------------

class TestPreflightReport:
    """Tests for PreflightReport warning/error aggregation."""

    def test_model_warning_counted(self):
        """Model warnings should be counted in warning_count."""
        report = PreflightReport(
            model_results=[
                ModelCheckResult(model="a", ok=True, warning="some warning"),
                ModelCheckResult(model="b", ok=True),
            ]
        )
        assert report.model_warning_count == 1
        assert report.warning_count == 1
        assert report.has_warnings is True
        assert report.has_errors is False

    def test_model_error_not_warning(self):
        """Model errors should not count as warnings."""
        report = PreflightReport(
            model_results=[
                ModelCheckResult(model="a", ok=False, error="missing key"),
            ]
        )
        assert report.model_error_count == 1
        assert report.model_warning_count == 0
        assert report.has_errors is True

    def test_empty_report(self):
        """Empty report has no errors or warnings."""
        report = PreflightReport()
        assert report.has_errors is False
        assert report.has_warnings is False
        assert report.warning_count == 0
