"""Tests for the profile-safe OMP configuration boundary."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from functools import partial

import pytest

from omnigent.omp_config import (
    OmpCommandError,
    OmpConfigParseError,
    OmpProfileConfig,
    OmpValidationError,
    list_omp_models,
    persist_omp_model,
    persist_omp_thinking_level,
)


class FakeOmpRunner:
    """Return scripted OMP responses without invoking a real OMP binary."""

    def __init__(self, responses: list[subprocess.CompletedProcess[str]]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        return next(self._responses)


def _result(
    argv: list[str], *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)


def _model_roles() -> str:
    return json.dumps(
        {
            "default": {"provider": "anthropic", "model": "claude-sonnet-4-5"},
            "background": {"provider": "openai", "model": "gpt-5.2"},
        }
    )


def test_persist_model_replaces_only_default_role() -> None:
    """Replacing the default model must retain every unrelated role object."""
    get_argv = ["omp", "config", "get", "modelRoles", "--json"]
    set_argv = ["omp", "config", "set", "modelRoles"]
    runner = FakeOmpRunner([_result(get_argv, stdout=_model_roles()), _result(set_argv)])

    OmpProfileConfig(runner=runner).persist_model("openai", "gpt-5.6")

    assert runner.calls[0] == get_argv
    assert runner.calls[1][:-1] == set_argv
    assert json.loads(runner.calls[1][-1]) == {
        "default": {"provider": "openai", "model": "gpt-5.6"},
        "background": {"provider": "openai", "model": "gpt-5.2"},
    }


def test_persist_model_raises_when_config_set_fails() -> None:
    """A rejected config write must not be reported as a successful update."""
    get_argv = ["omp", "config", "get", "modelRoles", "--json"]
    set_argv = ["omp", "config", "set", "modelRoles"]
    runner = FakeOmpRunner(
        [
            _result(get_argv, stdout=_model_roles()),
            _result(set_argv, stderr="permission denied", returncode=7),
        ]
    )

    with pytest.raises(OmpCommandError, match="config set") as exc_info:
        OmpProfileConfig(runner=runner).persist_model("openai", "gpt-5.6")

    assert exc_info.value.command_class == "config set"
    assert runner.calls[1][:-1] == set_argv


def test_named_profile_prefixes_every_omp_command() -> None:
    """A selected named profile must be applied to both read and write commands."""
    prefix = ["omp", "--profile", "work"]
    get_argv = [*prefix, "config", "get", "modelRoles", "--json"]
    set_argv = [*prefix, "config", "set", "modelRoles"]
    runner = FakeOmpRunner([_result(get_argv, stdout=_model_roles()), _result(set_argv)])

    OmpProfileConfig(profile="work", runner=runner).persist_model("openai", "gpt-5.6")

    assert runner.calls[0] == get_argv
    assert runner.calls[1][:-1] == set_argv


def test_empty_profile_uses_global_omp_configuration() -> None:
    """An empty profile selects the global configuration without a profile flag."""
    argv = ["omp", "config", "set", "defaultThinkingLevel", "high"]
    runner = FakeOmpRunner([_result(argv)])

    OmpProfileConfig(profile="", runner=runner).persist_thinking_level("high")

    assert runner.calls == [argv]


def test_malformed_model_roles_output_raises_actionable_parse_error() -> None:
    """Corrupt modelRoles JSON must never be replaced with a partial object."""
    argv = ["omp", "config", "get", "modelRoles", "--json"]
    runner = FakeOmpRunner([_result(argv, stdout="not json")])

    with pytest.raises(OmpConfigParseError, match="config get") as exc_info:
        OmpProfileConfig(runner=runner).persist_model("openai", "gpt-5.6")

    assert exc_info.value.command_class == "config get"
    assert len(runner.calls) == 1


def test_malformed_models_catalog_output_raises_actionable_parse_error() -> None:
    """Catalog consumers receive a typed error instead of silently empty options."""
    argv = ["omp", "models", "--json"]
    runner = FakeOmpRunner([_result(argv, stdout="{")])

    with pytest.raises(OmpConfigParseError, match="models") as exc_info:
        OmpProfileConfig(runner=runner).list_models()

    assert exc_info.value.command_class == "models"


def test_list_models_parses_structured_catalog() -> None:
    """Catalog entries preserve provider metadata and only accept valid records."""
    argv = ["omp", "models", "--json"]
    catalog = {
        "models": [
            {
                "provider": "openai",
                "id": "gpt-5.6",
                "displayName": "GPT 5.6",
                "providerName": "OpenAI",
                "aliases": ["gpt-latest"],
            }
        ]
    }
    runner = FakeOmpRunner([_result(argv, stdout=json.dumps(catalog))])

    models = OmpProfileConfig(runner=runner).list_models()

    assert [(model.provider, model.model_id, model.display_name) for model in models] == [
        ("openai", "gpt-5.6", "GPT 5.6")
    ]
    assert models[0].provider_name == "OpenAI"
    assert models[0].aliases == ["gpt-latest"]


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [("", "gpt-5.6"), ("open ai", "gpt-5.6"), ("openai", "gpt/5.6"), ("openai", "")],
)
def test_persist_model_rejects_invalid_identifiers(provider: str, model_id: str) -> None:
    """Invalid identifiers must fail before OMP receives a potentially unsafe argument."""
    runner = FakeOmpRunner([])

    with pytest.raises(OmpValidationError):
        OmpProfileConfig(runner=runner).persist_model(provider, model_id)

    assert runner.calls == []


@pytest.mark.parametrize("level", ["", "very-high", "high; rm -rf /", "turbo"])
def test_persist_thinking_level_rejects_unknown_levels(level: str) -> None:
    """Only OMP's documented thinking-level values may be persisted."""
    runner = FakeOmpRunner([])

    with pytest.raises(OmpValidationError):
        OmpProfileConfig(runner=runner).persist_thinking_level(level)

    assert runner.calls == []


def test_public_helpers_use_the_injected_runner() -> None:
    """The public API stays testable without an OMP installation or user config."""
    catalog_argv = ["omp", "models", "--json"]
    get_argv = ["omp", "config", "get", "modelRoles", "--json"]
    set_model_argv = ["omp", "config", "set", "modelRoles"]
    set_level_argv = ["omp", "config", "set", "defaultThinkingLevel", "off"]
    runner = FakeOmpRunner(
        [
            _result(
                catalog_argv,
                stdout=json.dumps({"models": [{"provider": "openai", "id": "gpt-5.6"}]}),
            ),
            _result(get_argv, stdout=_model_roles()),
            _result(set_model_argv),
            _result(set_level_argv),
        ]
    )

    models = list_omp_models(None, runner=runner)
    persist_omp_model(None, "openai", "gpt-5.6", runner=runner)
    persist_omp_thinking_level(None, "off", runner=runner)

    assert [model.model_id for model in models] == ["gpt-5.6"]
    assert runner.calls[0] == catalog_argv
    assert runner.calls[1] == get_argv
    assert runner.calls[2][:-1] == set_model_argv
    assert runner.calls[3] == set_level_argv


@pytest.mark.parametrize(
    ("method", "argv"),
    [
        ("list_models", ["omp", "models", "--json"]),
        ("persist_model", ["omp", "config", "get", "modelRoles", "--json"]),
        ("persist_thinking_level", ["omp", "config", "set", "defaultThinkingLevel", "high"]),
    ],
)
def test_nonzero_omp_commands_raise_typed_errors(
    method: str, argv: list[str]
) -> None:
    """Every OMP command class surfaces a non-zero exit as a typed failure."""
    runner = FakeOmpRunner([_result(argv, stderr="OMP failed", returncode=1)])
    config = OmpProfileConfig(runner=runner)
    action: Callable[[], object]
    if method == "list_models":
        action = config.list_models
    elif method == "persist_model":
        action = partial(config.persist_model, "openai", "gpt-5.6")
    else:
        action = partial(config.persist_thinking_level, "high")

    with pytest.raises(OmpCommandError) as exc_info:
        action()

    assert exc_info.value.returncode == 1
