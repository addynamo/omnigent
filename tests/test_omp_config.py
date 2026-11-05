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


def test_persist_model_accepts_tagged_ollama_identifiers() -> None:
    """Ollama-style tagged model ids like ``kimi-k2.7-code:cloud`` are valid."""
    get_argv = ["omp", "config", "get", "modelRoles", "--json"]
    set_argv = ["omp", "config", "set", "modelRoles"]
    runner = FakeOmpRunner([_result(get_argv, stdout=_model_roles()), _result(set_argv)])

    OmpProfileConfig(runner=runner).persist_model("ollama", "kimi-k2.7-code:cloud")

    assert json.loads(runner.calls[1][-1]) == {
        "default": {"provider": "ollama", "model": "kimi-k2.7-code:cloud"},
        "background": {"provider": "openai", "model": "gpt-5.2"},
    }


def test_persist_model_accepts_vendor_scoped_identifiers() -> None:
    """OpenRouter-style ids like ``~anthropic/claude-opus-latest`` are valid."""
    get_argv = ["omp", "config", "get", "modelRoles", "--json"]
    set_argv = ["omp", "config", "set", "modelRoles"]
    runner = FakeOmpRunner([_result(get_argv, stdout=_model_roles()), _result(set_argv)])

    OmpProfileConfig(runner=runner).persist_model("openrouter", "~anthropic/claude-opus-latest")

    assert json.loads(runner.calls[1][-1])["default"] == {
        "provider": "openrouter",
        "model": "~anthropic/claude-opus-latest",
    }


def test_list_models_accepts_vendor_scoped_model_ids() -> None:
    """Catalogs with OpenRouter vendor-scoped ids must still produce options."""
    argv = ["omp", "models", "--json"]
    catalog = [
        {"provider": "openai", "id": "gpt-5.6"},
        {"provider": "openrouter", "id": "ai21/jamba-large-1.7"},
        {"provider": "openrouter", "id": "~anthropic/claude-opus-latest"},
        {"provider": "ollama", "id": "x/flux2-klein:latest"},
    ]
    runner = FakeOmpRunner([_result(argv, stdout=json.dumps(catalog))])

    models = OmpProfileConfig(runner=runner).list_models()

    assert [(model.provider, model.model_id) for model in models] == [
        ("openai", "gpt-5.6"),
        ("openrouter", "ai21/jamba-large-1.7"),
        ("openrouter", "~anthropic/claude-opus-latest"),
        ("ollama", "x/flux2-klein:latest"),
    ]


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
    [
        ("", "gpt-5.6"),
        ("open ai", "gpt-5.6"),
        ("openai", "gpt 5.6"),
        ("openai", "-gpt-5.6"),
        ("openai", "/gpt-5.6"),
        ("openai", ""),
    ],
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
def test_nonzero_omp_commands_raise_typed_errors(method: str, argv: list[str]) -> None:
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


def _wrapped_model_roles() -> str:
    """Real ``omp config get modelRoles --json`` output wraps the record in an envelope."""
    return json.dumps(
        {
            "key": "modelRoles",
            "value": {
                "default": "ollama/kimi-k2.7-code:cloud:high",
                "slow": "ollama/deepseek-v4-flash:cloud:high",
            },
            "type": "record",
            "description": "",
        }
    )


def test_persist_model_unwraps_real_config_envelope() -> None:
    """Saving the default model must write the role map, never OMP's JSON envelope."""
    get_argv = ["omp", "config", "get", "modelRoles", "--json"]
    set_argv = ["omp", "config", "set", "modelRoles"]
    runner = FakeOmpRunner([_result(get_argv, stdout=_wrapped_model_roles()), _result(set_argv)])

    OmpProfileConfig(runner=runner).persist_model("openai", "gpt-5.6")

    assert json.loads(runner.calls[1][-1]) == {
        "default": "openai/gpt-5.6",
        "slow": "ollama/deepseek-v4-flash:cloud:high",
    }


def test_persist_role_sets_one_role_with_effort_suffix() -> None:
    """Assigning a role writes ``selector:effort`` for that role and keeps the others."""
    get_argv = ["omp", "config", "get", "modelRoles", "--json"]
    set_argv = ["omp", "config", "set", "modelRoles"]
    runner = FakeOmpRunner([_result(get_argv, stdout=_wrapped_model_roles()), _result(set_argv)])

    OmpProfileConfig(runner=runner).persist_role(
        "plan", "openrouter/~anthropic/claude-opus-latest:max"
    )

    assert runner.calls[1][:4] == set_argv
    assert json.loads(runner.calls[1][-1]) == {
        "default": "ollama/kimi-k2.7-code:cloud:high",
        "slow": "ollama/deepseek-v4-flash:cloud:high",
        "plan": "openrouter/~anthropic/claude-opus-latest:max",
    }


@pytest.mark.parametrize(
    ("role", "selector"),
    [
        ("", "openai/gpt-5.6"),
        ("bad role", "openai/gpt-5.6"),
        ("-default", "openai/gpt-5.6"),
        ("slow", "gpt-5.6"),  # selectors must be provider-qualified
        ("slow", "openai/-gpt"),
        ("slow", "open ai/gpt-5.6"),
    ],
)
def test_persist_role_rejects_invalid_values(role: str, selector: str) -> None:
    """Invalid role names or selectors fail before OMP is run."""
    runner = FakeOmpRunner([])

    with pytest.raises(OmpValidationError):
        OmpProfileConfig(runner=runner).persist_role(role, selector)

    assert runner.calls == []


def test_list_models_all_kinds_carries_kind_and_effort_ladder() -> None:
    """``kind="all"`` lists every catalog kind with each model's supported efforts."""
    argv = ["omp", "models", "--kind", "all", "--json"]
    catalog = [
        {
            "provider": "ollama",
            "id": "deepseek-v4-flash:cloud",
            "kind": "chat",
            "thinking": ["low", "high", "max"],
        },
        {"provider": "openai-codex", "id": "gpt-image-2", "kind": "image"},
        {"provider": "web", "id": "brave", "kind": "search"},
    ]
    runner = FakeOmpRunner([_result(argv, stdout=json.dumps(catalog))])

    models = OmpProfileConfig(runner=runner).list_models(kind="all")

    assert runner.calls == [argv]
    assert [(m.provider, m.model_id, m.kind, m.thinking) for m in models] == [
        ("ollama", "deepseek-v4-flash:cloud", "chat", ["low", "high", "max"]),
        ("openai-codex", "gpt-image-2", "image", []),
        ("web", "brave", "search", []),
    ]


@pytest.mark.parametrize("level", ["max", "auto"])
def test_persist_thinking_level_accepts_omp18_levels(level: str) -> None:
    """OMP 18 adds ``max`` and ``auto`` as default thinking levels."""
    argv = ["omp", "config", "set", "defaultThinkingLevel", level]
    runner = FakeOmpRunner([_result(argv)])

    OmpProfileConfig(runner=runner).persist_thinking_level(level)

    assert runner.calls == [argv]
