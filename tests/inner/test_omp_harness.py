"""Tests for the OMP headless harness wrap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI

from omnigent.inner.datamodel import OSEnvSandboxSpec, OSEnvSpec
from omnigent.inner.omp_executor import OmpExecutor
from omnigent.inner.omp_harness import _build_omp_executor, create_app
from omnigent.runtime.workflow import _build_omp_spawn_env
from omnigent.spec.types import AgentSpec, ExecutorSpec


@pytest.fixture
def clean_omp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "HARNESS_OMP_CWD",
        "HARNESS_OMP_OS_ENV",
        "HARNESS_OMP_SKILLS_FILTER",
        "HARNESS_OMP_BUNDLE_DIR",
        "HARNESS_OMP_AGENT_NAME",
        "OMNIGENT_OMP_PATH",
        "HARNESS_OMP_PATH",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_create_app_returns_fastapi() -> None:
    app = create_app()
    assert isinstance(app, FastAPI)


def test_build_omp_executor_uses_env_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_omp_env: None
) -> None:
    monkeypatch.setenv("OMNIGENT_OMP_PATH", "/fake/bin/omp")
    monkeypatch.setenv("HARNESS_OMP_CWD", str(tmp_path))
    monkeypatch.setenv("HARNESS_OMP_AGENT_NAME", "test-agent")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setenv("HARNESS_OMP_BUNDLE_DIR", str(bundle))
    monkeypatch.setenv("HARNESS_OMP_SKILLS_FILTER", '["core"]')
    monkeypatch.setenv(
        "HARNESS_OMP_OS_ENV",
        '{"type":"caller_process","sandbox":{"type":"none"},"fork":false}',
    )

    executor = _build_omp_executor()
    assert isinstance(executor, OmpExecutor)
    assert executor._cwd == tmp_path
    assert executor.command == ["/fake/bin/omp", "--mode", "rpc"]


def test_build_omp_spawn_env(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    spec = AgentSpec(
        spec_version=1,
        name="agent-x",
        executor=ExecutorSpec(model="openai/gpt-5.3-codex"),
        skills_filter=["core", "extra"],
        os_env=OSEnvSpec(
            type="caller_process",
            cwd=str(tmp_path),
            sandbox=OSEnvSandboxSpec(type="none"),
            fork=False,
        ),
    )
    env = _build_omp_spawn_env(spec, cwd=tmp_path, workdir=bundle)

    assert env["HARNESS_OMP_MODEL"] == "openai/gpt-5.3-codex"
    assert env["HARNESS_OMP_AGENT_NAME"] == "agent-x"
    assert env["HARNESS_OMP_CWD"] == str(tmp_path)
    assert env["HARNESS_OMP_BUNDLE_DIR"] == str(bundle)
    assert env["HARNESS_OMP_SKILLS_FILTER"] == '["core", "extra"]'
    serialized_os_env = json.loads(env["HARNESS_OMP_OS_ENV"])
    assert serialized_os_env["type"] == "caller_process"
    assert serialized_os_env["cwd"] == str(tmp_path)
    assert serialized_os_env["sandbox"]["type"] == "none"
    assert serialized_os_env["fork"] is False


def test_build_omp_spawn_env_without_optional_fields(tmp_path: Path) -> None:
    spec = AgentSpec(spec_version=1, name="minimal", executor=ExecutorSpec())
    env = _build_omp_spawn_env(spec, cwd=None, workdir=None)
    assert env.get("HARNESS_OMP_MODEL") is None
    assert "HARNESS_OMP_CWD" not in env
    assert "HARNESS_OMP_BUNDLE_DIR" not in env
    assert env["HARNESS_OMP_AGENT_NAME"] == "minimal"
    assert env["HARNESS_OMP_SKILLS_FILTER"] == '"all"'


def test_build_omp_spawn_env_uses_cwd_override() -> None:
    """The ``cwd`` kwarg wins and is forwarded as HARNESS_OMP_CWD."""
    spec = AgentSpec(spec_version=1, name="cwd-test", executor=ExecutorSpec())
    cwd = Path("/tmp/custom")
    env = _build_omp_spawn_env(spec, cwd=cwd, workdir=None)
    assert env["HARNESS_OMP_CWD"] == str(cwd)
