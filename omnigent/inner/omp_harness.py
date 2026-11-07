"""OMP harness wrap: drives the headless ``omp`` RPC executor.

This module is the runtime entry point for the ``harness: omp`` headless
worker used by Polly, Debby, and other Omnigent agents. It mirrors the
shape of ``omnigent/inner/pi_harness.py`` but omits Pi's gateway and
Databricks-specific configuration; OMP's own profile/model/thinking
configuration remains authoritative.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI

from omnigent.inner.datamodel import OSEnvSandboxSpec, OSEnvSpec
from omnigent.inner.executor import Executor
from omnigent.inner.omp_executor import OmpExecutor
from omnigent.runtime.harnesses._executor_adapter import ExecutorAdapter

_logger = logging.getLogger(__name__)

# Env-var keys the wrap reads at executor construction time. Centralizing as
# constants so misconfigurations surface as a single grep target.
_ENV_CWD = "HARNESS_OMP_CWD"
_ENV_OS_ENV = "HARNESS_OMP_OS_ENV"
_ENV_SKILLS_FILTER = "HARNESS_OMP_SKILLS_FILTER"
_ENV_BUNDLE_DIR = "HARNESS_OMP_BUNDLE_DIR"
_ENV_AGENT_NAME = "HARNESS_OMP_AGENT_NAME"
_ENV_OMP_PATH = "OMNIGENT_OMP_PATH"
# Deprecated alias — read via resolve_harness_path() which warns on use.
_LEGACY_ENV_OMP_PATH = "HARNESS_OMP_PATH"

# Truthy strings matching the claude-sdk/codex/pi wraps' boolean parsers.
_TRUTHY_STRINGS = ("1", "true", "yes")


def _parse_truthy(env_var: str, default: bool) -> bool:
    raw = os.environ.get(env_var, "").strip().lower()
    if not raw:
        return default
    return raw in _TRUTHY_STRINGS


def _resolve_os_env() -> OSEnvSpec:
    """Resolve the inner-executor :class:`OSEnvSpec` from env config."""
    raw = os.environ.get(_ENV_OS_ENV, "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            _logger.warning(
                "%s is not valid JSON (%s); falling back to default os_env",
                _ENV_OS_ENV,
                exc,
            )
            payload = None
        if isinstance(payload, dict):
            sandbox_payload = payload.get("sandbox")
            sandbox = (
                OSEnvSandboxSpec(**sandbox_payload) if isinstance(sandbox_payload, dict) else None
            )
            return OSEnvSpec(
                type=str(payload.get("type", "caller_process")),
                cwd=payload.get("cwd"),
                sandbox=sandbox,
                fork=bool(payload.get("fork", False)),
            )
    # Default: enable natives, no sandbox. Matches Pi's default and keeps the
    # harness usable without an explicit os_env block.
    return OSEnvSpec(
        type="caller_process",
        cwd=None,
        sandbox=OSEnvSandboxSpec(type="none"),
        fork=False,
    )


def _resolve_skills_filter() -> str | list[str]:
    """Resolve the inner-executor ``skills_filter`` from env config."""
    raw = os.environ.get(_ENV_SKILLS_FILTER, "").strip()
    if not raw:
        return "all"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw if raw in {"all", "none"} else [raw]
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in {"all", "none"}:
        return value
    return str(value)


def _build_omp_executor() -> Executor:
    """Construct an :class:`OmpExecutor` from env-var config."""
    from omnigent.harness_startup_config import resolve_harness_path

    bundle_dir_raw = os.environ.get(_ENV_BUNDLE_DIR, "").strip()
    bundle_dir = Path(bundle_dir_raw) if bundle_dir_raw else None
    agent_name_raw = os.environ.get(_ENV_AGENT_NAME, "").strip()
    agent_name = agent_name_raw or None

    omp_path = resolve_harness_path("omp") or "omp"
    command = [omp_path, "--mode", "rpc"]

    cwd = os.environ.get(_ENV_CWD) or os.environ.get("OMNIGENT_RUNNER_WORKSPACE")
    return OmpExecutor(
        command=command,
        cwd=Path(cwd) if cwd else None,
        os_env=_resolve_os_env(),
        model=os.environ.get("HARNESS_OMP_MODEL") or None,
        bundle_dir=bundle_dir,
        agent_name=agent_name,
        skills_filter=_resolve_skills_filter(),
        extra_args=[],
    )


def create_app() -> FastAPI:
    """Build the harness FastAPI app used by the runner process manager."""
    adapter = ExecutorAdapter(_build_omp_executor, harness_label="OMP")
    return adapter.build()
