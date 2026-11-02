"""Focused tests for the terminal-first OMP native wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.omp_native import (
    OmpNativeResumeError,
    _materialize_omp_agent_spec,
    build_omp_native_launch,
)


def test_materialized_agent_is_terminal_first(tmp_path: Path) -> None:
    """The native wrapper has its own identity and never reuses headless ``omp``."""
    spec = _materialize_omp_agent_spec(tmp_path)

    content = spec.read_text(encoding="utf-8")
    assert "omp-native-ui" in content
    assert "omp-native" in content
    assert "terminal" in content


def test_native_launch_uses_exact_resume_argv(tmp_path: Path) -> None:
    """Terminal relaunches ask OMP to resume its stored session file exactly."""
    session_file = tmp_path / "omp-session.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")

    launch = build_omp_native_launch(session_file=session_file)

    assert launch.argv == ["omp", "--mode", "rpc", "--resume", str(session_file)]


def test_native_launch_fails_before_start_for_unreadable_resume(tmp_path: Path) -> None:
    """Resume errors point at the stored sessionFile and avoid transcript fallback."""
    with pytest.raises(OmpNativeResumeError, match="sessionFile"):
        build_omp_native_launch(session_file=tmp_path / "gone.jsonl")
