"""Focused contract tests for the terminal-first OMP RPC bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.omp_native_bridge import (
    OMP_NATIVE_BRIDGE_DIR_ENV_VAR,
    OMP_NATIVE_REQUEST_SESSION_ID_ENV_VAR,
    OmpNativeResumeError,
    OmpRpcBridge,
    bridge_dir_for_session_id,
    build_omp_native_spawn_env,
)


def test_bridge_dir_and_spawn_env_are_session_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner and bridge derive the same isolated location for one conversation."""
    import omnigent.omp_native_bridge as bridge_module

    monkeypatch.setattr(bridge_module, "_BRIDGE_ROOT", tmp_path / "omp-native")

    bridge_dir = bridge_dir_for_session_id("conv_omp")
    env = build_omp_native_spawn_env("conv_omp")

    assert bridge_dir.parent == tmp_path / "omp-native"
    assert bridge_dir != bridge_dir_for_session_id("conv_other")
    assert env == {
        OMP_NATIVE_BRIDGE_DIR_ENV_VAR: str(bridge_dir),
        OMP_NATIVE_REQUEST_SESSION_ID_ENV_VAR: "conv_omp",
    }


def test_bridge_launches_only_rpc_mode_and_resumes_exact_session_file(tmp_path: Path) -> None:
    """A durable OMP resume is exact argv, never reconstructed transcript context."""
    session_file = tmp_path / "session.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")

    bridge = OmpRpcBridge(
        session_id="conv_omp",
        executable="omp",
        session_file=session_file,
    )

    assert bridge.command == ["omp", "--mode", "rpc", "--resume", str(session_file)]


def test_bridge_rejects_missing_resume_file_before_launch(tmp_path: Path) -> None:
    """A missing durable OMP session gives an actionable error instead of replaying chat."""
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(OmpNativeResumeError, match=r"sessionFile.*missing|missing.*sessionFile"):
        OmpRpcBridge(session_id="conv_omp", session_file=missing)
