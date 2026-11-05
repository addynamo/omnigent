"""Tests for OMP-native terminal orchestration resume and metadata persistence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from omnigent.omp_native_bridge import OmpNativeResumeError, OmpRpcBridge
from omnigent.runner.native import orchestration


def _fake_launch_config(tmp_path: Path, external_session_id: str | None) -> Any:
    async def _config(*, session_id: str, server_client: Any) -> Any:
        del session_id, server_client
        return SimpleNamespace(
            workspace=tmp_path,
            external_session_id=external_session_id,
            terminal_launch_args=[],
            model_override=None,
            labels={},
        )

    return _config


class _RecordingServerClient:
    """Records PATCH bodies and answers with a configurable status."""

    class _Resp:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.patches: list[dict[str, Any]] = []

    async def patch(self, url: str, **kwargs: Any) -> Any:
        del url
        body = kwargs.get("json")
        if isinstance(body, dict):
            self.patches.append(body)
        return self._Resp(self.status_code)


@pytest.mark.asyncio
async def test_auto_create_refuses_missing_external_session_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A recorded sessionFile that vanished must raise instead of cold-starting."""
    missing = tmp_path / "gone.jsonl"
    monkeypatch.setattr(
        orchestration, "_pi_native_launch_config", _fake_launch_config(tmp_path, str(missing))
    )
    monkeypatch.setattr(OmpRpcBridge, "stored_session_file", lambda bridge_dir: None)

    with pytest.raises(OmpNativeResumeError, match="sessionFile"):
        await orchestration._auto_create_omp_terminal(
            "conv_omp_resume_lost",
            resource_registry=None,  # type: ignore[arg-type]
            publish_event=lambda *_args: None,
            server_client=None,
        )


@pytest.mark.asyncio
async def test_auto_create_refuses_stale_bridge_state_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A captured sessionFile that no longer exists must raise, not launch fresh."""
    missing = tmp_path / "gone.jsonl"
    monkeypatch.setattr(
        orchestration, "_pi_native_launch_config", _fake_launch_config(tmp_path, None)
    )
    monkeypatch.setattr(OmpRpcBridge, "stored_session_file", lambda bridge_dir: missing)

    with pytest.raises(OmpNativeResumeError, match="sessionFile"):
        await orchestration._auto_create_omp_terminal(
            "conv_omp_stale_state",
            resource_registry=None,  # type: ignore[arg-type]
            publish_event=lambda *_args: None,
            server_client=None,
        )


@pytest.mark.asyncio
async def test_auto_create_accepts_existing_session_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An existing resume file passes validation and the launch continues."""
    session_file = tmp_path / "omp-session.jsonl"
    session_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        orchestration, "_pi_native_launch_config", _fake_launch_config(tmp_path, str(session_file))
    )
    monkeypatch.setattr(OmpRpcBridge, "stored_session_file", lambda bridge_dir: None)

    def _stop_here() -> str:
        raise RuntimeError("stop-here")

    monkeypatch.setattr("omnigent.inner.omp_executor.resolve_omp_executable", _stop_here)
    with pytest.raises(RuntimeError, match="stop-here"):
        await orchestration._auto_create_omp_terminal(
            "conv_omp_resume_ok",
            resource_registry=None,  # type: ignore[arg-type]
            publish_event=lambda *_args: None,
            server_client=None,
        )


@pytest.mark.asyncio
async def test_persist_session_file_metadata_patches_external_session_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The captured sessionFile is recorded as the session's external_session_id."""
    session_file = tmp_path / "omp-session.jsonl"
    monkeypatch.setattr(OmpRpcBridge, "stored_session_file", lambda bridge_dir: session_file)
    server_client = _RecordingServerClient()

    await orchestration._persist_omp_session_file_metadata(
        server_client,  # type: ignore[arg-type]
        "conv_omp_persist",
        tmp_path,
        timeout_s=0.1,
        poll_interval_s=0.01,
    )

    assert server_client.patches == [{"external_session_id": str(session_file)}]


@pytest.mark.asyncio
async def test_persist_session_file_metadata_skips_uncaptured_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No capture within the deadline means no PATCH and no error."""
    monkeypatch.setattr(OmpRpcBridge, "stored_session_file", lambda bridge_dir: None)
    server_client = _RecordingServerClient()

    await orchestration._persist_omp_session_file_metadata(
        server_client,  # type: ignore[arg-type]
        "conv_omp_uncaptured",
        tmp_path,
        timeout_s=0.05,
        poll_interval_s=0.01,
    )

    assert server_client.patches == []


@pytest.mark.asyncio
async def test_persist_session_file_metadata_survives_rejected_patch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A set-once conflict (400) is logged, never raised into the runner loop."""
    session_file = tmp_path / "omp-session.jsonl"
    monkeypatch.setattr(OmpRpcBridge, "stored_session_file", lambda bridge_dir: session_file)
    server_client = _RecordingServerClient(status_code=400)

    await orchestration._persist_omp_session_file_metadata(
        server_client,  # type: ignore[arg-type]
        "conv_omp_rejected",
        tmp_path,
        timeout_s=0.1,
        poll_interval_s=0.01,
    )

    assert server_client.patches == [{"external_session_id": str(session_file)}]
