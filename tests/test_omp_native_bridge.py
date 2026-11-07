"""Focused contract tests for the terminal-first OMP RPC bridge."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from omnigent.omp_native_bridge import (
    OMP_NATIVE_BRIDGE_DIR_ENV_VAR,
    OMP_NATIVE_REQUEST_SESSION_ID_ENV_VAR,
    OmpNativeResumeError,
    OmpRpcBridge,
    bridge_dir_for_session_id,
    build_omp_native_spawn_env,
)
from omnigent.omp_native_terminal import OmpNativeTerminalClient


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


class _FakeOmpStdin:
    """Captures bridge writes and answers every command with a correlated response."""

    def __init__(self, stdout: asyncio.StreamReader) -> None:
        self._stdout = stdout
        self.writes: list[dict[str, Any]] = []

    def is_closing(self) -> bool:
        return False

    def write(self, data: bytes) -> None:
        frame = json.loads(data)
        self.writes.append(frame)
        request_id = frame.get("id")
        if isinstance(request_id, str):
            response = {"type": "response", "id": request_id, "success": True, "data": {}}
            self._stdout.feed_data((json.dumps(response) + "\n").encode("utf-8"))

    async def drain(self) -> None:
        return None


class _FakeOmpProcess:
    """Scriptable stand-in for the ``omp --mode rpc`` subprocess."""

    returncode: int | None = None
    stderr = None

    def __init__(self, limit: int | None) -> None:
        # Mirror create_subprocess_exec: the factory's ``limit`` sizes the stream.
        self.stdout = asyncio.StreamReader(limit=limit)
        self.stdin = _FakeOmpStdin(self.stdout)

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        self.returncode = self.returncode if self.returncode is not None else 0
        return self.returncode


def _start_bridge_with_fake_omp() -> tuple[OmpRpcBridge, dict[str, _FakeOmpProcess], Path]:
    """Start a real bridge over a short /tmp socket path backed by a fake OMP."""
    created: dict[str, _FakeOmpProcess] = {}

    async def _factory(*args: Any, **kwargs: Any) -> _FakeOmpProcess:
        del args
        process = _FakeOmpProcess(kwargs.get("limit"))
        process.stdout.feed_data(
            (json.dumps({"type": "ready", "protocolVersion": 1}) + "\n").encode("utf-8")
        )
        created["process"] = process
        return process

    bridge_dir = Path(tempfile.mkdtemp(prefix="omp_bridge_"))
    bridge = OmpRpcBridge(
        session_id="conv_oversized",
        bridge_dir=bridge_dir,
        executable="omp",
        process_factory=_factory,
    )
    return bridge, created, bridge_dir


async def test_bridge_relays_oversized_omp_frames_without_breaking() -> None:
    """OMP frames exceed asyncio's 64 KiB line default (command catalogs, state)."""
    bridge, created, bridge_dir = _start_bridge_with_fake_omp()
    await bridge.start()
    process = created["process"]
    client = OmpNativeTerminalClient(bridge_dir)
    await client.connect()
    huge = "x" * (200 * 1024)
    try:
        process.stdout.feed_data(
            (
                json.dumps({"type": "available_commands_update", "text": huge}) + "\n"
            ).encode("utf-8")
        )
        process.stdout.feed_eof()
        events: list[dict[str, Any]] = []

        async def _collect() -> None:
            async for event in client.events():
                events.append(event)
                if event.get("type") == "bridge_closed":
                    return

        await asyncio.wait_for(_collect(), timeout=10.0)
    finally:
        await client.close()
        await bridge.close()
        shutil.rmtree(bridge_dir, ignore_errors=True)

    kinds = [event.get("type") for event in events]
    assert "bridge_error" not in kinds
    assert any(
        event.get("frame", {}).get("text") == huge
        for event in events
        if event.get("type") == "event"
    )
    assert kinds[-1] == "bridge_closed"


async def test_bridge_accepts_oversized_client_prompts() -> None:
    """A pasted prompt can exceed asyncio's 64 KiB line default without dying."""
    bridge, created, bridge_dir = _start_bridge_with_fake_omp()
    await bridge.start()
    process = created["process"]
    client = OmpNativeTerminalClient(bridge_dir)
    await client.connect()
    huge = "y" * (200 * 1024)
    try:
        await client.send_command({"type": "prompt", "message": huge}, wait=True)
    finally:
        await client.close()
        await bridge.close()
        shutil.rmtree(bridge_dir, ignore_errors=True)

    assert any(frame.get("message") == huge for frame in process.stdin.writes)
