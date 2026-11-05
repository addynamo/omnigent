"""Tests for the omp-native terminal client's bridge protocol handling."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from omnigent.omp_native_terminal import OmpNativeTerminalClient

_BridgeHandler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


async def _write_json_line(writer: asyncio.StreamWriter, value: dict[str, Any]) -> None:
    writer.write((json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8"))
    await writer.drain()


async def _read_json_line(reader: asyncio.StreamReader) -> dict[str, Any]:
    raw = await reader.readline()
    if not raw:
        raise asyncio.IncompleteReadError(partial=b"", expected=1)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


@contextlib.asynccontextmanager
async def _bridge_with_handler(handler: _BridgeHandler, tmp_path: Path) -> AsyncIterator[Path]:
    """Run a scripted bridge socket that authenticates, then defers to *handler*.

    The directory lives under ``/tmp`` with a short name so the unix socket
    path stays within the OS limit.
    """
    del tmp_path
    bridge_dir = Path("/tmp") / f"omp_term_{os.getpid()}"
    shutil.rmtree(bridge_dir, ignore_errors=True)
    bridge_dir.mkdir()
    token = "super-secret"
    (bridge_dir / "token").write_text(token, encoding="utf-8")

    async def _serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            first = await _read_json_line(reader)
            assert first.get("type") == "authenticate" and first.get("token") == token
            await _write_json_line(writer, {"type": "authenticated", "sessionId": "conv_omp"})
            await handler(reader, writer)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    server = await asyncio.start_unix_server(_serve, path=str(bridge_dir / "bridge.sock"))
    try:
        yield bridge_dir
    finally:
        server.close()
        await server.wait_closed()
        shutil.rmtree(bridge_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_send_command_registers_waiter_before_write(tmp_path: Path) -> None:
    """An immediate bridge reply must not race ahead of the pending waiter."""

    async def _instant_reply(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        message = await _read_json_line(reader)
        assert message.get("type") == "command"
        await _write_json_line(
            writer, {"type": "command_response", "frame": {"type": "response", "success": True}}
        )

    async with _bridge_with_handler(_instant_reply, tmp_path) as bridge_dir:
        client = OmpNativeTerminalClient(bridge_dir)
        await client.connect()
        try:
            frame = await asyncio.wait_for(
                client.send_command({"type": "set_model"}, wait=True), timeout=5.0
            )
        finally:
            await client.close()
    assert frame == {"type": "response", "success": True}
    assert not client._pending_responses


@pytest.mark.asyncio
async def test_send_command_removes_waiter_when_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed write must not leave an orphaned waiter to desync later replies."""
    from omnigent import omp_native_terminal

    async def _hang(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read()

    async with _bridge_with_handler(_hang, tmp_path) as bridge_dir:
        client = OmpNativeTerminalClient(bridge_dir)
        await client.connect()

        async def _failing_write(writer: asyncio.StreamWriter, value: Any) -> None:
            raise ConnectionError("write failed")

        monkeypatch.setattr(omp_native_terminal, "_write_json_line", _failing_write)
        try:
            with pytest.raises(ConnectionError):
                await client.send_command({"type": "set_model"}, wait=True)
            assert not client._pending_responses
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_socket_close_wakes_pending_waiters_and_emits_bridge_closed(
    tmp_path: Path,
) -> None:
    """A dropped bridge connection must fail pending commands instead of hanging."""

    async def _no_reply(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Consume the command, then drop the connection with no bridge_closed frame.
        await _read_json_line(reader)

    async with _bridge_with_handler(_no_reply, tmp_path) as bridge_dir:
        client = OmpNativeTerminalClient(bridge_dir)
        await client.connect()
        try:
            task = asyncio.create_task(client.send_command({"type": "set_model"}, wait=True))
            with pytest.raises(ConnectionError):
                await asyncio.wait_for(task, timeout=5.0)
            events = client.events()
            message = await asyncio.wait_for(events.__anext__(), timeout=5.0)
        finally:
            await client.close()
    assert message.get("type") == "bridge_closed"
