"""Focused tests for the omp-native harness executor streaming contract."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from omnigent.inner.executor import (
    ExecutorConfig,
    ExecutorEvent,
    TextChunk,
    ToolCallComplete,
    ToolCallRequest,
    TurnComplete,
)
from omnigent.inner.omp_native_executor import OmpNativeExecutor
from omnigent.omp_native_bridge import OMP_NATIVE_BRIDGE_DIR_ENV_VAR


async def _write_json_line(writer: asyncio.StreamWriter, value: Mapping[str, Any]) -> None:
    writer.write((json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8"))
    await writer.drain()


async def _read_json_line(reader: asyncio.StreamReader) -> dict[str, Any]:
    raw = await reader.readline()
    if not raw:
        raise asyncio.IncompleteReadError(b"", expected=1)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


async def _fake_bridge_handler(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    token: str,
    started: asyncio.Event,
) -> None:
    """Authenticate, then stream events and command responses like the real bridge.

    The real bridge broadcasts ``event`` frames before the matching
    ``command_response`` for a ``wait=True`` command, so this fake does too.
    """
    try:
        first = await _read_json_line(reader)
        if first.get("type") != "authenticate" or first.get("token") != token:
            await _write_json_line(writer, {"type": "authentication_failed"})
            return
        await _write_json_line(writer, {"type": "authenticated", "sessionId": "conv_omp"})

        while True:
            command = await _read_json_line(reader)
            if command.get("type") == "close":
                return
            assert command.get("type") == "command"
            payload = command.get("payload", {})
            cmd_type = payload.get("type")
            if cmd_type == "set_system_prompt":
                # Broadcast a stray event before the command response to exercise demux.
                await _write_json_line(
                    writer,
                    {"type": "event", "frame": {"type": "noop"}},
                )
                await _write_json_line(
                    writer,
                    {
                        "type": "command_response",
                        "frame": {"type": "response", "success": True, "id": "sys1"},
                    },
                )
                continue
            if cmd_type == "set_model":
                assert payload.get("provider") == "ollama"
                assert payload.get("modelId") == "kimi-k2.7-code:cloud"
                await _write_json_line(
                    writer,
                    {
                        "type": "command_response",
                        "frame": {"type": "response", "success": True, "id": "m1"},
                    },
                )
                continue
            assert cmd_type == "prompt"
            assert payload.get("message") == "hi"

            # Exercise the demux path: emit events before the matching response.
            await _write_json_line(
                writer,
                {
                    "type": "event",
                    "frame": {
                        "type": "message_update",
                        "assistantMessageEvent": {"type": "text_delta", "delta": "Hello"},
                    },
                },
            )
            await _write_json_line(
                writer,
                {
                    "type": "command_response",
                    "frame": {"type": "response", "success": True, "id": "p1"},
                },
            )

            started.set()

            await _write_json_line(
                writer,
                {
                    "type": "event",
                    "frame": {
                        "type": "tool_execution_start",
                        "toolName": "Bash",
                        "arguments": {"command": "echo hi"},
                    },
                },
            )
            await _write_json_line(
                writer,
                {
                    "type": "event",
                    "frame": {
                        "type": "tool_execution_end",
                        "toolName": "Bash",
                        "status": "success",
                        "result": {"stdout": "hi"},
                    },
                },
            )
            await _write_json_line(
                writer,
                {"type": "event", "frame": {"type": "agent_end", "isTerminal": True}},
            )
            await _write_json_line(writer, {"type": "bridge_closed"})
            return
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def _serve_bridge(server: asyncio.Server, ready: asyncio.Event) -> None:
    async with server:
        ready.set()
        await server.serve_forever()


@contextlib.asynccontextmanager
async def _fake_bridge():
    """Start a fake OMP bridge and yield its bridge directory.

    The directory lives under ``/tmp`` with a short name so the unix socket
    path stays within the OS limit.
    """
    bridge_dir = Path("/tmp") / f"omp_test_{os.getpid()}"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    token = "super-secret"
    (bridge_dir / "token").write_text(token, encoding="utf-8")
    socket_path = bridge_dir / "bridge.sock"

    started = asyncio.Event()

    async def _handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _fake_bridge_handler(reader, writer, token, started)

    server = await asyncio.start_unix_server(_handler, path=str(socket_path))
    ready = asyncio.Event()
    task = asyncio.create_task(_serve_bridge(server, ready))
    await asyncio.wait_for(ready.wait(), timeout=5.0)
    try:
        yield bridge_dir
    finally:
        server.close()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        import shutil

        shutil.rmtree(bridge_dir, ignore_errors=True)


async def _collect(iterator: AsyncIterator[ExecutorEvent]) -> list[ExecutorEvent]:
    return [event async for event in iterator]


@pytest.mark.anyio
async def test_terminal_client_connects_to_fake_bridge() -> None:
    """Sanity-check that the test harness can be reached."""
    from omnigent.omp_native_terminal import OmpNativeTerminalClient

    async with _fake_bridge() as bridge_dir:
        client = OmpNativeTerminalClient(bridge_dir)
        await asyncio.wait_for(client.connect(), timeout=5.0)
        await client.close()


@pytest.mark.anyio
async def test_native_executor_streams_bridge_events() -> None:
    """The executor connects to the bridge, sends system/model/prompt, and translates events."""
    async with _fake_bridge() as bridge_dir:
        os.environ[OMP_NATIVE_BRIDGE_DIR_ENV_VAR] = str(bridge_dir)
        executor = OmpNativeExecutor()

        events = await _collect(
            executor.run_turn(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                system_prompt="You are a helpful assistant.",
                config=ExecutorConfig(model="ollama/kimi-k2.7-code:cloud"),
            )
        )

        assert executor.supports_streaming() is True
        assert executor.handles_tools_internally() is True

        chunks = [e for e in events if isinstance(e, TextChunk)]
        requests = [e for e in events if isinstance(e, ToolCallRequest)]
        completes = [e for e in events if isinstance(e, ToolCallComplete)]
        finals = [e for e in events if isinstance(e, TurnComplete)]

        assert [c.text for c in chunks] == ["Hello"]
        assert len(requests) == 1 and requests[0].name == "Bash"
        assert len(completes) == 1 and completes[0].status.name == "SUCCESS"
        assert len(finals) == 1 and finals[0].response == "Hello"
