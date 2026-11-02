"""Terminal client for an :mod:`omnigent.omp_native_bridge` socket.

This module deliberately never opens OMP's stdin/stdout.  It only speaks the
owner-authenticated local bridge protocol, allowing the terminal and browser to
remain attached to the same bridge-owned OMP process.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from omnigent.omp_native_bridge import _read_json_line, _write_json_line


class OmpNativeTerminalClient:
    """Authenticated local-socket terminal client for a running OMP bridge."""

    def __init__(self, bridge_dir: Path) -> None:
        self.bridge_dir = bridge_dir
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending_responses: asyncio.Queue[asyncio.Future[dict[str, Any]]] = asyncio.Queue()

    async def connect(self) -> None:
        """Connect and authenticate using the bridge's owner-only bearer token."""
        self._reader, self._writer = await asyncio.open_unix_connection(
            str(self.bridge_dir / "bridge.sock")
        )
        token = (self.bridge_dir / "token").read_text(encoding="utf-8")
        await _write_json_line(self._writer, {"type": "authenticate", "token": token})
        reply = await _read_json_line(self._reader)
        if reply.get("type") != "authenticated":
            raise RuntimeError("OMP native bridge authentication failed")
        self._reader_task = asyncio.create_task(
            self._read_loop(), name="omp-native-terminal-reader"
        )

    async def send_command(
        self, payload: dict[str, Any], *, wait: bool = False
    ) -> dict[str, Any] | None:
        """Send a command *to the bridge*, never directly to OMP stdio.

        When *wait* is True, blocks until the bridge returns the matching OMP
        command response and returns the unwrapped OMP frame.
        """
        if self._writer is None:
            raise RuntimeError("OMP native terminal client is not connected")
        await _write_json_line(self._writer, {"type": "command", "payload": payload, "wait": wait})
        if not wait:
            return None
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        await self._pending_responses.put(future)
        try:
            response = await future
        except asyncio.CancelledError:
            self._pending_responses = asyncio.Queue()
            raise
        if response.get("type") == "error":
            raise RuntimeError(response.get("error", "OMP bridge command failed"))
        if response.get("type") != "command_response":
            raise RuntimeError(f"Unexpected bridge response type: {response.get('type')}")
        return response.get("frame")

    async def events(self):
        """Yield bridge events in the exact order published by the bridge."""
        if self._reader is None:
            raise RuntimeError("OMP native terminal client is not connected")
        while True:
            yield await self._event_queue.get()

    async def close(self) -> None:
        """Disconnect without touching the bridge-owned OMP process."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
            self._reader_task = None
        if self._writer is not None:
            with contextlib.suppress(Exception):
                await _write_json_line(self._writer, {"type": "close"})
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        self._writer = None
        self._reader = None

    async def _read_loop(self) -> None:
        """Demultiplex bridge messages into command responses and events."""
        reader = self._reader
        if reader is None:
            return
        try:
            while True:
                message = await _read_json_line(reader)
                msg_type = message.get("type")
                if msg_type == "command_response" or msg_type == "error":
                    try:
                        pending = self._pending_responses.get_nowait()
                    except asyncio.QueueEmpty:
                        # No waiter for this response; treat errors as events and
                        # drop orphaned command_response frames silently.
                        if msg_type == "error":
                            await self._event_queue.put(message)
                        continue
                    if not pending.done():
                        pending.set_result(message)
                else:
                    await self._event_queue.put(message)
        except asyncio.CancelledError:
            raise
        except (ConnectionError, OSError, ValueError, asyncio.IncompleteReadError):
            # Stop the reader; consumers will see the queue drain and bridge_closed.
            return


async def _forward_stdin(client: OmpNativeTerminalClient) -> None:
    """Read terminal stdin lines and forward them to the bridge as OMP prompts."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
        except (KeyboardInterrupt, EOFError):
            return
        if not line:
            return
        text = line.rstrip("\n")
        if text:
            await client.send_command({"type": "prompt", "message": text}, wait=False)


async def run_omp_native_terminal(
    *,
    bridge_dir: Path,
    prompt: str | None = None,
    output: TextIO | None = None,
) -> None:
    """Render bridge events until the bridge closes; no direct OMP process is started."""
    output = output or sys.stdout
    client = OmpNativeTerminalClient(bridge_dir)
    await client.connect()
    if prompt is not None:
        await client.send_command({"type": "prompt", "message": prompt}, wait=False)
    try:
        stdin_task = asyncio.create_task(_forward_stdin(client))
        try:
            async for message in client.events():
                print(json.dumps(message, ensure_ascii=False), file=output, flush=True)
                if message.get("type") == "bridge_closed":
                    return
        finally:
            stdin_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stdin_task
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> None:
    """Attach the tmux terminal to the bridge created by runner lifecycle code."""
    parser = argparse.ArgumentParser(prog="python -m omnigent.omp_native_terminal")
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args(argv)
    asyncio.run(run_omp_native_terminal(bridge_dir=Path(args.bridge_dir), prompt=args.prompt))


if __name__ == "__main__":
    main()
