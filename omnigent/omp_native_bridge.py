"""Authenticated local RPC bridge for one durable terminal-first OMP process.

The bridge is the *only* object that owns OMP's stdio.  Browser-facing runner
code and the terminal client exchange newline-delimited JSON with this local
socket instead; this keeps a terminal and a browser attached to one OMP session.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from omnigent.inner.omp_executor import OMP_RPC_LINE_LIMIT_BYTES

OMP_NATIVE_BRIDGE_DIR_ENV_VAR = "HARNESS_OMP_NATIVE_BRIDGE_DIR"
OMP_NATIVE_REQUEST_SESSION_ID_ENV_VAR = "HARNESS_OMP_NATIVE_REQUEST_SESSION_ID"

_BRIDGE_ROOT = Path.home() / ".omnigent" / "omp-native"
_SOCKET_FILE = "bridge.sock"
_TOKEN_FILE = "token"
_STATE_FILE = "state.json"


class OmpNativeResumeError(RuntimeError):
    """Raised when a native OMP relaunch cannot safely resume its sessionFile."""


def bridge_dir_for_session_id(session_id: str) -> Path:
    """Return the deterministic owner-only bridge directory for a conversation."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    return _BRIDGE_ROOT / digest


def build_omp_native_spawn_env(session_id: str) -> dict[str, str]:
    """Return the session-scoped bridge coordinates supplied to runner children."""
    return {
        OMP_NATIVE_BRIDGE_DIR_ENV_VAR: str(bridge_dir_for_session_id(session_id)),
        OMP_NATIVE_REQUEST_SESSION_ID_ENV_VAR: session_id,
    }


def _validate_session_file(session_file: Path) -> Path:
    path = session_file.expanduser().resolve()
    if not path.is_file() or not os.access(path, os.R_OK):
        raise OmpNativeResumeError(
            f"Cannot resume OMP: stored sessionFile {path} is missing or unreadable. "
            "Restore that file or start a new OMP-native session; "
            "Omnigent will not rebuild context."
        )
    return path


def omp_rpc_command(
    *,
    executable: str = "omp",
    profile: str | None = None,
    extra_args: Sequence[str] | None = None,
    session_file: Path | str | None = None,
) -> list[str]:
    """Build the only supported native OMP command, optionally with exact resume state."""
    command = [executable]
    if profile:
        command.extend(["--profile", profile])
    if extra_args:
        command.extend(extra_args)
    command.extend(["--mode", "rpc"])
    if session_file is not None:
        command.extend(["--resume", str(_validate_session_file(Path(session_file)))])
    return command


def _write_owner_only(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


class OmpRpcBridge:
    """Own one OMP RPC subprocess and multiplex authenticated Unix-socket clients."""

    def __init__(
        self,
        *,
        session_id: str,
        bridge_dir: Path | None = None,
        executable: str = "omp",
        profile: str | None = None,
        extra_args: Sequence[str] | None = None,
        session_file: Path | str | None = None,
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        process_factory: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.session_id = session_id
        self.bridge_dir = bridge_dir or bridge_dir_for_session_id(session_id)
        self._session_file = (
            _validate_session_file(Path(session_file)) if session_file is not None else None
        )
        self._command = omp_rpc_command(
            executable=executable,
            profile=profile,
            extra_args=extra_args,
            session_file=self._session_file,
        )
        self._cwd = str(cwd) if cwd is not None else None
        self._env = dict(env) if env is not None else None
        self._process_factory = process_factory or asyncio.create_subprocess_exec
        self._process: Any | None = None
        self._server: asyncio.AbstractServer | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._client_lock = asyncio.Lock()
        self._next_request = 0
        self._closed = False
        self._ready = asyncio.Event()
        self._session_file_ready = asyncio.Event()

    @property
    def command(self) -> list[str]:
        """Return the exact argv of the sole OMP subprocess."""
        return list(self._command)

    @property
    def socket_path(self) -> Path:
        return self.bridge_dir / _SOCKET_FILE

    @property
    def token_path(self) -> Path:
        return self.bridge_dir / _TOKEN_FILE

    @property
    def state_path(self) -> Path:
        return self.bridge_dir / _STATE_FILE

    @property
    def session_file(self) -> Path | None:
        """Return the sessionFile captured from OMP's own ``get_state`` response."""
        return self._session_file

    @classmethod
    def stored_session_file(cls, bridge_dir: Path) -> Path | None:
        """Read a previously captured sessionFile without inventing transcript state."""
        try:
            raw = json.loads((bridge_dir / _STATE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        value = raw.get("sessionFile") if isinstance(raw, dict) else None
        return Path(value) if isinstance(value, str) and value else None

    async def start(self) -> None:
        """Start the socket and the one bridge-owned ``omp --mode rpc`` process."""
        if self._closed:
            raise RuntimeError("OMP native bridge is closed")
        if self._server is not None:
            return
        self.bridge_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.bridge_dir, 0o700)
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()
        _write_owner_only(self.token_path, secrets.token_urlsafe(32))
        self._server = await asyncio.start_unix_server(
            self._accept_client, path=str(self.socket_path), limit=OMP_RPC_LINE_LIMIT_BYTES
        )
        os.chmod(self.socket_path, 0o600)
        self._process = await self._process_factory(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=self._env,
            limit=OMP_RPC_LINE_LIMIT_BYTES,
        )
        self._reader_task = asyncio.create_task(
            self._read_omp(), name=f"omp-native-{self.session_id}"
        )
        await asyncio.wait_for(self._ready.wait(), timeout=30.0)
        # Capture durable OMP state before anyone can later choose a resume path.
        await self.send_command({"type": "get_state"}, timeout=30.0)

    async def wait_for_session_file(self, timeout: float = 30.0) -> Path:
        """Wait for and return OMP's durable sessionFile, or fail explicitly."""
        await asyncio.wait_for(self._session_file_ready.wait(), timeout=timeout)
        if self._session_file is None:
            raise OmpNativeResumeError("OMP did not provide a sessionFile for native resume.")
        return self._session_file

    async def send_command(
        self, payload: Mapping[str, Any], *, timeout: float = 60.0
    ) -> dict[str, Any]:
        """Write one command through the bridge and await its correlated OMP response."""
        await self.start()
        request_id = payload.get("id") if isinstance(payload.get("id"), str) else None
        if request_id is None:
            self._next_request += 1
            request_id = f"omp_native_{self._next_request}"
        frame = dict(payload)
        frame["id"] = request_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write_omp(frame)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)

    async def send_frame(self, payload: Mapping[str, Any]) -> None:
        """Forward a callback/response frame without letting a client own OMP stdio."""
        await self.start()
        await self._write_omp(dict(payload))

    async def available_models(self) -> list[dict[str, Any]]:
        """Return OMP model options in the native-picker shape."""
        response = await self.send_command({"type": "get_available_models"})
        if not response.get("success"):
            raise RuntimeError(response.get("error") or "OMP get_available_models failed")
        data = response.get("data", {})
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise RuntimeError("OMP get_available_models returned a non-list models field")
        options: list[dict[str, Any]] = []
        for entry in models:
            if not isinstance(entry, dict):
                continue
            provider = entry.get("provider")
            model_id = entry.get("id")
            if not isinstance(provider, str) or not isinstance(model_id, str):
                continue
            options.append(
                {
                    "id": f"{provider}/{model_id}",
                    "displayName": entry.get("displayName") or entry.get("name") or model_id,
                    "providerName": entry.get("providerName"),
                }
            )
        return options

    async def set_model(self, model: str) -> None:
        """Change the live OMP model from a ``provider/model_id`` string."""
        provider, _, model_id = model.partition("/")
        if not provider or not model_id:
            raise ValueError(f"OMP model must be provider/model_id, got {model!r}")
        response = await self.send_command(
            {"type": "set_model", "provider": provider, "modelId": model_id}
        )
        if not response.get("success"):
            raise RuntimeError(response.get("error") or "OMP set_model failed")

    async def set_thinking_level(self, level: str) -> None:
        """Change the live OMP thinking level."""
        response = await self.send_command({"type": "set_thinking_level", "level": level})
        if not response.get("success"):
            raise RuntimeError(response.get("error") or "OMP set_thinking_level failed")

    async def close(self, *, remove_files: bool = True) -> None:
        """Close clients, OMP, and owner-only transport material."""
        if self._closed:
            return
        self._closed = True
        for pending in self._pending.values():
            if not pending.done():
                pending.set_exception(RuntimeError("OMP native bridge closed"))
        self._pending.clear()
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        async with self._client_lock:
            for writer in self._clients:
                writer.close()
            self._clients.clear()
        process = self._process
        if process is not None:
            if getattr(process, "returncode", None) is None:
                process.terminate()
                with contextlib.suppress(asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(process.wait(), timeout=10.0)
            if getattr(process, "returncode", None) is None:
                process.kill()
        if remove_files:
            for path in (self.socket_path, self.token_path, self.state_path):
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

    async def _write_omp(self, frame: Mapping[str, Any]) -> None:
        process = self._process
        stdin = getattr(process, "stdin", None)
        if stdin is None or stdin.is_closing():
            raise RuntimeError("OMP native bridge stdin is unavailable")
        encoded = (json.dumps(frame, separators=(",", ":")) + "\n").encode("utf-8")
        async with self._write_lock:
            stdin.write(encoded)
            await stdin.drain()

    async def _read_omp(self) -> None:
        process = self._process
        stdout = getattr(process, "stdout", None)
        if stdout is None:
            return
        try:
            while line := await stdout.readline():
                try:
                    frame = json.loads(line)
                except ValueError:
                    await self._broadcast(
                        {"type": "bridge_error", "error": "OMP emitted invalid JSON"}
                    )
                    continue
                if not isinstance(frame, dict):
                    continue
                if frame.get("type") == "ready":
                    self._ready.set()
                if frame.get("type") == "response":
                    request_id = frame.get("id")
                    if isinstance(request_id, str):
                        pending = self._pending.get(request_id)
                        if pending is not None and not pending.done():
                            pending.set_result(frame)
                    self._capture_session_file(frame)
                await self._broadcast({"type": "event", "frame": frame})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - publish a bridge failure to all attached surfaces
            await self._broadcast(
                {"type": "bridge_error", "error": str(exc) or "OMP bridge failed"}
            )
        finally:
            if not self._closed:
                await self._broadcast({"type": "bridge_closed"})

    def _capture_session_file(self, frame: Mapping[str, Any]) -> None:
        data = frame.get("data")
        value = data.get("sessionFile") if isinstance(data, dict) else None
        if not isinstance(value, str) or not value:
            return
        self._session_file = Path(value)
        _write_owner_only(
            self.state_path,
            json.dumps(
                {"sessionFile": value, "sessionId": self.session_id}, separators=(",", ":")
            ),
        )
        self._session_file_ready.set()

    async def _accept_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            first = await asyncio.wait_for(_read_json_line(reader), timeout=10.0)
            token = first.get("token") if first.get("type") == "authenticate" else None
            expected = self.token_path.read_text(encoding="utf-8")
            if not isinstance(token, str) or not secrets.compare_digest(token, expected):
                await _write_json_line(writer, {"type": "authentication_failed"})
                return
            async with self._client_lock:
                self._clients.add(writer)
            await _write_json_line(writer, {"type": "authenticated", "sessionId": self.session_id})
            while True:
                message = await _read_json_line(reader)
                kind = message.get("type")
                if kind == "command":
                    payload = message.get("payload")
                    if not isinstance(payload, dict):
                        await _write_json_line(
                            writer, {"type": "error", "error": "command payload must be an object"}
                        )
                        continue
                    if message.get("wait"):
                        response = await self.send_command(payload)
                        await _write_json_line(
                            writer, {"type": "command_response", "frame": response}
                        )
                    else:
                        await self.send_frame(payload)
                elif kind == "close":
                    return
                else:
                    await _write_json_line(
                        writer, {"type": "error", "error": "unknown bridge message"}
                    )
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            return
        finally:
            async with self._client_lock:
                self._clients.discard(writer)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _broadcast(self, message: Mapping[str, Any]) -> None:
        async with self._client_lock:
            writers = tuple(self._clients)
        failed: list[asyncio.StreamWriter] = []
        for writer in writers:
            try:
                await _write_json_line(writer, message)
            except (ConnectionError, OSError):
                failed.append(writer)
        if failed:
            async with self._client_lock:
                for writer in failed:
                    self._clients.discard(writer)
                    writer.close()


async def _read_json_line(reader: asyncio.StreamReader) -> dict[str, Any]:
    raw = await reader.readline()
    if not raw:
        raise asyncio.IncompleteReadError(partial=b"", expected=1)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("bridge messages must be objects")
    return value


async def _write_json_line(writer: asyncio.StreamWriter, value: Mapping[str, Any]) -> None:
    writer.write((json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8"))
    await writer.drain()


async def _serve(args: argparse.Namespace) -> None:
    bridge = OmpRpcBridge(
        session_id=args.session_id,
        bridge_dir=Path(args.bridge_dir),
        executable=args.executable,
        profile=args.profile,
        extra_args=args.extra_arg,
        session_file=Path(args.resume) if args.resume else None,
        cwd=Path(args.cwd) if args.cwd else None,
    )
    await bridge.start()
    try:
        await asyncio.Event().wait()
    finally:
        await bridge.close()


def main(argv: list[str] | None = None) -> None:
    """Run a durable bridge server for a runner-owned native terminal."""
    parser = argparse.ArgumentParser(prog="python -m omnigent.omp_native_bridge")
    parser.add_argument("serve")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--executable", default="omp")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--resume")
    parser.add_argument("--cwd")
    args = parser.parse_args(argv)
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
