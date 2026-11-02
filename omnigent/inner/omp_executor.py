"""OmpExecutor: run agents through Oh My Pi's native RPC mode.

Spawns ``omp --mode rpc`` as a subprocess and communicates via the JSONL
protocol documented in ``omp://rpc.md``. OMP manages its own agent loop,
tool execution, context window, compaction, skills, profile state, and
persisted transcript internally. This executor translates the OMP event
stream into Omnigent :class:`ExecutorEvent` instances and exposes RPC
commands the native bridge needs (model/effort discovery and mutation,
resume via ``sessionFile``).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from omnigent._platform import resolve_cli_binary
from omnigent.json_types import JsonObject as _JsonObject
from omnigent.json_types import JsonValue

from ._subprocess_lifecycle import close_subprocess_transport
from .executor import (
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    Message,
    ReasoningChunk,
    TextChunk,
    ToolCallComplete,
    ToolCallRequest,
    ToolCallStatus,
    ToolSpec,
    TurnComplete,
)

logger = logging.getLogger(__name__)

# OMP process must stay alive across turns; default wait for clean shutdown.
_OMP_SHUTDOWN_TIMEOUT_S = 10.0

# A malicious ``count`` must not allocate an unbounded sparse segment list.
# Normal chunks fill frames near ``maxFrameBytes``, so this ceiling is far above
# any legitimate reassembly sequence while keeping allocation bounded.
_MAX_RPC_CHUNK_COUNT = 8_192

# OMP model catalog JSON is emitted by ``omp models --json``; we do not try
# to parse it here, but ``OmpProfileConfig`` (omnigent/omp_config.py) does.

# Callback wired by ``ExecutorAdapter._wire_sdk_executor``.
ToolExecutor: TypeAlias = Callable[  # type: ignore[explicit-any]
    [str, dict[str, Any]],
    Awaitable[dict[str, Any]] | dict[str, Any],
]

# Callback for extension UI round-trips (confirm/input/select/...).
ElicitationHandler: TypeAlias = Callable[[str, _JsonObject], Awaitable[Any]]  # type: ignore[explicit-any]


def _safe_json_dumps(obj: object) -> str:
    """Serialize a JSONL frame, raising only on unrecoverable errors."""
    return json.dumps(obj, separators=(",", ":"), default=str)


def _describe_exception(exc: BaseException) -> str:
    """Return a non-empty human-readable description of *exc*."""
    return str(exc) or repr(exc)


class OmpProtocolError(RuntimeError):
    """Raised when the OMP RPC wire contract is violated."""


@dataclass
class _OmpModelOption:
    """Internal normalized model option from ``get_available_models``."""

    provider: str
    model_id: str
    display_name: str
    provider_name: str | None = None
    aliases: list[str] = field(default_factory=list)


class _OmpChunkBuffer:
    """Reassemble one protocol-v2 logical frame from ``rpc_chunk`` pieces."""

    def __init__(self, chunk_id: str, count: int, byte_length: int) -> None:
        self.chunk_id = chunk_id
        self.count = count
        self.byte_length = byte_length
        self._segments: list[bytes | None] = [None] * count
        self.received = 0
        self._decoded_bytes: int = 0

    def add(self, index: int, data: str, byte_length: int) -> str | None:
        """Add a chunk and return the reassembled JSON string when complete.

        ``byte_length`` is the advertised total reassembled byte length, not
        the size of this individual segment.
        """
        if index < 0 or index >= self.count:
            raise OmpProtocolError(
                f"rpc_chunk {self.chunk_id} index {index} out of range (count {self.count})"
            )
        if self._segments[index] is not None:
            raise OmpProtocolError(f"rpc_chunk {self.chunk_id} index {index} duplicated")
        try:
            segment = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise OmpProtocolError(
                f"rpc_chunk {self.chunk_id} contains invalid base64 data"
            ) from exc
        self._segments[index] = segment
        self.received += 1
        self._decoded_bytes += len(segment)
        if self._decoded_bytes > byte_length:
            raise OmpProtocolError(
                f"rpc_chunk {self.chunk_id} decoded byte length exceeds advertised {byte_length}"
            )
        if self.received < self.count:
            return None
        decoded = b"".join(self._segments)  # type: ignore[arg-type]
        if len(decoded) != byte_length:
            raise OmpProtocolError(
                f"rpc_chunk {self.chunk_id} reassembled byte length {len(decoded)} "
                f"does not match advertised {byte_length}"
            )
        try:
            return decoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise OmpProtocolError(f"rpc_chunk {self.chunk_id} is not valid UTF-8") from exc


class _OmpRpcSession:
    """Own one ``omp --mode rpc`` subprocess and its JSONL stdio.

    Responsibilities:
    - Spawn and negotiate protocol v2.
    - Serialize outbound JSONL frames atomically.
    - Read stdout, reassemble v2 chunks, and route responses to pending
      request futures while enqueueing non-response frames.
    - Execute host-tool / extension UI callbacks on the event loop.
    """

    def __init__(
        self,
        command: Sequence[str],
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._command = list(command)
        self._cwd = cwd
        self._env = dict(env) if env else None
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._closed = False
        self._protocol_version = 1
        self._max_frame_bytes = 1_048_576
        self._max_reassembled_frame_bytes = 67_108_864
        self._next_id = 0
        self._pending: dict[str, asyncio.Future[_JsonObject]] = {}
        self._events: asyncio.Queue[_JsonObject] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._chunk_buffers: dict[str, _OmpChunkBuffer] = {}
        self._active_chunk_id: str | None = None
        self._failure: OmpProtocolError | None = None
        self._callback_tasks: set[asyncio.Task[None]] = set()
        # Callbacks wired after construction by the bridge/adapter.
        self._tool_executor: ToolExecutor | None = None
        self._elicitation_handler: ElicitationHandler | None = None

    @property
    def protocol_version(self) -> int:
        return self._protocol_version

    @property
    def command(self) -> list[str]:
        return list(self._command)

    @property
    def failed(self) -> bool:
        """Whether the reader has stopped due to a terminal protocol failure."""
        return self._failure is not None

    def set_tool_executor(self, executor: ToolExecutor | None) -> None:
        self._tool_executor = executor

    def set_elicitation_handler(self, handler: ElicitationHandler | None) -> None:
        self._elicitation_handler = handler

    async def start(self) -> None:
        """Spawn OMP and negotiate protocol v2."""
        if self._failure is not None:
            raise self._failure
        if self._process is not None:
            return
        cwd = str(self._cwd) if self._cwd else None
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=self._env,
            )
        except Exception as exc:
            raise OmpProtocolError(
                f"Failed to start OMP ({' '.join(self._command)}): {_describe_exception(exc)}"
            ) from exc

        self._reader_task = asyncio.create_task(self._read_loop())
        await self._negotiate_protocol()

    async def _negotiate_protocol(self) -> None:
        """Read the ready frame and opt in to v2 when advertised."""
        ready = await self._read_one_frame(timeout=30.0)
        if not isinstance(ready, dict) or ready.get("type") != "ready":
            raise OmpProtocolError(f"Expected OMP ready frame, got: {ready!r}")
        self._max_frame_bytes = int(ready.get("maxFrameBytes", self._max_frame_bytes))
        self._max_reassembled_frame_bytes = int(
            ready.get("maxReassembledFrameBytes", self._max_reassembled_frame_bytes)
        )
        supported = ready.get("supportedProtocolVersions", [])
        if 2 in supported:
            response = await self.send_command(
                {"type": "negotiate_protocol", "protocolVersion": 2}
            )
            if response.get("success") is True:
                self._protocol_version = 2
        logger.debug("OMP RPC negotiated protocol v%s", self._protocol_version)

    def _next_request_id(self) -> str:
        self._next_id += 1
        return f"omp_req_{self._next_id}"

    async def send_command(
        self,
        payload: _JsonObject,
        *,
        request_id: str | None = None,
        timeout: float | None = 60.0,
    ) -> _JsonObject:
        """Send a command and await its correlated response."""
        await self.start()
        req_id = request_id or self._next_request_id()
        payload = {**payload, "id": req_id}
        future: asyncio.Future[_JsonObject] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        try:
            await self._write_frame(payload)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(req_id, None)

    async def send_command_no_wait(
        self,
        payload: _JsonObject,
        *,
        request_id: str | None = None,
    ) -> str:
        """Send a command without awaiting a response; return its id.

        Callback result frames must retain the inbound callback id: OMP uses
        that id to resolve the pending host-tool, host-URI, or UI request.
        """
        await self.start()
        payload_id = payload.get("id")
        req_id = request_id or (payload_id if isinstance(payload_id, str) else None)
        req_id = req_id or self._next_request_id()
        payload = {**payload, "id": req_id}
        await self._write_frame(payload)
        return req_id

    async def _write_frame(self, payload: _JsonObject) -> None:
        line = _safe_json_dumps(payload) + "\n"
        encoded = line.encode("utf-8")
        if len(encoded) > self._max_frame_bytes:
            raise OmpProtocolError(
                f"Outbound frame exceeds OMP maxFrameBytes ({self._max_frame_bytes})"
            )
        async with self._write_lock:
            proc = self._process
            if proc is None or proc.stdin is None or proc.stdin.is_closing():
                raise OmpProtocolError("OMP subprocess stdin is closed")
            proc.stdin.write(encoded)
            await proc.stdin.drain()

    async def _read_loop(self) -> None:
        """Drain OMP stdout and dispatch frames until EOF or close."""
        try:
            proc = self._process
            if proc is None or proc.stdout is None:
                return
            while not self._closed:
                try:
                    raw = await proc.stdout.readline()
                except asyncio.CancelledError:
                    break
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Ignoring malformed JSONL frame from OMP: %s", exc)
                    continue
                if not isinstance(frame, dict):
                    logger.warning("Ignoring non-object JSONL frame from OMP: %r", frame)
                    continue
                await self._handle_frame(frame)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._failure = (
                exc
                if isinstance(exc, OmpProtocolError)
                else OmpProtocolError(_describe_exception(exc))
            )
            logger.exception("OMP RPC reader loop failed: %s", self._failure)
        finally:
            # A stopped reader is terminal for this process. Mark it failed so
            # the executor tears it down and creates a fresh session instead
            # of attaching future requests to a dead reader task.
            if not self._closed and self._failure is None:
                self._failure = OmpProtocolError("OMP RPC reader ended unexpectedly")
            failure = self._failure
            if failure is not None:
                for future in list(self._pending.values()):
                    if not future.done():
                        future.set_exception(failure)
            self._pending.clear()

    async def _handle_frame(self, frame: _JsonObject) -> None:
        ftype = frame.get("type")
        if self._active_chunk_id is not None and ftype != "rpc_chunk":
            raise OmpProtocolError(
                f"rpc_chunk sequence {self._active_chunk_id} interrupted by {ftype!r} frame"
            )
        if ftype == "rpc_chunk":
            frame = await self._reassemble_chunk(frame)
            if frame is None:
                return
            ftype = frame.get("type")

        # Host-facing callbacks (tool / extension UI / URI) are handled inline
        # because they may need to send responses back to OMP.
        if ftype == "host_tool_call":
            self._track_callback(self._handle_host_tool_call(frame))
            return
        if ftype == "host_tool_cancel":
            # Best-effort: no execution state tracked at this layer.
            return
        if ftype == "host_uri_request":
            self._track_callback(self._handle_host_uri_request(frame))
            return
        if ftype == "host_uri_cancel":
            return
        if ftype == "extension_ui_request":
            self._track_callback(self._handle_extension_ui_request(frame))
            return

        # Responses are routed by id; everything else is an event for consumers.
        if ftype == "response" and "id" in frame:
            req_id = str(frame["id"])
            future = self._pending.pop(req_id, None)
            if future is not None and not future.done():
                future.set_result(frame)
            else:
                # Response with no waiter becomes a generic event so callers
                # can observe late scheduling errors for prompt etc.
                await self._events.put(frame)
        else:
            await self._events.put(frame)

    def _track_callback(self, coroutine: Awaitable[None]) -> None:
        """Keep an OMP callback alive until it produces its RPC response."""
        task = asyncio.create_task(coroutine)
        self._callback_tasks.add(task)
        task.add_done_callback(self._callback_tasks.discard)

    async def _reassemble_chunk(self, chunk: _JsonObject) -> _JsonObject | None:
        chunk_id = chunk.get("chunkId")
        index = chunk.get("index")
        count = chunk.get("count")
        byte_length = chunk.get("byteLength")
        data = chunk.get("data")
        if (
            not isinstance(chunk_id, str)
            or type(index) is not int
            or type(count) is not int
            or type(byte_length) is not int
            or not isinstance(data, str)
            or not chunk_id
            or index < 0
            or count <= 0
            or byte_length < 0
        ):
            raise OmpProtocolError(f"Malformed rpc_chunk frame from OMP: {chunk!r}")
        if count > _MAX_RPC_CHUNK_COUNT:
            raise OmpProtocolError(
                f"rpc_chunk sequence {chunk_id} exceeds maximum chunk count {_MAX_RPC_CHUNK_COUNT}"
            )
        if byte_length > self._max_reassembled_frame_bytes:
            raise OmpProtocolError(
                f"rpc_chunk sequence {chunk_id} exceeds maxReassembledFrameBytes"
            )
        if self._active_chunk_id is None:
            self._active_chunk_id = chunk_id
        elif self._active_chunk_id != chunk_id:
            raise OmpProtocolError(
                f"rpc_chunk sequence {chunk_id} interleaved with {self._active_chunk_id}"
            )

        buf = self._chunk_buffers.get(chunk_id)
        if buf is None:
            buf = _OmpChunkBuffer(chunk_id, count, byte_length)
            self._chunk_buffers[chunk_id] = buf
        elif buf.count != count or buf.byte_length != byte_length:
            raise OmpProtocolError(f"rpc_chunk sequence {chunk_id} changed its declared shape")
        try:
            assembled = buf.add(index, data, byte_length)
        except OmpProtocolError:
            self._chunk_buffers.pop(chunk_id, None)
            self._active_chunk_id = None
            raise
        if assembled is None:
            return None

        self._chunk_buffers.pop(chunk_id, None)
        self._active_chunk_id = None
        try:
            decoded = json.loads(assembled)
        except json.JSONDecodeError as exc:
            raise OmpProtocolError(
                f"Reassembled v2 frame {chunk_id} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise OmpProtocolError(f"Reassembled v2 frame {chunk_id} is not a JSON object")
        return decoded

    async def _handle_host_tool_call(self, frame: _JsonObject) -> None:
        call_id = str(frame.get("id", ""))
        tool_name = str(frame.get("toolName", ""))
        _ = frame.get("toolCallId")
        arguments = frame.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        result: dict[str, Any]
        if self._tool_executor is None:
            result = {"isError": True, "error": f"No tool executor available for '{tool_name}'"}
        else:
            try:
                raw = self._tool_executor(tool_name, arguments)
                resolved = await raw if asyncio.iscoroutine(raw) or asyncio.isfuture(raw) else raw
                if not isinstance(resolved, dict):
                    resolved = {"result": resolved}
                result = {"result": resolved}
            except Exception as exc:  # noqa: BLE001
                result = {"isError": True, "error": _describe_exception(exc)}
        await self.send_command_no_wait(
            {
                "type": "host_tool_result",
                "id": call_id,
                **result,
            }
        )

    async def _handle_host_uri_request(self, frame: _JsonObject) -> None:
        req_id = str(frame.get("id", ""))
        # Host URI schemes are not implemented in this layer; reject cleanly.
        await self.send_command_no_wait(
            {
                "type": "host_uri_result",
                "id": req_id,
                "isError": True,
                "error": "Host URI schemes are not supported by this executor",
            }
        )

    async def _handle_extension_ui_request(self, frame: _JsonObject) -> None:
        req_id = str(frame.get("id", ""))
        method = str(frame.get("method", ""))
        if self._elicitation_handler is None:
            await self.send_command_no_wait(
                {
                    "type": "extension_ui_response",
                    "id": req_id,
                    "cancelled": True,
                }
            )
            return
        try:
            answer = await self._elicitation_handler(method, frame)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Extension UI handler failed for %s: %s", method, exc)
            await self.send_command_no_wait(
                {
                    "type": "extension_ui_response",
                    "id": req_id,
                    "cancelled": True,
                }
            )
            return
        # Normalize handler answer to the response shape OMP expects.
        response: _JsonObject = {"type": "extension_ui_response", "id": req_id}
        if isinstance(answer, dict):
            response.update(answer)
        elif isinstance(answer, str):
            response["value"] = answer
        elif answer is True:
            response["confirmed"] = True
        elif answer is False:
            response["confirmed"] = False
        else:
            response["cancelled"] = True
        await self.send_command_no_wait(response)

    async def _read_one_frame(self, timeout: float | None = None) -> _JsonObject:
        """Read the next non-response event frame (used during startup)."""
        return await asyncio.wait_for(self._events.get(), timeout=timeout)

    async def read_event(self, timeout: float | None = None) -> _JsonObject | None:
        """Pop the next event frame, or ``None`` on timeout."""
        try:
            return await asyncio.wait_for(self._events.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        """Terminate the subprocess and stop the reader."""
        self._closed = True
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        proc = self._process
        if proc is not None:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.close()
                with contextlib.suppress(Exception):
                    await proc.stdin.wait_closed()
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_OMP_SHUTDOWN_TIMEOUT_S)
                except asyncio.TimeoutError:
                    proc.kill()
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
            close_subprocess_transport(proc)
        for task in self._callback_tasks:
            task.cancel()
        if self._callback_tasks:
            await asyncio.gather(*self._callback_tasks, return_exceptions=True)
        self._callback_tasks.clear()
        self._chunk_buffers.clear()
        self._active_chunk_id = None
        self._process = None
        self._reader_task = None


class OmpExecutor(Executor):
    """Executor that drives OMP's RPC mode.

    :param command: Exact argv to launch OMP in RPC mode, e.g.
        ``("omp", "--mode", "rpc")``.
    :param cwd: Working directory for the subprocess.
    :param env: Extra environment variables for the subprocess.
    :param extra_args: Additional CLI arguments appended to *command*.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        extra_args: Sequence[str] | None = None,
        os_env: object | None = None,
        model: str | None = None,
        bundle_dir: Path | None = None,
        agent_name: str | None = None,
        skills_filter: str | list[str] = "all",
    ) -> None:
        self._base_command = list(command)
        self._cwd = Path(cwd) if cwd else None
        self._env = dict(env) if env else None
        self._extra_args = list(extra_args or [])
        # Kept as explicit construction state so the harness preserves the
        # established executor/spawn-env contract. OMP currently owns skill
        # discovery and sandbox configuration, so these are not translated
        # into undocumented RPC flags.
        self._os_env = os_env
        self._configured_model = model
        self._bundle_dir = bundle_dir
        self._agent_name = agent_name
        self._skills_filter = skills_filter
        self._session: _OmpRpcSession | None = None
        self._tool_executor: ToolExecutor | None = None
        self._elicitation_handler: ElicitationHandler | None = None
        self._last_session_file: str | None = None

    @property
    def command(self) -> list[str]:
        """Return the exact base RPC argv, without per-launch extra args."""
        return list(self._base_command)

    # ------------------------------------------------------------------
    # Executor contract
    # ------------------------------------------------------------------

    def supports_streaming(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    def handles_tools_internally(self) -> bool:
        return True

    def supports_live_message_queue(self) -> bool:
        return True

    def supports_tool_boundary_interrupt(self) -> bool:
        return True

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,  # noqa: ARG002
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        """Run one OMP agent turn.

        Sends the last user message to OMP, registers Omnigent tools as
        host-owned RPC tools, and yields translated events until the turn
        completes.
        """
        try:
            session = await self._ensure_session()
            # Capture resume state for every normal turn, not only native
            # callers that explicitly query it.
            await self.get_state()
        except OmpProtocolError as exc:
            yield ExecutorError(message=f"Failed to start OMP: {exc}")
            return
        if self._tool_executor is not None:
            session.set_tool_executor(self._tool_executor)
        if self._elicitation_handler is not None:
            session.set_elicitation_handler(self._elicitation_handler)

        user_message = _last_user_text(messages)
        if user_message is None:
            user_message = ""

        # ``set_host_tools`` replaces OMP's complete registration, including
        # when this turn intentionally exposes no host-owned tools.
        host_tools = [_tool_spec_to_host_tool(t) for t in tools]
        try:
            await session.send_command({"type": "set_host_tools", "tools": host_tools})
        except OmpProtocolError as exc:
            yield ExecutorError(message=f"Failed to register tools with OMP: {exc}")
            return

        # A request-specific override takes precedence over the AgentSpec
        # model passed through HARNESS_OMP_MODEL at executor construction.
        config = config or ExecutorConfig()
        selected_model = config.model or self._configured_model
        if selected_model:
            provider, model_id = _split_model(selected_model)
            if not provider or not model_id:
                yield ExecutorError(
                    message=f"OMP model must use provider/model form: {selected_model}"
                )
                return
            try:
                response = await session.send_command(
                    {
                        "type": "set_model",
                        "provider": provider,
                        "modelId": model_id,
                    }
                )
                if not response.get("success"):
                    raise OmpProtocolError(response.get("error", "OMP set_model failed"))
            except OmpProtocolError as exc:
                yield ExecutorError(message=f"Failed to set OMP model: {exc}")
                return

        # Send the user prompt and wait for the ack before streaming events.
        try:
            response = await session.send_command({"type": "prompt", "message": user_message})
        except OmpProtocolError as exc:
            yield ExecutorError(message=f"OMP prompt failed: {exc}")
            return
        if not response.get("success"):
            yield ExecutorError(message=f"OMP prompt failed: {response.get('error', 'rejected')}")
            return

        prompt_id = response.get("id") if isinstance(response.get("id"), str) else None
        response_data = response.get("data")
        if isinstance(response_data, dict) and response_data.get("agentInvoked") is False:
            yield TurnComplete(response=None)
            return

        response_text = ""
        while True:
            event = await session.read_event(timeout=300.0)
            if event is None:
                yield ExecutorError(message="OMP event stream timed out")
                return

            translated = self._translate_event(event)
            for item in translated:
                if isinstance(item, TextChunk):
                    response_text += item.text
                yield item

            if _is_local_prompt_result(event, prompt_id) or _is_terminal_agent_end(event):
                yield TurnComplete(response=response_text or None)
                return

    async def close_session(self, _session_key: str) -> None:
        """Close the persistent OMP subprocess for *session_key*."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def close(self) -> None:
        await self.close_session("")

    async def interrupt_session(self, session_key: str) -> bool:  # noqa: ARG002
        """Send ``abort`` to interrupt the active OMP turn."""
        session = self._session
        if session is None:
            return False
        try:
            await session.send_command({"type": "abort"}, timeout=10.0)
            return True
        except OmpProtocolError:
            return False

    async def enqueue_session_message(
        self,
        session_key: str,  # noqa: ARG002
        content: Any,
    ) -> bool:
        """Queue a follow-up/steer message into a live OMP session."""
        session = self._session
        if session is None:
            return False
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        try:
            await session.send_command({"type": "prompt", "message": text})
            return True
        except OmpProtocolError:
            return False

    # ------------------------------------------------------------------
    # Native bridge helpers
    # ------------------------------------------------------------------

    async def available_models(self) -> list[_OmpModelOption]:
        """Return models from OMP ``get_available_models``."""
        session = await self._ensure_session()
        response = await session.send_command({"type": "get_available_models"})
        if not response.get("success"):
            error = response.get("error", "OMP get_available_models failed")
            raise OmpProtocolError(error)
        data = response.get("data", {})
        models = data.get("models") if isinstance(data, dict) else None
        return _parse_model_catalog(models)

    async def set_model(self, provider: str, model_id: str) -> None:
        """Change the live OMP model."""
        session = await self._ensure_session()
        response = await session.send_command(
            {
                "type": "set_model",
                "provider": provider,
                "modelId": model_id,
            }
        )
        if not response.get("success"):
            raise OmpProtocolError(response.get("error", "OMP set_model failed"))

    async def set_thinking_level(self, level: str) -> None:
        """Change the live OMP thinking level."""
        session = await self._ensure_session()
        response = await session.send_command({"type": "set_thinking_level", "level": level})
        if not response.get("success"):
            raise OmpProtocolError(response.get("error", "OMP set_thinking_level failed"))

    async def get_state(self) -> _JsonObject:
        """Return the OMP session state (includes ``sessionFile``)."""
        session = await self._ensure_session()
        response = await session.send_command({"type": "get_state"})
        if not response.get("success"):
            raise OmpProtocolError(response.get("error", "OMP get_state failed"))
        data = response.get("data", {})
        session_file = data.get("sessionFile")
        if isinstance(session_file, str):
            self._last_session_file = session_file
        return data

    @property
    def last_session_file(self) -> str | None:
        return self._last_session_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> _OmpRpcSession:
        if self._session is not None and self._session.failed:
            failed_session = self._session
            self._session = None
            await failed_session.close()
        if self._session is None:
            command = [*self._base_command, *self._extra_args]
            session = _OmpRpcSession(command, cwd=self._cwd, env=self._env)
            self._session = session
            try:
                await session.start()
            except OmpProtocolError:
                self._session = None
                await session.close()
                raise
            # Wire callbacks each time a session is created.
            session.set_tool_executor(self._tool_executor)
            session.set_elicitation_handler(self._elicitation_handler)
        return self._session

    def _translate_event(self, event: _JsonObject) -> list[ExecutorEvent]:
        ftype = event.get("type")
        result: list[ExecutorEvent] = []
        if ftype == "message_update":
            ame = event.get("assistantMessageEvent", {}) if isinstance(event, dict) else {}
            if not isinstance(ame, dict):
                return result
            delta_type = ame.get("type")
            delta = ame.get("delta")
            if delta_type == "text_delta" and isinstance(delta, str) and delta:
                result.append(TextChunk(text=delta))
            elif delta_type == "thinking_delta" and isinstance(delta, str):
                result.append(ReasoningChunk(delta=delta, event_type="reasoning_text"))
            elif delta_type == "toolcall_delta":
                # Delta-level tool calls are informational only; lifecycle
                # events produce the actionable ToolCallRequest/Complete.
                pass
        elif ftype == "tool_execution_start":
            name = str(event.get("toolName", ""))
            args = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
            if name:
                result.append(ToolCallRequest(name=name, args=args))
        elif ftype == "tool_execution_end":
            status = event.get("status")
            result_data = event.get("result") if isinstance(event.get("result"), dict) else {}
            if status == "error":
                result.append(
                    ToolCallComplete(
                        name=str(event.get("toolName", "")),
                        status=ToolCallStatus.ERROR,
                        result=result_data,
                    )
                )
            elif status == "cancelled":
                result.append(
                    ToolCallComplete(
                        name=str(event.get("toolName", "")),
                        status=ToolCallStatus.CANCELLED,
                        result=result_data,
                    )
                )
            else:
                result.append(
                    ToolCallComplete(
                        name=str(event.get("toolName", "")),
                        status=ToolCallStatus.SUCCESS,
                        result=result_data,
                    )
                )
        elif ftype == "model_changed":
            # Native bridge may observe and publish metadata.
            pass
        elif ftype == "thinking_level_changed":
            pass
        elif ftype == "extension_error":
            error = event.get("error", "OMP extension error")
            result.append(ExecutorError(message=f"OMP extension error: {error}"))
        elif ftype == "response":
            # A response without a matching pending request (e.g. late prompt
            # scheduling error) is surfaced as an error event.
            if not event.get("success"):
                result.append(
                    ExecutorError(message=f"OMP command failed: {event.get('error', '')}")
                )
        elif ftype not in {"agent_start", "agent_end", "turn_start", "turn_end"}:
            # Unknown frame type: log but do not hang. Tests assert this.
            logger.debug("Unrecognized OMP event type %r", ftype)
        return result


def _last_user_text(messages: Sequence[Message]) -> str | None:
    """Extract text from the last user message."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                ]
                return "\n".join(texts) or None
    return None


def _split_model(model: str) -> tuple[str | None, str | None]:
    """Split ``provider/model_id`` into provider and model id."""
    if "/" in model:
        provider, model_id = model.split("/", 1)
        return provider, model_id
    return None, None


def _tool_spec_to_host_tool(tool: ToolSpec) -> _JsonObject:
    """Convert an Omnigent ToolSpec into OMP's ``set_host_tools`` shape."""
    name = str(tool.get("name", ""))
    return {
        "name": name,
        "label": tool.get("label") or name,
        "description": str(tool.get("description", "")),
        "parameters": tool.get("parameters", {}),
    }


def _parse_model_catalog(raw: JsonValue | None) -> list[_OmpModelOption]:
    """Parse OMP's structured model catalog into normalized options."""
    options: list[_OmpModelOption] = []
    if not isinstance(raw, list):
        return options
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider", ""))
        model_id = str(entry.get("id", ""))
        if not provider or not model_id:
            continue
        aliases = entry.get("aliases")
        options.append(
            _OmpModelOption(
                provider=provider,
                model_id=model_id,
                display_name=str(entry.get("displayName") or entry.get("name") or model_id),
                provider_name=(
                    str(entry.get("providerName")) if entry.get("providerName") else None
                ),
                aliases=[str(alias) for alias in aliases] if isinstance(aliases, list) else [],
            )
        )
    return options


def _is_terminal_agent_end(event: _JsonObject) -> bool:
    """Return True when an ``agent_end`` frame marks the turn boundary."""
    return (
        isinstance(event, dict)
        and event.get("type") == "agent_end"
        and event.get("isTerminal") is not False
    )


def _is_local_prompt_result(event: _JsonObject, prompt_id: str | None) -> bool:
    """Return whether *event* completes the correlated prompt locally."""
    return (
        prompt_id is not None
        and event.get("type") == "prompt_result"
        and event.get("id") == prompt_id
        and event.get("agentInvoked") is False
    )


# ---------------------------------------------------------------------------
# Public aliases used by the native bridge and configuration layer.
# ---------------------------------------------------------------------------

NativeModelOption: TypeAlias = _OmpModelOption


def resolve_omp_executable(
    *,
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> str:
    """Resolve the OMP CLI executable.

    :raises OSError: If the ``omp`` CLI cannot be found.
    """
    env = os.environ if env is None else env
    which = shutil.which if which is None else which
    resolved = resolve_cli_binary("omp", env_var="OMNIGENT_OMP_PATH", which=which)
    if resolved is None:
        raise OSError(
            "OMP requires the 'omp' CLI on PATH. Install OMP, add it to PATH, "
            "or set OMNIGENT_OMP_PATH=/path/to/omp."
        )
    return resolved
