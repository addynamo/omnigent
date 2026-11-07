"""Executor that bridges Omnigent web-UI turns into a resident OMP RPC bridge.

The native OMP process is already running in the session terminal, owned by
:class:`omnigent.omp_native_bridge.OmpRpcBridge`.  This executor connects to the
bridge's authenticated local socket and forwards user messages as OMP
``prompt`` commands.  It deliberately does not spawn a second OMP process.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from omnigent.inner.executor import (
    EnqueuedContent,
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    Message,
    TextChunk,
    ToolSpec,
    TurnComplete,
)
from omnigent.inner.omp_executor import (
    _is_local_prompt_result,
    _is_terminal_agent_end,
    _last_user_text,
    _split_model,
    _translate_omp_event,
)
from omnigent.omp_native_bridge import (
    OMP_NATIVE_BRIDGE_DIR_ENV_VAR,
    OMP_NATIVE_REQUEST_SESSION_ID_ENV_VAR,
)
from omnigent.omp_native_terminal import OmpNativeTerminalClient

logger = logging.getLogger(__name__)


class OmpNativeExecutor(Executor):
    """Harness-side executor for ``omnigent omp`` web-UI turns.

    :param bridge_dir: Directory containing ``bridge.sock`` and ``token``.
        Defaults to the value of ``HARNESS_OMP_NATIVE_BRIDGE_DIR``.
    """

    def __init__(self, bridge_dir: Path | None = None) -> None:
        self._bridge_dir = bridge_dir or _bridge_dir_from_env()
        self._request_session_id = _request_session_id_from_env()
        self._system_prompt_sent = False

    def supports_streaming(self) -> bool:
        """The bridge streams events to every attached surface."""
        return True

    def supports_tool_calling(self) -> bool:
        return True

    def handles_tools_internally(self) -> bool:
        """OMP executes tools inside its own process; we observe lifecycle events."""
        return True

    def supports_live_message_queue(self) -> bool:
        """Live steering messages can be forwarded into a running OMP session."""
        return True

    async def enqueue_session_message(self, session_key: str, content: EnqueuedContent) -> bool:
        """Queue a live steering message into the resident OMP bridge."""
        del session_key
        text = _content_to_text(content)
        if not text:
            return False
        await self._send_prompt(text)
        return True

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        """Forward the latest user message to OMP and stream translated events."""
        del tools
        user_message = _last_user_text(messages) or ""
        if not user_message:
            yield ExecutorError(message="OMP-native turn had no user text to send")
            return

        config = config or ExecutorConfig()
        client = OmpNativeTerminalClient(self._bridge_dir)
        try:
            await asyncio.wait_for(client.connect(), timeout=10.0)

            if config.model:
                provider, model_id = _split_model(config.model)
                if not provider or not model_id:
                    yield ExecutorError(
                        message=f"OMP model must use provider/model form: {config.model}"
                    )
                    return
                # Best-effort model forwarding via the bridge.  Failures are
                # surfaced as executor errors so the UI can report them.
                try:
                    response = await client.send_command(
                        {"type": "set_model", "provider": provider, "modelId": model_id},
                        wait=True,
                    )
                    if isinstance(response, dict) and not response.get("success"):
                        yield ExecutorError(
                            message=f"OMP set_model failed: {response.get('error', 'rejected')}"
                        )
                        return
                except RuntimeError as exc:
                    yield ExecutorError(message=f"OMP set_model failed: {exc}")
                    return

            if system_prompt and not self._system_prompt_sent:
                try:
                    await client.send_command(
                        {"type": "set_system_prompt", "message": system_prompt}, wait=True
                    )
                except RuntimeError as exc:
                    # OMP may not support this command; do not fail the turn.
                    logger.warning("OMP set_system_prompt failed: %s", exc)
                else:
                    self._system_prompt_sent = True

            prompt_response = await client.send_command(
                {"type": "prompt", "message": user_message}, wait=True
            )
            if not isinstance(prompt_response, dict):
                yield ExecutorError(message="OMP bridge did not acknowledge prompt")
                return
            if not prompt_response.get("success"):
                yield ExecutorError(
                    message=f"OMP prompt failed: {prompt_response.get('error', 'rejected')}"
                )
                return

            response_data = prompt_response.get("data")
            if isinstance(response_data, dict) and response_data.get("agentInvoked") is False:
                yield TurnComplete(response=None)
                return

            prompt_id_value = prompt_response.get("id")
            prompt_id = prompt_id_value if isinstance(prompt_id_value, str) else None
            response_text = ""
            async for wrapper in client.events():
                wrapper_type = wrapper.get("type")
                if wrapper_type == "bridge_error":
                    yield ExecutorError(
                        message=f"OMP bridge error: {wrapper.get('error', 'unknown')}"
                    )
                    continue
                if wrapper_type == "bridge_closed":
                    yield TurnComplete(response=response_text or None)
                    return
                if wrapper_type != "event":
                    continue
                raw = wrapper.get("frame")
                if not isinstance(raw, dict):
                    continue
                for event in _translate_omp_event(raw):
                    if isinstance(event, TextChunk):
                        response_text += event.text
                    yield event
                if _is_local_prompt_result(raw, prompt_id) or _is_terminal_agent_end(raw):
                    yield TurnComplete(response=response_text or None)
                    return
        except RuntimeError as exc:
            yield ExecutorError(message=f"OMP-native bridge error: {exc}")
            return
        except asyncio.TimeoutError:
            yield ExecutorError(message="Timed out connecting to OMP native bridge")
            return
        finally:
            await client.close()

    async def _send_prompt(self, text: str) -> None:
        client = OmpNativeTerminalClient(self._bridge_dir)
        try:
            await asyncio.wait_for(client.connect(), timeout=10.0)
            await client.send_command({"type": "prompt", "message": text}, wait=False)
        finally:
            await client.close()

    async def close(self) -> None:
        """No persistent subprocess to terminate; the bridge owns OMP."""


def _bridge_dir_from_env() -> Path:
    raw = os.environ.get(OMP_NATIVE_BRIDGE_DIR_ENV_VAR, "").strip()
    if not raw:
        raise RuntimeError(f"{OMP_NATIVE_BRIDGE_DIR_ENV_VAR} is required for omp-native harness")
    return Path(raw)


def _request_session_id_from_env() -> str | None:
    raw = os.environ.get(OMP_NATIVE_REQUEST_SESSION_ID_ENV_VAR, "").strip()
    return raw or None


def _content_to_text(content: EnqueuedContent) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "input_text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n\n".join(parts)
    if content is None:
        return ""
    return str(content)
