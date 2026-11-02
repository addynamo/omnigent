"""Tests for :class:`OmpExecutor` using a fake OMP JSONL subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from omnigent.inner.executor import (
    ExecutorConfig,
    ExecutorError,
    ReasoningChunk,
    TextChunk,
    ToolCallComplete,
    ToolCallRequest,
    ToolSpec,
    TurnComplete,
)
from omnigent.inner.omp_executor import OmpExecutor

FAKE_OMP = Path(__file__).with_name("fake_omp.py")


def _write_scenario(tmp_path: Path, scenario: dict[str, Any]) -> Path:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    return path


def _executor(tmp_path: Path, scenario: dict[str, Any]) -> OmpExecutor:
    scenario_path = _write_scenario(tmp_path, scenario)
    return OmpExecutor(
        command=[sys.executable, str(FAKE_OMP), str(scenario_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


async def _collect_events(ait: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in ait]


@pytest.fixture
async def closed_executor(tmp_path: Path) -> OmpExecutor:
    """Provide an executor and close it after the test."""
    executor = _executor(tmp_path, {"ready": _ready()})
    try:
        yield executor
    finally:
        await executor.close()


def _ready(
    *, supported_versions: list[int] | None = None, max_frame_bytes: int = 1048576
) -> dict[str, Any]:
    return {
        "type": "ready",
        "protocolVersion": 1,
        "supportedProtocolVersions": supported_versions or [1, 2],
        "maxFrameBytes": max_frame_bytes,
        "maxReassembledFrameBytes": 67108864,
    }


async def test_protocol_v2_ready_negotiation(tmp_path: Path) -> None:
    executor = _executor(tmp_path, {"ready": _ready(supported_versions=[1, 2])})
    # Trigger session start through a state query.
    state = await executor.get_state()
    assert state.get("sessionFile") == "/tmp/fake.omp"
    assert executor._session is not None
    assert executor._session.protocol_version == 2
    await executor.close()


async def test_protocol_v1_only_stays_on_v1(tmp_path: Path) -> None:
    executor = _executor(tmp_path, {"ready": _ready(supported_versions=[1])})
    await executor.get_state()
    assert executor._session is not None
    assert executor._session.protocol_version == 1
    await executor.close()


async def test_request_id_correlation(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "interactions": [
                {
                    "match": {"type": "get_state"},
                    "response": {"type": "response", "command": "get_state", "success": True},
                }
            ],
        },
    )
    session = await executor._ensure_session()
    response = await session.send_command({"type": "get_state"}, request_id="req-abc-123")
    assert response["id"] == "req-abc-123"
    assert response["success"] is True
    await executor.close()


async def test_prompt_ack_then_terminal_agent_end(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "interactions": [
                {
                    "match": {"type": "prompt"},
                    "response": {
                        "type": "response",
                        "command": "prompt",
                        "success": True,
                        "data": {"agentInvoked": True},
                    },
                    "events": [
                        {"type": "agent_start"},
                        {
                            "type": "message_update",
                            "assistantMessageEvent": {"type": "text_delta", "delta": "hello"},
                        },
                        {"type": "agent_end", "isTerminal": True},
                    ],
                }
            ],
        },
    )
    events = await _collect_events(
        executor.run_turn(
            messages=[{"role": "user", "content": "say hello"}],
            tools=[],
            system_prompt="",
        )
    )
    assert [type(e).__name__ for e in events] == ["TextChunk", "TurnComplete"]
    assert isinstance(events[0], TextChunk)
    assert events[0].text == "hello"
    assert isinstance(events[-1], TurnComplete)
    assert events[-1].response == "hello"
    await executor.close()


async def test_prompt_ack_no_agent_end_does_not_complete(tmp_path: Path) -> None:
    """A prompt ack without a terminal agent_end must not finish the turn."""
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "interactions": [
                {
                    "match": {"type": "prompt"},
                    "response": {
                        "type": "response",
                        "command": "prompt",
                        "success": True,
                        "data": {"agentInvoked": True},
                    },
                    "events": [
                        {
                            "type": "message_update",
                            "assistantMessageEvent": {"type": "text_delta", "delta": "ok"},
                        },
                        # Non-terminal agent_end: the turn should keep waiting.
                        {"type": "agent_end", "isTerminal": False},
                    ],
                }
            ],
        },
    )
    # Read with a short timeout to prove it does not finish.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            _collect_events(
                executor.run_turn(
                    messages=[{"role": "user", "content": "go"}],
                    tools=[],
                    system_prompt="",
                )
            ),
            timeout=0.5,
        )
    await executor.close()


async def test_v2_chunk_reassembly(tmp_path: Path) -> None:
    model_payload = {
        "type": "response",
        "command": "get_available_models",
        "success": True,
        "data": {
            "models": [
                {
                    "provider": "anthropic",
                    "id": "claude-sonnet-4-5",
                    "displayName": "a" * 500,
                }
            ]
        },
    }
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(max_frame_bytes=200),
            "interactions": [
                {
                    "match": {"type": "get_available_models"},
                    # The fake will chunk because the JSON line exceeds maxFrameBytes.
                    "response": model_payload,
                }
            ],
        },
    )
    models = await executor.available_models()
    assert len(models) == 1
    assert models[0].provider == "anthropic"
    assert models[0].model_id == "claude-sonnet-4-5"
    await executor.close()


async def test_message_update_text_and_thinking_delta(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "interactions": [
                {
                    "match": {"type": "prompt"},
                    "response": {
                        "type": "response",
                        "command": "prompt",
                        "success": True,
                        "data": {"agentInvoked": True},
                    },
                    "events": [
                        {
                            "type": "message_update",
                            "assistantMessageEvent": {"type": "thinking_delta", "delta": "think"},
                        },
                        {
                            "type": "message_update",
                            "assistantMessageEvent": {"type": "text_delta", "delta": "hi"},
                        },
                        {"type": "agent_end", "isTerminal": True},
                    ],
                }
            ],
        },
    )
    events = await _collect_events(
        executor.run_turn(messages=[{"role": "user", "content": "x"}], tools=[], system_prompt="")
    )
    assert [type(e).__name__ for e in events] == [
        "ReasoningChunk",
        "TextChunk",
        "TurnComplete",
    ]
    assert isinstance(events[0], ReasoningChunk)
    assert events[0].delta == "think"
    assert isinstance(events[1], TextChunk)
    assert events[1].text == "hi"
    await executor.close()


async def test_tool_lifecycle_events(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "interactions": [
                {
                    "match": {"type": "set_host_tools"},
                    "response": {"type": "response", "command": "set_host_tools", "success": True},
                },
                {
                    "match": {"type": "prompt"},
                    "response": {
                        "type": "response",
                        "command": "prompt",
                        "success": True,
                        "data": {"agentInvoked": True},
                    },
                    "events": [
                        {
                            "type": "tool_execution_start",
                            "toolName": "read",
                            "arguments": {"path": "/tmp/x"},
                        },
                        {
                            "type": "tool_execution_end",
                            "toolName": "read",
                            "status": "success",
                            "result": {"content": "data"},
                        },
                        {"type": "agent_end", "isTerminal": True},
                    ],
                },
            ],
        },
    )
    tools: list[ToolSpec] = [
        {
            "name": "read",
            "description": "read",
            "parameters": {"type": "object"},
        }
    ]
    events = await _collect_events(
        executor.run_turn(
            messages=[{"role": "user", "content": "read it"}],
            tools=tools,
            system_prompt="",
        )
    )
    assert [type(e).__name__ for e in events] == [
        "ToolCallRequest",
        "ToolCallComplete",
        "TurnComplete",
    ]
    assert isinstance(events[0], ToolCallRequest)
    assert events[0].name == "read"
    assert events[0].args == {"path": "/tmp/x"}
    assert isinstance(events[1], ToolCallComplete)
    assert events[1].status.value == "success"
    await executor.close()


async def test_host_tool_callback_failure(tmp_path: Path) -> None:
    log_path = tmp_path / "fake.log"
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "log": str(log_path),
            "interactions": [
                {
                    "match": {"type": "prompt"},
                    "response": {
                        "type": "response",
                        "command": "prompt",
                        "success": True,
                        "data": {"agentInvoked": True},
                    },
                    "events": [
                        {
                            "type": "host_tool_call",
                            "id": "host_1",
                            "toolCallId": "tc_1",
                            "toolName": "boom_tool",
                            "arguments": {"x": 1},
                        },
                        {
                            "wait_for": {"type": "host_tool_result"},
                            "then": [{"type": "agent_end", "isTerminal": True}],
                        },
                    ],
                }
            ],
        },
    )

    def _tool_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        assert name == "boom_tool"
        raise RuntimeError("host tool exploded")

    executor._tool_executor = _tool_executor  # type: ignore[attr-defined]

    events = await _collect_events(
        executor.run_turn(messages=[{"role": "user", "content": "x"}], tools=[], system_prompt="")
    )
    # Host-tool calls are executed by the executor internally; failures do not
    # surface as executor events, but the turn must still finish.
    assert events and isinstance(events[-1], TurnComplete)
    await executor.close()

    # Verify the host tool result frame was sent back with the error.
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    result_lines = [ln for ln in lines if '"type":"host_tool_result"' in ln]
    assert result_lines, "expected host_tool_result frame in fake log"
    frame = json.loads(result_lines[-1].split(": ", 1)[1])
    assert frame.get("isError") is True
    assert "exploded" in frame.get("error", "")


async def test_interrupt_via_abort(tmp_path: Path) -> None:
    log_path = tmp_path / "fake.log"
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "log": str(log_path),
            "interactions": [
                {
                    "match": {"type": "abort"},
                    "response": {"type": "response", "command": "abort", "success": True},
                }
            ],
        },
    )
    # Start session so there is something to interrupt.
    await executor.get_state()
    assert await executor.interrupt_session("session-key") is True
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    abort_lines = [ln for ln in lines if '"type":"abort"' in ln]
    assert abort_lines
    await executor.close()


async def test_subprocess_cleanup_after_close(tmp_path: Path) -> None:
    executor = _executor(tmp_path, {"ready": _ready()})
    await executor.get_state()
    process = executor._session._process if executor._session else None
    assert process is not None
    await executor.close()
    assert executor._session is None


async def test_malformed_and_unknown_frames_do_not_hang(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "interactions": [
                {
                    "match": {"type": "prompt"},
                    "response": {
                        "type": "response",
                        "command": "prompt",
                        "success": True,
                        "data": {"agentInvoked": True},
                    },
                    "events": [
                        {"raw": "this is not json"},
                        {"type": "weird_unknown_event", "foo": "bar"},
                        {
                            "type": "message_update",
                            "assistantMessageEvent": {"type": "text_delta", "delta": "ok"},
                        },
                        {"type": "agent_end", "isTerminal": True},
                    ],
                }
            ],
        },
    )
    events = await _collect_events(
        executor.run_turn(messages=[{"role": "user", "content": "x"}], tools=[], system_prompt="")
    )
    # The malformed line and unknown event must not hang the turn.
    texts = [e.text for e in events if isinstance(e, TextChunk)]
    assert texts == ["ok"]
    assert isinstance(events[-1], TurnComplete)
    await executor.close()


async def test_set_model_and_set_thinking_level(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "interactions": [
                {
                    "match": {"type": "set_model"},
                    "response": {"type": "response", "command": "set_model", "success": True},
                },
                {
                    "match": {"type": "set_thinking_level"},
                    "response": {
                        "type": "response",
                        "command": "set_thinking_level",
                        "success": True,
                    },
                },
            ],
        },
    )
    await executor.set_model("anthropic", "claude-sonnet-4-5")
    await executor.set_thinking_level("high")
    await executor.close()


async def test_command_failure_surfaces_as_executor_error(tmp_path: Path) -> None:
    """A failed prompt response must terminate the turn with an error."""
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "interactions": [
                {
                    "match": {"type": "prompt"},
                    "response": {
                        "type": "response",
                        "command": "prompt",
                        "success": False,
                        "error": "prompt refused",
                    },
                }
            ],
        },
    )
    events = await _collect_events(
        executor.run_turn(messages=[{"role": "user", "content": "x"}], tools=[], system_prompt="")
    )
    assert any(isinstance(e, ExecutorError) for e in events)
    await executor.close()


async def test_run_turn_model_override_from_config(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        {
            "ready": _ready(),
            "interactions": [
                {
                    "match": {
                        "type": "set_model",
                        "provider": "anthropic",
                        "modelId": "claude-sonnet-4-5",
                    },
                    "response": {
                        "type": "response",
                        "command": "set_model",
                        "success": True,
                    },
                },
                {
                    "match": {"type": "prompt"},
                    "response": {
                        "type": "response",
                        "command": "prompt",
                        "success": True,
                        "data": {"agentInvoked": True},
                    },
                    "events": [
                        {
                            "type": "message_update",
                            "assistantMessageEvent": {"type": "text_delta", "delta": "done"},
                        },
                        {"type": "agent_end", "isTerminal": True},
                    ],
                },
            ],
        },
    )
    config = ExecutorConfig(model="anthropic/claude-sonnet-4-5")
    events = await _collect_events(
        executor.run_turn(
            messages=[{"role": "user", "content": "x"}],
            tools=[],
            system_prompt="",
            config=config,
        )
    )
    assert isinstance(events[-1], TurnComplete)
    await executor.close()
