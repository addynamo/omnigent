"""Fake ``omp --mode rpc`` process for tests.

Driven by a JSON scenario file. Writes a ``ready`` frame, then reads JSONL
commands from stdin and emits scripted responses/events to stdout.

Usage::

    python fake_omp.py <scenario.json>

Scenario shape::

    {
      "ready": {"type": "ready", "supportedProtocolVersions": [1, 2], ...},
      "log": "/optional/log/path",
      "interactions": [
        {
          "match": {"type": "prompt"},
          "response": {"type": "response", "command": "prompt", "success": true},
          "events": [
            {"type": "agent_start"},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "hi"}},
            {"type": "agent_end", "isTerminal": true}
          ]
        }
      ]
    }

An event item may also be a wait directive that pauses until a matching
inbound frame is received::

    {"wait_for": {"type": "host_tool_result"}, "then": [{"type": "agent_end", "isTerminal": true}]}

A ``{"raw": "..."}`` item writes the literal string to stdout (useful for
malformed-line tests).
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any


def _load_scenario(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _log(path: str | None, message: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
            f.flush()


def _matches(frame: dict[str, Any], pattern: dict[str, Any]) -> bool:
    for key, value in pattern.items():
        if frame.get(key) != value:
            return False
    return True


def _write_line(line: str, max_frame_bytes: int, chunk_counter: list[int]) -> None:
    encoded = line.encode("utf-8")
    if len(encoded) <= max_frame_bytes:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        return

    chunk_id = f"chunk-{chunk_counter[0]}"
    chunk_counter[0] += 1
    total = len(encoded)
    chunk_size = max_frame_bytes
    count = (total + chunk_size - 1) // chunk_size
    for index in range(count):
        segment = encoded[index * chunk_size : (index + 1) * chunk_size]
        chunk = {
            "type": "rpc_chunk",
            "chunkId": chunk_id,
            "index": index,
            "count": count,
            "byteLength": total,
            "data": base64.b64encode(segment).decode("ascii"),
        }
        sys.stdout.write(json.dumps(chunk, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def _write_frame(frame: dict[str, Any], max_frame_bytes: int, chunk_counter: list[int]) -> None:
    _write_line(json.dumps(frame, separators=(",", ":"), default=str), max_frame_bytes, chunk_counter)


def _default_response(cmd: dict[str, Any]) -> dict[str, Any] | None:
    ctype = cmd.get("type")
    if ctype == "prompt":
        return {
            "type": "response",
            "command": "prompt",
            "success": True,
            "data": {"agentInvoked": True},
        }
    if ctype == "abort":
        return {"type": "response", "command": "abort", "success": True}
    if ctype in ("set_model", "set_thinking_level", "set_host_tools", "negotiate_protocol"):
        return {"type": "response", "command": ctype, "success": True}
    if ctype == "get_available_models":
        return {
            "type": "response",
            "command": "get_available_models",
            "success": True,
            "data": {"models": []},
        }
    if ctype == "get_state":
        return {
            "type": "response",
            "command": "get_state",
            "success": True,
            "data": {"sessionFile": "/tmp/fake.omp"},
        }
    return None


def _process_event_item(
    item: dict[str, Any],
    max_frame_bytes: int,
    chunk_counter: list[int],
    log_path: str | None,
) -> bool:
    """Emit one event item. Returns True if the item was a plain frame."""
    if "raw" in item:
        sys.stdout.write(item["raw"])
        if not item["raw"].endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        return True

    if "wait_for" in item:
        pattern = item["wait_for"]
        then_events = item.get("then", [])
        while True:
            line = sys.stdin.readline()
            if not line:
                return True
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            _log(log_path, f"wait_received: {json.dumps(frame, separators=(',', ':'), default=str)}")
            if _matches(frame, pattern):
                for ev in then_events:
                    _write_frame(ev, max_frame_bytes, chunk_counter)
                return True

    _write_frame(item, max_frame_bytes, chunk_counter)
    return True


def main() -> None:
    if len(sys.argv) < 2:
        print("fake_omp: missing scenario path", file=sys.stderr)
        sys.exit(1)

    scenario = _load_scenario(sys.argv[1])
    log_path = scenario.get("log")
    if log_path:
        Path(log_path).write_text("", encoding="utf-8")

    ready = scenario.get(
        "ready",
        {
            "type": "ready",
            "protocolVersion": 1,
            "supportedProtocolVersions": [1, 2],
            "maxFrameBytes": 1048576,
            "maxReassembledFrameBytes": 67108864,
        },
    )
    max_frame_bytes = int(ready.get("maxFrameBytes", 1048576))
    chunk_counter = [0]

    sys.stdout.write(json.dumps(ready, separators=(",", ":")) + "\n")
    sys.stdout.flush()

    interactions = list(scenario.get("interactions", []))

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_line(
                json.dumps(
                    {"type": "response", "command": "parse", "success": False, "error": str(exc)},
                    separators=(",", ":"),
                ),
                max_frame_bytes,
                chunk_counter,
            )
            continue

        if not isinstance(cmd, dict):
            continue

        _log(log_path, f"received: {json.dumps(cmd, separators=(',', ':'), default=str)}")

        matched = False
        for interaction in interactions:
            pattern = interaction.get("match", {})
            if not _matches(cmd, pattern):
                continue
            matched = True
            response = interaction.get("response")
            if response:
                response = dict(response)
                if "id" not in response and cmd.get("id"):
                    response["id"] = cmd["id"]
                _write_frame(response, max_frame_bytes, chunk_counter)
            for item in interaction.get("events", []):
                _process_event_item(item, max_frame_bytes, chunk_counter, log_path)
            break

        if matched:
            continue

        default = _default_response(cmd)
        if default:
            default = dict(default)
            if "id" not in default and cmd.get("id"):
                default["id"] = cmd["id"]
            _write_frame(default, max_frame_bytes, chunk_counter)


if __name__ == "__main__":
    main()
