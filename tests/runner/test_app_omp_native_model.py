"""Tests for omp-native model options and model/effort events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from omnigent.runner import create_runner_app
from omnigent.spec.types import AgentSpec, ExecutorSpec
from tests.runner.conftest import (
    _FakeProcessManager,
    _runner_client,
    _ScriptedHarnessClient,
)
from tests.runner.helpers import NullServerClient


class _FakeOmpClient:
    """Records OMP bridge commands and returns scripted responses."""

    def __init__(
        self,
        bridge_dir: Path,
        *,
        models: list[dict[str, Any]] | None = None,
        command_results: list[dict[str, Any]] | None = None,
    ) -> None:
        self.bridge_dir = bridge_dir
        self.connected = False
        self.closed = False
        self.commands: list[dict[str, Any]] = []
        self.models = models or []
        self.command_results = command_results or []

    async def connect(self) -> None:
        self.connected = True

    async def send_command(
        self,
        command: dict[str, Any],
        *,
        request_id: str | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        del request_id, wait
        self.commands.append(command)
        if command.get("type") == "get_available_models":
            return {"success": True, "data": {"models": self.models}}
        if self.command_results:
            return self.command_results.pop(0)
        return {"success": True}

    async def close(self) -> None:
        self.closed = True


def _omp_spec() -> AgentSpec:
    return AgentSpec(
        spec_version=1,
        name="t",
        executor=ExecutorSpec(type="omnigent", config={"harness": "omp-native"}),
    )


def _patch_omp_auto_create(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the runner from trying to launch a real OMP terminal in tests."""
    from omnigent import runner
    from omnigent.entities.session_resources import SessionResourceView

    async def _noop_auto_create(*_args: Any, **_kwargs: Any) -> SessionResourceView:
        return SessionResourceView(
            id="terminal_omp_main",
            type="terminal",
            session_id="",
            name="omp",
        )

    monkeypatch.setattr(
        runner.native.orchestration, "_auto_create_omp_terminal", _noop_auto_create
    )


@pytest.mark.asyncio
async def test_omp_native_model_options_query_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner exposes OMP's available models from the bridge RPC."""
    from omnigent import omp_native_terminal

    _patch_omp_auto_create(monkeypatch)
    conv_id = "conv_omp_model_options"
    expected = [{"id": "ollama/kimi-k2.7-code:cloud", "displayName": "Kimi K2.7 Code Cloud"}]
    created_clients: list[_FakeOmpClient] = []

    def _fake_client_class(bridge_dir: Path) -> _FakeOmpClient:
        client = _FakeOmpClient(
            bridge_dir,
            models=[
                {
                    "provider": "ollama",
                    "id": "kimi-k2.7-code:cloud",
                    "displayName": "Kimi K2.7 Code Cloud",
                }
            ],
        )
        created_clients.append(client)
        return client

    monkeypatch.setattr(omp_native_terminal, "OmpNativeTerminalClient", _fake_client_class)

    async def _resolver(agent_id: str, session_id: str | None = None) -> AgentSpec:
        del agent_id, session_id
        return _omp_spec()

    app = create_runner_app(
        process_manager=_FakeProcessManager(_ScriptedHarnessClient([])),  # type: ignore[arg-type]
        spec_resolver=_resolver,
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )

    async with _runner_client(app) as client:
        create_resp = await client.post(
            "/v1/sessions",
            json={"session_id": conv_id, "agent_id": "ag_1"},
        )
        assert create_resp.status_code == 201, create_resp.text
        response = await client.get(f"/v1/sessions/{conv_id}/omp-model-options")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"models": expected}
    assert len(created_clients) == 1
    assert created_clients[0].connected
    assert created_clients[0].closed
    assert created_clients[0].commands == [{"type": "get_available_models"}]


@pytest.mark.asyncio
async def test_omp_native_model_options_failure_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery failures return 503 so the AP server leaves its cache cold."""
    from omnigent import omp_native_terminal

    _patch_omp_auto_create(monkeypatch)
    conv_id = "conv_omp_model_options_fail"

    class _FailingOmpClient(_FakeOmpClient):
        async def connect(self) -> None:
            raise RuntimeError("bridge unavailable")

    def _fail_client(bridge_dir: Path) -> _FakeOmpClient:
        return _FailingOmpClient(bridge_dir)

    monkeypatch.setattr(omp_native_terminal, "OmpNativeTerminalClient", _fail_client)

    async def _resolver(agent_id: str, session_id: str | None = None) -> AgentSpec:
        del agent_id, session_id
        return _omp_spec()

    app = create_runner_app(
        process_manager=_FakeProcessManager(_ScriptedHarnessClient([])),  # type: ignore[arg-type]
        spec_resolver=_resolver,
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )

    async with _runner_client(app) as client:
        create_resp = await client.post(
            "/v1/sessions",
            json={"session_id": conv_id, "agent_id": "ag_1"},
        )
        assert create_resp.status_code == 201, create_resp.text
        response = await client.get(f"/v1/sessions/{conv_id}/omp-model-options")

    assert response.status_code == 503, response.text
    assert response.json()["error"] == "omp_native_model_options_failed"


class _PatchRecordingServerClient(NullServerClient):
    """Records PATCH bodies so we can assert model_override / reasoning_effort persistence."""

    def __init__(self) -> None:
        super().__init__()
        self.patches: list[dict[str, Any]] = []

    async def patch(self, url: str, **kwargs: Any) -> NullServerClient._Response:
        body = kwargs.get("json")
        if isinstance(body, dict):
            self.patches.append(body)
        return self._Response()


@pytest.mark.asyncio
async def test_omp_native_model_change_forwards_to_bridge_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model_change event reaches OMP, persists to config, and patches the session."""
    from omnigent import omp_config, omp_native_terminal

    _patch_omp_auto_create(monkeypatch)
    conv_id = "conv_omp_model_change"
    created_clients: list[_FakeOmpClient] = []

    def _fake_client_class(bridge_dir: Path) -> _FakeOmpClient:
        client = _FakeOmpClient(
            bridge_dir,
            command_results=[{"success": True}],
        )
        created_clients.append(client)
        return client

    monkeypatch.setattr(omp_native_terminal, "OmpNativeTerminalClient", _fake_client_class)

    persisted: list[tuple[str | None, str, str]] = []

    def _fake_persist_model(profile: str | None, provider: str, model_id: str) -> None:
        persisted.append((profile, provider, model_id))

    monkeypatch.setattr(omp_config, "persist_omp_model", _fake_persist_model)

    async def _resolver(agent_id: str, session_id: str | None = None) -> AgentSpec:
        del agent_id, session_id
        return _omp_spec()

    server_client = _PatchRecordingServerClient()
    app = create_runner_app(
        process_manager=_FakeProcessManager(_ScriptedHarnessClient([])),  # type: ignore[arg-type]
        spec_resolver=_resolver,
        server_client=server_client,  # type: ignore[arg-type]
    )

    async with _runner_client(app) as client:
        create_resp = await client.post(
            "/v1/sessions",
            json={"session_id": conv_id, "agent_id": "ag_1"},
        )
        assert create_resp.status_code == 201, create_resp.text
        resp = await client.post(
            f"/v1/sessions/{conv_id}/events",
            json={"type": "model_change", "model": "ollama/kimi-k2.7-code:cloud"},
        )

    assert resp.status_code == 204, resp.text
    assert len(created_clients) == 1
    assert created_clients[0].commands == [
        {"type": "set_model", "provider": "ollama", "modelId": "kimi-k2.7-code:cloud"}
    ]
    assert persisted == [(None, "ollama", "kimi-k2.7-code:cloud")]
    assert server_client.patches == [{"model_override": "ollama/kimi-k2.7-code:cloud"}]


@pytest.mark.asyncio
async def test_omp_native_effort_change_forwards_to_bridge_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An effort_change event reaches OMP, persists to config, and patches the session."""
    from omnigent import omp_config, omp_native_terminal

    _patch_omp_auto_create(monkeypatch)
    conv_id = "conv_omp_effort_change"
    created_clients: list[_FakeOmpClient] = []

    def _fake_client_class(bridge_dir: Path) -> _FakeOmpClient:
        client = _FakeOmpClient(
            bridge_dir,
            command_results=[{"success": True}],
        )
        created_clients.append(client)
        return client

    monkeypatch.setattr(omp_native_terminal, "OmpNativeTerminalClient", _fake_client_class)

    persisted: list[tuple[str | None, str]] = []

    def _fake_persist_level(profile: str | None, level: str) -> None:
        persisted.append((profile, level))

    monkeypatch.setattr(omp_config, "persist_omp_thinking_level", _fake_persist_level)

    async def _resolver(agent_id: str, session_id: str | None = None) -> AgentSpec:
        del agent_id, session_id
        return _omp_spec()

    server_client = _PatchRecordingServerClient()
    app = create_runner_app(
        process_manager=_FakeProcessManager(_ScriptedHarnessClient([])),  # type: ignore[arg-type]
        spec_resolver=_resolver,
        server_client=server_client,  # type: ignore[arg-type]
    )

    async with _runner_client(app) as client:
        create_resp = await client.post(
            "/v1/sessions",
            json={"session_id": conv_id, "agent_id": "ag_1"},
        )
        assert create_resp.status_code == 201, create_resp.text
        resp = await client.post(
            f"/v1/sessions/{conv_id}/events",
            json={"type": "effort_change", "effort": "high"},
        )

    assert resp.status_code == 204, resp.text
    assert len(created_clients) == 1
    assert created_clients[0].commands == [{"type": "set_thinking_level", "level": "high"}]
    assert persisted == [(None, "high")]
    assert server_client.patches == [{"reasoning_effort": "high"}]


@pytest.mark.asyncio
async def test_omp_native_model_change_rejects_invalid_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model ids without a provider/model_id separator are rejected before touching OMP."""
    _patch_omp_auto_create(monkeypatch)

    async def _resolver(agent_id: str, session_id: str | None = None) -> AgentSpec:
        del agent_id, session_id
        return _omp_spec()

    app = create_runner_app(
        process_manager=_FakeProcessManager(_ScriptedHarnessClient([])),  # type: ignore[arg-type]
        spec_resolver=_resolver,
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )

    async with _runner_client(app) as client:
        create_resp = await client.post(
            "/v1/sessions",
            json={"session_id": "conv_omp_bad_model", "agent_id": "ag_1"},
        )
        assert create_resp.status_code == 201, create_resp.text
        resp = await client.post(
            "/v1/sessions/conv_omp_bad_model/events",
            json={"type": "model_change", "model": "not-a-model"},
        )

    assert resp.status_code == 400
    assert resp.json()["error"] == "omp_native_model_invalid"
