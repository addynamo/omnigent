"""Terminal-first OMP native wrapper and launch helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import httpx
import yaml

from omnigent._native_resume_hint import echo_native_resume_hint
from omnigent._runner_startup import RunnerStartupProgress
from omnigent._wrapper_labels import OMP_NATIVE_WRAPPER_VALUE as _WRAPPER_LABEL_VALUE
from omnigent._wrapper_labels import WRAPPER_LABEL_KEY as _WRAPPER_LABEL_KEY
from omnigent.conversation_browser import conversation_url, open_conversation_link_if_enabled
from omnigent.entities.session_resources import terminal_resource_id
from omnigent.host.daemon_launch import (
    error_text,
    launch_or_reuse_daemon_runner,
    open_daemon_client,
    wait_for_host_online,
    wait_for_runner_online,
)
from omnigent.json_types import JsonObject as _JsonObject
from omnigent.native_coding_agents import native_shell_terminal_spec
from omnigent.native_terminal import (
    DAEMON_HOST_ONLINE_TIMEOUT_S as _DAEMON_HOST_ONLINE_TIMEOUT_S,
)
from omnigent.native_terminal import (
    DAEMON_RUNNER_ONLINE_TIMEOUT_S as _DAEMON_RUNNER_ONLINE_TIMEOUT_S,
)
from omnigent.native_terminal import (
    DAEMON_TERMINAL_READY_TIMEOUT_S as _DAEMON_TERMINAL_READY_TIMEOUT_S,
)
from omnigent.native_terminal import bind_session_runner as _bind_session_runner
from omnigent.native_terminal import (
    normalize_extra_args as _normalize_extra_args,
)
from omnigent.native_terminal import url_component
from omnigent.omp_native_bridge import (
    OmpNativeResumeError,
    bridge_dir_for_session_id,
    omp_rpc_command,
)
from omnigent.omp_native_terminal import OmpNativeTerminalClient

_logger = logging.getLogger(__name__)

_DEFAULT_OMP_COMMAND = "omp"
_OMP_PATH_ENV = "OMNIGENT_OMP_PATH"
# Deprecated alias — remove in v0.8.0.
_LEGACY_HARNESS_OMP_PATH_ENV = "HARNESS_OMP_PATH"
_AGENT_NAME = "omp-native-ui"
_TERMINAL_NAME = "omp"
_TERMINAL_SESSION_KEY = "main"
_SESSION_LABELS = {
    "omnigent.ui": "terminal",
    _WRAPPER_LABEL_KEY: _WRAPPER_LABEL_VALUE,
}


@dataclass(frozen=True)
class OmpNativeLaunch:
    """Resolved native OMP process launch."""

    executable: str
    argv: list[str]


@dataclass(frozen=True)
class LaunchedOmpTerminal:
    """Terminal resource returned by the Omnigent runner launch path."""

    terminal_id: str
    tmux_socket: Path | None
    tmux_target: str | None


@dataclass(frozen=True)
class PreparedOmpTerminal:
    """Prepared native OMP terminal attachment details."""

    session_id: str
    terminal_id: str
    tmux_socket: Path | None
    tmux_target: str | None
    reattached: bool


def _configured_omp_command(env: Mapping[str, str]) -> str:
    value = env.get(_OMP_PATH_ENV, "").strip()
    if value:
        return value
    legacy = env.get(_LEGACY_HARNESS_OMP_PATH_ENV, "").strip()
    if legacy:
        from omnigent.harness_startup_config import _warn_legacy_path

        _warn_legacy_path(_LEGACY_HARNESS_OMP_PATH_ENV, _OMP_PATH_ENV)
        return legacy
    return _DEFAULT_OMP_COMMAND


def resolve_omp_executable_path(
    *,
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> str:
    """Resolve the native OMP executable for the CLI wrapper.

    Prefer the canonical ``OMNIGENT_OMP_PATH`` env var, then the deprecated
    ``HARNESS_OMP_PATH``, then ``PATH``.
    """
    import shutil

    env = os.environ if env is None else env
    which = shutil.which if which is None else which
    command = _configured_omp_command(env)
    resolved = which(command)
    if resolved is None:
        raise click.ClickException(
            "Native OMP requires the 'omp' CLI on PATH. Install OMP, add it to PATH, "
            f"or set {_OMP_PATH_ENV}=/path/to/omp."
        )
    return resolved


def _build_terminal_launch_args(
    *,
    extra_args: tuple[str, ...] | None,
    profile: str | None,
    model: str | None,
    thinking: str | None,
) -> tuple[str, ...]:
    """Compose OMP pass-through args from explicit CLI options and extra args."""
    args: list[str] = list(extra_args or ())
    present = set(args)
    if profile is not None and "--profile" not in present:
        args.extend(["--profile", profile])
    if model is not None and "--model" not in present:
        args.extend(["--model", model])
    if thinking is not None and "--thinking" not in present:
        args.extend(["--thinking", thinking])
    return tuple(args)


def build_omp_native_launch(
    *,
    session_file: Path | str | None = None,
    executable: str = "omp",
    profile: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> OmpNativeLaunch:
    """Build the exact native OMP RPC argv, including a verified resume file."""
    argv = omp_rpc_command(
        executable=executable,
        profile=profile,
        extra_args=extra_args,
        session_file=session_file,
    )
    return OmpNativeLaunch(executable=argv[0], argv=argv)


def omp_native_terminal_command(
    *,
    session_id: str,
    bridge_dir: Path,
    workspace: Path,
    executable: str,
    profile: str | None = None,
    extra_args: Sequence[str] | None = None,
    session_file: Path | str | None = None,
    prompt: str | None = None,
) -> tuple[str, list[str]]:
    """Build the one tmux command: bridge daemon plus its socket-only terminal client."""
    bridge_args = [
        "-m",
        "omnigent.omp_native_bridge",
        "serve",
        "--session-id",
        session_id,
        "--bridge-dir",
        str(bridge_dir),
        "--cwd",
        str(workspace),
        "--executable",
        executable,
    ]
    if profile is not None:
        bridge_args.extend(["--profile", profile])
    for arg in extra_args or ():
        # ``=`` form so argparse accepts dash-prefixed values like ``--model=...``.
        bridge_args.append(f"--extra-arg={arg}")
    if session_file is not None:
        bridge_args.extend(["--resume", str(session_file)])
    # The runner's interpreter, never a PATH-resolved ``python``: the pane's
    # login shell may not resolve ``python`` at all. ``-I`` keeps the pane cwd
    # off sys.path so a workspace checkout can't shadow the installed package.
    python = sys.executable
    bridge_command = " ".join(shlex.quote(part) for part in [python, "-I", *bridge_args])
    terminal_args = [
        python,
        "-I",
        "-m",
        "omnigent.omp_native_terminal",
        "--bridge-dir",
        str(bridge_dir),
    ]
    if prompt is not None:
        terminal_args.extend(["--prompt", prompt])
    terminal_command = " ".join(shlex.quote(part) for part in terminal_args)
    return "/bin/sh", ["-lc", f"{bridge_command} & exec {terminal_command}"]


def _materialize_omp_agent_spec(tmpdir: Path) -> Path:
    """Write the terminal-first native OMP agent spec used by the wrapper."""
    path = tmpdir / "omp-native-ui.yaml"
    payload = {
        "name": _AGENT_NAME,
        "prompt": (
            "OMP is running in the session terminal. Browser and terminal input are "
            "multiplexed through one persistent OMP RPC bridge."
        ),
        "executor": {"harness": "omp-native"},
        "spawn": True,
        "os_env": {"type": "caller_process", "cwd": ".", "sandbox": {"type": "none"}},
        "terminals": native_shell_terminal_spec(),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def run_omp_native(
    *,
    server: str | None,
    session_id: str | None,
    extra_args: tuple[str, ...] | None = None,
    omp_args: tuple[str, ...] | None = None,
    resume_picker: bool = False,
    auto_open_conversation: bool = False,
    model: str | None = None,
    thinking: str | None = None,
    profile: str | None = None,
    prompt: str | None = None,
) -> None:
    """Launch OMP in an Omnigent terminal.

    :param server: Resolved Omnigent server URL.
    :param session_id: Optional existing Omnigent conversation id.
    :param omp_args: Deprecated alias for pass-through OMP CLI args.
    :param resume_picker: ``True`` runs the OMP-native picker.
    :param auto_open_conversation: When ``True``, open the browser conversation URL.
    :param model: Optional OMP model override.
    :param thinking: Optional OMP thinking level override.
    :param profile: Optional OMP profile.
    :param prompt: Optional initial prompt forwarded through the bridge terminal.
    """
    terminal_launch_args = _build_terminal_launch_args(
        extra_args=_normalize_extra_args(
            extra_args=extra_args, legacy_args=omp_args, legacy_param="omp_args"
        ),
        profile=profile,
        model=model,
        thinking=thinking,
    )
    _preflight_local_tools()
    if server is None:
        raise click.ClickException(
            "OMP requires a resolved Omnigent server URL. The CLI should call "
            "_ensure_backend before run_omp_native."
        )
    with TemporaryDirectory(prefix="omnigent-omp-native-") as tmpdir:
        spec_path = _materialize_omp_agent_spec(Path(tmpdir))
        _run_with_remote_server(
            server.rstrip("/"),
            spec_path,
            session_id=session_id,
            resume_picker=resume_picker,
            terminal_launch_args=terminal_launch_args,
            profile=profile,
            prompt=prompt,
            auto_open_conversation=auto_open_conversation,
        )


def _run_with_remote_server(
    base_url: str,
    spec_path: Path,
    *,
    session_id: str | None,
    resume_picker: bool,
    terminal_launch_args: tuple[str, ...],
    profile: str | None,
    prompt: str | None,
    auto_open_conversation: bool = False,
) -> None:
    from omnigent.chat import _bundle_agent, _remote_headers
    from omnigent.cli import _ensure_host_daemon
    from omnigent.host.identity import load_or_create_host_identity

    headers = _remote_headers(server_url=base_url, host_id=None)
    try:
        resolved_session_id = _resolve_session_id_for_resume(
            base_url=base_url,
            headers=headers,
            session_id=session_id,
            resume_picker=resume_picker,
        )
        if resolved_session_id is None and resume_picker and session_id is None:
            return

        async def _drive() -> None:
            from omnigent._runner_startup import runner_startup_progress

            with runner_startup_progress(initial_message="Preparing OMP...") as progress:
                _update_startup_progress(progress, "Connecting to local daemon...")
                _ensure_host_daemon(base_url)
                host_id = load_or_create_host_identity().host_id
                bundle = None if resolved_session_id is not None else _bundle_agent(spec_path)
                prepared = await _prepare_omp_terminal_via_daemon(
                    base_url=base_url,
                    headers=headers,
                    session_id=resolved_session_id,
                    session_bundle=bundle,
                    terminal_launch_args=terminal_launch_args,
                    profile=profile,
                    _prompt=prompt,
                    host_id=host_id,
                    workspace=str(Path.cwd().resolve()),
                    startup_progress=progress,
                )
            click.echo(f"Web UI: {conversation_url(base_url, prepared.session_id)}", err=True)
            open_conversation_link_if_enabled(
                base_url=base_url,
                conversation_id=prepared.session_id,
                enabled=auto_open_conversation,
                warn=lambda message: click.echo(message, err=True),
            )
            await _attach_terminal_resource(prepared)
            if resolved_session_id is None:
                echo_native_resume_hint(
                    native_command="omp",
                    session_id=prepared.session_id,
                    server=base_url,
                )

        asyncio.run(_drive())
    except httpx.ConnectError as exc:
        raise click.ClickException(
            f"Could not reach the omnigent server at {base_url}. "
            "Confirm the server is running and reachable from here "
            f"(e.g. `curl {base_url}/health`), and that --server is correct."
        ) from exc


async def _prepare_omp_terminal_via_daemon(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str | None,
    session_bundle: bytes | None,
    terminal_launch_args: tuple[str, ...],
    profile: str | None,
    _prompt: str | None,
    host_id: str,
    workspace: str,
    startup_progress: RunnerStartupProgress | None = None,
) -> PreparedOmpTerminal:

    persist_args = list(terminal_launch_args)
    timeout = httpx.Timeout(30.0, read=120.0)
    async with open_daemon_client(base_url, headers, host_id, timeout=timeout) as client:
        reattached = session_id is not None
        fresh_session = session_id is None
        if session_id is None:
            if session_bundle is None:
                raise click.ClickException("Creating an OMP session requires a session bundle.")
            _update_startup_progress(startup_progress, "Creating OMP session...")
            session_id = await _create_omp_session(
                client,
                session_bundle,
                terminal_launch_args=persist_args or None,
                profile=profile,
            )
        else:
            _update_startup_progress(startup_progress, "Loading OMP session...")
            payload = await _fetch_omp_session(client, session_id)
            labels = payload.get("labels") if isinstance(payload, dict) else None
            if (
                not isinstance(labels, dict)
                or labels.get(_WRAPPER_LABEL_KEY) != _WRAPPER_LABEL_VALUE
            ):
                raise click.ClickException(
                    f"Conversation {session_id!r} is not an omp-native session."
                )
            existing_terminal = await _find_running_omp_terminal(client, session_id)
            if existing_terminal is not None:
                if persist_args:
                    click.echo(
                        "Ignoring OMP launch args for an already-running terminal; "
                        "restart the session terminal to apply them.",
                        err=True,
                    )
                _update_startup_progress(startup_progress, "OMP terminal ready.")
                return PreparedOmpTerminal(
                    session_id=session_id,
                    terminal_id=existing_terminal.terminal_id,
                    tmux_socket=existing_terminal.tmux_socket,
                    tmux_target=existing_terminal.tmux_target,
                    reattached=True,
                )
            if persist_args:
                _update_startup_progress(startup_progress, "Updating OMP session...")
                resp = await client.patch(
                    f"/v1/sessions/{url_component(session_id)}",
                    json={"terminal_launch_args": persist_args},
                )
                if resp.status_code >= 400:
                    raise click.ClickException(
                        f"OMP session launch config update failed "
                        f"({resp.status_code}): {error_text(resp)}"
                    )

        await wait_for_host_online(client, host_id, timeout_s=_DAEMON_HOST_ONLINE_TIMEOUT_S)
        _update_startup_progress(startup_progress, "Starting runner...")
        runner_id = await launch_or_reuse_daemon_runner(
            client,
            host_id=host_id,
            session_id=session_id,
            workspace=workspace,
            fresh=fresh_session,
        )
        _update_startup_progress(startup_progress, "Waiting for runner...")
        await wait_for_runner_online(client, runner_id, timeout_s=_DAEMON_RUNNER_ONLINE_TIMEOUT_S)
        await _bind_session_runner(client, session_id, runner_id)
        _update_startup_progress(startup_progress, "Starting OMP terminal...")
        await _ensure_omp_terminal_on_runner(client, session_id)
        terminal = await _wait_for_omp_terminal_ready(
            client,
            session_id,
            timeout_s=_DAEMON_TERMINAL_READY_TIMEOUT_S,
        )
        _update_startup_progress(startup_progress, "OMP terminal ready.")
        if _prompt:
            await _inject_omp_initial_prompt(session_id, _prompt)
    return PreparedOmpTerminal(
        session_id=session_id,
        terminal_id=terminal.terminal_id,
        tmux_socket=terminal.tmux_socket,
        tmux_target=terminal.tmux_target,
        reattached=reattached,
    )


async def _create_omp_session(
    client: httpx.AsyncClient,
    bundle: bytes,
    *,
    terminal_launch_args: list[str] | None = None,
    profile: str | None = None,
) -> str:
    labels = dict(_SESSION_LABELS)
    if profile is not None:
        labels["omnigent.omp.profile"] = profile
    metadata: _JsonObject = {"labels": labels}
    if terminal_launch_args:
        metadata["terminal_launch_args"] = terminal_launch_args
    resp = await client.post(
        "/v1/sessions",
        data={"metadata": json.dumps(metadata)},
        files={"bundle": ("omp-native-ui.tar.gz", bundle, "application/gzip")},
        timeout=120.0,
    )
    if resp.status_code >= 400:
        raise click.ClickException(
            f"OMP session creation failed ({resp.status_code}): {error_text(resp)}"
        )
    body = resp.json()
    new_session_id = body.get("session_id")
    if not isinstance(new_session_id, str) or not new_session_id:
        raise click.ClickException("OMP session creation response did not include session_id.")
    return new_session_id


async def _fetch_omp_session(client: httpx.AsyncClient, session_id: str) -> _JsonObject:
    resp = await client.get(f"/v1/sessions/{url_component(session_id)}")
    if resp.status_code == 404:
        raise click.ClickException(f"Conversation {session_id!r} not found on the server.")
    if resp.status_code >= 400:
        text = error_text(resp)
        raise click.ClickException(
            f"Failed to fetch conversation {session_id!r} ({resp.status_code}): {text}"
        )
    payload = resp.json()
    if not isinstance(payload, dict):
        raise click.ClickException("Conversation fetch returned non-object JSON.")
    return payload


async def _ensure_omp_terminal_on_runner(client: httpx.AsyncClient, session_id: str) -> None:
    resp = await client.post(
        f"/v1/sessions/{url_component(session_id)}/resources/terminals",
        json={
            "terminal": _TERMINAL_NAME,
            "session_key": _TERMINAL_SESSION_KEY,
            "ensure_native_terminal": True,
        },
        timeout=60.0,
    )
    if resp.status_code >= 400:
        raise click.ClickException(
            f"OMP terminal ensure failed ({resp.status_code}): {error_text(resp)}"
        )


async def _wait_for_omp_terminal_ready(
    client: httpx.AsyncClient,
    session_id: str,
    *,
    timeout_s: float,
) -> LaunchedOmpTerminal:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        terminal = await _find_running_omp_terminal(client, session_id)
        if terminal is not None:
            return terminal
        await asyncio.sleep(0.2)
    raise click.ClickException(
        f"The runner did not create the OMP terminal for {session_id!r} within {timeout_s:.0f}s."
    )


async def _find_running_omp_terminal(
    client: httpx.AsyncClient,
    session_id: str,
) -> LaunchedOmpTerminal | None:
    terminal_id = omp_terminal_resource_id()
    resp = await client.get(
        f"/v1/sessions/{url_component(session_id)}"
        f"/resources/terminals/{url_component(terminal_id)}"
    )
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        text = error_text(resp)
        if resp.status_code in {409, 503} and (
            "not bound to a runner" in text or "offline" in text
        ):
            return None
        raise click.ClickException(f"Failed to fetch OMP terminal ({resp.status_code}): {text}")
    payload = resp.json()
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if isinstance(metadata, dict) and metadata.get("running") is False:
        return None
    return _launched_omp_terminal_from_payload(payload)


def _launched_omp_terminal_from_payload(payload: object) -> LaunchedOmpTerminal:
    if not isinstance(payload, dict):
        raise click.ClickException("OMP terminal launch returned non-object JSON.")
    terminal_id = payload.get("id")
    if not isinstance(terminal_id, str) or not terminal_id:
        raise click.ClickException("OMP terminal launch response did not include terminal id.")
    metadata = payload.get("metadata")
    tmux_socket: Path | None = None
    tmux_target: str | None = None
    if isinstance(metadata, dict):
        raw_socket = metadata.get("tmux_socket")
        raw_target = metadata.get("tmux_target")
        if isinstance(raw_socket, str) and raw_socket:
            tmux_socket = Path(raw_socket)
        if isinstance(raw_target, str) and raw_target:
            tmux_target = raw_target
    return LaunchedOmpTerminal(
        terminal_id=terminal_id,
        tmux_socket=tmux_socket,
        tmux_target=tmux_target,
    )


async def _attach_terminal_resource(prepared: PreparedOmpTerminal) -> None:
    direct_tmux_error = _direct_tmux_unavailable_reason(prepared)
    if direct_tmux_error is not None:
        raise click.ClickException(
            f"Runner-owned OMP terminal requires direct tmux attach, but {direct_tmux_error}"
        )
    if prepared.tmux_socket is None or prepared.tmux_target is None:
        raise click.ClickException("OMP tmux attach metadata was incomplete.")
    await _attach_direct_tmux(prepared.tmux_socket, prepared.tmux_target)


async def _attach_direct_tmux(socket_path: Path, tmux_target: str) -> None:
    env = dict(os.environ)
    env.pop("TMUX", None)
    process = await asyncio.create_subprocess_exec(
        "tmux",
        "-S",
        str(socket_path),
        "-f",
        os.devnull,
        "attach",
        "-t",
        tmux_target,
        env=env,
    )
    await process.wait()


def _direct_tmux_unavailable_reason(prepared: PreparedOmpTerminal) -> str | None:
    if prepared.tmux_socket is None:
        return "the terminal resource did not include a tmux socket path."
    if prepared.tmux_target is None:
        return "the terminal resource did not include a tmux target."
    if not prepared.tmux_socket.exists():
        return f"tmux socket {prepared.tmux_socket} is not reachable from this CLI process."
    import shutil

    if shutil.which("tmux") is None:
        return "tmux is not available on PATH."
    return None


def _resolve_session_id_for_resume(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str | None,
    resume_picker: bool,
) -> str | None:
    if session_id is not None:
        return session_id
    if not resume_picker:
        return None
    from omnigent_client import OmnigentClient

    from omnigent.repl._resume_picker import pick_conversation_by_wrapper_label_from_sdk

    async def _drive() -> str | None:
        async with OmnigentClient(
            base_url=base_url,
            headers=headers if headers else None,
        ) as client:
            return await pick_conversation_by_wrapper_label_from_sdk(
                client,
                wrapper_value=_WRAPPER_LABEL_VALUE,
                agent_name=_AGENT_NAME,
            )

    return asyncio.run(_drive())


def _update_startup_progress(
    startup_progress: RunnerStartupProgress | None,
    message: str,
) -> None:
    if startup_progress is not None:
        startup_progress.update(message)


async def _inject_omp_initial_prompt(session_id: str, prompt: str) -> None:
    """Forward the CLI ``--prompt`` value into the running OMP bridge."""
    bridge_dir = bridge_dir_for_session_id(session_id)
    client = OmpNativeTerminalClient(bridge_dir)
    try:
        await asyncio.wait_for(client.connect(), timeout=10.0)
        await client.send_command({"type": "prompt", "message": prompt}, wait=False)
    except (RuntimeError, asyncio.TimeoutError, OSError) as exc:
        _logger.warning("Could not inject initial OMP prompt for %s: %s", session_id, exc)
    finally:
        await client.close()


def _preflight_local_tools() -> None:
    import shutil

    if shutil.which("tmux") is None:
        raise click.ClickException(
            "tmux was not found on local PATH. The native OMP wrapper "
            "attaches to the runner-owned OMP tmux terminal."
        )


def omp_terminal_resource_id() -> str:
    """Return the deterministic terminal resource id for OMP."""
    return terminal_resource_id(_TERMINAL_NAME, _TERMINAL_SESSION_KEY)


def omp_native_bridge_dir_for_session(session_id: str) -> Path:
    """Return the native OMP bridge directory for a conversation."""
    return bridge_dir_for_session_id(session_id)


__all__ = [
    "_AGENT_NAME",
    "_TERMINAL_NAME",
    "_TERMINAL_SESSION_KEY",
    "_WRAPPER_LABEL_VALUE",
    "LaunchedOmpTerminal",
    "OmpNativeLaunch",
    "OmpNativeResumeError",
    "PreparedOmpTerminal",
    "_materialize_omp_agent_spec",
    "build_omp_native_launch",
    "omp_native_bridge_dir_for_session",
    "omp_native_terminal_command",
    "omp_terminal_resource_id",
    "resolve_omp_executable_path",
    "run_omp_native",
]
