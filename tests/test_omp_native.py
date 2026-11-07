"""Focused tests for the terminal-first OMP native wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.omp_native import (
    OmpNativeResumeError,
    _materialize_omp_agent_spec,
    build_omp_native_launch,
)


def test_materialized_agent_is_terminal_first(tmp_path: Path) -> None:
    """The native wrapper has its own identity and never reuses headless ``omp``."""
    spec = _materialize_omp_agent_spec(tmp_path)

    content = spec.read_text(encoding="utf-8")
    assert "omp-native-ui" in content
    assert "omp-native" in content
    assert "terminal" in content


def test_native_launch_uses_exact_resume_argv(tmp_path: Path) -> None:
    """Terminal relaunches ask OMP to resume its stored session file exactly."""
    session_file = tmp_path / "omp-session.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")

    launch = build_omp_native_launch(session_file=session_file)

    assert launch.argv == ["omp", "--mode", "rpc", "--resume", str(session_file)]


def test_native_launch_fails_before_start_for_unreadable_resume(tmp_path: Path) -> None:
    """Resume errors point at the stored sessionFile and avoid transcript fallback."""
    with pytest.raises(OmpNativeResumeError, match="sessionFile"):
        build_omp_native_launch(session_file=tmp_path / "gone.jsonl")


def test_terminal_command_forwards_flag_extra_args_to_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dash-prefixed passthrough args like ``--model=...`` must survive bridge argv parsing."""
    import shlex

    from omnigent import omp_native_bridge
    from omnigent.omp_native import omp_native_terminal_command

    _, args = omp_native_terminal_command(
        session_id="s1",
        bridge_dir=tmp_path,
        workspace=tmp_path,
        executable="omp",
        extra_args=["--model=openrouter/ai21/jamba-large-1.7"],
    )
    bridge_command = args[1].split(" & exec ", 1)[0]
    bridge_argv = shlex.split(bridge_command)
    bridge_argv = bridge_argv[bridge_argv.index("serve") :]  # drop interpreter prefix

    captured: dict[str, object] = {}

    async def _fake_serve(parsed: object) -> None:
        captured["extra_arg"] = parsed.extra_arg  # type: ignore[attr-defined]

    monkeypatch.setattr(omp_native_bridge, "_serve", _fake_serve)
    omp_native_bridge.main(bridge_argv)

    assert captured["extra_arg"] == ["--model=openrouter/ai21/jamba-large-1.7"]


def test_terminal_command_uses_runner_interpreter(tmp_path: Path) -> None:
    """Both pane processes run the runner's interpreter, not a PATH ``python``."""
    import shlex
    import sys

    from omnigent.omp_native import omp_native_terminal_command

    _, args = omp_native_terminal_command(
        session_id="s1",
        bridge_dir=tmp_path,
        workspace=tmp_path,
        executable="omp",
    )
    bridge_command, terminal_command = args[1].split(" & exec ", 1)
    for command in (bridge_command, terminal_command):
        argv = shlex.split(command)
        assert argv[0] == sys.executable
        assert "-I" in argv


@pytest.mark.parametrize(
    ("raw_args", "model_override", "expected"),
    [
        ([], None, (None, [])),
        (["--profile", "work", "--verbose"], None, ("work", ["--verbose"])),
        ([], "ollama/glm-5.1:cloud", (None, ["--model=ollama/glm-5.1:cloud"])),
        # An explicit launch-arg model wins over the persisted override.
        (["--model=openai/gpt-5.6"], "ollama/glm-5.1:cloud", (None, ["--model=openai/gpt-5.6"])),
        (
            ["--model", "openai/gpt-5.6"],
            "ollama/glm-5.1:cloud",
            (None, ["--model", "openai/gpt-5.6"]),
        ),
    ],
)
def test_omp_terminal_launch_args_apply_model_override(
    raw_args: list[str],
    model_override: str | None,
    expected: tuple[str | None, list[str]],
) -> None:
    """The model picked before launch reaches OMP as ``--model`` unless args already pin one."""
    from omnigent.runner.native.orchestration import _omp_terminal_launch_args

    assert _omp_terminal_launch_args(raw_args, model_override) == expected


def test_omp_role_assignment_persists_only_on_fresh_launch() -> None:
    """A role picked in the browser is saved once, before OMP's first start."""
    from omnigent.runner.native.orchestration import (
        OMP_ASSIGN_MODEL_LABEL_KEY,
        OMP_ASSIGN_ROLE_LABEL_KEY,
        _apply_omp_role_assignment,
    )

    labels = {
        OMP_ASSIGN_ROLE_LABEL_KEY: "slow",
        OMP_ASSIGN_MODEL_LABEL_KEY: "ollama/deepseek-v4-flash:cloud:high",
    }
    calls: list[tuple[object, ...]] = []

    def persist(*args: object) -> None:
        calls.append(args)

    assert _apply_omp_role_assignment(labels, "work", fresh=True, persist=persist) is True
    assert calls == [("work", "slow", "ollama/deepseek-v4-flash:cloud:high")]

    # Relaunches and resumes must not overwrite later TUI role changes.
    assert _apply_omp_role_assignment(labels, "work", fresh=False, persist=persist) is False
    assert len(calls) == 1


@pytest.mark.parametrize(
    "labels",
    [
        {},
        {"omnigent.omp.assign_role": "slow"},
        {"omnigent.omp.assign_model": "ollama/x"},
        {"omnigent.omp.assign_role": "", "omnigent.omp.assign_model": "ollama/x"},
    ],
)
def test_omp_role_assignment_skips_incomplete_labels(labels: dict[str, str]) -> None:
    """Without both a role and a model, nothing is written to OMP's config."""
    from omnigent.runner.native.orchestration import _apply_omp_role_assignment

    calls: list[object] = []
    assert _apply_omp_role_assignment(labels, None, fresh=True, persist=calls.append) is False
    assert calls == []


def test_omp_role_assignment_failure_does_not_block_launch() -> None:
    """A rejected role save is reported, and the session still launches."""
    from omnigent.omp_config import OmpValidationError
    from omnigent.runner.native.orchestration import _apply_omp_role_assignment

    def persist(*_args: object) -> None:
        raise OmpValidationError("bad")

    labels = {"omnigent.omp.assign_role": "slow", "omnigent.omp.assign_model": "bad"}
    assert _apply_omp_role_assignment(labels, None, fresh=True, persist=persist) is False
