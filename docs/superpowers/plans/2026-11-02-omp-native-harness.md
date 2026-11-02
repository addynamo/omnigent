# OMP Native Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `omnigent omp` as a terminal-first OMP RPC harness whose terminal and browser conversation coordinate one durable OMP session, with persistent profile-backed model/effort settings and headless OMP workers available to Polly and Debby.

**Architecture:** Add a focused OMP RPC executor and harness wrapper for headless work, then layer a native-session bridge over that protocol for the top-level terminal/browser session. The bridge exclusively owns OMP's JSONL stdio; a separate terminal client connects to its per-session socket, while browser controls are routed through the same bridge. Registry/CLI/session wiring follows existing native harness conventions, and profile writes use the OMP CLI rather than editing its files.

**Tech Stack:** Python 3.12, FastAPI, asyncio subprocesses and Unix-domain sockets, Click, existing Omnigent runner/session registry, OMP documented JSONL RPC protocol, pytest, React/TypeScript only where an existing picker mapping requires it.

**Spec:** `docs/superpowers/specs/2026-11-02-omp-native-harness-design.md`

## Global Constraints

- Launch OMP only with documented `omp --mode rpc`; the bridge alone owns that process's stdin/stdout.
- `omp-native` is the top-level terminal/browser harness; `omp` is the headless worker harness for Polly and Debby.
- Persist model and effort only through `omp [--profile <profile>] config set`; never write OMP configuration files directly.
- Persist profile and OMP `sessionFile` in conversation metadata/labels; resume with `omp --mode rpc --resume <sessionFile>`.
- Browser and terminal input must target the same bridge/process; never start a second OMP process for either surface.
- Update Omnigent metadata only after both the live OMP RPC command and the OMP CLI persistence command succeed.
- Effective 2026-11-06, Polly and Debby default to `omp` because Claude is cancelled; retain explicit harness/model overrides and do not implement runtime quota probing.
- Tests use fake OMP subprocess/CLI seams and never access a cloud model or the developer's real OMP configuration.

## Review Focus

- Runner restart with a missing or unreadable persisted `sessionFile` must produce an actionable resume failure and must not silently rebuild context — Task 3.
- A terminal and browser prompt arriving while OMP is streaming must preserve the selected steer/follow-up policy and never corrupt JSONL command framing — Task 3.
- A live `set_model` or `set_thinking_level` success followed by failed `omp config set` must leave Omnigent metadata unchanged and report persistence failure — Task 4.
- A malformed, oversized, or unknown OMP RPC frame must fail only the affected session with diagnostics and must not leave bridge clients waiting forever — Task 1.
- A default Polly or Debby invocation after the 2026-11-06 configuration change must select the headless `omp` harness and never create a terminal-first `omp-native` child — Task 6.

---

### Task 1: Implement the headless OMP RPC executor

**Files:**
- Create: `omnigent/inner/omp_executor.py`
- Create: `omnigent/inner/omp_harness.py`
- Modify: `omnigent/runtime/workflow.py`
- Modify: `omnigent/harness_aliases.py`
- Modify: `omnigent/harness_plugins.py`
- Test: `tests/inner/test_omp_executor.py`
- Test: `tests/inner/test_omp_harness.py`
- Test: `tests/test_harness_plugins.py`

**Interfaces:**
- Consumes: the OMP JSONL commands/events documented in `omp://rpc.md`; existing `Executor`, `ExecutorEvent`, `ExecutorAdapter`, and spawn-environment conventions.
- Produces: `OmpExecutor(Executor)` for `harness: omp`; `build_omp_spawn_env(spec: AgentSpec, *, cwd: Path | None, workdir: Path | None) -> dict[str, str]`; harness registry/module resolution for `omp`.

- [ ] **Step 1: Write failing executor protocol tests**

```python
async def test_omp_executor_prompts_and_translates_text_reasoning_tool_and_completion() -> None:
    executor = OmpExecutor(command=("fake-omp", "--mode", "rpc"), cwd=Path("/work"))
    events = [event async for event in executor.execute([Message.user("inspect src")])]
    assert any(isinstance(event, TextChunk) and event.text == "done" for event in events)
    assert any(isinstance(event, ReasoningChunk) for event in events)
    assert isinstance(events[-1], TurnComplete)

async def test_omp_executor_rejects_unknown_or_malformed_rpc_frame_without_hanging() -> None:
    ...
```

Cover ready-frame negotiation, request-id correlation, `prompt` acknowledgement versus terminal `agent_end`, RPC v2 chunk reassembly, `message_update` translation, tool lifecycle, extension/host-tool callback failure, interrupt, and subprocess cleanup. Use a scripted fake JSONL process; do not invoke `omp`.

- [ ] **Step 2: Run the focused executor tests to verify they fail**

Run: `uv run pytest tests/inner/test_omp_executor.py -q`

Expected: FAIL because `omnigent.inner.omp_executor` does not exist.

- [ ] **Step 3: Implement `OmpExecutor` and the harness wrapper**

Create `OmpExecutor` by extracting only reusable JSONL subprocess mechanics from `PiExecutor` and implementing OMP's documented commands/events directly: negotiate protocol v2, call `get_state`, submit `prompt` / `steer` / `follow_up`, cancel with `abort`, and translate terminal `agent_end` into `TurnComplete`. Preserve session-file state returned by `get_state`; expose typed methods `available_models()`, `set_model(provider: str, model_id: str)`, and `set_thinking_level(level: str)` for native callers.

`omp_harness.create_app()` must create this executor from a dedicated `HARNESS_OMP_*` spawn environment. Add the `omp` harness module and environment builder using the Pi harness's sandbox/cwd/skills conventions without inheriting Pi's gateway-specific configuration.

- [ ] **Step 4: Run the focused executor and harness tests**

Run: `uv run pytest tests/inner/test_omp_executor.py tests/inner/test_omp_harness.py tests/test_harness_plugins.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the headless OMP executor**

```bash
git add omnigent/inner/omp_executor.py omnigent/inner/omp_harness.py omnigent/runtime/workflow.py omnigent/harness_aliases.py omnigent/harness_plugins.py tests/inner/test_omp_executor.py tests/inner/test_omp_harness.py tests/test_harness_plugins.py
git commit -m "feat: add headless OMP RPC harness"
```

### Task 2: Add profile-safe OMP configuration and catalog access

**Files:**
- Create: `omnigent/omp_config.py`
- Test: `tests/test_omp_config.py`

**Interfaces:**
- Consumes: an optional OMP profile, model provider/id, thinking level, and an injectable subprocess runner.
- Produces: `OmpProfileConfig`, `list_omp_models(profile: str | None) -> list[NativeModelOption]`, `persist_omp_model(profile: str | None, provider: str, model_id: str) -> None`, and `persist_omp_thinking_level(profile: str | None, level: str) -> None`.

- [ ] **Step 1: Write failing OMP configuration tests**

```python
def test_persist_omp_model_preserves_other_model_roles() -> None:
    runner = FakeRunner(config={"modelRoles": {"default": "old", "slow": "keep"}})
    persist_omp_model("work", "ollama", "kimi-k2.7-code:cloud", runner=runner)
    assert runner.config_set_value("modelRoles") == {
        "default": "ollama/kimi-k2.7-code:cloud", "slow": "keep"
    }

def test_failed_config_set_raises_without_returning_success() -> None:
    ...
```

Cover global versus named profiles, malformed `config get` and `models --json` output, invalid provider/model identifiers, non-zero commands, and thinking-level validation.

- [ ] **Step 2: Run the configuration tests to verify they fail**

Run: `uv run pytest tests/test_omp_config.py -q`

Expected: FAIL because `omnigent.omp_config` does not exist.

- [ ] **Step 3: Implement the OMP CLI boundary**

Run all OMP commands as argv lists, never a shell. Prepend `omp --profile <profile>` only for a non-empty selected profile. Build the catalog from OMP's structured model data; read the complete `modelRoles` object before replacing it with the same roles except `default`; validate config-set success before returning. Make errors typed/actionable so runner and browser callers can show the failed OMP command class without exposing secrets.

- [ ] **Step 4: Run the configuration tests**

Run: `uv run pytest tests/test_omp_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit profile-safe OMP configuration**

```bash
git add omnigent/omp_config.py tests/test_omp_config.py
git commit -m "feat: manage persistent OMP profile settings"
```

### Task 3: Build the persistent native bridge and terminal client

**Files:**
- Create: `omnigent/omp_native_bridge.py`
- Create: `omnigent/omp_native_terminal.py`
- Create: `omnigent/omp_native.py`
- Modify: `omnigent/runner/native/orchestration.py`
- Modify: `omnigent/runner/native/__init__.py`
- Modify: `omnigent/native_coding_agents.py`
- Modify: `omnigent/_wrapper_labels.py`
- Modify: `omnigent/harness_plugins.py`
- Test: `tests/test_omp_native_bridge.py`
- Test: `tests/test_omp_native.py`
- Test: `tests/runner/test_app_sessions_native_terminals_autocreate.py`
- Test: `tests/runner/test_app_sessions_native_workflow_init.py`

**Interfaces:**
- Consumes: `OmpExecutor` RPC primitives, a session id, optional profile/session file, terminal resources, and browser-originated control requests.
- Produces: `OmpRpcBridge`, `bridge_dir_for_session_id(session_id: str) -> Path`, `run_omp_native(...) -> None`, `build_omp_native_spawn_env(session_id: str) -> dict[str, str]`, `_launch_omp(ctx: NativeLaunchContext) -> SessionResourceView`, and native identity `omp-native-ui` / wrapper label `omp-native-ui`.

- [ ] **Step 1: Write failing bridge and resume tests**

```python
async def test_terminal_and_browser_commands_share_one_omp_rpc_process() -> None:
    bridge = await OmpRpcBridge.start(...)
    await bridge.submit_terminal_prompt("terminal turn")
    await bridge.submit_browser_prompt("browser turn")
    assert fake_omp.received_messages == ["terminal turn", "browser turn"]

async def test_resume_uses_persisted_session_file() -> None:
    launch = build_omp_rpc_launch(session_file=Path("/sessions/a.jsonl"), profile="work")
    assert launch.argv == ["omp", "--profile", "work", "--mode", "rpc", "--resume", "/sessions/a.jsonl"]
```

Cover per-session socket authentication/cleanup, one bridge owner, extension UI round trips, host-tool callbacks, terminal disconnect/reconnect, browser cancellation, ordering during streaming, native terminal creation, session-file capture, missing resume file, and bridge failure publication.

- [ ] **Step 2: Run bridge and native lifecycle tests to verify they fail**

Run: `uv run pytest tests/test_omp_native_bridge.py tests/test_omp_native.py tests/runner/test_app_sessions_native_terminals_autocreate.py tests/runner/test_app_sessions_native_workflow_init.py -q`

Expected: FAIL because the OMP native bridge and launch adapter do not exist.

- [ ] **Step 3: Implement the bridge, terminal client, and native lifecycle adapter**

Make `OmpRpcBridge` the exclusive OMP stdio owner. It must multiplex its local authenticated socket clients, expose commands/events to the runner, relay OMP extension UI and host-tool requests through existing policy/tool infrastructure, and persist the initial `sessionFile` before accepting later resume requests. Implement `omp_native_terminal` as the terminal resource process that connects to the bridge and renders the OMP event stream; it must never write directly to OMP's stdin.

Follow Pi/Claude native conventions for terminal resource labels, native agent registration, launch locking, terminal-required semantics, cleanup, and resume picker scope. Add `_launch_omp` through the native-provider dispatch seam rather than a new harness-name conditional.

- [ ] **Step 4: Run bridge and native lifecycle tests**

Run: `uv run pytest tests/test_omp_native_bridge.py tests/test_omp_native.py tests/runner/test_app_sessions_native_terminals_autocreate.py tests/runner/test_app_sessions_native_workflow_init.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the OMP native bridge**

```bash
git add omnigent/omp_native_bridge.py omnigent/omp_native_terminal.py omnigent/omp_native.py omnigent/runner/native/orchestration.py omnigent/runner/native/__init__.py omnigent/native_coding_agents.py omnigent/_wrapper_labels.py omnigent/harness_plugins.py tests/test_omp_native_bridge.py tests/test_omp_native.py tests/runner/test_app_sessions_native_terminals_autocreate.py tests/runner/test_app_sessions_native_workflow_init.py
git commit -m "feat: add persistent OMP native bridge"
```

### Task 4: Wire browser model and effort controls through the OMP bridge

**Files:**
- Modify: `omnigent/runner/app.py`
- Modify: `omnigent/server/routes/_sessions/common.py`
- Modify: `omnigent/server/routes/_sessions/helpers.py`
- Modify: `omnigent/host/connect.py`
- Test: `tests/runner/test_app_sessions_native_events_lifecycle.py`
- Test: `tests/runner/test_app_sessions_native_events_options.py`
- Test: `tests/host/test_connect.py`
- Test: `tests/server/routes/test_sessions_model_options.py`

**Interfaces:**
- Consumes: an `omp-native` session, `OmpRpcBridge`, `OmpProfileConfig`, and existing model/effort event payloads.
- Produces: `/v1/sessions/{session_id}/omp-model-options`, OMP-specific model/effort event handlers, model-options cache readiness, and error responses that preserve prior session metadata on failure.

- [ ] **Step 1: Write failing browser-control tests**

```python
async def test_omp_model_change_persists_only_after_live_rpc_success(client) -> None:
    response = await client.post("/v1/sessions/conv_omp/events", json={
        "type": "model_change", "model": "ollama/kimi-k2.7-code:cloud"
    })
    assert response.status_code == 200
    assert bridge.model == ("ollama", "kimi-k2.7-code:cloud")
    assert config.persisted_default == "ollama/kimi-k2.7-code:cloud"

async def test_omp_effort_persistence_failure_keeps_previous_session_effort(client) -> None:
    ...
```

Cover wrong-harness model-options requests, catalog failure, missing bridge, rejected `set_model` / `set_thinking_level`, config persistence failure after a live update, and host model-options discovery.

- [ ] **Step 2: Run the focused browser-control tests to verify they fail**

Run: `uv run pytest tests/runner/test_app_sessions_native_events_lifecycle.py tests/runner/test_app_sessions_native_events_options.py tests/host/test_connect.py tests/server/routes/test_sessions_model_options.py -q`

Expected: FAIL because `omp-native` routes and handlers are absent.

- [ ] **Step 3: Implement OMP browser control dispatch**

Add the OMP wrapper label to the existing model-options endpoint mapping and provide the runner endpoint through `OmpRpcBridge.available_models()`. Route model and effort events to bridge methods that execute the live RPC mutation first, persist with `omp_config`, and only then patch/publish Omnigent metadata. Surface live or persistence failure as a non-success response; do not reuse the existing best-effort-forward-after-metadata path.

- [ ] **Step 4: Run the focused browser-control tests**

Run: `uv run pytest tests/runner/test_app_sessions_native_events_lifecycle.py tests/runner/test_app_sessions_native_events_options.py tests/host/test_connect.py tests/server/routes/test_sessions_model_options.py -q`

Expected: PASS.

- [ ] **Step 5: Commit browser control coordination**

```bash
git add omnigent/runner/app.py omnigent/server/routes/_sessions/common.py omnigent/server/routes/_sessions/helpers.py omnigent/host/connect.py tests/runner/test_app_sessions_native_events_lifecycle.py tests/runner/test_app_sessions_native_events_options.py tests/host/test_connect.py tests/server/routes/test_sessions_model_options.py
git commit -m "feat: coordinate OMP browser controls"
```

### Task 5: Add the `omnigent omp` launcher and top-level session behavior

**Files:**
- Modify: `omnigent/cli_native.py`
- Modify: `omnigent/cli.py`
- Modify: `omnigent/repl.py`
- Modify: `omnigent/harness_startup_config.py`
- Test: `tests/cli/test_cli.py`
- Test: `tests/test_wrapper_labels.py`
- Test: `tests/e2e_ui/chat/test_harness_support_level_split.py`
- Test: `tests/e2e_ui/start_session/test_start_session.py`

**Interfaces:**
- Consumes: `run_omp_native(server, session_id, resume_picker, model, thinking, profile, prompt, extra_args, auto_open_conversation)` and the OMP native agent registration.
- Produces: `omnigent omp` CLI command, OMP-only resume filtering, startup configuration keys `harness.omp-native.command` / `args`, and browser-visible OMP display/wrapper identity.

- [ ] **Step 1: Write failing launcher and UI identity tests**

```python
def test_omp_command_resolves_config_and_launches_native_wrapper(runner, monkeypatch) -> None:
    result = runner.invoke(cli, ["omp", "--model", "ollama/kimi-k2.7-code:cloud", "--thinking", "high"])
    assert result.exit_code == 0
    assert captured["model"] == "ollama/kimi-k2.7-code:cloud"
    assert captured["thinking"] == "high"

async def test_start_session_renders_omp_and_stamps_omp_native_wrapper(page) -> None:
    ...
```

Cover `--resume` filtering, `--profile`, mutually exclusive legacy session/resume inputs, missing OMP executable, pass-through args, harness-config command override, browser open behavior, and native-agent display name.

- [ ] **Step 2: Run the launcher and UI identity tests to verify they fail**

Run: `uv run pytest tests/cli/test_cli.py tests/test_wrapper_labels.py tests/e2e_ui/chat/test_harness_support_level_split.py tests/e2e_ui/start_session/test_start_session.py -q`

Expected: FAIL because the `omp` command and native identity do not exist.

- [ ] **Step 3: Implement `omnigent omp` through native command conventions**

Register the command next to Claude/Codex/Pi with the shared server/resume validation behavior. Resolve executable/args from the established harness-startup config, create/attach the top-level native session, and pass model, thinking, profile, and initial prompt into the OMP launch record. Ensure session labels select terminal-first OMP-native behavior and browser routes display `OMP`, not raw internal identifiers.

- [ ] **Step 4: Run the launcher and UI identity tests**

Run: `uv run pytest tests/cli/test_cli.py tests/test_wrapper_labels.py tests/e2e_ui/chat/test_harness_support_level_split.py tests/e2e_ui/start_session/test_start_session.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the OMP native launcher**

```bash
git add omnigent/cli_native.py omnigent/cli.py omnigent/repl.py omnigent/harness_startup_config.py tests/cli/test_cli.py tests/test_wrapper_labels.py tests/e2e_ui/chat/test_harness_support_level_split.py tests/e2e_ui/start_session/test_start_session.py
git commit -m "feat: launch OMP with Omnigent"
```

### Task 6: Make OMP the Polly and Debby default

**Files:**
- Create: `examples/polly/agents/omp/config.yaml`
- Create: `examples/debby/agents/omp/config.yaml`
- Modify: `examples/polly/config.yaml`
- Modify: `examples/debby/config.yaml`
- Test: `tests/omnigent/test_example_polly.py`
- Test: `tests/spec/test_debby_example.py`
- Test: `tests/test_opencode_polly_debby_worker.py`
- Test: `tests/e2e/test_polly_e2e.py`

**Interfaces:**
- Consumes: the `harness: omp` headless executor and existing Polly/Debby worker selection/config formats.
- Produces: default OMP brain/worker entries effective 2026-11-06, with explicit harness/model overrides retaining their existing precedence.
- [ ] **Step 1: Write failing default-selection tests**

```python
def test_polly_and_debby_default_to_headless_omp() -> None:
    for path in (POLLY_CONFIG, DEBBY_CONFIG):
        config = yaml.safe_load(path.read_text())
        assert config["executor"]["config"]["harness"] == "omp"

def test_explicit_worker_override_remains_authoritative() -> None:
    assert resolve_worker_harness(explicit="codex") == "codex"
```

Test that both OMP worker files use `harness: omp` rather than `omp-native`; that Polly's availability guidance detects `omp`; and that Debby's two-head contract becomes OMP plus GPT without preserving a Claude dependency.

- [ ] **Step 2: Run worker tests to verify they fail**

Run: `uv run pytest tests/omnigent/test_example_polly.py tests/spec/test_debby_example.py tests/test_opencode_polly_debby_worker.py tests/e2e/test_polly_e2e.py -q`

Expected: FAIL because the bundled defaults and OMP worker configurations do not exist.

- [ ] **Step 3: Add OMP defaults and preserve explicit overrides**

Mirror the scoped-worker contract from Polly's Pi agent, with OMP-specific name/description and `harness: omp`. Change Polly's brain and default worker routing from Claude to OMP, update its binary preflight to require `omp`, and retain explicit `--harness` / configured override precedence. Change Debby's brain and first debate head from Claude to OMP while retaining GPT as the independent second head; update user-facing prompt text and setup requirements to remove the cancelled Claude dependency. Record 2026-11-06 in the bundled configuration/documentation comment as the effective default-change date.

- [ ] **Step 4: Run worker tests**

Run: `uv run pytest tests/omnigent/test_example_polly.py tests/spec/test_debby_example.py tests/test_opencode_polly_debby_worker.py tests/e2e/test_polly_e2e.py -q`

Expected: PASS.

- [ ] **Step 5: Commit OMP default worker support**

```bash
git add examples/polly/agents/omp/config.yaml examples/debby/agents/omp/config.yaml examples/polly/config.yaml examples/debby/config.yaml tests/omnigent/test_example_polly.py tests/spec/test_debby_example.py tests/test_opencode_polly_debby_worker.py tests/e2e/test_polly_e2e.py
git commit -m "feat: default polly and debby to OMP"
```

### Task 7: Verify the integrated native harness flow

**Files:**
- Modify: `docs/superpowers/specs/2026-11-02-omp-native-harness-design.md`
- Test: `tests/e2e_ui/chat/test_omp_native_session.py`

**Interfaces:**
- Consumes: completed `omp`, `omp-native`, OMP bridge, launcher, browser controls, and worker registrations.
- Produces: an end-to-end regression that covers the design's top-level coordination contract and documents any implementation-only clarifications discovered while executing it.

- [ ] **Step 1: Write the end-to-end native-session test**

```python
async def test_omp_terminal_browser_and_resume_share_one_native_session(page, fake_omp) -> None:
    session = await launch_omp_native(profile="work")
    await session.terminal.submit("from terminal")
    await expect(page.get_by_test_id("chat-transcript")).to_contain_text("from terminal")
    await page.get_by_test_id("composer").fill("from browser")
    await page.get_by_test_id("composer-send").click()
    assert fake_omp.prompt_messages[-1] == "from browser"
    resumed = await resume_omp_native(session.id)
    assert resumed.omp_argv_contains("--resume", fake_omp.session_file)
```

Also pin model/effort persistence, an extension prompt round trip, and an observable failure when the stored OMP session file is unavailable.

- [ ] **Step 2: Run the end-to-end test to verify it fails before final wiring**

Run: `uv run pytest tests/e2e_ui/chat/test_omp_native_session.py -q`

Expected: FAIL until the full OMP native flow is wired.

- [ ] **Step 3: Complete integration gaps exposed by the end-to-end contract**

Make only the targeted fixes required for the test and design acceptance criteria. Update the design document if an implementation-only transport detail needs to be recorded; do not broaden user-facing scope.

- [ ] **Step 4: Run focused regression and smoke verification**

Run: `uv run pytest tests/inner/test_omp_executor.py tests/test_omp_config.py tests/test_omp_native_bridge.py tests/test_omp_native.py tests/cli/test_cli.py tests/e2e_ui/chat/test_omp_native_session.py tests/omnigent/test_example_polly.py tests/spec/test_debby_example.py -q`

Expected: PASS.

Then run manually with a non-production profile:

```bash
omnigent omp --profile <test-profile>
```

Verify that the created browser URL and terminal share a single conversation; change model and effort in the browser; exit and run `omnigent omp --profile <test-profile> --resume`; confirm the OMP session resumes with the selected persisted model and effort.

- [ ] **Step 5: Commit final integration verification**

```bash
git add docs/superpowers/specs/2026-11-02-omp-native-harness-design.md tests/e2e_ui/chat/test_omp_native_session.py
git commit -m "test: verify OMP native session coordination"
```

## Self-Review

- **Spec coverage:** Tasks 1–5 cover the top-level native RPC bridge, browser coordination, profile persistence, model/effort controls, durable OMP session-file resume, and launcher. Task 6 makes the headless OMP worker the Polly/Debby default effective 2026-11-06 while preserving explicit overrides. Task 7 exercises the end-to-end contract. No requirement is unassigned.
- **Step scan:** Every task begins with a named failing behavioral test, names the exact production interfaces/files, verifies focused passing tests, and commits an independently reviewable deliverable.
- **Type consistency:** `OmpExecutor` owns protocol-level commands; `OmpRpcBridge` owns the long-lived top-level process/socket; `omp_config` owns durable profile mutations; `run_omp_native` owns launch/resume. Later tasks consume only those names.
- **Review Focus:** Task 1 covers malformed frames, Task 3 covers missing session files and terminal/browser concurrency, Task 4 covers mutation/persistence ordering, and Task 6 pins the default headless-worker boundary.
- **Proportion:** The plan records integration decisions and tests without prescribing implementation bodies; protocol parsing and terminal rendering remain local implementation details constrained by the documented RPC contract.
