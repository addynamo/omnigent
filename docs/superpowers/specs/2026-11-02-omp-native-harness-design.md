# OMP Native Harness Design

## Intent

Add Oh My Pi (OMP) as a first-class Omnigent native harness. Running `omnigent omp` must create and coordinate an OMP agent session just as `omnigent claude` does: the terminal and the Omnigent conversation URL (for example, `http://localhost:6767/c/<session-id>`) are two views of one live session.

## Goals

- Add `omnigent omp` as a terminal-first native launcher with `--server`, `--resume`, `--model`, `--thinking`, `--profile`, `--prompt`, and native-CLI pass-through arguments.
- Register `omp-native` as the terminal-first harness and expose an `OMP` native coding-agent entry in existing agent pickers, session routing, terminal lifecycle, resume flow, and capability metadata. Register `omp` as the matching headless RPC harness and, effective 2026-11-06, make it the default brain/worker harness for Polly and Debby while preserving explicit harness overrides.
- Drive OMP with its documented native RPC protocol (`omp --mode rpc`), not as an indirect model provider. Omnigent owns the persistent RPC bridge and terminal RPC client; OMP owns its agent loop, tools, compaction, skills, profile state, and persisted transcript.
- Coordinate the terminal and `/c/<session-id>` as simultaneous views of one RPC session: terminal and browser turns enter the same OMP queue, and streamed text/reasoning, tool-card state, native extension UI requests, permission requests, cancellation, errors, and session status are mirrored in the browser.
- Populate the OMP model control from OMP's `get_available_models` RPC command. A model changed in Omnigent must update the live RPC session with `set_model` and persist the selected primary model to the chosen profile's `modelRoles.default`.
- Map Omnigent's effort control to OMP thinking levels (`minimal`, `low`, `medium`, `high`, `xhigh`, and `max` where supported), apply it through `set_thinking_level`, and persist the selected profile's `defaultThinkingLevel`.
- Treat the selected profile as an explicit launch/configuration boundary: configuration reads and writes must invoke `omp --profile <profile> config ...`; an omitted profile uses OMP's global/default profile.
- Persist the OMP `sessionFile` reported by `get_state` as the conversation's external-session reference. A resume starts OMP with `--resume <sessionFile>` so runner, terminal, and browser restarts restore OMP-native history and state instead of rebuilding context from the Omnigent transcript.

## Non-goals

- Do not copy Claude's authentication, gateway, transcript-forwarder, or provider model vocabulary.
- Do not mutate OMP configuration files directly; use OMP CLI configuration commands so OMP owns the schema and profile storage rules.
- Do not turn OMP's `smol`, `slow`, `vision`, fallback-chain, skills, or tool configuration into separate Omnigent controls in this delivery. They remain OMP-native configuration and continue to work unchanged.
- Do not create a generic ACP wrapper, a temporary shim, or a transcript-rebuild recovery path. `omp-native` is a registered native harness with a dedicated RPC bridge, terminal session identity, and OMP-native resume contract.

## Architecture

Use OMP's documented native RPC protocol because it exposes the controls and state Omnigent must coordinate: model and thinking-level mutation, model discovery, durable `sessionFile`, streaming agent/tool events, queue control, extension UI, and host-tool callbacks. Add an `OmpRpcBridge` that resolves the executable, launches `omp --mode rpc [global options]`, owns its JSONL stdin/stdout, and translates OMP frames into Omnigent session events.

The bridge exposes a per-session local socket to an `omp_native_terminal` client launched in Omnigent's terminal resource. That client, rather than OMP, owns the human terminal: it renders bridge events and turns terminal input into bridge commands. Browser turns and controls invoke the same bridge, so no second OMP process or competing stdin reader exists. The CLI command creates or resumes an `omp-native` session, binds the selected profile and OMP session file in durable metadata/labels, and on resume launches `omp --mode rpc --resume <sessionFile>`.

The server/runner provides OMP model and effort controls through the RPC bridge; a browser setting is persisted to OMP's profile only after the live RPC command succeeds, and Omnigent session metadata changes only after persistence succeeds. Effective 2026-11-06, Polly and Debby default to their headless `omp` brain/worker configuration because Claude is cancelled; explicit harness/model selection continues to override this default, and no runtime quota probing is introduced.

## Acceptance Criteria

1. `omnigent omp --help` is available; `omnigent omp` starts an `omp-native` session, and `omnigent omp --resume` lists only OMP-native conversations.
2. A launched OMP turn sent from the terminal appears at `/c/<session-id>`, and a browser-originated turn reaches the same persistent OMP RPC process; streamed text, reasoning, tools, native extension prompts, permission prompts, errors, and cancellation are reflected in the browser.
3. The native-agent selector renders `OMP`, a top-level created session uses the standard terminal-first wrapper/session identity for `omp-native`, and, effective 2026-11-06, Polly and Debby default to headless `omp` execution while preserving an explicit override.
4. Model options come from OMP RPC `get_available_models`; selecting an option succeeds through RPC `set_model` before it persists `modelRoles.default` through `omp [--profile] config set` without overwriting unrelated roles.
5. Changing effort succeeds through RPC `set_thinking_level` before it persists `defaultThinkingLevel` through `omp [--profile] config set`.
6. New direct `omp` terminal sessions launched with the same profile observe the model and effort selected through Omnigent.
7. Reopen/resume starts OMP with the persisted OMP session file and retains the model, effort, OMP-native history, and relationship to the same Omnigent conversation.
8. Missing OMP, malformed OMP catalog/config output, rejected live config changes, or CLI persistence failures produce explicit launcher/UI errors and do not report a successful setting change.

## Constraints

- Reuse the native harness registry, session metadata, capability model, terminal/resource lifecycle, and existing browser conversation transport; implement a focused OMP RPC bridge rather than an ACP adapter.
- Match existing CLI-native command conventions and test organization.
- Tests must use fake OMP RPC subprocess/CLI seams; no test may require a real cloud model or mutate the developer's real OMP configuration.
- The browser conversation remains accessible at its normal Omnigent URL and is the authority for shared Omnigent history and session state; OMP remains authority for its profile configuration, native transcript, and agent execution state.
