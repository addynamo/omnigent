"""Idle-session reaper — periodically stops sessions whose runner has sat idle
past a configured TTL.

Every session (``omni run``, ``omnigent attach``, a scheduled-task firing, a
Slack thread, ...) keeps its runner process resident after the launching
command exits, so follow-up turns don't cold-start. There was previously no
automatic cleanup: idle sessions accumulated resident runner + native-bridge
processes forever, discoverable only via ``omnigent host status --sessions``
and reapable only by hand via ``omnigent host stop-session <id>``. This module
runs that same stop on a timer for sessions idle past
:data:`DEFAULT_IDLE_SESSION_TTL_S`.

Deliberately scoped to single-user/no-auth deployments for now (see
:func:`reap_idle_sessions_once`'s ``local_single_user_enabled`` gate in
``app.py``'s wiring) — reaping in an accounts-enabled multi-tenant deployment
needs a real system identity to authorize the stop call, which is future work.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable

import httpx

from omnigent.entities import Conversation
from omnigent.stores.conversation_store import ConversationStore

# Matches ``omnigent/server/routes/_sessions/common.py``'s ``_STOP_SESSION_TYPE``.
_STOP_SESSION_EVENT_BODY = {"type": "stop_session", "data": {}}

_logger = logging.getLogger(__name__)

# How long a session may sit idle (no item append, no title change — anything
# that bumps ``Conversation.updated_at``) with a live runner before it's
# reaped. 0 or negative disables the reaper; unset defaults to this value.
DEFAULT_IDLE_SESSION_TTL_S = 2 * 60 * 60  # 2 hours

# How often the reaper sweeps for idle sessions.
DEFAULT_SWEEP_INTERVAL_S = 5 * 60  # 5 minutes

# How many runner-bound sessions to inspect per sweep. Generous relative to
# any single-host deployment's expected live-session count; a deployment large
# enough to need more should tune ``OMNIGENT_IDLE_REAPER_INTERVAL_S`` down
# instead of raising this, so a sweep stays cheap.
_SWEEP_PAGE_LIMIT = 500


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to *default* on unset/invalid."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        _logger.warning("idle_session_reaper: invalid %s=%r, using default %s", name, raw, default)
        return default


def idle_session_ttl_seconds() -> float:
    """The configured idle TTL, from ``OMNIGENT_IDLE_SESSION_TTL_S`` or the default."""
    return _env_float("OMNIGENT_IDLE_SESSION_TTL_S", DEFAULT_IDLE_SESSION_TTL_S)


def idle_reaper_sweep_interval_seconds() -> float:
    """The configured sweep interval, from ``OMNIGENT_IDLE_REAPER_INTERVAL_S`` or the default."""
    return _env_float("OMNIGENT_IDLE_REAPER_INTERVAL_S", DEFAULT_SWEEP_INTERVAL_S)


# ``(session_id) -> None`` — stops one session's runner. Injectable for tests.
StopSession = Callable[[str], Awaitable[None]]


def _runner_bound_candidates(
    conversation_store: ConversationStore,
    *,
    tunnel_registry: object,
) -> list[Conversation]:
    """Every ``kind="default"`` conversation with a runner tunnel live on this replica.

    Paginates ``conversation_store.list_conversations`` sorted oldest-updated-first
    (so the sweep naturally reaps the longest-idle sessions first if it's ever
    cut short by ``_SWEEP_PAGE_LIMIT``), filters to rows with a bound
    ``runner_id``, then confirms the tunnel is actually live on *this* replica —
    a runner already dead has nothing to reap and isn't this reaper's job to
    explain (its ``runner_online`` will correct itself via ``host status`` on
    the next read).

    Also excludes any conversation whose ``live_status`` is ``"running"`` or
    ``"waiting"`` — a turn in progress or a session parked on a pending
    elicitation can leave ``updated_at`` stale well past the TTL, since
    ``ConversationStore.set_session_live_status`` is documented to never bump
    ``updated_at``. ``updated_at`` staleness alone is not a reliable idleness
    signal for those states.
    """
    candidates: list[Conversation] = []
    after: str | None = None
    while len(candidates) < _SWEEP_PAGE_LIMIT:
        page = conversation_store.list_conversations(
            limit=min(100, _SWEEP_PAGE_LIMIT - len(candidates)),
            after=after,
            kind="default",
            sort_by="updated_at",
            order="asc",
        )
        for conv in page.data:
            if conv.live_status in ("running", "waiting"):
                continue
            if conv.runner_id and tunnel_registry.get(conv.runner_id) is not None:  # type: ignore[attr-defined]
                candidates.append(conv)
        if not page.has_more or not page.data:
            break
        after = page.last_id
    return candidates


async def reap_idle_sessions_once(
    *,
    conversation_store: ConversationStore,
    tunnel_registry: object,
    stop_session: StopSession,
    ttl_seconds: float,
    now: Callable[[], float] = time.time,
) -> int:
    """Stop every currently-idle, runner-bound session. Returns the count reaped.

    :param conversation_store: Source of truth for conversations + their
        ``runner_id``/``updated_at``.
    :param tunnel_registry: This replica's live runner-tunnel registry (only
        used for its ``.get(runner_id)`` liveness check).
    :param stop_session: Stops one session by id; see :data:`StopSession`.
    :param ttl_seconds: Idle threshold in seconds. Non-positive disables
        reaping (returns 0 without listing candidates).
    :param now: Injectable clock for tests.
    """
    if ttl_seconds <= 0:
        return 0
    cutoff = now() - ttl_seconds
    candidates = await asyncio.to_thread(
        _runner_bound_candidates, conversation_store, tunnel_registry=tunnel_registry
    )
    reaped = 0
    for conv in candidates:
        if conv.updated_at > cutoff:
            # List is oldest-first: once we hit a fresh one, everything after
            # it is fresher still.
            break
        try:
            await stop_session(conv.id)
        except Exception:
            _logger.exception("idle_session_reaper: failed to stop idle session %s", conv.id)
            continue
        reaped += 1
        _logger.info(
            "idle_session_reaper: stopped idle session %s (idle %.0fs, ttl %.0fs)",
            conv.id,
            now() - conv.updated_at,
            ttl_seconds,
        )
    return reaped


async def reap_idle_sessions_periodically(
    *,
    conversation_store: ConversationStore,
    tunnel_registry: object,
    stop_session: StopSession,
    ttl_seconds: float | None = None,
    interval_seconds: float | None = None,
) -> None:
    """Run :func:`reap_idle_sessions_once` on a fixed interval until cancelled.

    Mirrors :func:`omnigent.server.performance_metrics.publish_server_metrics_periodically`'s
    shape — a bare ``while True: sleep; do the thing`` task, not a class with
    its own lifecycle, matching how this server wires other periodic
    background work in ``app.py``'s lifespan.

    :param ttl_seconds: Overrides :func:`idle_session_ttl_seconds` when given
        (mainly for tests); ``None`` reads the env-configured default.
    :param interval_seconds: Overrides :func:`idle_reaper_sweep_interval_seconds`
        when given; ``None`` reads the env-configured default.
    """
    ttl = ttl_seconds if ttl_seconds is not None else idle_session_ttl_seconds()
    interval = (
        interval_seconds if interval_seconds is not None else idle_reaper_sweep_interval_seconds()
    )
    if ttl <= 0:
        _logger.info("idle_session_reaper: disabled (OMNIGENT_IDLE_SESSION_TTL_S <= 0)")
        return
    if interval <= 0:
        _logger.warning(
            "idle_session_reaper: invalid interval_seconds=%s, using default %s",
            interval,
            DEFAULT_SWEEP_INTERVAL_S,
        )
        interval = DEFAULT_SWEEP_INTERVAL_S
    while True:
        await asyncio.sleep(interval)
        try:
            reaped = await reap_idle_sessions_once(
                conversation_store=conversation_store,
                tunnel_registry=tunnel_registry,
                stop_session=stop_session,
                ttl_seconds=ttl,
            )
            if reaped:
                _logger.info("idle_session_reaper: reaped %d idle session(s)", reaped)
        except Exception:
            _logger.exception("idle_session_reaper: sweep failed")


def make_local_stop_session(app: object) -> StopSession:
    """Build a :data:`StopSession` that posts ``stop_session`` in-process via ASGI.

    Reuses the exact, already-tested ``POST /v1/sessions/{id}/events`` route
    (the same one ``omnigent host stop-session`` hits over real HTTP) rather
    than re-threading the lower-level runner-client-resolution internals in
    ``routes/_sessions/orchestration.py`` — those have subtle
    persist-before-forward invariants documented right on
    ``_dispatch_session_event_to_runner_impl`` that aren't worth duplicating
    for a periodic sweep. Going through ASGI in-process avoids a real network
    hop while still exercising every bit of auth/validation logic a real
    client would.

    Single-user/no-auth only for now — see this module's docstring.
    """

    async def _stop(session_id: str) -> None:
        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(
            transport=transport, base_url="http://idle-session-reaper"
        ) as client:
            response = await client.post(
                f"/v1/sessions/{session_id}/events",
                json=_STOP_SESSION_EVENT_BODY,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"stop_session event for {session_id!r} failed: "
                    f"{response.status_code} {response.text}"
                )

    return _stop
