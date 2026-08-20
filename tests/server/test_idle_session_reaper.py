"""Tests for :mod:`omnigent.server.idle_session_reaper`.

Exercises :func:`reap_idle_sessions_once` against fakes for the conversation
store, tunnel registry, and stop-session seam — no real DB, no real ASGI app,
no wall-clock sleeps. Also covers :func:`reap_idle_sessions_periodically`'s
own differentiating behavior: the per-iteration try/except that keeps one
failed sweep from permanently killing the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field

from omnigent.server.idle_session_reaper import (
    reap_idle_sessions_once,
    reap_idle_sessions_periodically,
)


@dataclass
class _FakeConversation:
    """The slice of ``Conversation`` the reaper reads."""

    id: str
    updated_at: float
    runner_id: str | None


@dataclass
class _FakePage:
    data: list[_FakeConversation]
    has_more: bool = False
    last_id: str | None = None


class FakeConversationStore:
    """Single-page stand-in for ``ConversationStore.list_conversations``.

    Real pagination (multi-page sweeps) isn't exercised here — the reaper's
    paging loop is a thin, standard cursor walk with no idle-specific logic in
    it, so a single fake page keeps these tests focused on the sweep decision
    (idle vs. fresh, runner-bound vs. not) rather than re-testing pagination.
    """

    def __init__(self, conversations: list[_FakeConversation]) -> None:
        self._conversations = conversations

    def list_conversations(self, **kwargs: object) -> _FakePage:
        # The reaper always requests sort_by="updated_at", order="asc" (see
        # ``_runner_bound_candidates``) and relies on that ordering to break
        # early once it hits a fresh conversation — sort here the same way a
        # real store would, rather than trusting caller-supplied order.
        ordered = sorted(self._conversations, key=lambda c: c.updated_at)
        return _FakePage(data=ordered)


class FakeTunnelRegistry:
    """Stands in for the live runner-tunnel registry's ``.get(runner_id)``."""

    def __init__(self, online_runner_ids: set[str]) -> None:
        self._online = online_runner_ids

    def get(self, runner_id: str) -> object | None:
        return object() if runner_id in self._online else None


@dataclass
class StopSessionSpy:
    """Records every session id passed to the stop-session seam."""

    stopped: list[str] = field(default_factory=list)
    raise_for: set[str] = field(default_factory=set)

    async def __call__(self, session_id: str) -> None:
        if session_id in self.raise_for:
            raise RuntimeError(f"boom stopping {session_id}")
        self.stopped.append(session_id)


_NOW = 1_800_000_000.0


def _fake_now() -> float:
    return _NOW


async def test_reaps_only_sessions_idle_past_ttl() -> None:
    ttl = 3600.0
    store = FakeConversationStore(
        [
            _FakeConversation(id="idle-1", updated_at=_NOW - ttl - 100, runner_id="r1"),
            _FakeConversation(id="fresh-1", updated_at=_NOW - ttl + 100, runner_id="r2"),
            _FakeConversation(id="idle-2", updated_at=_NOW - ttl - 1, runner_id="r3"),
        ]
    )
    tunnels = FakeTunnelRegistry(online_runner_ids={"r1", "r2", "r3"})
    stop = StopSessionSpy()

    reaped = await reap_idle_sessions_once(
        conversation_store=store,
        tunnel_registry=tunnels,
        stop_session=stop,
        ttl_seconds=ttl,
        now=_fake_now,
    )

    assert reaped == 2
    assert set(stop.stopped) == {"idle-1", "idle-2"}


async def test_skips_sessions_with_no_runner_bound() -> None:
    """A session with ``runner_id is None`` has nothing resident to reap."""
    ttl = 3600.0
    store = FakeConversationStore(
        [_FakeConversation(id="no-runner", updated_at=_NOW - ttl - 100, runner_id=None)]
    )
    tunnels = FakeTunnelRegistry(online_runner_ids=set())
    stop = StopSessionSpy()

    reaped = await reap_idle_sessions_once(
        conversation_store=store,
        tunnel_registry=tunnels,
        stop_session=stop,
        ttl_seconds=ttl,
        now=_fake_now,
    )

    assert reaped == 0
    assert stop.stopped == []


async def test_skips_sessions_whose_runner_tunnel_is_already_gone() -> None:
    """A runner_id that isn't live on this replica's tunnel registry is
    already offline — nothing for this reaper to stop, and it isn't this
    reaper's job to explain a dead tunnel that will self-correct on the next
    ``host status`` read."""
    ttl = 3600.0
    store = FakeConversationStore(
        [_FakeConversation(id="dead-tunnel", updated_at=_NOW - ttl - 100, runner_id="r-dead")]
    )
    tunnels = FakeTunnelRegistry(online_runner_ids=set())  # r-dead is NOT live
    stop = StopSessionSpy()

    reaped = await reap_idle_sessions_once(
        conversation_store=store,
        tunnel_registry=tunnels,
        stop_session=stop,
        ttl_seconds=ttl,
        now=_fake_now,
    )

    assert reaped == 0
    assert stop.stopped == []


async def test_disabled_when_ttl_is_non_positive() -> None:
    store = FakeConversationStore([_FakeConversation(id="idle-1", updated_at=0, runner_id="r1")])
    tunnels = FakeTunnelRegistry(online_runner_ids={"r1"})
    stop = StopSessionSpy()

    for ttl in (0.0, -1.0):
        reaped = await reap_idle_sessions_once(
            conversation_store=store,
            tunnel_registry=tunnels,
            stop_session=stop,
            ttl_seconds=ttl,
            now=_fake_now,
        )
        assert reaped == 0
    assert stop.stopped == []


async def test_a_failed_stop_does_not_abort_the_rest_of_the_sweep() -> None:
    ttl = 3600.0
    store = FakeConversationStore(
        [
            _FakeConversation(id="idle-fails", updated_at=_NOW - ttl - 300, runner_id="r1"),
            _FakeConversation(id="idle-ok", updated_at=_NOW - ttl - 100, runner_id="r2"),
        ]
    )
    tunnels = FakeTunnelRegistry(online_runner_ids={"r1", "r2"})
    stop = StopSessionSpy(raise_for={"idle-fails"})

    reaped = await reap_idle_sessions_once(
        conversation_store=store,
        tunnel_registry=tunnels,
        stop_session=stop,
        ttl_seconds=ttl,
        now=_fake_now,
    )

    # Only the successful stop counts; the failing one is logged, not raised.
    assert reaped == 1
    assert stop.stopped == ["idle-ok"]


async def test_periodic_sweep_survives_a_failed_iteration_and_keeps_going() -> None:
    """A sweep-level failure (e.g. the store raising) must be logged and
    swallowed by ``reap_idle_sessions_periodically``'s own try/except, not
    left to kill the background task — the loop should reach a later,
    successful iteration."""
    calls = 0

    class RaisingThenSucceedingStore:
        def list_conversations(self, **kwargs: object) -> _FakePage:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("boom")
            return _FakePage(data=[])

    tunnels = FakeTunnelRegistry(online_runner_ids=set())
    stop = StopSessionSpy()

    task = asyncio.create_task(
        reap_idle_sessions_periodically(
            conversation_store=RaisingThenSucceedingStore(),
            tunnel_registry=tunnels,
            stop_session=stop,
            ttl_seconds=3600.0,
            interval_seconds=0.0001,
        )
    )
    try:
        for _ in range(1000):
            if calls >= 2:
                break
            await asyncio.sleep(0)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert calls >= 2
