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

import omnigent.server.idle_session_reaper as idle_session_reaper
from omnigent.server.idle_session_reaper import (
    DEFAULT_IDLE_SESSION_TTL_S,
    _env_float,
    _runner_bound_candidates,
    reap_idle_sessions_once,
    reap_idle_sessions_periodically,
    reaper_safe_for_auth_mode,
)


def test_reaper_safe_for_auth_mode_requires_both_single_user_and_header(monkeypatch) -> None:
    """``OMNIGENT_LOCAL_SINGLE_USER`` alone is not enough — an operator can
    (per ``create_auth_provider``'s own docstring) explicitly force
    accounts/OIDC auth while the single-user marker is also set, in which
    case an unauthenticated request does NOT succeed as the reserved "local"
    user and the reaper's cookie-less dispatch would get 401'd. Only the
    header+single-user combination is actually safe."""
    monkeypatch.setenv("OMNIGENT_LOCAL_SINGLE_USER", "1")
    monkeypatch.setenv("OMNIGENT_AUTH_PROVIDER", "header")
    assert reaper_safe_for_auth_mode() is True

    monkeypatch.setenv("OMNIGENT_AUTH_PROVIDER", "accounts")
    assert reaper_safe_for_auth_mode() is False

    monkeypatch.setenv("OMNIGENT_AUTH_PROVIDER", "oidc")
    assert reaper_safe_for_auth_mode() is False

    monkeypatch.setenv("OMNIGENT_AUTH_PROVIDER", "header")
    monkeypatch.setenv("OMNIGENT_LOCAL_SINGLE_USER", "0")
    assert reaper_safe_for_auth_mode() is False

    monkeypatch.delenv("OMNIGENT_LOCAL_SINGLE_USER", raising=False)
    assert reaper_safe_for_auth_mode() is False


def test_env_float_rejects_non_finite_values(monkeypatch) -> None:
    """``float()`` parses "nan"/"inf"/"-inf" without raising — a nan TTL would
    make every ``ttl <= 0`` and ``updated_at > cutoff`` comparison downstream
    false, silently reaping every live session instead of disabling the
    reaper or falling back to a sane default."""
    for raw in ("nan", "NaN", "inf", "-inf", "Infinity"):
        monkeypatch.setenv("OMNIGENT_IDLE_SESSION_TTL_S", raw)
        assert _env_float("OMNIGENT_IDLE_SESSION_TTL_S", DEFAULT_IDLE_SESSION_TTL_S) == (
            DEFAULT_IDLE_SESSION_TTL_S
        )


def test_env_float_accepts_normal_values(monkeypatch) -> None:
    monkeypatch.setenv("OMNIGENT_IDLE_SESSION_TTL_S", "42.5")
    assert _env_float("OMNIGENT_IDLE_SESSION_TTL_S", DEFAULT_IDLE_SESSION_TTL_S) == 42.5


@dataclass
class _FakeConversation:
    """The slice of ``Conversation`` the reaper reads."""

    id: str
    updated_at: float
    runner_id: str | None
    live_status: str | None = None


@dataclass
class _FakePage:
    data: list[_FakeConversation]
    has_more: bool = False
    last_id: str | None = None


class FakeConversationStore:
    """Real-pagination stand-in for ``ConversationStore.list_conversations`` +
    ``get_conversation``, keyed by id so a test can mutate one entry between
    the listing pass and the reaper's own pre-stop freshness re-check —
    exactly the race window the re-check exists to close.

    Honors ``limit``/``after`` like the real store's cursor pagination, so
    tests can exercise the multi-page walk in ``_runner_bound_candidates``
    (e.g. confirming it stops after inspecting ``_SWEEP_PAGE_LIMIT`` rows
    total, not just ``_SWEEP_PAGE_LIMIT`` matches).
    """

    def __init__(self, conversations: list[_FakeConversation]) -> None:
        self._by_id = {c.id: c for c in conversations}

    def list_conversations(
        self, *, limit: int = 20, after: str | None = None, **kwargs: object
    ) -> _FakePage:
        # The reaper always requests sort_by="updated_at", order="asc" (see
        # ``_runner_bound_candidates``) and relies on that ordering to break
        # early once it hits a fresh conversation — sort here the same way a
        # real store would, rather than trusting caller-supplied order.
        ordered = sorted(self._by_id.values(), key=lambda c: c.updated_at)
        start = 0
        if after is not None:
            ids = [c.id for c in ordered]
            start = ids.index(after) + 1
        page = ordered[start : start + limit]
        has_more = start + limit < len(ordered)
        return _FakePage(data=page, has_more=has_more, last_id=page[-1].id if page else None)

    def get_conversation(self, conversation_id: str) -> _FakeConversation | None:
        return self._by_id.get(conversation_id)

    def mark_active(self, conversation_id: str, *, now: float) -> None:
        """Simulate a follow-up turn starting after the listing pass: bumps
        ``updated_at`` to *now* and flips ``live_status`` to ``"running"``."""
        conv = self._by_id[conversation_id]
        self._by_id[conversation_id] = _FakeConversation(
            id=conv.id, updated_at=now, runner_id=conv.runner_id, live_status="running"
        )


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


async def test_skips_sessions_that_are_running_or_waiting_despite_stale_updated_at() -> None:
    """``live_status`` is the authoritative "turn in progress" signal and is
    documented to never bump ``updated_at`` — a session mid-turn or parked on
    a pending elicitation must not be reaped just because ``updated_at`` looks
    stale."""
    ttl = 3600.0
    store = FakeConversationStore(
        [
            _FakeConversation(
                id="running", updated_at=_NOW - ttl - 100, runner_id="r1", live_status="running"
            ),
            _FakeConversation(
                id="waiting", updated_at=_NOW - ttl - 200, runner_id="r2", live_status="waiting"
            ),
            _FakeConversation(
                id="idle", updated_at=_NOW - ttl - 300, runner_id="r3", live_status="idle"
            ),
            _FakeConversation(
                id="never-reported", updated_at=_NOW - ttl - 400, runner_id="r4", live_status=None
            ),
        ]
    )
    tunnels = FakeTunnelRegistry(online_runner_ids={"r1", "r2", "r3", "r4"})
    stop = StopSessionSpy()

    reaped = await reap_idle_sessions_once(
        conversation_store=store,
        tunnel_registry=tunnels,
        stop_session=stop,
        ttl_seconds=ttl,
        now=_fake_now,
    )

    assert reaped == 2
    assert set(stop.stopped) == {"idle", "never-reported"}


async def test_sweep_page_limit_bounds_rows_inspected_not_just_matches(monkeypatch) -> None:
    """``_SWEEP_PAGE_LIMIT`` must bound how many conversation rows a sweep
    scans, not how many runner-bound matches it finds — otherwise a
    deployment with few live runners but a large conversation history would
    rescan its entire history every sweep, since filtered-out (non-runner-
    bound) rows wouldn't otherwise stop the walk."""
    monkeypatch.setattr(idle_session_reaper, "_SWEEP_PAGE_LIMIT", 5)
    # None of these are runner-bound, so zero would ever become candidates —
    # if the walk were bounded by matches instead of rows inspected, it would
    # keep paginating through all 12 looking for a match that never comes.
    real_store = FakeConversationStore(
        [
            _FakeConversation(id=f"no-runner-{i}", updated_at=float(i), runner_id=None)
            for i in range(12)
        ]
    )
    rows_returned = 0

    class _CountingStore:
        def list_conversations(self, **kwargs: object) -> _FakePage:
            nonlocal rows_returned
            page = real_store.list_conversations(**kwargs)
            rows_returned += len(page.data)
            return page

    tunnels = FakeTunnelRegistry(online_runner_ids=set())

    candidates = await asyncio.to_thread(
        _runner_bound_candidates, _CountingStore(), tunnel_registry=tunnels
    )

    assert candidates == []
    # The patched limit is 5, not 12 — proves the walk stops on rows
    # inspected, not on ``len(candidates)`` (which never grows here and so
    # would never bound the old, buggy implementation's pagination).
    assert rows_returned == 5


async def test_revalidates_freshness_immediately_before_stopping() -> None:
    """A candidate can go active between the listing pass and its own stop
    dispatch — e.g. a user starts a follow-up turn while an earlier
    candidate's stop is still being processed. The pre-stop re-check must
    catch this and skip it, not kill a runner that's newly active again."""
    ttl = 3600.0
    store = FakeConversationStore(
        [
            _FakeConversation(id="first", updated_at=_NOW - ttl - 200, runner_id="r1"),
            _FakeConversation(id="second", updated_at=_NOW - ttl - 100, runner_id="r2"),
        ]
    )
    tunnels = FakeTunnelRegistry(online_runner_ids={"r1", "r2"})
    stop = StopSessionSpy()

    real_call = stop.__call__

    async def _stop_and_mutate(session_id: str) -> None:
        await real_call(session_id)
        if session_id == "first":
            # Simulate "second" starting a new turn while "first"'s stop was
            # in flight — this must be observed by the pre-stop re-check for
            # "second", not just the stale snapshot from the listing pass.
            store.mark_active("second", now=_NOW)

    reaped = await reap_idle_sessions_once(
        conversation_store=store,
        tunnel_registry=tunnels,
        stop_session=_stop_and_mutate,
        ttl_seconds=ttl,
        now=_fake_now,
    )

    assert reaped == 1
    assert stop.stopped == ["first"]


async def test_skips_candidate_deleted_between_listing_and_stop() -> None:
    """``get_conversation`` returning ``None`` (row gone) must be a clean
    skip, not an attribute-error crash on the re-check."""
    ttl = 3600.0
    store = FakeConversationStore(
        [_FakeConversation(id="vanishes", updated_at=_NOW - ttl - 100, runner_id="r1")]
    )

    class _DeletesAfterListing:
        """Answers ``list_conversations`` from the real fake, but always
        reports the row gone on the freshness re-check — the listing pass
        must still see it (so the candidate loop runs) while the re-check
        catches its disappearance."""

        def list_conversations(self, **kwargs: object) -> _FakePage:
            return store.list_conversations(**kwargs)

        def get_conversation(self, conversation_id: str) -> None:
            return None

    tunnels = FakeTunnelRegistry(online_runner_ids={"r1"})
    stop = StopSessionSpy()

    reaped = await reap_idle_sessions_once(
        conversation_store=_DeletesAfterListing(),
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
