"""
Tests for L{klein._changes}, the two-phase resource change protocol.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from io import BytesIO
from typing import Any, ClassVar, cast
from unittest.mock import Mock
from urllib.parse import parse_qs

from twisted.internet.defer import Deferred, fail, succeed
from twisted.trial.unittest import SynchronousTestCase
from twisted.web.server import NOT_DONE_YET, Request
from twisted.web.test.test_web import DummyChannel

from .. import Klein
from .._changes import (
    DELETE,
    Applied,
    CandidatePlan,
    Change,
    ChangeApplicationError,
    CommitResult,
    DuplicateIdentifier,
    EmptyChanges,
    StaleSnapshot,
    TwoPhaseChangeAPI,
    TwoPhaseResourceStore,
)


def succeedNow(testCase: SynchronousTestCase, deferred: Deferred[Any]) -> Any:
    """
    Assert C{deferred} has already fired and return its result.
    """
    return testCase.successResultOf(deferred)


def failNow(
    testCase: SynchronousTestCase,
    deferred: Deferred[Any],
    exceptionType: type[BaseException],
) -> BaseException:
    """
    Assert C{deferred} has already failed with C{exceptionType}.
    """
    failure = testCase.failureResultOf(deferred)
    failure.trap(exceptionType)
    return cast(BaseException, failure.value)


class MockRequest(Request):
    """
    A minimal synchronous L{Request} for rendering Klein resources.
    """

    def __init__(self, path: bytes, method: bytes = b"GET") -> None:
        super().__init__(DummyChannel(), False)
        self.site = Mock()
        self.gotLength(0)
        self.content = BytesIO(b"")
        path, query = (path.split(b"?", 1) + [b""])[:2]
        self.args = parse_qs(query)
        self.setHost(b"localhost", 80, False)
        self.uri = path
        self.prepath = []
        self.postpath = path.split(b"/")[1:]
        self.method = method
        self.clientproto = b"HTTP/1.1"
        self._written = BytesIO()

    def write(self, data: bytes) -> None:
        self._written.write(data)

    def finish(self) -> None:
        self.finished = True

    def body(self) -> dict[str, Any]:
        """
        Decode the written response as JSON.
        """
        return cast(
            dict[str, Any],
            json.loads(self._written.getvalue().decode("utf-8")),
        )


def constant(value: Any) -> Callable[[Any, object | None], Any]:
    """
    Build an executor that always returns C{value}.
    """

    def execute(current: Any, context: object | None) -> Any:
        return value

    return execute


class PlanPreparationTests(SynchronousTestCase):
    """
    Tests for phase one, L{TwoPhaseResourceStore.plan}.
    """

    def test_emptyInputRejected(self) -> None:
        """
        Preparing a plan with no changes fails with L{EmptyChanges}, bumps no
        version, and records no committed result.
        """
        store = TwoPhaseResourceStore()
        failureResult = store.plan([])
        failNow(self, failureResult, EmptyChanges)
        self.assertEqual(store.version, 0)
        self.assertIsNone(store.committedResult("plan-1"))

    def test_emptyIterableRejected(self) -> None:
        """
        Any empty iterable (not just an empty list) is rejected.
        """
        store = TwoPhaseResourceStore()
        failNow(self, store.plan(iter(())), EmptyChanges)

    def test_duplicateIdentifiersRejected(self) -> None:
        """
        Two changes targeting the same identifier fail planning with
        L{DuplicateIdentifier} naming the identifier, without executing
        anything.
        """
        executed: list[str] = []

        def execute(current: Any, context: object | None) -> Any:
            executed.append("called")
            return "x"

        store = TwoPhaseResourceStore()
        failureResult = store.plan(
            [Change("same", execute), Change("same", execute)]
        )
        error = failNow(self, failureResult, DuplicateIdentifier)
        assert isinstance(error, DuplicateIdentifier)
        self.assertEqual(error.identifier, "same")
        self.assertEqual(executed, [])
        self.assertEqual(store.version, 0)

    def test_planIsVersionedCandidate(self) -> None:
        """
        A plan captures the snapshot version, per-resource versions and the
        request context, but neither executes changes nor bumps the version.
        """
        store = TwoPhaseResourceStore()
        succeedNow(self, seed(store))

        context = object()
        changes = [Change("new", constant("v1")), Change("old", constant("v2"))]
        plan = succeedNow(self, store.plan(changes, context))

        self.assertIsInstance(plan, CandidatePlan)
        self.assertEqual(plan.identifier, "plan-2")
        self.assertEqual(plan.version, 1)
        self.assertEqual(dict(plan.resourceVersions), {"new": 0, "old": 1})
        self.assertIs(plan.context, context)
        self.assertEqual(
            [change.identifier for change in plan.changes],
            [
                "new",
                "old",
            ],
        )
        # Nothing was applied during planning.
        self.assertIsNone(store.get("new"))
        self.assertEqual(store.version, 1)

    def test_planIdentifiersAreUnique(self) -> None:
        """
        Every prepared plan gets its own token.
        """
        store = TwoPhaseResourceStore()
        first = succeedNow(self, store.plan([Change("a", constant(1))]))
        second = succeedNow(self, store.plan([Change("a", constant(2))]))
        self.assertNotEqual(first.identifier, second.identifier)


def seed(store: TwoPhaseResourceStore) -> Deferred[CommitResult]:
    """
    Create an initial C{old} resource at store version 1.
    """
    planned = store.plan([Change("old", constant("seed"))])
    return planned.addCallback(store.commit)


class CommitTests(SynchronousTestCase):
    """
    Tests for phase two, L{TwoPhaseResourceStore.commit}.
    """

    def test_commitAppliesAndBumpsVersion(self) -> None:
        """
        A successful commit publishes every staged value, gives each
        resource a per-resource version, and bumps the store version once.
        """
        store = TwoPhaseResourceStore()
        plan = succeedNow(
            self,
            store.plan(
                [Change("a", constant("A")), Change("b", constant("B"))]
            ),
        )
        result = succeedNow(self, store.commit(plan))

        self.assertIsInstance(result, CommitResult)
        self.assertFalse(result.replayed)
        self.assertEqual(result.planIdentifier, plan.identifier)
        self.assertEqual(result.version, 1)
        self.assertEqual(tuple(result.results), ("A", "B"))
        self.assertEqual(store.version, 1)
        self.assertEqual(store.get("a"), "A")
        self.assertEqual(store.get("b"), "B")
        self.assertEqual(store.resourceVersion("a"), 1)
        self.assertEqual(store.resourceVersion("b"), 1)

    def test_commitUpdatesAndDeletes(self) -> None:
        """
        Updates increment the resource version and L{DELETE} removes it.
        """
        store = TwoPhaseResourceStore()
        succeedNow(self, seed(store))

        plan = succeedNow(
            self,
            store.plan(
                [Change("old", constant(DELETE)), Change("new", constant("N"))]
            ),
        )
        succeedNow(self, store.commit(plan))

        self.assertIsNone(store.get("old"))
        self.assertEqual(store.resourceVersion("old"), 0)
        self.assertEqual(store.get("new"), "N")
        self.assertEqual(store.version, 2)

    def test_commitPassesCurrentValueAndContext(self) -> None:
        """
        Executors receive the current value and the plan's request context.
        """
        store = TwoPhaseResourceStore()
        succeedNow(self, seed(store))
        seen: list[tuple[Any, Any]] = []
        context = object()

        def execute(current: Any, givenContext: object | None) -> Any:
            seen.append((current, givenContext))
            return "updated"

        plan = succeedNow(self, store.plan([Change("old", execute)], context))
        succeedNow(self, store.commit(plan))
        self.assertEqual(seen, [("seed", context)])

    def test_staleSnapshotRejected(self) -> None:
        """
        Committing a plan whose snapshot version has moved fails with
        L{StaleSnapshot} and writes nothing.
        """
        store = TwoPhaseResourceStore()
        stale = succeedNow(self, store.plan([Change("a", constant("A"))]))

        # A different commit advances the store version first.
        other = succeedNow(self, store.plan([Change("z", constant("Z"))]))
        succeedNow(self, store.commit(other))

        error = failNow(self, store.commit(stale), StaleSnapshot)
        assert isinstance(error, StaleSnapshot)
        self.assertEqual(error.expectedVersion, 0)
        self.assertEqual(error.currentVersion, 1)
        self.assertIsNone(store.get("a"))
        self.assertEqual(store.get("z"), "Z")
        self.assertEqual(store.version, 1)

    def test_partialFailureLeavesNoHalfState(self) -> None:
        """
        When one change fails, no earlier or later change is published.
        """

        def broken(current: Any, context: object | None) -> Any:
            raise RuntimeError("boom")

        store = TwoPhaseResourceStore()
        plan = succeedNow(
            self,
            store.plan(
                [
                    Change("a", constant("A")),
                    Change("b", broken),
                    Change("c", constant("C")),
                ]
            ),
        )
        error = failNow(self, store.commit(plan), ChangeApplicationError)
        assert isinstance(error, ChangeApplicationError)
        self.assertEqual(error.identifier, "b")
        self.assertEqual(store.version, 0)
        self.assertIsNone(store.get("a"))
        self.assertIsNone(store.get("b"))
        self.assertIsNone(store.get("c"))

    def test_compensationRunsInReverseWithContext(self) -> None:
        """
        Compensating actions for already-applied changes run in reverse
        order, receiving the request context; an async compensation is
        awaited.
        """
        events: list[str] = []
        context = object()

        def compensated(
            name: str, asyncRollback: bool
        ) -> Callable[[Any, object | None], Any]:
            def execute(current: Any, givenContext: object | None) -> Any:
                def rollback(
                    rollbackContext: object | None,
                ) -> Deferred[None] | None:
                    self.assertIs(rollbackContext, context)
                    events.append(f"undo-{name}")
                    if asyncRollback:
                        return succeed(None)
                    return None

                return Applied(name, rollback)

            return execute

        def broken(current: Any, givenContext: object | None) -> Any:
            raise RuntimeError("boom")

        store = TwoPhaseResourceStore()
        plan = succeedNow(
            self,
            store.plan(
                [
                    Change("a", compensated("a", False)),
                    Change("b", compensated("b", True)),
                    Change("c", broken),
                ],
                context,
            ),
        )
        failNow(self, store.commit(plan), ChangeApplicationError)
        self.assertEqual(events, ["undo-b", "undo-a"])
        self.assertEqual(store.version, 0)

    def test_failingCompensationDoesNotMaskOriginal(self) -> None:
        """
        A compensating action that itself fails is logged but the original
        application failure is still reported.
        """

        def execute(current: Any, context: object | None) -> Any:
            def rollback(rollbackContext: object | None) -> None:
                raise RuntimeError("rollback failed")

            return Applied("A", rollback)

        def broken(current: Any, context: object | None) -> Any:
            raise RuntimeError("apply failed")

        store = TwoPhaseResourceStore()
        plan = succeedNow(
            self, store.plan([Change("a", execute), Change("b", broken)])
        )
        error = failNow(self, store.commit(plan), ChangeApplicationError)
        assert isinstance(error, ChangeApplicationError)
        self.assertEqual(error.identifier, "b")
        self.assertEqual(store.version, 0)
        # The failed compensating action was logged rather than masking the
        # original application failure.
        logged = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0].value.args, ("rollback failed",))

    def test_idempotentRetryReplaysResult(self) -> None:
        """
        Committing an already-committed plan again returns the original
        result flagged as a replay and never executes a change twice.
        """
        executions: list[str] = []

        def execute(current: Any, context: object | None) -> Any:
            executions.append("a")
            return "A"

        store = TwoPhaseResourceStore()
        plan = succeedNow(self, store.plan([Change("a", execute)]))
        first = succeedNow(self, store.commit(plan))
        second = succeedNow(self, store.commit(plan))
        third = succeedNow(self, store.commit(plan))

        self.assertFalse(first.replayed)
        self.assertTrue(second.replayed)
        self.assertTrue(third.replayed)
        self.assertEqual(first.version, second.version)
        self.assertEqual(executions, ["a"])
        self.assertEqual(store.version, 1)

    def test_failedCommitIsRetryable(self) -> None:
        """
        A failed commit is not recorded; fixing the failure and committing
        the same plan succeeds.
        """
        flags = {"fail": True}
        executions: list[str] = []

        def flaky(current: Any, context: object | None) -> Any:
            executions.append("b")
            if flags["fail"]:
                raise RuntimeError("temporary")
            return "B"

        store = TwoPhaseResourceStore()
        plan = succeedNow(self, store.plan([Change("b", flaky)]))
        failNow(self, store.commit(plan), ChangeApplicationError)
        self.assertEqual(store.version, 0)

        flags["fail"] = False
        result = succeedNow(self, store.commit(plan))
        self.assertEqual(result.version, 1)
        self.assertEqual(store.get("b"), "B")
        self.assertEqual(executions, ["b", "b"])


class AsyncChangeTests(SynchronousTestCase):
    """
    Tests for changes whose executors return L{Deferred}s.
    """

    def test_asynchronousFailurePreservesContextAndIsRetryable(
        self,
    ) -> None:
        """
        A failure delivered asynchronously reports the originating request
        context and plan on L{ChangeApplicationError}, publishes nothing,
        and the same plan may be retried afterwards.
        """
        context = object()
        flags = {"fail": True}
        rollbacks: list[object | None] = []
        bContexts: list[object | None] = []

        def first(current: Any, givenContext: object | None) -> Applied:
            def rollback(
                rollbackContext: object | None,
            ) -> None:
                rollbacks.append(rollbackContext)

            return Applied("A", rollback)

        def flaky(current: Any, givenContext: object | None) -> Deferred[str]:
            bContexts.append(givenContext)
            if flags["fail"]:
                return fail(RuntimeError("later"))
            return succeed("B")

        store = TwoPhaseResourceStore()
        plan = succeedNow(
            self,
            store.plan([Change("a", first), Change("b", flaky)], context),
        )

        firstAttempt = store.commit(plan)
        # The failing Deferred is already fired, so the chain resolves, but
        # the failure still travels through the asynchronous path.
        error = failNow(self, firstAttempt, ChangeApplicationError)
        assert isinstance(error, ChangeApplicationError)
        self.assertIs(error.context, context)
        self.assertIs(error.plan, plan)
        self.assertEqual(error.identifier, "b")
        error.failure.trap(RuntimeError)
        self.assertEqual(store.version, 0)
        self.assertEqual(rollbacks, [context])

        flags["fail"] = False
        result = succeedNow(self, store.commit(plan))
        self.assertEqual(result.version, 1)
        self.assertEqual(store.get("a"), "A")
        self.assertEqual(store.get("b"), "B")
        # The context survived into the retry as well.
        self.assertEqual(bContexts, [context, context])

    def test_inFlightCommitSerializesConcurrentCommit(self) -> None:
        """
        While one commit is awaiting an asynchronous change, a concurrent
        commit cannot slip in; once the first commits, the queued plan is
        stale and rejected without writing.
        """
        firstChange: Deferred[str] = Deferred()
        store = TwoPhaseResourceStore()
        firstPlan = succeedNow(
            self, store.plan([Change("a", lambda c, x: firstChange)])
        )
        secondPlan = succeedNow(self, store.plan([Change("b", constant("B"))]))

        firstAttempt = store.commit(firstPlan)
        secondAttempt = store.commit(secondPlan)
        self.assertNoResult(firstAttempt)
        self.assertNoResult(secondAttempt)

        firstChange.callback("A")

        firstResult = succeedNow(self, firstAttempt)
        self.assertEqual(firstResult.version, 1)
        error = failNow(self, secondAttempt, StaleSnapshot)
        assert isinstance(error, StaleSnapshot)
        self.assertEqual((error.expectedVersion, error.currentVersion), (0, 1))
        self.assertIsNone(store.get("b"))
        self.assertEqual(store.get("a"), "A")

    def test_failedAsyncCommitDoesNotBlockQueue(self) -> None:
        """
        A failed asynchronous commit releases the serialization slot so the
        next queued commit proceeds.
        """
        blocker: Deferred[None] = Deferred()

        def later(current: Any, context: object | None) -> Deferred[None]:
            return blocker

        store = TwoPhaseResourceStore()
        failingPlan = succeedNow(
            self,
            store.plan([Change("x", later), Change("y", later)]),
        )
        queuedPlan = succeedNow(self, store.plan([Change("q", constant("Q"))]))
        firstAttempt = store.commit(failingPlan)
        secondAttempt = store.commit(queuedPlan)
        self.assertNoResult(secondAttempt)

        blocker.errback(RuntimeError("nope"))
        failNow(self, firstAttempt, ChangeApplicationError)

        result = succeedNow(self, secondAttempt)
        self.assertEqual(result.version, 1)
        self.assertEqual(store.get("q"), "Q")

    def test_replayWhileAnotherCommitIsQueued(self) -> None:
        """
        Re-committing a completed plan that is behind a still-running
        commit waits its turn and replays the stored result.
        """
        gate: Deferred[str] = Deferred()
        store = TwoPhaseResourceStore()
        replayedPlan = succeedNow(
            self, store.plan([Change("a", constant("A"))])
        )
        completed = succeedNow(self, store.commit(replayedPlan))
        self.assertEqual(completed.version, 1)

        blockingPlan = succeedNow(
            self, store.plan([Change("b", lambda c, x: gate)])
        )
        blocking = store.commit(blockingPlan)
        replay = store.commit(replayedPlan)
        self.assertNoResult(replay)

        gate.callback("B")
        succeedNow(self, blocking)
        replayed = succeedNow(self, replay)
        self.assertTrue(replayed.replayed)
        self.assertEqual(replayed.version, 1)


class RenderedChangeApp:
    """
    A Klein application exposing the protocol over HTTP.
    """

    router: ClassVar[Klein] = Klein()

    def __init__(self) -> None:
        self.store = TwoPhaseResourceStore()
        self.api = TwoPhaseChangeAPI(self.store)
        self.failNext = True
        self.contexts: list[object | None] = []
        self.rollbacks: list[object | None] = []

    @router.route("/plan", methods=["POST"])
    async def planWidget(self, request: Request) -> Any:
        def widget(current: Any, context: object | None) -> str:
            return "widget-value"

        return await self.api.renderPlan(request, [Change("widget", widget)])

    @router.route("/plan-empty", methods=["POST"])
    async def planEmpty(self, request: Request) -> Any:
        return await self.api.renderPlan(request, [])

    @router.route("/plan-duplicate", methods=["POST"])
    async def planDuplicate(self, request: Request) -> Any:
        change = Change("same", constant("x"))
        return await self.api.renderPlan(request, [change, change])

    @router.route("/plan-other", methods=["POST"])
    async def planOther(self, request: Request) -> Any:
        return await self.api.renderPlan(
            request, [Change("other", constant("other-value"))]
        )

    @router.route("/plan-flaky", methods=["POST"])
    async def planFlaky(self, request: Request) -> Any:
        def first(current: Any, context: object | None) -> Applied:
            def rollback(
                rollbackContext: object | None,
            ) -> None:
                self.rollbacks.append(rollbackContext)

            return Applied("A", rollback)

        def flaky(current: Any, context: object | None) -> Deferred[str]:
            self.contexts.append(context)
            if self.failNext:
                return fail(RuntimeError("boom"))
            return succeed("B")

        return await self.api.renderPlan(
            request, [Change("a", first), Change("b", flaky)]
        )

    @router.route("/commit/<token>", methods=["POST"])
    async def commit(self, request: Request, token: str) -> Any:
        return await self.api.renderCommit(request, token)


class RenderedChangeAPITests(SynchronousTestCase):
    """
    End-to-end tests for L{TwoPhaseChangeAPI} rendered by Klein resources.
    """

    def render(self, app: RenderedChangeApp, path: bytes) -> MockRequest:
        request = MockRequest(path, b"POST")
        result = app.router.resource().render(request)
        self.assertEqual(result, NOT_DONE_YET)
        return request

    def test_planThenCommitRendered(self) -> None:
        """
        Planning renders a versioned candidate; committing its token writes
        the resource, and a repeat commit renders an idempotent replay.
        """
        app = RenderedChangeApp()

        planned = self.render(app, b"/plan")
        self.assertEqual(planned.code, 200)
        self.assertEqual(
            planned.body(),
            {"changes": ["widget"], "plan": "plan-1", "version": 0},
        )

        committed = self.render(app, b"/commit/plan-1")
        self.assertEqual(committed.code, 200)
        self.assertEqual(
            committed.body(),
            {"plan": "plan-1", "replayed": False, "version": 1},
        )
        self.assertEqual(app.store.get("widget"), "widget-value")

        replayed = self.render(app, b"/commit/plan-1")
        self.assertEqual(replayed.code, 200)
        self.assertEqual(replayed.body()["replayed"], True)

    def test_emptyAndDuplicateRender422(self) -> None:
        """
        Empty input and duplicate identifiers render 422 responses and
        write nothing.
        """
        app = RenderedChangeApp()

        empty = self.render(app, b"/plan-empty")
        self.assertEqual(empty.code, 422)
        self.assertEqual(empty.body(), {"error": "no changes supplied"})

        duplicate = self.render(app, b"/plan-duplicate")
        self.assertEqual(duplicate.code, 422)
        self.assertEqual(
            duplicate.body(),
            {"error": "duplicate identifier", "identifier": "same"},
        )
        self.assertEqual(app.store.version, 0)

    def test_staleCommitRenders409(self) -> None:
        """
        A commit against an expired snapshot renders 409, and the client can
        recover by preparing a new plan.
        """
        app = RenderedChangeApp()

        first = self.render(app, b"/plan")
        self.assertEqual(first.body()["version"], 0)
        other = self.render(app, b"/plan-other")
        self.assertEqual(other.body()["plan"], "plan-2")

        self.assertEqual(self.render(app, b"/commit/plan-2").code, 200)
        stale = self.render(app, b"/commit/plan-1")
        self.assertEqual(stale.code, 409)
        self.assertEqual(
            stale.body(),
            {
                "currentVersion": 1,
                "error": "snapshot expired; prepare a new plan",
                "expectedVersion": 0,
            },
        )

        # A fresh plan commits cleanly against the new version.
        replanned = self.render(app, b"/plan")
        self.assertEqual(replanned.body()["version"], 1)
        self.assertEqual(self.render(app, b"/commit/plan-3").code, 200)

    def test_unknownTokenRenders404(self) -> None:
        """
        Committing a token that was never prepared renders 404.
        """
        app = RenderedChangeApp()
        unknown = self.render(app, b"/commit/never-prepared")
        self.assertEqual(unknown.code, 404)
        self.assertIn("never-prepared", unknown.body()["error"])

    def test_asyncFailureRetainsContextAndRetrySucceeds(self) -> None:
        """
        An asynchronous application failure renders a retryable 500 while
        retaining the planning request as context; repeating the commit
        after the failure clears succeeds.
        """
        app = RenderedChangeApp()
        planningRequest = self.render(app, b"/plan-flaky")
        self.assertEqual(planningRequest.code, 200)

        failed = self.render(app, b"/commit/plan-1")
        self.assertEqual(failed.code, 500)
        body = failed.body()
        self.assertTrue(body["retryable"])
        self.assertEqual(body["identifier"], "b")
        # Nothing was published.
        self.assertEqual(app.store.version, 0)
        self.assertIsNone(app.store.get("a"))
        # The planning request was retained through failure and
        # compensation.
        self.assertEqual(app.contexts, [planningRequest])
        self.assertEqual(app.rollbacks, [planningRequest])

        app.failNext = False
        retried = self.render(app, b"/commit/plan-1")
        self.assertEqual(retried.code, 200)
        self.assertEqual(retried.body()["version"], 1)
        self.assertEqual(app.store.get("a"), "A")
        self.assertEqual(app.store.get("b"), "B")
        # The retry ran against the same retained context.
        self.assertEqual(app.contexts, [planningRequest, planningRequest])
