"""
Tests for budgeted execution control of resource rendering.
"""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any, cast
from unittest.mock import Mock

from twisted.internet.defer import Deferred, fail
from twisted.internet.error import ConnectionLost
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from twisted.web.http_headers import Headers
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.test.test_web import DummyChannel

from .. import Klein
from .._budgets import (
    BudgetCancelled,
    BudgetedRender,
    BudgetError,
    BudgetExceeded,
    BudgetInitializationError,
    BudgetShutdown,
    CountBudgetExceeded,
    MemoryBudgetExceeded,
    RenderBudget,
    StageFailed,
    TimeBudgetExceeded,
)
from .._interfaces import IKleinRequest
from .._resource import KleinResource


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


class RenderBudgetTests(SynchronousTestCase):
    def test_unlimitedByDefault(self) -> None:
        budget = RenderBudget()
        budget.start()
        budget.accept()
        budget.accept()
        budget.charge(10**9)
        budget.check()  # must not raise

    def test_startIsIdempotent(self) -> None:
        clock = FakeClock()
        budget = RenderBudget(time_limit=5.0, clock=clock)
        budget.start()
        clock.advance(4.0)
        budget.start()  # does not reset
        self.assertEqual(budget.elapsed, 4.0)

    def test_timeLimit(self) -> None:
        clock = FakeClock()
        budget = RenderBudget(time_limit=2.5, clock=clock)
        budget.start()
        clock.advance(2.5)
        error = self.assertRaises(TimeBudgetExceeded, budget.check)
        self.assertEqual(error.dimension, "time")
        self.assertEqual(error.limit, 2.5)
        self.assertEqual(error.used, 2.5)

    def test_timeLimitStillAvailable(self) -> None:
        clock = FakeClock()
        budget = RenderBudget(time_limit=2.5, clock=clock)
        budget.start()
        clock.advance(2.0)
        budget.check()

    def test_countLimit(self) -> None:
        budget = RenderBudget(count_limit=2)
        budget.accept()
        budget.accept()
        error = self.assertRaises(CountBudgetExceeded, budget.check)
        self.assertEqual(error.dimension, "count")
        self.assertEqual((error.limit, error.used), (2, 2))

    def test_countLimitAllowsExactly(self) -> None:
        budget = RenderBudget(count_limit=1)
        budget.accept()
        self.assertEqual(budget.used_count, 1)

    def test_memoryLimitCharged(self) -> None:
        budget = RenderBudget(memory_limit=10)
        budget.charge(11)
        error = self.assertRaises(MemoryBudgetExceeded, budget.check)
        self.assertEqual(error.dimension, "memory")
        self.assertEqual((error.limit, error.used), (10, 11))

    def test_memoryLimitIncludesExternalUsage(self) -> None:
        budget = RenderBudget(memory_limit=10, external_memory=lambda: 11)
        self.assertRaises(MemoryBudgetExceeded, budget.check)

    def test_memoryReleasedIsUncharged(self) -> None:
        budget = RenderBudget(memory_limit=10)
        budget.charge(10)
        budget.check()
        budget.charge(-10)
        budget.check()

    def test_invalidLimits(self) -> None:
        self.assertRaises(ValueError, RenderBudget, time_limit=0)
        self.assertRaises(ValueError, RenderBudget, count_limit=0)
        self.assertRaises(ValueError, RenderBudget, memory_limit=-1)

    def test_dimensionsCheckedInFixedOrder(self) -> None:
        clock = FakeClock()
        budget = RenderBudget(
            time_limit=1.0,
            count_limit=1,
            memory_limit=1,
            clock=clock,
        )
        budget.start()
        budget.accept()
        clock.advance(2.0)
        budget.charge(2)
        # Time wins over count and memory.
        self.assertRaises(TimeBudgetExceeded, budget.check)


class BudgetedRenderTests(SynchronousTestCase):
    def release(self, events: list[str], name: str) -> Callable[[Any], None]:
        def _release(resource: object) -> None:
            events.append(name)

        return _release

    def test_successRunsOnceAndCaches(self) -> None:
        calls = []

        def work() -> int:
            calls.append("call")
            return 7

        controller = BudgetedRender(RenderBudget(count_limit=10))
        first = controller.run("only", work)
        self.assertEqual(self.successResultOf(first), 7)

        # The completed stage never executes again and consumes no more
        # budget.
        second = controller.run("only", lambda: self.fail("ran twice"))
        self.assertEqual(self.successResultOf(second), 7)
        self.assertEqual(calls, ["call"])
        self.assertEqual(controller.budget.used_count, 1)

    def test_inFlightStageIsJoinedNotReRun(self) -> None:
        pending: Deferred[str] = Deferred()

        def work() -> Deferred[str]:
            return pending

        controller = BudgetedRender(RenderBudget())
        first = controller.run("once", work)
        second = controller.run("once", lambda: self.fail("ran twice"))
        self.assertIs(first, second)
        self.assertNoResult(first)

        pending.callback("done")
        self.assertEqual(self.successResultOf(second), "done")

    def test_preStageGateStopsBeforeWork(self) -> None:
        ran: list[str] = []
        events: list[str] = []
        controller = BudgetedRender(RenderBudget(count_limit=1))
        controller.track("first", self.release(events, "first"))

        def work_a() -> str:
            ran.append("a")
            return "a"

        def work_b() -> str:
            ran.append("b")
            return "b"

        self.assertEqual(
            self.successResultOf(controller.run("a", work_a)),
            "a",
        )
        failure = self.failureResultOf(controller.run("b", work_b))
        failure.trap(CountBudgetExceeded)

        self.assertEqual(ran, ["a"])
        self.assertTrue(controller.closed)
        # Temporaries of the already finished render are reclaimed.
        self.assertEqual(events, ["first"])

    def test_timeGate(self) -> None:
        clock = FakeClock()

        def factory() -> BudgetedRender:
            return BudgetedRender(RenderBudget(time_limit=1.0, clock=clock))

        controller = factory()
        self.successResultOf(controller.run("a", lambda: "a"))
        clock.advance(2.0)
        failure = self.failureResultOf(controller.run("b", lambda: "b"))
        failure.trap(TimeBudgetExceeded)
        self.assertTrue(controller.closed)

    def test_memoryGateAfterInitialization(self) -> None:
        events: list[str] = []
        controller = BudgetedRender(RenderBudget(memory_limit=50))

        def initialize() -> None:
            controller.track("buffer", self.release(events, "buffer"), size=60)

        failure = self.failureResultOf(
            controller.run(
                "big", lambda: self.fail("work ran"), initialize=initialize
            )
        )
        failure.trap(MemoryBudgetExceeded)
        self.assertTrue(controller.closed)
        self.assertEqual(events, ["buffer"])
        self.assertEqual(controller.budget.charged_memory, 0)

        # The render is shut down, so further stages are shutdown races.
        late = self.failureResultOf(controller.run("late", lambda: None))
        late.trap(BudgetShutdown)

    def test_cancelBeforeStage(self) -> None:
        controller = BudgetedRender(RenderBudget())
        controller.cancel()
        failure = self.failureResultOf(controller.run("x", lambda: "x"))
        failure.trap(BudgetCancelled)

    def test_cancelDuringStageReleasesTemporaries(self) -> None:
        pending: Deferred[str] = Deferred()
        events: list[str] = []

        controller = BudgetedRender(RenderBudget(), request="the-request")
        deferred = controller.run("slow", lambda: pending)
        controller.track("temp", self.release(events, "temp"), size=12)
        self.assertNoResult(deferred)

        controller.cancel()
        failure = self.failureResultOf(deferred)
        failure.trap(BudgetCancelled)

        self.assertTrue(controller.cancelled)
        self.assertTrue(controller.closed)
        self.assertEqual(events, ["temp"])
        self.assertEqual(controller.budget.charged_memory, 0)
        self.assertEqual(controller.request, "the-request")

    def test_cancelIsIdempotent(self) -> None:
        controller = BudgetedRender(RenderBudget())
        controller.cancel()
        controller.cancel()
        controller.close()
        self.assertTrue(controller.cancelled)

    def test_initializationFailureIsDistinct(self) -> None:
        events: list[str] = []
        controller = BudgetedRender(RenderBudget())

        def initialize() -> None:
            controller.track("partial", self.release(events, "partial"), size=3)
            raise RuntimeError("setup exploded")

        failure = self.failureResultOf(
            controller.run(
                "broken",
                lambda: self.fail("work ran"),
                initialize=initialize,
            )
        )
        self.assertIs(
            failure.trap(BudgetInitializationError),
            BudgetInitializationError,
        )
        error = failure.value
        self.assertEqual(error.stage, "broken")
        self.assertIsInstance(error.cause, RuntimeError)

        self.assertTrue(controller.closed)
        self.assertEqual(events, ["partial"])

        late = self.failureResultOf(controller.run("late", lambda: None))
        late.trap(BudgetShutdown)

    def test_runAfterCloseIsShutdown(self) -> None:
        controller = BudgetedRender(RenderBudget())
        controller.close()
        failure = self.failureResultOf(controller.run("late", lambda: None))
        failure.trap(BudgetShutdown)

    def test_lateSuccessAfterCancellationIsShutdownRace(self) -> None:
        # The work finishes successfully in the same instant that
        # cancellation reaches it: its canceller delivers the result before
        # the CancelledError fallback fires.  That is a shutdown race, not a
        # cancellation.
        def canceller(deferred: Deferred[str]) -> None:
            deferred.callback("late-success")

        pending: Deferred[str] = Deferred(canceller)
        controller = BudgetedRender(RenderBudget())
        deferred = controller.run("late", lambda: pending)

        controller.cancel()
        failure = self.failureResultOf(deferred)
        failure.trap(BudgetShutdown)
        self.assertEqual(failure.value.stage, "late")

    def test_lateFailureAfterCancellationIsShutdownRace(self) -> None:
        def canceller(deferred: Deferred[str]) -> None:
            deferred.errback(Failure(RuntimeError("late failure")))

        pending: Deferred[str] = Deferred(canceller)
        controller = BudgetedRender(RenderBudget())
        deferred = controller.run("late", lambda: pending)

        controller.cancel()
        failure = self.failureResultOf(deferred)
        failure.trap(BudgetShutdown)
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)

    def test_raceAfterTerminalResultIsRecorded(self) -> None:
        controller = BudgetedRender(RenderBudget())
        deferred = controller.run("late", lambda: Deferred(lambda _: None))
        controller.cancel()
        outer = deferred  # already delivered BudgetCancelled
        self.failureResultOf(outer).trap(BudgetCancelled)
        stage = controller._stages["late"]
        self.assertIsNone(controller._race(stage, outer, False, "x"))
        self.assertEqual(controller.races, [("late", "completed")])

    def test_fourTerminalReasonsAreDistinct(self) -> None:
        reasons: list[type[Exception]] = []

        cancelled = BudgetedRender(RenderBudget())
        cancelled.cancel()
        reasons.append(
            type(self.failureResultOf(cancelled.run("a", lambda: None)).value)
        )

        exceeded = BudgetedRender(RenderBudget(count_limit=1))
        self.successResultOf(exceeded.run("a", lambda: None))
        reasons.append(
            type(self.failureResultOf(exceeded.run("b", lambda: None)).value)
        )

        init_fail = BudgetedRender(RenderBudget())

        def boom() -> None:
            raise RuntimeError()

        reasons.append(
            type(
                self.failureResultOf(
                    init_fail.run("a", lambda: None, initialize=boom)
                ).value
            )
        )

        shutdown = BudgetedRender(RenderBudget())
        shutdown.close()
        reasons.append(
            type(self.failureResultOf(shutdown.run("a", lambda: None)).value)
        )

        self.assertEqual(len(set(reasons)), 4)
        self.assertEqual(
            set(reasons),
            {
                BudgetCancelled,
                CountBudgetExceeded,
                BudgetInitializationError,
                BudgetShutdown,
            },
        )
        self.assertTrue(issubclass(CountBudgetExceeded, BudgetExceeded))

    def test_asyncFailureRetainsContextAndIsRetryable(self) -> None:
        request = object()
        attempts = []

        def flaky() -> str | Deferred[str]:
            attempts.append("try")
            if len(attempts) == 1:
                return fail(RuntimeError("boom"))
            return "recovered"

        controller = BudgetedRender(
            RenderBudget(count_limit=5), request=request
        )
        failure = self.failureResultOf(controller.run("work", flaky))
        stage_failure = failure.trap(StageFailed)
        self.assertIs(stage_failure, StageFailed)
        error = cast(StageFailed, failure.value)
        self.assertEqual(error.stage, "work")
        self.assertIs(error.request, request)
        self.assertEqual(error.attempt, 1)
        self.assertIsInstance(error.failure.value, RuntimeError)
        self.assertEqual(str(error.failure.value), "boom")

        # The failed attempt released its claim; the stage is retryable.
        self.assertTrue(controller._stages["work"].failed)
        retry_deferred = controller.retry("work")
        self.assertEqual(self.successResultOf(retry_deferred), "recovered")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(controller.budget.used_count, 2)

        # After success the stage is complete and cannot run again.
        again = controller.run("work", lambda: self.fail("again"))
        self.assertEqual(self.successResultOf(again), "recovered")
        self.assertEqual(controller.budget.used_count, 2)

    def test_retryWithoutNameRetriesLastFailed(self) -> None:
        attempts = []

        def flaky() -> str:
            attempts.append("try")
            if len(attempts) == 1:
                raise RuntimeError("first")
            return "ok"

        controller = BudgetedRender(RenderBudget(count_limit=3))
        failure = self.failureResultOf(controller.run("w", flaky))
        failure.trap(StageFailed)
        self.assertEqual(self.successResultOf(controller.retry()), "ok")

    def test_retryIsBoundedByCount(self) -> None:
        controller = BudgetedRender(RenderBudget(count_limit=1), request="req")
        failure = self.failureResultOf(
            controller.run("w", lambda: fail(RuntimeError()))
        )
        failure.trap(StageFailed)

        bounded = self.failureResultOf(controller.retry("w"))
        bounded.trap(CountBudgetExceeded)
        self.assertTrue(controller.closed)

    def test_runAfterFailureImpliesRetry(self) -> None:
        attempts = []

        def work() -> str:
            attempts.append("try")
            if len(attempts) == 1:
                raise RuntimeError()
            return "ok"

        controller = BudgetedRender(RenderBudget(count_limit=3))
        first = self.failureResultOf(controller.run("w", work))
        first.trap(StageFailed)
        second = controller.run("w")
        self.assertEqual(self.successResultOf(second), "ok")

    def test_retryMistakes(self) -> None:
        controller = BudgetedRender(RenderBudget())
        self.failureResultOf(controller.retry("nope")).trap(BudgetError)

        self.successResultOf(controller.run("ok", lambda: 1))
        self.assertEqual(self.successResultOf(controller.retry("ok")), 1)

        pending: Deferred[None] = Deferred()
        controller2 = BudgetedRender(RenderBudget())
        in_flight = controller2.run("pending", lambda: pending)
        # An in-flight stage is joined, not retried.
        self.assertIs(controller2.retry("pending"), in_flight)
        pending.callback(None)

    def test_failureReleasesOnlyThatStage(self) -> None:
        events: list[str] = []

        controller = BudgetedRender(RenderBudget(count_limit=5))
        controller.track("global", self.release(events, "global"))
        self.successResultOf(
            controller.run(
                "ok",
                lambda: controller.track(
                    "ok-resource",
                    self.release(events, "ok-resource"),
                ),
            )
        )
        self.failureResultOf(
            controller.run(
                "bad",
                lambda: fail(RuntimeError()),
            )
        ).trap(StageFailed)

        # The failed attempt's temporaries are released immediately;
        # successful and controller level temporaries survive.
        self.assertEqual(events, [])
        controller.close()
        self.assertEqual(sorted(events), ["global", "ok-resource"])

    def test_releaseErrorsAreLoggedNotRaised(self) -> None:
        def bad_release(resource: object) -> None:
            raise RuntimeError("cannot release")

        controller = BudgetedRender(RenderBudget())
        controller.track("thing", bad_release, size=4)
        controller.close()  # must not raise
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)
        self.assertEqual(controller.budget.charged_memory, 0)

    def test_openWrapsInitializationFailure(self) -> None:
        with self.assertRaises(BudgetInitializationError):
            BudgetedRender.open(cast(RenderBudget, object()))

    def test_brokenClockIsInitializationFailure(self) -> None:
        def broken_clock() -> float:
            raise RuntimeError("no clock")

        with self.assertRaises(BudgetInitializationError):
            BudgetedRender(RenderBudget(clock=broken_clock))

    def test_trackAfterShutdownRejected(self) -> None:
        controller = BudgetedRender(RenderBudget())
        controller.close()
        self.assertRaises(
            BudgetShutdown,
            controller.track,
            "x",
            lambda _: None,
        )

    def test_repr(self) -> None:
        controller = BudgetedRender(RenderBudget())
        self.assertIn("running", repr(controller))
        controller.close()
        self.assertIn("closed", repr(controller))


class MockRequest(Request):
    finished: bool
    startedWriting: bool
    processingFailed: Mock
    setResponseCode: Mock
    setHeader: Mock

    def __init__(self, path: bytes, method: bytes = b"GET") -> None:
        super().__init__(DummyChannel(), False)

        self.site = cast(Site, Mock(spec=Site))
        self.gotLength(0)
        self.content = BytesIO()
        self.args = {}
        self.selfHeaders = Headers({})
        self.setHost(b"localhost", 8080, False)
        self.uri = path
        self.prepath = []
        self.postpath = path.split(b"/")[1:]
        self.method = method
        self.clientproto = b"HTTP/1.1"

        self.setHeader = Mock(wraps=self.setHeader)
        self.setResponseCode = Mock(wraps=self.setResponseCode)

        self._written = BytesIO()
        self.finishCount = 0
        self.writeCount = 0

        self.processingFailed = Mock(wraps=self.processingFailed)

    def finish(self) -> None:
        self.finishCount += 1
        if not self.startedWriting:
            self.write(b"")
        if not self.finished:
            self.finished = True
            self._cleanup()

    def write(self, data: bytes) -> None:
        self.writeCount += 1
        self.startedWriting = True
        if not self.finished:
            self._written.write(data)
        else:
            raise RuntimeError("write after finish")

    def getWrittenData(self) -> bytes:
        return self._written.getvalue()


def _render(resource: KleinResource, request: MockRequest) -> Any:
    result = resource.render(request)
    assert result is NOT_DONE_YET
    return result


class KleinResourceBudgetTests(SynchronousTestCase):
    def test_unbudgetedUnchanged(self) -> None:
        app = Klein()

        @app.route("/")
        def root(request: object) -> bytes:
            return b"unbudgeted"

        request = MockRequest(b"/")
        resource = KleinResource(app)
        _render(resource, request)
        self.assertEqual(request.getWrittenData(), b"unbudgeted")
        self.assertTrue(request.finished)
        self.assertIsNone(IKleinRequest(request).execution_budget)

    def test_budgetedHappyPath(self) -> None:
        observed: list[BudgetedRender] = []
        app = Klein()

        @app.route("/")
        def root(request: object) -> bytes:
            observed.append(IKleinRequest(request).execution_budget)
            return b"budgeted"

        request = MockRequest(b"/")
        resource = KleinResource(
            app,
            execution_budget=lambda: RenderBudget(count_limit=3),
        )
        _render(resource, request)
        self.assertEqual(request.getWrittenData(), b"budgeted")
        self.assertEqual(request.code, 200)
        self.assertEqual(len(observed), 1)
        controller = observed[0]
        self.assertIs(controller.request, request)
        self.assertTrue(controller.closed)
        # route, render, write.
        self.assertEqual(controller.budget.used_count, 3)

    def test_countLimitExceededReturns503(self) -> None:
        app = Klein()

        @app.route("/")
        def root(request: object) -> bytes:
            return b"ok"

        request = MockRequest(b"/")
        resource = KleinResource(
            app,
            execution_budget=lambda: RenderBudget(count_limit=1),
        )
        _render(resource, request)

        self.assertEqual(request.setResponseCode.call_args.args[0], 503)
        self.assertEqual(request.getWrittenData(), b"Render budget exceeded.\n")
        self.assertTrue(request.finished)

    def test_timeLimitExceededReturns503(self) -> None:
        ticks = {"n": 0}

        def clock() -> float:
            ticks["n"] += 1
            return 0.0 if ticks["n"] == 1 else 100.0

        called: list[bool] = []
        app = Klein()

        @app.route("/")
        def root(request: object) -> bytes:
            called.append(True)
            return b"ok"

        request = MockRequest(b"/")
        resource = KleinResource(
            app,
            execution_budget=lambda: RenderBudget(time_limit=1.0, clock=clock),
        )
        _render(resource, request)

        self.assertEqual(request.setResponseCode.call_args.args[0], 503)
        self.assertEqual(called, [])

    def test_errorHandlerRunsAsGatedStage(self) -> None:
        app = Klein()

        @app.route("/boom")
        def boom(request: object) -> None:
            raise RuntimeError("boom")

        @app.handle_errors(RuntimeError)
        def on_boom(request: object, failure: Failure) -> bytes:
            return b"recovered"

        request = MockRequest(b"/boom")
        resource = KleinResource(
            app,
            execution_budget=lambda: RenderBudget(count_limit=3),
        )
        _render(resource, request)

        self.assertEqual(request.getWrittenData(), b"recovered")
        controller = IKleinRequest(request).execution_budget
        self.assertEqual(controller.budget.used_count, 3)

    def test_errorHandlerExceedingBudgetReturns503(self) -> None:
        app = Klein()

        @app.route("/boom")
        def boom(request: object) -> None:
            raise RuntimeError("boom")

        @app.handle_errors(RuntimeError)
        def on_boom(request: object, failure: Failure) -> bytes:
            return b"recovered"

        request = MockRequest(b"/boom")
        resource = KleinResource(
            app,
            execution_budget=lambda: RenderBudget(count_limit=2),
        )
        _render(resource, request)

        self.assertEqual(request.setResponseCode.call_args.args[0], 503)

    def test_cancellationStopsRenderAndReclaims(self) -> None:
        cancelled: list[bool] = []

        def canceller(deferred: Deferred[None]) -> None:
            cancelled.append(True)

        pending: Deferred[None] = Deferred(canceller)
        released: list[str] = []
        app = Klein()

        @app.route("/slow")
        def slow(request: object) -> Deferred[None]:
            controller = IKleinRequest(request).execution_budget
            controller.track("temp", lambda _: released.append("temp"), size=7)
            return pending

        request = MockRequest(b"/slow")
        resource = KleinResource(app, execution_budget=lambda: RenderBudget())
        _render(resource, request)
        self.assertNoResult(pending)

        request.connectionLost(Failure(ConnectionLost("gone")))

        self.assertEqual(cancelled, [True])
        self.assertEqual(request.finishCount, 0)
        self.assertEqual(request.getWrittenData(), b"")
        request.processingFailed.assert_not_called()
        self.assertEqual(released, ["temp"])
        controller = IKleinRequest(request).execution_budget
        self.assertTrue(controller.cancelled)
        self.flushLoggedErrors(ConnectionLost)

    def test_budgetFactoryFailureIsInitializationFailure(self) -> None:
        app = Klein()

        @app.route("/")
        def root(request: object) -> bytes:
            return b"ok"

        def broken_factory() -> RenderBudget:
            raise RuntimeError("no budget for you")

        request = MockRequest(b"/")
        resource = KleinResource(app, execution_budget=broken_factory)
        _render(resource, request)

        request.processingFailed.assert_called_once()
        self.assertEqual(request.setResponseCode.call_args.args[0], 500)
        self.flushLoggedErrors(BudgetInitializationError)

    def test_asyncFailureRetainsRequestContext(self) -> None:
        app = Klein()
        captured: list[BudgetedRender] = []

        @app.route("/flaky")
        def flaky(request: object) -> None:
            captured.append(IKleinRequest(request).execution_budget)
            raise RuntimeError("first try fails")

        request = MockRequest(b"/flaky")
        resource = KleinResource(
            app,
            execution_budget=lambda: RenderBudget(count_limit=10),
        )
        _render(resource, request)

        # The default failure handling renders a 500 for the unhandled
        # RuntimeError; the controller still knows which request it was,
        # and the failed stage is marked retryable.
        self.assertEqual(request.code, 500)
        controller = captured[0]
        self.assertIs(controller.request, request)
        self.assertTrue(controller._stages["route"].failed)
        self.flushLoggedErrors(RuntimeError)
