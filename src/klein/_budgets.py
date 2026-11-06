# -*- test-case-name: klein.test.test_budgets -*-
"""
Budgeted execution control for L{klein.KleinResource} rendering.

A resource render is broken into named I{stages} (route lookup, endpoint
execution, rendering, error handling, writing the response, ...).  A
L{RenderBudget} expresses caller supplied limits on how long the render may
take (wall-clock time), how many stages may be accepted (retries included),
and how many bytes of temporary resource it may hold.  A L{BudgetedRender}
runs those stages, checking the budget I{before} each stage is accepted.

A stage goes through the following lifecycle::

    accepted -> completed
            \\-> failed -> (retry -> accepted ...)

An accepted stage executes at most once and completes at most once:
re-entering a running stage returns its in-flight L{Deferred}, and re-running
a completed stage returns its cached result without consuming any more
budget.

Terminal failures each have a distinct reason:

  - L{BudgetCancelled} when the render is cancelled (client disconnect),
  - L{BudgetExceeded} (and its time/count/memory subclasses) when a limit
    is reached before a stage can be accepted,
  - L{BudgetInitializationError} when a stage cannot initialize,
  - L{BudgetShutdown} when work is requested, or settles, after the
    controller has already been shut down.

Every terminal failure releases all tracked temporary resources.  A
non-terminal asynchronous stage failure is wrapped in L{StageFailed}, which
retains the request context and may be L{BudgetedRender.retry retried}.
"""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic
from typing import Any

from attr import Factory, attrib, attrs

from twisted.internet.defer import (
    CancelledError,
    Deferred,
    fail,
    maybeDeferred,
    succeed,
)
from twisted.python import log
from twisted.python.failure import Failure

__all__ = ()


class BudgetError(Exception):
    """
    Base class for all errors raised by budgeted render execution.
    """


class BudgetCancelled(BudgetError):
    """
    The budgeted render was cancelled before it could finish.
    """


class BudgetExceeded(BudgetError):
    """
    A stage could not be accepted because a budget limit was reached.

    @cvar dimension: Which limit was reached, C{"time"}, C{"count"} or
        C{"memory"}.
    """

    dimension = ""

    def __init__(self, *, dimension: str, limit: float, used: float) -> None:
        self.dimension = dimension
        self.limit = limit
        self.used = used
        super().__init__(
            f"{dimension} budget exhausted: used {used!r}, limit {limit!r}"
        )


class TimeBudgetExceeded(BudgetExceeded):
    """
    The wall-clock time limit was reached before a stage was accepted.
    """

    def __init__(self, *, limit: float, used: float) -> None:
        super().__init__(dimension="time", limit=limit, used=used)


class CountBudgetExceeded(BudgetExceeded):
    """
    The stage-acceptance count limit was reached (retries count).
    """

    def __init__(self, *, limit: int, used: int) -> None:
        super().__init__(dimension="count", limit=limit, used=used)


class MemoryBudgetExceeded(BudgetExceeded):
    """
    The temporary-memory limit was reached.
    """

    def __init__(self, *, limit: int, used: int) -> None:
        super().__init__(dimension="memory", limit=limit, used=used)


class BudgetInitializationError(BudgetError):
    """
    The controller, or one of its stages, could not be initialized.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(message)


class BudgetShutdown(BudgetError):
    """
    Work was requested, or settled, after the budget controller had been
    shut down.
    """

    def __init__(self, message: str, *, stage: str | None = None) -> None:
        self.stage = stage
        super().__init__(message)


class StageFailed(BudgetError):
    """
    An asynchronous stage failed.

    This is non-terminal: the originating request context is retained and
    the stage may be retried via L{BudgetedRender.retry}.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        request: object,
        failure: Failure,
        attempt: int,
    ) -> None:
        self.stage = stage
        self.request = request
        self.failure = failure
        self.attempt = attempt
        super().__init__(message)


def _noExternalMemory() -> int:
    return 0


@attrs(eq=False)
class RenderBudget:
    """
    Caller supplied execution limits for a single render.

    Limits are checked immediately before each stage is accepted.  Any
    combination of limits may be given; unspecified limits are unlimited.

    @param time_limit: Maximum wall-clock duration, in seconds, measured
        with C{clock} from the controller's start.
    @param count_limit: Maximum number of stage acceptances; retries count.
    @param memory_limit: Maximum bytes of temporary memory.
    @param clock: Zero-argument callable returning monotonic time.
    @param external_memory: Zero-argument callable returning bytes in use
        that are not accounted for via L{BudgetedRender.track}, for example
        a C{tracemalloc} or RSS measurement.
    """

    time_limit: float | None = attrib(default=None)
    count_limit: int | None = attrib(default=None)
    memory_limit: int | None = attrib(default=None)
    clock: Callable[[], float] = attrib(default=monotonic)
    external_memory: Callable[[], int] = attrib(default=_noExternalMemory)

    _start: float | None = attrib(init=False, default=None)
    used_count: int = attrib(init=False, default=0)
    charged_memory: int = attrib(init=False, default=0)

    def __attrs_post_init__(self) -> None:
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("time_limit must be greater than zero")
        if self.count_limit is not None and self.count_limit <= 0:
            raise ValueError("count_limit must be greater than zero")
        if self.memory_limit is not None and self.memory_limit < 0:
            raise ValueError("memory_limit must not be negative")

    def start(self) -> None:
        """
        Record the starting time.  Idempotent.
        """
        if self._start is None:
            self._start = self.clock()

    @property
    def elapsed(self) -> float:
        """
        Seconds since L{start}, or C{0.0} if not started.
        """
        if self._start is None:
            return 0.0
        return self.clock() - self._start

    def used_memory(self) -> int:
        """
        Bytes currently accounted to this render.
        """
        return self.charged_memory + self.external_memory()

    def check(self) -> None:
        """
        Raise a L{BudgetExceeded} subclass if any limit has been reached.

        Dimensions are checked in a fixed order - time, then count, then
        memory - so the reason is unambiguous when several apply.
        """
        if self.time_limit is not None and self.elapsed >= self.time_limit:
            raise TimeBudgetExceeded(limit=self.time_limit, used=self.elapsed)
        if self.count_limit is not None and self.used_count >= self.count_limit:
            raise CountBudgetExceeded(
                limit=self.count_limit, used=self.used_count
            )
        used_memory = self.used_memory()
        if self.memory_limit is not None and used_memory > self.memory_limit:
            raise MemoryBudgetExceeded(
                limit=self.memory_limit, used=used_memory
            )

    def accept(self) -> None:
        """
        Accept one more stage: check all limits and consume one count.
        """
        self.check()
        self.used_count += 1

    def charge(self, size: int) -> None:
        """
        Account for C{size} bytes.  Negative values release prior bytes.
        """
        self.charged_memory += size


@attrs(eq=False)
class _Temporary:
    """
    A temporary resource that must be released on failure or shutdown.
    """

    resource: object = attrib()
    release: Callable[[object], None] = attrib()
    size: int = attrib()
    released: bool = attrib(init=False, default=False)


@attrs(eq=False)
class _Stage:
    """
    Bookkeeping for one named render stage.
    """

    name: str = attrib()
    work: Callable[[], Any] | None = attrib(default=None)
    initializer: Callable[[], None] | None = attrib(default=None)
    accepted: bool = attrib(init=False, default=False)
    completed: bool = attrib(init=False, default=False)
    failed: bool = attrib(init=False, default=False)
    attempts: int = attrib(init=False, default=0)
    result: object = attrib(init=False, default=None)
    waiting: Deferred[Any] | None = attrib(init=False, default=None)
    temporaries: list[_Temporary] = attrib(init=False, default=Factory(list))


class BudgetedRender:
    """
    Execute the stages of one resource render under a L{RenderBudget}.

    @param budget: The limits for this render.
    @param request: The request being rendered, retained on failures.
    """

    def __init__(
        self, budget: RenderBudget, request: object | None = None
    ) -> None:
        if not isinstance(budget, RenderBudget):
            raise BudgetInitializationError(
                f"expected a RenderBudget, got {budget!r}"
            )
        self._budget = budget
        self._request = request
        self._stages: dict[str, _Stage] = {}
        self._temporaries: list[_Temporary] = []
        self._active: _Stage | None = None
        self._cancelled = False
        self._closed = False
        self._terminal: BudgetError | None = None
        self._lastFailed: str | None = None

        #: Late stage settlements observed after shutdown, as
        #: ``(stage name, "completed" | "failed")`` tuples.
        self.races: list[tuple[str, str]] = []

        try:
            budget.start()
        except Exception as e:
            raise BudgetInitializationError(
                "render budget could not be started", cause=e
            ) from e

    @classmethod
    def open(
        cls, budget: RenderBudget, request: object | None = None
    ) -> BudgetedRender:
        """
        Construct a controller, translating construction failures to
        L{BudgetInitializationError}.
        """
        try:
            return cls(budget, request)
        except BudgetInitializationError:
            raise
        except Exception as e:
            raise BudgetInitializationError(
                "budgeted render could not be initialized", cause=e
            ) from e

    @property
    def budget(self) -> RenderBudget:
        return self._budget

    @property
    def request(self) -> object | None:
        return self._request

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def closed(self) -> bool:
        return self._closed

    def watch(self, request: object) -> None:
        """
        Cancel this controller when C{request}'s connection is lost.
        """
        request.notifyFinish().addErrback(  # type: ignore[attr-defined]
            lambda _: self.cancel()
        )

    def track(
        self,
        resource: object,
        release: Callable[[object], None],
        *,
        size: int = 0,
        stage: str | None = None,
    ) -> object:
        """
        Register a temporary resource.

        @param resource: The resource to release later.
        @param release: One-argument callable invoked with C{resource} when
            the owning stage fails or the render shuts down.
        @param size: Bytes of memory the resource occupies, charged to the
            memory budget until it is released.
        @param stage: Owning stage name.  Defaults to the stage currently
            accepting; a resource without an owning stage lives until the
            whole render shuts down.

        @return: C{resource}, for convenient inline registration.
        """
        if self._closed:
            raise BudgetShutdown(
                "cannot track a temporary resource after shutdown",
                stage=stage,
            )
        temporary = _Temporary(resource=resource, release=release, size=size)
        owner = self._owner(stage)
        if owner is not None:
            owner.temporaries.append(temporary)
        else:
            self._temporaries.append(temporary)
        self._budget.charge(size)
        return resource

    def run(
        self,
        name: str,
        work: Callable[[], Any] | None = None,
        *,
        initialize: Callable[[], None] | None = None,
    ) -> Deferred[Any]:
        """
        Run the named stage under the budget.

        @param name: Unique stage name.
        @param work: Zero-argument callable performing the stage; may return
            a L{Deferred}.  Required the first time a stage is run and may
            be omitted afterwards.
        @param initialize: Optional zero-argument callable setting up the
            stage (for example binding the URL mapper).  Failures become
            L{BudgetInitializationError}.  It runs again on every retry,
            after the previous attempt's temporaries were released.
        """
        stage = self._stages.get(name)
        if stage is None:
            if work is None:
                return fail(
                    BudgetInitializationError(
                        f"unknown stage {name!r}", stage=name
                    )
                )
            stage = _Stage(name=name, work=work, initializer=initialize)
            self._stages[name] = stage
        elif stage.completed:
            # An accepted stage completes exactly once.
            return succeed(stage.result)
        elif stage.waiting is not None:
            # The stage is already in flight; join its result instead of
            # executing it a second time.
            return stage.waiting
        else:
            if work is not None:
                stage.work = work
                stage.initializer = initialize
            if stage.failed:
                return self._retry(stage)
        return self._accept(stage)

    def retry(self, name: str | None = None) -> Deferred[Any]:
        """
        Retry a stage that previously failed asynchronously.

        The request context is retained and the retry consumes a fresh
        stage acceptance (and is therefore bounded by the count limit).
        """
        if name is None:
            name = self._lastFailed
        if name is None or name not in self._stages:
            return fail(BudgetError("no failed stage is available to retry"))
        stage = self._stages[name]
        if stage.completed:
            return succeed(stage.result)
        if stage.waiting is not None:
            return stage.waiting
        if not stage.failed:
            return fail(BudgetError(f"stage {name!r} has not failed"))
        return self._retry(stage)

    def cancel(self) -> None:
        """
        Cancel the render: stop in-flight stages and release temporaries.
        Idempotent.
        """
        if self._closed or self._cancelled:
            return
        self._cancelled = True
        self._terminate(BudgetCancelled("budgeted rendering was cancelled"))

    def close(self) -> None:
        """
        Shut the controller down normally and release everything.
        Idempotent.
        """
        self._terminate(None)

    def _owner(self, stage: str | None) -> _Stage | None:
        if stage is not None:
            return self._stages.get(stage)
        if self._active is not None:
            return self._active
        return None

    def _gate(self) -> None:
        if self._cancelled:
            raise BudgetCancelled("budgeted rendering was cancelled")
        if self._closed:
            raise BudgetShutdown("budgeted render has been shut down")
        self._budget.check()

    def _retry(self, stage: _Stage) -> Deferred[Any]:
        self._lastFailed = None
        return self._accept(stage)

    def _accept(self, stage: _Stage) -> Deferred[Any]:
        try:
            self._gate()
        except BudgetError as e:
            if not self._closed:
                self._terminate(e)
            return fail(e)

        stage.accepted = True
        stage.failed = False
        stage.attempts += 1
        self._budget.accept()

        if stage.initializer is not None:
            self._active = stage
            try:
                stage.initializer()
            except Exception as e:
                self._active = None
                error = BudgetInitializationError(
                    f"stage {stage.name!r} could not be initialized",
                    stage=stage.name,
                    cause=e,
                )
                self._terminate(error)
                return fail(error)
            # Initialization may have charged memory beyond the limit.
            try:
                self._budget.check()
            except BudgetExceeded as e:
                self._active = None
                self._terminate(e)
                return fail(e)

        self._active = stage
        work = stage.work
        assert work is not None
        inner = maybeDeferred(work)

        outer: Deferred[Any] = Deferred(lambda _: inner.cancel())
        stage.waiting = outer
        inner.addBoth(self._settled, stage, outer)
        inner.addBoth(self._propagate, outer)
        outer.addErrback(self._translateCancellation)
        return outer

    def _translateCancellation(self, result: Failure) -> Failure:
        if result.check(CancelledError):
            reason: BudgetError
            if self._terminal is not None:
                reason = self._terminal
            else:
                reason = BudgetCancelled("budgeted rendering was cancelled")
            return Failure(reason)
        return result

    def _settled(
        self,
        result: Any,
        stage: _Stage,
        outer: Deferred[Any],
    ) -> Any:
        stage.waiting = None
        self._active = None

        if isinstance(result, Failure):
            if result.check(CancelledError):
                if self._terminal is None and not self._closed:
                    self._terminate(
                        BudgetCancelled("budgeted rendering was cancelled")
                    )
                return result

            if self._closed:
                return self._race(stage, outer, True, result)

            # Non-terminal asynchronous failure: release the attempt's
            # temporaries but retain the request context for a retry.
            stage.accepted = False
            stage.failed = True
            self._lastFailed = stage.name
            self._releaseStage(stage)
            error = StageFailed(
                f"stage {stage.name!r} failed on attempt "
                f"{stage.attempts}; it may be retried",
                stage=stage.name,
                request=self._request,
                failure=result,
                attempt=stage.attempts,
            )
            return Failure(error)

        if self._closed:
            return self._race(stage, outer, False, result)

        stage.failed = False
        stage.completed = True
        stage.result = result
        self._lastFailed = None
        return result

    def _race(
        self,
        stage: _Stage,
        outer: Deferred[Any],
        failed: bool,
        result: Any,
    ) -> Any:
        detail = "failed" if failed else "completed"
        if failed:
            log.err(
                result,
                f"stage {stage.name!r} failed after the budget shut down",
            )
        if outer.called:
            # The terminal result was already delivered; record the late
            # settlement instead of raising inside a dead callback chain.
            self.races.append((stage.name, detail))
            return None
        return Failure(
            BudgetShutdown(
                f"stage {stage.name!r} {detail} after the budget was "
                "shut down",
                stage=stage.name,
            )
        )

    def _propagate(self, result: Any, outer: Deferred[Any]) -> None:
        if outer.called:
            # Late settlement racing with cancellation; the terminal
            # result was already delivered to the caller.
            return None
        if isinstance(result, Failure):
            outer.errback(result)
        else:
            outer.callback(result)
        return None

    def _terminate(self, reason: BudgetError | None) -> None:
        if self._closed:
            return
        self._closed = True
        self._terminal = reason
        if isinstance(reason, BudgetCancelled):
            self._cancelled = True
        waiting = [
            stage.waiting
            for stage in self._stages.values()
            if stage.waiting is not None
        ]
        for waiting_deferred in waiting:
            waiting_deferred.cancel()
        self._releaseAll()

    def _releaseStage(self, stage: _Stage) -> None:
        temporaries = stage.temporaries[:]
        stage.temporaries.clear()
        for temporary in reversed(temporaries):
            self._release(temporary)

    def _releaseAll(self) -> None:
        pending = list(self._temporaries)
        self._temporaries.clear()
        for stage in self._stages.values():
            pending.extend(stage.temporaries)
            stage.temporaries.clear()
        for temporary in reversed(pending):
            self._release(temporary)

    def _release(self, temporary: _Temporary) -> None:
        if temporary.released:
            return
        temporary.released = True
        try:
            temporary.release(temporary.resource)
        except Exception as e:
            log.err(
                e,
                "error while releasing a temporary render resource",
            )
        finally:
            self._budget.charge(-temporary.size)

    def __repr__(self) -> str:
        state: str
        if self._cancelled:
            state = "cancelled"
        elif self._closed:
            state = "closed"
        else:
            state = "running"
        return f"<BudgetedRender {state} stages={list(self._stages)}>"
