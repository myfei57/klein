# -*- test-case-name: klein.test.test_changes -*-
"""
A two-phase change protocol for resource rendering.

Applying a batch of changes to rendered resources happens in two explicit
phases:

    1. U{prepare <TwoPhaseResourceStore.plan>} validates the requested changes
       against a versioned snapshot of the current resource state and returns a
       L{CandidatePlan}.  Nothing is written in this phase.

    2. U{commit <TwoPhaseResourceStore.commit>} applies a previously prepared
       plan.  Immediately before writing, the snapshot version recorded on the
       plan is compared against the store's current version; if another commit
       moved the state on, the plan is rejected with L{StaleSnapshot} and no
       write occurs.

The commit itself is atomic with respect to the store: changes execute
against a private staging area and are only published after every change has
succeeded, so an empty batch, a duplicate identifier, or a failure partway
through a batch can never leave a half-applied resource collection.  Changes
which performed externally visible work may supply a compensating action via
L{Applied.rollback}; such actions run in reverse order when a later change
fails.

Successful commits are idempotent: committing the same plan more than once
returns the original L{CommitResult} (flagged as a replay) without executing
any change a second time.  Failed commits are not recorded, and the plan -
along with the request context it was prepared with - remains usable as a
retry.  When a (possibly asynchronous) change fails, the resulting
L{ChangeApplicationError} retains the originating context so a retry happens
with the same request information.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias

from attrs import define, evolve, field

from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.web.iweb import IRequest

from ._dihttp import Response
from ._util import eagerDeferredCoroutine

#: Type of the request context carried through plan preparation, application,
#: failure, compensation, and retry.  This is typically an L{IRequest}, but
#: any object (or C{None}) is acceptable.
ChangeContext: TypeAlias = object | None

if TYPE_CHECKING:
    #: Result of a L{Change.execute} callable.  May be synchronous (the new
    #: resource value, an L{Applied}, or L{DELETE}) or asynchronous (a
    #: L{Deferred} resolving to one of those).
    ExecuteResult: TypeAlias = "Deferred[Any] | Applied | Any"

    #: A compensating action receives the same request context the change
    #: executed with and may perform synchronous or asynchronous cleanup.
    Rollback: TypeAlias = Callable[[ChangeContext], Deferred[None] | None]

log = Logger()


class _Delete:
    """
    Sentinel value returned by a L{Change.execute} callable to indicate that
    the targeted resource should be removed by the commit.
    """

    _instance: _Delete | None = None

    def __new__(cls) -> _Delete:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "DELETE"

    def __bool__(self) -> bool:
        return False


#: Return this from a L{Change.execute} callable to delete the resource.
DELETE = _Delete()


class ChangeProtocolError(Exception):
    """
    Base class for all failures raised by the two-phase change protocol.
    """


class EmptyChanges(ChangeProtocolError):
    """
    A plan was requested for an empty collection of changes.

    Preparing an empty plan is rejected rather than represented as a no-op
    candidate, so that callers cannot accidentally commit nothing.
    """

    def __init__(self) -> None:
        super().__init__("cannot prepare a plan containing no changes")


class DuplicateIdentifier(ChangeProtocolError):
    """
    More than one L{Change} in the same plan targeted the same identifier.

    Plans with ambiguous targets are rejected before anything is applied.
    """

    def __init__(self, identifier: str) -> None:
        super().__init__(
            f"duplicate change identifier in the same plan: {identifier!r}"
        )
        self.identifier = identifier


class StaleSnapshot(ChangeProtocolError):
    """
    A plan was committed against a resource snapshot that had already moved.

    The caller must prepare a new L{CandidatePlan} against the current state
    and may then commit that one instead.
    """

    def __init__(self, expectedVersion: int, currentVersion: int) -> None:
        super().__init__(
            "stale snapshot: plan was prepared at version "
            f"{expectedVersion}, current version is {currentVersion}"
        )
        self.expectedVersion = expectedVersion
        self.currentVersion = currentVersion


class ChangeApplicationError(ChangeProtocolError):
    """
    Applying one of the changes in a committed plan failed.

    Nothing was published: the store retains its pre-commit state.  The
    failed commit was not recorded, so L{TwoPhaseResourceStore.commit} may be
    called again with the same L{plan} (possibly after the underlying problem
    has been resolved).
    """

    def __init__(
        self,
        identifier: str,
        context: ChangeContext,
        plan: CandidatePlan,
        failure: Failure,
    ) -> None:
        super().__init__(
            f"applying change {identifier!r} failed: "
            f"{failure.getErrorMessage()}"
        )
        self.identifier = identifier
        self.context = context
        self.plan = plan
        self.failure = failure


@define(frozen=True)
class Applied:
    """
    The outcome of a L{Change.execute} callable when it needs more than a bare
    new value.

    @param value: the new resource value, or L{DELETE} to remove it.

    @param rollback: an optional compensating action, invoked (in reverse
        order) if a later change in the same commit fails.  It receives the
        plan's request context.
    """

    value: Any
    rollback: Rollback | None = None


@define(frozen=True)
class Change:
    """
    A single proposed resource change.

    @param identifier: the targeted resource identifier.  Identifiers must be
        unique within a single plan.

    @param execute: called during commit with the resource's current value
        (C{None} if it does not exist) and the plan's request context.  It
        returns the new resource value (or L{DELETE}), optionally wrapped in
        L{Applied}, and may return a L{Deferred} of any of those.
    """

    identifier: str
    execute: Callable[[Any, ChangeContext], ExecuteResult]


@define(frozen=True)
class CandidatePlan:
    """
    A prepared, not-yet-committed batch of changes.

    @param identifier: a unique token identifying this plan, used for
        idempotent commits and retries.

    @param changes: the validated changes in application order.

    @param version: the version of the resource snapshot this plan was
        prepared against.

    @param resourceVersions: the per-resource versions observed while
        preparing the plan (C{0} for resources that did not exist).

    @param context: the request context that will be handed to each change's
        L{Change.execute} (and any compensating action).
    """

    identifier: str
    changes: Sequence[Change]
    version: int
    resourceVersions: Mapping[str, int]
    context: ChangeContext


@define(frozen=True)
class CommitResult:
    """
    The outcome of a successful L{TwoPhaseResourceStore.commit}.

    @param planIdentifier: the L{CandidatePlan.identifier} that committed.

    @param version: the store version resulting from the commit.

    @param replayed: C{True} when this result was produced by re-committing an
        already-committed plan; no change executed again.

    @param results: the value produced by each change in order (C{None} for a
        deletion).
    """

    planIdentifier: str
    version: int
    replayed: bool
    results: Sequence[Any]


@define(frozen=True)
class ResourceSnapshot:
    """
    An immutable point-in-time view of a L{TwoPhaseResourceStore}.
    """

    version: int
    resources: Mapping[str, Any]
    resourceVersions: Mapping[str, int]


@define
class _Record:
    """
    A stored resource value along with its version.
    """

    value: Any
    version: int


async def _ignore(ready: Deferred[None]) -> None:
    """
    Wait for C{ready}, allowing (but discarding) a prior failure so that one
    failed commit does not block later commits in the serialization chain.
    """
    try:
        await ready
    except Exception:  # pragma: no cover - exercised via failure tests
        pass


@define
class TwoPhaseResourceStore:
    """
    An in-memory, versioned resource store speaking the two-phase change
    protocol.

    The store maintains a monotonically increasing L{version}; each successful
    commit increases it by one.  Commits are serialized, so a commit whose
    changes are still in flight asynchronously prevents a concurrent commit
    from observing a half-prepared state.
    """

    _resources: dict[str, _Record] = field(factory=dict, init=False)
    _revision: int = field(default=0, init=False)
    _committed: dict[str, CommitResult] = field(factory=dict, init=False)
    _tail: Deferred[None] | None = field(default=None, init=False)
    _counter: int = field(default=0, init=False, eq=False)

    @property
    def version(self) -> int:
        """
        The current snapshot version of the store.
        """
        return self._revision

    def get(self, identifier: str) -> Any:
        """
        Return the current value of the resource with C{identifier}, or
        C{None} if it does not exist.
        """
        record = self._resources.get(identifier)
        if record is None:
            return None
        return record.value

    def resourceVersion(self, identifier: str) -> int:
        """
        Return the version of the identified resource (C{0} if absent).
        """
        record = self._resources.get(identifier)
        if record is None:
            return 0
        return record.version

    def snapshot(self) -> ResourceSnapshot:
        """
        Return an immutable L{ResourceSnapshot} of the current state.
        """
        return ResourceSnapshot(
            version=self._revision,
            resources=MappingProxyType(
                {key: record.value for key, record in self._resources.items()}
            ),
            resourceVersions=MappingProxyType(
                {key: record.version for key, record in self._resources.items()}
            ),
        )

    def committedResult(self, planIdentifier: str) -> CommitResult | None:
        """
        Return the recorded L{CommitResult} for an already-committed plan
        identifier, or C{None} if the plan has not committed successfully.
        """
        return self._committed.get(planIdentifier)

    @eagerDeferredCoroutine
    async def plan(
        self,
        changes: Iterable[Change],
        context: ChangeContext = None,
    ) -> CandidatePlan:
        """
        Phase one: validate C{changes} and return a versioned
        L{CandidatePlan} without writing anything.

        @raises EmptyChanges: if no changes were supplied.

        @raises DuplicateIdentifier: if two changes target the same
            identifier.
        """
        requested = tuple(changes)
        if not requested:
            raise EmptyChanges()

        seen: set[str] = set()
        for change in requested:
            if change.identifier in seen:
                raise DuplicateIdentifier(change.identifier)
            seen.add(change.identifier)

        self._counter += 1
        return CandidatePlan(
            identifier=f"plan-{self._counter}",
            changes=requested,
            version=self._revision,
            resourceVersions=MappingProxyType(
                {
                    change.identifier: self.resourceVersion(change.identifier)
                    for change in requested
                }
            ),
            context=context,
        )

    def commit(self, plan: CandidatePlan) -> Deferred[CommitResult]:
        """
        Phase two: atomically apply C{plan} if its snapshot is still current.

        Commits are serialized against one another.  A failed commit leaves
        both the state and the plan untouched and reports a
        L{ChangeApplicationError}; committing the same plan again retries it.
        Committing a plan that has already committed returns the original
        result with L{CommitResult.replayed} set.

        @raises StaleSnapshot: if L{CandidatePlan.version} no longer matches
            the store's current version.
        """
        ready = self._tail
        gate: Deferred[None] = Deferred()
        self._tail = gate

        @eagerDeferredCoroutine
        async def run() -> CommitResult:
            if ready is not None:
                await _ignore(ready)
            try:
                return await self._commit(plan)
            finally:
                if not gate.called:
                    gate.callback(None)

        return run()

    @eagerDeferredCoroutine
    async def _commit(self, plan: CandidatePlan) -> CommitResult:
        """
        Apply C{plan} against a private staging area and publish it only on
        complete success.  Called while this store's commit chain holds the
        serialization slot.
        """
        cached = self._committed.get(plan.identifier)
        if cached is not None:
            # Idempotent retry: the plan already committed successfully.
            return evolve(cached, replayed=True)

        if plan.version != self._revision:
            raise StaleSnapshot(plan.version, self._revision)

        staged = dict(self._resources)
        rollbacks: list[tuple[str, Rollback]] = []
        results: list[Any] = []
        failingIdentifier = ""
        try:
            for change in plan.changes:
                failingIdentifier = change.identifier
                existing = staged.get(change.identifier)
                outcome = change.execute(
                    None if existing is None else existing.value,
                    plan.context,
                )
                if isinstance(outcome, Deferred):
                    outcome = await outcome
                if isinstance(outcome, Applied):
                    value = outcome.value
                    if outcome.rollback is not None:
                        rollbacks.append((change.identifier, outcome.rollback))
                else:
                    value = outcome

                if isinstance(value, _Delete):
                    results.append(None)
                    staged.pop(change.identifier, None)
                else:
                    results.append(value)
                    priorVersion = 0 if existing is None else existing.version
                    staged[change.identifier] = _Record(value, priorVersion + 1)

            # Re-check the snapshot immediately before publishing.  The
            # serialization chain makes this redundant within a single store
            # today, but it makes the conflict guarantee explicit and keeps
            # holding if the store gains other mutation paths.
            if plan.version != self._revision:
                raise StaleSnapshot(plan.version, self._revision)
        except StaleSnapshot:
            raise
        except Exception as exc:
            await self._rollback(plan.context, rollbacks)
            raise ChangeApplicationError(
                failingIdentifier, plan.context, plan, Failure()
            ) from exc

        self._resources.clear()
        self._resources.update(staged)
        self._revision += 1
        committed = CommitResult(
            planIdentifier=plan.identifier,
            version=self._revision,
            replayed=False,
            results=tuple(results),
        )
        self._committed[plan.identifier] = committed
        return committed

    @eagerDeferredCoroutine
    async def _rollback(
        self,
        context: ChangeContext,
        rollbacks: Sequence[tuple[str, Rollback]],
    ) -> None:
        """
        Run compensating actions in reverse order.

        A failing compensating action is logged but does not mask the
        original application failure or stop later compensations.
        """
        for identifier, rollback in reversed(rollbacks):
            try:
                outcome = rollback(context)
                if isinstance(outcome, Deferred):
                    await outcome
            except Exception:
                log.failure(
                    "compensating action for change {identifier!r} failed",
                    identifier=identifier,
                )


def _jsonBody(payload: Mapping[str, Any]) -> bytes:
    """
    Serialize C{payload} as a JSON response body.
    """
    return json.dumps(payload, sort_keys=True).encode("utf-8")


@define
class TwoPhaseChangeAPI:
    """
    A small HTTP rendering binding for L{TwoPhaseResourceStore}, suitable for
    use directly from Klein route handlers.

    Prepared plans are kept server-side keyed by their token; committing (and
    retrying) is a matter of rendering that token again.  Protocol failures
    render as ordinary responses rather than propagating to Klein's error
    machinery: empty/duplicate input is C{422}, an unknown plan token C{404},
    a stale snapshot C{409} (the client should re-plan), and an application
    failure C{500} with an explicitly retryable body.
    """

    store: TwoPhaseResourceStore
    _pending: dict[str, CandidatePlan] = field(factory=dict, init=False)

    @eagerDeferredCoroutine
    async def renderPlan(
        self,
        request: IRequest,
        changes: Iterable[Change],
        context: ChangeContext = None,
    ) -> Response:
        """
        Prepare C{changes} and render the resulting candidate plan.

        The request itself is used as the change context when C{context} is
        not supplied, so asynchronous failures and retries retain the
        originating request.
        """
        try:
            plan = await self.store.plan(
                changes, request if context is None else context
            )
        except EmptyChanges:
            return Response(
                code=422,
                headers={"Content-Type": "application/json"},
                body=_jsonBody({"error": "no changes supplied"}),
            )
        except DuplicateIdentifier as duplicate:
            return Response(
                code=422,
                headers={"Content-Type": "application/json"},
                body=_jsonBody(
                    {
                        "error": "duplicate identifier",
                        "identifier": duplicate.identifier,
                    }
                ),
            )
        self._pending[plan.identifier] = plan
        return Response(
            code=200,
            headers={"Content-Type": "application/json"},
            body=_jsonBody(
                {
                    "plan": plan.identifier,
                    "version": plan.version,
                    "changes": [change.identifier for change in plan.changes],
                }
            ),
        )

    @eagerDeferredCoroutine
    async def renderCommit(
        self,
        request: IRequest,
        planIdentifier: str,
    ) -> Response:
        """
        Commit the prepared plan identified by C{planIdentifier}, rendering
        the result.

        Repeating a rendered commit retries a failed attempt or replays a
        successful one, so a lost response is safe to re-request.
        """
        plan = self._pending.get(planIdentifier)
        if plan is None:
            return Response(
                code=404,
                headers={"Content-Type": "application/json"},
                body=_jsonBody({"error": f"unknown plan {planIdentifier!r}"}),
            )
        try:
            result = await self.store.commit(plan)
        except StaleSnapshot as stale:
            return Response(
                code=409,
                headers={"Content-Type": "application/json"},
                body=_jsonBody(
                    {
                        "error": "snapshot expired; prepare a new plan",
                        "expectedVersion": stale.expectedVersion,
                        "currentVersion": stale.currentVersion,
                    }
                ),
            )
        except ChangeApplicationError as failed:
            return Response(
                code=500,
                headers={"Content-Type": "application/json"},
                body=_jsonBody(
                    {
                        "error": str(failed),
                        "identifier": failed.identifier,
                        "retryable": True,
                    }
                ),
            )
        return Response(
            code=200,
            headers={"Content-Type": "application/json"},
            body=_jsonBody(
                {
                    "plan": result.planIdentifier,
                    "version": result.version,
                    "replayed": result.replayed,
                }
            ),
        )


__all__ = (
    "Applied",
    "CandidatePlan",
    "Change",
    "ChangeApplicationError",
    "ChangeProtocolError",
    "CommitResult",
    "DELETE",
    "DuplicateIdentifier",
    "EmptyChanges",
    "ResourceSnapshot",
    "StaleSnapshot",
    "TwoPhaseChangeAPI",
    "TwoPhaseResourceStore",
)
