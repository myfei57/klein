# -*- test-case-name: klein.test.test_versioning -*-
"""
A concurrent read-model for incremental resource rendering.

The rendering problem this solves looks like this:

    - a single writer publishes rendered versions as they become available;

    - several independent readers (each typically corresponding to one
      request) want to be delivered the increments that appeared since the
      version they last saw, identified by an opaque L{Cursor};

    - a reader that fell behind while the writer kept publishing must not
      block (or back-pressure) the writer in any way; it only receives a
      recoverable L{ReaderLagging} error telling it where to resume;

    - asynchronous rendering failures must leave the reader's position
      untouched, preserve the originating request context, and remain fully
      retryable against the same immutable versions.

The model:

    - L{VersionLog.publish} commits a value at the next position, producing an
      immutable L{PublishedVersion}.  Publication is synchronous and never
      waits for readers.

    - Only a bounded number of versions is retained (C{history}).  A reader
      whose L{Cursor} points before the oldest retained version fails with
      L{ReaderLagging}; L{ReaderLagging.recoveryCursor} lets it resume from
      the oldest version still available.

    - L{VersionLog.offer} accepts versions whose asynchronous production
      completed out of order.  Values are buffered in a bounded reorder
      window and only a contiguous prefix is committed, so every reader
      always observes versions in a single, consistent order regardless of
      the arrival order of the inputs.

    - L{VersionLog.close} is terminal: readers first drain all committed but
      undelivered versions and then receive L{PublisherClosed}.  A pending
      read may be cancelled; cancellation only removes the waiter, the
      cursor is not advanced, and the read may be retried.

    - Version labels are non-negative integers.  With C{modulus=0} (the
      default) they grow without bound.  With a positive C{modulus} labels
      wrap to 0 after C{modulus - 1} and L{PublishedVersion.epoch} is
      incremented; cursors are opaque tokens that remain unambiguous across
      wraps, so a slow reader spanning a wrap still gets L{ReaderLagging}
      rather than silently skipping or re-reading versions.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from itertools import count
from typing import Any, Generic, TypeVar, cast, overload

from attr import frozen

from twisted.internet.defer import (
    CancelledError,
    Deferred,
    fail,
    maybeDeferred,
    succeed,
)
from twisted.python.failure import Failure

from ._util import eagerDeferredCoroutine

__all__ = (
    "Cursor",
    "DuplicateVersion",
    "InvalidCursor",
    "PublishedVersion",
    "PublisherClosed",
    "ReaderLagging",
    "ReorderWindowExceeded",
    "RenderFailed",
    "RenderedUpdate",
    "VersionLog",
    "VersionReader",
    "VersionUpdate",
    "VersioningError",
)

T = TypeVar("T")
R = TypeVar("R")

_logIds = count(1)


class VersioningError(Exception):
    """
    Base class for all errors raised by the versioned read-model.
    """


class ReaderLagging(VersioningError):
    """
    A reader's cursor is older than the oldest version still retained.

    This error is recoverable: reset the reader to L{recoveryCursor} (or
    resubscribe) and read again.  The versions between the stale cursor and
    L{oldestAvailable} have been discarded and can no longer be replayed.
    """

    def __init__(
        self,
        cursor: Cursor,
        oldestAvailable: PublishedVersion[Any],
        latest: PublishedVersion[Any],
        recoveryCursor: Cursor,
    ) -> None:
        self.cursor = cursor
        self.oldestAvailable = oldestAvailable
        self.latest = latest
        self.recoveryCursor = recoveryCursor
        super().__init__(
            f"reader cursor at position {cursor.position} is behind the "
            f"oldest retained version at position "
            f"{oldestAvailable.position}; resume at "
            f"{recoveryCursor.position}"
        )


class InvalidCursor(VersioningError):
    """
    A cursor was presented that does not name a readable position on this
    log: it was issued by a different log, or it lies beyond the latest
    published version.
    """

    def __init__(self, cursor: Cursor, reason: str) -> None:
        self.cursor = cursor
        self.reason = reason
        super().__init__(f"invalid cursor {cursor!r}: {reason}")


class PublisherClosed(VersioningError):
    """
    The log has been closed.

    All committed versions have already been delivered (or remain readable
    once, for a reader that has not drained them yet); no further versions
    will ever be published.
    """

    def __init__(self, lastPosition: int | None) -> None:
        self.lastPosition = lastPosition
        super().__init__(
            "version log is closed"
            + (
                f" after position {lastPosition}"
                if lastPosition is not None
                else " without any published versions"
            )
        )


class DuplicateVersion(VersioningError):
    """
    A version was offered more than once, or an attempt was made to publish
    a position that has already been committed.
    """


class ReorderWindowExceeded(VersioningError):
    """
    An out-of-order version arrived so far ahead of the commit frontier that
    accepting it would require unbounded buffering.

    Publication of already committed versions is unaffected; the offending
    offer was not recorded and may be retried once earlier versions arrive,
    if the producer can supply them.
    """


class RenderFailed(VersioningError):
    """
    Asynchronously rendering a published version failed.

    The reader's cursor was not advanced, L{requestContext} is the context
    object the reader was subscribed with (typically the request being
    rendered), L{requestedCursor} is the cursor the failing read started
    from, and L{failure} is the underlying failure.  The read is fully
    retryable: reading again renders the same immutable versions again.
    """

    def __init__(
        self,
        requestContext: object | None,
        requestedCursor: Cursor,
        version: PublishedVersion[Any],
        failure: Failure,
    ) -> None:
        self.requestContext = requestContext
        self.requestedCursor = requestedCursor
        self.version = version
        self.failure = failure
        super().__init__(
            f"rendering of version epoch={version.epoch} "
            f"version={version.version} (position {version.position}) "
            f"failed for context {requestContext!r}: "
            f"{failure.getErrorMessage()}"
        )


@frozen
class Cursor:
    """
    An opaque, immutable reader position.

    Cursors are issued by a L{VersionLog} (via L{VersionUpdate.cursor} or
    L{VersionLog.startCursor}) and must be presented back to the same log.
    C{position} is unique for the lifetime of a log, including across label
    wraps, so cursors never become ambiguous when version labels wrap.
    """

    logId: int
    position: int
    epoch: int
    version: int


@frozen
class PublishedVersion(Generic[T]):
    """
    An immutable, published value at one position.

    C{version} is the (possibly wrapped) version label, C{epoch} is the
    wrap generation (zero until the first wrap) and C{position} is the
    log-unique monotonic position.  C{value} must never be mutated after
    publication.
    """

    version: int
    epoch: int
    position: int
    value: T


@frozen
class VersionUpdate(Generic[T]):
    """
    A contiguous, ascending batch of L{PublishedVersion} delivered to a
    reader, together with the cursor that results from consuming it.

    C{closed} is true exactly when this batch ends at the log's final
    version.
    """

    versions: tuple[PublishedVersion[T], ...]
    cursor: Cursor
    closed: bool


@frozen
class RenderedUpdate(Generic[T, R]):
    """
    Like L{VersionUpdate}, but carrying the renderer's output for each
    delivered version in C{results}.
    """

    versions: tuple[PublishedVersion[T], ...]
    results: tuple[R, ...]
    cursor: Cursor
    closed: bool


class VersionLog(Generic[T]):
    """
    A bounded log of immutable, ordered published versions.

    @param history: maximum number of committed versions retained for slow
        readers; must be at least 1.  It may exceed C{modulus}: retained
        labels then repeat, but epochs (and cursors) stay unambiguous.

    @param modulus: when 0 version labels grow without bound; when positive
        labels run from C{0} to C{modulus - 1} and then wrap to C{0} with a
        new epoch.

    @param reorderWindow: maximum number of positions (counting from the
        commit frontier) accepted out of order by L{offer}; defaults to
        C{history}.  With wrapping it must not exceed C{modulus}, so a
        buffered label names exactly one position.
    """

    def __init__(
        self,
        *,
        history: int = 64,
        modulus: int = 0,
        reorderWindow: int | None = None,
    ) -> None:
        if history < 1:
            raise ValueError("history must be at least 1")
        if modulus < 0:
            raise ValueError("modulus must be 0 (unbounded) or positive")
        if reorderWindow is None:
            reorderWindow = min(history, modulus) if modulus else history
        if reorderWindow < 1:
            raise ValueError("reorderWindow must be at least 1")
        if modulus and reorderWindow > modulus:
            raise ValueError("reorderWindow must not exceed modulus")

        self._id = next(_logIds)
        self._history = history
        self._modulus = modulus
        self._reorderWindow = reorderWindow
        self._retained: deque[PublishedVersion[T]] = deque()
        self._pending: dict[int, T] = {}
        self._nextPosition = 0
        self._closed = False
        self._waiters: set[Deferred[None]] = set()

    @property
    def closed(self) -> bool:
        """
        Whether L{close} has been called.
        """
        return self._closed

    @property
    def latest(self) -> PublishedVersion[T] | None:
        """
        The latest committed version, or L{None} if nothing has been
        published yet.
        """
        if not self._retained:
            return None
        return self._retained[-1]

    @property
    def oldest(self) -> PublishedVersion[T] | None:
        """
        The oldest version still retained.
        """
        if not self._retained:
            return None
        return self._retained[0]

    def startCursor(self) -> Cursor:
        """
        The cursor positioned before the first version: reading with it
        delivers from the oldest version still retained.
        """
        return Cursor(self._id, -1, -1, -1)

    def subscribe(
        self,
        requestContext: object | None = None,
        *,
        cursor: Cursor | None = None,
    ) -> VersionReader[T]:
        """
        Create an independent reader.

        @param requestContext: arbitrary opaque object (typically the
            request being rendered) that is handed back to renderers and
            preserved on L{RenderFailed}.

        @param cursor: resume from an existing L{Cursor}; L{None} starts at
            the oldest retained version.
        """
        if cursor is None:
            token = self.startCursor()
        else:
            token = self._validate(cursor)
        return VersionReader(self, requestContext, token)

    def publish(self, value: T) -> PublishedVersion[T]:
        """
        Commit C{value} as the next in-order version.

        Never blocks on or waits for readers.  Retention beyond C{history}
        is evicted immediately, regardless of any reader's position.
        """
        if self._closed:
            raise PublisherClosed(self._latestPosition())
        # The frontier position is never buffered: offer() flushes it the
        # moment it arrives, so inserting here always commits immediately.
        self._pending[self._nextPosition] = value
        committed = self._flush()
        return committed[0]

    def offer(self, version: int, value: T) -> bool:
        """
        Offer a version whose production may have completed out of order.

        C{version} is an absolute position when the log is unbounded, or a
        wrapped label within the open reorder window when the log wraps.
        Only the value (and any contiguous values already buffered) is
        committed; gaps remain buffered.  Returns true if the offered
        version was committed.

        @raises DuplicateVersion: the version was already committed or is
            already buffered.
        @raises ReorderWindowExceeded: the version is more than
            C{reorderWindow} positions ahead of the commit frontier.
        """
        if self._closed:
            raise PublisherClosed(self._latestPosition())
        if self._modulus == 0:
            if version < 0:
                raise ValueError("versions must be non-negative")
            position = version
        else:
            if not 0 <= version < self._modulus:
                raise ValueError(
                    f"version label {version} outside 0..{self._modulus - 1}"
                )
            frontier = self._nextPosition
            distance = (version - self._label(frontier)) % self._modulus
            position = frontier + distance

        if position < self._nextPosition:
            raise DuplicateVersion(f"position {position} was already committed")
        if position in self._pending:
            raise DuplicateVersion(f"position {position} already offered")
        if position >= self._nextPosition + self._reorderWindow:
            raise ReorderWindowExceeded(
                f"position {position} is more than {self._reorderWindow} "
                f"ahead of the frontier {self._nextPosition}"
            )
        self._pending[position] = value
        self._flush()
        return position < self._nextPosition

    def close(self) -> None:
        """
        Close the log.

        Idempotent.  Buffered values that never completed a contiguous
        prefix are abandoned: they were never visible to any reader.
        Pending readers are woken so that they drain committed versions
        before observing L{PublisherClosed}.  Any further L{publish} or
        L{offer} raises L{PublisherClosed}.
        """
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        self._wake()

    def _latestPosition(self) -> int | None:
        latest = self.latest
        if latest is None:
            return None
        return latest.position

    def _label(self, position: int) -> int:
        if self._modulus == 0:
            return position
        return position % self._modulus

    def _epoch(self, position: int) -> int:
        if self._modulus == 0:
            return 0
        return position // self._modulus

    def _makeCursor(self, position: int) -> Cursor:
        return Cursor(
            self._id, position, self._epoch(position), self._label(position)
        )

    def _validate(self, cursor: Cursor) -> Cursor:
        if cursor.logId != self._id:
            raise InvalidCursor(cursor, "cursor was issued by another log")
        return cursor

    def _flush(self) -> list[PublishedVersion[T]]:
        committed: list[PublishedVersion[T]] = []
        while self._nextPosition in self._pending:
            position = self._nextPosition
            value = self._pending.pop(position)
            published = PublishedVersion(
                self._label(position),
                self._epoch(position),
                position,
                value,
            )
            self._retained.append(published)
            if len(self._retained) > self._history:
                self._retained.popleft()
            self._nextPosition += 1
            committed.append(published)
        if committed:
            self._wake()
        return committed

    def _wake(self) -> None:
        # Snapshot and clear before firing: callbacks run after the state
        # commit, and readers that re-register register a fresh waiter.
        # A woken reader's callback may cancel another pending waiter, so
        # skip triggers that fired re-entrantly; publication must not fail
        # because of what readers do on wake.
        waiters = self._waiters
        self._waiters = set()
        for waiter in waiters:
            if not waiter.called:
                waiter.callback(None)

    def _wait(self) -> Deferred[None]:
        def cancel(trigger: Deferred[None]) -> None:
            self._waiters.discard(trigger)
            trigger.errback(CancelledError())

        trigger: Deferred[None] = Deferred(cancel)
        self._waiters.add(trigger)
        return trigger

    def _collect(
        self,
        reader: VersionReader[T],
        maxItems: int | None,
        commit: bool,
    ) -> VersionUpdate[T] | None:
        """
        Snapshot and optionally commit the versions available to a reader.

        Returns L{None} when the log is open but no version is currently
        available; raises L{PublisherClosed} when closed and drained,
        L{ReaderLagging} and L{InvalidCursor} as appropriate.
        """
        cursor = self._validate(reader.cursor)
        retained = self._retained
        position = cursor.position

        if not retained:
            if position >= 0:
                raise InvalidCursor(
                    cursor, "no versions have ever been published"
                )
            if self._closed:
                raise PublisherClosed(None)
            return None

        oldest = retained[0]
        latest = retained[-1]

        if position == -1:
            # The start (or resync) cursor: deliver from the oldest
            # retained version, never a lag error.
            versions = tuple(retained)
        elif position > latest.position:
            raise InvalidCursor(
                cursor,
                f"cursor position {position} is beyond the latest "
                f"published position {latest.position}",
            )
        elif position < oldest.position - 1:
            raise ReaderLagging(
                cursor,
                oldest,
                latest,
                self._makeCursor(oldest.position - 1),
            )
        else:
            versions = tuple(pv for pv in retained if pv.position > position)
        if not versions:
            if self._closed:
                raise PublisherClosed(latest.position)
            return None

        if maxItems is not None:
            versions = versions[:maxItems]

        last = versions[-1]
        update = VersionUpdate(
            versions,
            self._makeCursor(last.position),
            self._closed and last.position == latest.position,
        )
        if commit:
            reader._cursor = update.cursor
        return update


class VersionReader(Generic[T]):
    """
    An independent cursor over a L{VersionLog}.

    Readers do not retain versions and never influence the writer: each
    reader only mutates its own cursor.
    """

    def __init__(
        self,
        log: VersionLog[T],
        requestContext: object | None,
        cursor: Cursor,
    ) -> None:
        self._log = log
        self.requestContext = requestContext
        self._cursor = cursor

    @property
    def cursor(self) -> Cursor:
        """
        The cursor of the last version delivered to this reader.
        """
        return self._cursor

    def reset(self, cursor: Cursor | None = None) -> None:
        """
        Reset this reader's position, e.g. to
        L{ReaderLagging.recoveryCursor}.  L{None} returns to the start.
        """
        if cursor is None:
            self._cursor = self._log.startCursor()
        else:
            self._cursor = self._log._validate(cursor)

    def poll(self, maxItems: int | None = None) -> VersionUpdate[T] | None:
        """
        Non-blocking read: deliver currently available versions, return
        L{None} if the log is open with nothing new, or raise the same
        errors as L{read}.
        """
        return self._log._collect(self, maxItems, commit=True)

    def read(self, maxItems: int | None = None) -> Deferred[VersionUpdate[T]]:
        """
        Deliver the next increment.

        Succeeds immediately when versions are already available; otherwise
        returns a L{Deferred} that fires when the writer publishes or
        closes the log.  Cancelling a pending read removes the waiter
        without advancing the cursor and may be retried.
        """
        try:
            update = self._log._collect(self, maxItems, commit=True)
        except Exception:
            return fail()

        if update is not None:
            return succeed(update)

        trigger = self._log._wait()

        def woke(_: object) -> Deferred[VersionUpdate[T]]:
            return self.read(maxItems)

        trigger.addCallback(woke)
        return cast("Deferred[VersionUpdate[T]]", trigger)

    @overload
    def rendered(
        self,
        renderer: Callable[[object | None, PublishedVersion[T]], Deferred[R]],
        maxItems: int | None = None,
    ) -> Deferred[RenderedUpdate[T, R]]: ...

    @overload
    def rendered(
        self,
        renderer: Callable[[object | None, PublishedVersion[T]], R],
        maxItems: int | None = None,
    ) -> Deferred[RenderedUpdate[T, R]]: ...

    @eagerDeferredCoroutine
    async def rendered(
        self,
        renderer: Callable[[object | None, PublishedVersion[T]], Any],
        maxItems: int | None = None,
    ) -> RenderedUpdate[T, Any]:
        """
        Read the next increment and render each immutable version with
        C{renderer}, which receives this reader's request context and the
        L{PublishedVersion} and may return a value or a L{Deferred}.

        The cursor is committed only after every version has rendered
        successfully.  If rendering fails, the returned L{Deferred} fails
        with L{RenderFailed} (which preserves the request context, the
        starting cursor, the failed version and the underlying failure);
        the cursor is unchanged and the operation may be retried, rendering
        the same immutable versions again.  Cancellation propagates as
        L{twisted.internet.defer.CancelledError}, likewise without moving
        the cursor.
        """
        while True:
            snapshot = self._log._collect(self, maxItems, commit=False)
            if snapshot is None:
                await self._log._wait()
                continue

            results: list[Any] = []
            requested = self._cursor
            try:
                for published in snapshot.versions:
                    outcome = maybeDeferred(
                        renderer, self.requestContext, published
                    )
                    results.append(await outcome)
            except CancelledError:
                raise
            except Exception:
                raise RenderFailed(
                    self.requestContext,
                    requested,
                    snapshot.versions[len(results)],
                    Failure(),
                ) from None

            last = snapshot.versions[-1]
            self._cursor = self._log._makeCursor(last.position)
            return RenderedUpdate(
                snapshot.versions,
                tuple(results),
                self._cursor,
                snapshot.closed,
            )
