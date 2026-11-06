"""
Tests for L{klein._versioning}, the concurrent incremental-read model for
resource rendering.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

from twisted.internet.defer import (
    CancelledError,
    Deferred,
    fail,
    succeed,
)
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase

from klein._versioning import (
    DuplicateVersion,
    InvalidCursor,
    PublishedVersion,
    PublisherClosed,
    ReaderLagging,
    RenderFailed,
    ReorderWindowExceeded,
    VersionLog,
    VersionUpdate,
)


def labels(update: VersionUpdate[Any]) -> list[tuple[int, int]]:
    """
    Extract ``(epoch, label)`` pairs from a version update for assertions.
    """
    return [(pv.epoch, pv.version) for pv in update.versions]


class PublishTests(SynchronousTestCase):
    """
    Committed versions are immutable, ordered and numbered monotonically.
    """

    def test_publishNumbersVersions(self) -> None:
        log: VersionLog[str] = VersionLog()
        first = log.publish("a")
        second = log.publish("b")

        self.assertEqual(
            first,
            PublishedVersion(version=0, epoch=0, position=0, value="a"),
        )
        self.assertEqual(
            second,
            PublishedVersion(version=1, epoch=0, position=1, value="b"),
        )
        self.assertEqual(log.oldest, first)
        self.assertEqual(log.latest, second)

    def test_publishedValueIsNotCopiedOrMutated(self) -> None:
        payload = ("immutable", (1, 2))
        log: VersionLog[tuple[str, tuple[int, ...]]] = VersionLog()
        log.publish(payload)
        # The exact object published is what readers receive.
        latest = log.latest
        assert latest is not None
        self.assertIs(latest.value, payload)

    def test_publishNeverWaitsForReaders(self) -> None:
        """
        Even with a permanently passive pending reader, the writer keeps
        publishing synchronously: readers exert no back-pressure.
        """
        log: VersionLog[int] = VersionLog(history=3)
        waiter = log.subscribe("request")
        pending = waiter.read()
        self.assertNoResult(pending)

        for index in range(50):
            published = log.publish(index)
            self.assertEqual(published.position, index)

        # The slow reader is now behind the horizon, not blocking anyone.
        with self.assertRaises(ReaderLagging):
            waiter.poll()
        self.assertEqual(log.latest.position, 49)  # type: ignore[union-attr]


class IncrementalReadTests(SynchronousTestCase):
    """
    Multiple readers take independent increments from their own cursors.
    """

    def test_independentCursors(self) -> None:
        log: VersionLog[str] = VersionLog()
        slow = log.subscribe("slow-request")
        fast = log.subscribe("fast-request")

        for value in ("a", "b", "c"):
            log.publish(value)

        fastAll = self.successResultOf(fast.read())
        self.assertEqual([pv.value for pv in fastAll.versions], ["a", "b", "c"])
        self.assertEqual(fastAll.cursor.position, 2)
        self.assertFalse(fastAll.closed)

        # The fast reader's consumption does not move the slow reader.
        slowFirst = self.successResultOf(slow.read(maxItems=1))
        self.assertEqual([pv.value for pv in slowFirst.versions], ["a"])
        self.assertEqual(slow.cursor.position, 0)

        slowRest = self.successResultOf(slow.read())
        self.assertEqual([pv.value for pv in slowRest.versions], ["b", "c"])
        self.assertEqual(slow.cursor.position, 2)

    def test_pendingReadFiresOnPublish(self) -> None:
        log: VersionLog[str] = VersionLog()
        reader = log.subscribe()
        pending = reader.read()
        self.assertNoResult(pending)

        log.publish("wake")

        update = self.successResultOf(pending)
        self.assertEqual([pv.value for pv in update.versions], ["wake"])
        self.assertEqual(reader.cursor.position, 0)

    def test_pendingReaderWakesWithBurstCoalesced(self) -> None:
        log: VersionLog[int] = VersionLog()
        reader = log.subscribe()
        pending = reader.read()
        self.assertNoResult(pending)

        # Versions 1..3 complete out of order and buffer; when version 0
        # arrives the whole contiguous prefix commits in one wake, so the
        # pending reader receives the burst as a single increment.
        for position in (3, 2, 1):
            self.assertFalse(log.offer(position, position))
        self.assertTrue(log.offer(0, 0))

        update = self.successResultOf(pending)
        self.assertEqual([pv.value for pv in update.versions], [0, 1, 2, 3])
        self.assertEqual(reader.cursor.position, 3)

    def test_pollReturnsNoneWhenCaughtUp(self) -> None:
        log: VersionLog[int] = VersionLog()
        reader = log.subscribe()
        self.assertIsNone(reader.poll())
        log.publish(0)
        self.successResultOf(reader.read())
        self.assertIsNone(reader.poll())

    def test_emptyLogProperties(self) -> None:
        log: VersionLog[int] = VersionLog()
        self.assertFalse(log.closed)
        self.assertIsNone(log.latest)
        self.assertIsNone(log.oldest)

    def test_newSubscriberStartsAtOldestRetained(self) -> None:
        log: VersionLog[int] = VersionLog(history=2)
        for index in range(5):
            log.publish(index)

        reader = log.subscribe()
        update = self.successResultOf(reader.read())
        # Versions 0..2 were evicted; a fresh reader starts mid-stream
        # without an error.
        self.assertEqual([pv.value for pv in update.versions], [3, 4])
        self.assertEqual(reader.cursor.position, 4)


class SlowReaderTests(SynchronousTestCase):
    """
    A reader behind the retention horizon gets a recoverable lag error and
    nothing else.
    """

    def test_lagErrorIsRecoverable(self) -> None:
        log: VersionLog[int] = VersionLog(history=3)
        reader = log.subscribe("laggy-request")

        log.publish(0)
        first = self.successResultOf(reader.read())
        self.assertEqual([pv.value for pv in first.versions], [0])

        for index in range(1, 6):
            log.publish(index)

        lag = self.failureResultOf(reader.read(), ReaderLagging).value
        self.assertEqual(lag.cursor.position, 0)
        self.assertEqual(lag.oldestAvailable.position, 3)
        self.assertEqual(lag.latest.position, 5)
        self.assertEqual(lag.recoveryCursor.position, 2)
        # The failed read did not move the cursor.
        self.assertEqual(reader.cursor.position, 0)

        reader.reset(lag.recoveryCursor)
        update = self.successResultOf(reader.read())
        self.assertEqual([pv.value for pv in update.versions], [3, 4, 5])
        self.assertEqual(reader.cursor.position, 5)

    def test_resetToStart(self) -> None:
        log: VersionLog[int] = VersionLog(history=2)
        reader = log.subscribe()
        log.publish(0)
        log.publish(1)
        self.successResultOf(reader.read())
        reader.reset()
        update = self.successResultOf(reader.read())
        self.assertEqual([pv.value for pv in update.versions], [0, 1])


class CloseTests(SynchronousTestCase):
    """
    Closing is terminal: committed versions drain first, then readers see
    L{PublisherClosed}.
    """

    def test_drainBeforeTerminal(self) -> None:
        log: VersionLog[int] = VersionLog()
        log.publish(0)
        log.publish(1)
        reader = log.subscribe()
        self.successResultOf(reader.read(maxItems=1))
        log.close()

        remaining = self.successResultOf(reader.read())
        self.assertEqual([pv.value for pv in remaining.versions], [1])
        self.assertTrue(remaining.closed)

        self.failureResultOf(reader.read(), PublisherClosed)

    def test_closedBatchFlagOnlyAtEnd(self) -> None:
        log: VersionLog[int] = VersionLog()
        for index in range(3):
            log.publish(index)
        log.close()
        reader = log.subscribe()

        first = self.successResultOf(reader.read(maxItems=2))
        self.assertFalse(first.closed)
        last = self.successResultOf(reader.read())
        self.assertTrue(last.closed)
        self.failureResultOf(reader.read(), PublisherClosed)

    def test_pendingReaderGetsTerminalOnClose(self) -> None:
        log: VersionLog[int] = VersionLog()
        reader = log.subscribe()
        pending = reader.read()
        self.assertNoResult(pending)

        log.close()

        self.failureResultOf(pending, PublisherClosed)
        self.failureResultOf(reader.read(), PublisherClosed)

    def test_newSubscriberToClosedLogDrains(self) -> None:
        log: VersionLog[int] = VersionLog()
        log.publish(0)
        log.close()

        reader = log.subscribe()
        update = self.successResultOf(reader.read())
        self.assertEqual([pv.value for pv in update.versions], [0])
        self.assertTrue(update.closed)
        self.failureResultOf(reader.read(), PublisherClosed)

    def test_newSubscriberToEmptyClosedLog(self) -> None:
        log: VersionLog[int] = VersionLog()
        log.close()
        reader = log.subscribe()
        failure = self.failureResultOf(reader.read(), PublisherClosed)
        self.assertIsNone(failure.value.lastPosition)

    def test_publishAfterCloseRejected(self) -> None:
        log: VersionLog[int] = VersionLog()
        log.publish(0)
        log.close()
        with self.assertRaises(PublisherClosed):
            log.publish(1)
        with self.assertRaises(PublisherClosed):
            log.offer(1, 1)
        # Idempotent.
        log.close()
        self.assertTrue(log.closed)

    def test_publishOnEmptyClosedLog(self) -> None:
        log: VersionLog[int] = VersionLog()
        log.close()
        with self.assertRaises(PublisherClosed) as publishing:
            log.publish(1)
        self.assertIsNone(publishing.exception.lastPosition)
        with self.assertRaises(PublisherClosed):
            log.offer(0, 0)

    def test_gappedPendingValuesAbandonedAtClose(self) -> None:
        log: VersionLog[int] = VersionLog()
        # Version 2 buffered behind a missing version 1; close must not
        # somehow publish it.
        self.assertFalse(log.offer(2, 2))
        log.close()
        reader = log.subscribe()
        self.failureResultOf(reader.read(), PublisherClosed)


class CancelTests(SynchronousTestCase):
    """
    Cancelling a pending read only removes the waiter; the cursor does not
    move and the read can be retried.
    """

    def test_cancelPendingRead(self) -> None:
        log: VersionLog[str] = VersionLog()
        reader = log.subscribe()
        pending = reader.read()
        self.assertNoResult(pending)

        pending.cancel()

        self.failureResultOf(pending, CancelledError)
        self.assertEqual(reader.cursor, log.startCursor())
        self.assertEqual(log._waiters, set())

    def test_retryAfterCancellationDelivers(self) -> None:
        log: VersionLog[str] = VersionLog()
        reader = log.subscribe()
        pending = reader.read()
        pending.cancel()
        self.failureResultOf(pending, CancelledError)

        log.publish("after-cancel")

        update = self.successResultOf(reader.read())
        self.assertEqual([pv.value for pv in update.versions], ["after-cancel"])

    def test_cancelOneReaderDoesNotDisturbOthers(self) -> None:
        log: VersionLog[str] = VersionLog()
        first = log.subscribe()
        second = log.subscribe()
        firstPending = first.read()
        secondPending = second.read()

        firstPending.cancel()
        self.failureResultOf(firstPending, CancelledError)
        self.assertNoResult(secondPending)

        log.publish("value")
        update = self.successResultOf(secondPending)
        self.assertEqual([pv.value for pv in update.versions], ["value"])

    def test_wokenReaderCancellingAnotherDoesNotBreakPublish(self) -> None:
        """
        Reader callbacks run during publication; if a woken reader cancels
        another pending waiter, publication still completes and continues.
        """
        log: VersionLog[int] = VersionLog()
        first = log.subscribe()
        second = log.subscribe()
        firstPending = first.read()
        secondPending = second.read()
        outcomes: list[VersionUpdate[int] | Failure] = []

        def cancelSecond(result: object) -> object:
            secondPending.cancel()
            return result

        def cancelFirst(result: object) -> object:
            firstPending.cancel()
            return result

        firstPending.addCallback(cancelSecond)
        secondPending.addCallback(cancelFirst)
        firstPending.addBoth(outcomes.append)
        secondPending.addBoth(outcomes.append)

        log.publish(1)

        cancellations = [
            each
            for each in outcomes
            if isinstance(each, Failure) and each.check(CancelledError)
        ]
        updates = [each for each in outcomes if not isinstance(each, Failure)]
        self.assertEqual(len(cancellations), 1)
        self.assertEqual(len(updates), 1)
        self.assertEqual([pv.value for pv in updates[0].versions], [1])

        # The writer is unaffected by reader-side cancellation.
        following = log.publish(2)
        self.assertEqual(following.value, 2)
        self.assertEqual(following.position, 1)


class WraparoundTests(SynchronousTestCase):
    """
    Version labels wrap; epochs advance; cursors stay unambiguous.
    """

    def test_labelsWrapAndEpochsAdvance(self) -> None:
        log: VersionLog[int] = VersionLog(history=8, modulus=4)
        for index in range(10):
            log.publish(index)

        reader = log.subscribe()
        update = self.successResultOf(reader.read())
        self.assertEqual(
            labels(update),
            [
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
            ],
        )
        self.assertEqual(
            [pv.position for pv in update.versions], list(range(2, 10))
        )

    def test_slowReaderAcrossWrapLagsRatherThanAliasing(self) -> None:
        log: VersionLog[int] = VersionLog(history=4, modulus=4)
        reader = log.subscribe()
        log.publish(0)
        log.publish(1)
        self.successResultOf(reader.read())
        self.assertEqual(reader.cursor.position, 1)

        # Publish through positions 2..7; retained is epoch 1's 0..3.
        for index in range(2, 8):
            log.publish(index)

        lag = self.failureResultOf(reader.read(), ReaderLagging).value
        # The stale epoch-0 cursor must not be mistaken for epoch 1's
        # label 1, which would silently skip epoch 1's version 0.
        self.assertEqual(lag.oldestAvailable.epoch, 1)
        self.assertEqual(lag.oldestAvailable.version, 0)
        self.assertEqual(lag.recoveryCursor.position, 3)

        reader.reset(lag.recoveryCursor)
        update = self.successResultOf(reader.read())
        self.assertEqual(
            [(pv.epoch, pv.version, pv.value) for pv in update.versions],
            [(1, 0, 4), (1, 1, 5), (1, 2, 6), (1, 3, 7)],
        )

    def test_constructionConstraints(self) -> None:
        with self.assertRaises(ValueError):
            VersionLog(history=0)
        with self.assertRaises(ValueError):
            VersionLog(modulus=-1)
        with self.assertRaises(ValueError):
            VersionLog(history=4, modulus=4, reorderWindow=5)
        with self.assertRaises(ValueError):
            VersionLog(reorderWindow=0)

        # History may span wraps: epochs keep retained labels unambiguous.
        log: VersionLog[int] = VersionLog(history=8, modulus=4)
        self.assertEqual(log._reorderWindow, 4)


class ReorderTests(SynchronousTestCase):
    """
    Out-of-order inputs commit as a contiguous prefix and every reader sees
    one consistent version order regardless of arrival order.
    """

    def test_gapsBufferedPrefixCommitted(self) -> None:
        log: VersionLog[str] = VersionLog()
        reader = log.subscribe()

        self.assertFalse(log.offer(2, "c"))
        pending = reader.read()
        self.assertNoResult(pending)
        self.assertTrue(log.offer(0, "a"))
        # The pending read wakes and consumes the committed prefix.
        self.assertEqual(
            [pv.value for pv in self.successResultOf(pending).versions],
            ["a"],
        )
        self.assertTrue(log.offer(1, "b"))
        update = self.successResultOf(reader.read())
        self.assertEqual([pv.value for pv in update.versions], ["b", "c"])

    def test_arrivalPermutationsUnbounded(self) -> None:
        seen: list[list[str]] = []
        for order in permutations(range(4)):
            log: VersionLog[str] = VersionLog()
            for position in order:
                log.offer(position, f"v{position}")
            reader = log.subscribe()
            update = self.successResultOf(reader.read())
            seen.append([pv.value for pv in update.versions])

        self.assertEqual(len(seen), 24)
        self.assertEqual(seen, [["v0", "v1", "v2", "v3"]] * 24)

    def test_arrivalPermutationsAcrossWrap(self) -> None:
        seen: list[list[tuple[int, int]]] = []
        for order in permutations(range(4)):
            log: VersionLog[int] = VersionLog(history=8, modulus=4)
            # Commit epoch 0 first; the frontier is now epoch 1.
            for index in range(4):
                log.publish(index)
            for label in order:
                log.offer(label, 100 + label)
            reader = log.subscribe()
            update = self.successResultOf(reader.read())
            seen.append(labels(update))

        expected = [(0, 0), (0, 1), (0, 2), (0, 3)] + [
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
        ]
        self.assertEqual(seen, [expected] * 24)

    def test_duplicateOfferRejected(self) -> None:
        log: VersionLog[int] = VersionLog()
        self.assertTrue(log.offer(0, 0))
        with self.assertRaises(DuplicateVersion):
            log.offer(0, 9)

        self.assertFalse(log.offer(2, 2))
        with self.assertRaises(DuplicateVersion):
            log.offer(2, 9)

    def test_reorderWindowBounded(self) -> None:
        log: VersionLog[int] = VersionLog(history=4, reorderWindow=4)
        # Frontier 0: positions 0..3 accepted, position 4 rejected.
        with self.assertRaises(ReorderWindowExceeded):
            log.offer(4, 4)

        # The rejected offer does not poison the log.
        log.publish(0)
        latest = log.latest
        assert latest is not None
        self.assertEqual(latest.position, 0)
        log.offer(1, 1)

    def test_wrappedLabelOutsideWindowRejected(self) -> None:
        log: VersionLog[int] = VersionLog(history=4, modulus=4)
        with self.assertRaises(ValueError):
            log.offer(4, 4)

    def test_negativePositionRejectedWhenUnbounded(self) -> None:
        log: VersionLog[int] = VersionLog()
        with self.assertRaises(ValueError):
            log.offer(-1, 0)


class InvalidCursorTests(SynchronousTestCase):
    """
    Cursors from other logs or from the future are rejected.
    """

    def test_cursorFromAnotherLog(self) -> None:
        first: VersionLog[int] = VersionLog()
        second: VersionLog[int] = VersionLog()
        cursor = first.startCursor()
        with self.assertRaises(InvalidCursor):
            second.subscribe(cursor=cursor)

    def test_cursorBeyondLatest(self) -> None:
        log: VersionLog[int] = VersionLog(history=4)
        log.publish(0)
        forged = log._makeCursor(9)
        reader = log.subscribe(cursor=forged)
        self.failureResultOf(reader.read(), InvalidCursor)

    def test_namedCursorOnNeverPublishedLog(self) -> None:
        log: VersionLog[int] = VersionLog()
        reader = log.subscribe(cursor=log._makeCursor(0))
        self.failureResultOf(reader.read(), InvalidCursor)

    def test_resetRejectsForeignCursor(self) -> None:
        first: VersionLog[int] = VersionLog()
        second: VersionLog[int] = VersionLog()
        reader = first.subscribe()
        with self.assertRaises(InvalidCursor):
            reader.reset(second.startCursor())


class RenderedTests(SynchronousTestCase):
    """
    Async rendering: failures keep the request context and the cursor
    untouched, and the same immutable versions render again on retry.
    """

    def test_renderedResultsPairWithVersions(self) -> None:
        log: VersionLog[int] = VersionLog()
        for value in (1, 2):
            log.publish(value)
        reader = log.subscribe("the-request")

        def renderer(
            context: object | None, published: PublishedVersion[int]
        ) -> str:
            self.assertEqual(context, "the-request")
            return f"rendered-{published.value}"

        update = self.successResultOf(reader.rendered(renderer))
        self.assertEqual(update.results, ("rendered-1", "rendered-2"))
        self.assertEqual(reader.cursor.position, 1)

    def test_asyncRenderFailurePreservesContextAndIsRetryable(self) -> None:
        log: VersionLog[int] = VersionLog()
        for value in range(3):
            log.publish(value)
        request = object()
        reader = log.subscribe(request)
        attempts: list[int] = []

        def renderer(
            context: object | None, published: PublishedVersion[int]
        ) -> Deferred[str] | str:
            attempts.append(published.value)
            failuresSoFar = [each for each in attempts if each == 1]
            if published.value == 1 and len(failuresSoFar) == 1:
                return fail(RuntimeError("boom"))
            return succeed(f"ok-{published.value}")

        failed = self.failureResultOf(
            reader.rendered(renderer), RenderFailed
        ).value
        self.assertIs(failed.requestContext, request)
        self.assertEqual(failed.requestedCursor, log.startCursor())
        self.assertEqual(failed.version.position, 1)
        self.assertTrue(failed.failure.check(RuntimeError))
        self.assertEqual(failed.failure.getErrorMessage(), "boom")
        # Nothing was committed.
        self.assertEqual(reader.cursor, log.startCursor())

        update = self.successResultOf(reader.rendered(renderer))
        # Version 0 was rendered again because the batch had not committed.
        self.assertEqual(attempts, [0, 1, 0, 1, 2])
        self.assertEqual(update.results, ("ok-0", "ok-1", "ok-2"))
        self.assertEqual(reader.cursor.position, 2)

    def test_synchronousFailureAlsoWrapped(self) -> None:
        log: VersionLog[int] = VersionLog()
        log.publish(7)
        reader = log.subscribe("ctx")

        def renderer(
            context: object | None, published: PublishedVersion[int]
        ) -> str:
            raise ValueError("sync boom")

        failed = self.failureResultOf(
            reader.rendered(renderer), RenderFailed
        ).value
        self.assertIs(failed.requestContext, "ctx")
        self.assertEqual(failed.version.value, 7)
        self.assertTrue(failed.failure.check(ValueError))
        self.assertEqual(reader.cursor, log.startCursor())
        # Still retryable.
        self.successResultOf(reader.rendered(lambda ctx, pv: "fixed"))

    def test_cancelWhileWaitingDoesNotCommit(self) -> None:
        log: VersionLog[int] = VersionLog()
        reader = log.subscribe()
        pending = reader.rendered(lambda ctx, pv: "rendered")
        self.assertNoResult(pending)

        pending.cancel()
        self.failureResultOf(pending, CancelledError)
        self.assertEqual(reader.cursor, log.startCursor())

        log.publish(1)
        update = self.successResultOf(
            reader.rendered(lambda ctx, pv: pv.value * 10)
        )
        self.assertEqual(update.results, (10,))

    def test_pendingRenderedWokenByPublish(self) -> None:
        log: VersionLog[int] = VersionLog()
        reader = log.subscribe()
        pending = reader.rendered(lambda ctx, pv: pv.value * 2)
        self.assertNoResult(pending)

        log.publish(21)

        update = self.successResultOf(pending)
        self.assertEqual(update.results, (42,))
        self.assertFalse(update.closed)
        self.assertEqual(reader.cursor.position, 0)

    def test_cancelDuringRenderCancelsRenderer(self) -> None:
        log: VersionLog[int] = VersionLog()
        log.publish(0)
        reader = log.subscribe()
        renderCancelled: list[bool] = []

        def rendering(
            context: object | None, published: PublishedVersion[int]
        ) -> Deferred[None]:
            def cancel(inner: Deferred[None]) -> None:
                renderCancelled.append(True)
                inner.errback(CancelledError())

            return Deferred(cancel)

        pending = reader.rendered(rendering)
        self.assertNoResult(pending)
        pending.cancel()

        self.failureResultOf(pending, CancelledError)
        self.assertEqual(renderCancelled, [True])
        self.assertEqual(reader.cursor, log.startCursor())

    def test_renderedDrainAndClose(self) -> None:
        log: VersionLog[int] = VersionLog()
        log.publish(0)
        log.publish(1)
        log.close()
        reader = log.subscribe()

        first = self.successResultOf(
            reader.rendered(lambda ctx, pv: pv.value, maxItems=1)
        )
        self.assertFalse(first.closed)
        second = self.successResultOf(reader.rendered(lambda ctx, pv: pv.value))
        self.assertTrue(second.closed)
        self.failureResultOf(
            reader.rendered(lambda ctx, pv: pv.value), PublisherClosed
        )

    def test_failureAfterLagDoesNotCommit(self) -> None:
        """
        If versions are evicted while a reader is mid-render, the consumed
        batch still completes and the next read surfaces the lag.
        """
        log: VersionLog[int] = VersionLog(history=3)
        log.publish(0)
        reader = log.subscribe()
        held: Deferred[str] = Deferred()
        pending = reader.rendered(lambda ctx, pv: held)
        self.assertNoResult(pending)

        for index in range(1, 5):
            log.publish(index)
        held.callback("done")

        update = self.successResultOf(pending)
        self.assertEqual(update.results, ("done",))
        # Cursor reflects what was consumed; the next read reports lag.
        self.assertEqual(reader.cursor.position, 0)
        self.failureResultOf(reader.read(), ReaderLagging)
