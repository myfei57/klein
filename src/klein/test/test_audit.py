"""
Tests for the versioned, best-effort Klein resource-rendering audit
stream (L{klein._audit}).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any, cast
from urllib.parse import parse_qs

import treq
from werkzeug.exceptions import NotFound

from twisted.internet.defer import Deferred, fail
from twisted.internet.error import ConnectionLost
from twisted.trial.unittest import SynchronousTestCase
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.test.test_web import DummyChannel

from .. import Klein, KleinRenderable
from .._audit import (
    AUDIT_FORMAT_VERSION,
    DEFAULT_SENSITIVE_KEYS,
    REDACTED,
    SOURCE,
    AuditConflictError,
    AuditCursor,
    AuditEvent,
    AuditGapError,
    AuditReader,
    AuditValidationError,
    DuplicateCursorError,
    IAuditSink,
    InvalidCursorError,
    InvalidSequenceError,
    MemoryAuditSink,
    PendingDelivery,
    RenderAuditLog,
    RenderAuditor,
    ResultCategory,
    RetryingAuditSink,
    export_ndjson,
    redact_deep,
    redact_mapping,
)
from .._resource import KleinResource
from .util import makeStub

#
# Request test double
#


class DummyRequest(Request):
    """
    A trimmed, cross-platform request double (no UNIX socket dependency)
    sufficient to drive L{KleinResource.render}.
    """

    site: Site
    finished: bool
    startedWriting: bool

    def __init__(
        self,
        path: bytes,
        method: bytes = b"GET",
        host: bytes = b"localhost",
        port: int = 80,
        isSecure: bool = False,
        headers: Mapping[bytes, Sequence[bytes]] | None = None,
    ) -> None:
        super().__init__(DummyChannel(), False)
        path, qpath = (path.split(b"?", 1) + [b""])[:2]
        self.site = cast(Site, object())
        self.gotLength(0)
        self.content = BytesIO()
        self.args = parse_qs(qpath)
        self.requestHeaders = Headers(dict(headers or {}))
        self.selfHeaders = Headers()
        self.setHost(host, port, isSecure)
        self.uri = path + (b"?" + qpath if qpath else b"")
        self.prepath = []
        self.postpath = path.split(b"/")[1:]
        self.method = method
        self.clientproto = b"HTTP/1.1"
        self._written = BytesIO()

    def write(self, data: bytes) -> None:
        self.startedWriting = True
        if not self.finished:
            self._written.write(data)

    def finish(self) -> None:
        if not self.startedWriting:
            self.write(b"")
        if not self.finished:
            self.finished = True
            self._cleanup()

    def getWrittenData(self) -> bytes:
        return self._written.getvalue()


def _render(resource: KleinResource, request: Request) -> Any:
    result = resource.render(request)
    if isinstance(result, bytes):
        return result
    if result is NOT_DONE_YET:
        return None
    raise AssertionError("unreachable")  # pragma: no cover


#
# Redaction
#


class RedactionTests(SynchronousTestCase):
    def test_redactsKnownSensitiveKeys(self) -> None:
        result = redact_mapping(
            {
                "Authorization": "Bearer abc",
                "X-Auth-Token": "t",
                "api_key": "k",
                "Cookie": "session=1",
                "set-cookie": "x=1",
                "X-Forwarded-For": "1.2.3.4",
            }
        )
        self.assertEqual(result["Authorization"], REDACTED)
        self.assertEqual(result["X-Auth-Token"], REDACTED)
        self.assertEqual(result["api_key"], REDACTED)
        self.assertEqual(result["Cookie"], REDACTED)
        self.assertEqual(result["set-cookie"], REDACTED)
        self.assertEqual(result["X-Forwarded-For"], "1.2.3.4")

    def test_normalizesUnderscoresAndCase(self) -> None:
        result = redact_mapping(
            {"CLIENT_SECRET": "s", "Access_Token": "t", "ApiKey": "k"}
        )
        self.assertEqual({v for v in result.values()}, {REDACTED})

    def test_redactsNestedStructures(self) -> None:
        result = redact_deep(
            {
                "outer": [
                    {"password": "hunter2", "ok": "yes"},
                    "plain",
                ],
                "nested": {"deep": {"proxy_authorization": "p"}},
            }
        )
        self.assertEqual(result["outer"][0]["password"], REDACTED)
        self.assertEqual(result["outer"][0]["ok"], "yes")
        self.assertEqual(result["outer"][1], "plain")
        self.assertEqual(
            result["nested"]["deep"]["proxy_authorization"], REDACTED
        )

    def test_bytesKeysSupported(self) -> None:
        result = redact_mapping({b"authorization": b"Bearer x"})
        self.assertEqual(result["authorization"], REDACTED)

    def test_customSensitiveSet(self) -> None:
        result = redact_mapping(
            {"x-internal-secret": "42", "authorization": "ok"},
            sensitive=frozenset({"x-internal-secret"}),
        )
        self.assertEqual(result["x-internal-secret"], REDACTED)
        self.assertEqual(result["authorization"], "ok")

    def test_defaultSetCoversCommonFields(self) -> None:
        for key in (
            "authorization",
            "cookie",
            "password",
            "token",
            "secret",
            "api_key",
            "client_secret",
        ):
            self.assertIn(key.replace("_", "-"), DEFAULT_SENSITIVE_KEYS)


class RequestContextTests(SynchronousTestCase):
    def test_contextIsRedacted(self) -> None:
        from .._audit import build_request_context

        request = DummyRequest(
            b"/login?token=abc&user=bob",
            headers={
                b"Authorization": [b"Bearer s3cr3t"],
                b"Cookie": [b"session=1"],
                b"X-Request-ID": [b"req-123"],
                b"Accept": [b"application/json"],
            },
        )
        context = build_request_context(request)
        headers = {
            key.lower(): value for key, value in context["headers"].items()
        }

        self.assertEqual(context["method"], "GET")
        self.assertEqual(context["path"], "/login")
        self.assertEqual(context["request_id"], "req-123")
        self.assertEqual(context["query"]["token"], REDACTED)
        self.assertEqual(context["query"]["user"], ["bob"])
        self.assertEqual(headers["authorization"], REDACTED)
        self.assertEqual(headers["cookie"], REDACTED)
        self.assertEqual(headers["x-request-id"], ["req-123"])
        self.assertEqual(headers["accept"], ["application/json"])


#
# Events and summaries
#


class EventSummaryTests(SynchronousTestCase):
    def _event(self) -> AuditEvent:
        return AuditEvent(
            sequence=1,
            source=SOURCE.RENDER_START,
            category=ResultCategory.RENDERED.value,
            summary="",
            timestamp=123.5,
            context={"method": "GET"},
            detail={"endpoint": "index"},
        )

    def test_summaryIsStableAndValid(self) -> None:
        event = self._event()
        summary = event.computed_summary()
        self.assertEqual(len(summary), 64)
        # Recomputing over identical contents is deterministic.
        self.assertEqual(event.computed_summary(), summary)
        object.__setattr__(event, "summary", summary)
        self.assertTrue(event.is_valid())
        event.validate()

    def test_tamperingIsDetected(self) -> None:
        event = self._event()
        object.__setattr__(event, "summary", event.computed_summary())
        tampered_context = dict(event.context)
        tampered_context["method"] = "POST"
        object.__setattr__(event, "context", tampered_context)
        self.assertFalse(event.is_valid())
        with self.assertRaises(AuditValidationError):
            event.validate()

    def test_toDictIsJsonSafe(self) -> None:
        event = self._event()
        payload = event.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["sequence"], 1)
        self.assertEqual(payload["version"], AUDIT_FORMAT_VERSION)


#
# The log
#


class RenderAuditLogTests(SynchronousTestCase):
    def test_emptyStream(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        self.assertEqual(len(log), 0)
        self.assertEqual(log.snapshot(), ())
        self.assertEqual(log.read(), ())
        log.validate_contiguous()
        log.validate_events()
        self.assertEqual(log.pending_sequences(), ())

    def test_stableGapFreeSequences(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        events = [
            log.record(SOURCE.RENDER_START, ResultCategory.RENDER_STARTED)
            for _ in range(5)
        ]
        self.assertEqual([e.sequence for e in events], [1, 2, 3, 4, 5])
        self.assertTrue(all(e.is_valid() for e in events))

    def test_storageBoundaryRedacts(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        event = log.record(
            SOURCE.RESPONSE,
            ResultCategory.RENDERED,
            context={"headers": {"authorization": "Bearer topsecret"}},
            detail={"token": "t"},
        )
        self.assertEqual(event.context["headers"]["authorization"], REDACTED)
        self.assertEqual(event.detail["token"], REDACTED)
        self.assertNotIn(b"topsecret", export_ndjson(log))

    def test_reservedCommitsMayArriveOutOfOrder(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        s1 = log.reserve()
        s2 = log.reserve()
        s3 = log.reserve()
        self.assertEqual(log.pending_sequences(), (1, 2, 3))
        error = self.assertRaises(AuditGapError, log.validate_contiguous)
        self.assertEqual(error.missing, (1, 2, 3))

        e3 = log.commit(s3, SOURCE.RESPONSE, ResultCategory.RENDERED)
        e1 = log.commit(s1, SOURCE.RENDER_START, ResultCategory.RENDER_STARTED)
        e2 = log.commit(s2, SOURCE.RESPONSE, ResultCategory.RENDERED)

        # Snapshot restores total order regardless of commit order.
        self.assertEqual([e.sequence for e in log.snapshot()], [1, 2, 3])
        log.validate_contiguous()
        self.assertEqual(log.snapshot(), (e1, e2, e3))

    def test_commitIsIdempotentButConflictsOnChange(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        s1 = log.reserve()
        e1 = log.commit(s1, SOURCE.RENDER_START, ResultCategory.RENDER_STARTED)
        again = log.commit(
            s1, SOURCE.RENDER_START, ResultCategory.RENDER_STARTED
        )
        self.assertIs(again, e1)
        with self.assertRaises(AuditConflictError):
            log.commit(s1, SOURCE.RESPONSE, ResultCategory.RENDERED)

    def test_commitRejectsUnknownSequence(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        with self.assertRaises(InvalidSequenceError):
            log.commit(42, SOURCE.RENDER_START, ResultCategory.RENDER_STARTED)

    def test_readRespectsAfterAndLimit(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        for _ in range(5):
            log.record(SOURCE.RENDER_START, ResultCategory.RENDER_STARTED)
        self.assertEqual(
            [e.sequence for e in log.read(after=2, limit=2)], [3, 4]
        )
        self.assertEqual([e.sequence for e in log.read(after=4)], [5])

    def test_concurrentSnapshotsAreConsistentAndComplete(self) -> None:
        thread_count = 8
        observer_count = 2
        per_thread = 50
        log = RenderAuditLog()
        start = threading.Barrier(thread_count + observer_count)
        observed_snapshots: list[tuple[int, ...]] = []
        snapshot_errors: list[AssertionError] = []
        observers: list[threading.Thread] = []

        def observe() -> None:
            start.wait()
            for _ in range(20):
                snapshot = tuple(e.sequence for e in log.snapshot())
                # Every concurrent snapshot must be sorted and unique.
                if snapshot != tuple(sorted(set(snapshot))):
                    snapshot_errors.append(
                        AssertionError(
                            f"unsorted/duplicate snapshot: {snapshot}"
                        )
                    )
                observed_snapshots.append(snapshot)

        def write() -> None:
            start.wait()
            for _ in range(per_thread):
                log.record(
                    SOURCE.RENDER_START,
                    ResultCategory.RENDER_STARTED,
                    context={"authorization": "x"},
                )

        for _ in range(observer_count):
            thread = threading.Thread(target=observe)
            thread.start()
            observers.append(thread)

        writers = [threading.Thread(target=write) for _ in range(thread_count)]
        for thread in writers:
            thread.start()
        for thread in writers:
            thread.join()
        for thread in observers:
            thread.join()

        self.assertEqual(snapshot_errors, [])
        total = thread_count * per_thread
        self.assertEqual(len(log), total)
        self.assertEqual(
            [e.sequence for e in log.snapshot()],
            list(range(1, total + 1)),
        )
        log.validate_contiguous()
        log.validate_events()
        self.assertTrue(observed_snapshots)
        # Sensitive data never escapes under concurrency either.
        self.assertNotIn(b'"x"', export_ndjson(log))


#
# Cursors
#


class CursorTests(SynchronousTestCase):
    def test_roundtrip(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        token = log.cursor_for(7)
        cursor = log.decode_cursor(token)
        self.assertEqual(cursor.last_sequence, 7)
        self.assertEqual(cursor.version, AUDIT_FORMAT_VERSION)

    def test_tamperedTokenRejected(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        token = log.cursor_for(7)
        body, signature = token.split(".", 1)
        flipped = signature[:-1] + ("0" if signature[-1] != "0" else "1")
        with self.assertRaises(InvalidCursorError):
            log.decode_cursor(f"{body}.{flipped}")
        with self.assertRaises(InvalidCursorError):
            log.decode_cursor("not-a-cursor")
        with self.assertRaises(InvalidCursorError):
            AuditCursor.from_token(
                token, "wrong-anchor", expected_version=AUDIT_FORMAT_VERSION
            )

    def test_staleVersionRejectedWithRecoveryHint(self) -> None:
        v1 = RenderAuditLog(
            version=1, clock=lambda: 0.0, anchor="stream-identity"
        )
        v2 = RenderAuditLog(
            version=2, clock=lambda: 0.0, anchor="stream-identity"
        )
        token = v2.cursor_for(3)
        error = self.assertRaises(InvalidCursorError, v1.decode_cursor, token)
        self.assertEqual(error.cursor_version, 2)
        self.assertEqual(error.stream_version, 1)


#
# Pagination
#


class AuditReaderTests(SynchronousTestCase):
    def _populated(self, count: int) -> RenderAuditLog:
        log = RenderAuditLog(clock=lambda: 0.0)
        for _ in range(count):
            log.record(SOURCE.RENDER_START, ResultCategory.RENDER_STARTED)
        return log

    def test_emptyFlowPagination(self) -> None:
        reader = AuditReader(RenderAuditLog(clock=lambda: 0.0))
        page = reader.read_page(None, limit=10)
        self.assertEqual(page.events, ())
        self.assertIsNone(page.next_cursor)
        # Restarting an empty read remains legal.
        again = reader.read_page(None, limit=10)
        self.assertEqual(again.events, ())

    def test_walkAllPagesInOrder(self) -> None:
        log = self._populated(5)
        reader = AuditReader(log)
        seen: list[int] = []
        cursor: str | None = None
        pages = 0
        while True:
            page = reader.read_page(cursor, limit=2)
            pages += 1
            seen.extend(e.sequence for e in page.events)
            cursor = page.next_cursor
            if cursor is None:
                break
        self.assertEqual(seen, [1, 2, 3, 4, 5])
        self.assertEqual(pages, 3)

    def test_exactPageMultipleEndsCleanly(self) -> None:
        log = self._populated(4)
        reader = AuditReader(log)
        first = reader.read_page(None, limit=2)
        self.assertIsNotNone(first.next_cursor)
        second = reader.read_page(first.next_cursor, limit=2)
        self.assertIsNotNone(second.next_cursor)
        third = reader.read_page(second.next_cursor, limit=2)
        self.assertEqual(third.events, ())
        self.assertIsNone(third.next_cursor)

    def test_duplicateCursorDetected(self) -> None:
        log = self._populated(4)
        reader = AuditReader(log)
        first = reader.read_page(None, limit=2)
        assert first.next_cursor is not None
        reader.read_page(first.next_cursor, limit=2)
        with self.assertRaises(DuplicateCursorError):
            reader.read_page(first.next_cursor, limit=2)
        # Resetting the reader recovers pagination.
        reader.reset()
        recovered = reader.read_page(first.next_cursor, limit=2)
        self.assertEqual([e.sequence for e in recovered.events], [3, 4])

    def test_staleCursorIsRecoverableByRestarting(self) -> None:
        v1 = self._populated(2)
        v2 = RenderAuditLog(version=2, clock=lambda: 0.0, anchor=v1._anchor)
        v2.record(SOURCE.RENDER_START, ResultCategory.RENDER_STARTED)
        reader = AuditReader(v1)
        stale = v2.cursor_for(1)
        with self.assertRaises(InvalidCursorError):
            reader.read_page(stale)
        # The start cursor always works, so order can be rebuilt.
        restarted = reader.read_page(None)
        self.assertEqual([e.sequence for e in restarted.events], [1, 2])

    def test_invalidLimitRejected(self) -> None:
        reader = AuditReader(self._populated(1))
        with self.assertRaises(ValueError):
            reader.read_page(None, limit=0)


#
# Export
#


class ExportTests(SynchronousTestCase):
    def test_exportEmpty(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        self.assertEqual(export_ndjson(log), b"")

    def test_exportIsReadOnlyNdjson(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        log.record(
            SOURCE.RENDER_START,
            ResultCategory.RENDER_STARTED,
            context={"headers": {"authorization": "Bearer secret"}},
        )
        log.record(SOURCE.RESPONSE, ResultCategory.RENDERED)
        before = log.snapshot()

        blob = export_ndjson(log)

        after = log.snapshot()
        self.assertEqual(before, after)
        self.assertEqual(len(log), 2)

        lines = blob.decode("utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        records = [json.loads(line) for line in lines]
        self.assertEqual([r["sequence"] for r in records], [1, 2])
        for record in records:
            AuditEvent(
                sequence=record["sequence"],
                source=record["source"],
                category=record["category"],
                summary=record["summary"],
                timestamp=record["timestamp"],
                context=record["context"],
                detail=record["detail"],
                version=record["version"],
            ).validate()
        self.assertIn(REDACTED, blob.decode("utf-8"))
        self.assertNotIn("Bearer secret", blob.decode("utf-8"))


#
# Sinks and retry
#


class ExplodingSink:
    def __init__(self, *, asynchronous: bool = False) -> None:
        self.asynchronous = asynchronous
        self.calls = 0

    def deliver(self, event: AuditEvent) -> Deferred[None]:
        self.calls += 1
        if self.asynchronous:
            return fail(RuntimeError("async boom"))
        raise RuntimeError("sync boom")


class SwitchableSink:
    impl: IAuditSink

    def __init__(self, initial: IAuditSink) -> None:
        self.impl = initial

    def deliver(self, event: AuditEvent) -> object:
        return self.impl.deliver(event)


class RetryingSinkTests(SynchronousTestCase):
    def _event(self, sequence: int = 1) -> AuditEvent:
        return AuditEvent(
            sequence=sequence,
            source=SOURCE.RESPONSE,
            category=ResultCategory.RENDERED.value,
            summary="0",
            timestamp=1.0,
            context={"path": "/boom", "method": "GET"},
        )

    def test_failureQueuesWithContextAndNeverErrbacks(self) -> None:
        clock = iter([10.0, 11.0])
        sink = RetryingAuditSink(ExplodingSink(), clock=lambda: next(clock))
        d = sink.deliver(self._event())
        self.assertIs(self.successResultOf(d), None)
        pending = sink.pending
        self.assertEqual(len(pending), 1)
        entry = pending[0]
        self.assertIsInstance(entry, PendingDelivery)
        self.assertEqual(entry.attempts, 1)
        # Request context is preserved with the queued event.
        self.assertEqual(entry.event.context["path"], "/boom")
        self.assertEqual(entry.event.context["method"], "GET")
        self.assertEqual(entry.first_failed_at, 10.0)
        self.assertEqual(entry.last_failed_at, 10.0)
        self.flushLoggedErrors(RuntimeError)

    def test_asyncFailureIsQueued(self) -> None:
        sink = RetryingAuditSink(ExplodingSink(asynchronous=True))
        d = sink.deliver(self._event())
        self.assertIs(self.successResultOf(d), None)
        self.assertEqual(len(sink.pending), 1)
        self.assertEqual(sink.pending[0].attempts, 1)
        self.flushLoggedErrors(RuntimeError)

    def test_repeatedFailureIncrementsAttempts(self) -> None:
        sink = RetryingAuditSink(ExplodingSink())
        event = self._event()
        sink.deliver(event)
        sink.deliver(event)
        self.assertEqual(sink.pending[0].attempts, 2)
        still_pending = self.successResultOf(sink.retry_pending())
        self.assertEqual(len(still_pending), 1)
        self.assertEqual(still_pending[0].attempts, 3)
        self.flushLoggedErrors(RuntimeError)

    def test_retryDeliversPreservingContextAndOrder(self) -> None:
        memory = MemoryAuditSink()
        switch = SwitchableSink(ExplodingSink())
        sink = RetryingAuditSink(switch)
        first = self._event(1)
        second = self._event(2)
        sink.deliver(first)
        sink.deliver(second)
        self.assertEqual(len(sink.pending), 2)
        self.flushLoggedErrors(RuntimeError)

        switch.impl = memory
        remaining = self.successResultOf(sink.retry_pending())

        self.assertEqual(remaining, ())
        delivered = memory.events
        self.assertEqual([event.sequence for event in delivered], [1, 2])
        self.assertEqual(delivered[0].context["path"], "/boom")

    def test_successfulDeliveryIsNotPending(self) -> None:
        sink = RetryingAuditSink(MemoryAuditSink())
        d = sink.deliver(self._event())
        self.assertIs(self.successResultOf(d), None)
        self.assertEqual(sink.pending, ())


class RenderAuditorTests(SynchronousTestCase):
    def test_recordForwardsToSink(self) -> None:
        memory = MemoryAuditSink()
        auditor = RenderAuditor(sink=memory, clock=lambda: 0.0)
        event = auditor.record(
            SOURCE.RENDER_START, ResultCategory.RENDER_STARTED
        )
        self.assertIsNotNone(event)
        self.assertEqual(memory.events, (event,))

    def test_sinkRaisingDoesNotBreakRecording(self) -> None:
        class RaisingSink:
            def deliver(self, event: AuditEvent) -> None:
                raise RuntimeError("sink down")

        auditor = RenderAuditor(sink=RaisingSink(), clock=lambda: 0.0)
        event = auditor.record(
            SOURCE.RENDER_START, ResultCategory.RENDER_STARTED
        )
        self.assertIsNotNone(event)
        self.assertEqual(len(auditor.log), 1)
        self.flushLoggedErrors(RuntimeError)

    def test_reservedSequenceSurvivesDelayedCommit(self) -> None:
        auditor = RenderAuditor(clock=lambda: 0.0)
        reserved = auditor.log.reserve()
        event = auditor.record(
            SOURCE.RESPONSE,
            ResultCategory.RENDERED,
            sequence=reserved,
        )
        assert event is not None
        self.assertEqual(event.sequence, 1)


#
# Integration with KleinResource
#


class KleinResourceAuditTests(SynchronousTestCase):
    def setUp(self) -> None:
        self.app = Klein()
        self.auditor = RenderAuditor(sink=MemoryAuditSink())
        self.resource = KleinResource(self.app, self.auditor)

    def _sink(self) -> MemoryAuditSink:
        sink = self.auditor._sink
        return cast(MemoryAuditSink, sink)

    def test_successIsAuditedWithContext(self) -> None:
        @self.app.route("/hello")
        def hello(request: IRequest) -> KleinRenderable:
            return b"world"

        stub = makeStub(self.resource)
        response = self.successResultOf(
            stub.request(
                "GET",
                "https://example.com/hello",
                headers=Headers({b"Authorization": [b"Bearer s3cr3t"]}),
            )
        )
        self.assertEqual(response.code, 200)
        body = self.successResultOf(treq.content(response))
        self.assertEqual(body, b"world")

        events = self.auditor.log.snapshot()
        self.assertEqual(
            [event.category for event in events],
            [
                ResultCategory.RENDER_STARTED.value,
                ResultCategory.RENDERED.value,
            ],
        )
        started, rendered = events
        self.assertEqual(started.source, SOURCE.RENDER_START)
        self.assertEqual(rendered.source, SOURCE.RESPONSE)
        self.assertEqual(rendered.detail["endpoint"], "hello")
        self.assertEqual(rendered.detail["response_code"], 200)
        self.assertFalse(rendered.detail["delegated"])
        self.assertEqual(started.context["method"], "GET")
        self.assertEqual(started.context["path"], "/hello")
        headers = {
            key.lower(): value
            for key, value in started.context["headers"].items()
        }
        self.assertEqual(headers["authorization"], REDACTED)
        self.assertEqual(
            [event.sequence for event in self._sink().events], [1, 2]
        )
        self.assertNotIn(b"s3cr3t", export_ndjson(self.auditor.log))

    def test_querySecretsNeverReachStoredEvents(self) -> None:
        @self.app.route("/login")
        def login(request: IRequest) -> KleinRenderable:
            return b"ok"

        stub = makeStub(self.resource)
        response = self.successResultOf(
            stub.request(
                "GET",
                "https://example.com/login?token=topsecret&user=amy",
            )
        )
        self.assertEqual(response.code, 200)
        blob = export_ndjson(self.auditor.log)
        self.assertNotIn(b"topsecret", blob)
        self.assertIn(REDACTED.encode("utf-8"), blob)
        self.assertIn(b"amy", blob)

    def test_handledErrorIsAudited(self) -> None:
        @self.app.route("/boom")
        def boom(request: IRequest) -> KleinRenderable:
            raise ValueError("kaboom")

        @self.app.handle_errors(ValueError)
        def handled(request: IRequest, failure: object) -> KleinRenderable:
            return b"recovered"

        stub = makeStub(self.resource)
        response = self.successResultOf(
            stub.request("GET", "https://example.com/boom")
        )
        self.assertEqual(response.code, 200)
        self.assertEqual(
            self.successResultOf(treq.content(response)), b"recovered"
        )
        categories = [event.category for event in self.auditor.log.snapshot()]
        self.assertEqual(
            categories,
            [
                ResultCategory.RENDER_STARTED.value,
                ResultCategory.ERROR_HANDLED.value,
            ],
        )
        terminal = self.auditor.log.snapshot()[-1]
        self.assertEqual(terminal.detail["error"], "ValueError")
        self.assertEqual(terminal.detail["endpoint"], "boom")

    def test_unhandledErrorIsAudited(self) -> None:
        @self.app.route("/fatal")
        def fatal(request: IRequest) -> KleinRenderable:
            raise ValueError("uncatchable")

        stub = makeStub(self.resource)
        response = self.successResultOf(
            stub.request("GET", "https://example.com/fatal")
        )
        self.assertEqual(response.code, 500)
        categories = [event.category for event in self.auditor.log.snapshot()]
        self.assertEqual(
            categories,
            [
                ResultCategory.RENDER_STARTED.value,
                ResultCategory.ERROR_UNHANDLED.value,
            ],
        )
        self.flushLoggedErrors(ValueError)

    def test_httpExceptionIsAuditedAsHandled(self) -> None:
        stub = makeStub(self.resource)
        response = self.successResultOf(
            stub.request("GET", "https://example.com/missing")
        )
        self.assertEqual(response.code, 404)
        terminal = self.auditor.log.snapshot()[-1]
        self.assertEqual(terminal.category, ResultCategory.ERROR_HANDLED.value)
        self.assertEqual(terminal.detail["response_code"], 404)
        self.assertEqual(terminal.detail["error"], NotFound.__name__)

    def test_urlDecodeFailureIsAudited(self) -> None:
        request = DummyRequest(b"/\xff")
        result = _render(self.resource, request)
        self.assertEqual(result, b"Non-UTF-8 encoding in URL.")
        self.assertEqual(request.code, 400)
        events = self.auditor.log.snapshot()
        self.assertEqual(
            [event.category for event in events],
            [
                ResultCategory.RENDER_STARTED.value,
                ResultCategory.URL_DECODE_FAILED.value,
            ],
        )
        self.assertIn("PATH_INFO", events[1].detail["components"])
        self.flushLoggedErrors(UnicodeDecodeError)

    def test_interruptedRenderingIsAudited(self) -> None:
        waiting: Deferred[bytes] = Deferred()

        @self.app.route("/wait")
        def wait(request: IRequest) -> KleinRenderable:
            return waiting

        request = DummyRequest(b"/wait")
        _render(self.resource, request)
        self.assertFalse(request.finished)
        request.connectionLost(ConnectionLost())
        # processing_failed deliberately suppresses the cancellation on
        # a lost connection, so the observable signal is the audit
        # category rather than a failure on the handler deferred.
        self.assertTrue(waiting.called)

        categories = [event.category for event in self.auditor.log.snapshot()]
        self.assertEqual(
            categories,
            [
                ResultCategory.RENDER_STARTED.value,
                ResultCategory.RENDER_INTERRUPTED.value,
            ],
        )
        interrupted = self.auditor.log.snapshot()[-1]
        self.assertTrue(interrupted.detail["cancelled"])

    def test_asyncSinkFailurePreservesContextAndIsRetryable(
        self,
    ) -> None:
        memory = MemoryAuditSink()
        switch = SwitchableSink(ExplodingSink(asynchronous=True))
        retrying = RetryingAuditSink(switch)
        auditor = RenderAuditor(sink=retrying, clock=lambda: 0.0)
        resource = KleinResource(self.app, auditor)

        @self.app.route("/boom")
        def boom(request: IRequest) -> KleinRenderable:
            return b"still rendered"

        stub = makeStub(resource)
        response = self.successResultOf(
            stub.request("GET", "https://example.com/boom")
        )
        self.assertEqual(response.code, 200)
        self.assertEqual(
            self.successResultOf(treq.content(response)),
            b"still rendered",
        )

        pending = retrying.pending
        self.assertEqual(len(pending), 2)
        contexts = {entry.event.context["path"] for entry in pending}
        self.assertEqual(contexts, {"/boom"})
        self.flushLoggedErrors(RuntimeError)

        switch.impl = memory
        remaining = self.successResultOf(retrying.retry_pending())
        self.assertEqual(remaining, ())
        self.assertEqual(len(memory.events), 2)

    def test_brokenAuditorCannotChangeMainFlow(self) -> None:
        class ExplodingAuditor:
            def record(
                self,
                source: str,
                category: str,
                *,
                context: Mapping[str, object] | None = None,
                detail: Mapping[str, object] | None = None,
                sequence: int | None = None,
            ) -> None:
                raise RuntimeError("auditor down")

        @self.app.route("/")
        def index(request: IRequest) -> KleinRenderable:
            return b"unchanged"

        resource = KleinResource(self.app, ExplodingAuditor())
        stub = makeStub(resource)
        response = self.successResultOf(
            stub.request("GET", "https://example.com/")
        )
        self.assertEqual(response.code, 200)
        self.assertEqual(
            self.successResultOf(treq.content(response)), b"unchanged"
        )
        self.flushLoggedErrors(RuntimeError)

    def test_exportDoesNotChangeMainFlow(self) -> None:
        @self.app.route("/")
        def index(request: IRequest) -> KleinRenderable:
            return b"same"

        resource = KleinResource(self.app, self.auditor)
        stub = makeStub(resource)
        for _ in range(2):
            response = self.successResultOf(
                stub.request("GET", "https://example.com/")
            )
            self.assertEqual(response.code, 200)
            self.assertEqual(
                self.successResultOf(treq.content(response)), b"same"
            )
            export_ndjson(self.auditor.log)
        self.assertEqual(len(self.auditor.log), 4)

    def test_auditingIsOptIn(self) -> None:
        plain = KleinResource(self.app)
        self.assertIsNone(plain._auditor)
        # Default resources of the same app remain equal (the auditor
        # attribute defaults to None on both sides).
        self.assertEqual(plain, KleinResource(self.app))

    def test_readerPagesRealEvents(self) -> None:
        @self.app.route("/a")
        def a(request: IRequest) -> KleinRenderable:
            return b"a"

        @self.app.route("/b")
        def b(request: IRequest) -> KleinRenderable:
            return b"b"

        stub = makeStub(self.resource)
        for path in ("/a", "/b"):
            response = self.successResultOf(
                stub.request("GET", f"https://example.com{path}")
            )
            self.assertEqual(response.code, 200)

        reader = self.auditor.new_reader(page_size=2)
        first = reader.read_page(None)
        self.assertEqual(len(first.events), 2)
        self.assertIsNotNone(first.next_cursor)
        second = reader.read_page(first.next_cursor)
        self.assertEqual(len(second.events), 2)
        self.assertIsNotNone(second.next_cursor)
        # Exact-fit page is followed by an empty terminal page.
        third = reader.read_page(second.next_cursor)
        self.assertEqual(third.events, ())
        self.assertIsNone(third.next_cursor)
        with self.assertRaises(DuplicateCursorError):
            reader.read_page(first.next_cursor)


#
# Order recovery across delayed (asynchronous) commits
#


class OrderRecoveryTests(SynchronousTestCase):
    def test_delayedCommitsRecoverTotalOrder(self) -> None:
        log = RenderAuditLog()
        sequences = [log.reserve() for _ in range(4)]

        # Simulate commits completing in reverse (as a retry of an
        # earlier, asynchronously failed delivery might).
        for sequence in reversed(sequences):
            log.commit(
                sequence,
                SOURCE.RESPONSE,
                ResultCategory.RENDERED,
                context={"path": f"/{sequence}"},
            )

        snapshot = log.snapshot()
        self.assertEqual([event.sequence for event in snapshot], [1, 2, 3, 4])
        log.validate_contiguous()
        log.validate_events()
        self.assertEqual(
            [event.context["path"] for event in snapshot],
            ["/1", "/2", "/3", "/4"],
        )

    def test_partialSnapshotThenFillRecovers(self) -> None:
        log = RenderAuditLog(clock=lambda: 0.0)
        s1 = log.reserve()
        s2 = log.reserve()
        log.commit(s2, SOURCE.RESPONSE, ResultCategory.RENDERED)
        # Only sequence 2 visible while 1 is still in flight.
        self.assertEqual([event.sequence for event in log.snapshot()], [2])
        self.assertRaises(AuditGapError, log.validate_contiguous)
        log.commit(s1, SOURCE.RENDER_START, ResultCategory.RENDER_STARTED)
        self.assertEqual([event.sequence for event in log.snapshot()], [1, 2])
        log.validate_contiguous()
