# -*- test-case-name: klein.test.test_audit -*-
"""
A versioned, best-effort audit trail for L{KleinResource} rendering.

The audit trail records, for every resource-rendering attempt:

    - a stable, gap-free event sequence number,

    - the source location (rendering pipeline stage) that produced the
      event,

    - a result category (started, rendered, handled error, unhandled
      error, interrupted, URL decode failure),

    - a validation summary (SHA-256 over the event's canonical
      serialization) that lets readers detect tampering,

    - a redacted snapshot of the request context (sensitive headers and
    query fields are masked before the event is ever stored).

Events live in an in-memory L{RenderAuditLog}.  They may be paged
through with opaque, integrity-protected cursors (L{AuditReader});
re-using a cursor is rejected, and a cursor from a different stream
version is reported as stale.  Events may be exported read-only via
L{export_ndjson}; exporting never mutates the log or any request state.

Auditing is strictly best-effort.  L{RenderAuditor.record} swallows
every failure, and asynchronous delivery failures are captured by
L{RetryingAuditSink}, which preserves the full event (including its
request context) for later retry.  Nothing in this module can change
the outcome of request rendering.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Optional, Protocol
from urllib.parse import parse_qsl

import attr

from twisted.internet.defer import Deferred, maybeDeferred, succeed
from twisted.python import log
from twisted.web.iweb import IRequest

#
# Versioning and result categories
#

AUDIT_FORMAT_VERSION: int = 1
"""
Version of the on-the-wire audit record format.  Cursors embed this
number, so a reader holding a cursor from an incompatible stream version
fails fast with L{StaleCursorError} and can recover by restarting from
the beginning.
"""

DEFAULT_PAGE_SIZE: int = 100

REDACTED: str = "***REDACTED***"
"""
Replacement value for sensitive fields.
"""


class ResultCategory(str, Enum):
    """
    The coarse result category of an audited rendering event.
    """

    RENDER_STARTED = "render-started"
    RENDERED = "rendered"
    URL_DECODE_FAILED = "url-decode-failed"
    ERROR_HANDLED = "error-handled"
    ERROR_UNHANDLED = "error-unhandled"
    RENDER_INTERRUPTED = "render-interrupted"


class SOURCE:
    """
    Stable source locations within the Klein rendering pipeline.

    These identify the pipeline stage that emitted an event.  They are
    strings (rather than line numbers) so they remain stable as the
    implementation evolves.
    """

    RENDER_START = "klein._resource.KleinResource.render#start"
    URL_DECODE = "klein._resource.KleinResource.render#url-decode"
    RESPONSE = "klein._resource.KleinResource.render#write-response"
    PROCESSING_FAILED = "klein._resource.KleinResource.render#processing-failed"


#
# Errors
#


class AuditError(Exception):
    """
    Base class for audit-stream errors.
    """


class AuditValidationError(AuditError):
    """
    An event's validation summary does not match its contents.
    """


class AuditGapError(AuditError):
    """
    The committed stream is missing one or more sequence numbers.

    @ivar missing: The sequence numbers that have not been committed.
    """

    def __init__(self, missing: Sequence[int]) -> None:
        self.missing = tuple(missing)
        super().__init__(f"missing audit event sequences: {self.missing}")


class AuditConflictError(AuditError):
    """
    A sequence number was committed twice with different contents.
    """


class InvalidSequenceError(AuditError):
    """
    A commit referenced a sequence number that was never reserved.
    """


class InvalidCursorError(AuditError):
    """
    A cursor token was malformed, forged, or tampered with.
    """


class StaleCursorError(InvalidCursorError):
    """
    A cursor belongs to a different stream format version.

    Recovery consists of discarding the cursor and restarting pagination
    from the beginning (C{cursor=None}).

    @ivar cursor_version: The format version embedded in the cursor.
    @ivar stream_version: The format version of the current stream.
    """

    def __init__(self, cursor_version: int, stream_version: int) -> None:
        self.cursor_version = cursor_version
        self.stream_version = stream_version
        super().__init__(
            f"cursor version {cursor_version} does not match stream "
            f"version {stream_version}; restart pagination from the "
            f"beginning to recover"
        )


class DuplicateCursorError(InvalidCursorError):
    """
    A cursor that had already been consumed was presented again.

    @ivar cursor: The offending cursor token.
    """

    def __init__(self, cursor: str) -> None:
        self.cursor = cursor
        super().__init__(f"audit cursor has already been consumed: {cursor!r}")


#
# Redaction
#

DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        # credentials / sessions
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-auth-token",
        "x-api-key",
        # common secret-bearing form/query field names
        "password",
        "passwd",
        "secret",
        "token",
        "access-token",
        "refresh-token",
        "id-token",
        "api-key",
        "apikey",
        "client-secret",
        "code",
    }
)


def _normalize_sensitive_key(key: str) -> str:
    return key.strip().lower().replace("_", "-")


def _as_text(key: object) -> Optional[str]:
    if isinstance(key, str):
        return key
    if isinstance(key, bytes):
        return key.decode("latin1")
    return None


def redact_mapping(
    values: Mapping[Any, Any],
    sensitive: frozenset[str] = DEFAULT_SENSITIVE_KEYS,
) -> dict[str, Any]:
    """
    Return a copy of C{values} with sensitive entries masked.

    Matching is case-insensitive and treats C{_} and C{-} as equivalent,
    so C{X-Auth-Token}, C{x_auth_token} and C{API-Key} are all redacted.
    Values of matching keys are replaced with L{REDACTED} regardless of
    their type; non-mapping, non-sequence values are preserved.
    """
    redacted: dict[str, Any] = {}
    normalized = {_normalize_sensitive_key(k) for k in sensitive}
    for key, value in values.items():
        text_key = _as_text(key)
        if text_key is None:
            # Non-string keys carry no useful schema information; keep
            # redacting their contents defensively.
            redacted[str(key)] = REDACTED
            continue
        if _normalize_sensitive_key(text_key) in normalized:
            redacted[text_key] = REDACTED
        else:
            redacted[text_key] = redact_deep(value, sensitive)
    return redacted


def redact_deep(
    value: Any,
    sensitive: frozenset[str] = DEFAULT_SENSITIVE_KEYS,
) -> Any:
    """
    Recursively redact sensitive keys in mappings and sequences.

    Bytes are preserved as bytes (they may be opaque identifiers); only
    dictionary entries whose key is sensitive are masked.
    """
    if isinstance(value, Mapping):
        return redact_mapping(value, sensitive)
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return [redact_deep(item, sensitive) for item in value]
    return value


def _request_headers(request: IRequest) -> dict[str, list[str]]:
    headers = request.requestHeaders
    raw = headers.getAllRawHeaders()
    result: dict[str, list[str]] = {}
    for name, values in raw:
        name_text = (
            name.decode("latin1") if isinstance(name, bytes) else str(name)
        )
        result[name_text] = [
            value.decode("latin1") if isinstance(value, bytes) else str(value)
            for value in values
        ]
    return result


def build_request_context(
    request: IRequest,
    *,
    sensitive: frozenset[str] = DEFAULT_SENSITIVE_KEYS,
) -> dict[str, Any]:
    """
    Build a JSON-safe, redacted snapshot of a request's context.

    The context captures the method, path (query string split out and
    separately redacted), an inbound C{X-Request-ID} when present, and
    all request headers with sensitive ones masked.  This function never
    raises on malformed input: undecodable path bytes are replaced
    rather than propagated.
    """
    method = request.method
    if isinstance(method, bytes):
        method_text = method.decode("latin1")
    else:
        method_text = str(method)

    raw_uri = getattr(request, "uri", b"/")
    if isinstance(raw_uri, str):
        raw_uri = raw_uri.encode("utf-8", "replace")
    raw_path, sep, raw_query = raw_uri.partition(b"?")
    path_text = raw_path.decode("utf-8", "replace")

    query: dict[str, list[str]] = {}
    if sep:
        for name, value in parse_qsl(
            raw_query.decode("latin1"), keep_blank_values=True
        ):
            query.setdefault(name, []).append(value)

    request_id = request.getHeader("x-request-id")
    if isinstance(request_id, bytes):
        request_id = request_id.decode("latin1")

    context = {
        "method": method_text,
        "path": path_text,
        "query": redact_mapping(query, sensitive),
        "request_id": request_id,
        "headers": redact_mapping(_request_headers(request), sensitive),
    }
    return context


#
# Events and validation summaries
#


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(_as_text(key) if _as_text(key) is not None else key): (
                _json_safe(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence):
        return [_json_safe(item) for item in value]
    return repr(value)


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _as_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """
    Typed attrs converter: store an owned mutable copy of a mapping
    while keeping the attribute's declared L{Mapping} type.
    """
    return dict(value)


@attr.s(frozen=True, slots=True)
class AuditEvent:
    """
    One immutable, versioned audit record.

    @ivar sequence: Stable, gap-free event number within the stream.
    @ivar source: The source location that produced the event.
    @ivar category: A L{ResultCategory} value.
    @ivar summary: Hex SHA-256 of the event's canonical serialization.
    @ivar timestamp: Seconds since the epoch, from the log's clock.
    @ivar context: Redacted request context.
    @ivar detail: Redacted, JSON-safe extra information.
    @ivar version: Audit record format version.
    """

    sequence: int = attr.ib()
    source: str = attr.ib()
    category: str = attr.ib()
    summary: str = attr.ib()
    timestamp: float = attr.ib()
    context: Mapping[str, Any] = attr.ib(factory=dict, converter=_as_mapping)
    detail: Mapping[str, Any] = attr.ib(factory=dict, converter=_as_mapping)
    version: int = attr.ib(default=AUDIT_FORMAT_VERSION)

    def _summary_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sequence": self.sequence,
            "source": self.source,
            "category": self.category,
            "timestamp": self.timestamp,
            "context": self.context,
            "detail": self.detail,
        }

    def computed_summary(self) -> str:
        """
        Recompute the validation summary from the event contents.
        """
        return hashlib.sha256(_canonical(self._summary_payload())).hexdigest()

    def is_valid(self) -> bool:
        """
        Whether the stored summary matches the current contents.
        """
        return hmac.compare_digest(self.summary, self.computed_summary())

    def validate(self) -> None:
        """
        Raise L{AuditValidationError} if the summary does not match.
        """
        if not self.is_valid():
            raise AuditValidationError(
                f"audit event {self.sequence} failed summary validation"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-safe dict representation of the event.
        """
        return {
            "version": self.version,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "source": self.source,
            "category": self.category,
            "context": _json_safe(self.context),
            "detail": _json_safe(self.detail),
            "summary": self.summary,
        }


#
# Cursors
#


@attr.s(frozen=True, slots=True)
class AuditCursor:
    """
    An opaque, integrity-protected position in an audit stream.
    """

    version: int = attr.ib()
    last_sequence: int = attr.ib()

    def to_token(self, anchor: str) -> str:
        body = (
            urlsafe_b64encode(
                _canonical(
                    {
                        "version": self.version,
                        "last_sequence": self.last_sequence,
                    }
                )
            )
            .rstrip(b"=")
            .decode("ascii")
        )
        signature = hmac.new(
            anchor.encode("ascii"), body.encode("ascii"), hashlib.sha256
        ).hexdigest()
        return f"{body}.{signature}"

    @classmethod
    def from_token(
        cls,
        token: str,
        anchor: str,
        *,
        expected_version: int,
    ) -> "AuditCursor":
        """
        Decode and verify a cursor token.

        @raise InvalidCursorError: The token is malformed or its
            signature does not match the stream.
        @raise StaleCursorError: The token's format version differs from
            C{expected_version}.
        """
        try:
            body, signature = token.split(".", 1)
        except (ValueError, AttributeError) as e:
            raise InvalidCursorError(f"malformed cursor: {token!r}") from e

        expected_signature = hmac.new(
            anchor.encode("ascii"), body.encode("ascii"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            raise InvalidCursorError("cursor signature mismatch")

        padding = "=" * (-len(body) % 4)
        try:
            raw = urlsafe_b64decode(body + padding)
            payload = json.loads(raw.decode("ascii"))
            version = int(payload["version"])
            last_sequence = int(payload["last_sequence"])
        except Exception as e:
            raise InvalidCursorError(
                f"cursor payload undecodable: {token!r}"
            ) from e

        if last_sequence < 0:
            raise InvalidCursorError("cursor position may not be negative")
        if version != expected_version:
            raise StaleCursorError(version, expected_version)
        return cls(version=version, last_sequence=last_sequence)


#
# The log
#


class RenderAuditLog:
    """
    A thread-safe, versioned store of L{AuditEvent}s.

    Sequence numbers are stable: they are reserved before an event is
    committed, so an event whose commit is delayed (for example by an
    asynchronous delivery failure and retry) keeps its position in the
    total order.  Commits may therefore arrive out of order; readers
    always observe events ordered by sequence number.
    """

    def __init__(
        self,
        *,
        version: int = AUDIT_FORMAT_VERSION,
        clock: Callable[[], float] = time.time,
        anchor: str | None = None,
    ) -> None:
        """
        @param anchor: A secret signing material for cursor tokens.
            A random one is generated when omitted.  Passing an
            explicit anchor lets a migrated stream validate cursors
            while still distinguishing format versions.
        """
        self._version = version
        self._clock = clock
        self._lock = threading.RLock()
        self._events: dict[int, AuditEvent] = {}
        self._next_sequence = 1
        self._anchor = anchor if anchor is not None else os.urandom(16).hex()

    @property
    def version(self) -> int:
        return self._version

    def reserve(self) -> int:
        """
        Reserve and return the next stable sequence number.
        """
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            return sequence

    def commit(
        self,
        sequence: int,
        source: str,
        category: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        detail: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> AuditEvent:
        """
        Commit an event against a previously reserved sequence.

        Committing the same sequence twice with identical contents is
        idempotent and returns the original event; committing it with
        different contents raises L{AuditConflictError}.

        @raise InvalidSequenceError: C{sequence} was never reserved.
        """
        with self._lock:
            if sequence < 1 or sequence >= self._next_sequence:
                raise InvalidSequenceError(
                    f"sequence {sequence} was never reserved"
                )

            existing = self._events.get(sequence)
            if existing is not None:
                if existing.source != source or existing.category != (
                    category.value if isinstance(category, Enum) else category
                ):
                    raise AuditConflictError(
                        f"sequence {sequence} already committed with "
                        "different contents"
                    )
                return existing

            event = AuditEvent(
                sequence=sequence,
                source=source,
                category=(
                    category.value
                    if isinstance(category, Enum)
                    else str(category)
                ),
                timestamp=self._clock() if timestamp is None else timestamp,
                context=redact_deep(context or {}),
                detail=redact_deep(detail or {}),
                summary="",
                version=self._version,
            )
            summary = event.computed_summary()
            object.__setattr__(event, "summary", summary)
            self._events[sequence] = event
            return event

    def record(
        self,
        source: str,
        category: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        detail: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> AuditEvent:
        """
        Reserve and commit an event atomically.
        """
        sequence = self.reserve()
        return self.commit(
            sequence,
            source,
            category,
            context=context,
            detail=detail,
            timestamp=timestamp,
        )

    def snapshot(self) -> tuple[AuditEvent, ...]:
        """
        Return a consistent point-in-time snapshot of committed events in
        sequence order.

        Safe to call concurrently with commits from other threads.
        """
        with self._lock:
            return tuple(
                self._events[sequence] for sequence in sorted(self._events)
            )

    def read(
        self, *, after: int = 0, limit: Optional[int] = None
    ) -> tuple[AuditEvent, ...]:
        """
        Read up to C{limit} committed events with sequence numbers
        greater than C{after}, in sequence order.
        """
        events = self.snapshot()
        if limit is None:
            return tuple(e for e in events if e.sequence > after)
        return tuple(e for e in events if e.sequence > after)[:limit]

    def pending_sequences(self) -> tuple[int, ...]:
        """
        Sequence numbers that have been reserved but not committed.
        """
        with self._lock:
            return tuple(
                sequence
                for sequence in range(1, self._next_sequence)
                if sequence not in self._events
            )

    def validate_contiguous(self) -> None:
        """
        Assert every reserved sequence has been committed.

        @raise AuditGapError: One or more reserved sequences are missing.
        """
        with self._lock:
            missing = tuple(
                sequence
                for sequence in range(1, self._next_sequence)
                if sequence not in self._events
            )
        if missing:
            raise AuditGapError(missing)

    def validate_events(self) -> None:
        """
        Validate the summary of every committed event.
        """
        for event in self.snapshot():
            event.validate()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def cursor_for(self, last_sequence: int) -> str:
        """
        Build an opaque cursor token positioned after C{last_sequence}.
        """
        return AuditCursor(
            version=self._version, last_sequence=last_sequence
        ).to_token(self._anchor)

    def decode_cursor(self, token: str) -> AuditCursor:
        """
        Decode a cursor token issued by this log.
        """
        return AuditCursor.from_token(
            token, self._anchor, expected_version=self._version
        )


#
# Pagination
#


@attr.s(frozen=True, slots=True)
class AuditPage:
    """
    One page of an audit export.

    @ivar events: The events in this page, in sequence order.
    @ivar next_cursor: Token to fetch the next page, or C{None} when the
        end of the stream has been reached.
    """

    events: tuple[AuditEvent, ...] = attr.ib()
    next_cursor: Optional[str] = attr.ib(default=None)


class AuditReader:
    """
    A stateful cursor pager over a L{RenderAuditLog}.

    Each non-start cursor may only be presented once: re-presenting a
    consumed cursor raises L{DuplicateCursorError}, which protects
    export jobs from silently double-processing events.  C{None} is the
    start cursor and may always be used to (re)start a read.
    """

    def __init__(
        self, log: RenderAuditLog, *, default_page_size: int = DEFAULT_PAGE_SIZE
    ) -> None:
        self._log = log
        self._default_page_size = default_page_size
        self._consumed: set[str] = set()
        self._lock = threading.Lock()

    def reset(self) -> None:
        """
        Forget all consumed cursors (for example after recovering from a
        L{StaleCursorError}).
        """
        with self._lock:
            self._consumed.clear()

    def read_page(
        self,
        cursor: Optional[str] = None,
        *,
        limit: Optional[int] = None,
    ) -> AuditPage:
        if limit is None:
            limit = self._default_page_size
        if limit < 1:
            raise ValueError("page limit must be at least 1")

        if cursor is None:
            after = 0
        else:
            with self._lock:
                if cursor in self._consumed:
                    raise DuplicateCursorError(cursor)
            # Raises InvalidCursorError / StaleCursorError on bad input.
            position = self._log.decode_cursor(cursor)
            after = position.last_sequence
            with self._lock:
                self._consumed.add(cursor)

        events = self._log.read(after=after, limit=limit)
        if len(events) == limit:
            next_cursor = self._log.cursor_for(events[-1].sequence)
        else:
            next_cursor = None
        return AuditPage(events=events, next_cursor=next_cursor)


#
# Export
#


def export_ndjson(log: RenderAuditLog) -> bytes:
    """
    Export every committed event as newline-delimited JSON.

    This is strictly read-only: it takes a snapshot and serializes it.
    It never reserves sequences, mutates events, or touches request
    state, so exporting cannot alter the rendering main flow.  An empty
    stream exports as C{b""}.
    """
    lines = [
        json.dumps(
            event.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for event in log.snapshot()
    ]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


#
# Asynchronous delivery and retry
#


class IAuditSink(Protocol):
    """
    A destination for audited events.

    Deliver may return synchronously or a L{Deferred}; it may be
    invoked from arbitrary threads only if the implementation is
    thread-safe.
    """

    def deliver(self, event: AuditEvent) -> object:  # pragma: no cover
        ...


class MemoryAuditSink:
    """
    A simple sink that keeps delivered events in memory.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[AuditEvent] = []

    def deliver(self, event: AuditEvent) -> Deferred[None]:
        with self._lock:
            self._events.append(event)
        return succeed(None)

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._events)


@attr.s(eq=False, slots=True)
class PendingDelivery:
    """
    An event whose asynchronous delivery failed and is awaiting retry.

    The full event - including its redacted request context - is
    retained, so retries carry the original request context.
    """

    event: AuditEvent = attr.ib()
    attempts: int = attr.ib(default=0)
    last_failure: Optional[object] = attr.ib(default=None)
    first_failed_at: float = attr.ib(default=0.0)
    last_failed_at: float = attr.ib(default=0.0)


class RetryingAuditSink:
    """
    A sink wrapper that records failed deliveries for later retry.

    L{deliver} never errbacks: when the wrapped sink fails, the event
    (with its complete request context) is retained in L{pending} and
    the failure is logged.  L{retry_pending} re-attempts all pending
    deliveries in FIFO order; events whose retry succeeds are removed,
    the rest remain pending for another attempt.
    """

    def __init__(
        self,
        inner: IAuditSink,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._inner = inner
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: list[PendingDelivery] = []

    def _record_failure(self, failure: object, event: AuditEvent) -> None:
        now = self._clock()
        with self._lock:
            for pending in self._pending:
                if pending.event.sequence == event.sequence:
                    pending.attempts += 1
                    pending.last_failure = failure
                    pending.last_failed_at = now
                    break
            else:
                self._pending.append(
                    PendingDelivery(
                        event=event,
                        attempts=1,
                        last_failure=failure,
                        first_failed_at=now,
                        last_failed_at=now,
                    )
                )
        log.err(failure, f"audit delivery failed for event {event.sequence}")

    def _clear_failure(self, event: AuditEvent) -> None:
        with self._lock:
            self._pending = [
                pending
                for pending in self._pending
                if pending.event.sequence != event.sequence
            ]

    def deliver(self, event: AuditEvent) -> Deferred[None]:
        """
        Attempt delivery; on failure retain the event for retry.

        A successful delivery (including a successful retry) removes any
        previously queued L{PendingDelivery} for the same event.

        @return: A L{Deferred} that always fires successfully with
            C{None}, whether delivery succeeded or was queued.
        """
        d: Deferred[Any] = maybeDeferred(self._inner.deliver, event)

        def _failed(failure: object) -> None:
            self._record_failure(failure, event)

        def _succeeded(result: object) -> None:
            self._clear_failure(event)

        d.addCallbacks(_succeeded, _failed)
        return d.addCallback(lambda _: None)

    @property
    def pending(self) -> tuple[PendingDelivery, ...]:
        with self._lock:
            return tuple(self._pending)

    def retry_pending(self) -> Deferred[tuple[PendingDelivery, ...]]:
        """
        Re-attempt every pending delivery in FIFO order.

        @return: A L{Deferred} firing with the deliveries that are still
            pending after the retry attempt.
        """
        with self._lock:
            to_retry = list(self._pending)

        d: Deferred[Any] = succeed(None)
        for pending in to_retry:

            def _deliver(
                _: object, event: AuditEvent = pending.event
            ) -> Deferred[None]:
                return self.deliver(event)

            d.addCallback(_deliver)

        def _done(_: object) -> tuple[PendingDelivery, ...]:
            return self.pending

        return d.addCallback(_done)


#
# Facade
#


class IRenderAuditor(Protocol):
    """
    Structural interface consumed by L{klein._resource.KleinResource}.

    L{RenderAuditor} is the production implementation; tests and
    applications may supply any object with a fail-safe C{record}
    method of the same shape.
    """

    def record(
        self,
        source: str,
        category: str,
        *,
        context: Mapping[str, Any] | None = None,
        detail: Mapping[str, Any] | None = None,
        sequence: int | None = None,
    ) -> AuditEvent | None:
        """
        Record one audit event.  Must never raise.
        """
        ...


class RenderAuditor:
    """
    Best-effort entry point used by L{KleinResource}.

    Records events into a L{RenderAuditLog} and forwards them to an
    optional L{IAuditSink}.  Every method is fail-safe: neither local
    recording errors nor asynchronous sink errors can propagate into
    request rendering.
    """

    def __init__(
        self,
        *,
        log: Optional[RenderAuditLog] = None,
        sink: Optional[IAuditSink] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._log = log if log is not None else RenderAuditLog(clock=clock)
        self._sink = sink

    @property
    def log(self) -> RenderAuditLog:
        return self._log

    def new_reader(self, *, page_size: int = DEFAULT_PAGE_SIZE) -> AuditReader:
        return AuditReader(self._log, default_page_size=page_size)

    def record(
        self,
        source: str,
        category: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        detail: Optional[Mapping[str, Any]] = None,
        sequence: Optional[int] = None,
    ) -> Optional[AuditEvent]:
        """
        Record and forward one audit event.

        @param sequence: An optional previously reserved sequence
            number, used when the event's position must be stable across
            asynchronous delay.

        @return: The committed event, or C{None} if recording itself
            failed (the failure is logged).
        """
        try:
            if sequence is None:
                event = self._log.record(
                    source, category, context=context, detail=detail
                )
            else:
                event = self._log.commit(
                    sequence,
                    source,
                    category,
                    context=context,
                    detail=detail,
                )
        except Exception:
            log.err(None, "failed to record audit event")
            return None

        if self._sink is not None:
            try:
                d = maybeDeferred(self._sink.deliver, event)
                d.addErrback(log.err, "unhandled audit sink failure")
            except Exception:
                log.err(None, "audit sink raised synchronously")
        return event
