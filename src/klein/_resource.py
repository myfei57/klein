# -*- test-case-name: klein.test.test_resource -*-
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from werkzeug.exceptions import HTTPException

from twisted.internet import defer
from twisted.internet.defer import Deferred, maybeDeferred, succeed
from twisted.python import log
from twisted.python.failure import Failure
from twisted.web import server
from twisted.web.iweb import IRenderable, IRequest
from twisted.web.resource import IResource, Resource, getChildForRequest
from twisted.web.template import renderElement

from ._budgets import (
    BudgetCancelled,
    BudgetedRender,
    BudgetExceeded,
    BudgetInitializationError,
    BudgetShutdown,
    RenderBudget,
    StageFailed,
)
from ._dihttp import Response
from ._interfaces import IKleinRequest

if TYPE_CHECKING:
    # NB: circular import, must not be imported at runtime.
    from ._app import (
        ErrorMethods,
        Klein,
        KleinRouteHandler,
        RouteMetadata,
    )


def route_metadata(handler: KleinRouteHandler) -> RouteMetadata:
    return handler  # type: ignore[return-value]


def ensure_utf8_bytes(v: str | bytes) -> bytes:
    """
    Coerces a value which is either a C{str} or C{bytes} to a C{bytes}.
    If ``v`` is a C{str} object it is encoded as utf-8.
    """
    if isinstance(v, str):
        v = v.encode("utf-8")
    return v


def encode_http_exception_body(exception: HTTPException) -> bytes:
    """
    Render the default response body for a werkzeug L{HTTPException}.

    Accessed via L{getattr} because the C{iter_encoded} method is missing
    from some werkzeug type stubs.
    """
    response = exception.get_response({})
    iter_encoded: Callable[[], Iterable[bytes]] = getattr(
        response, "iter_encoded"
    )
    return ensure_utf8_bytes(b"".join(iter_encoded()))


class _StandInResource:
    """
    A standin for a Resource.

    This is a sentinel value for L{KleinResource}, to say that we are rendering
    a L{Resource}, which may close the connection itself later.
    """


StandInResource = _StandInResource()


class URLDecodeError(Exception):
    """
    Raised if one or more string parts of the URL could not be decoded.
    """

    __slots__ = ["errors"]

    def __init__(self, errors: Sequence[tuple[str, Failure]]) -> None:
        """
        @param errors: Sequence of decoding errors, expressed as tuples
            of names and an associated failure.
        """
        super().__init__(errors)
        self.errors = errors

    def __repr__(self) -> str:
        return f"<URLDecodeError(errors={self.errors!r})>"


def extractURLparts(request: IRequest) -> tuple[str, str, int, str, str]:
    """
    Extracts and decodes URI parts from C{request}.

    All strings must be UTF8-decodable.

    @param request: A Twisted Web request.

    @raise URLDecodeError: If one of the parts could not be decoded as UTF-8.

    @return: L{tuple} of the URL scheme, the server name, the server port, the
        path info and the script name.
    """
    server_name = request.getRequestHostname()
    if hasattr(request.getHost(), "port"):
        server_port = request.getHost().port
    else:
        server_port = 0
    is_secure = bool(request.isSecure())
    if (is_secure, server_port) not in [
        (True, 443),
        (False, 80),
    ] or server_port == 0:
        server_name = b"%s:%d" % (server_name, server_port)

    script_name = b""
    if request.prepath:
        script_name = b"/".join(request.prepath)

        # TODO: coverage
        if not script_name.startswith(b"/"):  # pragma: no branch
            script_name = b"/" + script_name

    path_info = b""
    # TODO: coverage
    if request.postpath:  # pragma: no branch
        path_info = b"/".join(request.postpath)

        # TODO: coverage
        if not path_info.startswith(b"/"):  # pragma: no branch
            path_info = b"/" + path_info

    url_scheme = "https" if is_secure else "http"

    utf8Failures = []
    try:
        server_name = server_name.decode("utf-8")
    except UnicodeDecodeError:
        utf8Failures.append(("SERVER_NAME", Failure()))
    try:
        path_text = path_info.decode("utf-8")
    except UnicodeDecodeError:
        utf8Failures.append(("PATH_INFO", Failure()))
    try:
        script_text = script_name.decode("utf-8")
    except UnicodeDecodeError:
        utf8Failures.append(("SCRIPT_NAME", Failure()))

    if utf8Failures:
        raise URLDecodeError(utf8Failures)

    return url_scheme, server_name, server_port, path_text, script_text


class KleinResource(Resource):
    """
    A ``Resource`` that can do URL routing.
    """

    isLeaf = True

    def __init__(
        self,
        app: Klein,
        execution_budget: Callable[[], RenderBudget] | None = None,
    ) -> None:
        """
        @param execution_budget: Optional zero-argument factory returning a
            fresh L{RenderBudget} for each request.  When supplied, every
            render stage is checked against that budget before it runs; when
            C{None} rendering proceeds without any budget.
        """
        super().__init__()
        self._app = app
        self._executionBudget = execution_budget

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KleinResource):
            return vars(self) == vars(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def render(self, request: IRequest) -> int | bytes:
        """
        Render the request based on the underlying L{Klein} application, in a
        multi-step process:

            1. convert twisted input request information into something legible
               to werkzeug

            2. bind that info (URL, method, query args, etc) into a request
               mapper, and look up a werkzeug endpoint (i.e.: klein
               C{@route}-decorated method) to invoke

            3. invoke that endpoint, getting something renderable

            4. render that thing

            5. handle any errors in lookup or rendering with declared error
               handlers

            6. render the thing that the error handler returned

        When an execution budget was supplied at construction time, each of
        these stages is checked against that budget before it is accepted.
        """
        try:
            url_parts = extractURLparts(request)
        except URLDecodeError as e:
            for what, fail in e.errors:
                log.err(fail, f"Invalid encoding in {what}.")
            request.setResponseCode(400)
            return b"Non-UTF-8 encoding in URL."

        if self._executionBudget is None:
            return self._renderUnbudgeted(request, url_parts)
        return self._renderBudgeted(request, url_parts)

    def _renderUnbudgeted(
        self,
        request: IRequest,
        url_parts: tuple[str, str, int, str, str],
    ) -> int:
        """
        Render without any execution budget; the historical behavior.
        """
        (
            url_scheme,
            server_name,
            server_port,
            path_info,
            script_name,
        ) = url_parts

        # Bind our mapper.
        mapper = self._app.url_map.bind(
            server_name,
            script_name,
            path_info=path_info,
            default_method=request.method.decode("utf-8"),
            url_scheme=url_scheme,
        )
        # Make the mapper available to the view.
        kleinRequest = IKleinRequest(request)
        kleinRequest.mapper = mapper

        # Make sure we'll notice when the connection goes away unambiguously.
        request_finished = [False]

        def _finish(result: object) -> None:
            request_finished[0] = True

        def _execute() -> Deferred:
            # Actually doing the match right here. This can cause an exception
            # to percolate up. If that happens it will be handled below in
            # processing_failed, either by a user-registered error handler or
            # one of our defaults.
            rule, kwargs = mapper.match(return_rule=True)
            endpoint = rule.endpoint

            # Try pretty hard to fix up prepath and postpath.
            segment_count = route_metadata(
                self._app.endpoints[endpoint]
            ).segment_count
            request.prepath.extend(request.postpath[:segment_count])
            request.postpath = request.postpath[segment_count:]

            request.notifyFinish().addBoth(  # type: ignore[attr-defined]
                _finish,
            )

            # Standard Twisted Web stuff. Defer the method action, giving us
            # something renderable or printable. Return NOT_DONE_YET and set up
            # the incremental renderer.
            d = maybeDeferred(
                self._app.execute_endpoint, endpoint, request, **kwargs
            )

            request.notifyFinish().addErrback(  # type: ignore[attr-defined]
                lambda _: d.cancel(),
            )

            return d

        d = maybeDeferred(_execute)

        # typing note: returns Any because Response._applyToRequest returns Any
        def process(r: object) -> Any:
            """
            Recursively go through r and any child Resources until something
            returns something renderable (L{IResource}, L{IRenderable},
            L{bytes}, L{str}, or L{Iterable} of same), then render it and let
            the result of that bubble back up.
            """
            # isinstance() is faster than providedBy(), so this speeds up the
            # very common case of returning pre-rendered results, at the cost
            # of slightly slowing down other cases.
            if isinstance(r, (bytes, str)):
                return r

            if isinstance(r, Response):
                r = r._applyToRequest(request)

            if IResource.providedBy(r):
                request.render(  # type: ignore[attr-defined]
                    getChildForRequest(r, request)
                )
                return StandInResource

            if IRenderable.providedBy(r):
                renderElement(request, r)
                return StandInResource

            return r

        d.addCallback(process)

        def processing_failed(
            failure: Failure, error_handlers: ErrorMethods
        ) -> Deferred | None:
            # The failure processor writes to the request.  If the
            # request is already finished we should suppress failure
            # processing.  We don't return failure here because there
            # is no way to surface this failure to the user if the
            # request is finished.
            if request_finished[0]:
                # TODO: coverage
                if not failure.check(defer.CancelledError):  # pragma: no branch
                    log.err(
                        failure, "Unhandled Error Processing Request."
                    )  # pragma: no cover
                return None

            # If there are no more registered handlers, apply some defaults
            if len(error_handlers) == 0:
                if failure.check(HTTPException):
                    he = failure.value
                    assert isinstance(he, HTTPException)
                    request.setResponseCode(he.code)

                    resp = he.get_response({})

                    for header, value in resp.headers:
                        request.setHeader(
                            ensure_utf8_bytes(header), ensure_utf8_bytes(value)
                        )

                    return succeed(encode_http_exception_body(he))
                else:
                    request.processingFailed(  # type: ignore[attr-defined]
                        failure,
                    )
                    return None

            error_handler = error_handlers[0]

            # Each error handler is a tuple of
            # (list_of_exception_types, handler_fn)
            if failure.check(*error_handler[0]):
                handler_func = error_handler[1]
                d = maybeDeferred(
                    self._app.execute_error_handler,
                    handler_func,
                    request,
                    failure,
                )

                d.addCallback(process)

                return d.addErrback(processing_failed, error_handlers[1:])

            return processing_failed(failure, error_handlers[1:])

        d.addErrback(processing_failed, self._app._error_handlers)

        def write_response(
            r: _StandInResource | str | bytes | int | None,
        ) -> None:
            if r is StandInResource:
                return

            if isinstance(r, str):
                r = r.encode("utf-8")

            if isinstance(r, bytes):
                request.write(r)

            if not request_finished[0]:
                request.finish()

        d.addCallback(write_response)
        d.addErrback(log.err, _why="Unhandled Error writing response")

        return server.NOT_DONE_YET

    def _renderBudgeted(
        self,
        request: IRequest,
        url_parts: tuple[str, str, int, str, str],
    ) -> int:
        """
        Render with every stage gated by the caller supplied execution
        budget.
        """
        (
            url_scheme,
            server_name,
            server_port,
            path_info,
            script_name,
        ) = url_parts

        # Make sure we'll notice when the connection goes away.
        request_finished = [False]

        def _finish(result: object) -> None:
            request_finished[0] = True

        assert self._executionBudget is not None
        try:
            budget = self._executionBudget()
            controller = BudgetedRender.open(budget, request)
        except Exception as e:
            if not isinstance(e, BudgetInitializationError):
                e = BudgetInitializationError(
                    "execution budget could not be initialized", cause=e
                )
            request.processingFailed(Failure(e))  # type: ignore[attr-defined]
            return server.NOT_DONE_YET

        request.notifyFinish().addBoth(  # type: ignore[attr-defined]
            _finish,
        )
        request.notifyFinish().addErrback(  # type: ignore[attr-defined]
            lambda _: controller.cancel(),
        )

        mapper_box: list[Any] = [None]

        def _bind_mapper() -> None:
            mapper = self._app.url_map.bind(
                server_name,
                script_name,
                path_info=path_info,
                default_method=request.method.decode("utf-8"),
                url_scheme=url_scheme,
            )
            # Make the mapper, and this controller, available to the view.
            kleinRequest = IKleinRequest(request)
            kleinRequest.mapper = mapper
            kleinRequest.execution_budget = controller
            mapper_box[0] = mapper

        def _route() -> Deferred:
            # Actually doing the match right here. This can cause an
            # exception to percolate up. If that happens it will be handled
            # below in processing_failed, either by a user-registered error
            # handler or one of our defaults.
            mapper = mapper_box[0]
            rule, kwargs = mapper.match(return_rule=True)
            endpoint = rule.endpoint

            # Try pretty hard to fix up prepath and postpath.
            segment_count = route_metadata(
                self._app.endpoints[endpoint]
            ).segment_count
            request.prepath.extend(request.postpath[:segment_count])
            request.postpath = request.postpath[segment_count:]

            # Standard Twisted Web stuff. Defer the method action, giving us
            # something renderable or printable. Return NOT_DONE_YET and set
            # up the incremental renderer.
            return maybeDeferred(
                self._app.execute_endpoint, endpoint, request, **kwargs
            )

        # typing note: returns Any because Response._applyToRequest returns Any
        def process(r: object) -> Any:
            """
            Recursively go through r and any child Resources until something
            returns something renderable (L{IResource}, L{IRenderable},
            L{bytes}, L{str}, or L{Iterable} of same), then render it and let
            the result of that bubble back up.
            """
            # isinstance() is faster than providedBy(), so this speeds up the
            # very common case of returning pre-rendered results, at the cost
            # of slightly slowing down other cases.
            if isinstance(r, (bytes, str)):
                return r

            if isinstance(r, Response):
                r = r._applyToRequest(request)

            if IResource.providedBy(r):
                request.render(  # type: ignore[attr-defined]
                    getChildForRequest(r, request)
                )
                return StandInResource

            if IRenderable.providedBy(r):
                renderElement(request, r)
                return StandInResource

            return r

        def write_response(
            r: _StandInResource | str | bytes | int | None,
        ) -> None:
            if r is StandInResource:
                return

            if isinstance(r, str):
                r = r.encode("utf-8")

            if isinstance(r, bytes):
                request.write(r)

            if not request_finished[0]:
                request.finish()

        def processing_failed(
            failure: Failure, error_handlers: ErrorMethods
        ) -> Deferred | None:
            # Asynchronous stage failures retain their originating failure;
            # unwrap it so error handlers see the real exception.
            if failure.check(StageFailed):
                stage_failure = failure.value
                assert isinstance(stage_failure, StageFailed)
                failure = stage_failure.failure

            # The failure processor writes to the request.  If the
            # request is already finished we should suppress failure
            # processing.  We don't return failure here because there
            # is no way to surface this failure to the user if the
            # request is finished.
            if request_finished[0]:
                # TODO: coverage
                if not failure.check(defer.CancelledError):  # pragma: no branch
                    log.err(
                        failure, "Unhandled Error Processing Request."
                    )  # pragma: no cover
                return None

            # If there are no more registered handlers, apply some defaults
            if len(error_handlers) == 0:
                if failure.check(HTTPException):
                    he = failure.value
                    assert isinstance(he, HTTPException)
                    request.setResponseCode(he.code)

                    resp = he.get_response({})

                    for header, value in resp.headers:
                        request.setHeader(
                            ensure_utf8_bytes(header), ensure_utf8_bytes(value)
                        )

                    return succeed(encode_http_exception_body(he))
                else:
                    request.processingFailed(  # type: ignore[attr-defined]
                        failure,
                    )
                    return None

            error_handler = error_handlers[0]

            # Each error handler is a tuple of
            # (list_of_exception_types, handler_fn)
            if failure.check(*error_handler[0]):
                handler_func = error_handler[1]
                handler_index = len(self._app._error_handlers) - len(
                    error_handlers
                )

                def _run_error_handler() -> Deferred:
                    handler_deferred = maybeDeferred(
                        self._app.execute_error_handler,
                        handler_func,
                        request,
                        failure,
                    )
                    handler_deferred.addCallback(process)
                    return handler_deferred

                error_deferred = controller.run(
                    f"error:{handler_index}", _run_error_handler
                )
                return error_deferred.addErrback(
                    processing_failed, error_handlers[1:]
                )

            return processing_failed(failure, error_handlers[1:])

        terminated = [False]

        def terminal_failure(failure: Failure) -> Failure | None:
            # The four terminal budget reasons each get their own response
            # (or deliberate suppression), and the controller has already
            # released all temporary resources.  Handling a terminal
            # failure stops the chain: the write stage must not run.
            if failure.check(BudgetCancelled):
                terminated[0] = True
                return None

            if failure.check(BudgetShutdown):
                terminated[0] = True
                log.err(failure, "Budgeted render stopped during shutdown.")
                return None

            if failure.check(BudgetExceeded):
                terminated[0] = True
                if not request_finished[0]:
                    request.setResponseCode(503)
                    request.setHeader(
                        b"content-type", b"text/plain; charset=utf-8"
                    )
                    request.write(b"Render budget exceeded.\n")
                    request.finish()
                return None

            if failure.check(BudgetInitializationError):
                terminated[0] = True
                if not request_finished[0]:
                    request.processingFailed(  # type: ignore[attr-defined]
                        failure
                    )
                return None

            return failure

        def write_stage(
            result: _StandInResource | str | bytes | int | None,
        ) -> Deferred | None:
            # Do not write after a terminal budget reason was handled.
            if terminated[0]:
                return None
            return controller.run("write", lambda: write_response(result))

        d = controller.run("route", _route, initialize=_bind_mapper)
        d.addCallback(
            lambda result: controller.run("render", lambda: process(result))
        )
        d.addErrback(terminal_failure)
        d.addErrback(processing_failed, self._app._error_handlers)
        d.addCallback(write_stage)
        d.addErrback(terminal_failure)
        d.addErrback(log.err, _why="Unhandled Error writing response")
        d.addBoth(lambda _: controller.close())

        return server.NOT_DONE_YET
