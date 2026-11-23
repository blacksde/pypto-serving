# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Opt-in request timing; no payloads, per-token logging, or device synchronization."""
from __future__ import annotations

import contextvars
import functools
import inspect
import hashlib
import json
import os
import re
import time
import typing
from contextlib import contextmanager

HEADER = "x-pypto-timing-id"
_current = contextvars.ContextVar("request_timing", default=None)
_engines = {}


@functools.lru_cache(maxsize=1)
def clock_domain():
    try:
        with open("/proc/sys/kernel/random/boot_id") as stream:
            return stream.read().strip()
    except OSError:
        return "unavailable"


class Trace:
    def __init__(self, identifier, component, path):
        self.identifier, self.component, self.path = identifier, component, path
        self.start = time.monotonic_ns()
        self.wall = time.time_ns()
        self.calibration_ns = time.monotonic_ns() - self.start
        self.wall -= self.calibration_ns // 2
        self.events, self.seen, self.engine_ids = [], set(), []
        self.dropped = 0
        self.steps = {}

    def event(self, name, start=None, **fields):
        if len(self.events) >= 512:
            self.dropped += 1
            return
        end = time.monotonic_ns() - self.start
        self.events.append(dict(name=name, start_ns=end if start is None else start - self.start,
                                end_ns=end, **fields))

    def once(self, name, **fields):
        if name not in self.seen:
            if len(self.events) >= 512:
                self.dropped += 1
                return
            self.seen.add(name)
            self.event(name, **fields)

    def finish(self):
        for identifier in self.engine_ids:
            _engines.pop(identifier, None)
        mono_end = time.monotonic_ns()
        drift = time.time_ns() - (self.wall + mono_end - self.start)
        row = dict(schema=1, trace_id=self.identifier, component=self.component, path=self.path,
                   clock_domain=clock_domain(), wall_drift_ns=drift,
                   pid=os.getpid(), mono_start_ns=self.start, wall_start_ns=self.wall,
                   calibration_ns=self.calibration_ns, events=self.events, dropped=self.dropped)
        try:
            # One best-effort append after the response, never on each token.
            fd = os.open(os.environ["SERVING_TIMING_FILE"], os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(fd, (json.dumps(row, separators=(",", ":")) + "\n").encode())
            finally:
                os.close(fd)
        except (OSError, KeyError):
            pass


def current_headers():
    trace = _current.get()
    return {HEADER: trace.identifier} if trace is not None else {}


@contextmanager
def background(identifier, name):
    """One opt-in record for work that outlives the original HTTP trace."""
    if not identifier or not os.environ.get("SERVING_TIMING_FILE"):
        yield
        return
    trace = Trace(identifier, "pypto-background", name)
    token = _current.set(trace)
    try:
        trace.once("background.begin")
        yield
    finally:
        trace.once("background.end")
        _current.reset(token)
        trace.finish()


def mark(name, *, engine_id=None, **fields):
    trace = _engines.get(engine_id) if engine_id is not None else _current.get()
    if trace is not None:
        trace.once(name, **fields)


def bind_engine(identifier, tokens=None):
    trace = _current.get()
    if trace is not None:
        _engines[identifier] = trace
        trace.engine_ids.append(identifier)
        fields = {}
        if tokens is not None:
            encoded = json.dumps([int(t) for t in tokens], separators=(",", ":")).encode()
            fields["prompt_digest"] = hashlib.sha256(encoded).hexdigest()
        trace.event("engine.bound", engine_id=identifier, **fields)


@contextmanager
def span(name):
    trace = _current.get()
    if trace is None:
        yield
        return
    start = time.monotonic_ns()
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        trace.event(name, start, outcome=outcome)


def timed(name, *, engine=False):
    def decorate(function):
        if inspect.iscoroutinefunction(function):
            @functools.wraps(function)
            async def asynchronous(*args, **kwargs):
                with span(name):
                    return await function(*args, **kwargs)
            _signature(asynchronous, function)
            return asynchronous
        @functools.wraps(function)
        def synchronous(*args, **kwargs):
            token = None
            if engine:
                owner = args[1] if len(args) > 1 else kwargs.get("request_id")
                identifier = owner if isinstance(owner, str) else getattr(getattr(owner, "binding", None), "request", None)
                trace = _engines.get(identifier)
                if trace is not _current.get():
                    token = _current.set(trace)
            try:
                with span(name):
                    return function(*args, **kwargs)
            finally:
                if token is not None:
                    _current.reset(token)
        _signature(synchronous, function)
        return synchronous
    return decorate


class TimingMiddleware:
    """Pure ASGI wrapper preserving streaming and disconnect semantics."""

    def __init__(self, app, component):
        self.app, self.component = app, component

    async def __call__(self, scope, receive, send):
        headers = dict(scope.get("headers", ()))
        identifier = headers.get(HEADER.encode(), b"").decode("ascii", errors="replace")
        if (scope["type"] != "http" or not os.environ.get("SERVING_TIMING_FILE")
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", identifier)):
            return await self.app(scope, receive, send)
        trace = Trace(identifier, self.component, scope.get("path", ""))
        token = _current.set(trace)
        trace.once("http.receive")

        async def observed_send(message):
            if message["type"] == "http.response.start":
                trace.once("http.headers", status=message["status"])
            elif message["type"] == "http.response.body" and message.get("body"):
                trace.once("http.first_body")
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                trace.once("http.end")
        try:
            await self.app(scope, receive, observed_send)
        finally:
            trace.once("http.closed")
            _current.reset(token)
            trace.finish()


def step_begin(output):
    if not _engines:
        return
    for scheduled in output.scheduled_requests:
        trace = _engines.get(scheduled.request.request_id)
        if trace is not None and (scheduled.is_prefill or "engine.first_token" not in trace.seen):
            trace.steps[(id(output), scheduled.request.request_id)] = time.monotonic_ns()


def step_end(output):
    if not _engines:
        return
    for scheduled in output.scheduled_requests:
        trace = _engines.get(scheduled.request.request_id)
        if trace is not None:
            start = trace.steps.pop((id(output), scheduled.request.request_id), None)
            if start is not None:
                trace.event("engine.prefill.host_wait" if scheduled.is_prefill else "engine.decode.host_wait",
                            start, tokens=scheduled.num_new_tokens)


def _signature(wrapper, function):
    # FastAPI evaluates endpoint annotations in the wrapper module otherwise.
    try:
        hints = typing.get_type_hints(function)
        signature = inspect.signature(function)
        wrapper.__signature__ = signature.replace(
            parameters=[parameter.replace(annotation=hints.get(name, parameter.annotation))
                        for name, parameter in signature.parameters.items()],
            return_annotation=hints.get("return", signature.return_annotation))
    except (NameError, TypeError):
        pass
