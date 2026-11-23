# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local session execution: two short durable operations, one inference call."""

import asyncio
import anyio
from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import asdict
import hashlib
import hmac
import json
import logging
import os
import time
from types import SimpleNamespace
import uuid

from fastapi.responses import JSONResponse, StreamingResponse

from pypto_serving.router.http import RouterError
from pypto_serving.serving import request_timing as timing
from .session_protocol import canonical, input_messages, output_messages
from .state import authorize
from .state_protocol import Continuation

logger = logging.getLogger(__name__)


class SessionRuntime:
    def __init__(self, server, store=None):
        # Loading the optional package happens only for the explicitly enabled
        # session mode. Ordinary serving never imports a database driver.
        from stateful_session import timing as state_timing
        from stateful_session.store import SessionStore

        self.server = server
        self.secret = os.environ["PYPTO_SESSION_SECRET"]
        self.store = store or SessionStore(
            os.environ["PYPTO_SESSION_DATABASE_URL"],
            os.getenv("PYPTO_SESSION_SCHEMA", "pypto_session_v3"),
            self.secret,
            ttl=float(os.getenv("PYPTO_SESSION_TTL_SECONDS", "3600")),
            execution_timeout=float(os.getenv("PYPTO_SESSION_TIMEOUT_SECONDS", "120")),
        )
        state_timing.configure(timing)
        self.db_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="session-db")
        self.token_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="session-tokenize")
        self.db_slots = asyncio.Semaphore(8)
        self.requests = asyncio.Semaphore(16)
        self.candidates = asyncio.Queue(maxsize=64)
        self.pending = set()
        self.background = None

    async def db(self, operation, *args, **kwargs):
        name = operation.__name__
        timing.mark("session." + name + ".submit")
        await asyncio.wait_for(self.db_slots.acquire(), 1)
        context = contextvars.copy_context()

        def run():
            timing.mark("session." + name + ".worker")
            with timing.span("session." + name):
                if name == "commit_history":
                    from stateful_session.database import execution_budget

                    with execution_budget(max(0, args[0].deadline - time.time())):
                        return operation(*args, **kwargs)
                return operation(*args, **kwargs)

        future = asyncio.get_running_loop().run_in_executor(self.db_executor, context.run, run)
        future.add_done_callback(lambda unused: self.db_slots.release())
        result = await asyncio.shield(future)
        timing.mark("session." + name + ".returned")
        return result

    async def start(self):
        if self.server.session_kv_mode != "input":
            raise ValueError("Local sessions require non-PD input mode")
        await self.db(self.store.start)
        self.background = asyncio.create_task(self.maintenance())

    async def close(self):
        if self.pending:
            await asyncio.gather(*self.pending, return_exceptions=True)
        if self.background:
            self.background.cancel()
            await asyncio.gather(self.background, return_exceptions=True)
        while not self.candidates.empty():
            receipt, _, _, _, _ = self.candidates.get_nowait()
            self.server.release_receipt(receipt)
        await self.db(self.store.close)
        self.token_executor.shutdown(wait=True)
        self.db_executor.shutdown(wait=True)

    async def maintenance(self):
        from stateful_session.database import CatalogError

        while True:
            candidate = None
            try:
                try:
                    candidate = await asyncio.wait_for(self.candidates.get(), 0.25)
                except asyncio.TimeoutError:
                    pass
                if candidate:
                    receipt, response_id, version, deadline, trace_id = candidate
                    published = False
                    try:
                        with timing.background(trace_id, "session.publish"):
                            if time.monotonic() < deadline:
                                published = await self.db(
                                    self.store.publish_kv, response_id, version, receipt
                                )
                    finally:
                        if not published:
                            self.server.release_receipt(receipt)
                        self.candidates.task_done()
                await self.db(self.store.sweep)
                for job in await self.db(self.store.claim_releases):
                    binding = job["binding"]
                    # A previous boot's volatile pages no longer exist. Never
                    # send its naked page IDs to a new owner.
                    replica_idx = self.server.replica_for_receipt(binding)
                    if replica_idx is not None:
                        objects = self.server._objects(replica_idx)
                        if binding["boot"] == objects.node.boot and binding["registry"] == objects.node.registry:
                            self.server.release_receipt(binding)
                    await self.db(self.store.ack_release, job["id"], job["claim"])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A failed publication is disposable; release jobs retain their
                # fenced lease for retry, and session validity stays in SQL.
                code = exc.code if isinstance(exc, CatalogError) else type(exc).__name__
                logger.warning("Session maintenance failed: %s", code)
                await asyncio.sleep(0.25)

    async def prepare(self, request):
        server = self.server
        # Parser/constraint checks retain their existing async preflight. The
        # expensive template and tokenizer run on a dedicated bounded worker.
        parser = server._output_parser_spec(request)
        constraint = server._constraint_spec(request, parser)
        if constraint is not None:
            await server._preflight_constraint(constraint)
        context = contextvars.copy_context()

        def render():
            timing.mark("session.tokenize.worker")
            with timing.span("session.template"):
                prompt = server._apply_chat_template(
                    request.messages,
                    request.chat_template_kwargs,
                    reasoning_effort=request.reasoning_effort,
                    tools=request.tools,
                )
                rendering = server.rendering_profile(request)
            config = server._resolve_generate_config(request)
            server._validate_generation(config)
            tokens = list(server.engine._tokenize_prompt(prompt) or ())
            if not tokens:
                raise ValueError("Session requires non-empty input")
            server._budget(tokens, config)
            return tokens, config, parser, constraint, rendering

        timing.mark("session.tokenize.submit")
        result = await asyncio.get_running_loop().run_in_executor(self.token_executor, context.run, render)
        timing.mark("session.tokenize.returned")
        return result

    def validate_history(self, request, parent, rendering):
        incoming = canonical([message.model_dump(exclude_none=True) for message in request.messages])
        ctx = request.stateful_context
        # Instructions belong to this request and are not inherited by the
        # Gateway. Persistent input/system items still remain in history.
        leading = [dict(role="system", content=ctx.instructions)] if ctx.instructions else []
        previous = []
        if parent:
            history = parent.history
            if (
                history["model_identity"] != self.server.model_identity
                or history["rendering"] != rendering
                or history["mode"] != "single-input-v3"
            ):
                raise RouterError(409, "model_identity_mismatch", "Parent identity or rendering differs")
            previous = history["history_messages"] + history["output_messages"]
        if incoming != leading + previous + input_messages(ctx.input_items):
            raise RouterError(
                409, "history_integrity_mismatch", "Messages differ from trusted parent and input"
            )
        return incoming[len(leading) :]

    async def handle(self, request, http):
        authorize(http.headers)
        if request.model != self.server.model_id:
            raise ValueError("Session model differs from the configured owner")
        if not self.background:
            raise RouterError(503, "session_unavailable", "Session state is not started")
        await asyncio.wait_for(self.requests.acquire(), 1)
        timing.mark("session.prepare.begin")
        begin = asyncio.create_task(
            self.db(
                self.store.begin_request,
                request.stateful_context.response_id,
                request.stateful_context.tenant_id,
                request.stateful_context.parent_response_id,
                store=request.stateful_context.store,
                expected_version=request.stateful_context.expected_parent_version,
            )
        )
        execution = None
        try:
            prepared = asyncio.create_task(self.prepare(request))
            # return_exceptions waits for both branches, so a successful begin
            # cannot be lost when rendering fails and leave an untracked owner.
            results = await asyncio.shield(asyncio.gather(begin, prepared, return_exceptions=True))
            if not isinstance(results[0], BaseException):
                execution, parent = results[0]
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            timing.mark("session.prepare.joined")
            with timing.span("session.history.validate"):
                history_messages = self.validate_history(request, parent, prepared.result()[4])
            settled = asyncio.Event()
            if not request.stream:
                result = None
                async for result in self.generate(
                    request, execution, parent, prepared.result(), history_messages, settled
                ):
                    pass
                return JSONResponse(result)
            frames = self.generate(request, execution, parent, prepared.result(), history_messages, settled)

            async def stream():
                try:
                    async for frame in frames:
                        yield "data: " + json.dumps(frame, separators=(",", ":")) + "\n\n"
                except Exception as exc:
                    code = getattr(exc, "code", "session_execution_failed")
                    yield "data: " + json.dumps(dict(error=dict(code=code, message=str(exc)))) + "\n\n"
                finally:
                    await frames.aclose()
                yield "data: [DONE]\n\n"

            runtime = self

            class SessionResponse(StreamingResponse):
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        with anyio.CancelScope(shield=True):
                            try:
                                await frames.aclose()
                                if not settled.is_set():
                                    await runtime.db(runtime.store.abort_request, execution)
                            finally:
                                runtime.requests.release()

            return SessionResponse(stream(), media_type="text/event-stream")
        except BaseException:
            # A cancelled caller must not release the admission permit while
            # its tokenizer job still occupies the bounded worker queue.
            if "prepared" in locals() and not prepared.done():
                try:
                    await asyncio.shield(prepared)
                except BaseException:
                    pass
            if execution is None:
                try:
                    execution, _ = await asyncio.shield(begin)
                except BaseException:
                    pass
            if execution:
                await asyncio.shield(self.db(self.store.abort_request, execution))
            raise
        finally:
            if not request.stream or execution is None or "frames" not in locals():
                self.requests.release()

    async def generate(self, request, execution, parent, prepared, history_messages, settled):
        from .native import SingleBinding

        tokens, config, parser, constraint, rendering = prepared
        ctx = request.stateful_context
        parent_receipt = parent.binding if parent else None
        replica_idx = self.server.replica_for_receipt(parent_receipt) if parent_receipt else None
        if replica_idx is None:
            replica_idx = self.server.select_replica()
        core = self.server._cores[replica_idx]
        objects = self.server._objects(replica_idx)
        salt = hmac.new(self.secret.encode(), ctx.tenant_id.encode(), hashlib.sha256).hexdigest()
        scope = dict(
            tenant_id=ctx.tenant_id,
            cache_salt_digest=hashlib.sha256(salt.encode()).hexdigest(),
            version_id=execution.owner,
        )
        prefix, temporary_pin = None, None
        selection = "first" if parent is None else "cold"
        if parent is not None:
            if parent_receipt and objects.parent_precheck(parent_receipt, scope)["state"] == "live":
                common = 0
                for old, new in zip(parent.history["prompt_token_ids"], tokens):
                    if old != new:
                        break
                    common += 1
                mount = min(common, parent_receipt["retained_tokens"], len(tokens) - 1) // 128 * 128
                with timing.span("session.mount"):
                    prefix = objects.mount(parent_receipt, scope, tokens, mount_tokens=mount)
                    temporary_pin = objects.cache.retain_group_block_snapshot(
                        prefix.block_ids, prefix.partition
                    )
                selection = "warm"
            elif ctx.policy == "strict":
                raise RouterError(409, "session_kv_lost", "Parent KV is not available")
        rid = "session-" + uuid.uuid4().hex
        timing.mark(
            "session.selection", selection=selection, mounted_tokens=prefix.token_count if prefix else 0
        )
        entry = SimpleNamespace(
            binding=SingleBinding(rid, execution.owner),
            native_scope=scope,
            continuation=Continuation(tuple(tokens), asdict(config), None, None),
            native_parent=parent_receipt if prefix else None,
            mounted_tokens=prefix.token_count if prefix else 0,
            native=None,
            error=None,
            done=asyncio.Event(),
        )
        core.scheduler.native_requests[rid] = entry

        def queued():
            nonlocal temporary_pin
            if temporary_pin:
                objects.cache.release_group_block_snapshot(temporary_pin)
                temporary_pin = None

        outputs = core.add_request(
            rid,
            "",
            config,
            prompt_token_ids=tokens,
            output_parser_spec=parser,
            constraint_spec=constraint,
            confirmed_prefix=prefix,
            on_queued=queued,
        )
        committed = False
        handed_off = False
        commit_task = None
        final = None
        previous_text = previous_reasoning = ""
        try:
            while True:
                remaining = execution.deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Session execution budget exhausted")
                try:
                    output = await asyncio.wait_for(anext(outputs), remaining)
                except StopAsyncIteration:
                    break
                text = output.text_delta if parser else (output.text or "")[len(previous_text) :]
                reasoning = (
                    output.reasoning_delta if parser else (output.reasoning or "")[len(previous_reasoning) :]
                )
                previous_text, previous_reasoning = output.text or "", output.reasoning or ""
                if request.stream and (text or reasoning or output.tool_call_deltas):
                    timing.mark("server.first_content")
                    calls = [
                        self.server._tool_delta(item).model_dump(exclude_none=True)
                        for item in output.tool_call_deltas
                        if request.parallel_tool_calls or item.index == 0
                    ]
                    yield self.chunk(
                        ctx.response_id,
                        dict(
                            role="assistant",
                            content=text or None,
                            reasoning_content=reasoning or None,
                            tool_calls=calls or None,
                        ),
                        None,
                    )
                if output.finished:
                    final = output
            if final is None or final.finish_reason not in {
                "FINISHED_EOS",
                "FINISHED_STOP",
                "FINISHED_LENGTH",
            }:
                raise RouterError(502, "session_generation_failed", "Engine did not finish successfully")
            if entry.error is not None:
                # Only allocation failure for the optional retained copy is
                # disposable. Integrity/ACK failures remain fatal.
                if not isinstance(entry.error, MemoryError):
                    raise entry.error
                timing.mark("session.kv.discarded", reason="retention_capacity")
            calls = [
                dict(id=call.id, type="function", function=dict(name=call.name, arguments=call.arguments))
                for call in (final.tool_calls if request.parallel_tool_calls else final.tool_calls[:1])
            ]
            history = dict(
                schema=3,
                mode="single-input-v3",
                parent_id=ctx.parent_response_id,
                parent_version=parent.version if parent else None,
                messages=[m.model_dump(exclude_none=True) for m in request.messages],
                history_messages=history_messages,
                output_messages=output_messages(final.text, calls),
                input_items=ctx.input_items,
                output_items=dict(text=final.text, reasoning=final.reasoning, tool_calls=calls),
                prompt_token_ids=tokens,
                output_token_ids=list(final.token_ids),
                generation=asdict(config),
                model_identity=self.server.model_identity,
                rendering=rendering,
                finish_reason=final.finish_reason,
            )

            async def finish_commit():
                nonlocal committed, handed_off
                version = await self.db(self.store.commit_history, execution, history)
                committed = True
                if entry.native is not None and ctx.store:
                    try:
                        self.candidates.put_nowait(
                            (
                                entry.native,
                                ctx.response_id,
                                version,
                                time.monotonic() + 5,
                                timing.current_headers().get(timing.HEADER),
                            )
                        )
                        handed_off = True
                        timing.mark("session.publish.enqueued")
                    except asyncio.QueueFull:
                        timing.mark("session.kv.discarded", reason="publish_queue_full")
                return version

            commit_task = asyncio.create_task(finish_commit())
            self.pending.add(commit_task)
            commit_task.add_done_callback(self.pending.discard)
            await asyncio.wait_for(asyncio.shield(commit_task), max(0.001, execution.deadline - time.time()))
            timing.mark("session.history.committed")
            usage = dict(
                prompt_tokens=len(tokens),
                completion_tokens=len(final.token_ids),
                total_tokens=len(tokens) + len(final.token_ids),
            )
            reason = self.server._chat_finish_reason(final)
            if request.stream:
                yield self.chunk(ctx.response_id, {}, reason)
                yield dict(
                    id=ctx.response_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=self.server.model_id,
                    choices=[],
                    usage=usage,
                )
            else:
                yield dict(
                    id=ctx.response_id,
                    object="chat.completion",
                    created=int(time.time()),
                    model=self.server.model_id,
                    choices=[
                        dict(
                            index=0,
                            message=dict(
                                role="assistant",
                                content=final.text or None,
                                reasoning_content=final.reasoning or None,
                                tool_calls=calls or None,
                            ),
                            finish_reason=reason,
                        )
                    ],
                    usage=usage,
                )
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await outputs.aclose()
                finally:
                    queued()
                    core.scheduler.native_requests.pop(rid, None)
                    try:
                        # Disconnection cannot cancel synchronous SQL. Resolve
                        # its bounded commit before transferring candidate ownership.
                        if commit_task is not None and not commit_task.done():
                            try:
                                await asyncio.shield(commit_task)
                            except Exception:
                                pass
                        if not committed:
                            try:
                                status = await self.db(self.store.execution_status, execution)
                                committed = status is not None and status["status"] == "committed"
                            except Exception:
                                pass
                            await self.db(self.store.abort_request, execution)
                    finally:
                        # Even a database outage must release the optional local
                        # copy. The fenced execution can converge after recovery.
                        if not handed_off:
                            objects.discard_request(rid)
                        entry.done.set()
                        settled.set()

    def chunk(self, response_id, delta, finish):
        return dict(
            id=response_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=self.server.model_id,
            choices=[dict(index=0, delta=delta, finish_reason=finish)],
        )
