# Copyright (c) PyPTO Contributors.
"""Single-rank local session serving with explicit KV ownership."""

from dataclasses import dataclass
import os
from types import SimpleNamespace
import uuid

from fastapi import Request as HTTPRequest
from fastapi.responses import JSONResponse

from pypto_serving.router.http import RouterError
from pypto_serving.serving.sched.scheduler import RequestStatus, Scheduler
from .model_identity import load_model_identity
from .server import ChatCompletionRequest, ServingServer
from .state import NativeObjects, authorize
from .state_protocol import Component, Contract, Node, Topology, digest


@dataclass(frozen=True)
class SingleBinding:
    request: str
    input_digest: str


class SessionScheduler(Scheduler):
    """Keep input KV pages after the real prefill acknowledgement."""

    def __init__(self, *, session_kv_mode="input", **kwargs):
        if session_kv_mode != "input":
            raise ValueError("Local sessions support input KV retention only")
        super().__init__(**kwargs)
        self.session_kv_mode = session_kv_mode
        self.native_requests = {}
        self.native_objects = None

    def update_from_output(self, output, tokens):
        for scheduled in output.scheduled_requests:
            request = scheduled.request
            entry = self.native_requests.get(request.request_id)
            if (
                entry is None
                or entry.error is not None
                or request.status.is_finished
                or request.status is RequestStatus.PREEMPTED
                or entry.native is not None
            ):
                continue
            end = scheduled.num_computed_tokens + scheduled.num_new_tokens
            if end < request.num_prompt_tokens or not scheduled.is_prefill:
                continue
            try:
                entry.native = self.native_objects.retain(
                    entry,
                    SimpleNamespace(
                        final=True,
                        end=request.num_prompt_tokens,
                        blocks=scheduled.block_ids_by_group,
                        partition=scheduled.cache_partition,
                    ),
                )
            except Exception as exc:
                entry.error = exc
        return super().update_from_output(output, tokens)


class NativeServer(ServingServer):
    """Expose ordinary Chat plus the database-backed C session path."""

    def __init__(self, engine, model_id, generate, *, session_kv_mode="input"):
        self.core = engine.single_core()
        if len(engine.config.worker_device_ids()) != 1:
            raise ValueError("Local sessions require one serving rank")
        if not isinstance(self.core.scheduler, SessionScheduler):
            raise ValueError("Local sessions require SessionScheduler")
        if engine.config.executor_cls != "PyptoQwen14BExecutor":
            raise ValueError("Local sessions require the Qwen3-14B executor")
        if session_kv_mode != "input":
            raise ValueError("Local sessions support input KV retention only")
        self.session_kv_mode = session_kv_mode
        self.boot = uuid.uuid4().hex
        self.model_identity = load_model_identity(engine.config.model_dir, model_id, engine.tokenizer)
        self.objects = None
        self.sessions = None
        super().__init__(engine, model_id, generate)
        if os.getenv("PYPTO_SESSION_DATABASE_URL"):
            from .session import SessionRuntime
            from stateful_session.database import CatalogError

            self.sessions = SessionRuntime(self)
            self.app.router.add_event_handler("startup", self.sessions.start)
            self.app.router.add_event_handler("shutdown", self.sessions.close)

            async def state_error(request, exc):
                return JSONResponse(
                    {"error": {"code": exc.code, "message": str(exc)}},
                    status_code=exc.status_code,
                )

            self.app.add_exception_handler(CatalogError, state_error)

    async def _chat_completions(self, request: ChatCompletionRequest, http: HTTPRequest):
        if request.stateful_context is None:
            return await super()._chat_completions(request)
        if self.sessions is None:
            raise RouterError(503, "session_disabled", "Local session extension is not enabled")
        return await self.sessions.handle(request, http)

    async def session_delete(self, request: HTTPRequest):
        authorize(request.headers)
        if self.sessions is None:
            raise RouterError(503, "session_disabled", "Local sessions are disabled")
        raw = await request.json()
        response_id = raw.get("response_id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("response_id is required")
        await self.sessions.db(self.sessions.store.request_delete, response_id, raw.get("tenant_id"))
        return JSONResponse({"deleted": True, "response_id": response_id})

    async def session_status(self, request: HTTPRequest):
        authorize(request.headers)
        if self.sessions is None:
            raise RouterError(503, "session_disabled", "Local sessions are disabled")
        return JSONResponse(
            {
                "boot": self.boot,
                "live_objects": len(self.objects.objects) if self.objects else 0,
                "pending_publications": self.sessions.candidates.qsize(),
                "active_executions": len(self.core.scheduler.native_requests),
                "database_pool": self.sessions.store.database.snapshot(),
            }
        )

    def _register_exception_handlers(self):
        super()._register_exception_handlers()

        @self.app.exception_handler(RouterError)
        async def native_error(request, exc):
            return JSONResponse(
                {"object": "error", "code": exc.code, "message": str(exc)},
                status_code=exc.status,
            )

    def _register_routes(self):
        super()._register_routes()
        self.app.add_api_route("/internal/stateful/v3/delete", self.session_delete, methods=["POST"])
        self.app.add_api_route("/internal/stateful/v3/status", self.session_status, methods=["POST"])

    def _objects(self):
        if not self.engine.is_ready():
            raise RouterError(503, "native_unavailable", "Single engine is unavailable")
        if self.objects is None:
            cache = self.core.kv_cache_manager
            runtime = self.engine.config.resolve_runtime_config()
            contract = Contract(
                "qwen3-14b-paged",
                self.model_id,
                digest(
                    {
                        "deployment": "single",
                        "session_kv_mode": self.session_kv_mode,
                        "runtime": runtime,
                        "groups": cache.group_specs,
                    }
                ),
                (Component("key", "kv", "kv"), Component("value", "kv", "kv")),
            )
            node = Node(
                "single",
                "single",
                self.boot,
                contract,
                uuid.uuid4().hex,
                Topology.for_tp(1, 1),
            )
            self.objects = NativeObjects(
                cache,
                node,
                SimpleNamespace(active={}),
                self.model_identity["fingerprint"],
                session_kv_mode="input",
            )
            self.core.scheduler.native_objects = self.objects
        return self.objects

    def _budget(self, tokens, config):
        if len(tokens) + config.max_new_tokens > self.engine.config.resolve_runtime_config().max_seq_len:
            raise RouterError(
                400,
                "context_length_exceeded",
                "Input tokens plus output budget exceed model context",
            )

    @staticmethod
    def _validate_generation(config):
        if config.temperature != 0.0 or config.top_p != 1.0 or config.top_k is not None:
            raise ValueError("Local Qwen sessions require greedy sampling")

    async def _health(self):
        ready = self.engine.is_ready()
        return JSONResponse(
            {"status": "ok" if ready else "not_ready", "deployment": "single"},
            status_code=200 if ready else 503,
        )
