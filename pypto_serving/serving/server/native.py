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
from .session_adapter import SessionKVAdapter
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

    def __init__(self, engine, model_id, generate, *, session_kv_mode="input", model_family="qwen"):
        self._cores = engine.cores
        if not self._cores:
            raise ValueError("Local sessions require at least one serving core")
        if any(not isinstance(core.scheduler, SessionScheduler) for core in self._cores):
            raise ValueError("Local sessions require SessionScheduler on every serving core")
        self.core = self._cores[0]
        self.model_family = model_family
        if session_kv_mode != "input":
            raise ValueError("Local sessions support input KV retention only")
        self.session_kv_mode = session_kv_mode
        self.boot = uuid.uuid4().hex
        self.model_identity = load_model_identity(engine.config.model_dir, model_id, engine.tokenizer)
        self.objects = None
        self._objects_by_replica = {}
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
                "live_objects": sum(len(objects.objects) for objects in self._objects_by_replica.values()),
                "pending_publications": self.sessions.candidates.qsize(),
                "active_executions": sum(len(core.scheduler.native_requests) for core in self._cores),
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

    def _objects(self, replica_idx=0):
        if not self.engine.is_ready():
            raise RouterError(503, "native_unavailable", "Session engine is unavailable")
        if not 0 <= replica_idx < len(self._cores):
            raise RouterError(503, "native_unavailable", "Session replica is unavailable")
        if replica_idx not in self._objects_by_replica:
            core = self._cores[replica_idx]
            cache = core.kv_cache_manager
            runtime = self.engine.config.resolve_runtime_config()
            adapter = SessionKVAdapter.from_cache(self.model_family, self.model_id, cache)
            devices = core.config.worker_device_ids()
            contract = Contract(
                adapter.adapter_id,
                self.model_id,
                digest(
                    {
                        "deployment": "replica",
                        "replica_idx": replica_idx,
                        "model_family": self.model_family,
                        "session_kv_mode": self.session_kv_mode,
                        "runtime": runtime,
                        "groups": cache.group_specs,
                    }
                ),
                (Component("key", "kv", "kv"), Component("value", "kv", "kv")),
            )
            node = Node(
                f"replica-{replica_idx}",
                "replica",
                self.boot,
                contract,
                uuid.uuid4().hex,
                Topology.for_tp(len(devices), len(devices)),
            )
            objects = NativeObjects(
                cache,
                node,
                SimpleNamespace(active={}),
                self.model_identity["fingerprint"],
                session_kv_mode="input",
                adapter=adapter,
            )
            self._objects_by_replica[replica_idx] = objects
            core.scheduler.native_objects = objects
            if replica_idx == 0:
                self.objects = objects
        return self._objects_by_replica[replica_idx]

    def select_replica(self):
        return self.engine.select_replica()

    def replica_for_receipt(self, receipt):
        node_id = receipt.get("node_id") if isinstance(receipt, dict) else None
        if isinstance(node_id, str) and node_id.startswith("replica-"):
            try:
                index = int(node_id.rsplit("-", 1)[1])
            except ValueError:
                return None
            return index if 0 <= index < len(self._cores) else None
        return None

    def release_receipt(self, receipt):
        replica_idx = self.replica_for_receipt(receipt)
        if replica_idx is None:
            return
        self._objects(replica_idx).release(receipt)

    def close_objects(self):
        for objects in tuple(self._objects_by_replica.values()):
            objects.close()

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
