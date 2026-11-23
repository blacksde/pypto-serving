# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Local session execution uses real SQL and CPU worker ACKs."""

import asyncio
from functools import partial
import json
import os
import uuid

import pytest

from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.config.types import GenerateConfig, KVCacheGroupSpec, KVCacheSpec, RuntimeConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig, ReplicaEngineCore
from pypto_serving.serving.server.native import NativeServer, SessionScheduler
from pypto_serving.serving.server.server import ChatCompletionRequest
from tests.unit.serving.server.session_fixtures import Tokenizer, Worker, checkpoint, incoming


def test_local_history_barrier_hot_continuation_and_cold_fallback(tmp_path, monkeypatch):
    url = os.getenv("PYPTO_SESSION_TEST_DATABASE_URL")
    if not url:
        pytest.skip("PYPTO_SESSION_TEST_DATABASE_URL is required")
    monkeypatch.setenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "single-stream-credential")
    monkeypatch.setenv("PYPTO_SESSION_DATABASE_URL", url)
    monkeypatch.setenv("PYPTO_SESSION_SCHEMA", "session_test_" + uuid.uuid4().hex)
    monkeypatch.setenv("PYPTO_SESSION_SECRET", "test-secret")
    checkpoint(tmp_path)

    async def run():
        worker = Worker()
        worker.released.set()
        runtime = RuntimeConfig(
            page_size=128,
            max_batch_size=2,
            max_seq_len=512,
            max_num_batched_tokens=512,
            kv_cache_groups=(KVCacheGroupSpec("kv", (0,), KVCacheSpec(128, 128), 4, num_blocks=16),),
        )
        config = EngineConfig(
            model_id="Qwen3-14B",
            model_dir=str(tmp_path),
            runtime_config=runtime,
            max_num_running_reqs=2,
            max_num_scheduled_tokens=512,
            enable_prefix_cache=False,
        )
        engine = AsyncLLMEngine(
            config,
            Tokenizer(),
            core_factory=partial(
                ReplicaEngineCore,
                scheduler_factory=partial(SessionScheduler, session_kv_mode="input"),
                worker_factory=worker.factory,
            ),
        )
        await engine.start()
        server = NativeServer(engine, "Qwen3-14B", GenerateConfig())
        await server.sessions.start()
        body = dict(
            model="Qwen3-14B",
            messages=[dict(role="user", content="first")],
            max_completion_tokens=2,
            temperature=0,
            top_p=1,
            chat_template_kwargs={"enable_thinking": False},
            stateful_context=dict(
                protocol="pypto-session-v3",
                response_id="resp_A",
                tenant_id="one",
                input_items=[dict(type="message", role="user", content="first")],
            ),
        )
        try:
            response = await server._chat_completions(
                ChatCompletionRequest.model_validate(body), incoming("/chat", {})
            )
            result = json.loads(response.body)
            assert result["choices"][0]["finish_reason"] == "length"
            probe, parent = await server.sessions.db(
                server.sessions.store.begin_request, "probe", "one", "resp_A"
            )
            assert parent.history["output_token_ids"] == [7, 8]
            await server.sessions.db(server.sessions.store.abort_request, probe)
            await asyncio.wait_for(server.sessions.candidates.join(), 3)
            body["messages"] += [dict(role="assistant", content="7 8"), dict(role="user", content="next")]
            body["stateful_context"].update(
                response_id="resp_B",
                parent_response_id="resp_A",
                policy="strict",
                input_items=[dict(type="message", role="user", content="next")],
            )
            await server._chat_completions(ChatCompletionRequest.model_validate(body), incoming("/chat", {}))
            await asyncio.wait_for(server.sessions.candidates.join(), 3)
            probe, child = await server.sessions.db(
                server.sessions.store.begin_request, "probe_b", "one", "resp_B"
            )
            assert child.binding["mounted_tokens"] == 128
            await server.sessions.db(server.sessions.store.abort_request, probe)
            server._objects().release(parent.binding) if parent.binding else None
            # Force physical loss independently of the committed parent history.
            probe, old = await server.sessions.db(
                server.sessions.store.begin_request, "probe_old", "one", "resp_A"
            )
            await server.sessions.db(server.sessions.store.abort_request, probe)
            server._objects().release(old.binding)
            body["stateful_context"].update(response_id="resp_C", policy="availability")
            await server._chat_completions(ChatCompletionRequest.model_validate(body), incoming("/chat", {}))
            body["stateful_context"].update(response_id="resp_D", policy="strict")
            with pytest.raises(Exception, match="Parent KV"):
                await server._chat_completions(
                    ChatCompletionRequest.model_validate(body), incoming("/chat", {})
                )
            assert not server.core.scheduler.native_requests
        finally:
            await server.sessions.close()
            if server.objects:
                server.objects.close()
            await engine.stop()

    asyncio.run(run())


def test_local_session_routes_parent_to_the_same_replica(tmp_path, monkeypatch):
    url = os.getenv("PYPTO_SESSION_TEST_DATABASE_URL")
    if not url:
        pytest.skip("PYPTO_SESSION_TEST_DATABASE_URL is required")
    monkeypatch.setenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "single-stream-credential")
    monkeypatch.setenv("PYPTO_SESSION_DATABASE_URL", url)
    monkeypatch.setenv("PYPTO_SESSION_SCHEMA", "replica_test_" + uuid.uuid4().hex)
    monkeypatch.setenv("PYPTO_SESSION_SECRET", "test-secret")
    checkpoint(tmp_path)

    async def run():
        workers = [Worker(), Worker()]
        for worker in workers:
            worker.released.set()

        def worker_factory(config):
            return workers[config.dp_rank].factory(config)

        runtime = RuntimeConfig(
            page_size=128,
            max_batch_size=2,
            max_seq_len=512,
            max_num_batched_tokens=512,
            kv_cache_groups=(
                KVCacheGroupSpec(
                    "kv", (0,), KVCacheSpec(128, 128), 4,
                    num_blocks=16, num_partitions=2, max_requests_per_partition=2,
                ),
            ),
        )
        config = EngineConfig(
            model_id="Qwen3-14B",
            model_dir=str(tmp_path),
            runtime_config=runtime,
            parallel_config=ParallelConfig(
                data_parallel_size=2, tensor_parallel_size=1, devices=(0, 1),
            ),
            max_num_running_reqs=2,
            max_num_scheduled_tokens=512,
            enable_prefix_cache=False,
        )
        engine = AsyncLLMEngine(
            config,
            Tokenizer(),
            core_factory=partial(
                ReplicaEngineCore,
                scheduler_factory=partial(SessionScheduler, session_kv_mode="input"),
                worker_factory=worker_factory,
            ),
        )
        await engine.start()
        server = NativeServer(engine, "Qwen3-14B", GenerateConfig())
        await server.sessions.start()
        body = dict(
            model="Qwen3-14B",
            messages=[dict(role="user", content="first")],
            max_completion_tokens=2,
            temperature=0,
            top_p=1,
            chat_template_kwargs={"enable_thinking": False},
            stateful_context=dict(
                protocol="pypto-session-v3",
                response_id="resp_replica_A",
                tenant_id="one",
                input_items=[dict(type="message", role="user", content="first")],
            ),
        )
        try:
            response = await server._chat_completions(
                ChatCompletionRequest.model_validate(body), incoming("/chat", {})
            )
            assert json.loads(response.body)["choices"][0]["finish_reason"] == "length"
            await asyncio.wait_for(server.sessions.candidates.join(), 3)
            probe, parent = await server.sessions.db(
                server.sessions.store.begin_request, "probe_replica", "one", "resp_replica_A"
            )
            await server.sessions.db(server.sessions.store.abort_request, probe)
            parent_replica = server.replica_for_receipt(parent.binding)
            assert parent_replica in (0, 1)

            body["messages"] += [
                dict(role="assistant", content="7 8"),
                dict(role="user", content="next"),
            ]
            body["stateful_context"].update(
                response_id="resp_replica_B",
                parent_response_id="resp_replica_A",
                policy="strict",
                input_items=[dict(type="message", role="user", content="next")],
            )
            await server._chat_completions(
                ChatCompletionRequest.model_validate(body), incoming("/chat", {})
            )
            await asyncio.wait_for(server.sessions.candidates.join(), 3)
            probe, child = await server.sessions.db(
                server.sessions.store.begin_request, "probe_replica_b", "one", "resp_replica_B"
            )
            await server.sessions.db(server.sessions.store.abort_request, probe)
            assert server.replica_for_receipt(child.binding) == parent_replica
            assert child.binding["mounted_tokens"] == 128
        finally:
            await server.sessions.close()
            server.close_objects()
            await engine.stop()

    asyncio.run(run())

@pytest.mark.parametrize("outcome", ["success", "failure", "disconnect"])
def test_local_stream_holds_success_until_history_commit(tmp_path, monkeypatch, outcome):
    import threading
    from stateful_session.database import CatalogError

    url = os.getenv("PYPTO_SESSION_TEST_DATABASE_URL")
    if not url:
        pytest.skip("PYPTO_SESSION_TEST_DATABASE_URL is required")
    monkeypatch.setenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "single-stream-credential")
    monkeypatch.setenv("PYPTO_SESSION_DATABASE_URL", url)
    monkeypatch.setenv("PYPTO_SESSION_SCHEMA", "stream_test_" + uuid.uuid4().hex)
    monkeypatch.setenv("PYPTO_SESSION_SECRET", "test-secret")
    checkpoint(tmp_path)

    async def run():
        worker = Worker()
        worker.released.set()
        runtime = RuntimeConfig(
            page_size=128,
            max_batch_size=2,
            max_seq_len=512,
            max_num_batched_tokens=512,
            kv_cache_groups=(KVCacheGroupSpec("kv", (0,), KVCacheSpec(128, 128), 4, num_blocks=16),),
        )
        config = EngineConfig(
            model_id="Qwen3-14B",
            model_dir=str(tmp_path),
            runtime_config=runtime,
            max_num_running_reqs=2,
            max_num_scheduled_tokens=512,
            enable_prefix_cache=False,
        )
        engine = AsyncLLMEngine(
            config,
            Tokenizer(),
            core_factory=partial(
                ReplicaEngineCore,
                scheduler_factory=partial(SessionScheduler, session_kv_mode="input"),
                worker_factory=worker.factory,
            ),
        )
        await engine.start()
        server = NativeServer(engine, "Qwen3-14B", GenerateConfig())
        await server.sessions.start()
        entered, release = threading.Event(), threading.Event()
        commit = server.sessions.store.commit_history
        execution_contexts = []

        def blocked(context, history):
            execution_contexts.append(context)
            entered.set()
            assert release.wait(3)
            if outcome == "failure":
                raise CatalogError("injected_commit_failure", "Commit rejected")
            return commit(context, history)

        monkeypatch.setattr(server.sessions.store, "commit_history", blocked)
        request = ChatCompletionRequest.model_validate(
            dict(
                model="Qwen3-14B",
                stream=True,
                messages=[dict(role="user", content="first")],
                max_completion_tokens=2,
                temperature=0,
                top_p=1,
                chat_template_kwargs={"enable_thinking": False},
                stateful_context=dict(
                    protocol="pypto-session-v3",
                    response_id="resp_stream",
                    tenant_id="one",
                    input_items=[dict(type="message", role="user", content="first")],
                ),
            )
        )
        frames = []
        response = await server._chat_completions(request, incoming("/chat", {}))

        async def collect():
            async for frame in response.body_iterator:
                if frame.startswith("data: {"):
                    frames.append(json.loads(frame[6:]))

        collector = asyncio.create_task(collect())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert frames and not any(c.get("finish_reason") for f in frames for c in f.get("choices", []))
            if outcome == "disconnect":
                collector.cancel()
                await asyncio.sleep(0.01)
                assert any(not task.cancelled() for task in server.sessions.pending)
            release.set()
            await asyncio.gather(collector, return_exceptions=True)
            status = await server.sessions.db(server.sessions.store.execution_status, execution_contexts[0])
            if outcome == "failure":
                assert status["status"] == "failed"
                assert any("error" in f for f in frames)
                assert not any(c.get("finish_reason") for f in frames for c in f.get("choices", []))
            else:
                assert status["status"] == "committed"
                if outcome == "success":
                    assert any(
                        c.get("finish_reason") == "length" for f in frames for c in f.get("choices", [])
                    )
            assert not server.core.scheduler.native_requests
        finally:
            release.set()
            await response.body_iterator.aclose()
            await server.sessions.close()
            if server.objects:
                server.objects.close()
            await engine.stop()

    asyncio.run(run())
