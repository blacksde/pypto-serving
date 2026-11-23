# Copyright (c) PyPTO Contributors.
"""Model-family KV contract tests for local sessions."""

from types import SimpleNamespace

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.router.http import RouterError
from pypto_serving.serving.server.session_adapter import SessionKVAdapter


def test_qwen_adapter_describes_single_page_group():
    cache = SimpleNamespace(
        group_specs=(KVCacheGroupSpec("kv", (0,), KVCacheSpec(128, 128), 4),)
    )

    adapter = SessionKVAdapter.from_cache("qwen", "Qwen3-14B", cache)

    assert adapter.adapter_id == "qwen3-14b-paged-input-v1"
    assert adapter.block_size == 128
    assert adapter.group_names == ("kv",)
    assert adapter.supports_warm_prefix is True
    assert adapter.topology_metadata((2, 3))["rank_count"] == 2


def test_grouped_adapter_can_describe_a_model_with_replay_only_state():
    cache = SimpleNamespace(
        group_specs=(
            KVCacheGroupSpec("attention", (0,), KVCacheSpec(128, 128), 4),
            KVCacheGroupSpec(
                "window",
                (1,),
                KVCacheSpec(128, 128),
                2,
                sliding_window=128,
                prefix_cacheable=False,
            ),
        )
    )

    adapter = SessionKVAdapter.from_cache("glm53", "GLM", cache)

    assert adapter.group_names == ("attention", "window")
    assert adapter.cacheable_groups == ("attention",)
    assert adapter.supports_warm_prefix is False


def test_qwen_adapter_rejects_a_non_qwen_page_layout():
    cache = SimpleNamespace(
        group_specs=(KVCacheGroupSpec("kv", (0,), KVCacheSpec(640, 640), 4),)
    )

    with pytest.raises(RouterError, match="128-token"):
        SessionKVAdapter.from_cache("qwen", "Qwen3-14B", cache)
