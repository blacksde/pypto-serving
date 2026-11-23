# Copyright (c) PyPTO Contributors.
"""Model-specific KV contracts used by local stateful sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pypto_serving.router.http import RouterError


@dataclass(frozen=True)
class SessionKVAdapter:
    """Validate and describe the KV layout shared by session receipts."""

    model_family: str
    model_id: str
    adapter_id: str
    block_size: int
    group_names: tuple[str, ...]
    cacheable_groups: tuple[str, ...]
    supports_warm_prefix: bool

    @classmethod
    def from_cache(cls, model_family, model_id, cache):
        groups = tuple(cache.group_specs)
        if not groups:
            raise RouterError(400, "session_kv_unsupported", "Model has no grouped KV cache contract")
        names = tuple(group.name for group in groups)
        if len(set(names)) != len(names):
            raise RouterError(400, "session_kv_unsupported", "KV cache group names must be unique")
        cacheable = tuple(group.name for group in groups if group.prefix_cacheable)
        if not cacheable:
            raise RouterError(400, "session_kv_unsupported", "Model has no cacheable KV group")
        capacities = {group.spec.token_capacity for group in groups if group.prefix_cacheable}
        if len(capacities) != 1:
            raise RouterError(400, "session_kv_unsupported", "Cacheable KV groups must share token capacity")
        if any(
            group.sliding_window is not None
            or group.is_eagle_group
            or group.is_checkpoint_state
            for group in groups
            if group.prefix_cacheable
        ):
            raise RouterError(
                400,
                "session_kv_unsupported",
                "Cacheable session groups must be non-rolling and non-checkpoint state",
            )
        adapter_id = f"{model_family}-grouped-input-v1"
        if model_family == "qwen":
            if names != ("kv",) or capacities != {128}:
                raise RouterError(400, "session_kv_unsupported", "Qwen sessions require one 128-token kv group")
            adapter_id = "qwen3-14b-paged-input-v1"
        return cls(
            model_family=model_family,
            model_id=model_id,
            adapter_id=adapter_id,
            block_size=next(iter(capacities)),
            group_names=names,
            cacheable_groups=cacheable,
            supports_warm_prefix=len(cacheable) == len(names),
        )

    def retained_boundary(self, confirmed_tokens: int) -> int:
        return (confirmed_tokens // self.block_size) * self.block_size

    def validate_tables(self, tables: dict[str, Sequence[int]]) -> None:
        if set(tables) != set(self.group_names):
            raise ValueError("KV receipt groups do not match the model session contract")
        for name in self.group_names:
            if name not in self.cacheable_groups and tables[name]:
                raise ValueError(f"non-cacheable group {name!r} cannot be retained as a prefix")

    def topology_metadata(self, device_ids: Sequence[int]) -> dict:
        devices = tuple(int(device) for device in device_ids)
        return {
            "rank_count": len(devices),
            "device_ids": list(devices),
            "group_names": list(self.group_names),
            "cacheable_groups": list(self.cacheable_groups),
            "block_size": self.block_size,
        }
