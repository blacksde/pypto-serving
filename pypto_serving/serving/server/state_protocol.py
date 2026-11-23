# Copyright (c) PyPTO Contributors.
"""Small local identity protocol for single-rank session KV ownership."""

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json


def _builtins(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _builtins(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtins(item) for item in value]
    return value


def digest(value):
    encoded = json.dumps(_builtins(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class Component:
    name: str
    group: str
    atomic: str
    final: bool = False
    replicated: bool = True


@dataclass(frozen=True)
class Contract:
    adapter: str
    model: str
    layout: str
    components: tuple[Component, ...]


@dataclass(frozen=True)
class Topology:
    ranks: int
    groups: tuple[tuple[int, ...], ...]

    @classmethod
    def for_tp(cls, ranks, tp):
        if type(ranks) is not int or type(tp) is not int or ranks <= 0 or tp <= 0 or ranks % tp:
            raise ValueError("invalid ranks/TP configuration")
        return cls(ranks, tuple(tuple(range(start, start + tp)) for start in range(0, ranks, tp)))


@dataclass(frozen=True)
class Node:
    node_id: str
    role: str
    boot: str
    contract: Contract
    registry: str
    topology: Topology
    provider: str = "local"
    prefix: bool = True
    healthy: bool = True


@dataclass(frozen=True)
class Binding:
    request: str
    input_digest: str


@dataclass(frozen=True)
class Continuation:
    tokens: tuple[int, ...]
    generation: dict
    parser: object = None
    constraint: object = None
