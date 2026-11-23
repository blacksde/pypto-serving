# Copyright (c) PyPTO Contributors.
"""Authenticated local ownership of retained Qwen KV pages."""

from dataclasses import dataclass
import hashlib
import hmac
import os
import re
import secrets
import uuid

from pypto_serving.router.http import RouterError
from pypto_serving.serving import request_timing as timing
from .state_protocol import Binding, digest


def authorize(headers):
    token = os.getenv("PYPTO_STATEFUL_INTERNAL_TOKEN", "")
    supplied = headers.get("authorization", "")
    if not token or not secrets.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
        raise RouterError(401, "invalid_internal_credentials", "Invalid internal credentials")


def owner(payload):
    values = {}
    for name in ("tenant_id", "cache_salt", "version_id"):
        value = payload.get(name)
        if not isinstance(value, str) or not value or len(value) > 256:
            raise ValueError(f"{name} must be a non-empty string of at most 256 characters")
        values[name] = value
    values["cache_salt_digest"] = hashlib.sha256(values.pop("cache_salt").encode()).hexdigest()
    return values


@dataclass
class _Object:
    receipt: dict
    blocks: dict[str, list[int]]
    partition: int
    binding: Binding
    tokens: tuple[int, ...]
    snapshot: object


class NativeObjects:
    """Keep one signed, boot-scoped reference to each retained input prefix."""

    def __init__(self, cache, node, pins, model_identity=None, *, session_kv_mode="input"):
        if (
            node.contract.adapter != "qwen3-14b-paged"
            or cache.group_names != ("kv",)
            or cache.group_specs[0].spec.token_capacity != 128
        ):
            raise ValueError("Local retention requires the Qwen3-14B paged cache layout")
        if session_kv_mode != "input":
            raise ValueError("Local retention supports input KV only")
        self.session_kv_mode = session_kv_mode
        self.cache, self.node, self.pins = cache, node, pins
        self.model_identity = model_identity
        self.secret = secrets.token_bytes(32)
        self.objects: dict[str, _Object] = {}
        self.requests: dict[str, str] = {}

    def _signature(self, receipt, nonce):
        return hmac.new(self.secret, (nonce + ":" + digest(receipt)).encode(), hashlib.sha256).hexdigest()

    def parent_precheck(self, parent, scope):
        if parent is None:
            return {"state": "absent"}
        if (
            parent.get("node_id") == self.node.node_id
            and parent.get("model") == self.node.contract.model
            and (parent.get("boot") != self.node.boot or parent.get("registry") != self.node.registry)
        ):
            return {"state": "missing", "reason": "owner_boot_changed"}
        key = self._check(parent)
        if any(parent.get(key_name) != scope[key_name] for key_name in ("tenant_id", "cache_salt_digest")):
            raise RouterError(403, "native_owner_mismatch", "Parent belongs to another owner")
        status = self.status(parent)
        if status["request_active"]:
            raise RouterError(409, "native_parent_busy", "Parent has not retired")
        return {"state": "live" if key in self.objects else "missing", "reason": None}

    @timing.timed("native.retain", engine=True)
    def retain(self, entry, chunk):
        previous = self.requests.get(entry.binding.request)
        if previous is not None:
            raise RuntimeError("Request already owns a local object")
        actual = tuple(entry.continuation.tokens)
        if not chunk.final or chunk.end != len(actual):
            raise ValueError("Local retain requires the confirmed prompt")
        boundary = chunk.end // 128 * 128
        tables = {"kv": list(chunk.blocks["kv"][: boundary // 128])}
        if len(tables["kv"]) != boundary // 128:
            raise ValueError("Confirmed prompt does not have enough complete physical pages")
        receipt = dict(
            entry.native_scope,
            node_id=self.node.node_id,
            boot=self.node.boot,
            registry=self.node.registry,
            model=self.node.contract.model,
            layout=self.node.contract.layout,
            request_id=entry.binding.request,
            partition=chunk.partition,
            durability="volatile",
            retained_tokens=boundary,
            confirmed_tokens=chunk.end,
            block_size=128,
            prefix_digest=digest(list(actual[:boundary])),
        )
        if self.model_identity is not None:
            receipt["model_identity_digest"] = self.model_identity
        if getattr(entry, "native_parent", None) is not None:
            receipt.update(mounted_tokens=entry.mounted_tokens, mounted_object_id=entry.native_parent["object_id"])
        snapshot = self.cache.retain_group_block_snapshot(tables, chunk.partition)
        key = None
        try:
            nonce = uuid.uuid4().hex
            key = nonce + "." + self._signature(receipt, nonce)
            receipt["object_id"] = key
            self.objects[key] = _Object(
                receipt, tables, chunk.partition, entry.binding, actual[:boundary], snapshot
            )
            self.requests[entry.binding.request] = key
        except BaseException:
            if key is not None:
                self.objects.pop(key, None)
                self.requests.pop(entry.binding.request, None)
            self.cache.release_group_block_snapshot(snapshot)
            raise
        return dict(receipt)

    def _check(self, receipt):
        if not isinstance(receipt, dict):
            raise ValueError("native must be an object receipt")
        expected = dict(
            node_id=self.node.node_id,
            boot=self.node.boot,
            registry=self.node.registry,
            model=self.node.contract.model,
            layout=self.node.contract.layout,
        )
        if self.model_identity is not None:
            expected["model_identity_digest"] = self.model_identity
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise RouterError(409, "native_identity_mismatch", "Native object belongs to another engine")
        unsigned = dict(receipt)
        key = unsigned.pop("object_id", "")
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{32}\.[0-9a-f]{64}", key):
            raise RouterError(404, "native_object_not_found", "Unknown native object")
        nonce, signature = key.split(".")
        if not hmac.compare_digest(signature, self._signature(unsigned, nonce)):
            raise RouterError(403, "native_owner_mismatch", "Native object receipt does not match its owner")
        return key

    def status(self, receipt):
        key = self._check(receipt)
        record = self.objects.get(key)
        request = receipt["request_id"]
        return {
            "live": record is not None,
            "session_references": int(record is not None),
            "request_active": self.cache.group_request_partition(request) is not None,
            "transfer_references": int(request in self.pins.active),
            "free_blocks": self.cache.group_num_free_blocks("kv", receipt["partition"]),
            "pages": {name: list(ids) for name, ids in record.blocks.items()} if record else {},
        }

    def release(self, receipt):
        key = self._check(receipt)
        record = self.objects.get(key)
        if record is not None:
            self.cache.release_group_block_snapshot(record.snapshot)
            del self.objects[key]
            self.requests.pop(record.receipt["request_id"], None)
        return self.status(receipt)

    @timing.timed("native.mount.verify", engine=True)
    def mount(self, receipt, scope, tokens, *, mount_tokens=None):
        from pypto_serving.serving.memory.kv_cache import ConfirmedGroupPrefix

        key = self._check(receipt)
        record = self.objects.get(key)
        if record is None:
            raise RouterError(409, "session_kv_lost", "The parent local object is no longer live")
        if any(receipt.get(name) != scope[name] for name in ("tenant_id", "cache_salt_digest")):
            raise RouterError(403, "native_owner_mismatch", "The parent object has a different owner")
        boundary = receipt["retained_tokens"]
        if mount_tokens is None:
            if boundary >= len(tokens) or digest(list(tokens[:boundary])) != receipt["prefix_digest"]:
                raise RouterError(409, "native_prefix_mismatch", "Prompt differs from the confirmed parent prefix")
        else:
            common = 0
            for old, new in zip(record.tokens, tokens):
                if old != new:
                    break
                common += 1
            expected = min(common, max(0, len(tokens) - 1)) // 128 * 128
            if type(mount_tokens) is not int or mount_tokens != expected:
                raise RouterError(409, "native_prefix_mismatch", "Requested mount differs from the actual prefix")
            boundary = mount_tokens
        return ConfirmedGroupPrefix(
            boundary,
            record.partition,
            {name: tuple(ids[: boundary // 128]) for name, ids in record.blocks.items()},
        )

    def discard_request(self, request):
        key = self.requests.get(request)
        if key is not None:
            self.release(self.objects[key].receipt)

    def close(self):
        for record in list(self.objects.values()):
            self.release(record.receipt)
