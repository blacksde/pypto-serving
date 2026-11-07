# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the online documentation for details.
# -----------------------------------------------------------------------------------------------------------
"""Local host-staged KV transfer for single-host PD deployments.

The Mooncake provider moves device bytes between allocation owners through a
native P2P engine that must be installed on every participating node and
driven through PyPTO chip-child services. A single-host 1P1D deployment can
instead stage pages through each serving worker's own L3 ``DistributedWorker``:
the source reads device memory with ``copy_from``, the destination writes it
back with ``copy_to``, and the destination acknowledges every frame so
completion semantics match the control protocol (a job is done only after
the destination confirms the write). Regions keep the same ``(endpoint,
address)`` handle shape as Mooncake, so protocol version 3 peers interoperate
unchanged.

The provider runs inside the serving worker process itself -- there is no
separate allocation-owner child and therefore no requirement for the PyPTO
``chip_service_factories``/``retain_service_tensor`` interfaces. Device
addresses are allocation bases the L3 worker already tracks, and every copy
takes the worker's per-device dispatch gate (held from run submission until
settlement), so transfers from the PD write pool serialize safely against
dispatches.
"""

import ctypes
import logging
import os
import socket
import struct
import threading

from pypto_serving.model.common.runner.l3_dispatch import device_gate
from pypto_serving.tools.observe import span
from .transfer import Region, TransferError
from .worker import WorkerRuntime

logger = logging.getLogger(__name__)

_FRAME = struct.Struct("!QI")  # destination device address, payload length
_ACK = struct.Struct("!I")  # receiver status for one frame
_FRAME_LIMIT = 4 << 20  # host staging chunk for one TCP frame
_CONNECT_TIMEOUT = 10.0
_RECV_TIMEOUT = 120.0

_staging = threading.local()


def _staging_buffer(size):
    """Reusable per-thread host staging buffer of at least ``size`` bytes."""
    buffer = getattr(_staging, "buffer", None)
    if buffer is None or len(buffer) < size:
        buffer = _staging.buffer = ctypes.create_string_buffer(max(size, _FRAME_LIMIT))
    return buffer


class HostTransfer:
    """Byte-range provider staging through the caller's own L3 worker."""

    #: This provider runs beside the allocations; no PyPTO chip services needed.
    REQUIRES_CHIP_SERVICES = False

    @staticmethod
    def check_available():
        from importlib import import_module

        try:
            module = import_module("pypto.runtime.distributed_runner")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("local PD transfer requires a pypto runtime: " + str(exc)) from exc
        worker = getattr(module, "DistributedWorker", None)
        if worker is None or not all(hasattr(worker, name) for name in ("copy_to", "copy_from")):
            raise ValueError("local PD transfer requires PyPTO DistributedWorker copy interfaces")

    def __init__(self, hostname, worker):
        if not all(hasattr(worker, name) for name in ("copy_to", "copy_from")):
            raise ValueError("local PD transfer requires the L3 worker copy interfaces")
        self.worker = worker
        # Copies must not race an in-flight dispatch on this worker: take the
        # same per-worker gate the dispatch path holds from submit to settle.
        self.gate = device_gate(worker)
        self.poisoned = False
        self.hostname = hostname
        self.local, self.remote = {}, {}
        self.lock = threading.Lock()
        self.connections = {}
        self.stopped = threading.Event()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("", 0))
        self.listener.listen(16)
        self.endpoint = f"{hostname}:{self.listener.getsockname()[1]}"
        self.thread = threading.Thread(target=self._serve, name="pd-local-receive", daemon=True)
        self.thread.start()

    def _check(self):
        if self.poisoned:
            raise TransferError("provider unavailable", uncertain=True)

    def register(self, boot, rank, name, address, extent, worker_id=0):
        self._check()
        if (not boot or type(address) is not int or address <= 0 or address % 64
                or type(extent) is not int or extent <= 0 or extent % 64
                or address + extent > 1 << 64 or name in self.local):
            raise ValueError("invalid or duplicate device region")
        region = Region(boot, rank, name, extent,
                        {"endpoint": self.endpoint, "address": address, "worker_id": int(worker_id)})
        self.local[name] = region
        return region

    def install(self, regions):
        self._check()
        for region in regions:
            address, endpoint = region.handle.get("address"), region.handle.get("endpoint")
            if (not region.boot or region.rank < 0 or not isinstance(endpoint, str) or not endpoint
                    or type(address) is not int or address <= 0 or address % 64
                    or region.extent <= 0 or region.extent % 64 or address + region.extent > 1 << 64):
                raise ValueError("invalid destination region")
            old = self.remote.get(region.key)
            if old is not None and old != region:
                raise ValueError("registration changed within the same boot identity")
        self.remote.update((r.key, r) for r in regions)

    def _connection(self, endpoint):
        connection = self.connections.get(endpoint)
        if connection is None:
            host, _, port = endpoint.rpartition(":")
            client = socket.create_connection((host, int(port)), timeout=_CONNECT_TIMEOUT)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.settimeout(_RECV_TIMEOUT)
            connection = self.connections[endpoint] = client
        return connection

    def write(self, peer, copies):
        self._check()
        # (source region, destination region, in-region source offset,
        # in-region destination offset, total length) per copy.
        plan, endpoints = [], set()
        for copy in copies:
            src = self.local[copy.component]
            dst = self.remote[peer, copy.destination_rank, copy.component]
            if (copy.source_rank != src.rank or min(copy.source_offset, copy.destination_offset) < 0
                    or copy.length <= 0 or copy.source_offset + copy.length > src.extent
                    or copy.destination_offset + copy.length > dst.extent):
                raise ValueError("transfer exceeds registered owner/region")
            plan.append((src, dst, copy.source_offset, copy.destination_offset, copy.length))
            endpoints.add(dst.handle["endpoint"])
        if not copies:
            return
        if len(endpoints) != 1:
            raise ValueError("one local batch must target one rank owner")
        connection = self._connection(endpoints.pop())
        try:
            for src, dst, src_offset, dst_offset, length in plan:
                for offset in range(0, length, _FRAME_LIMIT):
                    size = min(_FRAME_LIMIT, length - offset)
                    payload = self._read_device(src, src_offset + offset, size)
                    connection.sendall(
                        _FRAME.pack(dst.handle["address"] + dst_offset + offset, size) + payload
                    )
                    status, = _ACK.unpack(self._receive_exact(connection, _ACK.size))
                    if status != 0:
                        raise TransferError("destination rejected a staged write", uncertain=True)
        except BaseException as exc:
            self.poisoned = True
            raise TransferError(f"local completion uncertain: {exc}", uncertain=True) from exc

    def _read_device(self, region, offset, size):
        buffer = _staging_buffer(size)
        with self.gate:
            self.worker.copy_from(ctypes.addressof(buffer), region.handle["address"], size,
                                  src_offset=offset, worker_id=region.handle["worker_id"])
        return buffer.raw[:size]

    def _write_device(self, region, offset, payload):
        buffer = _staging_buffer(len(payload))
        ctypes.memmove(ctypes.addressof(buffer), payload, len(payload))
        with self.gate:
            self.worker.copy_to(region.handle["address"], ctypes.addressof(buffer), len(payload),
                                dst_offset=offset, worker_id=region.handle["worker_id"])

    @staticmethod
    def _receive_exact(connection, size):
        parts = bytearray()
        while len(parts) < size:
            part = connection.recv(size - len(parts))
            if not part:
                raise EOFError("destination transfer connection closed")
            parts.extend(part)
        return bytes(parts)

    def _serve(self):
        while not self.stopped.is_set():
            try:
                connection, _ = self.listener.accept()
            except OSError:
                return
            connection.settimeout(_RECV_TIMEOUT)
            try:
                self._serve_connection(connection)
            except (EOFError, OSError, TransferError, ValueError):
                pass
            finally:
                connection.close()

    def _serve_connection(self, connection):
        while True:
            header = self._receive_exact(connection, _FRAME.size)
            address, length = _FRAME.unpack(header)
            if length > _FRAME_LIMIT:
                raise ValueError("invalid local transfer frame")
            payload = self._receive_exact(connection, length)
            status = 0
            try:
                self._dispatch_write(address, payload)
            except Exception as exc:  # noqa: BLE001
                logger.error("local PD destination write failed: %s", exc)
                status = 1
            connection.sendall(_ACK.pack(status))

    def _dispatch_write(self, address, payload):
        for region in self.local.values():
            base, extent = region.handle["address"], region.extent
            if base <= address < base + extent:
                self._write_device(region, address - base, payload)
                return
        raise ValueError("local transfer targets an unregistered address")

    def release(self):
        self._check()
        self.local.clear()
        self.remote.clear()

    def stop(self):
        self.stopped.set()
        try:
            self.listener.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.listener.close()
        with self.lock:
            connections, self.connections = self.connections, {}
        for connection in connections.values():
            connection.close()
        self.thread.join(5)


class HostOwner:
    """Owner that runs its provider beside the allocation, in this process.

    Mirrors the wire-protocol owner (``register``/``release``/``call``) but
    without a chip-child service: the provider object lives here and is
    created lazily at the first register, which supplies the L3 worker whose
    ``copy_from``/``copy_to`` move the bytes.
    """

    def __init__(self, boot, rank, hostname, provider="local"):
        self.boot, self.rank = boot, rank
        self.hostname, self.provider_name = hostname, provider
        self.provider = None
        self.worker = None
        self.lock = threading.Lock()
        self.retained = []
        self.uncertain = False

    def _bind(self, runtime):
        if self.provider is None:
            if self.provider_name != "local":
                raise ValueError("HostOwner only serves the local provider")
            self.worker = runtime
            self.provider = HostTransfer(self.hostname, runtime)
        elif self.worker is not runtime:
            raise ValueError("owner bound to a different L3 worker")

    def call(self, operation, **payload):
        with self.lock:
            if self.uncertain:
                raise TransferError("owner gated", uncertain=True)
            try:
                if operation == "probe":
                    return os.getpid()
                if operation == "register":
                    region = self.provider.register(self.boot, self.rank, payload["name"],
                                                    payload["address"], payload["extent"],
                                                    worker_id=payload.get("worker_id", 0))
                    self.retained.append(payload["name"])
                    return region
                if operation == "install":
                    self.provider.install(payload["regions"])
                    return None
                if operation == "write":
                    copies = payload["copies"]
                    with span(logger, "pd.local.write", cat="native-transfer",
                              **payload["observation"], bytes=sum(c.length for c in copies),
                              segments=len(copies)):
                        self.provider.write(payload["peer"], copies)
                    return None
                if operation == "release":
                    if self.provider is not None:
                        self.provider.release()
                    self.retained.clear()
                    return None
                if operation == "profile":
                    return None  # transfer spans share this process's profiler
                if operation == "stop":
                    return None
                raise ValueError(f"unknown owner operation {operation}")
            except BaseException as exc:
                self.uncertain = not isinstance(exc, TransferError) or exc.uncertain
                if self.uncertain:
                    raise TransferError(f"owner completion unknown: {exc}", uncertain=True) from exc
                raise

    def register(self, runtime, tensor, name):
        """Register a device tensor; the runtime/token protocol is chip-service only."""
        self._bind(runtime)
        return self.call("register", address=tensor.data_ptr, extent=tensor.nbytes, name=name,
                         worker_id=getattr(tensor, "worker_id", 0))

    def release(self):
        self.call("release")
        self.retained.clear()

    def ready(self):
        # The provider starts lazily at the first register; nothing to probe.
        return os.getpid()

    def close(self):
        with self.lock:
            if self.provider is not None:
                self.provider.stop()
                self.provider = None


class HostWorkerRuntime(WorkerRuntime):
    """Owner runtime whose transfers run inside this serving worker process."""

    def _new_owner(self, rank):
        return HostOwner(self.settings.boot, rank, self.settings.hostname, self.settings.provider)

    def worker_options(self):
        return {}  # no chip services; the DistributedWorker stays untouched
