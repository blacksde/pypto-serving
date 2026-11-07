# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the online documentation for details.
# -----------------------------------------------------------------------------------------------------------

"""Host-staged local KV transfer framing checks (no device required)."""

import threading
import time

import pytest

from pypto_serving.model.common.runner.l3_dispatch import PendingL3Dispatch, device_gate
from pypto_serving.pd.local_transfer import HostTransfer
from pypto_serving.pd.protocol import Copy
from pypto_serving.pd.transfer import Region, TransferError

POOL = 1 << 24  # 16 MiB pools


class FakeWorker:
    """Device-memory double tracking copy_to/copy_from against host bytes."""

    def __init__(self):
        self.device = bytearray(POOL)
        self.base = 1 << 20  # nonzero, 64-byte aligned
        self.copies = []

    def copy_from(self, dst_host, src_dev, nbytes, *, src_offset=0, **_):
        import ctypes

        self.copies.append(("from", src_dev, src_offset, nbytes))
        start = src_dev + src_offset - self.base
        assert 0 <= start and start + nbytes <= len(self.device), "read outside allocation"
        ctypes.memmove(dst_host, bytes(self.device[start:start + nbytes]), nbytes)

    def copy_to(self, dst_dev, src_host, nbytes, *, dst_offset=0, **_):
        import ctypes

        self.copies.append(("to", dst_dev, dst_offset, nbytes))
        start = dst_dev + dst_offset - self.base
        assert 0 <= start and start + nbytes <= len(self.device), "write outside allocation"
        self.device[start:start + nbytes] = ctypes.string_at(src_host, nbytes)


@pytest.fixture
def pair():
    """A sender and a receiver HostTransfer wired over real loopback sockets."""
    sender_worker, receiver_worker = FakeWorker(), FakeWorker()
    sender = HostTransfer("127.0.0.1", sender_worker)
    receiver = HostTransfer("127.0.0.1", receiver_worker)
    try:
        sender.register("boot-p", 0, "kv", sender_worker.base, POOL)
        receiver.register("boot-d", 0, "kv", receiver_worker.base, POOL)
        sender.install((Region("boot-d", 0, "kv", POOL,
                               {"endpoint": receiver.endpoint, "address": receiver_worker.base,
                                "worker_id": 0}),))
        yield sender, receiver, sender_worker, receiver_worker
    finally:
        sender.stop()
        receiver.stop()


def test_roundtrip_offsets_survive_framing(pair):
    sender, receiver, source, destination = pair
    source.device = bytearray(index & 0xFF for index in range(POOL))
    # Copy 5 MiB (frame limit is 4 MiB) from source offset 1 MiB to
    # destination offset 2 MiB: the in-region offsets must ride every frame.
    copy = Copy("kv", 0, 0, (1 << 20), (2 << 20), (5 << 20))
    sender.write("boot-d", [copy])
    expected = source.device[copy.source_offset:copy.source_offset + copy.length]
    got = destination.device[copy.destination_offset:copy.destination_offset + copy.length]
    assert bytes(got) == bytes(expected)
    reads = [c for c in source.copies if c[0] == "from"]
    writes = [c for c in destination.copies if c[0] == "to"]
    assert [c[3] for c in reads] == [4 << 20, 1 << 20]
    assert [c[3] for c in writes] == [4 << 20, 1 << 20]
    assert all(c[1] == source.base for c in reads) and all(c[1] == destination.base for c in writes)
    assert reads[0][2] == copy.source_offset and writes[0][2] == copy.destination_offset
    assert reads[1][2] == copy.source_offset + (4 << 20)
    assert writes[1][2] == copy.destination_offset + (4 << 20)


def test_out_of_range_copy_is_rejected_without_poisoning(pair):
    sender, _, source, destination = pair
    with pytest.raises(ValueError):
        sender.write("boot-d", [Copy("kv", 0, 0, 0, 0, POOL + 64)])
    # A rejected plan is a control error, not a native failure: the provider
    # must keep serving later, in-range copies.
    copy = Copy("kv", 0, 0, 0, 4096, 4096)
    sender.write("boot-d", [copy])
    assert bytes(destination.device[:4096]) == bytes(source.device[:4096])


def test_receiver_rejects_unregistered_address(pair):
    sender, receiver, source, destination = pair
    # Forge a frame to an address inside no registered region: the receiver
    # must reject the write and the sender must surface uncertain completion.
    handle = sender.remote[("boot-d", 0, "kv")].handle
    handle["address"] = destination.base + POOL + 64
    with pytest.raises(TransferError) as error:
        sender.write("boot-d", [Copy("kv", 0, 0, 0, 0, 64)])
    assert error.value.uncertain
    assert sender.poisoned
    assert receiver.poisoned is False  # the receiver stays serviceable


def test_concurrent_writes_serialize(pair):
    sender, receiver, source, destination = pair
    copies = [Copy("kv", 0, 0, index * 128, (1 << 16) + index * 128, 128)
              for index in range(32)]
    errors = []

    def run():
        try:
            sender.write("boot-d", copies)
        except TransferError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    for copy in copies:
        expected = source.device[copy.source_offset:copy.source_offset + copy.length]
        got = destination.device[copy.destination_offset:copy.destination_offset + copy.length]
        assert bytes(got) == bytes(expected)


def test_device_copies_wait_for_dispatch_gate(pair):
    """A copy in flight on the worker defers to a dispatch holding the gate."""
    sender, receiver, source, destination = pair
    source.device = bytearray((index * 7) & 0xFF for index in range(4096))
    gate = device_gate(destination)
    gate.acquire()

    def dispatch_step():
        time.sleep(0.4)  # a run settles after this long
        gate.release()

    worker_thread = threading.Thread(target=dispatch_step)
    start = time.monotonic()
    worker_thread.start()
    sender.write("boot-d", [Copy("kv", 0, 0, 0, 0, 4096)])
    elapsed = time.monotonic() - start
    worker_thread.join()
    assert elapsed >= 0.4, "the destination copy must wait for the dispatch gate"
    assert bytes(destination.device[:4096]) == bytes(source.device[:4096])


class _FakeHandle:
    def __init__(self, error=None):
        self.error = error

    def result(self):
        if self.error is not None:
            raise self.error


def test_pending_dispatch_releases_gate():
    """An async L3 dispatch releases its gate when the run settles (or fails)."""
    from pypto_serving.model.common.runner.l3_dispatch import threading as _threading

    for error in (None, RuntimeError("run failed")):
        gate = _threading.Lock()
        gate.acquire()
        pending = PendingL3Dispatch(
            worker=FakeWorker(), handle=_FakeHandle(error), uploaded=(), name="probe",
            gate=gate,
        )
        if error is None:
            pending.wait()
            pending.wait()  # an idempotent re-wait neither touches the gate nor raises
        else:
            with pytest.raises(RuntimeError):
                pending.wait()
            with pytest.raises(RuntimeError):
                pending.wait()  # replays the stored run error; gate untouched
        assert not gate.locked(), "gate must be free once the dispatch settles"
