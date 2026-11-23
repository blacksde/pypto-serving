# Copyright (c) PyPTO Contributors.
"""CPU fixtures for the database-backed local session tests."""

import hashlib
import json
from queue import Queue
from types import SimpleNamespace
import threading

from starlette.requests import Request

from pypto_serving.serving.server.ipc import ShutdownCommand, StepResult, decode_command, encode_result


class Tokenizer:
    eos_token_id = None
    bos_token_id = None

    def __init__(self, prompt_tokens=147):
        self.prompt_tokens = prompt_tokens
        self.tokenizer = SimpleNamespace(
            chat_template="cpu-template",
            backend_tokenizer=SimpleNamespace(
                to_str=lambda: json.dumps(dict(type="cpu-test", base_tokens=self.prompt_tokens))
            ),
            get_chat_template=lambda **kw: "cpu-template",
        )

    def encode(self, text):
        messages = json.loads(text)
        tokens = list(range(self.prompt_tokens))
        return tokens if len(messages) == 1 else tokens + [7, 8] + list(range(200, 221))

    def decode(self, ids):
        return " ".join(str(value) for value in ids)

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)


class Worker:
    """External worker IPC peer whose final decode acknowledgement is controllable."""

    def __init__(self):
        self.alive = True
        self.pid = 4242
        self.released = threading.Event()
        self.pending = Queue()
        self.last = {}
        self.commands = []
        self.fail_last = False

    def factory(self, config):
        ready = threading.Event()
        ready.set()
        return self, SimpleNamespace(put=self.put), SimpleNamespace(get=self.get), Queue(), ready, [16, 16]

    def put(self, raw):
        command = decode_command(raw)
        if isinstance(command, ShutdownCommand):
            self.alive = False
            return
        self.commands.append(command)
        tokens = {}
        for item in command.prefill_requests:
            self.last[item.request_id] = 7
            tokens[item.request_id] = [7]
        for item in command.decode_requests:
            value = self.last[item.request_id] + 1
            self.last[item.request_id] = value
            tokens[item.request_id] = [value]
        self.pending.put(
            (
                encode_result(
                    StepResult(
                        new_tokens=tokens,
                        step_id=command.step_id,
                        error="owned worker fault" if self.fail_last and command.decode_requests else None,
                    )
                ),
                bool(command.decode_requests),
            )
        )

    def get(self, timeout=None):
        value, held = self.pending.get(timeout=timeout)
        if held and not self.released.wait(5):
            raise TimeoutError("CPU worker ACK was not released")
        return value

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def close(self):
        pass

    def terminate(self):
        self.alive = False

    kill = terminate


def checkpoint(path):
    weight = b"CPU fixture weights"
    (path / "model.safetensors").write_bytes(weight)
    documents = {
        "config.json": {"vocab_size": 1024},
        "tokenizer.json": {"type": "cpu-test"},
        "tokenizer_config.json": {"chat_template": "cpu-template"},
        "model.safetensors.index.json": {"weight_map": {"weight": "model.safetensors"}},
        "download-verification.json": {
            "repo": "cpu-fixture",
            "revision": "revision-one",
            "files": [
                {
                    "file": "model.safetensors",
                    "bytes": len(weight),
                    "sha256": hashlib.sha256(weight).hexdigest(),
                    "upstream_lfs_sha256_verified": True,
                }
            ],
        },
    }
    for name, value in documents.items():
        (path / name).write_text(json.dumps(value))


def incoming(path, body):
    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"authorization", b"Bearer single-stream-credential")],
        },
        receive,
    )
