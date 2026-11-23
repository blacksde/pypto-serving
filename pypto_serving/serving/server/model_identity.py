# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Content identity for the checkpoint and the tokenizer actually loaded by Serving."""

import hashlib
import inspect
import json
from pathlib import Path

from .state_protocol import digest


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_model_identity(model_dir, model, tokenizer):
    """Verify pinned shard contents once at native startup, never per request."""
    from .server import ServingServer

    root = Path(model_dir)
    manifest_path = root / "download-verification.json"
    manifest = json.loads(manifest_path.read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())
    files = {row["file"]: row for row in manifest["files"]}
    weights = {}
    for name in sorted(set(index["weight_map"].values())):
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Native weight index contains an invalid shard name")
        record = files.get(name)
        actual = root / name
        if (record is None or record.get("upstream_lfs_sha256_verified") is not True
                or actual.stat().st_size != record.get("bytes")):
            raise ValueError("Native weights lack a pinned, verified manifest entry")
        value = file_digest(actual)
        if value != record.get("sha256"):
            raise ValueError("Native weight contents differ from the pinned revision")
        weights[name] = value
    if not weights or not manifest.get("repo") or not manifest.get("revision"):
        raise ValueError("Native weights require a non-empty pinned checkpoint identity")
    metadata = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "model.safetensors.index.json", "vocab.json", "merges.txt", "added_tokens.json",
                 "special_tokens_map.json", "chat_template.jinja"):
        path = root / name
        if path.is_file():
            metadata[name] = file_digest(path)
    if not all(name in metadata for name in ("config.json", "tokenizer.json", "tokenizer_config.json")):
        raise ValueError("Native identity requires model and tokenizer metadata")
    loaded = tokenizer.tokenizer
    template = loaded.chat_template
    if not template:
        raise ValueError("Native identity requires the actual loaded chat template")
    identity = dict(schema_version=1, model=model, repository=manifest["repo"], revision=manifest["revision"],
        weights=weights, metadata=metadata, weight_manifest=file_digest(manifest_path),
        actual_tokenizer=digest(json.loads(loaded.backend_tokenizer.to_str())),
        chat_template=digest(template), tokenizer_adapter=hashlib.sha256(
            inspect.getsource(type(tokenizer)).encode()).hexdigest(),
        chat_adapter=file_digest(Path(inspect.getfile(ServingServer))))
    identity["fingerprint"] = "sha256:" + digest(identity)
    return identity
