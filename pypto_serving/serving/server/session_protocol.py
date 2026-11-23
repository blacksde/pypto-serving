# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Versioned trusted Chat extension; never enters template keyword arguments."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StatefulContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol: Literal["pypto-session-v3"]
    response_id: str = Field(min_length=1, max_length=256)
    parent_response_id: str | None = Field(default=None, min_length=1, max_length=256)
    tenant_id: str = Field(min_length=1, max_length=256)
    input_items: list[dict]
    instructions: str | None = None
    policy: Literal["strict", "availability"] = "availability"
    store: bool = True
    expected_parent_version: str | None = None


def canonical(messages):
    result = []
    for message in messages:
        value = dict(message)
        content = value.get("content") or ""
        if isinstance(content, list):
            if any(part.get("type") not in {"text", "input_text", "output_text"} for part in content):
                raise ValueError("Unsupported history content")
            content = "".join(part["text"] for part in content)
        item = dict(role=value["role"], content=content)
        if value.get("tool_calls"):
            item["tool_calls"] = value["tool_calls"]
        if value.get("tool_call_id"):
            item["tool_call_id"] = value["tool_call_id"]
        result.append(item)
    return result


def input_messages(items):
    messages = []
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            messages.append(dict(role=item.get("role", "user"), content=item.get("content", "")))
        elif kind == "function_call":
            messages.append(
                dict(
                    role="assistant",
                    content="",
                    tool_calls=[
                        dict(
                            id=item["call_id"],
                            type="function",
                            function=dict(name=item["name"], arguments=item["arguments"]),
                        )
                    ],
                )
            )
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            content = item.get("output", "")
            if not isinstance(content, (str, list)):
                content = json.dumps(content, separators=(",", ":"), ensure_ascii=False)
            messages.append(dict(role="tool", content=content, tool_call_id=item["call_id"]))
        elif kind == "custom_tool_call":
            messages.append(
                dict(
                    role="assistant",
                    content="",
                    tool_calls=[
                        dict(
                            id=item["call_id"],
                            type="function",
                            function=dict(
                                name=item["name"],
                                arguments=json.dumps(
                                    dict(input=item["input"]), separators=(",", ":"), ensure_ascii=False
                                ),
                            ),
                        )
                    ],
                )
            )
        elif kind != "reasoning":
            raise ValueError("Unsupported session input item")
    return canonical(messages)


def output_messages(text, calls):
    # Match the Gateway's public Responses-to-Chat history view: reasoning is
    # stored as recovery evidence but not replayed into the next Chat prompt.
    result = [dict(role="assistant", content=text)] if text else []
    result.extend(dict(role="assistant", content="", tool_calls=[call]) for call in calls)
    return result
