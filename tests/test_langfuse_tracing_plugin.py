from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_plugin_module():
    plugin_path = (
        Path(__file__).resolve().parents[1]
        / ".hermes"
        / "plugins"
        / "langfuse_tracing"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location("langfuse_tracing_plugin", plugin_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_serialize_messages_parses_tool_message_json():
    plugin = _load_plugin_module()

    messages = [
        {"role": "user", "content": "find files"},
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"total_count": 2, "files": ["a.py", "b.py"], "truncated": false}',
        },
    ]

    serialized = plugin._serialize_messages(messages)

    assert serialized[1]["role"] == "tool"
    assert serialized[1]["tool_call_id"] == "call_123"
    assert serialized[1]["content"]["total_count"] == 2
    assert serialized[1]["content"]["files"] == ["a.py", "b.py"]
    assert serialized[1]["content"]["truncated"] is False


def test_safe_value_parses_nested_json_strings_when_enabled():
    plugin = _load_plugin_module()

    value = {
        "outer": '{"items": [{"name": "alpha"}], "count": 1}',
        "plain": "hello",
    }

    parsed = plugin._safe_value(value, parse_json_strings=True)

    assert parsed["outer"]["items"][0]["name"] == "alpha"
    assert parsed["outer"]["count"] == 1
    assert parsed["plain"] == "hello"


def test_safe_value_parses_json_with_trailing_hint_text():
    plugin = _load_plugin_module()

    value = (
        '{"total_count": 179, "truncated": true}\n\n'
        "[Hint: Results truncated. Use offset=20 to see more.]"
    )

    parsed = plugin._safe_value(value, parse_json_strings=True)

    assert parsed["total_count"] == 179
    assert parsed["truncated"] is True
    assert parsed["_hint"] == "[Hint: Results truncated. Use offset=20 to see more.]"


def test_start_child_observation_uses_parent_observation_api():
    plugin = _load_plugin_module()

    events = []

    class FakeRootSpan:
        def start_observation(self, **kwargs):
            events.append(("root_start", kwargs))
            return {"label": "child"}

    class FakeClient:
        pass

    state = plugin.TraceState(
        trace_id="trace-123",
        root_ctx=None,
        root_span=FakeRootSpan(),
    )

    observation = plugin._start_child_observation(
        state,
        client=FakeClient(),
        name="LLM call 1",
        as_type="generation",
        input_value={"role": "user", "content": "hi"},
        metadata={"provider": "openai"},
        model="gpt-5.4",
        model_parameters={"provider": "openai"},
    )

    assert observation["label"] == "child"
    assert events[0][0] == "root_start"
    assert events[0][1]["name"] == "LLM call 1"
    assert events[0][1]["as_type"] == "generation"
    assert events[0][1]["input"] == {"role": "user", "content": "hi"}
    assert events[0][1]["metadata"] == {"provider": "openai"}


def test_on_post_tool_call_parses_json_with_trailing_hint():
    plugin = _load_plugin_module()
    updates = {}

    class FakeObservation:
        def update(self, **kwargs):
            updates.update(kwargs)

        def end(self):
            updates["ended"] = True

    task_id = "task-hint"
    state = plugin.TraceState(trace_id="trace-hint", root_ctx=None, root_span=object())
    state.tools["call_hint"] = FakeObservation()
    plugin._TRACE_STATE[task_id] = state

    try:
        plugin.on_post_tool_call(
            task_id=task_id,
            tool_name="search_files",
            tool_call_id="call_hint",
            args={"pattern": "provider"},
            result='{"total_count": 179, "truncated": true}\n\n[Hint: Results truncated. Use offset=20 to see more.]',
        )
    finally:
        plugin._TRACE_STATE.pop(task_id, None)

    assert updates["output"]["total_count"] == 179
    assert updates["output"]["truncated"] is True
    assert updates["output"]["_hint"] == "[Hint: Results truncated. Use offset=20 to see more.]"
    assert updates["metadata"]["tool_name"] == "search_files"
    assert updates["ended"] is True


def test_safe_value_formats_read_file_payload_as_preview():
    plugin = _load_plugin_module()

    content = "\n".join(
        [
            "     1|alpha",
            "     2|beta",
            "     3|gamma",
        ]
    )
    value = {
        "content": content,
        "total_lines": 3,
        "file_size": 17,
        "truncated": False,
        "is_binary": False,
        "is_image": False,
    }

    parsed = plugin._safe_value(value, parse_json_strings=True)

    assert "content" not in parsed
    assert parsed["returned_lines"] == {"start": 1, "end": 3, "count": 3}
    assert parsed["content_preview"]["lines"][0] == {"line": 1, "text": "alpha"}
    assert parsed["content_preview"]["lines"][2] == {"line": 3, "text": "gamma"}


def test_on_post_tool_call_formats_read_file_output_with_preview_and_args():
    plugin = _load_plugin_module()
    updates = {}

    class FakeObservation:
        def update(self, **kwargs):
            updates.update(kwargs)

        def end(self):
            updates["ended"] = True

    task_id = "task-read-file"
    state = plugin.TraceState(trace_id="trace-read-file", root_ctx=None, root_span=object())
    state.tools["call_read_file"] = FakeObservation()
    plugin._TRACE_STATE[task_id] = state

    lines = [f"{line:6d}|line {line}" for line in range(1, 51)]
    result = json.dumps({
        "content": "\n".join(lines),
        "total_lines": 50,
        "file_size": 400,
        "truncated": False,
        "is_binary": False,
        "is_image": False,
    })

    try:
        plugin.on_post_tool_call(
            task_id=task_id,
            tool_name="read_file",
            tool_call_id="call_read_file",
            args={"path": "agent/models_dev.py", "offset": 1, "limit": 260},
            result=result,
        )
    finally:
        plugin._TRACE_STATE.pop(task_id, None)

    assert updates["output"]["path"] == "agent/models_dev.py"
    assert updates["output"]["offset"] == 1
    assert updates["output"]["limit"] == 260
    assert updates["output"]["returned_lines"] == {"start": 1, "end": 50, "count": 50}
    assert updates["output"]["content_preview"]["head"][0] == {"line": 1, "text": "line 1"}
    assert updates["output"]["content_preview"]["tail"][-1] == {"line": 50, "text": "line 50"}
    assert updates["output"]["content_preview"]["omitted_line_count"] == 10
    assert "content" not in updates["output"]
    assert updates["metadata"]["tool_name"] == "read_file"
    assert updates["ended"] is True
