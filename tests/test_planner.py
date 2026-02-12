import asyncio
import json

import pytest

from bamboo.tools import planner as planner_mod


def test_extract_first_json_object_from_code_fence():
    text = """Here you go:

```json
{"route":"FAST_PATH","confidence":0.9,"tool_calls":[{"tool":"panda_task_status","arguments":{}}],"retrieval_query":null,"reuse_policy":{"allow_final_answer_reuse":false,"allow_pattern_reuse":true,"requires_fresh_evidence":true},"explain":"ok"}
```
"""
    extracted = planner_mod.extract_first_json_object(text)
    parsed = json.loads(extracted)
    assert parsed["route"] == "FAST_PATH"


def test_plan_schema_contains_expected_fields():
    schema = planner_mod.get_plan_json_schema()
    assert schema.get("title") == "Plan"
    props = schema.get("properties", {})
    for key in ("route", "confidence", "tool_calls", "retrieval_query", "reuse_policy", "explain"):
        assert key in props


def test_plan_semantic_validation_requires_tool_calls_for_plan_route():
    bad = {
        "route": "PLAN",
        "confidence": 0.5,
        "tool_calls": [],
        "retrieval_query": None,
        "reuse_policy": {
            "allow_final_answer_reuse": False,
            "allow_pattern_reuse": True,
            "requires_fresh_evidence": True,
        },
        "explain": "",
    }
    with pytest.raises(Exception):
        planner_mod.Plan.model_validate(bad)


def test_planner_tool_repairs_invalid_first_response(monkeypatch):
    # Avoid entry-point scanning in unit tests.
    monkeypatch.setattr(
        planner_mod,
        "_collect_tool_catalog",
        lambda namespaces=None: [
            {
                "name": "panda_task_status",
                "description": "Get task metadata",
                "inputSchema": {"type": "object", "properties": {"task_id": {"type": "integer"}}},
            }
        ],
    )

    calls = {"n": 0}

    async def fake_call_default_llm(messages, temperature, max_tokens):  # pylint: disable=unused-argument
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return json.dumps(
            {
                "route": "FAST_PATH",
                "confidence": 0.91,
                "tool_calls": [{"tool": "panda_task_status", "arguments": {"task_id": 123}}],
                "retrieval_query": None,
                "reuse_policy": {
                    "allow_final_answer_reuse": False,
                    "allow_pattern_reuse": True,
                    "requires_fresh_evidence": True,
                },
                "explain": "Task ID detected.",
            }
        )

    monkeypatch.setattr(planner_mod, "_call_default_llm", fake_call_default_llm)

    tool = planner_mod.bamboo_plan_tool
    res = asyncio.run(tool.call({"question": "task 123 status?"}))
    assert isinstance(res, list)
    assert res and res[0]["type"] == "text"
    plan = json.loads(res[0]["text"])
    assert plan["route"] == "FAST_PATH"
    assert calls["n"] == 2
