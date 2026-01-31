import pytest
from bamboo.tools import loader


class DummyEP:
    def __init__(self, name, value, group="bamboo.tools"):
        self.name = name
        self.value = value
        self.group = group

    def load(self):
        return "dummy_tool_obj"


def test_loader_list_and_find(monkeypatch):
    class EPs:
        def select(self, group=None):
            if group == loader.PRIMARY_GROUP:
                return [
                    DummyEP(
                        "atlas.task_status",
                        "askpanda_atlas.task_status:panda_task_status_tool",
                        group=loader.PRIMARY_GROUP,
                    )
                ]
            if group == loader.LEGACY_GROUP:
                return []
            return []

    monkeypatch.setattr("bamboo.tools.loader.entry_points", lambda: EPs())

    items = loader.list_tool_entry_points()
    assert any(i["name"] == "atlas.task_status" for i in items)

    resolved = loader.find_tool_by_name("task_status", namespace="atlas")
    assert resolved is not None
    assert resolved.name == "task_status"
    assert resolved.namespace == "atlas"
