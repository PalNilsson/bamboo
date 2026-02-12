import asyncio
import json

from askpanda_atlas import task_status as ts_mod


class DummyResp:
    def __init__(self, status_code=200, headers=None, text="{}", json_data=None):
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self._text = text
        self._json = json_data

    @property
    def text(self):
        return self._text

    def json(self):
        if self._json is not None:
            return self._json
        raise ValueError("No JSON payload")


def test_task_status_success_json(monkeypatch):
    sample = {
        "task": {"jeditaskid": 1234, "status": "finished"},
        "datasets": [
            {"status": "finished", "nfilesfailed": 0, "nfilesfinished": 10, "datasetname": "d1"},
        ],
    }

    def fake_get(url, timeout=30, headers=None, allow_redirects=True):  # pylint: disable=unused-argument
        return DummyResp(status_code=200, json_data=sample, text=json.dumps(sample))

    monkeypatch.setattr("requests.get", fake_get)

    tool = ts_mod.panda_task_status_tool
    res = asyncio.run(tool.call({"task_id": 1234, "query": "status?"}))

    assert "evidence" in res
    ev = res["evidence"]
    assert ev["task_id"] == 1234
    assert ev.get("status") == "finished"
    assert "monitor_url" in ev


def test_task_status_non_json(monkeypatch):
    html = "<html><body>error</body></html>"

    def fake_get(url, timeout=30, headers=None, allow_redirects=True):  # pylint: disable=unused-argument
        return DummyResp(status_code=200, headers={"content-type": "text/html"}, text=html, json_data=None)

    monkeypatch.setattr("requests.get", fake_get)

    tool = ts_mod.panda_task_status_tool
    res = asyncio.run(tool.call({"task_id": 9999, "query": "status?"}))

    ev = res.get("evidence", {})
    assert ev.get("http_status") == 200
    assert ev.get("content_type", "").startswith("text/html")
    assert "response_snippet" in ev


def test_task_status_404(monkeypatch):
    def fake_get(url, timeout=30, headers=None, allow_redirects=True):  # pylint: disable=unused-argument
        return DummyResp(status_code=404, headers={"content-type": "text/html"}, text="", json_data=None)

    monkeypatch.setattr("requests.get", fake_get)

    tool = ts_mod.panda_task_status_tool
    res = asyncio.run(tool.call({"task_id": 1111, "query": "status?"}))

    ev = res.get("evidence", {})
    assert ev.get("http_status") == 404
    assert ev.get("not_found") is True
