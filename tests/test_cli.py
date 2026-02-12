import json

from bamboo import cli as bamboo_cli


def test_cli_tools_list_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "bamboo.cli.loader.list_tool_entry_points",
        lambda: [
            {
                "group": "bamboo.tools",
                "name": "atlas.task_status",
                "value": "askpanda_atlas.task_status:panda_task_status_tool",
            }
        ],
    )

    # bamboo_cli.main may either return an int or raise SystemExit (when called as __main__).
    try:
        result = bamboo_cli.main(["tools", "list", "--json"])
        # If it returned, expect 0
        assert result == 0
    except SystemExit as se:
        assert se.code == 0

    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed[0]["name"] == "atlas.task_status"
