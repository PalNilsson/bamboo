"""Bamboo CLI core.

Minimal command-line interface for inspecting the Bamboo installation.
Currently supports:

  - `python -m bamboo.cli tools list`

This prints discovered tool entry points from the primary group and legacy group.
"""

from __future__ import annotations

import argparse
import json
import sys

from bamboo.tools import loader


def cmd_tools_list(args: argparse.Namespace) -> int:
    """List discovered tool entry points and print them.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    items = loader.list_tool_entry_points()
    if args.json:
        print(json.dumps(items, indent=2, sort_keys=True))
    else:
        if not items:
            print("No tool entry points found.")
            return 0
        # Simple aligned output
        w_group = max(len(i.get("group", "")) for i in items)
        w_name = max(len(i.get("name", "")) for i in items)
        for i in items:
            print(f"{i['group']:<{w_group}}  {i['name']:<{w_name}}  {i['value']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the Bamboo CLI entry point.

    Args:
        argv: Optional list of CLI arguments excluding the program name.

    Returns:
        Process exit code.
    """
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="bamboo")
    sub = p.add_subparsers(dest="cmd", required=True)

    tools = sub.add_parser("tools", help="Tooling commands")
    tools_sub = tools.add_subparsers(dest="tools_cmd", required=True)

    tools_list = tools_sub.add_parser("list", help="List discovered tool entry points")
    tools_list.add_argument("--json", action="store_true", help="Output JSON")
    tools_list.set_defaults(func=cmd_tools_list)

    ns = p.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
