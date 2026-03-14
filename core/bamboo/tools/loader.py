"""Tool discovery and loading from entry points.

Provides utilities to discover and load Bamboo tools from entry points,
supporting both primary and legacy tool groups.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from importlib.metadata import entry_points

PRIMARY_GROUP = "bamboo.tools"
LEGACY_GROUP = "askpanda.tools"


@dataclass(frozen=True)
class ResolvedTool:
    """Metadata for a resolved tool loaded from an entry point.

    Attributes:
        name: The tool name (without namespace prefix).
        namespace: The tool namespace/group.
        obj: The loaded tool object instance.
        entry_point: String representation of the entry point (group:name=value).
    """

    name: str
    namespace: str
    obj: Any
    entry_point: str


def _iter_entry_points(groups: Iterable[str]) -> Any:
    """Iterate over entry points across multiple groups.

    Handles both new (3.10+) and legacy entry point APIs for compatibility.

    Args:
        groups: Iterable of entry point group names to search.

    Yields:
        Entry point objects matching the specified groups.
    """
    eps = entry_points()
    for g in groups:
        try:
            selected = eps.select(group=g)  # type: ignore[attr-defined]
        except Exception:  # pylint: disable=broad-exception-caught
            selected = eps.get(g, []) if isinstance(eps, dict) else []
        yield from selected


def list_tool_entry_points() -> list[dict[str, str]]:
    """List all discovered tool entry points across primary and legacy groups.

    Returns:
        List of dicts with keys 'group', 'name', and 'value' for each entry point.
    """
    out: list[dict[str, str]] = []
    for ep in _iter_entry_points([PRIMARY_GROUP, LEGACY_GROUP]):
        out.append({"group": ep.group, "name": ep.name, "value": ep.value})
    return out


def find_tool_by_name(tool_name: str, namespace: str | None = None) -> ResolvedTool | None:
    """Find and load a tool by name, optionally filtered by namespace.

    Searches for entry points matching the given tool name across primary and
    legacy groups.

    The ``tool_name`` argument should be the **unqualified** name (e.g.
    ``"task_status"``).  If you pass a fully-qualified name like
    ``"atlas.task_status"`` without setting ``namespace``, pass it as-is and
    leave ``namespace`` as ``None``; the function will split it internally.

    Args:
        tool_name: Name of the tool to find.  May include a namespace prefix
            separated by a dot (e.g. ``'atlas.task_status'``).  If so, and
            ``namespace`` is not set, the prefix is treated as the namespace.
        namespace: Optional namespace to restrict the search.

    Returns:
        ResolvedTool with the loaded tool object and metadata, or None if not found.
    """
    # If tool_name is fully qualified and no explicit namespace was given,
    # split it so the match logic below works correctly.
    if "." in tool_name and namespace is None:
        namespace, tool_name = tool_name.split(".", 1)

    for group in (PRIMARY_GROUP, LEGACY_GROUP):
        for ep in _iter_entry_points([group]):
            ep_name: str = ep.name
            if namespace:
                if ep_name != f"{namespace}.{tool_name}":
                    continue
            else:
                # Unqualified search: match the suffix after the last dot.
                suffix = ep_name.rsplit(".", 1)[-1] if "." in ep_name else ep_name
                if suffix != tool_name:
                    continue

            try:
                obj: Any = ep.load()
            except Exception:  # pylint: disable=broad-exception-caught
                continue

            ns: str
            name: str
            ns, _, name = ep_name.partition(".")
            return ResolvedTool(
                name=name or tool_name,
                namespace=ns,
                obj=obj,
                entry_point=f"{ep.group}:{ep.name}={ep.value}",
            )
    return None
