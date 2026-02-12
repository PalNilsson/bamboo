"""Task status tool wrapper for PanDA task monitoring.

Improved canonical wrapper for bamboo.tools.task_status that ensures the
exported tool object `panda_task_status_tool` exposes a robust get_definition()
including an 'inputSchema' key so UI/registry validation passes.

This file wraps an existing implementation if present (preferred as task_status_impl
or similar). If none found, it provides a helpful stub.
"""
from __future__ import annotations
import asyncio
import inspect
import importlib
from typing import Any, Optional

# Try to import a real implementation under a different name first to avoid self-import.
_real = None
_import_errors = []

candidates = [
    "bamboo.tools.task_status_atlas",
    "bamboo.tools.task_status_orig",
    "bamboo.tools.task_status_original",
    "bamboo.tools.task_status_backup",
    # do not import "bamboo.tools.task_status" here to avoid recursion
]

for name in candidates:
    try:
        mod = importlib.import_module(name)
        if mod is not None and getattr(mod, "__name__", "") != __name__:
            _real = mod
            break
    except Exception as e:
        _import_errors.append((name, repr(e)))
        continue

# If no alternative module found, try to import plain 'task_status' (may point to original in some setups)
if _real is None:
    try:
        mod_plain = importlib.import_module("task_status")
        if getattr(mod_plain, "__name__", "") != __name__:
            _real = mod_plain
    except Exception as e:
        _import_errors.append(("task_status", repr(e)))
        _real = None


def _find_callable_in_module(mod: Any) -> Optional[Any]:
    """Find a callable function in a module for task status.

    Searches for a callable in this order:
    - panda_task_status_tool.call
    - module-level call()
    - Common function names (panda_task_status, get_task_status, task_status, run, handle)

    Args:
        mod: Module object to search.

    Returns:
        Callable function, or None if not found.
    """
    # If module defines a canonical tool object with .call, use that
    if hasattr(mod, "panda_task_status_tool") and hasattr(mod.panda_task_status_tool, "call"):
        return getattr(mod, "panda_task_status_tool").call  # may be async or sync
    # module-level call()
    if hasattr(mod, "call") and callable(mod.call):
        return mod.call
    # common function names
    for name in ("panda_task_status", "get_task_status", "task_status", "run", "handle"):
        if hasattr(mod, name) and callable(getattr(mod, name)):
            return getattr(mod, name)
    return None


async def _stub_call(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return a stub error response when no implementation is found.

    Args:
        arguments: Tool input arguments provided by the caller.

    Returns:
        Dictionary containing error information and debugging details.
    """
    return {
        "error": "no underlying task_status implementation found",
        "message": "Restore the original implementation as task_status_impl.py or ensure the module exports callable functions.",
        "import_attempts": _import_errors,
        "provided_arguments": arguments,
    }


def _wrap_callable(fn: Any) -> Any:
    """Wrap a sync or async callable into an async function.

    Args:
        fn: Function or coroutine function to wrap.

    Returns:
        Async wrapper function that accepts arguments dict.
    """
    if inspect.iscoroutinefunction(fn):
        async def _async_fn(args: dict[str, Any]) -> Any:
            return await fn(args)
        return _async_fn
    else:
        async def _async_fn(args: dict[str, Any]) -> Any:
            return await asyncio.to_thread(fn, args)
        return _async_fn


_detected_callable = None
if _real is not None:
    _detected_callable = _find_callable_in_module(_real)

if _detected_callable is not None:
    _async_caller = _wrap_callable(_detected_callable)
else:
    _async_caller = _stub_call


# Create the canonical tool object with a robust get_definition including inputSchema
class _Tool:
    """PanDA task status tool wrapper with MCP-compatible definition.

    Wraps underlying task_status implementations and provides a robust tool
    definition including inputSchema for UI/registry validation.
    """

    def __init__(self) -> None:
        """Initialize the tool with a definition from real or default implementation."""
        # attempt to reuse definition from real module if present
        if _real is not None and hasattr(_real, "get_definition"):
            try:
                base_def: dict[str, Any] = _real.get_definition() or {}
            except Exception:
                base_def = {}
        else:
            base_def = {}

        # Ensure required fields exist and include a sensible inputSchema
        self._def: dict[str, Any] = {
            "name": base_def.get("name", "panda_task_status"),
            "description": base_def.get("description", "PanDA task status (wrapped)"),
            "inputSchema": base_def.get("inputSchema", {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "PanDA task id"},
                    "query": {"type": "string", "description": "Original user query string (optional)"},
                },
                # task_id is the primary required field for this tool
                "required": ["task_id"],
                "additionalProperties": True
            }),
            "examples": base_def.get("examples", [
                {"task_id": 48254358, "query": "what is the status of task 48254358?"}
            ]),
            "tags": base_def.get("tags", ["panda", "monitoring", "task"]),
        }

        # If the real module indicated a detected source, add that for debugging
        if _real is not None:
            try:
                self._def.setdefault("metadata", {})["detected_module"] = getattr(_real, "__name__", "unknown")
            except Exception:
                pass

    def get_definition(self) -> dict[str, Any]:
        """Get the MCP tool definition.

        Returns:
            Tool definition dict with name, description, inputSchema, and metadata.
        """
        return self._def

    async def call(self, arguments: dict[str, Any]) -> Any:
        """Execute the task status tool with provided arguments.

        Attempts to delegate to the underlying implementation if available,
        otherwise returns a stub error response.

        Args:
            arguments: Tool input dict with required 'task_id' key.

        Returns:
            Task status result from underlying implementation, or error dict if not found.
        """
        # Prefer passing the arguments through to the underlying implementation if present.
        if _real is not None:
            # If the real module exposes an awaitable call, try to use it
            try:
                # If real exposes panda_task_status_tool with .call
                if hasattr(_real, "panda_task_status_tool") and hasattr(_real.panda_task_status_tool, "call"):
                    fn: Any = _real.panda_task_status_tool.call
                    if inspect.iscoroutinefunction(fn):
                        return await fn(arguments)
                    else:
                        return await asyncio.to_thread(fn, arguments)
                # module-level call
                if hasattr(_real, "call") and callable(_real.call):
                    fn = _real.call
                    if inspect.iscoroutinefunction(fn):
                        return await fn(arguments)
                    else:
                        return await asyncio.to_thread(fn, arguments)
                # fallback: detect common function names
                for name in ("panda_task_status", "get_task_status", "task_status", "run", "handle"):
                    if hasattr(_real, name) and callable(getattr(_real, name)):
                        fn = getattr(_real, name)
                        if inspect.iscoroutinefunction(fn):
                            return await fn(arguments)
                        else:
                            return await asyncio.to_thread(fn, arguments)
            except Exception as e:
                # If the underlying implementation raises, return the exception info for debugging
                return {"error": "underlying task_status raised", "exception": repr(e), "provided_arguments": arguments}

        # No real implementation found — return stub response
        return await _stub_call(arguments)


# For backwards-compatibility, also expose module-level async call
async def call(arguments: dict[str, Any]) -> Any:
    """Execute the task status tool at module level for backwards compatibility.

    Args:
        arguments: Tool input dict with required 'task_id' key.

    Returns:
        Task status result from the tool.
    """
    return await panda_task_status_tool.call(arguments)


panda_task_status_tool = _Tool()
