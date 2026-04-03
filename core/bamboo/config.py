"""AskPanDA MCP Server configuration.

This module defines a frozen dataclass `Config` that centralizes runtime
configuration for the AskPanDA MCP server. Defaults are intentionally safe
and driven by environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # type: ignore[import]
except ImportError:  # Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

try:
    from importlib.metadata import version as _pkg_version
    _bamboo_version: str = _pkg_version("bamboo")
except Exception:  # package not installed or metadata unavailable
    _bamboo_version = "0.0.0.dev0"


def load_askpanda_config() -> dict:
    """Load AskPanDA configuration from `pyproject.toml`.

    Returns:
        Dictionary of configuration values under `tool.askpanda`, or an empty dict if the file does not exist.
    """
    if tomllib is None:
        return {}
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return {}
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    return data.get("tool", {}).get("askpanda", {})


_CONFIG = load_askpanda_config()

DEFAULT_NAMESPACE = _CONFIG.get("default_namespace")


@dataclass(frozen=True)
class Config:  # pylint: disable=too-many-instance-attributes
    """Configuration for AskPanDA MCP Server.

    Attributes:
        SERVER_NAME (str): Service name exposed to MCP clients.
        SERVER_VERSION (str): Service version string.
        ENABLE_REAL_PANDA (bool): Toggle real PanDA integration (off by default).
        ENABLE_REAL_LLM (bool): Toggle real LLM integration (off by default).
        KNOWLEDGE_BASE_PATH (str): Path to static knowledge base used for RAG.
        QUEUE_DATA_PATH (str): Path to queue metadata (e.g. queuedata.json).
        LLM_DEFAULT_PROFILE (str): Default LLM profile name.
        LLM_FAST_PROFILE (str): Profile name used for fast (low-latency) LLMs.
        LLM_REASONING_PROFILE (str): Profile name used for reasoning-heavy LLMs.
        LLM_DEFAULT_PROVIDER (str): Default LLM provider id (e.g. "openai").
        LLM_DEFAULT_MODEL (str): Default LLM model string.
        LLM_FAST_PROVIDER (str): Provider id for fast profile.
        LLM_FAST_MODEL (str): Fast model string.
        LLM_REASONING_PROVIDER (str): Provider id for reasoning profile.
        LLM_REASONING_MODEL (str): Reasoning model string.
        OPENAI_COMPAT_BASE_URL (str): Optional OpenAI-compatible endpoint URL.
    """

    SERVER_NAME: str = os.getenv("ASKPANDA_SERVER_NAME", "askpanda-mcp-server")
    SERVER_VERSION: str = os.getenv("ASKPANDA_SERVER_VERSION", _bamboo_version)

    # Toggle real integrations later
    ENABLE_REAL_PANDA: bool = os.getenv("ASKPANDA_ENABLE_REAL_PANDA", "0") == "1"
    ENABLE_REAL_LLM: bool = os.getenv("ASKPANDA_ENABLE_REAL_LLM", "0") == "1"

    # Where your static knowledge base lives (for future RAG)
    KNOWLEDGE_BASE_PATH: str = os.getenv("ASKPANDA_KB_PATH", "data/kb")

    # Where queuedata.json or other site metadata might live
    QUEUE_DATA_PATH: str = os.getenv("ASKPANDA_QUEUE_DATA_PATH", "data/queuedata.json")

    # LLM profiles: JSON string or path later; start with env for simplicity.
    LLM_DEFAULT_PROFILE: str = os.getenv("ASKPANDA_LLM_DEFAULT_PROFILE", "default")
    LLM_FAST_PROFILE: str = os.getenv("ASKPANDA_LLM_FAST_PROFILE", "fast")
    LLM_REASONING_PROFILE: str = os.getenv("ASKPANDA_LLM_REASONING_PROFILE", "reasoning")

    # Model strings (minimal starting point)
    #
    # Compatibility note:
    # Some environments export LLM_DEFAULT_PROVIDER/LLM_DEFAULT_MODEL (without
    # the ASKPANDA_ prefix). Support both, with ASKPANDA_* taking precedence.
    LLM_DEFAULT_PROVIDER: str = os.getenv(
        "ASKPANDA_LLM_DEFAULT_PROVIDER",
        os.getenv("LLM_DEFAULT_PROVIDER", "openai"),
    )
    LLM_DEFAULT_MODEL: str = os.getenv(
        "ASKPANDA_LLM_DEFAULT_MODEL",
        os.getenv("LLM_DEFAULT_MODEL", "gpt-4.1-mini"),
    )

    LLM_FAST_PROVIDER: str = os.getenv(
        "ASKPANDA_LLM_FAST_PROVIDER",
        os.getenv("LLM_FAST_PROVIDER", LLM_DEFAULT_PROVIDER),
    )
    LLM_FAST_MODEL: str = os.getenv(
        "ASKPANDA_LLM_FAST_MODEL",
        os.getenv("LLM_FAST_MODEL", LLM_DEFAULT_MODEL),
    )

    LLM_REASONING_PROVIDER: str = os.getenv(
        "ASKPANDA_LLM_REASONING_PROVIDER",
        os.getenv("LLM_REASONING_PROVIDER", LLM_DEFAULT_PROVIDER),
    )
    LLM_REASONING_MODEL: str = os.getenv(
        "ASKPANDA_LLM_REASONING_MODEL",
        os.getenv("LLM_REASONING_MODEL", "gpt-4.1"),
    )

    # Optional OpenAI-compatible endpoint (for Llama/Mistral via vLLM/Ollama/etc.)
    OPENAI_COMPAT_BASE_URL: str = os.getenv("ASKPANDA_OPENAI_COMPAT_BASE_URL", "")
