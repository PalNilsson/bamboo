"""AskPanDA MCP Server configuration.

This module defines a frozen dataclass `Config` that centralizes runtime
configuration for the AskPanDA MCP server. Defaults are intentionally safe
and driven by environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import tomllib
from pathlib import Path


def load_askpanda_config() -> dict:
    """Load AskPanDA configuration from `pyproject.toml`.

    Returns:
        Dictionary of configuration values under `tool.askpanda`, or an empty dict if the file does not exist.
    """
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
    SERVER_VERSION: str = os.getenv("ASKPANDA_SERVER_VERSION", "0.1.0")

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
    LLM_DEFAULT_PROVIDER: str = os.getenv("ASKPANDA_LLM_DEFAULT_PROVIDER", "openai")
    LLM_DEFAULT_MODEL: str = os.getenv("ASKPANDA_LLM_DEFAULT_MODEL", "gpt-4.1-mini")

    LLM_FAST_PROVIDER: str = os.getenv("ASKPANDA_LLM_FAST_PROVIDER", "openai")
    LLM_FAST_MODEL: str = os.getenv("ASKPANDA_LLM_FAST_MODEL", "gpt-4.1-mini")

    LLM_REASONING_PROVIDER: str = os.getenv("ASKPANDA_LLM_REASONING_PROVIDER", "openai")
    LLM_REASONING_MODEL: str = os.getenv("ASKPANDA_LLM_REASONING_MODEL", "gpt-4.1")

    # Optional OpenAI-compatible endpoint (for Llama/Mistral via vLLM/Ollama/etc.)
    OPENAI_COMPAT_BASE_URL: str = os.getenv("ASKPANDA_OPENAI_COMPAT_BASE_URL", "")
