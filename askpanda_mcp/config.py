"""AskPanDA MCP Server configuration.

This skeleton intentionally keeps defaults safe:
- No external network calls unless explicitly enabled via env vars.
"""
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
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
