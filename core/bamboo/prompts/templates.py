"""Prompt templates.

In MCP, prompts are a discoverable interface for clients.
Keep prompts small and composable; use tools for data access.
"""
from __future__ import annotations
from typing import Any


def _text_msg(text: str) -> dict[str, Any]:
    """Create a text message dict for MCP prompt messages.

    Args:
        text: The text content of the message.

    Returns:
        Dictionary with role 'assistant' and text content.
    """
    return {"role": "assistant", "content": {"type": "text", "text": text}}


async def get_bamboo_system_prompt() -> dict[str, Any]:
    """Get the system prompt for AskPanDA assistant.

    Returns:
        Dictionary with 'messages' list containing the system prompt defining
        AskPanDA's role and behavior for PanDA/ATLAS workflow operations.
    """
    return {
        "messages": [
            _text_msg(
                "You are AskPanDA, an assistant for PanDA/ATLAS workflow operations. "
                "Prefer calling tools for factual data (task status, queue info, pilots). "
                "If data is missing, ask for identifiers (task id, job id, site) and propose next steps."
            )
        ]
    }


async def get_failure_triage_prompt(log_text: str) -> dict[str, Any]:
    """Get a prompt template for analyzing failure logs.

    Produces a structured analysis prompt for triaging workflow failures including
    classification, root causes, mitigation steps, and metadata collection guidance.

    Args:
        log_text: The failure log text to be analyzed.

    Returns:
        Dictionary with 'messages' list containing the analysis prompt.
    """
    return {
        "messages": [
            _text_msg(
                "Analyze the following failure log and produce: "
                "(1) classification, (2) likely root causes, (3) immediate mitigation, "
                "(4) what additional metadata to collect.\n\n"
                f"LOG:\n{log_text}"
            )
        ]
    }
