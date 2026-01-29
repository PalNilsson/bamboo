"""Direct LLM passthrough tool.

This tool is intentionally simple: it forwards the full chat prompt (history)
to the configured **default** LLM profile and returns the model's raw text.

Use-cases:
  1) Sanity-check that LLM configuration (keys, provider adapters, networking)
     works end-to-end through MCP.
  2) Provide an explicit "bypass reasoning engine" path later, when the
     orchestration layer starts selecting tools.
"""
from __future__ import annotations

from typing import Any, cast

from askpanda_mcp.prompts.templates import get_askpanda_system_prompt
from askpanda_mcp.tools.base import text_content, coerce_messages

from askpanda_mcp.llm.runtime import get_llm_manager, get_llm_selector
from askpanda_mcp.llm.types import GenerateParams, Message


class LLMPassthroughTool:
    """Calls the default LLM with the full provided prompt.

    This tool forwards either a provided `messages` chat history or a single
    `question` string (wrapped as a user message) to the project's configured
    default LLM profile. The raw text response from the model is returned as
    a single text content block.
    """

    @staticmethod
    def get_definition() -> dict[str, Any]:
        """Return the MCP tool discovery definition.

        The returned mapping describes the tool's name, description and the
        expected input schema so clients can discover and validate calls.

        Returns:
            Dict[str, Any]: Tool definition compatible with MCP discovery.
        """
        return {
            "name": "askpanda_llm_answer",
            "description": (
                "Send the full prompt (optionally including chat history) to the "
                "default LLM profile and return the raw response text."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "User question (used if messages is not provided).",
                    },
                    "messages": {
                        "type": "array",
                        "description": (
                            "Optional full chat history as a list of {role, content}. "
                            "If provided, it is sent to the LLM as-is (plus a system prompt)."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature.",
                        "default": 0.2,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Optional max tokens for the completion.",
                    },
                },
            },
        }

    async def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute the passthrough call against the default LLM.

        Args:
            arguments: Tool arguments; either a `messages` list or a `question`
                string is required. Optional `temperature` and `max_tokens`
                controls the generation parameters.

        Returns:
            List[Dict[str, Any]]: A one-element list containing the model's
            raw text response annotated with a debug header.

        Raises:
            ValueError: If neither `question` nor non-empty `messages` is provided.
            RuntimeError: If the configured LLM selector does not expose a
                registry or the manager cannot obtain a client.
        """
        selector = get_llm_selector()
        manager = get_llm_manager()

        # Determine which profile is considered "default".
        default_profile = getattr(selector, "default_profile", "default")
        registry = getattr(selector, "registry", None)
        if registry is None:
            raise RuntimeError("LLM selector does not expose a registry.")

        model_spec = registry.get(default_profile)
        debug = f"[LLM selected] provider={model_spec.provider} model={model_spec.model}"
        client = await manager.get_client(model_spec)

        temperature = float(arguments.get("temperature", 0.2))
        max_tokens = arguments.get("max_tokens")
        max_tokens_int = int(max_tokens) if max_tokens is not None else None

        # Build message list.
        sys_prompt = await get_askpanda_system_prompt()
        sys_text = getattr(sys_prompt, "text", None) or getattr(sys_prompt, "content", None)
        if isinstance(sys_text, list):
            # Some MCP prompt objects use content items.
            sys_text = "\n".join([str(x.get("text", "")) for x in sys_text if isinstance(x, dict)])
        system_message: Message = {"role": "system", "content": str(sys_text or "")}

        messages_arg = arguments.get("messages")
        messages: list[Message] = [system_message]
        if isinstance(messages_arg, list) and messages_arg:
            messages.extend(cast(list[Message], coerce_messages(messages_arg)))
        else:
            question = str(arguments.get("question", "")).strip()
            if not question:
                raise ValueError("Either 'question' or non-empty 'messages' must be provided.")
            messages.append({"role": "user", "content": question})

        resp = await client.generate(
            messages=messages,
            params=GenerateParams(temperature=temperature, max_tokens=max_tokens_int),
        )
        return text_content(f"{debug}\n\n{resp.text}")


askpanda_llm_answer_tool = LLMPassthroughTool()
