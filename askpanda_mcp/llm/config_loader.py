"""LLM configuration loader.

This module converts application configuration into a :class:`~askpanda_mcp.llm.registry.ModelRegistry`.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, MutableMapping

from askpanda_mcp.llm.registry import ModelRegistry
from askpanda_mcp.llm.types import ModelSpec


_MISSING: object = object()


def _cfg_get(config: object, name: str, default: object = _MISSING) -> Any:
    """Gets an attribute from either a config instance or a config class.

    Args:
        config: Config instance or class-like object.
        name: Attribute name.
        default: Default value if attribute is not present.

    Returns:
        Attribute value.

    Raises:
        AttributeError: If the attribute is missing and no default was provided.
    """
    if hasattr(config, name):
        return getattr(config, name)
    if default is not _MISSING:
        return default
    raise AttributeError(f"Missing config attribute: {name}")


def _parse_profiles_json(value: str) -> dict[str, ModelSpec]:
    """Parses JSON mapping profile name -> ModelSpec-like dict.

    Expected JSON format:
        {
          "default": {"provider": "mistral", "model": "mistral-large-latest"},
          "fast": {"provider": "mistral", "model": "mistral-small-latest"},
          "reasoning": {"provider": "mistral", "model": "mistral-large-latest"}
        }

    Optional keys per profile:
        - base_url
        - api_key_env
        - extra (object)

    Args:
        value: JSON string.

    Returns:
        Mapping of profile names to ModelSpec objects.

    Raises:
        ValueError: If JSON is invalid or required keys are missing.
    """
    raw: Mapping[str, Any] = json.loads(value)
    profiles: dict[str, ModelSpec] = {}

    for profile_name, spec in raw.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"Profile '{profile_name}' must be an object.")
        provider = spec.get("provider")
        model = spec.get("model")
        if not provider or not model:
            raise ValueError(f"Profile '{profile_name}' must include 'provider' and 'model'.")
        profiles[profile_name] = ModelSpec(
            provider=str(provider),
            model=str(model),
            base_url=spec.get("base_url"),
            api_key_env=spec.get("api_key_env"),
            extra=dict(spec.get("extra", {})) if spec.get("extra") else None,
        )
    return profiles


def build_model_registry_from_config(config: object) -> ModelRegistry:
    """Builds a ModelRegistry from application configuration.

    Supports two config styles:

    1) JSON mapping (recommended when you add more profiles):
       - ``LLM_PROFILES_JSON``: JSON string defining profiles.
       E.g. export LLM_PROFILES_JSON='{
          "default": {"provider":"mistral","model":"mistral-large-latest"},
          "fast": {"provider":"mistral","model":"mistral-large-latest"},
          "reasoning": {"provider":"mistral","model":"mistral-large-latest"}
          }'
    2) Simple per-profile fields (minimal / env-friendly):
       - ``LLM_DEFAULT_PROVIDER`` / ``LLM_DEFAULT_MODEL``
       - ``LLM_FAST_PROVIDER`` / ``LLM_FAST_MODEL``
       - ``LLM_REASONING_PROVIDER`` / ``LLM_REASONING_MODEL``

    Also supports ``OPENAI_COMPAT_BASE_URL`` for providers using OpenAI-compatible
    endpoints (e.g., vLLM/Ollama/LM Studio).

    Args:
        config: Config instance or class-like object exposing attributes.

    Returns:
        ModelRegistry containing at least the ``default`` profile.

    Raises:
        ValueError: If required configuration is missing or invalid.
    """
    profiles: MutableMapping[str, ModelSpec]

    profiles_json = _cfg_get(config, "LLM_PROFILES_JSON", "")
    if isinstance(profiles_json, str) and profiles_json.strip():
        profiles = _parse_profiles_json(profiles_json.strip())
    else:
        default_provider = _cfg_get(config, "LLM_DEFAULT_PROVIDER", None)
        default_model = _cfg_get(config, "LLM_DEFAULT_MODEL", None)
        if not default_provider or not default_model:
            raise ValueError(
                "LLM profile configuration missing. Provide LLM_PROFILES_JSON, or set "
                "LLM_DEFAULT_PROVIDER and LLM_DEFAULT_MODEL (and optionally FAST/REASONING). "
                "Example: LLM_DEFAULT_PROVIDER='mistral', LLM_DEFAULT_MODEL='mistral-large-latest'."
            )

        profiles = {
            "default": ModelSpec(provider=str(default_provider), model=str(default_model)),
            "fast": ModelSpec(
                provider=str(_cfg_get(config, "LLM_FAST_PROVIDER", default_provider)),
                model=str(_cfg_get(config, "LLM_FAST_MODEL", default_model)),
            ),
            "reasoning": ModelSpec(
                provider=str(_cfg_get(config, "LLM_REASONING_PROVIDER", default_provider)),
                model=str(_cfg_get(config, "LLM_REASONING_MODEL", default_model)),
            ),
        }

    openai_compat_base_url = _cfg_get(config, "OPENAI_COMPAT_BASE_URL", "")
    if isinstance(openai_compat_base_url, str) and openai_compat_base_url.strip():
        base_url = openai_compat_base_url.strip()
        profiles = {
            name: (
                ModelSpec(
                    provider=spec.provider,
                    model=spec.model,
                    base_url=base_url,
                    api_key_env=spec.api_key_env,
                    extra=spec.extra,
                )
                if spec.provider == "openai_compat" and not spec.base_url
                else spec
            )
            for name, spec in profiles.items()
        }

    if "default" not in profiles:
        raise ValueError("ModelRegistry must define a 'default' profile.")

    return ModelRegistry(profiles=dict(profiles))
