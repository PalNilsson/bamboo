"""LLM configuration loader.

This module converts application configuration (env vars and/or a Config object)
into a :class:`~bamboo.llm.registry.ModelRegistry`.

Precedence:
  1) Environment variables (LLM_*), if set and non-empty
  2) Attributes on the provided `config` object
  3) Defaults defined in this module
"""

from __future__ import annotations

import json
import os
from typing import Any
import dataclasses

from bamboo.llm.registry import ModelRegistry
from bamboo.llm.types import ModelSpec


_MISSING: object = object()


def _get(config: Any, name: str, default: Any = _MISSING) -> Any:
    """Get a configuration value, preferring environment variables.

    Args:
        config: Config object or module with attributes.
        name: Attribute / environment variable name.
        default: Default to return if not found.

    Returns:
        The resolved value.

    Raises:
        AttributeError: If missing and no default is provided.
    """
    env_val = os.getenv(name)
    if env_val is not None and env_val != "":
        return env_val

    if hasattr(config, name):
        val = getattr(config, name)
        # Treat empty strings as "unset" so env/default can take over.
        if val is not None and val != "":
            return val

    if default is not _MISSING:
        return default

    raise AttributeError(f"Missing config value: {name}")


def _parse_profiles_json(value: str) -> dict[str, ModelSpec]:
    """Parse JSON mapping profile name to ModelSpec-like dict.

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
    """
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("LLM_PROFILES_JSON must be a JSON object")

    profiles: dict[str, ModelSpec] = {}
    for profile_name, spec_dict in raw.items():
        if not isinstance(spec_dict, dict):
            raise ValueError(f"Profile '{profile_name}' must be an object")
        provider = str(spec_dict.get("provider", "")).strip()
        model = str(spec_dict.get("model", "")).strip()
        if not provider or not model:
            raise ValueError(f"Profile '{profile_name}' must include provider and model")

        profiles[profile_name] = ModelSpec(
            provider=provider,
            model=model,
            base_url=spec_dict.get("base_url"),
            api_key_env=spec_dict.get("api_key_env"),
            extra=spec_dict.get("extra") or {},
        )

    return profiles


def build_model_registry_from_config(config: Any) -> ModelRegistry:
    """Build a ModelRegistry from application configuration.

    This supports two configuration modes:

    1) JSON mode (recommended for later): set LLM_PROFILES_JSON
    2) Env/attribute mode (Option A): set LLM_DEFAULT_PROVIDER/MODEL, etc.

    Args:
        config: Global application configuration object (class, module, or instance).

    Returns:
        A ModelRegistry.
    """
    profiles_json = os.getenv("LLM_PROFILES_JSON") or getattr(config, "LLM_PROFILES_JSON", "")
    profiles_json = str(profiles_json or "").strip()

    if profiles_json:
        profiles = _parse_profiles_json(profiles_json)
    else:
        default_provider = str(_get(config, "LLM_DEFAULT_PROVIDER", "openai")).strip()
        default_model = str(_get(config, "LLM_DEFAULT_MODEL", "gpt-4.1-mini")).strip()

        fast_provider = str(_get(config, "LLM_FAST_PROVIDER", default_provider)).strip()
        fast_model = str(_get(config, "LLM_FAST_MODEL", default_model)).strip()

        reasoning_provider = str(_get(config, "LLM_REASONING_PROVIDER", default_provider)).strip()
        reasoning_model = str(_get(config, "LLM_REASONING_MODEL", default_model)).strip()

        profiles = {
            "default": ModelSpec(provider=default_provider, model=default_model),
            "fast": ModelSpec(provider=fast_provider, model=fast_model),
            "reasoning": ModelSpec(provider=reasoning_provider, model=reasoning_model),
        }

    # If using openai_compat, allow a global base URL to be supplied.
    compat_base_url = str(_get(config, "ASKPANDA_OPENAI_COMPAT_BASE_URL", "") or "").strip()
    if compat_base_url:
        for name, spec in list(profiles.items()):
            if spec.provider == "openai_compat" and not spec.base_url:
                # ModelSpec is frozen; create a replaced copy with the base_url set.
                profiles[name] = dataclasses.replace(spec, base_url=compat_base_url)

    return ModelRegistry(profiles=profiles)
