"""Registry of named model profiles."""

from __future__ import annotations

from dataclasses import dataclass

from askpanda_mcp.llm.types import ModelSpec


@dataclass(frozen=True)
class ModelRegistry:
    """Registry of named model profiles.

    Example:
        profiles["fast"] -> ModelSpec(provider="anthropic", model="claude-...")
    """
    profiles: dict[str, ModelSpec]

    def get(self, profile: str) -> ModelSpec:
        """Gets a profile ModelSpec.

        Args:
            profile: Profile name.

        Returns:
            ModelSpec for the profile.

        Raises:
            KeyError: If profile not found.
        """
        return self.profiles[profile]
