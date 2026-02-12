# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Authors
# - Paul Nilsson, paul.nilsson@cern.ch, 2026

"""Authentication helpers for Bamboo services.

This module implements a small Bearer-token allowlist suitable for internal
services. Tokens are loaded at process start from either:

  - A tokens file specified by environment variable BAMBOO_MCP_TOKENS_FILE
  - An inline token list specified by environment variable BAMBOO_MCP_TOKENS

The stdio entrypoint does not use HTTP headers and therefore cannot enforce
Bearer tokens; the HTTP entrypoint should enforce them on inbound requests.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple


class TokenAuthError(ValueError):
    """Raised when authentication fails."""


@dataclass(frozen=True)
class TokenAuthConfig:
    """Configuration for TokenAuth.

    Attributes:
        tokens_file_env: Environment variable name for tokens file path.
        tokens_env: Environment variable name for inline token list.
    """

    tokens_file_env: str = "BAMBOO_MCP_TOKENS_FILE"
    tokens_env: str = "BAMBOO_MCP_TOKENS"


def _parse_tokens_line(line: str) -> Optional[Tuple[str, str]]:
    """Parse a single tokens file line.

    Supported formats:
      - "client_id: token"
      - "client_id token"

    Blank lines and comment lines starting with "#" are ignored.

    Args:
        line: The input line.

    Returns:
        Tuple of (client_id, token), or None if the line is blank/comment.

    Raises:
        ValueError: If the line is non-empty but not parseable.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if ":" in stripped:
        client, token = stripped.split(":", 1)
        client_id = client.strip()
        tok = token.strip()
        if not client_id or not tok:
            raise ValueError("Invalid tokens line (missing client_id or token).")
        return client_id, tok

    parts = stripped.split()
    if len(parts) != 2:
        raise ValueError("Invalid tokens line format.")
    return parts[0].strip(), parts[1].strip()


def _load_tokens_from_file(path: Path) -> Dict[str, str]:
    """Load tokens from a tokens.txt file.

    The returned mapping is token -> client_id.

    Args:
        path: File path.

    Returns:
        Mapping from token string to client_id.
    """
    mapping: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_tokens_line(line)
        if parsed is None:
            continue
        client_id, token = parsed
        mapping[token] = client_id
    return mapping


def _load_tokens_from_env_value(value: str) -> Dict[str, str]:
    """Load tokens from an inline environment variable.

    Format:
      - Comma-separated entries
      - Each entry may be "client_id:token" or just "token"
        If only "token" is given, client_id is "unknown".

    Example:
      BAMBOO_MCP_TOKENS="bamboo-cli:AAA,bamboo-ui:BBB,CCC"

    Args:
        value: Environment value.

    Returns:
        Mapping from token string to client_id.
    """
    mapping: Dict[str, str] = {}
    for raw in value.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if ":" in entry:
            client_id, token = entry.split(":", 1)
            mapping[token.strip()] = client_id.strip() or "unknown"
        else:
            mapping[entry] = "unknown"
    return mapping


@dataclass(frozen=True)
class TokenAuth:
    """Bearer token allowlist authentication.

    Attributes:
        token_to_client: Mapping from token -> client_id.
    """

    token_to_client: Mapping[str, str]

    @classmethod
    def from_env(cls, config: TokenAuthConfig | None = None) -> "TokenAuth":
        """Create TokenAuth by loading configuration from environment.

        Precedence:
          1) tokens file referenced by BAMBOO_MCP_TOKENS_FILE
          2) inline tokens in BAMBOO_MCP_TOKENS
          3) empty allowlist (auth disabled)

        Args:
            config: Optional config overrides.

        Returns:
            TokenAuth instance.
        """
        cfg = config or TokenAuthConfig()
        file_path = os.environ.get(cfg.tokens_file_env, "").strip()
        inline = os.environ.get(cfg.tokens_env, "").strip()

        if file_path:
            return cls(_load_tokens_from_file(Path(file_path)))
        if inline:
            return cls(_load_tokens_from_env_value(inline))
        return cls({})

    @property
    def enabled(self) -> bool:
        """Check if any tokens are configured for authentication."""
        return bool(self.token_to_client)

    def verify_token(self, token: str) -> str:
        """Verify a raw token and return the client_id.

        Args:
            token: Token string.

        Returns:
            client_id for the token.

        Raises:
            TokenAuthError: If token is invalid.
        """
        for allowed, client_id in self.token_to_client.items():
            if secrets.compare_digest(token, allowed):
                return client_id
        raise TokenAuthError("Invalid token.")

    def verify_bearer_token(self, authorization_header: str | None) -> str:
        """Verify an Authorization header in Bearer format.

        Args:
            authorization_header: Value of the Authorization header.

        Returns:
            client_id for the token. If auth is disabled, returns "auth-disabled".

        Raises:
            TokenAuthError: If the header is missing/malformed or token is invalid.
        """
        if not self.enabled:
            return "auth-disabled"

        if not authorization_header:
            raise TokenAuthError("Missing Authorization header.")

        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise TokenAuthError("Expected 'Authorization: Bearer <token>'.")

        return self.verify_token(parts[1])
