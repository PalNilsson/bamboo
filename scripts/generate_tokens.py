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

#!/usr/bin/env python3
"""
Generate high-entropy Bearer tokens for Bamboo MCP deployments.

This script generates cryptographically strong tokens intended for use as
Authorization Bearer tokens (shared secrets) when calling Bamboo MCP endpoints.

Examples:
  # Create tokens for common clients and print a server tokens.txt format
  python scripts/generate_tokens.py \
    --client bamboo-cli --client bamboo-ui --client bamboo-ci \
    --format tokens.txt > tokens.txt

  # Generate a .env snippet for local dev (single token)
  python scripts/generate_tokens.py --client bamboo-dev --format env > .env

  # Generate JSON for programmatic use
  python scripts/generate_tokens.py --client bamboo-cli --client bamboo-ui --format json

Notes:
  - Token entropy is specified in bits (default: 256 bits).
  - Internally the script generates ceil(bits/8) random bytes via secrets.token_urlsafe(nbytes).
  - Treat tokens like passwords: do not commit them to git or log them.
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class TokenRecord:
    """A named token record.

    Attributes:
        client_id: Identifier for the client (e.g. "bamboo-cli").
        token: The generated secret token.
    """

    client_id: str
    token: str


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    """De-duplicate a sequence while preserving the original order.

    Args:
        items: The input items.

    Returns:
        A list with duplicates removed, preserving first occurrence order.
    """
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def bits_to_nbytes(bits: int) -> int:
    """Convert requested entropy in bits to a byte count.

    The conversion rounds up to ensure at least the requested entropy.

    Args:
        bits: Entropy in bits.

    Returns:
        Number of bytes (ceil(bits/8)).

    Raises:
        ValueError: If bits is not positive.
    """
    if bits <= 0:
        raise ValueError("bits must be > 0")
    return int(math.ceil(bits / 8))


def generate_tokens(client_ids: List[str], bits: int) -> List[TokenRecord]:
    """Generate a token for each client id.

    Args:
        client_ids: List of client identifiers.
        bits: Desired entropy in bits (e.g. 256).

    Returns:
        A list of TokenRecord entries (client_id, token).
    """
    nbytes = bits_to_nbytes(bits)
    return [TokenRecord(cid, secrets.token_urlsafe(nbytes)) for cid in client_ids]


def parse_tokens_txt_line(line: str) -> Tuple[str, str]:
    """Parse a line from a tokens.txt file.

    Supported formats:
      - "client_id: token"
      - "client_id token"

    Lines that are empty or comments are considered invalid for parsing.

    Args:
        line: The line to parse.

    Returns:
        Tuple (client_id, token).

    Raises:
        ValueError: If the line is empty/comment or has an invalid format.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        raise ValueError("empty/comment")

    if ":" in stripped:
        client, token = stripped.split(":", 1)
        client_id = client.strip()
        tok = token.strip()
        if not client_id or not tok:
            raise ValueError("invalid format")
        return client_id, tok

    parts = stripped.split()
    if len(parts) != 2:
        raise ValueError("invalid format")

    return parts[0].strip(), parts[1].strip()


def emit_tokens_txt(records: Sequence[TokenRecord]) -> str:
    """Emit tokens in server-friendly tokens.txt format.

    Args:
        records: Token records to emit.

    Returns:
        A string in "client_id: token" format, one per line.
    """
    lines = [f"{r.client_id}: {r.token}" for r in records]
    return "\n".join(lines) + "\n"


def emit_env(records: Sequence[TokenRecord]) -> str:
    """Emit tokens as environment variable assignments.

    If a single token is generated, emits:
      BAMBOO_TOKEN="..."

    If multiple tokens are generated, emits:
      BAMBOO_TOKEN_<CLIENT_ID>="..."

    Args:
        records: Token records to emit.

    Returns:
        A string containing environment variable assignments.
    """
    records_list = list(records)
    if len(records_list) == 1:
        return f'BAMBOO_TOKEN="{records_list[0].token}"\n'

    lines: List[str] = []
    for r in records_list:
        key = "BAMBOO_TOKEN_" + r.client_id.upper().replace("-", "_")
        lines.append(f'{key}="{r.token}"')
    return "\n".join(lines) + "\n"


def emit_json(records: Sequence[TokenRecord]) -> str:
    """Emit tokens as JSON mapping client_id -> token.

    Args:
        records: Token records to emit.

    Returns:
        JSON string (pretty-printed).
    """
    payload: Dict[str, str] = {r.client_id: r.token for r in records}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def emit_markdown(records: Sequence[TokenRecord]) -> str:
    """Emit tokens as Markdown (useful for secure notes).

    Args:
        records: Token records to emit.

    Returns:
        Markdown formatted string listing tokens by client id.
    """
    lines: List[str] = ["# Bamboo tokens", ""]
    for r in records:
        lines.append(f"- **{r.client_id}**: `{r.token}`")
    lines.append("")
    return "\n".join(lines)


def load_existing_tokens_from_stdin() -> set[str]:
    """Load token values from stdin, interpreting stdin as tokens.txt content.

    Non-parseable lines are ignored (e.g. blank lines, comments).

    Returns:
        A set of tokens extracted from stdin.
    """
    existing: set[str] = set()
    for line in sys.stdin.read().splitlines():
        try:
            _, tok = parse_tokens_txt_line(line)
        except ValueError:
            continue
        existing.add(tok)
    return existing


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Generate Bamboo Bearer tokens.")
    parser.add_argument(
        "--client",
        action="append",
        dest="clients",
        default=[],
        help="Client identifier (repeatable). Example: --client bamboo-cli",
    )
    parser.add_argument(
        "--format",
        choices=["tokens.txt", "env", "json", "md"],
        default="tokens.txt",
        help="Output format.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=256,
        help="Entropy in bits (default: 256).",
    )
    parser.add_argument(
        "--min-bits",
        type=int,
        default=128,
        help="Refuse to generate tokens below this entropy (default: 128).",
    )
    parser.add_argument(
        "--check-unique",
        action="store_true",
        help=(
            "Sanity-check that generated tokens do not collide with an existing "
            "tokens.txt provided on stdin."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Optional argv override (for testing). If None, uses sys.argv.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.bits < args.min_bits:
        parser.error(
            f"Refusing to generate tokens with {args.bits} bits (<{args.min_bits}). "
            "Use --min-bits to override if you really want this."
        )

    raw_clients: List[str] = args.clients or ["bamboo-dev"]
    cleaned_clients = [c.strip() for c in raw_clients if c.strip()]
    clients = dedupe_preserve_order(cleaned_clients)
    if not clients:
        parser.error("No valid --client values provided.")

    records = generate_tokens(clients, int(args.bits))

    if args.check_unique:
        existing_tokens = load_existing_tokens_from_stdin()
        collisions = [r for r in records if r.token in existing_tokens]
        if collisions:
            # Extremely unlikely, but keep behavior deterministic.
            parser.error("Token collision detected (extremely unlikely). Re-run generation.")

    fmt: str = args.format
    if fmt == "tokens.txt":
        sys.stdout.write(emit_tokens_txt(records))
    elif fmt == "env":
        sys.stdout.write(emit_env(records))
    elif fmt == "json":
        sys.stdout.write(emit_json(records))
    elif fmt == "md":
        sys.stdout.write(emit_markdown(records))
    else:
        parser.error(f"Unsupported format: {fmt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
