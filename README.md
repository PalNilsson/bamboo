# Bamboo

**Bamboo** is a lightweight MCP-based runtime with a plugin architecture for AI-assisted scientific tools.

It is experiment- and workflow-agnostic by design. AskPanDA/ATLAS/Vera Rubin/EPIC and CGSim integrations are implemented as **plugins**, not hard-coded dependencies.

Note: as of March 2026, Bamboo is in early development. The core system is stable, but plugins and documentation are still being built.
There are only few tools available at this time. The current focus is expanding the infrastructure (esp. orchestration using tool families planning,
which will eventually enable highly complex prompts).

I will add a separate examples README when more tools are available.

## Key ideas

- Tool-first architecture
- Plugins provide domain logic
- LLMs are used for *summarization and explanation*, not as sources of truth
- Structured evidence is returned alongside natural-language answers

## Quick start (development)
```bash
pip install -e ./core
pip install -e ./packages/askpanda_atlas

# Configure environment (copy and fill in API keys)
cp bamboo_env_example.sh bamboo_env.sh
source bamboo_env.sh

# List available tools
python -m bamboo tools list

# Start the MCP stdio server
python -m bamboo.server
```

To inspect the server interactively using MCP Inspector:
```bash
npx @modelcontextprotocol/inspector python3 -m bamboo.server
```

## Documentation

- [Developer guide](docs/developer.md)
- [MCP server & LLM internals](docs/mcp.md)
- [Writing plugins](docs/plugins.md)
- [Authentication & Security](docs/security.md)
- [Interfaces](docs/interfaces.md)

## Plugins

- **askpanda_atlas** — ATLAS / PanDA workflows
- **askpanda_verarubin** — Vera Rubin (planned)
- **askpanda_epic** — EPIC / EIC (planned)
- **cgsim** — SimGrid-based workflows (non-PanDA)
