# Bamboo

**Bamboo** is a plugin-based orchestration framework for AI-assisted workflows.

It is experiment- and workflow-agnostic by design. AskPanDA/ATLAS/Vera Rubin/EPIC and CGSim integrations are implemented as **plugins**, not hard-coded dependencies.

## Key ideas

- Tool-first architecture
- Plugins provide domain logic
- LLMs are used for *summarization and explanation*, not as sources of truth
- Structured evidence is returned alongside natural-language answers

## Quick start (development)

```bash
pip install -e ./core
pip install -e ./packages/askpanda_atlas
python -m bamboo tools list
```

## Documentation

- [Developer guide](docs/developer.md)
- [MCP server & LLM internals](docs/mcp.md)
- [Writing plugins](docs/plugins.md)

## Plugins

- **askpanda_atlas** — ATLAS / PanDA workflows
- **askpanda_verarubin** — Vera Rubin (planned)
- **askpanda_epic** — EPIC / EIC (planned)
- **cgsim** — SimGrid-based workflows (non-PanDA)

## Philosophy

Bamboo is the framework. AskPanDA is one possible application built as a plugin family.

This separation allows Bamboo to be reused outside HEP and PanDA-based systems.
