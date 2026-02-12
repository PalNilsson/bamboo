# Bamboo

**Bamboo** is a lightweight MCP-based runtime with a plugin architecture for AI-assisted scientific tools.

It is experiment- and workflow-agnostic by design. AskPanDA/ATLAS/Vera Rubin/EPIC and CGSim integrations are implemented as **plugins**, not hard-coded dependencies.

Note: as of February 2026, Bamboo is in early development. The core system is stable, but plugins and documentation are still being built.
There are only few tools available at this time. The current focus is expanding the infrastructure (esp. orchestration using tool families planning,
which will eventually enable highly complex prompts).
The only tool that is fully working at this time is the ATLAS PanDA task status tool. It can be run like this:

```bash
pip install -e ./core
pip install -e ./packages/askpanda_atlas
python3 -m bamboo tools call atlas.task_status --arguments '{"task_id":123456}' (broken example, to be fixed)
```

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
python -m bamboo tools list
```

## Documentation

- [Developer guide](docs/developer.md)
- [MCP server & LLM internals](docs/mcp.md)
- [Writing plugins](docs/plugins.md)
- [Authentication & Security](docs/security.md)
-
## Plugins

- **askpanda_atlas** — ATLAS / PanDA workflows
- **askpanda_verarubin** — Vera Rubin (planned)
- **askpanda_epic** — EPIC / EIC (planned)
- **cgsim** — SimGrid-based workflows (non-PanDA)
