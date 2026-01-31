# Writing plugins for Bamboo

Plugins provide domain-specific functionality.

Each plugin is a separate Python package with its own `pyproject.toml` and registers tools via entry points.

## Entry points

Plugins register tools under the `bamboo.tools` entry-point group:

```toml
[project.entry-points."bamboo.tools"]
"atlas.task_status" = "askpanda_atlas.task_status:panda_task_status_tool"
```

Naming convention:

- Entry point name: `<namespace>.<tool_name>`
- Example: `atlas.task_status`

## Tool contract

A tool should expose:

```python
def get_definition() -> dict
async def call(arguments: dict) -> dict
```

Return shape:

```json
{
  "text": "Human-readable summary or tool output",
  "evidence": { "structured": "data" }
}
```

## Namespaces

Namespaces prevent collisions and make it clear which plugin owns a tool:

- `atlas.*` — ATLAS / PanDA tooling
- `cgsim.*` — SimGrid / CGSim tooling
- `verarubin.*` — Vera Rubin tooling
- `epic.*` — EPIC / EIC tooling
