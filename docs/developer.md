# Developer guide

This guide covers local development setup, editable installs, and testing.

## Development environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools
```

## Editable installs (important)

Bamboo uses **editable installs** so that plugins register entry points.

```bash
pip install -e ./core
pip install -e ./packages/askpanda_atlas
```

Install additional plugins the same way:

```bash
pip install -e ./packages/cgsim
```

## When you MUST re-run `pip install -e`

Re-run editable installs after changing:

- any `pyproject.toml`
- entry-point definitions (`bamboo.tools`)
- plugin dependencies
- package layout / module paths

You do NOT need to reinstall if you only change Python source files.

## CLI

List discovered tools:

```bash
python -m bamboo tools list
python -m bamboo tools list --json
```

## Running unit tests

Install test dependencies:

```bash
pip install -U pytest pytest-mock
```

Run tests from the repo root:

```bash
pytest -q tests/
```

## Testing strategy

- Unit-test tools by mocking external services (e.g. BigPanDA HTTP)
- Keep tool tests deterministic and fast
- Integration-test MCP endpoints separately (optional)
