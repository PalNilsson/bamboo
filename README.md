# Ask PanDA Toolkit

AskPanDA Toolkit is an MCP-based service and UI for answering experiment-specific questions.
It currently supports a direct **LLM passthrough mode** (used to validate end-to-end
plumbing) and is designed to evolve toward **tool-based orchestration backed by PanDA
and experiment-specific plugins**.

The system supports multiple LLM providers (currently Mistral by default) and multiple
deployment modes (stdio MCP and HTTP Streamable MCP).

---

## Repository Layout

```
askpanda_mcp/
  core.py                       # MCP server + tool registration
  server.py                     # stdio MCP entrypoint
  entrypoints/http.py           # ASGI Streamable HTTP MCP entrypoint
  llm/                          # LLM abstraction, providers, registry, manager
  tools/                        # MCP tools (LLM passthrough, future PanDA tools)

interfaces/
  streamlit/chat.py             # Streamlit UI
  shared/mcp_client.py          # MCP client helper

askpanda_llm_env_example.sh     # Example environment variable configuration
README.md
```

---

## Prerequisites

- Python **3.11+** (recommended)
- Virtual environment (`venv` or `conda`)
- At least one LLM API key (Mistral is the default)

---

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Make sure `streamlit`, `uvicorn`, and your chosen LLM SDK
(e.g. `mistralai`) are installed.

---

## Environment Variables (LLMs)

AskPanDA uses **Option A** configuration: LLM profiles are defined via environment variables.

### 1. Copy the example file

```bash
cp askpanda_llm_env_example.sh askpanda_llm_env.sh
```

### 2. Edit and add API keys

At minimum, set:

```bash
export MISTRAL_API_KEY="<your-key>"
```

### 3. Load the environment

```bash
source askpanda_llm_env.sh
```

> For Kubernetes, map these variables from Secrets instead of sourcing a file.

---

## Running the MCP Server

You can run AskPanDA in **stdio mode** or **HTTP (Streamable MCP) mode**.

### Option 1: stdio MCP (local development)

```bash
export PYTHONPATH=$PWD
python -m askpanda_mcp.server
```

This mode is useful for:
- Local development
- MCP hosts that only support stdio

---

### Option 2: HTTP MCP (Uvicorn)

This is required for Streamlit, Open WebUI, and Kubernetes deployments.

```bash
export PYTHONPATH=$PWD
uvicorn askpanda_mcp.entrypoints.http:app --host 0.0.0.0 --port 8000
```

Recommended dev flags:

```bash
uvicorn askpanda_mcp.entrypoints.http:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info
```

#### Health check

```bash
curl http://localhost:8000/healthz
```

Expected response:

```
ok
```

---

## Running the Streamlit UI

In a separate terminal:

```bash
export PYTHONPATH=$PWD
streamlit run interfaces/streamlit/chat.py
```

Streamlit will open a browser window automatically.

---

## End-to-End Test: LLM Passthrough Mode

To verify that everything works before adding real PanDA tools:

1. Start the MCP server (stdio or HTTP)
2. Start Streamlit
3. In the UI, enable **“Bypass tool routing / send directly to LLM”**
4. Ask a question

What happens:

- The full chat history is sent to the MCP tool `askpanda_llm_answer`
- The server forwards it to the **default LLM profile**
- The response is returned verbatim

If this works, it confirms:

- Environment variables are correct
- LLM selection and client management work
- MCP request/response plumbing is correct
- Streamlit → MCP → LLM integration is functional

---

## Deployment Notes

### Why keep both stdio and HTTP entrypoints?

- **stdio MCP**
  - Minimal setup
  - Useful for local development and MCP-native hosts

- **HTTP MCP**
  - Required for web UIs
  - Supports health checks, auth middleware, observability
  - Required for Kubernetes deployments

Both entrypoints share the same server core and tool registry.

---

## Next Steps (Planned)

- Add real PanDA-backed tools (task status, queue info, log analysis)
- Introduce `askpanda_answer` orchestration tool
- Add experiment-specific plugin system (ATLAS, Rubin, ePIC)
- Add auth middleware (API key / JWT)
- Add Open WebUI integration

---

## Troubleshooting

### Import errors
Ensure you are in the repo root and `PYTHONPATH` is set:

```bash
export PYTHONPATH=$PWD
```

### LLM authentication failures
Check that the API key is loaded:

```bash
echo "$MISTRAL_API_KEY"
```

### HTTP server starts but Streamlit fails to connect
- Confirm the server URL and port
- Check logs from both Uvicorn and Streamlit
- Verify whether Streamlit is configured for stdio or HTTP MCP

---

## For Developers

This section documents development and quality-assurance workflows for contributors
and maintainers of this repository.

### Running unit tests

Unit tests are written using `pytest` and live under the `tests/` directory.

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_summarize.py
```

To run tests with more verbose output:

```bash
pytest -v
```

The tests do **not** make real LLM API calls; all provider interactions are mocked.

---

### Static analysis and code quality

This project uses several static-analysis tools. They are complementary and serve
different purposes:

- **flake8** — style, formatting, and complexity checks
- **pylint** — deeper code-quality and design checks
- **mypy** — static type checking
- **pydocstyle** — docstring style checks (Google-style)

Typical commands:

```bash
flake8
pylint your_module.py
mypy .
pydocstyle
```

Configuration for these tools is provided via:

- `pyproject.toml`
- `.flake8`
- `.pylintrc`
- `.pre-commit-config.yaml`

---

### Pre-commit hooks

This repository uses **pre-commit** to run quality checks automatically before
each commit.

To install the hooks locally:

```bash
pip install pre-commit
pre-commit install
```

To run all hooks manually on all files:

```bash
pre-commit run --all-files
```

---

### Type-checking notes

This repository is primarily script-oriented rather than a distributable Python
package. To avoid issues with directory naming and package inference, `mypy`
is configured to treat files as standalone scripts via `pyproject.toml`.

Optional third-party SDKs (for example `mistralai` or `google-genai`) may lack
complete type stubs; missing imports are intentionally ignored by `mypy`.

---

### Adding new modules or scripts

When adding new code:

- Prefer **pure functions** where possible (simpler testing)
- Add unit tests for non-trivial logic
- Use modern type hints (`list[str]`, `str | None`, etc.)
- Add a module-level docstring explaining the script’s purpose
- Keep CLI parsing isolated in a `main()` function

---

### Recommended local workflow

A typical local workflow before committing changes:

```bash
pytest
flake8
pylint your_module.py
mypy .
pydocstyle
pre-commit run
```

---

## License

Internal / project-specific (update as appropriate).
