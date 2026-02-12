# AskPanDA Interfaces

This folder contains UI/front-end interfaces that talk to the AskPanDA MCP server.

## Layout

- `interfaces/shared/` - shared client helpers (e.g., MCP connection wrappers)
- `interfaces/streamlit/` - Streamlit UI
- `interfaces/open_webui/` - placeholder for a future Open WebUI integration

## Streamlit

From your repo root:

```bash
pip install -r requirements-ui.txt  # or your preferred env management
streamlit run interfaces/streamlit/chat.py
```

By default, the UI spawns the MCP server via stdio:

- command: `python`
- args: `["-m", "bamboo.server"]`

If your MCP server module/package name differs, update those args in the sidebar.
