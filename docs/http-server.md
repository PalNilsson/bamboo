# HTTP server deployment

This guide covers running Bamboo as a shared HTTP server so multiple users can
connect without running their own local server process.  The primary use case
is a testbed deployment at BNL or a similar facility.

The stdio transport (used by the TUI by default) spawns a private server
subprocess per user session and needs no network configuration.  The HTTP
transport runs a single persistent server process that any number of clients
can connect to over the network.

---

## Architecture

```
┌─────────────────────┐        HTTP/MCP         ┌──────────────────────────┐
│  User A — TUI       │ ──────────────────────► │                          │
├─────────────────────┤                          │  Bamboo HTTP server      │
│  User B — TUI       │ ──────────────────────► │  (uvicorn ASGI app)      │
├─────────────────────┤                          │                          │
│  User C — curl /    │ ──────────────────────► │  bamboo.entrypoints.http │
│  MCP Inspector      │                          └──────────────────────────┘
└─────────────────────┘
```

The server holds **one shared LLM configuration and one shared PanDA
connection**.  Each client gets an isolated MCP session (its own session ID
and conversation state) but shares the underlying resources.

---

## Prerequisites

Install `uvicorn` — the ASGI server used to run the HTTP entrypoint:

```bash
pip install uvicorn
```

All other dependencies are the same as for stdio mode.  Make sure the full
environment is configured (LLM keys, `PANDA_BASE_URL`, etc.) before starting.

---

## Starting the server

```bash
uvicorn bamboo.entrypoints.http:app --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` binds to all network interfaces so remote clients can
  connect.  Use `--host 127.0.0.1` to restrict to localhost only.
- `--port 8000` — change if the port is already in use.

The MCP endpoint is at:

```
http://<your-hostname-or-ip>:8000/mcp
```

### Finding your address

```bash
hostname -f                                          # FQDN (recommended at BNL)
ip addr show | grep "inet " | grep -v 127.0.0.1     # IP address
```

### Running persistently

For a testbed that should survive terminal disconnection:

```bash
# With nohup — logs go to bamboo_server.log
nohup uvicorn bamboo.entrypoints.http:app \
  --host 0.0.0.0 --port 8000 \
  > bamboo_server.log 2>&1 &

echo "Server PID: $!"   # note this to kill it later

# With screen — easier to inspect and reattach
screen -S bamboo-server
uvicorn bamboo.entrypoints.http:app --host 0.0.0.0 --port 8000
# Ctrl+A D to detach
# screen -r bamboo-server to reattach
```

### Multiple workers

For higher concurrency (multiple simultaneous users):

```bash
uvicorn bamboo.entrypoints.http:app \
  --host 0.0.0.0 --port 8000 \
  --workers 4
```

Note: `--workers` forks separate processes.  Each worker has its own LLM
connection pool and PanDA MCP session.  Use a single worker during initial
testing to simplify log inspection.

---

## Authentication

If the server is reachable from outside your team, configure Bearer token
authentication.  When no tokens are configured, the server accepts all
requests (suitable for an isolated testbed subnet, not for public exposure).

### Option A — tokens file (recommended for multiple users)

Create a file with one entry per line:

```
# /etc/bamboo/tokens.txt
# Format: client_id: token   (or just: client_id token)
alice: s3cr3t-token-for-alice
bob:   s3cr3t-token-for-bob
ci:    s3cr3t-token-for-ci-runner
```

Then set the env var before starting the server:

```bash
export BAMBOO_MCP_TOKENS_FILE="/etc/bamboo/tokens.txt"
uvicorn bamboo.entrypoints.http:app --host 0.0.0.0 --port 8000
```

### Option B — inline token list (quick setup)

```bash
export BAMBOO_MCP_TOKENS="alice:s3cr3t-alice,bob:s3cr3t-bob"
uvicorn bamboo.entrypoints.http:app --host 0.0.0.0 --port 8000
```

Format: comma-separated `client_id:token` pairs.  If only a token (no colon)
is given, the client ID is recorded as `"unknown"`.

### Generating secure tokens

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Run once per user and distribute the token out-of-band.

---

## Firewall

At BNL you may need to open the port through the host firewall:

```bash
# RHEL / Rocky Linux (firewalld)
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload

# Ubuntu (ufw)
sudo ufw allow 8000/tcp
```

Verify the port is reachable from another machine:

```bash
curl http://<your-hostname>:8000/mcp
# Expected: HTTP 405 (correct — GET is not a valid MCP request)
```

---

## Connecting with the TUI

Users on other machines run:

```bash
python interfaces/textual/chat.py \
  --transport http \
  --http-url http://<your-hostname>:8000/mcp
```

Or set `MCP_URL` in the environment to avoid repeating the URL:

```bash
export MCP_URL="http://<your-hostname>:8000/mcp"
python interfaces/textual/chat.py --transport http
```

### With authentication

Pass the Bearer token via `--token` or set it in `bamboo_env.sh`:

```bash
# Via flag
python interfaces/textual/chat.py \
  --transport http \
  --http-url http://<your-hostname>:8000/mcp \
  --token s3cr3t-token-for-alice

# Via environment variable (add to bamboo_env.sh)
export MCP_BEARER_TOKEN="s3cr3t-token-for-alice"
python interfaces/textual/chat.py --transport http \
  --http-url http://<your-hostname>:8000/mcp
```

The token is sent as `Authorization: Bearer <token>` on every request.

---

## Connecting with MCP Inspector

Useful for debugging tool calls directly:

```bash
npx @modelcontextprotocol/inspector \
  --url http://<your-hostname>:8000/mcp
```

---

## Verifying the server is running

```bash
# Check the process
ps aux | grep uvicorn

# Check the port
ss -tlnp | grep 8000          # Linux
lsof -i :8000                  # macOS

# Tail the log (if using nohup)
tail -f bamboo_server.log
```

---

## Stopping the server

```bash
# If started with nohup, use the PID you noted earlier
kill <PID>

# Or find and kill by port
kill $(lsof -ti :8000)         # macOS
kill $(fuser 8000/tcp)         # Linux

# If running in screen
screen -r bamboo-server
# then Ctrl+C
```

---

## Environment variables reference

All standard Bamboo env vars apply.  HTTP-specific additions:

| Variable | Purpose |
|---|---|
| `BAMBOO_MCP_TOKENS_FILE` | Path to tokens file for Bearer auth |
| `BAMBOO_MCP_TOKENS` | Inline comma-separated `client_id:token` list |
| `MCP_URL` | Default server URL read by the TUI (`--http-url` default) |

See `bamboo_env_example.sh` for the full list of LLM, PanDA, and tracing
variables that also need to be set on the server.

---

## Troubleshooting

**`Connection refused` on the client**
: The server is not running, or the firewall is blocking the port.  Check
  `ps aux | grep uvicorn` on the server and verify the firewall rules.

**`HTTP 401 Unauthorized`**
: Auth is enabled on the server but the client sent no token or the wrong one.
  Verify `BAMBOO_MCP_TOKENS` / `BAMBOO_MCP_TOKENS_FILE` and the token the
  client is sending.

**`HTTP 405 Method Not Allowed` on GET**
: This is correct MCP behaviour — the endpoint only accepts POST.  The server
  is running fine.

**LLM errors on first question**
: The LLM environment variables (`MISTRAL_API_KEY` etc.) are not set in the
  server process environment.  Make sure you `source bamboo_env.sh` before
  starting uvicorn, or export the variables in the systemd unit / nohup
  invocation.

**`SSL_CERT_FILE` errors**
: If `SSL_CERT_FILE` is set in the shell environment and points to a
  non-existent file, the Mistral HTTP client will fail with
  `[Errno 2] No such file or directory`.  Run `unset SSL_CERT_FILE` before
  starting the server.
