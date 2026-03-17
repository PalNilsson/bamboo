#!/usr/bin/env bash
#
# Example environment configuration for AskPanDA LLM support
# Copy this file, remove `_example`, and fill in the API keys as needed.
# Remember to add this file to your .gitignore to avoid committing sensitive information.

########################################
# PANDA RELATED
########################################

export PANDA_BASE_URL="https://bigpanda.cern.ch"
export ASKPANDA_PANDA_RETRIES="2"
export ASKPANDA_PANDA_BACKOFF_SECONDS="0.8"

########################################
# LLM PROFILE SELECTION
########################################

# Which profile names the selector will use
export LLM_DEFAULT_PROFILE="default"
export LLM_FAST_PROFILE="fast"
export LLM_REASONING_PROFILE="reasoning"

########################################
# DEFAULT PROFILE (used if nothing else matches)
########################################

export LLM_DEFAULT_PROVIDER="mistral"
export LLM_DEFAULT_MODEL="mistral-large-latest"

########################################
# FAST PROFILE (classification, routing, lightweight tasks)
########################################

export LLM_FAST_PROVIDER="mistral"
export LLM_FAST_MODEL="mistral-large-latest"

########################################
# REASONING PROFILE (log analysis, synthesis, RAG answers)
########################################

export LLM_REASONING_PROVIDER="mistral"
export LLM_REASONING_MODEL="mistral-large-latest"

########################################
# MISTRAL CONFIGURATION
########################################

# Required when using provider="mistral"
export MISTRAL_API_KEY=""

# Optional concurrency / retry tuning
export ASKPANDA_MISTRAL_CONCURRENCY="4"
export ASKPANDA_MISTRAL_RETRIES="3"
export ASKPANDA_MISTRAL_BACKOFF_SECONDS="1.0"

########################################
# OPENAI CONFIGURATION
########################################

# Required when using provider="openai" or provider="openai_compat".
# Install: pip install -r requirements-openai.txt
export OPENAI_API_KEY=""

# Optional tuning for the OpenAI provider.
# export ASKPANDA_OPENAI_CONCURRENCY="8"
# export ASKPANDA_OPENAI_RETRIES="3"
# export ASKPANDA_OPENAI_BACKOFF_SECONDS="1.0"

########################################
# ANTHROPIC CONFIGURATION
########################################

# Required when using provider="anthropic".
# Install: pip install -r requirements-anthropic.txt
export ANTHROPIC_API_KEY=""

# Optional tuning for the Anthropic provider.
# export ASKPANDA_ANTHROPIC_CONCURRENCY="4"
# export ASKPANDA_ANTHROPIC_RETRIES="3"
# export ASKPANDA_ANTHROPIC_BACKOFF_SECONDS="1.0"

########################################
# GEMINI CONFIGURATION
########################################

# Required when using provider="gemini".
# Install: pip install -r requirements-gemini.txt
export GEMINI_API_KEY=""

# Optional tuning for the Gemini provider.
# export ASKPANDA_GEMINI_CONCURRENCY="4"
# export ASKPANDA_GEMINI_RETRIES="3"
# export ASKPANDA_GEMINI_BACKOFF_SECONDS="1.0"

########################################
# OPENAI-COMPATIBLE ENDPOINT (Llama / Mistral via vLLM, Ollama, etc.)
########################################

# Required when using provider="openai_compat".
# Uses the same openai SDK as the OpenAI provider.
# Install: pip install -r requirements-openai.txt
export ASKPANDA_OPENAI_COMPAT_BASE_URL=""
export OPENAI_COMPAT_API_KEY=""

# Optional tuning.
# export ASKPANDA_OPENAI_COMPAT_CONCURRENCY="8"
# export ASKPANDA_OPENAI_COMPAT_RETRIES="3"
# export ASKPANDA_OPENAI_COMPAT_BACKOFF_SECONDS="1.0"

########################################
# RAG / CHROMADB (panda_doc_search tool)
########################################

# Path to the ChromaDB persistent directory created by the ingestion script.
export BAMBOO_CHROMA_PATH="./chroma_db"

# Name of the ChromaDB collection to query.
export BAMBOO_CHROMA_COLLECTION="document_monitor_agent"

########################################
# DEBUG / SAFETY
########################################

# Uncomment for verbose debug logs if needed
# export ASKPANDA_DEBUG="1"

########################################
# TRACING
########################################

# Set to 1 to enable structured request/response tracing.
# When BAMBOO_TRACE_FILE is set, spans are written only to that file (stderr
# is left clean — required when running under the Textual TUI).
# When BAMBOO_TRACE_FILE is not set, spans are written to stderr instead.
# See docs/tracing.md for the full event schema and jq recipes.
# export BAMBOO_TRACE="1"
# export BAMBOO_TRACE_FILE="/tmp/bamboo_trace.jsonl"

# OpenTelemetry export (optional — requires pip install -r requirements-otel.txt).
# When set, spans are also exported via OTLP/gRPC to the given endpoint
# (Jaeger, Grafana Tempo, Honeycomb, Datadog, etc.) as a parent/child tree.
# export BAMBOO_OTEL_ENDPOINT="http://localhost:4317"
# export BAMBOO_OTEL_SERVICE_NAME="bamboo"   # default: bamboo
# export BAMBOO_OTEL_INSECURE="1"            # set to 0 to enable TLS

# Set to 1 to redirect the server's stderr to /dev/null.
# The Textual TUI sets this automatically when launching via stdio transport.
# Useful if running the server as a background subprocess in other contexts.
# export BAMBOO_QUIET="1"

echo "AskPanDA LLM environment variables loaded (example configuration)."