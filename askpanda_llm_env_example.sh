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
# OPENAI CONFIGURATION (disabled by default)
########################################

export OPENAI_API_KEY=""

########################################
# ANTHROPIC CONFIGURATION (disabled by default)
########################################

export ANTHROPIC_API_KEY=""

########################################
# GEMINI CONFIGURATION (disabled by default)
########################################

export GEMINI_API_KEY=""

########################################
# OPENAI-COMPATIBLE ENDPOINT (Llama / Mistral via vLLM, Ollama, etc.)
########################################

# Only used if provider="openai_compat"
export ASKPANDA_OPENAI_COMPAT_BASE_URL=""
export OPENAI_COMPAT_API_KEY=""

########################################
# DEBUG / SAFETY
########################################

# Uncomment for verbose debug logs if needed
# export ASKPANDA_DEBUG="1"

echo "AskPanDA LLM environment variables loaded (example configuration)."
