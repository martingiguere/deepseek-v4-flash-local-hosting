# Claude Code Proxy (CCP)

A zero-dependency Go proxy that translates Anthropic's Messages API to OpenAI's
Chat Completions API, enabling Claude Code to use DeepSeek V4 models on Ollama
Cloud or Azure AI Foundry (via Bifrost).

## Quick Start

```bash
# Build
make build

# Run with Ollama Cloud
CCP_BACKEND=ollama CCP_OLLAMA_API_KEY="sk-..." ./ccp

# Run with Azure AI Foundry via Bifrost
CCP_BACKEND=bifrost CCP_BIFROST_BASE_URL="http://localhost:8081" CCP_BIFROST_API_KEY="bf-" ./ccp
```

## Claude Code Setup

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=dummy
claude
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CCP_BACKEND` | `ollama` | Backend to use: `ollama` or `bifrost` |
| `CCP_OLLAMA_API_KEY` | — | Ollama Cloud API key |
| `CCP_OLLAMA_BASE_URL` | `https://ollama.com/v1` | Ollama Cloud base URL |
| `CCP_BIFROST_BASE_URL` | `http://localhost:8081` | Bifrost gateway URL |
| `CCP_BIFROST_API_KEY` | — | Bifrost API key |
| `CCP_PORT` | `8080` | Listen port |
| `CCP_LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warn`, `error` |
| `CCP_REQUEST_TIMEOUT` | `120` | Backend request timeout in seconds |
| `CCP_MAX_CONCURRENT` | `10` | Max concurrent requests |

## Model Mapping

| Claude Code model | Backend model |
|---|---|
| `claude-opus-*` | `deepseek-v4-pro` |
| `claude-sonnet-*` | `deepseek-v4-flash` |
| `claude-haiku-*` | `deepseek-v4-flash` |

## Testing

```bash
make test
```

## Architecture

```
Claude Code → CCP (:8080) → Ollama Cloud or Bifrost → Azure AI Foundry
```

CCP handles Anthropic↔OpenAI translation. Bifrost handles Azure auth, routing,
failover, and observability.

## Debugging

```bash
CCP_LOG_LEVEL=debug ./ccp
```

This logs full request and response bodies for troubleshooting.