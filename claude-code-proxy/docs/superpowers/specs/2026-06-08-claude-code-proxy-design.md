# Claude Code Proxy (CCP) — Design Document

## Overview

**CCP** (Claude Code Proxy) is a zero-dependency Go proxy that translates Anthropic's Messages API (`POST /v1/messages`) into OpenAI's Chat Completions API (`POST /v1/chat/completions`), enabling Claude Code to use DeepSeek V4 models hosted on Ollama Cloud or Azure AI Foundry (via Bifrost).

- **Project folder:** `~/projects/deepseek/claude-code-proxy/`
- **Binary:** `ccp`
- **Env var prefix:** `CCP_`

## Architecture

```
Claude Code
    │  ANTHROPIC_BASE_URL=http://localhost:8080
    │  ANTHROPIC_AUTH_TOKEN=dummy
    ▼
┌─────────────────────────────────────┐
│  ccp (Go, :8080)                    │
│  translate.go  — Anthropic↔OpenAI   │
│  backend.go    — Backend selection  │
│  main.go       — HTTP server, env   │
└──────┬──────────────┬───────────────┘
       │              │
       │ CCP_BACKEND= │ CCP_BACKEND=
       │ ollama       │ bifrost
       ▼              ▼
  Ollama Cloud    Bifrost (:8081)
  /v1/chat/           │
  completions         ▼
                 Azure AI Foundry
                 DeepSeek V4 Pro/Flash
```

Bifrost handles Azure-specific complexity (api-key auth, API version, deployment routing, failover, observability). CCP stays focused on Anthropic↔OpenAI translation.

### Configuration

```bash
# Ollama Cloud
CCP_BACKEND=ollama CCP_OLLAMA_API_KEY="sk-..." ./ccp

# Azure AI Foundry via Bifrost
CCP_BACKEND=bifrost CCP_BIFROST_BASE_URL="http://localhost:8081" CCP_BIFROST_API_KEY="bf-" ./ccp

# Optional
CCP_PORT=8080           # default
CCP_LOG_LEVEL=info      # debug, info, warn, error
CCP_REQUEST_TIMEOUT=120 # seconds
CCP_MAX_CONCURRENT=10
```

Claude Code side:
```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=dummy
claude
```

## Model Mapping

| Claude Code requests | → | Ollama / Bifrost model |
|---|---|---|
| `claude-opus-*` | → | `deepseek-v4-pro` |
| `claude-sonnet-*` | → | `deepseek-v4-flash` |
| `claude-haiku-*` | → | `deepseek-v4-flash` |
| Unknown model | → | `deepseek-v4-flash` (safe fallback) |

## Backend Abstraction

```
Backend {
    BaseURL     string
    AuthHeader  string
    AuthValue   string
    ChatPath    string
}
```

| Aspect | Ollama Cloud | Bifrost (Azure backend) |
|---|---|---|
| **Base URL** | `https://ollama.com/v1` | `http://localhost:8081` (configurable) |
| **Chat endpoint** | `/chat/completions` | `/v1/chat/completions` |
| **Auth header** | `Authorization: Bearer {key}` | `Authorization: Bearer {key}` |
| **Streaming** | SSE `data:` lines | Standard OpenAI SSE |
| **Model names** | `deepseek-v4-pro`, `deepseek-v4-flash` | Same (via Bifrost routing) |

## Request Translation: Anthropic → OpenAI

### Messages Flattening

#### Content Block Mapping

| Anthropic content block | OpenAI message role | Notes |
|---|---|---|
| `{type:"text", text:"..."}` in user message | `role:"user"`, `content:"..."` | String content |
| `{type:"tool_result", tool_use_id, content}` in user message | `role:"tool"`, `tool_call_id`, `content` | Each tool_result → separate message |
| Mixed text + tool_result in user message | Split: text → `role:"user"`, each tool_result → `role:"tool"` | Preserves ordering |
| Multiple tool_results in user message | Multiple `role:"tool"` messages, one per result | |
| `{type:"tool_use", id, name, input}` in assistant message | `role:"assistant"`, `tool_calls[{id,type,function}]` | Content may be `""` or text |
| `{type:"text", text:"..."}` in assistant message | `role:"assistant"`, `content:"..."` | |
| Mixed text + tool_use in assistant message | `role:"assistant"`, `content:"..."`, `tool_calls:[...]` | |
| `{type:"image"}` or `{type:"document"}` | **Rejected: 400 error** | Unsupported |

#### Adjacent Same-Role Merging

After flattening, adjacent messages with the same role are merged. Adjacent `user` messages have their content concatenated. Adjacent `tool` messages are kept separate (each references a different tool_call_id).

#### Empty Content / Tool-Only Assistant Messages

When an assistant message has only tool_uses (no text), `content` is set to `""`. When it has text + tool_uses, `content` is set to the text and `tool_calls` in parallel. When it has only text, normal pass-through.

### System Message

| Anthropic input | OpenAI output |
|---|---|
| `system: "string"` | `messages[0] = {role:"system", content:"..."}` |
| `system: [{type:"text", text:"a"}, {type:"text", text:"b"}]` | Concatenate: `"a\nb"` |

`cache_control` on system blocks is **stripped**.

### Tool Definitions

| Anthropic field | OpenAI field |
|---|---|
| `tools[].name` | `tools[].function.name` |
| `tools[].description` | `tools[].function.description` |
| `tools[].input_schema` | `tools[].function.parameters` |

### Tool Choice

| Anthropic value | OpenAI value |
|---|---|
| `"auto"` | `"auto"` |
| `"any"` | `"required"` |
| `{type:"tool", name:"X"}` | `{type:"function", function:{name:"X"}}` |

### Other Request Fields

| Anthropic field | Handling |
|---|---|
| `stop_sequences` | → `stop` (rename) |
| `max_tokens` | Pass through |
| `temperature` | Pass through |
| `top_p` | Pass through |
| `top_k` | **Drop** |
| `stream` | Pass through |
| `metadata.user_id` | Pass through |
| `thinking` | **Drop** — Ollama Cloud doesn't expose DeepSeek reasoning-mode control; V4 auto-reasons by default |
| `cache_control` | **Strip** — DeepSeek uses automatic prefix caching |
| `x-anthropic-billing-header` (cch stamp) | **Strip** — prevents prefix cache busting |
| `anthropic-beta` / `anthropic-version` headers | **Drop** |

## Response Translation: OpenAI → Anthropic

### Non-Streaming

#### Text Response

OpenAI `choices[0].message.content` → Anthropic `content: [{type:"text", text:"..."}]`, `stop_reason: "end_turn"`.

#### Tool Call Response

OpenAI `choices[0].message.tool_calls[]` → Anthropic `content: [{type:"tool_use", id, name, input}]`, `stop_reason: "tool_use"`. The arguments string is JSON-parsed into an `input` object.

#### Reasoning → Thinking Block

OpenAI `choices[0].message.reasoning` → placed as the first content block: `{type:"thinking", thinking:"..."}` followed by text/tool_use blocks.

### Finish Reason Mapping

| OpenAI `finish_reason` | Anthropic `stop_reason` |
|---|---|
| `"stop"` | `"end_turn"` |
| `"tool_calls"` | `"tool_use"` |
| `"length"` | `"max_tokens"` |

### Streaming Translation

#### Text Streaming Event Flow

```
Ollama SSE:                          Anthropic SSE:
───────────────────────────────────────────────────
                                     event: message_start
                                     data: {type:"message_start", message:{...}}

data: {choices:[{delta:{content:"H"}}]}
                                     event: content_block_start
                                     data: {type:"content_block_start", content_block:{type:"text", text:""}}

                                     event: content_block_delta
                                     data: {type:"content_block_delta", delta:{type:"text_delta", text:"H"}}

data: {choices:[{finish_reason:"stop"}]}
                                     event: content_block_stop
                                     data: {type:"content_block_stop"}

                                     event: message_delta
                                     data: {type:"message_delta", delta:{stop_reason:"end_turn"}}

                                     event: message_stop
                                     data: {type:"message_stop"}
data: [DONE]
```

#### Reasoning Streaming

Ollama emits `delta.reasoning` tokens before `delta.content` tokens. The proxy emits a `thinking` content block for reasoning deltas, then switches to a `text` content block when `delta.content` appears.

#### Tool Call Streaming — Incremental JSON Splitting

Ollama emits the entire tool call in a single chunk. The proxy splits it into incremental `content_block_delta` events with valid partial JSON:

```json
"{\"cmd\":\"ls\"}" →
  {"partial_json":"{\""}
  {"partial_json":"cmd"}
  {"partial_json":"\":\""}
  {"partial_json":"ls"}
  {"partial_json":"\"}"}
```

For multiple tool calls, emit separate `content_block_start/stop` pairs.

#### Ping

Emit `event: ping` every 5 seconds during idle stream.

## Concurrency & Infrastructure

| Feature | Implementation |
|---|---|
| **Parallel requests** | Go goroutines — each request independent |
| **Graceful shutdown** | SIGINT/SIGTERM → stop accepting, drain in-flight, exit |
| **Concurrent limit** | `CCP_MAX_CONCURRENT=10` — semaphore gate |
| **Request timeout** | `CCP_REQUEST_TIMEOUT=120s` — configurable |
| **Streaming body** | `io.Reader` passthrough, no buffering |

## Observability

| Feature | Implementation |
|---|---|
| **Health endpoint** | `GET /health` → `{"status":"ok","backend":"ollama|bifrost"}` |
| **Logger** | `log/slog` — structured JSON output |
| **Log fields** | `request_id`, `model`, `backend`, `duration_ms`, `status`, `tok_in`, `tok_out` |
| **Request IDs** | Forward `x-request-id` through chain |
| **Debug mode** | `CCP_LOG_LEVEL=debug` logs full request/response bodies |
| **Token counting** | Stub `POST /v1/messages/count_tokens` returning estimate |
| **Version flag** | `./ccp --version` |

## Error Handling

| Scenario | Response |
|---|---|
| Backend 4xx | Map body to Anthropic error: `{type:"error", error:{type:"api_error", message:"..."}}` |
| Backend 5xx | 502, Anthropic error shape |
| Backend timeout | 504, Anthropic error shape |
| Malformed request | 400, Anthropic error shape |
| Unsupported content (image/document) | 400, descriptive error |

## File Structure

```
~/projects/deepseek/claude-code-proxy/
├── main.go              # Entry, env parsing, backend selection, HTTP server
├── backend.go           # Backend struct, Ollama/Bifrost constructors
├── translate.go         # Anthropic ↔ OpenAI request/response translation
├── go.mod               # Zero external dependencies (stdlib only)
├── go.sum
├── test_ccp.py          # Multi-backend test tool (~35 tests, zero deps)
├── run_tests.sh         # go build && python3 test_ccp.py
├── Makefile             # build, test, run, install targets
└── README.md            # Setup and usage instructions
```

## Test Coverage

The test tool (`test_ccp.py`) is a single Python file with zero dependencies. It starts a mock backend server, starts the ccp binary, and runs 35+ test cases.

### Test Categories

1. **Turn-Based Conversation Flow (7 tests)**
2. **Model Mapping (5 tests)**
3. **Streaming SSE Remapping (5 tests)**
4. **Field Translation (11 tests)**
5. **Error/Edge Cases (5 tests)**
6. **Finish Reason Mapping (3 tests)**
7. **Backend-Specific (3 tests)**

Multi-backend via `--mock=ollama` or `--mock=bifrost`.

## Design Decisions Summary

| # | Decision | Rationale |
|---|---|---|
| Q1 | Map `reasoning` → `thinking` content block | Faithful Anthropic format, first in content[] |
| Q2 | Merge adjacent same-role messages | Cleaner conversation for the model |
| Q3 | Handle content per message (text, tool-only, mixed) | Correct for all message types |
| Q4 | Drop `thinking` parameter from requests | Ollama Cloud doesn't support reasoning-mode control |

## Limitations

- Image/document content blocks rejected
- `anthropic-beta` / `anthropic-version` headers dropped
- Thinking depth control not possible through Ollama Cloud
- `top_k` dropped (no OpenAI equivalent)
- `cache_control` stripped (DeepSeek auto-caches regardless)
