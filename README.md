# DeepSeek V4 + Claude Code: hosting & Anthropic-endpoint research

Research notes on running **Claude Code** against **DeepSeek V4** — self/local-hosted (vLLM, SGLang, llama.cpp), on **Azure Foundry**, or via **DeepSeek's cloud** — and on the proxies/routers that sit in between. Findings on the Anthropic-compatible API layer and caching are **verified at the source level** (LiteLLM, llama.cpp, claude-code-router), not just from docs.

📄 **Full write-up:** [deepseek-v4-flash-local-hosting.md](./deepseek-v4-flash-local-hosting.md) (Sections 1–13)

---

## Executive summary — running Claude Code on DeepSeek V4

Three things have to line up: **(1) where the model runs, (2) what speaks the Anthropic Messages API, and (3) whether caching survives.** They are independent — the API layer working tells you nothing about caching or architecture support.

### Decision matrix

| Backend | Anthropic endpoint via | Tool calling | Caching outcome | Verdict |
|---|---|---|---|---|
| **DeepSeek cloud** (`api.deepseek.com/anthropic`) | native (first-party) | yes | **automatic disk cache, ~98% off** | ✅ Best for cheap agent loops; zero ops |
| **Self-host vLLM/SGLang** | vLLM native `/v1/messages`, or LiteLLM | yes (parser flags) | prefix caching = **speed only, no $** | ✅ Full control; needs 2×H200-class GPUs |
| **Self-host llama.cpp** | native `/v1/messages` | yes | prefix caching, **actively defended** vs Claude Code cache-buster | ⚠️ Needs a V4 GGUF + CSA/HCA arch support (verify) |
| **Azure Foundry DeepSeek V4** | none for DeepSeek — needs a proxy | **No (Preview)** | cache_control stripped; V4 prefix caching unconfirmed; no LiteLLM price entry | ❌ Weakest path today |

### Proxy / tooling cheat-sheet

| Layer | When you need it | Caching note |
|---|---|---|
| **none** (env vars only) | backend already exposes native Anthropic (`api.deepseek.com/anthropic`, vLLM, llama.cpp) | best case |
| **Switcher** (`ccs`, deepclaude…) | flip Claude Code between native-Anthropic backends | passthrough; no transform |
| **LiteLLM** | OpenAI-only backend, multi-model routing, auth, cost tracking | **strips `cache_control` for non-Claude** |
| **claude-code-router** | per-task routing (background/think/longContext/webSearch) + transformers | **keeps `cache_control`** unless provider strips; ⚠️ v2.0.0 clamps `max_tokens` to 8192 |

The lessons behind these tables are in [§13 of the full doc](./deepseek-v4-flash-local-hosting.md#13-lessons-learned).

---

## What's in the full doc

| § | Topic |
|---|---|
| 1–2 | Model sizing reality check & hardware tiers |
| 3–5 | Two-layer architecture; download & serve (vLLM/SGLang); adding the Anthropic endpoint |
| 6–7 | Feature-parity caveats; recommended self-host path |
| 8 | Azure / Microsoft Foundry — why DeepSeek V4 gets no native Anthropic endpoint |
| 9 | Prompt caching across cloud / self-hosted / Azure |
| 10 | LiteLLM source-level verification (Anthropic→OpenAI proxy + caching) |
| 11 | llama.cpp's native Anthropic compatibility layer |
| 12 | Routers & switchers for Claude Code |
| 13 | Lessons learned |

> ⚠️ Research notes, not an official guide. Model details (DeepSeek V4, Azure Foundry availability, proxy internals) reflect the state at the time of writing and change fast — re-verify before relying on them.
