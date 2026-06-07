# Hosting DeepSeek V4‑Flash locally with an Anthropic-compatible endpoint

## TL;DR

DeepSeek-V4-Flash is a **284B-total / 13B-active MoE** model (released open-source, MIT, April 24 2026) that needs **~170–175 GB of VRAM** in its native FP4+FP8 format. You host it with **vLLM or SGLang**, then expose an **Anthropic Messages API** (`/v1/messages`) on top — either via **vLLM's built-in Anthropic endpoint** (simplest) or a **translation proxy** (LiteLLM / claude-code-proxy). The thing the "DeepSeek lab" does is exactly this: their `https://api.deepseek.com/anthropic` URL maps `claude-opus-*` → `deepseek-v4-pro` and `claude-sonnet/haiku-*` → `deepseek-v4-flash`. You replicate that mapping locally.

> **Executive summary, decision matrix, and proxy cheat-sheet** have moved to the [README](./README.md). The lessons behind them are in [§13](#13-lessons-learned).

---

## 1. What the model actually is (sizing reality check)

| | V4-Flash | V4-Pro (for contrast) |
|---|---|---|
| Total params | 284B | ~1.6T |
| Active params | 13B | 49B |
| Attention | CSA + HCA hybrid (compressed sparse) | same |
| Context | up to 1M tokens | 1M |
| Reasoning modes | Non-think / Think High / Think Max | same |
| Native precision | FP4 (experts) + FP8 (rest) | same |
| **VRAM (full)** | **~170–175 GB** | ~862 GB |

The 13B *active* parameters give you fast tokens/sec, but you still pay for **all 284B in VRAM** because every expert must be resident. This is the single most important planning fact.

---

## 2. Hardware tiers

**Comfortable (recommended, full quality):**
- **2× H200** (282 GB) — the configuration LMSYS used for Day-0 testing (TP=4 across the pair's GPUs)
- **2× RTX Pro 6000 Blackwell** (192 GB)
- **4× A100 80 GB** (320 GB) — vLLM prefers this because tensor parallelism wants power-of-two GPU counts

**Budget (quality tradeoff):**
- Community **INT4** quants (GGUF/AWQ/GPTQ) shrink it to **~80 GB** → fits on a single H100/H200 or 2× 48 GB cards, but with *measurable degradation on math, reasoning, and agentic tasks* — which is exactly what you'd use Claude Code for, so be cautious.

**Bare minimum (slow):**
- Single 48 GB+ GPU + **256 GB system RAM** using **KTransformers** with CPU offload + FP4. Functional, not pleasant for interactive coding.

For ~175 GB you also need to budget the **KV cache** — full 1M context adds ~10 GB; capping `--max-model-len` to 128K is the usual starting point.

---

## 3. Architecture: the two layers

```
                 Anthropic Messages API (/v1/messages)
Claude Code  ─────────────────────────────────────────►  ┌─────────────────┐
(or Anthropic SDK)   ANTHROPIC_BASE_URL                   │  Translation /  │
                                                          │  native endpoint│
                                                          └────────┬────────┘
                                                       OpenAI /v1/chat/completions
                                                                   │
                                                          ┌────────▼────────┐
                                                          │  vLLM / SGLang  │
                                                          │  V4-Flash weights│
                                                          └─────────────────┘
```

Claude Code and the Anthropic SDK speak the **Messages API** (separate `system`, different tool-call shape, different SSE streaming events). Inference engines natively speak **OpenAI Chat Completions**. You need something to bridge them. Two ways:

- **Option A — vLLM native** (vLLM now implements `/v1/messages` directly; no extra proxy)
- **Option B — a proxy** (LiteLLM, claude-code-proxy, UniClaudeProxy) in front of either engine

---

## 4. Step-by-step: download + serve the model

```bash
pip install "huggingface_hub"
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ./deepseek-v4-flash
```
Use the **Instruct FP4+FP8 mixed-precision** checkpoints for production.

### vLLM (V4 needs vLLM ≥ 0.8.0)

```bash
pip install "vllm>=0.8.0"

vllm serve ./deepseek-v4-flash \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 4 \        # = your GPU count; use 2 for 2×H200
  --max-model-len 131072 \          # raise toward 1M only if VRAM allows
  --trust-remote-code \
  --enable-auto-tool-choice \       # required for Claude Code tool use
  --tool-call-parser deepseek_v4 \  # use the parser matching the model
  --port 8000
```

### SGLang alternative (≥ 0.4.4)

SGLang shipped a Day-0 recipe and benchmarks ~29% faster than vLLM on H100 for general workloads. It supports DP/TP/SP/EP/PP/CP plus EAGLE/MTP speculative decoding. The exact flag set lives in the [SGLang Cookbook → DeepSeek-V4](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4) entry; the shape is `python -m sglang.launch_server --model-path ./deepseek-v4-flash --tp 4 --tool-call-parser ...` with FP4 MoE and MTP enabled.

Both engines expose an **OpenAI-compatible** server at `http://localhost:8000/v1`.

---

## 5. Adding the Anthropic endpoint (the part that matches DeepSeek lab)

### Option A — vLLM's built-in Anthropic Messages API (simplest)

vLLM serves `/v1/messages` **by default, no extra flag**. Just point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_AUTH_TOKEN=dummy          # vLLM doesn't enforce auth
export ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4-flash
export ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4-flash
export ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-v4-flash
claude
```

The three `ANTHROPIC_DEFAULT_*_MODEL` vars are how you reproduce DeepSeek's mapping locally. (DeepSeek maps opus→V4-Pro and sonnet/haiku→V4-Flash; with only Flash hosted, you point all three at `deepseek-v4-flash`. If you also host Pro, set the opus var to it.) Model names must match `--served-model-name` exactly and **must not contain slashes**.

### Option B — translation proxy (most faithful to the lab; engine-agnostic)

Use this if your engine lacks native `/v1/messages`, or you want DeepSeek-style model remapping, key auth, and request logging in one place.

**LiteLLM** (`config.yaml`):
```yaml
model_list:
  - model_name: deepseek-v4-flash
    litellm_params:
      model: openai/deepseek-v4-flash
      api_base: http://localhost:8000/v1
      api_key: dummy
```
```bash
litellm --config config.yaml --port 4000
export ANTHROPIC_BASE_URL=http://localhost:4000
```
LiteLLM accepts Anthropic-formatted `/v1/messages` requests and translates them to OpenAI for vLLM, returning Anthropic-shaped responses (including the streaming event remap) — which is precisely what DeepSeek's gateway does.

Lightweight purpose-built alternatives: **`claude-code-proxy`** and **`UniClaudeProxy`** (FastAPI, full tool-calling + streaming + ReAct XML fallback, hot-reload config).

---

## 6. Feature parity caveats (what won't fully work)

DeepSeek's own Anthropic endpoint documents these limits, and your local setup inherits the same constraints:

- **Supported:** `max_tokens`, `stop_sequences`, `stream`, `system`, `temperature`, `top_p`, tool definitions, `thinking` mode.
- **Partial:** `metadata` (only `user_id`), `output_config` (only `effort`), some `tool_choice` variants.
- **Not supported:** image/document content blocks, `anthropic-beta`/`anthropic-version` headers, MCP tools at the API layer.
- **Reasoning modes:** map Claude Code's effort to V4's *Non-think / Think High / Think Max* via the `thinking`/`effort` field (DeepSeek uses `CLAUDE_CODE_EFFORT_LEVEL`).

---

## 7. Recommended path

Given the goal of mirroring the DeepSeek lab experience:

1. **2× H200** (or 4× A100 80GB) running **vLLM ≥ 0.8.0** with the FP4+FP8 Instruct weights, `--max-model-len 131072`, tool-call parser + auto tool choice enabled.
2. **vLLM's native `/v1/messages`** for the Anthropic endpoint (skip a separate proxy unless you need multi-model routing/auth).
3. Set the three `ANTHROPIC_DEFAULT_*_MODEL` env vars to do the opus/sonnet/haiku→Flash mapping.
4. If you later add V4-Pro, front both with **LiteLLM** to replicate DeepSeek's exact opus→Pro / sonnet→Flash routing behind one URL and one key.

---

## 8. Azure / Microsoft Foundry: does DeepSeek V4 get a native Anthropic endpoint?

**Short answer: No.** Azure-hosted DeepSeek V4 does *not* expose a native Anthropic endpoint the way Azure-hosted **Claude Opus** does. On Microsoft Foundry the two model families are served on different API surfaces.

### The two are served differently

| Model on Foundry | API surface exposed |
|---|---|
| **Claude Opus 4.x / Sonnet** | **Native Anthropic Messages API** → `https://<resource>.services.ai.azure.com/anthropic/v1/messages` |
| **DeepSeek-V4-Pro / V4-Flash** (Preview) | **`chat-completion` (with reasoning content)** — the generic Azure AI / OpenAI-style chat completions surface |

Claude is the special case: Microsoft hosts Anthropic's first-party models and re-exposes Anthropic's *real* `/anthropic/v1/messages` surface, so Claude Code can point straight at it (via `ANTHROPIC_FOUNDRY_API_KEY` or the Azure default credential chain). DeepSeek V4 gets **no Anthropic surface** — it only answers on the standard Foundry chat-completions endpoint.

### Why this matters for Claude Code

1. **No `/v1/messages` for DeepSeek V4 on Azure.** You can't just set `ANTHROPIC_BASE_URL` to a Foundry DeepSeek endpoint. You'd need a **translation proxy** (LiteLLM / claude-code-proxy) in front of the Foundry chat-completions endpoint — the same Option B pattern from §5, just pointed at Azure instead of your own GPUs.
2. **Tool calling is currently `No`** for both `DeepSeek-V4-Pro` and `DeepSeek-V4-Flash` on Foundry (they're Preview). That's a real blocker for agentic Claude Code use. For contrast, `DeepSeek-V3.1` and `V3-0324` on Foundry list tool calling = `Yes`. Verify tool-calling support landed before relying on Foundry-hosted V4 for agent workflows.
3. **If you want DeepSeek V4 + a native Anthropic endpoint, use DeepSeek's own cloud, not Azure.** `https://api.deepseek.com/anthropic` natively speaks Anthropic, does the `claude-opus-*`→V4-Pro / `claude-sonnet|haiku-*`→V4-Flash mapping, and tool calling works on the compat layer.

### Options ranked

- **Native Anthropic + DeepSeek V4:** DeepSeek's own `api.deepseek.com/anthropic` (works today, has mapping + tools).
- **Native Anthropic + on Azure infra:** only **Claude** models qualify (the Opus path). DeepSeek can't replicate it.
- **DeepSeek V4 on Azure with Claude Code:** Foundry chat-completions endpoint **+ a LiteLLM/proxy shim** to translate Anthropic↔OpenAI — and check tool-calling support first.
- **Full control:** self-host (§1–7), where vLLM gives you a native `/v1/messages` directly.

---

## 9. Prompt caching: how it differs across self-hosted, DeepSeek cloud, and Azure

A common point of confusion (raised by the [deepclaude](https://github.com/aattaran/deepclaude) note *"DeepSeek has its own caching (automatic), but Anthropic's `cache_control` is ignored"*) is that **two unrelated caching mechanisms get conflated**:

1. **Anthropic explicit prompt caching** — the `cache_control: {type: "ephemeral"}` breakpoints Claude Code stamps onto the system prompt / tool defs / large context, with a TTL. *You* mark what to cache.
2. **DeepSeek-style automatic prefix caching** — no headers, no opt-in. The server hashes the prompt prefix; if a new request shares a prefix with a recent one, the overlapping KV is reused. Fully transparent.

deepclaude's note describes the **Claude Code → DeepSeek cloud** path: Claude Code keeps emitting `cache_control`, DeepSeek silently **drops it** (mechanism #1 is a no-op), but DeepSeek's **automatic** caching (#2) still kicks in — so agent loops stay cheap (~$0.004/M cache-hit vs ~$0.44/M uncached) without the Anthropic directives doing anything.

### How each deployment behaves

**Self-hosted (vLLM / SGLang):** same pattern, *different payoff*.
- `cache_control` is **ignored** — vLLM's native `/v1/messages` doesn't honor Anthropic ephemeral-cache semantics (open RFC [vllm#8333](https://github.com/vllm-project/vllm/issues/8333) to add it; not standard). A LiteLLM proxy strips it during OpenAI translation.
- **Automatic prefix caching IS the real analog**: vLLM `--enable-prefix-caching` (default-on in the V1 engine); SGLang RadixAttention is always on. Reuses KV blocks across shared-prefix requests.
- **Crucial difference: the win is latency/throughput, not a billing discount.** You own the GPUs — there's no per-token price, so no "$0.004/M" magic; prefix caching just lowers time-to-first-token by skipping redundant prefill.
- **Persistence caveat:** the prefix cache lives in GPU/CPU memory, is evicted under memory pressure and lost on restart (unless you configure KV offload to CPU/disk). DeepSeek cloud uses a *persistent disk* cache, so its hit rate across idle gaps is better.
- Bonus: vLLM `cache_salt` isolates cache reuse per user/tenant in multi-user setups.

**Azure / Microsoft Foundry DeepSeek V4:** the deepclaude statement mostly **doesn't apply**.
- `cache_control` is **moot** — Foundry serves DeepSeek V4 only via the OpenAI-style `chat-completion` API (see §8), which has no `cache_control` field. A proxy for Claude Code drops it anyway.
- DeepSeek's automatic cache savings are **not guaranteed on Azure**: Microsoft's Q&A states prefix/prompt caching is **not currently supported for DeepSeek V4 Pro on Foundry** (no ETA), and documented prompt-caching *discounts* cover only Azure-OpenAI models. Fireworks-served Foundry pricing does list a "cached input" rate (~$0.15/M vs $1.75/M for V4-Pro), but realizing hits isn't an officially supported behavior yet. The first-party economics that make deepclaude cheap are a property of **DeepSeek's own API**, not Azure.

### Summary

| | `cache_control` honored? | Automatic prefix caching? | What you actually save |
|---|---|---|---|
| **DeepSeek cloud** (`api.deepseek.com/anthropic`) | No (dropped) | Yes, persistent disk | **$$ — ~98% off cached tokens** |
| **Self-hosted vLLM/SGLang** | No (no-op / stripped) | Yes, in-memory (opt. disk offload) | **Latency/throughput, not $** |
| **Azure Foundry DeepSeek V4** | N/A (no Anthropic surface) | Not officially supported for V4 yet | **Uncertain — don't count on it** |

**Bottom line:** if cheap agent loops via caching are the goal, **DeepSeek's own cloud is the only option that delivers the deepclaude experience as written.** Self-hosting gives the speed benefit but no dollar discount; Azure gives neither the Anthropic path nor a confirmed DeepSeek-style cache discount for V4 yet.

---

## 10. LiteLLM source-level verification (Anthropic→OpenAI proxy + caching)

Verified against the LiteLLM source at commit `1cff02f` (2026-06-06). Conclusions are grounded in the actual code, not docs.

### LiteLLM does expose an Anthropic endpoint and transform it to OpenAI

Two routes are registered in `litellm/proxy/anthropic_endpoints/endpoints.py`:
- `/v1/messages` (beta)
- `/anthropic/v1/messages` (recommended)

Routing logic lives in `litellm/llms/anthropic/experimental_pass_through/messages/handler.py`. It calls `ProviderConfigManager.get_provider_anthropic_messages_config(...)`; providers with a **native** Anthropic handler (real Anthropic, Bedrock-Claude, Azure-Claude) pass through natively, while **everything else — including DeepSeek and Azure-AI DeepSeek — falls to the transformation path** (handler.py:487–500):

```python
if _should_route_to_responses_api(custom_llm_provider):   # OpenAI / Azure-OpenAI
    return LiteLLMMessagesToResponsesAPIHandler...
return LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(...)
```

So a DeepSeek request gets the Anthropic→OpenAI `/chat/completions` transformation. **This works with Claude Code**: point `ANTHROPIC_BASE_URL` at the LiteLLM proxy and it accepts Messages API requests, translating them for the DeepSeek backend.

### `cache_control` is provably stripped for DeepSeek

The deepclaude observation is enforced in code. In `adapters/transformation.py:335`:

```python
cache_control = source.get("cache_control") ...
if cache_control and model and self.is_anthropic_claude_model(model):
    target["cache_control"] = cache_control
```

→ `cache_control` is propagated **only if the target is an Anthropic Claude model.** For DeepSeek (any host), Claude Code's cache breakpoints are dropped during transformation.

It is stripped a second time on the Azure path. Azure Foundry serves DeepSeek **via Fireworks**, and `fireworks_ai/chat/transformation.py:259` does:
```python
filter_value_from_dict(cast(dict, message), "cache_control")
# Remove fields not permitted by FireworksAI (additionalProperties: false)
```
Additionally `AzureAIStudioConfig(OpenAIConfig)` has an auto-drop-on-422 retry that removes any field Foundry rejects. So `cache_control` cannot reach an Azure-Foundry DeepSeek backend even if forced.

### Cost-tracking gaps for V4 in `model_prices_and_context_window.json`

- **No `azure_ai/deepseek-v4-flash` or `azure_ai/deepseek-v4-pro` entry** exists (only `v3.2`, `v3.2-speciale`, `r1`, `v3`, `v3-0324`).
- **No direct `deepseek/...v4` entry** either (newest is `deepseek/deepseek-v3.2`).
- Existing `azure_ai/deepseek-v3.2` has `supports_prompt_caching: true` but `cache_read_input_token_cost: null` → cache hits are billed at full input cost in LiteLLM's tracking.

### What this means

| Question | Answer from the code |
|---|---|
| LiteLLM gives Claude Code a working Anthropic→DeepSeek bridge? | **Yes** (completion-transformation path). |
| Claude Code's `cache_control` reaches DeepSeek? | **No** — dropped in adapter (non-Claude gate) + Fireworks filter + Azure auto-drop. |
| Cache savings on **Azure Foundry** DeepSeek V4? | **Largely no** — cache_control gone, V4-on-Azure prefix caching unconfirmed (§8/§9), no V4-azure price entry. |
| Better route if caching matters? | **Direct DeepSeek cloud** (automatic disk cache works regardless of cache_control); register the v4 model + cache pricing manually in LiteLLM for accurate cost tracking. |

**Bottom line:** LiteLLM can technically front Azure-Foundry DeepSeek V4 for Claude Code, but the caching economics behind deepclaude **do not transfer to the Azure path** — `cache_control` is provably stripped, and Azure V4 has neither confirmed prefix caching nor a LiteLLM cost entry. For cheap cached agent loops, the **direct DeepSeek API** remains the only path where caching actually pays off.

---

## 11. llama.cpp's native Anthropic compatibility layer

Verified against the llama.cpp source at commit `98d5e8b` (2026-06-06). llama.cpp's `llama-server` ships a **genuine, fairly complete Anthropic Messages API** — full request + response + streaming translation, not a thin shim.

### What's there

**Endpoints** (`tools/server/server.cpp`):
```
196:  ctx_http.post("/v1/messages",              ...post_anthropic_messages);     // anthropic messages API
213:  ctx_http.post("/v1/messages/count_tokens", ...post_anthropic_count_tokens);
```
Auth via the Anthropic-style `X-Api-Key` header (`server-http.cpp:202`).

**Request conversion** — `server_chat_convert_anthropic_to_oai()` (`server-chat.cpp:325+`) maps the Anthropic schema to llama.cpp's internal OpenAI shape:
- `system` (string or array of text blocks)
- content blocks: `text`, `thinking` → `reasoning_content`, `image` (base64 **and** url → `image_url`), `tool_use` → `tool_calls`, `tool_result` → `tool` role messages
- top-level `tools` (`name`/`description`/`input_schema` → OpenAI function tools)
- `tool_choice` (`auto`; `any`/`tool` → `required`)
- `stop_sequences` → `stop`, `max_tokens` (default 4096), `temperature`/`top_p`/`top_k`/`stream`
- `thinking` (`enabled` + `budget_tokens` → `thinking_budget_tokens`), `metadata.user_id`

**Response conversion** — real round-trip: `to_json_anthropic()` / `to_json_anthropic_stream()` (`server-task.cpp:1137, 1203`) + `format_anthropic_sse()` emit genuine Anthropic responses and SSE events. Tested in `tools/server/tests/unit/test_compat_anthropic.py`.

**→ Works with Claude Code** by setting `ANTHROPIC_BASE_URL` to the `llama-server` — no proxy needed.

### Caching standout (contrast with §9–§10)

llama.cpp goes further than ignoring caching — it **protects its prefix cache from Claude Code**. `normalize_anthropic_billing_header()` (`server-chat.cpp:303`, PR [#21793](https://github.com/ggml-org/llama.cpp/pull/21793)) neutralizes the cache-busting stamp Claude Code injects:

```
// This is a claude message with a "cch=ef01a" attribute that breaks prefix caching.
// The cch stamp is a whitebox end-to-end integrity hint... its presence means the prefix
// cache will not get past it: It changes on each request.
```

Claude Code injects `x-anthropic-billing-header: ...; cch=<random>;` into the system prompt, mutating every request — which would bust llama.cpp's prefix cache each call. llama.cpp overwrites the `cch=` value with a constant (`fffff`) so the **KV prefix cache survives across requests**.

| Path | `cache_control`/Anthropic caching | Prefix caching with Claude Code |
|---|---|---|
| **LiteLLM → DeepSeek** | stripped (Claude-only gate + Fireworks filter) | backend-dependent |
| **vLLM/SGLang native** | ignored (no-op) | automatic, but cch stamp may reduce hits |
| **llama.cpp** | not honored (no billing concept) | **actively defended** — defeats the cch cache-buster |

### Caveats

- **API layer only** — it converts to llama.cpp's internal handling, so you still need a **GGUF** loaded. For DeepSeek V4-Flash that's a community INT4/GGUF quant (~80 GB, with the quality degradation from §2) **and** llama.cpp actually supporting V4's CSA+HCA hybrid attention — verify separately; a working Anthropic endpoint does not imply V4 architecture support.
- Not converted: `document` blocks, `redacted_thinking`, container/MCP tools, `anthropic-version`/beta headers, `disabled`/`none` `tool_choice`.
- Like vLLM/SGLang: no dollar discount, only speed — you own the hardware.

**Net:** llama.cpp's Anthropic layer is real and the best-engineered for Claude Code of the options reviewed (the only one that *defends* its cache against Claude Code's cache-buster). For DeepSeek V4 the gating question is GGUF + architecture support, not the API.

---

## 12. Routers & switchers for Claude Code (and how CCR handles caching)

Two genres of community tooling sit between Claude Code and alternative backends:

### Switchers (lightweight env-var flippers)

They don't transform anything — they flip `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_MODEL` to point Claude Code at a provider's **own** Anthropic-compatible endpoint:

- **[maxgfr/claude-code-switch](https://github.com/maxgfr/claude-code-switch)** (`ccs`) — zero-dependency sidecar; Anthropic, OpenRouter, DeepSeek, Z.AI, Kimi, Qwen, MiniMax, or custom endpoints.
- **[lucas-stellet/switcher](https://github.com/lucas-stellet/switcher)** — launches Claude Code with different providers.
- **[foreveryh/claude-code-switch](https://github.com/foreveryh/claude-code-switch)** (`ccm`) — Bash CLI exporting Anthropic-compatible env vars.
- **[aattaran/deepclaude](https://github.com/aattaran/deepclaude)** — switches Anthropic↔DeepSeek mid-session via a slash command (proxy on `localhost:3200`).

A switcher is enough when the target already exposes a native Anthropic endpoint (e.g. `api.deepseek.com/anthropic`).

### Router — `musistudio/claude-code-router` (CCR)

[github.com/musistudio/claude-code-router](https://github.com/musistudio/claude-code-router) — a proxy that **routes each request to a different backend by task**, with its own per-provider transformers. Verified against source at commit `e270dea` (v2.0.0, 2026-03-04).

**Routing scenarios** (`packages/core/src/utils/router.ts`), in priority order:
- per-project/session `Router` override from `config.json`
- `longContext` — when token count > `longContextThreshold` (default 60000) or last-usage input tokens exceed it
- `<CCR-SUBAGENT-MODEL>...</CCR-SUBAGENT-MODEL>` tag in the system prompt → explicit subagent model
- `background` — any `claude`+`haiku` model → cheap background model
- `webSearch` — when a `web_search` tool is present (higher priority than thinking)
- `think` — when `thinking` is set
- else `default`

**Anthropic endpoint:** `AnthropicTransformer` registers `endPoint = "/v1/messages"` and handles `x-api-key`/Bearer auth — so Claude Code points straight at CCR.

**Caching — CCR differs from LiteLLM (important):**
- The inbound `AnthropicTransformer` **preserves `cache_control`** — it carries it through on system text blocks (`anthropic.transformer.ts:64`) and tool messages (`:99`). CCR is permissive by default.
- Provider transformers **strip it only where the backend rejects it**: `groq.transformer.ts` and `vercel.transformer.ts` `delete` it; `openai.responses.transformer.ts` deletes it.
- **`deepseek.transformer.ts` does NOT touch `cache_control`** — it only clamps `max_tokens` to 8192 and reshapes `reasoning_content`↔`thinking` streaming. So routing to DeepSeek **forwards** `cache_control` (DeepSeek ignores it server-side and does its own automatic caching anyway — net effect identical to §9).

So vs LiteLLM (which gates on `is_anthropic_claude_model` and strips for all non-Claude): **CCR keeps `cache_control` unless a provider explicitly removes it**, and DeepSeek keeps it. Either way the practical caching outcome on DeepSeek is the same — automatic prefix/disk caching does the work; explicit `cache_control` is a server-side no-op.

> ⚠️ The pinned v2.0.0 `deepseek.transformer.ts` hardcodes a **`max_tokens` clamp to 8192** (a V3-era limit). For V4 (384K output) this would truncate — check for a newer release before using CCR with V4.

### Which to use

| Need | Tool |
|---|---|
| Point Claude Code at a native Anthropic endpoint | a **switcher** (`ccs`, etc.) |
| Switch Anthropic↔DeepSeek mid-session | **deepclaude** |
| Per-task routing (Flash background / Pro think / long-context) + transformation | **claude-code-router** |

---

## 13. Lessons learned

Distilled from §1–§12 — what actually matters if you want Claude Code on DeepSeek V4, whether self/local-hosted or on Azure, with or without a proxy in between.

1. **Three layers are independent — don't conflate them.** *Where the model runs* (GPUs/quant), *what speaks the Anthropic API* (native vs proxy), and *whether caching pays off* are separate questions. A working `/v1/messages` endpoint says nothing about architecture support or caching. Most confusion comes from assuming "the endpoint works" means "it's cheap and correct."

2. **"Anthropic-compatible" is the easy part; the model architecture is the hard part.** Every layer (vLLM, llama.cpp, LiteLLM, CCR, DeepSeek cloud) can translate the Messages API. The real gate for V4 is whether your engine supports its CSA+HCA attention and whether a usable quant exists — verify that first, not the API.

3. **Two caching mechanisms, never confused again:** Anthropic's explicit `cache_control` breakpoints vs DeepSeek-style automatic prefix/disk caching. On DeepSeek, `cache_control` is **always a no-op** (server-side ignored); automatic caching does all the work. So whether a proxy forwards or strips `cache_control` is mostly irrelevant to DeepSeek — but it tells you a lot about the proxy's design.

4. **Caching savings are a property of DeepSeek's *own cloud*, not of DeepSeek the model.** Self-hosting converts the benefit from **dollars to latency** (you already own the GPUs). Azure converts it to **nothing reliable** (no confirmed V4 prefix caching, no cost-tracking entry). If "95× cheaper agent loops" is the goal, only `api.deepseek.com` delivers it as advertised.

5. **Azure Foundry is the trap.** It looks like the enterprise-grade option but is the weakest for this use case: no native Anthropic surface for DeepSeek (Claude gets one, DeepSeek doesn't), **tool calling = No in Preview** (a hard blocker for Claude Code agents), and caching that doesn't transfer. Use Azure for Claude-on-Azure, not DeepSeek-for-Claude-Code.

6. **Proxies have opinions about `cache_control` — read the source.** LiteLLM **strips** it for any non-Claude model (`is_anthropic_claude_model` gate) and Fireworks strips it again; claude-code-router **preserves** it unless a provider transformer deletes it. Neither is "wrong," but you can't know without reading the code — which is why source-level verification beat docs every time here.

7. **llama.cpp is the quiet standout for local + Claude Code.** It's the only layer that *defends* its prefix cache against Claude Code's per-request `cch` cache-buster stamp (PR #21793). If you self-host and care about cache hit-rate under Claude Code specifically, that engineering matters.

8. **Match the tool to the actual need — don't over-build.** Native endpoint + env vars is enough for one backend. Reach for a **switcher** to flip between native-Anthropic backends, **LiteLLM** for OpenAI-only backends / cost tracking / auth, and **claude-code-router** only when you genuinely want per-task routing (cheap model for background, reasoning model for think). Every proxy adds a translation surface that can silently drop fields (tools, `cache_control`, `max_tokens`).

9. **Watch for stale hardcoded limits in proxies.** CCR v2.0.0 clamps `max_tokens` to 8192 (a V3-era constant) — it predates V4 and would truncate output. Pin proxy versions and re-verify against the model you're actually running.

10. **Practical recommendation:**
    - *Cheapest, least ops:* DeepSeek cloud `api.deepseek.com/anthropic` + a switcher.
    - *Full control / privacy:* self-host on vLLM (native `/v1/messages`) or llama.cpp (best cache behavior), FP4+FP8 weights, 2×H200-class.
    - *Fancy routing:* claude-code-router in front of either — after confirming its V4 handling.
    - *Azure:* only if you're committed to Foundry governance and can tolerate a proxy + no caching + waiting for tool-calling GA.

---

## Sources

- [DeepSeek-V4-Flash on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [DeepSeek-V4-Flash specs & VRAM (APXML)](https://apxml.com/models/deepseek-v4-flash)
- [Self-Hosting DeepSeek V4: vLLM, Hardware & Deployment (Lushbinary)](https://lushbinary.com/blog/deepseek-v4-self-hosting-guide-vllm-hardware-deployment/)
- [Run DeepSeek V4 Flash Locally — 2026 setup (Codersera)](https://codersera.com/blog/run-deepseek-v4-flash-locally-full-2026-setup-guide/)
- [DeepSeek-V4 on Day 0 with SGLang (LMSYS)](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) · [SGLang Cookbook: DeepSeek-V4](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [DeepSeek Anthropic API docs](https://api-docs.deepseek.com/guides/anthropic_api)
- [vLLM Claude Code / Anthropic Messages API integration](https://docs.vllm.ai/en/stable/serving/integrations/claude_code/)
- [Running Claude Code with local LLMs via vLLM + LiteLLM (DEV)](https://dev.to/dcruver/running-claude-code-with-local-llms-via-vllm-and-litellm-599b)
- [UniClaudeProxy (GitHub)](https://github.com/vibheksoni/UniClaudeProxy)
- [Foundry Models sold by Azure — DeepSeek + Claude API surfaces](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure)
- [Introducing DeepSeek V4 Flash and V4 Pro in Microsoft Foundry](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-deepseek-v4-flash-and-v4-pro-in-microsoft-foundry/4515174)
- [Deploy and use Claude models in Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-claude)
- [Claude Code on Microsoft Foundry](https://code.claude.com/docs/en/microsoft-foundry)
- [deepclaude repo (aattaran)](https://github.com/aattaran/deepclaude)
- [DeepSeek Context Caching on Disk](https://api-docs.deepseek.com/guides/kv_cache) · [pricing announcement](https://api-docs.deepseek.com/news/news0802)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) · [RFC #8333 Anthropic-style pinned caching](https://github.com/vllm-project/vllm/issues/8333)
- [How do I use prefix caching with DeepSeek V4 Pro? — Microsoft Q&A](https://learn.microsoft.com/en-in/answers/questions/5899380/how-do-i-use-prefix-caching-with-deepseek-v4-pro)
- [Prompt caching with Azure OpenAI (scope = OpenAI models only)](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/prompt-caching)
- [LiteLLM source (BerriAI/litellm)](https://github.com/BerriAI/litellm) — verified at commit `1cff02f` (2026-06-06): `litellm/proxy/anthropic_endpoints/endpoints.py`, `litellm/llms/anthropic/experimental_pass_through/messages/handler.py`, `.../adapters/transformation.py`, `litellm/llms/fireworks_ai/chat/transformation.py`, `litellm/llms/azure_ai/chat/transformation.py`, `model_prices_and_context_window.json`
- [llama.cpp source (ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp) — verified at commit `98d5e8b` (2026-06-06): `tools/server/server.cpp`, `server-chat.cpp`, `server-task.cpp`, `server-http.cpp`, `tests/unit/test_compat_anthropic.py` · [PR #21793 (prefix-cache cch normalization)](https://github.com/ggml-org/llama.cpp/pull/21793)
- [claude-code-router (musistudio/claude-code-router)](https://github.com/musistudio/claude-code-router) — verified at commit `e270dea` (v2.0.0, 2026-03-04): `packages/core/src/utils/router.ts`, `packages/core/src/transformer/{anthropic,deepseek,groq,vercel,openai.responses}.transformer.ts`
- Switchers: [maxgfr/claude-code-switch](https://github.com/maxgfr/claude-code-switch) · [lucas-stellet/switcher](https://github.com/lucas-stellet/switcher) · [foreveryh/claude-code-switch](https://github.com/foreveryh/claude-code-switch) · [aattaran/deepclaude](https://github.com/aattaran/deepclaude)
