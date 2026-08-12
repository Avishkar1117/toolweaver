---
title: Toolweaver
emoji: "\U0001F916"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
short_description: LangGraph agent that plans, picks tools, shows its trace
---

# Toolweaver

A LangGraph agent that takes a question, **plans, selects and chains tools,
observes the results, and loops until confident**, then returns a final answer
plus a full reasoning trace. Served behind a thin FastAPI `/ask` endpoint and
evaluated over a hand-designed task set.

The point of the project is the *mechanism*: clean tool-use orchestration and
rigorous agent evaluation, not a domain product.

## Live demo

- **Space:** https://huggingface.co/spaces/Avishkar1117/agentic-tool-use
- **Swagger UI:** https://avishkar1117-agentic-tool-use.hf.space/docs

Hosted on a free-tier Hugging Face Space (Docker SDK). The Space sleeps after
inactivity, so the first request may take ~30 s while the container wakes.

```bash
curl -X POST https://avishkar1117-agentic-tool-use.hf.space/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What optimizer trained the base Transformer models?"}'
```

## The agent loop

`plan → call_tool → observe → decide → (loop | answer)`, a LangGraph state
machine. Two hard guards keep it honest:

- **Step cap**: a max-steps limit; on exceeding it the agent returns a
  best-effort answer flagged `did_not_converge` instead of looping forever.
- **Tool failure as observation**: every tool call is wrapped; a failure is fed
  back as an observation so the agent can retry, switch tools, or answer around
  it. The graph never crashes on a tool error.

## Tools (4)

| Tool | What it does |
|---|---|
| `calculator` | Safe arithmetic (AST-evaluated, no `eval`). |
| `web_search` | External search via Tavily. |
| `doc_lookup` | Retrieves from a local document corpus, exposed over **MCP** (the agent is the MCP client; the server owns the Chroma + Gemini access). |
| `code_exec` | Runs model-generated Python in an isolated Docker sandbox (`--network none`, non-root, resource + wall-clock limits). |

### Hosted vs local: one image, two modes

`code_exec` needs the host Docker daemon to spawn its sandbox, which managed
hosting can't provide. So the **same image** runs two ways:

- **Hosted (HF Spaces):** `SANDBOX_ENABLED=false`, 3 tools live; `code_exec`
  returns a clean message and the agent answers around it.
- **Local (`docker compose`, socket mounted, `SANDBOX_ENABLED=true`):** all 4
  tools live.

## Evaluation

`eval/` runs the graph in-process (no HTTP), N times per task, and grades with
three graders: **gold** (programmatic), **LLM judge** (Gemini, a different model
family from the DeepSeek agent), and **trajectory** (did the required tool appear
in the trace?). The headline metric is the **per-prompting-strategy delta** over
the same task set. A 3-4 task smoke subset is the CI gate; the full eval is a
manual run.

```bash
python -m eval.runner          # full eval (needs API keys + Docker for code_exec tasks)
```

### Results (n=8/strategy, all 14 tasks)

| strategy | completion | avg steps | trajectory hit | non-convergence |
|---|---|---|---|---|
| baseline | 96% | 1.82 | 95% | 4% |
| plan_first | 98% | 1.73 | 100% | 2% |

13 of 14 tasks converge at 100% completion. The one that does not:

| task | completion | notes |
|---|---|---|
| `open-doc-vs-web` | 62% | LLM-judged, open-ended comparison of a corpus figure against a live web search result. The agent converges every run; the judge sometimes rejects the written comparison as incomplete. |

### Known limitations and further tuning

This task set is a reasonable first pass, not a finished instrument.

- **`open-doc-vs-web`'s score is a rubric ceiling, not an agent bug.** Tightening the judge's rubric, or giving it more structure, would likely move this number without changing agent behavior at all.
- **`selection-unprompted-calc` shows a real, measured strategy difference.** Under `baseline`, the agent sometimes skips the required `calculator` tool and does the arithmetic itself, still landing on the correct answer, so the wrong-process behavior was invisible in the completion number and only shows up in the trajectory-hit column. Under `plan_first`, it uses the tool on every run. This is exactly what a trajectory check is for: catching a right answer produced the wrong way.
- **The task set was hand-written for tool coverage, not tuned to this model's specific behavior or to any particular downstream use case.** It was still good enough to find real bugs during development (a retrieval snippet budget silently truncating relevant results, a planner that would not commit to noisy search results, a numeric grader that misread bolded vs. plain answers). A more targeted task set, shaped around DeepSeek's actual answer formatting habits and around whatever this agent ends up being used for, would be a better long-term instrument than the current general-purpose one.

## Run locally

```bash
cp .env.example .env           # fill in DEEPSEEK / TAVILY / GEMINI keys
docker compose up --build      # API on http://localhost:8000
```

## Configuration

All via environment / `.env` (never committed):

| Var | Purpose |
|---|---|
| `DEEPSEEK_API_KEY` | the agent's planner model (DeepSeek, OpenAI-compatible) |
| `TAVILY_API_KEY` | `web_search` backend |
| `GEMINI_API_KEY` | `doc_lookup` query embeddings + the eval LLM judge |
| `SANDBOX_ENABLED` | `false` on managed hosting, `true` for live `code_exec` |
| `DOC_CHROMA_PATH` | path to the bundled corpus (baked into the image) |
