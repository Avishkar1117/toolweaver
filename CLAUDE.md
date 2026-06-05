# P3 — Agentic Tool-Use Agent — Build Spec

> **For Claude Code.** This is the source of truth for building P3. Defer to it over any assumption. If something here conflicts with a habit or a "best practice," follow this doc. If something is genuinely ambiguous or missing, **ask before guessing** — do not invent scope.
>
> **Usage:** save this as `CLAUDE.md` in the repo root so it's read automatically, or keep this filename and reference it explicitly at the start of a session.

---

## 0. Operating principles (how to work on this repo)

Apply these four on every change. They are not decoration — they define acceptable output here.

1. **Think before coding.** Surface assumptions out loud. If a request has multiple valid interpretations or relies on an unstated decision, state it and ask. Do not silently pick one.
2. **Simplicity first.** Minimum code that solves the problem. No speculative abstractions, no error handling for impossible cases, no "production-ready" features that weren't asked for. The test: *would a senior engineer call this overcomplicated?*
3. **Surgical changes.** Touch only what the task needs. Match existing style. Do not refactor adjacent code "while you're here." Small, focused commits with conventional-commit messages.
4. **Goal-driven execution.** Every phase below has a Definition of Done (DoD). Build to the DoD, verify it, then move on. Do not start the next phase until the current one is verified.

**Two things are the owner's to provide, not yours to generate.** They are flagged `[OWNER]` below. The most important: **the evaluation tasks.** Do not auto-generate eval questions or answers. Build the harness; leave the task content for the owner.

---

## 1. What we're building

A **LangGraph agent** that takes a question, plans, selects and calls tools, observes the results, loops until confident, and emits a final answer **plus a full reasoning trace**. Served behind a thin FastAPI endpoint. Evaluated over a hand-designed task set.

**Framing / use case:** a research/QA assistant that answers multi-step questions by autonomously choosing and chaining tools (web search, document retrieval, computation).

**This is a showcase of the agentic-orchestration mechanism, not a domain product.** Its value is proving (a) clean tool-use orchestration and (b) rigorous agent evaluation. Keep it general and legible.

### Non-goals — do NOT build these

These are "extend later," not part of this build. Building any of them is a scope violation:

- No email / calendar / "personal assistant" tools. (Domain integration belongs to a later project.)
- No multi-agent collaboration, no long-term memory store, no human-in-the-loop approval step.
- No tool plugin system / dynamic tool discovery framework.
- No LLM-provider abstraction layer beyond a single config value (see §6).
- No advanced retrieval (BM25/hybrid/GraphRAG/reranking) — doc-lookup reuses the existing P2 store as-is.
- No gVisor/Firecracker/microVM sandbox (see §5).

---

## 2. Architecture — the load-bearing decision

**The agent core is a plain importable library (`src/agent/`). It imports nothing web/server-related.** The FastAPI app and the eval harness are two **independent clients** that import the library.

Why this matters: the eval must run the graph **headlessly and in-process** (fast, CI-able, no HTTP round-trips). The API is a thin wrapper. One brain, two mouths. Do not let FastAPI types or request objects leak into `src/agent/`.

---

## 3. The agent loop (LangGraph state machine)

Nodes: `plan → call_tool → observe → decide → (loop back to plan | answer)`.

### State object (typed)

Hold at least:
- `question` — the original input
- `messages` / step history
- `scratchpad` — the current plan / intermediate reasoning
- `observations` — accumulated tool outputs
- `step_count` — loop guard counter
- `last_tool_called` — for routing + the trajectory check
- `final_answer` and a `done` flag

What you carry forward vs drop each loop **is** context management — keep it deliberate; don't let `messages`/`observations` grow unbounded.

### Hard guards — both must be implemented

- **Step cap.** A max-steps limit checked in `decide`. On exceeding it, return a best-effort answer flagged `did_not_converge=True`. (This also produces the steps-to-convergence metric.) The agent must never loop forever.
- **Tool failure as observation.** Wrap every tool call. On failure (bad args, timeout, sandbox kill), catch the error and feed it back as an observation so the agent can retry, switch tools, or abort cleanly. **Never crash the graph on a tool error.**

---

## 4. Tools (3–4)

- `web_search` — external search.
- `calculator` — simple arithmetic. (Exists to force the planner to *choose* a tool; keep it minimal.)
- `doc_lookup` — queries **P2's existing Chroma store**. Reuse the existing persistent store and embeddings; do not rebuild a corpus.
- `code_exec` — a **thin client** to the sandbox (§5) that runs generated code.

All tools sit behind **one simple interface** (a `Protocol` or a thin base — not an abstract-class hierarchy), each with a **Pydantic arg schema**. Validate args against the schema **before** executing the tool.

### MCP

Expose **`doc_lookup` as an MCP server**, and have the agent **consume it as an MCP client** (both ends). This is the realistic MCP use case (a data/context source over the protocol) and reuses the P2 corpus.

**Only this one tool goes through MCP.** The rest stay as direct functions. Do not MCP everything.

---

## 5. Sandbox (code execution isolation)

A Docker container that runs **untrusted, model-generated code**. Its defining property is that it has **no ambient access**.

Requirements:
- `--network none` (no outbound network)
- no host filesystem mount (or minimal, read-only)
- CPU / memory / pids / wall-clock limits
- non-root user, dropped capabilities

`code_exec` passes code in, runs it, captures stdout/result/errors, returns them. Any data the code needs must be passed in **explicitly as input** — the sandbox cannot reach the corpus, the network, or the host.

**Include a test that proves isolation holds** (e.g. code attempting a network call fails).

**Scope guard:** a container with network off + resource limits + a timeout + non-root is **done**. Do not build anything heavier.

---

## 6. Model configuration

- **Agent model:** DeepSeek V4, **starting with the Flash variant**, accessed via DeepSeek's OpenAI-compatible API. Put the model behind **a single config/env value** so it can be swapped (e.g. to V4-Pro) without code changes.
- `[OWNER]` Before trusting it across the full eval: smoke-test **function-calling reliability against our actual tool schemas** (an agentic-coding benchmark is not proof of strict tool-calling). Default **"thinking" mode off** for simple tool calls to control latency/cost; reserve it for the plan step if needed.
- **Do not** hardcode the model and **do not** build a model-comparison framework. The required comparison in this project is **prompting strategies**, not models (see §8).

---

## 7. Serving (FastAPI)

Thin wrapper importing `src/agent/`. One endpoint (`POST /ask`) taking a question, returning `{answer, trace}`. Reuse P2's FastAPI/uvicorn patterns. **Deployment is the last step** (HF Spaces, Docker SDK, mirror P2) — do it after the loop and eval work, not before.

---

## 8. Evaluation — the differentiator. Read carefully.

> **The eval tasks are `[OWNER]`-provided.** They are hand-designed and human-verified. **DO NOT generate eval questions or answers, and do not bulk-create tasks from documents.** Your job is the harness, the graders, and the `tasks.yaml` *schema* — not the task content. Ship `tasks.yaml` with a documented schema and 1–2 placeholder examples for the owner to fill.

### Task schema (per task, in `tasks.yaml`)
- `id`
- `question`
- `type` — one of `gold` | `open` | `trajectory`
- `success` — for `gold`: expected value + match mode (exact / numeric-tolerance / substring/regex). For `open`: a **binary, concrete** rubric (e.g. "names the correct entity? cites a source?").
- `required_tool` — tool(s) the task should exercise (for the trajectory check).

### Graders (`graders.py`) — three kinds
- **Gold-answer:** programmatic check (exact / numeric tolerance / substring/regex). Expected to be the **majority** of tasks.
- **LLM judge:** uses a **different-family model from the agent** — reuse the owner's existing **Gemini** integration. Binary rubric per task. For the **minority** open-ended tasks only.
- **Trajectory:** did `required_tool` appear in the trace? Runs alongside the answer check (catches "right answer by hallucination").

### Runner (`runner.py`)
- Imports the graph **directly** (no HTTP).
- Runs each task **N times** (default 3) — the agent is stochastic.
- Records per run: outcome, `step_count`, trajectory hit, full trace.

### Report (`report.py` or inline)
- **Completion rate** (% of runs passing).
- **Average steps-to-convergence.**
- **Per-prompting-strategy delta** — prompts live in `prompts.py` and are swappable so two strategies can be compared. This delta is the headline result.

### CI
**Do not run the full eval in CI.** Run a **3–4 task smoke subset** as the PR gate (mirror P2's RAGAS-in-CI gate). The full eval is a manual/nightly run.

---

## 9. Repo structure (target)

Keep it flat. **Split a file only when it earns its existence** — by length (a module past ~200–300 lines) or by being independently reused/tested. Do not pre-split, and do not collapse for the sake of fewer files.

```
agentic-tool-use/
├── src/agent/
│   ├── graph.py        # State + nodes + graph assembly + step cap (one file while small)
│   ├── prompts.py      # planner prompts, kept separate so eval can swap strategies
│   ├── tools.py        # the 4 tools + the simple Tool interface (Protocol, not an ABC tower)
│   ├── sandbox.py      # the container runner / client
│   └── mcp_server.py   # the one MCP-exposed tool (doc_lookup)
├── api/main.py         # thin FastAPI wrapper, imports src/agent
├── eval/
│   ├── tasks.yaml      # [OWNER]-provided task content; ship schema + placeholders only
│   ├── runner.py       # runs the graph in-process, N times per task
│   └── graders.py      # gold-answer · llm_judge(rubric) · trajectory
├── tests/              # per-tool units, graph smoke test, sandbox-isolation test
├── docker/
│   ├── Dockerfile          # app, multi-stage (reuse P2)
│   └── sandbox.Dockerfile  # minimal, locked down
├── docker-compose.yml
├── pyproject.toml
├── .github/workflows/ci.yml
└── README.md
```

---

## 10. Tooling & conventions (reuse P2)

`uv`, `ruff`, `mypy`, `pytest`. Conventional commits; small, surgical commits. Multi-stage `Dockerfile` + `docker-compose`, mirroring P2's setup. `structlog` JSON logging. Lightweight call/cost tracking only if it's cheap to add.

---

## 11. Build order (phased — verify each DoD before the next phase)

1. **Skeleton.** Repo structure, `pyproject.toml`, the typed `State`, empty tool **stubs** (consistent interface, return canned data), graph wiring (nodes call the stubs), FastAPI `/ask` wrapper.
   **DoD:** project boots; the graph runs end-to-end against stub tools.
2. **Real tools, one at a time:** `calculator` → `web_search` → `doc_lookup` (wire to P2's Chroma) → `code_exec` + sandbox.
   **DoD per tool:** unit test passes; agent uses it in the loop.
3. **MCP.** Expose `doc_lookup` as an MCP server; agent consumes it as a client.
   **DoD:** agent answers a doc question via the MCP path; contract/isolation test passes.
4. **Loop hardening.** Step cap + tool-failure-as-observation + reasoning-trace output.
   **DoD:** a deliberately failing tool does not crash the graph; an over-long task returns `did_not_converge`.
5. **Eval harness.** `tasks.yaml` schema + the three graders + `runner.py` + `report.py`. (Task content from `[OWNER]`.)
   **DoD:** runner executes the owner's seed tasks and emits completion rate + avg steps; smoke subset wired into CI.
6. **Deploy.** Dockerize, push to HF Spaces.
   **DoD:** live endpoint answers a question.
7. **Docs.** README + record a Loom walkthrough.

---

## 12. `[OWNER]` inputs — do not block on or fabricate these

- **The eval task set** — hand-designed, human-verified. Provided by the owner. Build the harness around them; do not generate them.
- **DeepSeek V4 function-calling verification** + final Flash/Pro choice (decided by eval results).
- **API keys / secrets** — via environment only, never hardcoded or committed.

---

## 13. Definition of done (whole project)

A strong P3 artifact has: a working agent loop with the two hard guards; 3–4 tools with one exposed over MCP; an isolated code sandbox with an isolation test; an eval producing **completion rate + avg steps-to-convergence + a prompting-strategy delta** plus a trajectory check; a deployed `/ask` endpoint; and a clean README with a Loom walkthrough. Clean, tested, readable code — and an **efficient agent loop** (few LLM calls, tight context, the step cap doing its job) — matter more than file count or feature breadth.
