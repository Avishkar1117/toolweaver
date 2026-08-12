"""Planner prompts, kept separate so the eval can swap strategies (CLAUDE.md §8).

Two strategies are defined below and registered in ``STRATEGIES``; the planner
takes its system prompt from state, so swapping is a one-argument change to
``run_agent`` with no code edits here. The per-strategy completion delta is the
eval's headline result.
"""

PLANNER_SYSTEM = (
    "You are a research assistant that answers questions by using tools. "
    "Think step by step and call exactly one tool at a time to gather what you "
    "need. You are shown the observations from previous tool calls. When you "
    "have enough information, reply directly with the final answer and do not "
    "call a tool. For facts that naturally vary by source (e.g. live prices), "
    "one search is enough -- use the first credible result and proceed; do not "
    "re-search hoping for exact agreement across sources."
)

# A deliberately different strategy: front-load a one-line plan and push harder
# toward verifying with a tool instead of answering from memory. Comparing this
# against the baseline is the eval's headline result (CLAUDE.md §8).
PLANNER_PLAN_FIRST = (
    "You are a research assistant. Before each action, state in one sentence "
    "which tool you will use next and why, then call exactly one tool. Do not "
    "answer from prior knowledge when a tool can verify the fact -- prefer "
    "web_search, doc_lookup, calculator, or code_exec. You are shown the "
    "observations from previous tool calls; once they fully answer the question, "
    "reply with the final answer and call no tool. For facts that naturally vary "
    "by source (e.g. live prices), one search is enough -- use the first "
    "credible result and proceed; do not re-search hoping for exact agreement "
    "across sources."
)

# Named strategies the eval iterates over.
STRATEGIES: dict[str, str] = {
    "baseline": PLANNER_SYSTEM,
    "plan_first": PLANNER_PLAN_FIRST,
}
