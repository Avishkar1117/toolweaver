"""The agent loop as a LangGraph state machine.

``plan -> call_tool -> observe -> decide -> (loop back to plan | answer)``

``plan`` asks the DeepSeek model (bound to the tool schemas) for the next action:
call a tool, or answer. Observations from prior tool calls are fed back into the
plan prompt, so the model chooses tools and arguments itself. ``decide`` is the
hard loop guard (step cap); the model decides natural completion in ``plan``.
Hitting the cap flags ``did_not_converge`` and returns a best-effort answer;
tool failures are caught and fed back as observations so the graph never crashes.
"""

import operator
from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .config import settings
from .prompts import PLANNER_SYSTEM
from .tools import TOOLS, ToolResult, openai_tool_schemas

# Loop guard. Phase 4 adds did_not_converge flagging when it is exceeded.
MAX_STEPS = 5


class State(TypedDict):
    """What flows through the graph. ``messages`` and ``observations`` accumulate
    (reducer = list concatenation); everything else is overwritten each step,
    which is the deliberate context-management choice for the scratchpad."""

    question: str
    messages: Annotated[list[dict], operator.add]
    scratchpad: str
    observations: Annotated[list[dict], operator.add]
    step_count: int
    last_tool_called: str | None
    pending_args: dict | None
    planner_system: str
    final_answer: str | None
    done: bool
    did_not_converge: bool


_planner = None


def get_planner() -> Any:
    """Lazily build the DeepSeek planner bound to the tool schemas. Lazy so that
    importing this module needs no API key; offline tests monkeypatch this."""
    global _planner
    if _planner is None:
        from langchain_openai import ChatOpenAI

        _planner = ChatOpenAI(
            model=settings.agent_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
        ).bind_tools(openai_tool_schemas())
    return _planner


def _format_observations(observations: list[dict]) -> str:
    """Render observations for the plan prompt. A failed call shows its error,
    not its (empty) output -- otherwise the planner can't see *why* a tool failed
    and just retries blindly to the step cap. This is what makes the Phase 4
    tool-failure-as-observation guard actually usable by the model."""
    lines = []
    for o in observations:
        if o.get("ok", True):
            lines.append(f"- {o['tool']}({o['args']}) -> {o['output']}")
        else:
            lines.append(f"- {o['tool']}({o['args']}) -> ERROR: {o.get('error') or o['output']}")
    return "\n".join(lines)


# --- nodes -----------------------------------------------------------------


def plan(state: State) -> dict[str, Any]:
    """Ask the model for the next action: call a tool, or answer directly."""
    human = state["question"]
    context = _format_observations(state["observations"])
    if context:
        human = f"{state['question']}\n\nObservations so far:\n{context}"

    resp = get_planner().invoke([SystemMessage(state["planner_system"]), HumanMessage(human)])

    if resp.tool_calls:
        call = resp.tool_calls[0]  # one tool per step keeps the loop tight
        note = f"plan: call {call['name']}({call['args']})"
        return {
            "last_tool_called": call["name"],
            "pending_args": call["args"],
            "scratchpad": note,
            "messages": [{"role": "planner", "content": note}],
        }

    # No tool call -> the model is ready to answer.
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {
        "final_answer": text,
        "messages": [{"role": "planner", "content": "plan: answer"}],
    }


def call_tool(state: State) -> dict[str, Any]:
    """Run the selected tool, catching ANY failure -- bad args, a raised
    exception, a sandbox kill -- and feeding it back as an ok=False observation
    so the planner can retry, switch tools, or give up. The graph never crashes
    on a tool error (CLAUDE.md §3). A broad catch is correct here: the whole point
    is that no tool failure escapes the loop."""
    tool_name = cast(str, state["last_tool_called"])
    args = state["pending_args"] or {}
    try:
        result = TOOLS[tool_name].run(**args)
    except Exception as exc:
        result = ToolResult(ok=False, output="", error=f"{type(exc).__name__}: {exc}")
    obs = {
        "tool": tool_name,
        "args": args,
        "ok": result.ok,
        "output": result.output,
        "error": result.error,
    }
    return {
        "observations": [obs],
        "step_count": state["step_count"] + 1,
        "pending_args": None,
        "messages": [{"role": "tool", "content": result.output}],
    }


def observe(state: State) -> dict[str, Any]:
    """Summarise the latest tool output into the scratchpad (context management)."""
    last = state["observations"][-1]
    summary = f"observed {last['tool']}: {last['output']}"
    return {"scratchpad": summary, "messages": [{"role": "observer", "content": summary}]}


def decide(state: State) -> dict[str, Any]:
    """Hard loop guard: stop at the step cap (the model ends naturally in plan).
    Hitting the cap means we answer best-effort without the model having
    converged, so flag it -- this is also the steps-to-convergence signal."""
    capped = state["step_count"] >= MAX_STEPS
    return {"done": capped, "did_not_converge": capped}


def answer(state: State) -> dict[str, Any]:
    """Emit the final answer. If we stopped at the step cap without one, fall back
    to a best-effort answer from the last observation."""
    final = state["final_answer"] or _best_effort(state)
    return {
        "final_answer": final,
        "done": True,
        "messages": [{"role": "assistant", "content": final}],
    }


def _best_effort(state: State) -> str:
    if state["observations"]:
        return f"(step cap reached) latest observation: {state['observations'][-1]['output']}"
    return "(no answer)"


def _route_after_plan(state: State) -> str:
    return "answer" if state["final_answer"] else "call_tool"


def _route_after_decide(state: State) -> str:
    return "answer" if state["done"] else "plan"


# --- assembly --------------------------------------------------------------


def build_graph() -> Any:
    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("call_tool", call_tool)
    g.add_node("observe", observe)
    g.add_node("decide", decide)
    g.add_node("answer", answer)

    g.set_entry_point("plan")
    g.add_conditional_edges(
        "plan", _route_after_plan, {"call_tool": "call_tool", "answer": "answer"}
    )
    g.add_edge("call_tool", "observe")
    g.add_edge("observe", "decide")
    g.add_conditional_edges("decide", _route_after_decide, {"plan": "plan", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()


_GRAPH = build_graph()


def run_agent(question: str, planner_system: str = PLANNER_SYSTEM) -> State:
    """Run the graph headlessly and return the final state. Both the API and the
    eval harness call this -- no HTTP, one brain, two mouths."""
    initial: State = {
        "question": question,
        "messages": [],
        "scratchpad": "",
        "observations": [],
        "step_count": 0,
        "last_tool_called": None,
        "pending_args": None,
        "planner_system": planner_system,
        "final_answer": None,
        "done": False,
        "did_not_converge": False,
    }
    return cast(State, _GRAPH.invoke(initial))
