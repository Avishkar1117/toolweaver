"""Graph + /ask smoke tests.

The default run uses a fake planner so the loop is exercised offline (no API key,
no network) -- this is the CI-safe gate. The `live` test drives the real DeepSeek
planner and is excluded from the default run (see pyproject `-m "not live"`).
"""

import pytest
from fastapi.testclient import TestClient

from agent import graph
from agent.graph import run_agent
from agent.tools import TOOLS
from api.main import app


class _FakeMsg:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


class _FakePlanner:
    """Calls the calculator once, then answers -- exercises the whole loop."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return _FakeMsg(
                tool_calls=[{"name": "calculator", "args": {"expression": "40 + 2"}, "id": "1"}]
            )
        return _FakeMsg(content="The answer is 42.")


@pytest.fixture
def fake_planner(monkeypatch):
    planner = _FakePlanner()  # one shared instance so the call counter persists
    monkeypatch.setattr(graph, "get_planner", lambda: planner)


def test_all_four_tools_registered():
    assert set(TOOLS) == {"web_search", "calculator", "doc_lookup", "code_exec"}


def test_graph_runs_end_to_end(fake_planner):
    state = run_agent("What is 40 + 2?")
    assert state["final_answer"] == "The answer is 42."
    assert state["last_tool_called"] == "calculator"
    assert state["observations"][0]["output"] == "42"  # the real calculator ran
    assert state["step_count"] == 1
    assert state["done"] is True


def test_root_serves_demo_page():
    # GET / returns the self-contained HTML demo (no planner needed) -- this is
    # what HF embeds in the Space iframe; without it the root is a bare 404.
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Toolweaver" in resp.text


def test_ask_endpoint(fake_planner):
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "What is 40 + 2?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "The answer is 42."
    assert body["trace"]["last_tool_called"] == "calculator"


# --- loop hardening (Phase 4) ----------------------------------------------


class _FailThenAnswerPlanner:
    """Calls the real calculator with a div-by-zero expression (it raises), then
    answers -- exercises tool-failure-as-observation through the real tool path."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return _FakeMsg(
                tool_calls=[{"name": "calculator", "args": {"expression": "1/0"}, "id": "1"}]
            )
        return _FakeMsg(content="Could not compute: division by zero.")


class _NeverAnswersPlanner:
    """Always asks for another tool call, never answers -- drives the step cap."""

    def invoke(self, messages):
        return _FakeMsg(
            tool_calls=[{"name": "calculator", "args": {"expression": "1 + 1"}, "id": "1"}]
        )


def test_failing_tool_does_not_crash_graph(monkeypatch):
    planner = _FailThenAnswerPlanner()  # one instance so the call counter persists
    monkeypatch.setattr(graph, "get_planner", lambda: planner)
    state = run_agent("What is 1/0?")
    # The graph completed instead of raising ZeroDivisionError ...
    assert state["final_answer"] == "Could not compute: division by zero."
    # ... and the failure was fed back as an ok=False observation.
    failed = state["observations"][0]
    assert failed["ok"] is False
    assert "ZeroDivisionError" in failed["error"]
    assert state["did_not_converge"] is False  # it converged (model answered)


def test_failed_observation_surfaces_error_to_planner():
    """A failed call must show its error (not its empty output) in the plan prompt
    -- otherwise the planner can't see *why* a tool failed and just retries to the
    step cap. This is what lets the code_exec gate (and any failure) be relayed."""
    obs = [
        {
            "tool": "code_exec",
            "args": {"code": "print(1)"},
            "ok": False,
            "output": "",
            "error": "code_exec is disabled on this hosted demo",
        },
        {"tool": "calculator", "args": {"expression": "1+1"}, "ok": True, "output": "2"},
    ]
    text = graph._format_observations(obs)
    assert "ERROR: code_exec is disabled on this hosted demo" in text
    assert "calculator({'expression': '1+1'}) -> 2" in text


def test_step_cap_flags_did_not_converge(monkeypatch):
    planner = _NeverAnswersPlanner()
    monkeypatch.setattr(graph, "get_planner", lambda: planner)
    state = run_agent("Loop forever.")
    assert state["did_not_converge"] is True
    assert state["step_count"] == graph.MAX_STEPS  # capped, never exceeded
    assert state["final_answer"].startswith("(step cap reached)")
    assert state["done"] is True


@pytest.mark.live
def test_live_calculator_loop():
    """The real DeepSeek planner picks the calculator for an arithmetic question.
    Needs DEEPSEEK_API_KEY; excluded from the default run."""
    state = run_agent("Use a tool to compute 137 * 89, then tell me the result.")
    assert "calculator" in {o["tool"] for o in state["observations"]}
    # Strip thousands separators -- the model may render 12193 as "12,193".
    assert "12193" in state["final_answer"].replace(",", "")


@pytest.mark.live
def test_live_web_search_loop():
    """The real DeepSeek planner picks web_search and the loop completes.
    Needs DEEPSEEK_API_KEY + TAVILY_API_KEY; excluded from the default run.
    Asserts the trajectory (tool was used), not a time-sensitive answer."""
    state = run_agent("Use web search to find out who won the 2022 FIFA World Cup.")
    assert "web_search" in {o["tool"] for o in state["observations"]}
    assert state["final_answer"]


@pytest.mark.live
def test_live_doc_lookup_loop():
    """The real DeepSeek planner picks doc_lookup and reads P2's Chroma corpus.
    Needs DEEPSEEK_API_KEY + GEMINI_API_KEY + the rag-service chroma_store;
    excluded from the default run. Asserts the trajectory and that real snippets
    (not the empty sentinel) came back from the store."""
    state = run_agent(
        "Use the doc_lookup tool to find what the documents say about the project "
        "roadmap, then summarise what you find."
    )
    doc_obs = [o for o in state["observations"] if o["tool"] == "doc_lookup"]
    assert doc_obs and doc_obs[0]["output"] != "(no relevant documents)"
    assert state["final_answer"]


@pytest.mark.docker
@pytest.mark.live
def test_live_code_exec_loop():
    """The real planner writes Python and runs it in the Docker sandbox.
    Needs DEEPSEEK_API_KEY + a running Docker daemon + the built sandbox image;
    excluded from the default run. Run with `pytest -m "docker and live"`."""
    state = run_agent(
        "Use the code_exec tool to run Python that computes the sum of the "
        "integers from 1 to 100 and prints it."
    )
    assert "code_exec" in {o["tool"] for o in state["observations"]}
    assert "5050" in state["final_answer"]
