"""Eval-harness tests (all offline).

The graders are pure functions. The runner+report smoke drives the whole harness
with a fake planner -- no keys, no network -- so it is the CI PR-gate smoke
subset (CLAUDE.md §8: the full eval over the owner's tasks is run manually).
"""

from agent import graph
from eval import graders, report, runner
from eval.graders import Task, grade_gold, grade_open, trajectory_ok

# --- gold grader ------------------------------------------------------------


def test_gold_exact():
    t = Task(id="x", question="q", type="gold", success={"mode": "exact", "expected": "Paris"})
    assert grade_gold(t, "Paris")[0]
    assert not grade_gold(t, "Paris, France")[0]


def test_gold_substring():
    t = Task(id="x", question="q", type="gold", success={"mode": "substring", "expected": "Paris"})
    assert grade_gold(t, "The capital is Paris.")[0]
    assert not grade_gold(t, "The capital is London.")[0]


def test_gold_regex():
    t = Task(id="x", question="q", type="gold", success={"mode": "regex", "expected": r"\b42\b"})
    assert grade_gold(t, "the answer is 42 today")[0]
    assert not grade_gold(t, "the answer is 424")[0]


def test_gold_numeric_tolerance_and_separators():
    t = Task(
        id="x",
        question="q",
        type="gold",
        success={"mode": "numeric", "expected": 12193, "tolerance": 0},
    )
    assert grade_gold(t, "It is 12,193.")[0]  # thousands separator stripped (see memory)
    assert grade_gold(t, "12193")[0]
    t2 = Task(
        id="x",
        question="q",
        type="gold",
        success={"mode": "numeric", "expected": 2.5, "tolerance": 0.01},
    )
    assert grade_gold(t2, "about 2.504")[0]
    assert not grade_gold(t2, "about 2.6")[0]


# --- trajectory grader ------------------------------------------------------


def test_trajectory_hit_and_miss():
    t = Task(id="x", question="q", type="gold", required_tool="calculator", success={})
    assert t.required_tool == ["calculator"]  # str normalised to list
    assert trajectory_ok(t, {"calculator", "web_search"})
    assert not trajectory_ok(t, {"web_search"})


def test_trajectory_none_required_is_vacuously_true():
    t = Task(id="x", question="q", type="gold", success={})
    assert trajectory_ok(t, set())


# --- open grader (judge mocked, so it stays offline) ------------------------


def test_open_grader_relays_judge(monkeypatch):
    t = Task(id="x", question="q", type="open", success={"rubric": "ok?"})
    monkeypatch.setattr(graders, "_judge", lambda q, a, r: True)
    assert grade_open(t, "anything")[0]
    monkeypatch.setattr(graders, "_judge", lambda q, a, r: False)
    assert not grade_open(t, "anything")[0]


# --- runner + report smoke (fake planner) -----------------------------------


class _FakeMsg:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


class _CalcThenAnswer:
    """Calls the real calculator until an observation exists, then answers.
    Decides from the prompt (not a call counter), so it is correct across the N
    independent runs the runner makes."""

    def invoke(self, messages):
        human = messages[-1].content
        if "Observations so far" in human:
            return _FakeMsg(content="The answer is 42.")
        return _FakeMsg(
            tool_calls=[{"name": "calculator", "args": {"expression": "40 + 2"}, "id": "1"}]
        )


def test_runner_and_report_smoke(monkeypatch):
    monkeypatch.setattr(graph, "get_planner", lambda: _CalcThenAnswer())
    tasks = [
        Task(
            id="t1",
            question="What is 40 + 2?",
            type="gold",
            required_tool=["calculator"],
            success={"mode": "numeric", "expected": 42, "tolerance": 0},
        )
    ]
    records = runner.run_suite(tasks, strategies={"baseline": "sys"}, n=2)

    assert len(records) == 2  # one task x one strategy x two runs
    assert all(r.answer_ok and r.trajectory_ok for r in records)
    assert all(r.step_count == 1 for r in records)

    stats = report.summarize(records)
    assert len(stats) == 1
    assert stats[0].completion_rate == 1.0
    assert stats[0].trajectory_hit_rate == 1.0
    assert stats[0].hallucinations == 0

    text = report.format_report(records)
    assert "baseline" in text and "complete" in text
