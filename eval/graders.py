"""The three eval graders (CLAUDE.md §8).

  - gold        programmatic answer check (exact / numeric / substring / regex);
                expected to be the majority of tasks.
  - open        LLM judge -- Gemini, a different model family from the DeepSeek
                agent -- a binary rubric; for the minority open-ended tasks only.
  - trajectory  did the task's required tool(s) actually appear in the trace?
                Runs alongside the answer check to catch right-answer-by-
                hallucination.

Graders consume a validated ``Task`` and the agent's final state; they do no I/O
except the judge's single Gemini call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from agent.config import settings


class Task(BaseModel):
    """One eval task as loaded from tasks.yaml. Validating here turns a malformed
    task into a clear error instead of a confusing grader failure. The schema is
    documented in eval/tasks.yaml."""

    id: str
    question: str
    type: Literal["gold", "open", "trajectory"]
    success: dict[str, Any] = {}
    required_tool: list[str] = []

    @field_validator("required_tool", mode="before")
    @classmethod
    def _as_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        return [v] if isinstance(v, str) else list(v)


@dataclass
class GradeResult:
    answer_ok: bool
    trajectory_ok: bool
    detail: str


# --- gold -------------------------------------------------------------------


def grade_gold(task: Task, answer: str | None) -> tuple[bool, str]:
    """Check the answer against an expected value. ``mode`` is one of
    exact | numeric | substring | regex, declared in the task's success block."""
    s = task.success
    mode = s.get("mode", "exact")
    expected = str(s.get("expected", ""))
    ans = answer or ""
    if mode == "exact":
        ok = ans.strip() == expected.strip()
    elif mode == "substring":
        ok = expected in ans
    elif mode == "regex":
        ok = re.search(expected, ans) is not None
    elif mode == "numeric":
        ok = _numeric_match(ans, expected, float(s.get("tolerance", 0.0)))
    else:
        raise ValueError(f"unknown gold match mode: {mode!r}")
    return ok, f"gold[{mode}] expected={expected!r} ok={ok}"


def _numeric_match(answer: str, expected: str, tolerance: float) -> bool:
    """Compare the answer's final number against expected within tolerance.

    The agent states its result at the END, after narrating its work and echoing
    inputs: "...opened in 1889 and 1931, a difference of 42." Reading the FIRST
    number grabs those distractors (1889) and scores a correct answer wrong, so
    take the LAST number -- the concluding value. (Thousands separators are
    stripped; the model often adds them -- see memory.)

    We deliberately do NOT pick the number closest to ``expected``: that would
    pass any answer merely containing the value, i.e. grade to the answer key.
    Last-number still fails an answer that concludes wrongly."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", answer)
    if not nums:
        return False
    got = float(nums[-1].replace(",", ""))
    return abs(got - float(expected)) <= tolerance


# --- open (LLM judge) -------------------------------------------------------


def grade_open(task: Task, answer: str | None) -> tuple[bool, str]:
    """Binary LLM-judge check via Gemini (a different family from the agent). The
    rubric is a yes/no question authored per task. ``_judge`` is monkeypatched in
    offline tests."""
    rubric = task.success.get("rubric")
    if not rubric:
        raise ValueError(f"open task {task.id!r} has no success.rubric")
    try:
        verdict = _judge(task.question, answer or "", rubric)
    except Exception as e:
        # A judge outage (e.g. a 429 quota error) must not abort the whole suite
        # and discard every completed run. Degrade to a clear, greppable non-pass
        # so the report still prints (CLAUDE.md §8: the eval must be robust).
        return False, f"open judge UNAVAILABLE ({type(e).__name__}): {str(e)[:120]}"
    return verdict, f"open judge verdict={verdict} rubric={rubric!r}"


def _judge(question: str, answer: str, rubric: str) -> bool:
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key, transport="rest")
    model = genai.GenerativeModel(settings.judge_model)
    prompt = (
        "You are grading an AI assistant's answer against a strict binary rubric.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Rubric: {rubric}\n"
        "Reply with exactly one word: yes (rubric satisfied) or no (not satisfied)."
    )
    resp = model.generate_content(prompt)
    return resp.text.strip().lower().startswith("y")


# --- trajectory + dispatch --------------------------------------------------


def trajectory_ok(task: Task, tools_used: set[str]) -> bool:
    """Did every required tool actually run? No required tool -> nothing to check
    (vacuously true)."""
    return all(t in tools_used for t in task.required_tool)


def grade(task: Task, state: dict[str, Any]) -> GradeResult:
    """Grade one agent run: the answer (by task type) plus the trajectory check
    that always runs alongside it (CLAUDE.md §8)."""
    tools_used = {o["tool"] for o in state.get("observations", [])}
    traj = trajectory_ok(task, tools_used)
    if task.type == "gold":
        ans_ok, detail = grade_gold(task, state.get("final_answer"))
    elif task.type == "open":
        ans_ok, detail = grade_open(task, state.get("final_answer"))
    else:  # trajectory: using the required tool *is* the success criterion
        ans_ok, detail = traj, "trajectory-type: required tool present"
    return GradeResult(answer_ok=ans_ok, trajectory_ok=traj, detail=detail)
