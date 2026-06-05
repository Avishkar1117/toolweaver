"""Runs the eval harness in-process (CLAUDE.md §8).

Imports the agent graph directly -- no HTTP -- runs each task N times (the agent
is stochastic) under each prompting strategy, grades every run, and hands the
records to report.py. ``python -m eval.runner`` runs the full set; the CI smoke
calls these same functions with a fake planner (see tests/test_eval.py).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent.graph import run_agent
from agent.prompts import STRATEGIES

from .graders import Task, grade


@dataclass
class RunRecord:
    task_id: str
    strategy: str
    answer_ok: bool
    trajectory_ok: bool
    step_count: int
    did_not_converge: bool
    final_answer: str
    detail: str
    trace: dict[str, Any] = field(repr=False)  # full reasoning trace, kept out of repr


def load_tasks(path: str | Path) -> list[Task]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [Task(**t) for t in raw]


def run_task(task: Task, strategy: str, planner_system: str, n: int) -> list[RunRecord]:
    """Run one task ``n`` times under one strategy, grading each run."""
    records = []
    for _ in range(n):
        state = run_agent(task.question, planner_system=planner_system)
        result = grade(task, state)
        records.append(
            RunRecord(
                task_id=task.id,
                strategy=strategy,
                answer_ok=result.answer_ok,
                trajectory_ok=result.trajectory_ok,
                step_count=state["step_count"],
                did_not_converge=state["did_not_converge"],
                final_answer=state["final_answer"] or "",
                detail=result.detail,
                trace={"messages": state["messages"], "observations": state["observations"]},
            )
        )
    return records


def run_suite(
    tasks: list[Task], strategies: dict[str, str] | None = None, n: int = 3
) -> list[RunRecord]:
    """Run every task under every strategy, ``n`` runs each."""
    strategies = strategies or STRATEGIES
    records: list[RunRecord] = []
    for name, prompt in strategies.items():
        for task in tasks:
            records.extend(run_task(task, name, prompt, n))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the P3 agent eval (full set; not for CI).")
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks.yaml"))
    ap.add_argument("--runs", type=int, default=3, help="runs per task per strategy")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    if not tasks:
        raise SystemExit(f"no tasks in {args.tasks} -- the owner fills tasks.yaml (CLAUDE.md §8)")

    from .report import format_report

    records = run_suite(tasks, n=args.runs)
    print(format_report(records))


if __name__ == "__main__":
    main()
