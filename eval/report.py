"""Eval report: the headline metrics (CLAUDE.md §8).

  - completion rate           % of runs whose answer passed
  - avg steps-to-convergence  mean step_count
  - per-strategy delta         the headline -- strategies compared head to head

Plus two diagnostics: trajectory-hit rate and the count of right-answer-by-
hallucination runs (answer correct, required tool never used). Pure functions
over the runner's records, so they unit-test offline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .runner import RunRecord


@dataclass
class StrategyStats:
    strategy: str
    runs: int
    completion_rate: float
    avg_steps: float
    trajectory_hit_rate: float
    hallucinations: int  # answer correct but the required tool was never used
    non_convergence_rate: float


@dataclass
class TaskStats:
    task_id: str
    runs: int
    completion_rate: float
    avg_steps: float
    trajectory_hit_rate: float
    non_convergence_rate: float


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(records: list[RunRecord]) -> list[StrategyStats]:
    by_strategy: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_strategy[r.strategy].append(r)
    return [
        StrategyStats(
            strategy=name,
            runs=len(rs),
            completion_rate=_mean([r.answer_ok for r in rs]),
            avg_steps=_mean([r.step_count for r in rs]),
            trajectory_hit_rate=_mean([r.trajectory_ok for r in rs]),
            hallucinations=sum(1 for r in rs if r.answer_ok and not r.trajectory_ok),
            non_convergence_rate=_mean([r.did_not_converge for r in rs]),
        )
        for name, rs in by_strategy.items()
    ]


def summarize_by_task(records: list[RunRecord]) -> list[TaskStats]:
    """Per-task aggregate (across all strategies). Answers "is low completion a
    few hard tasks or a systematic step-cap issue?" -- the strategy table alone
    cannot tell those apart."""
    by_task: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_task[r.task_id].append(r)
    return [
        TaskStats(
            task_id=name,
            runs=len(rs),
            completion_rate=_mean([r.answer_ok for r in rs]),
            avg_steps=_mean([r.step_count for r in rs]),
            trajectory_hit_rate=_mean([r.trajectory_ok for r in rs]),
            non_convergence_rate=_mean([r.did_not_converge for r in rs]),
        )
        for name, rs in by_task.items()
    ]


def format_task_breakdown(records: list[RunRecord]) -> str:
    # Worst completion first, so the tasks dragging the average down surface at
    # the top; non-convergence ties break it for equal completion.
    stats = sorted(
        summarize_by_task(records),
        key=lambda s: (s.completion_rate, -s.non_convergence_rate),
    )
    header = (
        f"{'task':<34}{'runs':>6}{'complete':>10}{'avg_steps':>11}"
        f"{'traj_hit':>10}{'no_conv':>9}"
    )
    lines = ["=== per-task breakdown (both strategies) ===", "", header, "-" * len(header)]
    for s in stats:
        lines.append(
            f"{s.task_id:<34}{s.runs:>6}{s.completion_rate:>9.0%}{s.avg_steps:>11.2f}"
            f"{s.trajectory_hit_rate:>9.0%}{s.non_convergence_rate:>8.0%}"
        )
    lines.append("")
    return "\n".join(lines)


def format_report(records: list[RunRecord]) -> str:
    stats = summarize(records)
    header = (
        f"{'strategy':<14}{'runs':>6}{'complete':>10}{'avg_steps':>11}"
        f"{'traj_hit':>10}{'halluc':>8}{'no_conv':>9}"
    )
    lines = ["", "=== P3 agent eval ===", "", header, "-" * len(header)]
    for s in stats:
        lines.append(
            f"{s.strategy:<14}{s.runs:>6}{s.completion_rate:>9.0%}{s.avg_steps:>11.2f}"
            f"{s.trajectory_hit_rate:>9.0%}{s.hallucinations:>8}{s.non_convergence_rate:>8.0%}"
        )
    if len(stats) >= 2:
        best = max(stats, key=lambda s: s.completion_rate)
        worst = min(stats, key=lambda s: s.completion_rate)
        delta = best.completion_rate - worst.completion_rate
        lines += [
            "",
            f"strategy delta (completion): {best.strategy} beats {worst.strategy} "
            f"by {delta:.0%}",
        ]
    lines.append("")
    return "\n".join(lines)
