"""The P3 eval harness: tasks, graders, runner, report (CLAUDE.md §8).

A headless client of the agent library -- it imports the graph directly and runs
it in-process. Task content is owner-provided (see tasks.yaml); this package is
only the machinery.
"""
