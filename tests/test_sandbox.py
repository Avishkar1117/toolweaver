"""Sandbox isolation tests. These need a running Docker daemon and the built
image:

    docker build -f docker/sandbox.Dockerfile -t agent-sandbox:latest docker

They're excluded from the default run via the `docker` marker; run them with
`pytest -m docker`.
"""

import pytest

from agent import config
from agent.sandbox import run_code

pytestmark = pytest.mark.docker


def test_runs_simple_code():
    result = run_code("print(2 + 2)")
    assert result.exit_code == 0
    assert result.stdout.strip() == "4"
    assert not result.timed_out


def test_network_is_blocked():
    """The defining property of the sandbox: no ambient network access (§5)."""
    code = (
        "import urllib.request\n"
        "urllib.request.urlopen('http://example.com', timeout=5)\n"
        "print('REACHED')\n"
    )
    result = run_code(code)
    assert "REACHED" not in result.stdout  # the network call must not succeed
    assert result.exit_code != 0


def test_wall_clock_timeout(monkeypatch):
    """An infinite loop is killed by the wall-clock guard, not left running."""
    monkeypatch.setattr(config.settings, "sandbox_timeout", 3)
    result = run_code("while True:\n    pass\n")
    assert result.timed_out
