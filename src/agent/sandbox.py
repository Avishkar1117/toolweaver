"""Run untrusted, model-generated code in a locked-down Docker container.

The container has no ambient access: no network (``--network none``), no host
filesystem mount (code is piped in via stdin), dropped capabilities, a non-root
user, a read-only root fs, and CPU / memory / pid / wall-clock limits. Any data
the code needs must be in the code itself -- the sandbox cannot reach the corpus,
the network, or the host (CLAUDE.md §5).

The hardening flags live here (not baked into the image) so the whole security
boundary is auditable in one place.
"""

import subprocess
import uuid
from dataclasses import dataclass

from .config import settings


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def run_code(code: str) -> SandboxResult:
    """Execute ``code`` in the sandbox container and capture its output."""
    name = f"agent-sbx-{uuid.uuid4().hex[:12]}"
    args = [
        "docker", "run", "--rm", "-i",
        "--name", name,
        "--network", "none",                  # no outbound network
        "--memory", settings.sandbox_memory,
        "--cpus", str(settings.sandbox_cpus),
        "--pids-limit", str(settings.sandbox_pids),
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",                        # immutable root fs
        "--tmpfs", "/tmp:size=16m",           # small writable scratch
        settings.sandbox_image,
    ]
    try:
        proc = subprocess.run(
            args,
            input=code,
            capture_output=True,
            text=True,
            timeout=settings.sandbox_timeout,
        )
        return SandboxResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        # Wall-clock guard fired. Killing the `docker run` client doesn't stop
        # the container, so force-remove it by name.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return SandboxResult(
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + "\n[sandbox] timed out",
            exit_code=124,
            timed_out=True,
        )
