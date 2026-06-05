"""Agent core: a plain importable library. Imports nothing web/server-related.

The FastAPI app and the eval harness are independent clients of this package.
"""

from .graph import State, build_graph, run_agent
from .tools import TOOLS, Tool, ToolResult

__all__ = ["State", "build_graph", "run_agent", "TOOLS", "Tool", "ToolResult"]
