"""The agent's tools.

All four sit behind one interface (the ``Tool`` protocol) and validate their
args against a Pydantic schema before doing any work. Tools are promoted from
stubs to real implementations one at a time (CLAUDE.md §11); all four are real.
``doc_lookup`` is an MCP client (server in ``mcp_server.py``); ``code_exec`` runs
in the Docker sandbox (``sandbox.py``).
"""

import ast
import operator as _op
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .config import settings


class ToolResult(BaseModel):
    """Uniform return type for every tool."""

    ok: bool
    output: str
    error: str | None = None


class Tool(Protocol):
    """The one interface every tool implements."""

    name: str
    description: str
    args_schema: type[BaseModel]

    def run(self, **kwargs: Any) -> ToolResult: ...


# --- arg schemas (validated before each tool runs) -------------------------


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="search query")


class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="arithmetic expression to evaluate")


class DocLookupArgs(BaseModel):
    query: str = Field(..., description="question to retrieve documents for")


class CodeExecArgs(BaseModel):
    code: str = Field(..., description="python source to run in the sandbox")


# --- tools -----------------------------------------------------------------


_tavily_client: Any = None


def _get_tavily() -> Any:
    """Lazily build the Tavily client. Lazy so importing this module needs no
    key; offline tests monkeypatch this to avoid the network."""
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient

        _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    return _tavily_client


class WebSearchTool:
    name = "web_search"
    description = "Search the web for up-to-date or external information."
    args_schema = WebSearchArgs

    def run(self, **kwargs: Any) -> ToolResult:
        args = self.args_schema(**kwargs)
        resp = _get_tavily().search(query=args.query, max_results=settings.web_search_max_results)
        results = resp.get("results", [])
        if not results:
            return ToolResult(ok=True, output="(no results)")
        snippets = [
            f"{r.get('title', '')} ({r.get('url', '')})\n{r.get('content', '')}" for r in results
        ]
        return ToolResult(ok=True, output="\n\n".join(snippets))


# Safe arithmetic: walk the AST and evaluate only numbers, the binary operators
# + - * / % **, and unary +/-. No names, calls, or attribute access, so the
# calculator can't be turned into an arbitrary-code path. (Untrusted *code*
# execution is code_exec's job; see CLAUDE.md §5.)
_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
}


def _safe_eval(expr: str) -> float:
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return ev(ast.parse(expr, mode="eval").body)


class CalculatorTool:
    name = "calculator"
    description = "Evaluate a basic arithmetic expression (+, -, *, /, %, **)."
    args_schema = CalculatorArgs

    def run(self, **kwargs: Any) -> ToolResult:
        args = self.args_schema(**kwargs)
        return ToolResult(ok=True, output=str(_safe_eval(args.expression)))


def _mcp_doc_lookup(query: str) -> str:
    """Call the ``doc_lookup`` tool on the MCP server over stdio. Spawns the
    server per call (CLAUDE.md §4 -- doc_lookup is the one MCP-exposed tool).
    Offline tests monkeypatch this."""
    import asyncio

    return asyncio.run(_mcp_doc_lookup_async(query))


async def _mcp_doc_lookup_async(query: str) -> str:
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Pass our full environment to the server subprocess. MCP's default is a
    # scrubbed allowlist; in the Docker image the API keys AND PYTHONPATH arrive
    # as env vars (no editable install, no .env on disk), so without this the
    # server can neither import `agent` nor reach Gemini. Locally a .env file also
    # suffices -- this is what makes doc_lookup portable into the container.
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "agent.mcp_server"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("doc_lookup", {"query": query})
            parts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
            return "\n\n".join(parts)


class DocLookupTool:
    """doc_lookup goes over MCP: this tool is a thin MCP *client*; the retrieval
    lives in the MCP server (mcp_server.py). Its planner-facing schema is
    identical to the other tools -- only the execution path differs."""

    name = "doc_lookup"
    description = "Retrieve relevant snippets from the local document corpus."
    args_schema = DocLookupArgs

    def run(self, **kwargs: Any) -> ToolResult:
        args = self.args_schema(**kwargs)
        return ToolResult(ok=True, output=_mcp_doc_lookup(args.query))


class CodeExecTool:
    name = "code_exec"
    description = "Run a short stdlib-only Python snippet in an isolated sandbox; returns stdout."
    args_schema = CodeExecArgs

    def run(self, **kwargs: Any) -> ToolResult:
        args = self.args_schema(**kwargs)
        if not settings.sandbox_enabled:
            # Clean gate for managed hosting with no Docker daemon (CLAUDE.md §5):
            # a deliberate message the agent relays, not a raw sandbox error.
            return ToolResult(
                ok=False,
                output="",
                error=(
                    "code_exec is disabled on this hosted demo -- sandboxed code "
                    "execution runs only in the self-hosted/local deployment. "
                    "Answer the question without running code."
                ),
            )
        from . import sandbox  # lazy: importing tools needs no Docker

        result = sandbox.run_code(args.code)
        if result.timed_out:
            return ToolResult(ok=False, output=result.stdout, error="execution timed out")
        if result.exit_code != 0:
            err = result.stderr.strip() or f"exited with code {result.exit_code}"
            return ToolResult(ok=False, output=result.stdout, error=err)
        return ToolResult(ok=True, output=result.stdout.strip())


# --- registry --------------------------------------------------------------

TOOLS: dict[str, Tool] = {
    t.name: t for t in (WebSearchTool(), CalculatorTool(), DocLookupTool(), CodeExecTool())
}


def openai_tool_schemas() -> list[dict[str, Any]]:
    """The tools as OpenAI function-calling schemas, for binding to the planner LLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.model_json_schema(),
            },
        }
        for t in TOOLS.values()
    ]
