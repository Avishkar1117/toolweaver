"""Per-tool unit tests. Phase 2 promotes tools from stubs to real ones one at a
time; this file grows a section per tool as each becomes real."""

import pytest

from agent import mcp_server, sandbox, tools
from agent.tools import CalculatorTool, CodeExecTool, DocLookupTool, WebSearchTool

# --- calculator ------------------------------------------------------------


def test_calculator_basic():
    assert CalculatorTool().run(expression="40 + 2").output == "42"


def test_calculator_precedence_and_parens():
    assert CalculatorTool().run(expression="2 + 3 * 4").output == "14"
    assert CalculatorTool().run(expression="(2 + 3) * 4").output == "20"


def test_calculator_division_is_float():
    assert CalculatorTool().run(expression="10 / 4").output == "2.5"


def test_calculator_rejects_non_arithmetic():
    with pytest.raises(ValueError):
        CalculatorTool().run(expression="__import__('os').system('echo hi')")


# --- web_search (Tavily client mocked so the default run stays offline) -----


class _FakeTavily:
    def __init__(self, results):
        self._results = results

    def search(self, query, max_results):
        return {"results": self._results}


def test_web_search_formats_results(monkeypatch):
    fake = _FakeTavily(
        [
            {"title": "Paris", "url": "https://x/paris", "content": "Capital of France."},
            {"title": "France", "url": "https://x/france", "content": "A country."},
        ]
    )
    monkeypatch.setattr(tools, "_get_tavily", lambda: fake)
    result = WebSearchTool().run(query="capital of France")
    assert result.ok
    assert "Paris" in result.output
    assert "https://x/paris" in result.output
    assert "Capital of France." in result.output


def test_web_search_handles_no_results(monkeypatch):
    monkeypatch.setattr(tools, "_get_tavily", lambda: _FakeTavily([]))
    assert WebSearchTool().run(query="zzz").output == "(no results)"


# --- doc_lookup --------------------------------------------------------------
# doc_lookup goes over MCP: the retrieval+formatting lives in the MCP server
# (tested in-process here with a mocked retriever) and the tool is a thin client
# (tested with the MCP call mocked). The live client<->server roundtrip is in
# tests/test_mcp.py.


class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def query(self, query_embeddings, n_results):
        return {"documents": [self._docs]}


def test_doc_lookup_server_formats_snippets(monkeypatch):
    monkeypatch.setattr(mcp_server, "_embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(
        mcp_server, "_get_collection", lambda: _FakeCollection(["Alpha chunk about X.", "Beta."])
    )
    output = mcp_server._doc_lookup_impl("what is X?")
    assert "Alpha chunk about X." in output
    assert "Beta." in output


def test_doc_lookup_server_handles_no_hits(monkeypatch):
    monkeypatch.setattr(mcp_server, "_embed_query", lambda q: [0.1])
    monkeypatch.setattr(mcp_server, "_get_collection", lambda: _FakeCollection([]))
    assert mcp_server._doc_lookup_impl("zzz") == "(no relevant documents)"


def test_doc_lookup_tool_relays_mcp_result(monkeypatch):
    # The tool is a thin client: it relays whatever the MCP server returns.
    monkeypatch.setattr(tools, "_mcp_doc_lookup", lambda q: "snippet from MCP")
    result = DocLookupTool().run(query="anything")
    assert result.ok
    assert result.output == "snippet from MCP"


# --- code_exec (sandbox mocked so the default run needs no Docker) -----------


def _result(**kw):
    base = {"stdout": "", "stderr": "", "exit_code": 0, "timed_out": False}
    return sandbox.SandboxResult(**{**base, **kw})


def test_code_exec_returns_stdout(monkeypatch):
    monkeypatch.setattr(sandbox, "run_code", lambda code: _result(stdout="42\n"))
    result = CodeExecTool().run(code="print(40 + 2)")
    assert result.ok
    assert result.output == "42"


def test_code_exec_reports_error(monkeypatch):
    fail = lambda code: _result(stderr="NameError: x", exit_code=1)  # noqa: E731
    monkeypatch.setattr(sandbox, "run_code", fail)
    result = CodeExecTool().run(code="print(x)")
    assert not result.ok
    assert "NameError" in result.error


def test_code_exec_reports_timeout(monkeypatch):
    monkeypatch.setattr(sandbox, "run_code", lambda code: _result(exit_code=124, timed_out=True))
    result = CodeExecTool().run(code="while True: pass")
    assert not result.ok
    assert "timed out" in result.error


def test_code_exec_gated_when_sandbox_disabled(monkeypatch):
    # On managed hosting (sandbox_enabled=False) code_exec returns a clean message
    # without ever touching the sandbox -- no raw Docker error in the trace.
    monkeypatch.setattr(tools.settings, "sandbox_enabled", False)
    called = False

    def boom(code):  # the sandbox must NOT be invoked when gated
        nonlocal called
        called = True
        return _result()

    monkeypatch.setattr(sandbox, "run_code", boom)
    result = CodeExecTool().run(code="print(1)")
    assert not result.ok
    assert "self-hosted" in result.error
    assert called is False
