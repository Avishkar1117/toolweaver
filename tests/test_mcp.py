"""MCP contract test: a real client<->server roundtrip over stdio.

The server does the real Chroma retrieval, so this needs GEMINI_API_KEY + the
rag-service chroma_store. Marked `live`; excluded from the default run. Async
tests run under pytest-asyncio's auto mode (see pyproject).
"""

import json
import os
import sys

import pytest

pytestmark = pytest.mark.live


async def test_mcp_tool_and_resource_roundtrip():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "agent.mcp_server"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Contract: the server advertises the doc_lookup tool ...
            tools = await session.list_tools()
            assert "doc_lookup" in {t.name for t in tools.tools}

            # ... and the corpus resource (the second MCP primitive).
            resources = await session.list_resources()
            assert "corpus://info" in {str(r.uri) for r in resources.resources}

            # Calling the tool returns real snippets from P2's store.
            result = await session.call_tool("doc_lookup", {"query": "project roadmap"})
            text = "\n\n".join(
                c.text for c in result.content if getattr(c, "type", None) == "text"
            )
            assert text and text != "(no relevant documents)"

            # Reading the resource returns corpus metadata.
            res = await session.read_resource("corpus://info")
            meta = json.loads(res.contents[0].text)
            assert meta["collection"]
            assert meta["chunks"] > 0
