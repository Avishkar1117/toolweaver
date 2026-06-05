"""MCP server exposing ``doc_lookup`` over the protocol (CLAUDE.md §4).

This is the *one* tool we expose via MCP: a data/context source (P2's existing
corpus) is the realistic MCP use case. The server owns the Chroma + Gemini
access; the agent consumes it as a client (see ``DocLookupTool`` in tools.py).
The other tools stay as direct function calls -- we deliberately don't MCP
everything.

Two MCP primitives are exposed:
  - tool     ``doc_lookup(query)``  -- retrieve snippets (does the real work)
  - resource ``corpus://info``      -- read-only metadata about the corpus

Run as a stdio server: ``python -m agent.mcp_server``.
"""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import settings

mcp = FastMCP("doc-lookup")

# Total chars of retrieved text to hand back per lookup. This corpus is chunked
# by whole PDF page (~3k chars each), so a *per-chunk* head-cap would truncate
# answers that sit mid-page; a single total budget keeps context bounded (§13)
# while letting the most-relevant page come through whole.
_SNIPPET_BUDGET = 6000

_collection: Any = None


def _get_collection() -> Any:
    """Open P2's Chroma collection (cached, telemetry OFF). Telemetry logs to
    stdout, which here *is* the JSON-RPC channel -- a stray stdout write would
    corrupt the protocol and hang the client. Offline tests monkeypatch this."""
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.PersistentClient(
            path=settings.doc_chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(settings.doc_collection)
    return _collection


def _embed_query(text: str) -> list[float]:
    """Embed a query with the same Gemini model the corpus was built with, over
    REST. gRPC (the embedding default) deadlocks against the MCP server's asyncio
    Proactor loop on Windows -- both use IOCP -- so we go through llama-index for
    nothing here and call Gemini's REST endpoint directly. Offline tests
    monkeypatch this."""
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key, transport="rest")
    resp = genai.embed_content(
        model=settings.embedding_model, content=text, task_type="retrieval_query"
    )
    return resp["embedding"]


def _doc_lookup_impl(query: str) -> str:
    """Embed the query (REST) and pull the nearest chunks straight from Chroma --
    same store, same model, same vector space as P2; only the transport differs.
    Separated from the MCP registration so it's unit-testable in-process."""
    results = _get_collection().query(
        query_embeddings=[_embed_query(query)], n_results=settings.doc_top_k
    )
    docs = (results.get("documents") or [[]])[0]
    if not docs:
        return "(no relevant documents)"
    # Return chunks most-relevant-first under one total budget. Don't head-cap each
    # chunk: pages here run ~3k chars and the answer can sit deep in one (e.g. the
    # Adam optimizer fact ~2400 chars into its page), which a small per-chunk cap
    # would silently drop.
    budget, snippets = _SNIPPET_BUDGET, []
    for d in docs:
        d = d.strip()[:budget]
        snippets.append(d)
        budget -= len(d)
        if budget <= 0:
            break
    return "\n\n".join(snippets)


@mcp.tool()
async def doc_lookup(query: str) -> str:
    """Retrieve relevant snippets from the local document corpus."""
    import anyio

    # Offload the blocking retrieval (network embed + Chroma) to a worker thread
    # so it doesn't run on the server's event-loop thread. On the loop thread it
    # stalls ~60s against the active stdio I/O; in a thread it's ~2s.
    return await anyio.to_thread.run_sync(_doc_lookup_impl, query)


@mcp.resource("corpus://info")
def corpus_info() -> str:
    """Read-only metadata about the document corpus the agent can query."""
    collection = _get_collection()
    return json.dumps(
        {
            "collection": settings.doc_collection,
            "chunks": collection.count(),
            "embedding_model": settings.embedding_model,
            "top_k": settings.doc_top_k,
        }
    )


def _warm() -> None:
    """Pay the heavy import + collection open at startup, while no event loop is
    yet servicing the stdio pipes. Importing google/grpc lazily inside the first
    request -- under the server's active Proactor loop on Windows -- stalls ~90s;
    at startup it's <1s. After this, module imports are cached process-wide, so
    the first real request runs in a couple of seconds."""
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key, transport="rest")
    _get_collection()


if __name__ == "__main__":
    _warm()
    mcp.run()
