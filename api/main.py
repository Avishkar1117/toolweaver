"""Thin FastAPI wrapper over the agent core. One endpoint: POST /ask."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.graph import run_agent

app = FastAPI(title="agentic-tool-use")

_INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """A small self-contained demo page (HF embeds the app root in the Space
    iframe; without this, visitors hit FastAPI's bare 404). The raw API stays at
    /ask and /docs."""
    return FileResponse(_INDEX_HTML)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    trace: dict[str, Any]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    state = run_agent(req.question)
    trace = {
        "messages": state["messages"],
        "observations": state["observations"],
        "scratchpad": state["scratchpad"],
        "step_count": state["step_count"],
        "last_tool_called": state["last_tool_called"],
        "did_not_converge": state["did_not_converge"],
    }
    return AskResponse(answer=state["final_answer"] or "", trace=trace)
