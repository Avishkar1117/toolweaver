"""Thin FastAPI wrapper over the agent core. One endpoint: POST /ask."""

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from agent.graph import run_agent

app = FastAPI(title="agentic-tool-use")


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
