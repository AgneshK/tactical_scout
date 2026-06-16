import json
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage
from agent import tactical_agent

log = logging.getLogger(__name__)

app = FastAPI(title="Tactical Scout API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
async def health():
    """Lightweight readiness probe — lets the frontend wake a cold instance and detect
    when the model artifacts are loaded, without paying for a full agent turn."""
    from agent import _name_to_idx
    ready = _name_to_idx is not None
    return {
        "status": "ok" if ready else "degraded",
        "artifacts_loaded": ready,
        "players_indexed": len(_name_to_idx) if ready else 0,
    }


def _extract_similarity_payload(messages) -> dict | None:
    """Find the most recent get_similar_players ToolMessage and parse its JSON payload."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "get_similar_players":
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return None
            return data if isinstance(data, dict) and "clones" in data else None
    return None


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    t0 = time.perf_counter()
    try:
        inputs   = {"messages": [HumanMessage(content=request.message)]}
        response = await tactical_agent.ainvoke(inputs)
    except Exception as exc:
        log.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="The agent encountered an error. Please try again.")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info("elapsed=%.0fms  turns=%d", elapsed_ms, len(response["messages"]))

    return {
        "reply":   response["messages"][-1].content,
        "players": _extract_similarity_payload(response["messages"]),
    }
