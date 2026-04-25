import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
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

    return {"reply": response["messages"][-1].content}
