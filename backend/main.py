import json
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage
from agent import (
    tactical_agent,
    player_info,
    _parse_nation,
    _strip_country_prefix,
    _primary_pos,
)

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


# ── Dataset-explorer support (autocomplete, stats dashboard, leaderboards) ───────
# Season totals from the 2024/25 FBRef set powering the sidebar components.
_METRICS = {
    "xG":   "Expected Goals",
    "xAG":  "Expected Assists",
    "PrgC": "Progressive Carries",
    "PrgP": "Progressive Passes",
    "Tkl":  "Tackles",
    "KP":   "Key Passes",
}
_FLOAT_METRICS = {"xG", "xAG"}
_POS_GROUPS = {"FW", "MF", "DF"}


def _build_player_list() -> list[dict]:
    """Compact roster for client-side autocomplete (built once at startup)."""
    if player_info is None:
        return []
    out = []
    for _, row in player_info.iterrows():
        nat = _parse_nation(row.get("Nation", ""))
        out.append({
            "name":   row["Player"],
            "pos":    _primary_pos(str(row.get("Pos", ""))),
            "club":   row.get("Squad", "") or "",
            "league": _strip_country_prefix(row.get("Comp", "")),
            "flag":   nat["flag_url"],
            "nation": nat["label"],
        })
    return out


def _build_dataset_stats() -> dict:
    """Aggregate dataset shape for the engine dashboard (built once at startup)."""
    if player_info is None:
        return {}
    df = player_info
    leagues = df["Comp"].dropna().apply(_strip_country_prefix).value_counts()
    positions = df["Pos"].dropna().apply(_primary_pos).value_counts()
    return {
        "players":   int(len(df)),
        "clubs":     int(df["Squad"].nunique()),
        "leagues":   [{"name": k, "count": int(v)} for k, v in leagues.items()],
        "positions": [{"pos": k, "count": int(v)} for k, v in positions.items()],
        "minutes":   int(df["Min"].sum()),
        "avg_age":   round(float(df["Age"].mean()), 1),
        "latent_dim": 8,
        "metrics":   [{"key": k, "label": v} for k, v in _METRICS.items()],
    }


_PLAYER_LIST = _build_player_list()
_DATASET_STATS = _build_dataset_stats()
log.info("Explorer data ready — %d players, %d metrics", len(_PLAYER_LIST), len(_METRICS))


@app.get("/players")
async def players():
    """Full roster (name, primary position, club, nation) for autocomplete."""
    return {"count": len(_PLAYER_LIST), "players": _PLAYER_LIST}


@app.get("/stats")
async def stats():
    """Dataset overview powering the engine dashboard."""
    if not _DATASET_STATS:
        raise HTTPException(status_code=503, detail="Dataset unavailable — artifacts not loaded.")
    return _DATASET_STATS


@app.get("/leaders")
async def leaders(metric: str = "xG", pos: str | None = None, limit: int = 10):
    """Top performers by a season-total metric, optionally filtered by position group."""
    if player_info is None:
        raise HTTPException(status_code=503, detail="Dataset unavailable — artifacts not loaded.")
    if metric not in _METRICS:
        metric = "xG"
    limit = max(1, min(int(limit), 25))

    df = player_info
    pos = (pos or "").upper().strip()
    if pos in _POS_GROUPS:
        df = df[df["Pos"].apply(lambda s: _primary_pos(str(s)) == pos)]

    top = df.nlargest(limit, metric)
    leaders_out = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        nat = _parse_nation(row.get("Nation", ""))
        raw = row[metric]
        leaders_out.append({
            "rank":   rank,
            "name":   row["Player"],
            "club":   row.get("Squad", "") or "",
            "pos":    _primary_pos(str(row.get("Pos", ""))),
            "flag":   nat["flag_url"],
            "nation": nat["label"],
            "value":  round(float(raw), 1) if metric in _FLOAT_METRICS else int(raw),
        })
    return {
        "metric":       metric,
        "metric_label": _METRICS[metric],
        "pos":          pos or None,
        "leaders":      leaders_out,
    }


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
