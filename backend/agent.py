import logging
import time
import torch
import torch.nn.functional as F
import joblib
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Artifact loading ───────────────────────────────────────────────────────────

def _load_artifacts():
    player_info = joblib.load("artifacts/player_info.pkl")
    embeddings  = torch.load("artifacts/embeddings.pt", weights_only=True)
    name_to_idx = {name.lower(): idx for idx, name in enumerate(player_info["Player"].tolist())}
    log.info("Artifacts loaded — %d players in index", len(name_to_idx))
    return player_info, embeddings, name_to_idx

try:
    player_info, embeddings, _name_to_idx = _load_artifacts()
except FileNotFoundError:
    log.critical("Artifacts missing — run autoencoder.py first.")
    player_info = embeddings = _name_to_idx = None


# ── Position helpers ───────────────────────────────────────────────────────────

def _primary_pos(pos_str: str) -> str:
    """First position listed in the FBRef string is the player's primary role."""
    return pos_str.split(",")[0].strip() if pos_str else ""

def _pos_compatible(source_pos: str, candidate_pos: str) -> bool:
    """
    True when the source's primary position appears anywhere in the candidate's
    position string.

    Examples:
      source "FW"    vs candidate "FW,MF" → "FW" in ["FW","MF"] → True
      source "FW"    vs candidate "MF"    → "FW" in ["MF"]      → False
      source "MF,FW" vs candidate "FW,MF" → "MF" in ["FW","MF"] → True
      source "DF"    vs candidate "MF,DF" → "DF" in ["MF","DF"] → True
    """
    primary = _primary_pos(source_pos)
    if not primary:
        return True  # no position data — don't filter
    return primary in [p.strip() for p in candidate_pos.split(",")]


# ── Similarity helpers ─────────────────────────────────────────────────────────

_STAT_COLS = ["Goals", "Assists", "xG", "xAG", "PrgC", "PrgP", "Tkl", "KP"]

def _confidence_label(score: float) -> str:
    if score >= 0.97: return "Exceptional"
    if score >= 0.93: return "Strong"
    if score >= 0.88: return "Good"
    return "Moderate"

def _player_card(row, rank: int, score: float) -> str:
    pct   = score * 100
    label = _confidence_label(score)
    bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))

    lines = [
        f"  #{rank}  {row['Player']}",
        f"      Similarity : {pct:.1f}%  [{bar}]  {label}",
        f"      Position   : {row.get('Pos', 'N/A')}  |  Club : {row.get('Squad', 'N/A')}  |  League : {row.get('Comp', 'N/A')}",
    ]
    if "Age" in row.index:
        lines.append(f"      Age        : {row.get('Age', 'N/A')}  |  Nationality : {row.get('Nation', 'N/A')}")
    if "MP" in row.index and "Min" in row.index:
        lines.append(f"      Apps/Mins  : {row.get('MP', 'N/A')} apps / {int(row.get('Min', 0))} min")
    avail = [c for c in _STAT_COLS if c in row.index and row.get(c) is not None]
    if avail:
        lines.append(f"      Per-90     : {'  |  '.join(f'{c}: {row[c]:.2f}' for c in avail)}")
    return "\n".join(lines)


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def get_similar_players(player_name: str) -> str:
    """
    Finds the top 5 statistically similar football players for a given player using
    cosine similarity over autoencoder embeddings trained on 2024/25 season data.
    Results are STRICTLY filtered to the same positional group as the query player —
    a forward query will only return forwards, a midfielder query only midfielders, etc.
    Returns similarity scores, confidence labels, club, league, and per-90 metrics.
    """
    if _name_to_idx is None:
        return "ERROR: Player database unavailable — artifacts not loaded."

    # ── Name resolution (case-insensitive, last-name fallback) ────────────────
    key = player_name.strip().lower()
    idx = _name_to_idx.get(key)

    if idx is None:
        last_name  = key.split()[-1]
        candidates = [n for n in _name_to_idx if last_name in n]
        if len(candidates) == 1:
            idx = _name_to_idx[candidates[0]]
            log.info("Fuzzy matched '%s' → '%s'", player_name, player_info.iloc[idx]["Player"])
        else:
            suggestions = ", ".join(
                player_info.iloc[_name_to_idx[c]]["Player"] for c in candidates[:4]
            )
            hint = f"  Possible matches: {suggestions}" if suggestions else ""
            return f"Player '{player_name}' not found in the 2024/25 database.{hint}"

    t0     = time.perf_counter()
    target = embeddings[idx].unsqueeze(0)
    sims   = F.cosine_similarity(target, embeddings)

    source     = player_info.iloc[idx]
    source_pos = str(source.get("Pos", ""))
    primary    = _primary_pos(source_pos)

    # ── Position-filtered candidate selection ─────────────────────────────────
    # Pull a pool of 30, filter by position, widen to 50 if too few survive.
    def _filtered_pool(pool_size: int):
        n   = min(pool_size + 1, len(_name_to_idx))
        top = torch.topk(sims, n)
        return [
            (i.item(), s.item())
            for i, s in zip(top.indices[1:], top.values[1:])
            if _pos_compatible(source_pos, str(player_info.iloc[i.item()].get("Pos", "")))
        ]

    filtered = _filtered_pool(30)
    if len(filtered) < 2:
        filtered = _filtered_pool(50)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    top5       = filtered[:5]

    header = (
        f"TACTICAL SIMILARITY REPORT\n"
        f"{'─' * 52}\n"
        f"  Query player : {source['Player']}\n"
        f"  Position     : {source_pos}  (primary: {primary})\n"
        f"  Club         : {source.get('Squad', 'N/A')}  |  League : {source.get('Comp', 'N/A')}\n"
        f"  Filter       : position-locked to '{primary}' group\n"
        f"  Candidates   : {len(filtered)} position-compatible players found\n"
        f"  Search time  : {elapsed_ms:.1f}ms  |  DB size : {len(_name_to_idx)} players\n"
        f"{'─' * 52}\n"
        f"TOP {len(top5)} TACTICAL CLONES (same positional group):\n"
    )

    cards = [_player_card(player_info.iloc[i], rank, s) for rank, (i, s) in enumerate(top5, 1)]

    log.info(
        "Similarity search '%s' (%s): %d position-compatible results in %.1fms, top=%.3f",
        source["Player"], primary, len(filtered), elapsed_ms, top5[0][1] if top5 else 0,
    )
    return header + "\n\n".join(cards)


@tool
def search_player_news(player_name: str) -> str:
    """
    Searches the web for real-time injury updates, transfer rumours, or suspension news
    for a specific football player. Call this for each recommended clone to check
    their current availability before finalising a recommendation.
    """
    query = f"{player_name} football injury transfer news 2025"
    t0    = time.perf_counter()

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=5, timelimit="m"))
    except Exception as exc:
        log.warning("DuckDuckGo search failed for '%s': %s", player_name, exc)
        return f"Web search unavailable ({exc}). Rely on training-data knowledge for {player_name}."

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info("DDG search '%s': %d results in %.0fms", player_name, len(raw), elapsed_ms)

    if not raw:
        return f"NEWS — {player_name}: No recent results found. Search time: {elapsed_ms:.0f}ms"

    lines = [
        f"NEWS SEARCH — {player_name}",
        f"  Query: {query}  |  {len(raw)} results  |  {elapsed_ms:.0f}ms",
        "─" * 52,
    ]
    for i, r in enumerate(raw, 1):
        lines.append(f"  {i}. {r['title']}")
        lines.append(f"     {r['body']}")
        if r.get("href"):
            lines.append(f"     Source: {r['href']}")
    return "\n".join(lines)


# ── Agent assembly ─────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_retries=2,
)

SYSTEM_PROMPT = SystemMessage(content="""You are an elite AI football tactical scout. \
Each question is independent — do not invent or carry over context from previous turns.

TOOLS:
  - get_similar_players  : returns position-filtered tactical clones with similarity scores and per-90 metrics
  - search_player_news   : returns live injury, transfer, and availability intelligence

POSITION RULE (non-negotiable):
  The similarity tool already locks results to the query player's positional group.
  You must NEVER recommend a player from a different position. If the query is a forward,
  every recommendation must be a forward. If a midfielder, only midfielders. Do not override,
  ignore, or work around this constraint under any circumstances.

FORMAT: Respond in Markdown. Use this exact structure:

## Scout Report
*[One sentence stating the tactical context — position, role, and what is being sought.]*

### Analysis Steps
- `get_similar_players("[name]")` — [N position-compatible clones found, top score X% (Label)]
- `search_player_news("[name]")` — [summary of news found] *(one line per tool call)*

---

### Tactical Clones — [Query Player] *([Position])*

**#1 [Player Name]** | *[Position]* · [Club] · [League]
**Similarity:** [X.X%] — *[Label]*
**Why similar:** [2–3 sentences. Cite at least two per-90 stats in bold, e.g. **xG: 0.48**, **PrgC: 6.2**. Explain the tactical overlap.]
**Availability:** [news summary or "No concerns flagged"]

*(repeat for each clone)*

---

### Recommendation
[2–4 sentences. Name the top choice(s), explain the tactical fit, and flag any availability risk. \
If similarity scores are below 88%, note that the match quality is moderate.]

RULES:
- Cite exact similarity % and label for every player
- Bold every per-90 stat value you reference
- Prefix any injury or transfer news with **⚠ ALERT:**
- If fewer than 3 position-compatible clones were found, say so explicitly
- Never fabricate stats, names, or news not present in the tool output
- Markdown only — no plain text blocks""")

tactical_agent = create_react_agent(
    model=llm,
    tools=[get_similar_players, search_player_news],
    prompt=SYSTEM_PROMPT,
)
