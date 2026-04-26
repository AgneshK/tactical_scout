import json
import logging
import math
import time

import pandas as pd
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
    """
    primary = _primary_pos(source_pos)
    if not primary:
        return True
    return primary in [p.strip() for p in candidate_pos.split(",")]

def _league_match(comp: str, league_filter: str) -> bool:
    """Substring match against FBRef Comp ('eng Premier League', 'de Bundesliga', etc.)."""
    if not league_filter:
        return True
    if not comp:
        return False
    return league_filter.strip().lower() in comp.lower()


# ── Display helpers ────────────────────────────────────────────────────────────

# FBRef uses lowercase 2-letter ISO codes for most countries, but UK home nations
# need the gb-{region} variant flagcdn supports.
_HOME_NATIONS = {"eng": "gb-eng", "wls": "gb-wls", "sct": "gb-sct", "nir": "gb-nir"}

def _parse_nation(nation: str) -> dict:
    """FBRef nation string 'eng ENG' → {code, label, flag_url}."""
    if not nation or not isinstance(nation, str):
        return {"code": "", "label": "", "flag_url": ""}
    parts = nation.strip().split()
    raw   = parts[0].lower() if parts else ""
    code  = _HOME_NATIONS.get(raw, raw)
    label = parts[1].upper() if len(parts) > 1 else raw.upper()
    flag  = f"https://flagcdn.com/24x18/{code}.png" if code else ""
    return {"code": code, "label": label, "flag_url": flag}

def _strip_country_prefix(s) -> str:
    """'eng Premier League' → 'Premier League'. Leaves bare strings alone."""
    if not s or not isinstance(s, str):
        return ""
    parts = s.split(" ", 1)
    return parts[1] if len(parts) > 1 and parts[0].islower() else s

def _safe(v):
    """Convert NaN/missing to None; preserve everything else."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


# ── Similarity helpers ─────────────────────────────────────────────────────────

_STAT_COLS = ["Goals", "Assists", "xG", "xAG", "PrgC", "PrgP", "Tkl", "KP"]

def _compute_percentiles(df: pd.DataFrame, stat_cols) -> dict:
    """Per-primary-position percentile rank (0-100) for each per-90 stat.

    Players are ranked against peers in their primary position group, so an FW's
    Goals percentile is meaningful relative to other FWs (not vs DFs).
    """
    if df is None or df.empty:
        return {}
    pos_series = df["Pos"].apply(_primary_pos)
    pcts = {}
    for stat in stat_cols:
        if stat not in df.columns:
            continue
        ranks = df.groupby(pos_series)[stat].rank(pct=True, method="average") * 100
        for idx, pct in ranks.items():
            if pd.isna(pct):
                continue
            name = str(df.iloc[idx]["Player"]).lower()
            pcts.setdefault(name, {})[stat] = round(float(pct), 1)
    return pcts

_percentiles = _compute_percentiles(player_info, _STAT_COLS) if player_info is not None else {}
if _percentiles:
    log.info("Percentiles computed for %d players across %d stats", len(_percentiles), len(_STAT_COLS))


def _confidence_label(score: float) -> str:
    if score >= 0.97: return "Exceptional"
    if score >= 0.93: return "Strong"
    if score >= 0.88: return "Good"
    return "Moderate"

def _row_to_dict(row, similarity: float | None = None, rank: int | None = None) -> dict:
    nation = _parse_nation(row.get("Nation", ""))
    age    = _safe(row.get("Age"))
    mp     = _safe(row.get("MP"))
    mins   = _safe(row.get("Min"))

    stats = {}
    for c in _STAT_COLS:
        if c in row.index:
            v = _safe(row.get(c))
            if v is not None:
                stats[c] = round(float(v), 2)

    out = {
        "name":        row["Player"],
        "position":    row.get("Pos", ""),
        "club":        row.get("Squad", ""),
        "league":      _strip_country_prefix(row.get("Comp", "")),
        "nation":      nation,
        "age":         int(age) if age is not None else None,
        "matches":     int(mp) if mp is not None else None,
        "minutes":     int(mins) if mins is not None else None,
        "stats":       stats,
        "percentiles": _percentiles.get(str(row["Player"]).lower(), {}),
    }
    if similarity is not None:
        out["similarity"] = round(float(similarity), 4)
        out["confidence"] = _confidence_label(similarity)
    if rank is not None:
        out["rank"] = rank
    return out


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def get_similar_players(player_name: str, league_filter: str | None = None) -> str:
    """
    Finds statistically similar football players using cosine similarity over autoencoder
    embeddings trained on 2024/25 season data. Results are STRICTLY filtered to the same
    positional group as the query player.

    Args:
        player_name: The player whose tactical clones you want.
        league_filter: Optional league name to restrict candidates to. Matched as a
            case-insensitive substring against the FBRef Comp field. Pass this
            whenever the user names a league constraint. Examples:
              "Bundesliga", "Premier League", "La Liga", "Serie A", "Ligue 1".
            Common aliases the user may say — translate before passing:
              EPL → "Premier League", BL → "Bundesliga", LaLiga → "La Liga".

    Returns JSON with {query, clones[], primary_pos, league_filter, candidate_pool, ...}.
    """
    if _name_to_idx is None:
        return json.dumps({"error": "Player database unavailable — artifacts not loaded."})

    # ── Name resolution: exact > token-exact > token-prefix > substring ───────
    # Tiered to avoid silently matching "Rodri" → "Rodrigo Bentancur".
    key = player_name.strip().lower()
    idx = _name_to_idx.get(key)

    if idx is None:
        last = key.split()[-1]
        tiers = [
            [n for n in _name_to_idx if key in n.split()],                           # exact word token
            [n for n in _name_to_idx if any(t.startswith(key) for t in n.split())],  # token prefix
            [n for n in _name_to_idx if last in n],                                  # last-name substring
        ]
        # First non-empty tier wins; pick only if it's unambiguous (single match).
        candidates = next((c for c in tiers if c), [])

        if len(candidates) == 1:
            idx = _name_to_idx[candidates[0]]
            log.info("Fuzzy matched '%s' → '%s'", player_name, player_info.iloc[idx]["Player"])
        else:
            sugs = [player_info.iloc[_name_to_idx[c]]["Player"] for c in candidates[:6]]
            return json.dumps({
                "error": (
                    f"'{player_name}' is not in the 2024/25 dataset. "
                    f"Players with under 500 minutes played (or any goalkeeper) are excluded "
                    f"— this notably affects players who missed most of the season through injury "
                    f"(e.g. Rodri's ACL). Multiple similar names match the query."
                ),
                "queried_name": player_name,
                "suggestions": sugs,
                "must_ask_user": True,
            })

    t0     = time.perf_counter()
    target = embeddings[idx].unsqueeze(0)
    sims   = F.cosine_similarity(target, embeddings)

    source     = player_info.iloc[idx]
    source_pos = str(source.get("Pos", ""))
    primary    = _primary_pos(source_pos)

    league = (league_filter or "").strip()

    def _filtered_pool(pool_size: int):
        n   = min(pool_size + 1, len(_name_to_idx))
        top = torch.topk(sims, n)
        return [
            (i.item(), s.item())
            for i, s in zip(top.indices[1:], top.values[1:])
            if _pos_compatible(source_pos, str(player_info.iloc[i.item()].get("Pos", "")))
            and _league_match(str(player_info.iloc[i.item()].get("Comp", "")), league)
        ]

    # Widen the pool aggressively when filtering by league — in-league candidates
    # may not appear in the global top-30 by cosine similarity.
    initial_pool = 200 if league else 30
    filtered     = _filtered_pool(initial_pool)
    if len(filtered) < 2:
        filtered = _filtered_pool(len(_name_to_idx))

    if not filtered:
        msg = f"No position-compatible players found for '{source['Player']}'"
        if league:
            msg += f" in league matching '{league}'"
        return json.dumps({"error": msg, "league_filter": league or None})

    elapsed_ms = (time.perf_counter() - t0) * 1000
    top5       = filtered[:5]

    payload = {
        "query":           _row_to_dict(source),
        "clones":          [_row_to_dict(player_info.iloc[i], s, rank) for rank, (i, s) in enumerate(top5, 1)],
        "primary_pos":     primary,
        "league_filter":   league or None,
        "candidate_pool":  len(filtered),
        "search_time_ms":  round(elapsed_ms, 1),
        "db_size":         len(_name_to_idx),
    }

    log.info(
        "Similarity search '%s' (%s, league=%r): %d position-compatible results in %.1fms, top=%.3f",
        source["Player"], primary, league or None, len(filtered), elapsed_ms, top5[0][1] if top5 else 0,
    )
    return json.dumps(payload)


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

SYSTEM_PROMPT = SystemMessage(content="""You are an elite AI football tactical scout.

# MANDATORY WORKFLOW
You MUST invoke `get_similar_players` BEFORE writing any response about a player.
NEVER answer from your own knowledge. NEVER write placeholder text like "Top clone's name",
"[Player Name]", or "X.X%" — those are forbidden. Use the ACTUAL player names and stat
values returned by the tool. If you do not have tool output, your only valid action is
to call the tool.

# TOOLS
- get_similar_players(player_name, league_filter=None): returns JSON with {query, clones[5], ...}.
  Each clone has: name, position, club, league, nation, age, stats{Goals, Assists, xG, xAG,
  PrgC, PrgP, Tkl, KP}, similarity (0-1), confidence (Exceptional/Strong/Good/Moderate), rank.
  PASS league_filter whenever the user names a league. Examples of user phrasing → filter value:
    "in the Bundesliga"   → league_filter="Bundesliga"
    "from La Liga"        → league_filter="La Liga"
    "EPL alternative"     → league_filter="Premier League"
    "Serie A clone"       → league_filter="Serie A"
    "Ligue 1 option"      → league_filter="Ligue 1"
  If no league is mentioned, omit league_filter.
- search_player_news(player_name): plain-text injury/transfer news. Use for the TOP pick only.

# POSITION RULE
The similarity tool already locks results to the query player's position. Never override.

# AMBIGUOUS NAMES — CRITICAL
If get_similar_players returns an error JSON containing `suggestions` or `must_ask_user: true`,
you MUST stop and ask the user to clarify. You may NOT call the tool again with one of the
suggestions, and you may NOT pick a "closest" name yourself. The user asked about a specific
player — silently substituting a different one is a serious failure.

When this happens, respond ONLY in this format (no Scout Report, no clones, no analysis):

## Player not found
*[Repeat the tool's error message verbatim — including the note about the 500-minute cutoff.]*

Did you mean one of these?
- [Suggestion 1]
- [Suggestion 2]
- [Suggestion 3]

*Or please clarify the player you're asking about.*

# UI CONTEXT
The frontend renders ONE card — the top-ranked clone (rank #1) — showing its club, league,
country flag, position, age, similarity %, and per-90 stats. Your prose must NOT repeat that
meta. Focus the recommendation on the #1 clone. You may briefly contrast 1–2 alternatives in
text for context, but the headline pick is always #1 (unless it has a serious news flag, in
which case state explicitly that you'd defer to #2 and explain why).

# RESPONSE FORMAT (markdown only)
Below is a worked example showing the EXACT shape and level of detail expected. Substitute
real values from your tool calls — never copy these placeholders.

## Scout Report
*A press-resistant deep-lying playmaker to dictate Bundesliga tempo, profiled against Rodri.*

### Analysis Steps
- get_similar_players("Rodri", league_filter="Bundesliga") — 5 in-league candidates, top match 92.1% (Strong)
- search_player_news("Joshua Kimmich") — minor knock, available this weekend

---

### Top Pick: Joshua Kimmich
Mirrors Rodri's tempo control. **PrgP: 8.4** and **KP: 2.1** highlight his game-dictating
passing range, with **Tkl: 2.6** showing the defensive bite that anchors the midfield pivot.
> ⚠ ALERT: minor hamstring knock, expected to play.

### Runners-up
- **Granit Xhaka** — Deep distributor with **PrgP: 7.1** and **xAG: 0.18**, but **Tkl: 1.8** is below Rodri's defensive output.
- **Aleksandar Pavlović** — Younger profile, **PrgP: 6.8** and **KP: 1.6**.

*(Always use REAL player names from the tool. Bold real stat values. Omit the ⚠ ALERT line entirely when no news concern. Skip the "Runners-up" section if fewer than 2 alternatives exist.)*

---

### Recommendation
*1–3 sentences. Reaffirm the top pick and the tactical fit. Flag availability risk if any.
If all similarity scores are below 88%, explicitly note the moderate match quality.*

# HARD RULES
- ALWAYS call get_similar_players first — no exceptions
- Use REAL player names and REAL stat values from tool output
- Bold every per-90 stat value you reference
- Never invent stats, names, leagues, or news
- Do not restate club, league, country, or age in prose — the cards show those
- Omit the ⚠ ALERT line when there is no news to flag (do not write "no concerns" inside an alert block)
- Markdown only — no plain text blocks, no leaked tool-call syntax""")

tactical_agent = create_react_agent(
    model=llm,
    tools=[get_similar_players, search_player_news],
    prompt=SYSTEM_PROMPT,
)
