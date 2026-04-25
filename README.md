# Tactical Scout

An AI-powered football scouting tool that finds statistically similar players — **tactical clones** — using a PyTorch autoencoder and a LangGraph ReAct agent. Ask it in plain English; it responds with a structured scout report.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-8a2be2?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Groq%20llama--3.3--70b-f97316?style=flat-square)

**Live demo:** [tactical-scout-two.vercel.app](https://tactical-scout-two.vercel.app/)

---

## What it does

1. **Type a question** — e.g. *"Find me a cheaper alternative to Bukayo Saka"*
2. The **ReAct agent** (Groq / Llama 3.3 70B) decides to call the similarity tool
3. The tool runs **cosine similarity** over 8-dimensional autoencoder embeddings trained on 200+ FBRef features from the 2024/25 season
4. Results are **position-locked** — a forward query only returns forwards
5. The agent fetches **live news** (injuries, transfers) for each candidate via DuckDuckGo
6. A structured **Markdown scout report** is returned with similarity scores, per-90 stats, and a recommendation

---

## Architecture

```
User (browser)
    │  POST /chat  { message }
    ▼
FastAPI  (backend/main.py)
    │  HumanMessage → LangGraph ReAct agent
    ▼
Agent  (backend/agent.py)
    ├── Tool: get_similar_players(player_name)
    │       └── cosine_similarity over embeddings.pt
    │           filtered to same positional group
    └── Tool: search_player_news(player_name)
            └── DuckDuckGo live web search
```

**ML pipeline** (`backend/autoencoder.py` / `tune_model.py`)

| Step | Detail |
|---|---|
| Data | FBRef 2024/25 · outfield players only · 500+ minutes played |
| Normalisation | Goals & Assists converted to per-90 |
| Preprocessing | `PowerTransformer` (Yeo-Johnson) → `RobustScaler` |
| Architecture | `num_features → 32 → 16 → 8 → 16 → 32 → num_features` |
| Latent space | 8 dimensions |
| Training | 100 epochs · batch 32 · Adam lr=0.001 · MSE loss |
| Output | `artifacts/embeddings.pt` · `artifacts/player_info.pkl` |

A separate tuner (`tune_model.py`) runs an **Optuna hyperparameter search** with scout-weighted MSE loss — attacking and creativity features (xG, xAG, PrgC, KP, SCA, GCA) carry 2× reconstruction weight so the latent space reflects tactical function rather than statistical volume.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | PyTorch, scikit-learn, Optuna |
| Agent | LangGraph, LangChain, Groq API |
| Backend | FastAPI, Uvicorn |
| Frontend | Vanilla HTML / CSS / JS, marked.js |
| Data | FBRef (via CSV) |

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/your-username/tactical-scout.git
cd tactical-scout
pip install -r requirements.txt
```

### 2. Configure environment variables

Create `backend/.env`:

```
GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Train the autoencoder

Must be run once before starting the backend. Generates the embedding artifacts.

```bash
python backend/autoencoder.py
```

Output: `backend/artifacts/embeddings.pt` and `backend/artifacts/player_info.pkl`

> The autoencoder uses CUDA automatically if available, otherwise falls back to CPU.

### 4. Start the backend

```bash
uvicorn backend.main:app --reload
```

API available at `http://localhost:8000`

### 5. Open the frontend

```bash
python -m http.server 3000 -d frontend
```

Then open `http://localhost:3000` in your browser. Or simply open `frontend/index.html` directly.

---

## Hyperparameter Tuning (optional)

To search for a better autoencoder configuration using Optuna:

```bash
# Default: 50 trials
python backend/tune_model.py

# Custom run
python backend/tune_model.py --n-trials 100 --timeout 3600

# Resume a previous study
python backend/tune_model.py --study-name my_run
```

The tuner retrains the best configuration on the full dataset and overwrites the artifacts. Best parameters are saved to `backend/artifacts/best_params.json`.

**What gets searched:** latent dimension, learning rate, hidden layer sizes, dropout, weight decay, batch size, and optimizer (Adam vs AdamW).

---

## Example Query

> *"Who are the top 5 tactical alternatives to Rodri in the Premier League?"*

The agent returns a scout report with similarity percentages, per-90 metrics (xG, xAG, PrgC, PrgP, Tkl, KP), live availability news, and a recommendation — all in Markdown.

---

## Project Structure

```
tactical-scout/
├── backend/
│   ├── agent.py          # LangGraph ReAct agent + tools
│   ├── autoencoder.py    # ML training pipeline
│   ├── tune_model.py     # Optuna hyperparameter search
│   ├── main.py           # FastAPI app
│   ├── artifacts/        # Generated embeddings (gitignored)
│   └── .env              # API keys (gitignored)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/
│   └── players_data-2024_2025.csv
└── requirements.txt
```

---

## Deployment

| Service | URL |
|---|---|
| Frontend | [tactical-scout-two.vercel.app](https://tactical-scout-two.vercel.app/) |
| Backend API | [tactical-scout.onrender.com](https://tactical-scout.onrender.com/) |

The frontend is hosted on Vercel and the backend on Render. Both redeploy automatically on every push to `main`.

> **Note:** The Render free tier spins down after 15 minutes of inactivity. The first request after idle may take 30–60 seconds to respond.

---

## License

MIT
