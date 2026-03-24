# Vaani Sahayak (वाणी सहायक) — Project Context

## What This Is
A sovereign AI-powered Hindi voice assistant for navigating Indian government welfare schemes.
Users speak/type queries in Hindi → system retrieves relevant schemes → Param-1 generates a Hindi answer → Indic Parler-TTS speaks it back.

## Sovereign AI Narrative (Key Demo Angle)
Everything runs on Indian AI models:
- **LLM**: Param-1-2.9B-Instruct (bharatgenai/Param-1-2.9B-Instruct on HuggingFace) — BharatGen consortium (IIT Madras + IIT Bombay + IIT Kanpur etc.)
- **TTS**: Indic Parler-TTS (ai4bharat/indic-parler-tts) — AI4Bharat model
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (can swap to ai4bharat/indic-sentence-bert later)
- **Data**: shrijayan/gov_myscheme — 723 schemes from myscheme.gov.in
No OpenAI, no external APIs, runs fully offline on a Mac with Apple Silicon.

## Architecture: Option B — JSON RAG-lite
Query → Embedding → Cosine similarity over 723 scheme embeddings → Top-3 schemes injected into Param-1 context → Hindi text response → Indic Parler-TTS → audio.

## Project Structure
```
vaani/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # All constants/model names
│   ├── models/
│   │   ├── param_model.py       # Param-1 inference wrapper
│   │   └── tts_model.py         # Indic Parler-TTS wrapper
│   ├── retrieval/
│   │   ├── embeddings.py        # Pre-compute + load scheme embeddings
│   │   └── retriever.py         # Query → top-k schemes
│   ├── data/
│   │   ├── schemes.json         # 723 schemes (downloaded from HuggingFace)
│   │   └── scheme_embeddings.npy # Pre-computed embeddings (generated once)
│   └── requirements.txt
├── scripts/
│   ├── download_data.py         # Pull HuggingFace dataset → schemes.json
│   └── precompute_embeddings.py # Run once: embed all 723 schemes
└── frontend/
    ├── index.html
    ├── package.json
    └── src/
        ├── App.jsx
        └── components/
            ├── QueryInput.jsx
            ├── ResponsePanel.jsx
            ├── SchemeCard.jsx
            └── AudioPlayer.jsx
```

## Key Data
- **Source**: `shrijayan/gov_myscheme` on HuggingFace (Parquet/JSON/CSV)
- **Schema per scheme**: name, description, eligibility, benefits, application_process, documents, faqs, official_link, category, state
- **Count**: 723 schemes covering Central + State governments
- **Categories**: Agriculture, Health, Education, Social Welfare, Women & Child, Housing, Finance, Skills, etc.

## API Endpoints
- `POST /ask` — `{query: str, language: "hi"|"en"}` → `{text: str, audio_base64: str, schemes: [...], latency_ms: int}`
- `GET /schemes` — list all schemes with pagination
- `GET /schemes/{id}` — single scheme detail
- `GET /health` — model load status check

## Platform
- macOS with Apple Silicon (MPS acceleration)
- Python 3.11+
- Node 18+ for frontend

## Quick Start (after setup)
```bash
# 1. Install deps
pip install -r backend/requirements.txt

# 2. Download data + precompute embeddings (one-time)
python scripts/download_data.py
python scripts/precompute_embeddings.py   # uses sentence-transformers locally

# 3. Start vLLM servers (two terminals)
vllm serve bharatgenai/Param-1-2.9B-Instruct --port 8001
vllm serve sentence-transformers/all-MiniLM-L6-v2 --task embed --port 8002

# 4. Start FastAPI
uvicorn backend.main:app --reload --port 8000

# 5. Start frontend
cd frontend && npm install && npm run dev
```

## Model Serving Architecture
| Component | How it runs | Port |
|-----------|-------------|------|
| Param-1 LLM | servers/server_param1.py (transformers + MPS) | 8001 |
| Embeddings | sentence-transformers in-process | — |
| TTS (Indic Parler-TTS) | In-process (FastAPI loads it) | — |
| FastAPI backend | uvicorn | 8000 |
| React frontend | Vite dev server | 5173 |

**Why not vLLM/MLX LM for Param-1?**
Param-1 uses a custom architecture `ParamBharatGenForCausalLM` that is not supported by either
vLLM 0.16.0 or mlx-lm 0.31.0. `servers/server_param1.py` wraps transformers directly and exposes
an OpenAI-compatible `/v1/chat/completions` API on port 8001 — same interface, no external dep.

The FastAPI backend calls it via the `openai` client with `base_url="http://localhost:8001/v1"`.

## Build Timeline
- Day 1: Data download + verify Param-1 and TTS models run locally
- Day 2: Retrieval layer + FastAPI endpoints
- Day 3: TTS integration + voice pipeline
- Day 4: React frontend
- Day 5: Polish, testing, demo prep

## Prior Art Reference
OpenNyAI Jugalbandi: scraped myScheme, built FastAPI + WhatsApp bot, used Bhashini STT + OpenAI. Vaani replaces OpenAI with Param-1 and modernizes TTS with Indic Parler-TTS.

## Notes
- Param-1 context window: 2,048 tokens. Keep each scheme summary to ~150 tokens. Inject top-3 schemes = ~450 tokens, leaving ~1,500 for instruction + response.
- TTS inference on MPS can be slow (~5-10s for a 3-sentence response). Cache audio for repeated queries.
- Hindi retrieval: all-MiniLM-L6-v2 handles bilingual queries decently. If Hindi queries miss, add BM25 hybrid search (rank_bm25 lib).
