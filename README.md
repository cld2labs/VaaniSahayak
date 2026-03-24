<p align="center">
  <img src="docs/assets/InnovationHub-HeaderImage.png" width="800" alt="Innovation Hub">
</p>

# Vaani Sahayak (वाणी सहायक)

**Sovereign AI-Powered Hindi Voice Assistant for Indian Government Welfare Schemes**

Vaani Sahayak helps citizens navigate 2000+ Indian government welfare schemes through natural Indic voice conversations — powered entirely by Indian AI models, running fully offline.

---

## Table of Contents

- [Vaani Sahayak (वाणी सहायक)](#vaani-sahayak-वाणी-सहायक)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Sovereign AI Stack](#sovereign-ai-stack)
  - [How It Works](#how-it-works)
  - [Architecture](#architecture)
  - [Get Started](#get-started)
    - [Prerequisites](#prerequisites)
    - [Quick Start (Local)](#quick-start-local)
    - [Docker Deployment](#docker-deployment)
  - [Project Structure](#project-structure)
  - [API Reference](#api-reference)
    - [Example: Ask a Question](#example-ask-a-question)
  - [Usage](#usage)
  - [Environment Variables](#environment-variables)
  - [Model Serving](#model-serving)
  - [Technology Stack](#technology-stack)
    - [Backend](#backend)
    - [Frontend](#frontend)
  - [Performance Notes](#performance-notes)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [License](#license)
  - [Disclaimer](#disclaimer)

---

## Overview

Citizens across India often struggle to discover which government welfare schemes they're eligible for. Vaani Sahayak solves this by letting users **ask questions in Hindi or Telugu** and receiving **spoken answers** about relevant schemes — eligibility, benefits, application process, and required documents.

**Key highlights:**
- Voice-in, voice-out interaction in Hindi and Telugu
- Retrieves from 2,000+ real schemes scraped from [myscheme.gov.in](https://www.myscheme.gov.in/)
- Supports Enterprise Inference (EI) GPU stack for production TTS, with local MPS fallback
- Streaming responses with sentence-by-sentence audio playback

---

## Sovereign AI Stack

Everything runs on Indian AI models — no OpenAI, no external APIs:

| Component | Model | Origin |
|-----------|-------|--------|
| **LLM** | [Param-1-2.9B-Instruct](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct) | BharatGen (IIT Madras + IIT Bombay + IIT Kanpur) |
| **TTS** | [Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts) | AI4Bharat |
| **Embeddings** | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Sentence-Transformers |
| **Data** | [gov_myscheme](https://huggingface.co/datasets/shrijayan/gov_myscheme) | 2,000+ schemes from myscheme.gov.in |

---

## How It Works

```
1. User speaks/types a query in Hindi or Telugu
        ↓
2. Query is embedded using all-MiniLM-L6-v2
        ↓
3. Cosine similarity retrieves the top-3 most relevant schemes
        ↓
4. Schemes are injected into Param-1's context → Hindi/Telugu answer generated
        ↓
5. Indic Parler-TTS speaks the answer back sentence-by-sentence
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                       │
│               (Vite + Tailwind CSS)                      │
│                    :5173 / :3002                          │
└────────────────────────┬────────────────────────────────┘
                         │  HTTP / SSE
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend (:8000)                  │
│                                                          │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │ Retriever │  │  Param-1 LLM  │  │   TTS Router     │  │
│  │ (cosine)  │  │  Client (:8001│  │  EI → Local      │  │
│  └──────────┘  └───────────────┘  └──────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┼─────────────────┐
        ▼                ▼                 ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐
│ Embeddings   │ │  Param-1     │ │  TTS (priority order)  │
│ (in-process) │ │  LLM server  │ │  1. EI GPU Stack       │
│ MiniLM-L6-v2 │ │  (:8001 MPS) │ │     (APISIX + Keycloak)│
└──────────────┘ └──────────────┘ │  2. Local TTS (:8003)  │
                                  │     (:8003, MPS)       │
                                  └────────────────────────┘
```

---

## Get Started

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4) — MPS acceleration
- **Python 3.11+**
- **Node 18+** (for the frontend)
- **~16 GB RAM** recommended (Param-1 ≈ 6 GB, TTS ≈ 4 GB, embeddings ≈ 1 GB)
- **HuggingFace account** with accepted model licenses

### Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/cld2labs/VaaniSahayak.git
cd VaaniSahayak

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r backend/requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your HF_TOKEN

# 5. Download scheme data and pre-compute embeddings (one-time)
python scripts/download_data.py
python scripts/precompute_embeddings.py

# 6. Start the Param-1 LLM server (Terminal 1)
python servers/server_param1.py --preload --port 8001

# 7. Start the TTS server (Terminal 2)
python servers/server_tts.py --preload --port 8003

# 8. Start the FastAPI backend (Terminal 3)
uvicorn backend.main:app --reload --port 8000

# 9. Start the React frontend (Terminal 4)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** in your browser.

### Docker Deployment

> **Note:** `servers/server_param1.py` and `servers/server_tts.py` must run natively on your Mac for MPS (Metal) acceleration. Docker on macOS uses Linux containers where MPS is unavailable.

```bash
# 1. Start model servers natively (two terminals)
python servers/server_param1.py --preload --port 8001
python servers/server_tts.py --preload --port 8003

# 2. Start backend + frontend via Docker
docker compose up --build
```

- **Backend:** http://localhost:8000
- **Frontend:** http://localhost:3002

### EI GPU TTS Deployment

The `Cuda/` directory contains everything needed to deploy Indic Parler-TTS on a CUDA GPU server. See [`Cuda/README.md`](Cuda/README.md) for the full deployment guide.

**Quick version:**

```bash
# 1. Create HF token secret
kubectl create secret generic hf-token \
  --from-literal=HUGGINGFACEHUB_API_TOKEN=hf_YOUR_TOKEN

# 2. Build Docker image on the GPU machine
cd Cuda && docker build -t tts:local .

# 3. Deploy
kubectl apply -f k8s/tts-deployment.yaml
kubectl apply -f k8s/tts-service.yaml
kubectl apply -f k8s/tts-apisix-route.yaml

# 4. Verify (look for "READY" in logs)
kubectl logs -f deployment/tts-deployment

# 5. Point Vaani backend to it (in .env)
EI_TTS_URL=http://api.example.com:32237/v1/tts
EI_TTS_TOKEN=<your-bearer-token>
```

The backend's TTS router automatically prefers the EI GPU path over the local MPS server when `EI_TTS_URL` is configured.

**Supported GPUs:** V100 (split SDPA + KV cache), A10/A100/H100 (+ Flash Attention 2 + torch.compile).

---

## Project Structure

```
vaani/
├── backend/
│   ├── main.py                    # FastAPI app — all endpoints
│   ├── config.py                  # Central configuration
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── models/
│   │   ├── param_model.py         # Param-1 inference wrapper
│   │   └── tts_model.py           # Multi-backend TTS routing
│   ├── retrieval/
│   │   ├── embeddings.py          # Load pre-computed scheme embeddings
│   │   └── retriever.py           # Cosine similarity retrieval
│   ├── services/
│   │   └── keycloak_auth.py       # OAuth2 token manager (EI stack)
│   └── data/
│       ├── schemes.json           # 2,000+ schemes (generated by download_data.py)
│       └── scheme_embeddings.npy  # Pre-computed embeddings (generated once)
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── src/
│       ├── App.jsx                # Main app — streaming, suggestions, language toggle
│       ├── main.jsx
│       ├── index.css
│       └── components/
│           ├── QueryInput.jsx     # Hindi/Telugu text input with language toggle
│           ├── ResponsePanel.jsx  # Streaming answer + audio playback
│           ├── SchemeCard.jsx     # Scheme result card
│           └── ModelBadge.jsx     # Sovereign AI model attribution
├── Cuda/                            # EI GPU TTS deployment package (see Cuda/README.md)
│   ├── Dockerfile                   # NVIDIA CUDA 12.1 + PyTorch 2.3.1 image
│   ├── server_tts_cuda.py          # CUDA-optimized TTS server
│   ├── text_normalize.py           # Hindi text normalization for TTS
│   ├── requirements.txt            # Pinned Python deps for CUDA build
│   ├── README.md                   # Full deployment guide
│   └── k8s/
│       ├── tts-deployment.yaml     # K8s Deployment (GPU node, env vars)
│       ├── tts-service.yaml        # K8s ClusterIP Service
│       └── tts-apisix-route.yaml   # APISIX gateway route (/v1/tts)
├── scripts/
│   ├── download_data.py           # Pull myscheme.gov.in dataset → schemes.json
│   ├── precompute_embeddings.py   # Embed all 2,086 schemes (one-time)
│   ├── test_remote_tts.py         # Benchmark TTS: sequential / parallel / batch
│   └── archive/                   # Retired test scripts
├── servers/
│   ├── server_param1.py           # Param-1 OpenAI-compatible server (MPS)
│   ├── server_tts.py              # Indic Parler-TTS server (MPS, local fallback)
│   └── text_normalize.py          # Hindi text normalization (shared)
├── docker-compose.yml
├── Dockerfile.tts
├── .env.example
├── .gitignore
├── README.md
├── CONTRIBUTING.md
├── LICENSE.md
├── SECURITY.md
├── DISCLAIMER.md
├── TERMS_AND_CONDITIONS.md
└── TROUBLESHOOTING.md
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Single query → text + audio + schemes |
| `POST` | `/ask/stream` | Streaming token-by-token generation (SSE) |
| `POST` | `/ask/speak` | Interleaved LLM + TTS — sentence-by-sentence audio (SSE) |
| `POST` | `/narrate` | Text-to-speech streaming |
| `GET` | `/schemes` | Paginated scheme listing with category filter |
| `GET` | `/schemes/suggestions` | Auto-generated Hindi/Telugu sample queries |
| `GET` | `/schemes/{id}` | Single scheme detail |
| `GET` | `/categories` | List all scheme categories |
| `GET` | `/health` | Model load status check |

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "गरीब परिवारों के लिए कौन सी योजनाएं हैं?", "language": "hi"}'
```

**Response:**
```json
{
  "text": "गरीब परिवारों के लिए कई योजनाएं उपलब्ध हैं...",
  "audio_base64": "UklGRi4A...",
  "schemes": [...],
  "latency_ms": 3200
}
```

---

## Usage

1. **Open the app** at http://localhost:5173
2. **Toggle language** between हिंदी (Hindi) and తెలుగు (Telugu)
3. **Type or pick a suggestion** — e.g., "महिलाओं के लिए कौन सी योजनाएं हैं?"
4. **View the streaming response** — answer appears token-by-token
5. **Listen** — audio plays sentence-by-sentence as each is synthesized
6. **Browse scheme cards** — see name, category, similarity score, and official links

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace access token (for gated models) |
| `VLLM_LLM_URL` | `http://localhost:8001/v1` | Param-1 server endpoint |
| `TTS_SERVER_URL` | `http://localhost:8003` | Local TTS server endpoint |
| `HOST` | `0.0.0.0` | Backend bind address |
| `PORT` | `8000` | Backend port |
| `EI_TTS_URL` | — | Enterprise Inference GPU TTS endpoint (optional) |
| `EI_TTS_TOKEN` | — | Static Bearer token for EI TTS (optional) |
| `EI_KEYCLOAK_URL` | — | Keycloak base URL for EI auth (optional) |
| `EI_KEYCLOAK_REALM` | `ei` | Keycloak realm (optional) |
| `EI_CLIENT_ID` | — | OAuth2 client ID (optional) |
| `EI_CLIENT_SECRET` | — | OAuth2 client secret (optional) |
| `EI_VERIFY_SSL` | `true` | SSL verification for EI endpoints |

---

## Model Serving

| Component | Server | Port | Device |
|-----------|--------|------|--------|
| Param-1 LLM | `servers/server_param1.py` | 8001 | MPS (Apple Silicon) |
| Indic Parler-TTS (EI) | EI GPU Stack (APISIX + Keycloak) | remote | CUDA GPU |
| Indic Parler-TTS (local) | `servers/server_tts.py` | 8003 | MPS (Apple Silicon) |
| Embeddings | sentence-transformers (in-process) | — | CPU |
| FastAPI Backend | uvicorn | 8000 | — |
| React Frontend | Vite dev / Nginx | 5173 / 3002 | — |

**TTS routing priority:**
1. **Enterprise Inference (EI) GPU stack** — Keycloak-authenticated, GPU-accelerated Indic Parler-TTS behind an APISIX gateway. Supports batch synthesis, parallel sentence synthesis, and SSE streaming. Configure via `EI_TTS_URL` + Keycloak or static token.
2. **Local `servers/server_tts.py`** — MPS-accelerated fallback on Apple Silicon. Used when EI is not configured.

**Why custom model servers instead of vLLM?**
Param-1 uses a custom architecture (`ParamBharatGenForCausalLM`) not supported by vLLM or mlx-lm. `servers/server_param1.py` wraps HuggingFace Transformers directly and exposes an OpenAI-compatible `/v1/chat/completions` API — same interface, no compatibility issues.

---

## Technology Stack

### Backend
- **FastAPI** — async API framework with SSE streaming
- **HuggingFace Transformers** — model inference
- **sentence-transformers** — embedding computation
- **PyTorch** — MPS-accelerated inference on Apple Silicon
- **OpenAI Python SDK** — client for vLLM-compatible servers
- **Pydantic v2** — request/response validation

### Frontend
- **React 18** — UI framework
- **Vite** — build tool and dev server
- **Tailwind CSS** — utility-first styling
- **Nginx** — production static file server (Docker)

---

## Performance Notes

- **Param-1 context window:** 2,048 tokens. Each scheme summary is capped at ~150 tokens. Top-3 schemes ≈ 450 tokens, leaving ~1,500 for instruction + response.
- **TTS latency (EI):** Sub-second per sentence on GPU. Batch and parallel synthesis modes overlap network I/O with GPU compute for faster end-to-end audio.
- **TTS latency (local):** ~5–10 seconds per 3-sentence response on MPS. The streaming `/ask/speak` endpoint sends audio chunk-by-chunk so playback begins before full synthesis completes.
- **MPS memory management:** `servers/server_param1.py` supports `/suspend` and `/resume` endpoints to offload weights to CPU when TTS needs GPU memory.
- **Retrieval:** Cosine similarity over 2,000+ pre-computed 384-dim embeddings runs in <10 ms.

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

---

## Disclaimer

This project is for **educational and demonstration purposes only**. See [DISCLAIMER.md](DISCLAIMER.md) and [TERMS_AND_CONDITIONS.md](TERMS_AND_CONDITIONS.md).

---

Built with care by [Cloud2 Labs](https://github.com/cld2labs)
