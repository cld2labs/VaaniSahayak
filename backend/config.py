"""Central configuration for Vaani Sahayak backend."""
import os
from pathlib import Path

ROOT = Path(__file__).parent

# --- Model IDs ---
PARAM_MODEL_ID = "bharatgenai/Param-1-2.9B-Instruct"
TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
TTS_TOKENIZER_ID = "ai4bharat/indic-parler-tts"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# --- Model server endpoints ---
# param1-server (transformers + MPS): python server_param1.py --preload --port 8001
VLLM_LLM_URL = os.getenv("VLLM_LLM_URL", "http://localhost:8001/v1")
# TTS server (Indic Parler-TTS + MPS): python server_tts.py --preload --port 8003
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://localhost:8003")

# --- Data paths ---
SCHEMES_JSON = ROOT / "data" / "schemes.json"
EMBEDDINGS_NPY = ROOT / "data" / "scheme_embeddings.npy"

# --- Retrieval ---
TOP_K_SCHEMES = 10               # how many schemes the retriever returns
MAX_SCHEMES_IN_PROMPT = 3        # how many fit in Param-1's 2048-token window
MAX_SCHEME_TOKENS = 150          # per-scheme token budget in the prompt

# --- Param-1 generation ---
MAX_NEW_TOKENS = 512   # hard cap — enough for a detailed Hindi answer
TEMPERATURE = 0.6  # model card recommended value

# --- TTS ---
# Voice description passed to Indic Parler-TTS (runs in-process, vLLM doesn't support TTS)
TTS_DESCRIPTION = (
    "A calm and clear female Hindi voice, speaking slowly and distinctly, "
    "suitable for government information delivery. Studio quality."
)
TTS_LANGUAGE = "hi"  # ISO 639-1 code for Hindi

# --- Enterprise Inference (EI) TTS ---
# When set, the backend calls the GPU-accelerated EI stack instead of the
# local server_tts.py. Leave blank to use local TTS.
#
# EI_TTS_URL        : APISIX gateway URL for TTS  (e.g. https://apisix.example.com/indic-parler-tts)
# EI_KEYCLOAK_URL   : Keycloak base URL            (e.g. https://keycloak.example.com)
# EI_KEYCLOAK_REALM : Keycloak realm name          (e.g. "ei")
# EI_CLIENT_ID      : OAuth2 client_id
# EI_CLIENT_SECRET  : OAuth2 client_secret
# EI_VERIFY_SSL     : set "false" for self-signed certs in dev
EI_TTS_URL        = os.getenv("EI_TTS_URL", "")
EI_TTS_TOKEN      = os.getenv("EI_TTS_TOKEN", "")   # static Bearer token (no Keycloak needed)
EI_KEYCLOAK_URL   = os.getenv("EI_KEYCLOAK_URL", "")
EI_KEYCLOAK_REALM = os.getenv("EI_KEYCLOAK_REALM", "ei")
EI_CLIENT_ID      = os.getenv("EI_CLIENT_ID", "")
EI_CLIENT_SECRET  = os.getenv("EI_CLIENT_SECRET", "")
EI_VERIFY_SSL     = os.getenv("EI_VERIFY_SSL", "true").lower() == "true"

# True when EI_TTS_URL is set with either a static token OR full Keycloak creds
EI_TTS_ENABLED = bool(EI_TTS_URL and (EI_TTS_TOKEN or (EI_KEYCLOAK_URL and EI_CLIENT_ID and EI_CLIENT_SECRET)))

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
