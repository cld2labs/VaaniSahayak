"""Download model weights at Docker build time.

Called by Dockerfile.tts when BAKE_WEIGHTS=true.
Downloads:
  1. ai4bharat/indic-parler-tts  (TTS model + tokenizer)
  2. google/flan-t5-large         (description encoder tokenizer)
"""
import os
import sys

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_HOME = os.environ.get("HF_HOME", "/app/hf_cache")
TTS_MODEL_ID = os.environ.get("TTS_MODEL_ID", "ai4bharat/indic-parler-tts")

os.makedirs(HF_HOME, exist_ok=True)

from huggingface_hub import login, snapshot_download

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print(f"[download] Authenticated with HuggingFace Hub")
else:
    print("[download] No HF_TOKEN — attempting anonymous download")

print(f"[download] Downloading {TTS_MODEL_ID}...")
snapshot_download(TTS_MODEL_ID, token=HF_TOKEN or None, cache_dir=HF_HOME)
print(f"[download] {TTS_MODEL_ID} done.")

# Load model config to find the description encoder model ID (google/flan-t5-large)
import torch
from parler_tts import ParlerTTSForConditionalGeneration

print(f"[download] Loading model config to find description encoder...")
m = ParlerTTSForConditionalGeneration.from_pretrained(
    TTS_MODEL_ID,
    token=HF_TOKEN or None,
    torch_dtype=torch.float32,   # CPU load, just to read config
)
desc_model_id = m.config.text_encoder._name_or_path
del m
print(f"[download] Description encoder: {desc_model_id}")

print(f"[download] Downloading {desc_model_id}...")
snapshot_download(desc_model_id, token=HF_TOKEN or None, cache_dir=HF_HOME)
print(f"[download] {desc_model_id} done.")

print("[download] All weights cached successfully.")
