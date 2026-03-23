# Production TTS Server — V100 + Kubernetes
# Optimizations: split attention (SDPA decoder + eager encoder), static cache,
# torch.compile, use_cache probe, pre-encoded description, warmup

import argparse
import asyncio
import base64
import gc
import io
import json
import logging
import os
import sys
import time
import traceback
from threading import Thread

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from text_normalize import normalize_for_tts

# ── Patch StaticCache for transformers 4.46 compatibility ──
# transformers 4.46 renamed StaticCache.max_batch_size → batch_size,
# but parler-tts git HEAD still accesses .max_batch_size
try:
    from transformers.cache_utils import StaticCache
    if not hasattr(StaticCache, 'max_batch_size'):
        StaticCache.max_batch_size = property(lambda self: self.batch_size)
        logging.getLogger("tts").info("[patch] Added StaticCache.max_batch_size alias → batch_size")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tts")

# ─────────────────────────────────────────────────────────
# ENV CONFIG
# ─────────────────────────────────────────────────────────

HF_TOKEN       = os.getenv("HF_TOKEN")
TTS_MODEL_ID   = os.getenv("TTS_MODEL_ID", "ai4bharat/indic-parler-tts")
TTS_PORT       = int(os.getenv("TTS_PORT", "8003"))

ATTN_IMPL      = os.getenv("TTS_ATTN_IMPL", "auto")
COMPILE_MODE   = os.getenv("TTS_COMPILE_MODE", "default")
MAX_CONCURRENT = int(os.getenv("TTS_MAX_CONCURRENT", "1"))

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN must be set")

# ── Global CUDA optimizations ──
torch.set_grad_enabled(False)  # inference-only server, never need gradients
if os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────────────────
# MODEL STATE
# ─────────────────────────────────────────────────────────

_model = None
_tts_tokenizer = None
_desc_tokenizer = None
_sample_rate = None
_compiled = False
_use_cache = False
_has_static_cache = False

_sem = asyncio.Semaphore(MAX_CONCURRENT)

# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────

_desc_enc_cached = None


def _try_load_model(attn_impl, device, dtype):
    """Attempt to load the model with a specific attention implementation."""
    from parler_tts import ParlerTTSForConditionalGeneration
    return ParlerTTSForConditionalGeneration.from_pretrained(
        TTS_MODEL_ID,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        token=HF_TOKEN,
    ).to(device)


def _probe_use_cache(dummy_enc) -> bool:
    """Test whether use_cache=True works without crashing."""
    try:
        _model.generate(
            input_ids=_desc_enc_cached.input_ids,
            attention_mask=_desc_enc_cached.attention_mask,
            prompt_input_ids=dummy_enc.input_ids,
            prompt_attention_mask=dummy_enc.attention_mask,
            max_new_tokens=5,
            use_cache=True,
        )
        log.info("[TTS] ✓ use_cache=True probe PASSED — KV cache enabled")
        return True
    except Exception as e:
        log.warning("[TTS] ✗ use_cache=True probe FAILED: %s", e)
        log.warning("[TTS]   → Falling back to use_cache=False (slower)")
        return False


def load():
    global _model, _tts_tokenizer, _desc_tokenizer, _sample_rate
    global _compiled, _desc_enc_cached, _use_cache, _has_static_cache

    if _model is not None:
        return

    if not torch.cuda.is_available():
        log.error("CUDA NOT AVAILABLE — CHECK GPU SCHEDULING")
        sys.exit(1)

    device = "cuda"
    dtype = torch.float16

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"[TTS] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    log.info(f"[TTS] PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    # ── Attention backend selection ──
    # Priority: split (SDPA decoder + eager encoder) → global sdpa → eager
    # Split attention gives SDPA speedup on the decoder where generation
    # time is spent, while avoiding T5EncoderModel's lack of SDPA support.
    attn_strategies = []
    if ATTN_IMPL == "auto":
        attn_strategies = [
            ({"decoder": "sdpa", "text_encoder": "eager"}, "split (decoder=sdpa, encoder=eager)"),
            ("sdpa", "global sdpa"),
            ("flash_attention_2", "flash_attention_2"),
            ("eager", "eager"),
        ]
    elif ATTN_IMPL == "eager":
        attn_strategies = [("eager", "eager")]
    else:
        attn_strategies = [(ATTN_IMPL, ATTN_IMPL), ("eager", "eager")]

    chosen_attn_label = None
    for attn_config, label in attn_strategies:
        try:
            log.info(f"[TTS] Trying attention backend: {label}")
            _model = _try_load_model(attn_config, device, dtype)
            chosen_attn_label = label
            log.info(f"[TTS] ✓ {label} loaded successfully")
            break
        except Exception as e:
            log.warning(f"[TTS] ✗ {label} FAILED: {type(e).__name__}: {e}")
            _model = None

    if _model is None:
        log.error("[TTS] ALL attention backends failed — cannot start")
        sys.exit(1)

    _model.eval()

    _tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID, token=HF_TOKEN)
    _desc_tokenizer = AutoTokenizer.from_pretrained(
        _model.config.text_encoder._name_or_path,
        token=HF_TOKEN
    )
    _sample_rate = _model.config.sampling_rate

    # ── Pre-encode voice description ──
    desc = "Divya's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
    _desc_enc_cached = _desc_tokenizer(desc, return_tensors="pt").to(device)
    log.info("[TTS] Description pre-encoded (cached)")

    # ── Probe use_cache ──
    dummy_enc = _tts_tokenizer("test", return_tensors="pt").to(device)
    _use_cache = _probe_use_cache(dummy_enc)

    # ── Static cache (requires compatible parler-tts + transformers versions) ──
    try:
        _model.generation_config.cache_implementation = "static"
        # Probe it with a real generate call to catch API mismatches
        _model.generate(
            input_ids=_desc_enc_cached.input_ids,
            attention_mask=_desc_enc_cached.attention_mask,
            prompt_input_ids=dummy_enc.input_ids,
            prompt_attention_mask=dummy_enc.attention_mask,
            max_new_tokens=5,
            use_cache=True,
        )
        _has_static_cache = True
        log.info("[TTS] ✓ Static cache enabled and verified")
    except Exception as e:
        _model.generation_config.cache_implementation = None
        _has_static_cache = False
        log.warning(f"[TTS] ✗ Static cache FAILED: {type(e).__name__}: {e}")
        log.warning("[TTS]   → Using dynamic cache. Fix: upgrade transformers to match parler-tts git HEAD")

    # ── torch.compile ──
    # Disabled on V100: even with static cache, variable prompt lengths cause
    # recompilation that exceeds any speedup. Effective on Ampere+ (A10/A100/H100).
    if COMPILE_MODE != "none" and _has_static_cache:
        compute_cap = torch.cuda.get_device_capability(0)
        if compute_cap[0] >= 8:  # Ampere+
            try:
                log.info(f"[TTS] Compiling model.forward with mode={COMPILE_MODE}...")
                _model.forward = torch.compile(_model.forward, mode=COMPILE_MODE)
                _compiled = True
                log.info("[TTS] ✓ torch.compile succeeded")
            except Exception as e:
                log.warning(f"[TTS] ✗ torch.compile FAILED: {type(e).__name__}: {e}")
        else:
            log.info(f"[TTS] Skipping torch.compile — GPU compute capability {compute_cap[0]}.{compute_cap[1]} < 8.0 (recompilation penalty outweighs gains)")

    # ── Warmup ──
    log.info("[TTS] Running warmup inference...")
    warmup_runs = 2 if _compiled else 1
    try:
        for i in range(warmup_runs):
            t0 = time.time()
            _model.generate(
                input_ids=_desc_enc_cached.input_ids,
                attention_mask=_desc_enc_cached.attention_mask,
                prompt_input_ids=dummy_enc.input_ids,
                prompt_attention_mask=dummy_enc.attention_mask,
                max_new_tokens=10,
                use_cache=_use_cache,
            )
            log.info(f"[TTS] Warmup {i+1}/{warmup_runs} done ({time.time()-t0:.1f}s)")
        del dummy_enc
        torch.cuda.empty_cache()
    except Exception as e:
        log.warning(f"[TTS] Warmup failed ({e}), first request may be slow")

    log.info("=" * 60)
    log.info(f"[TTS] READY")
    log.info(f"[TTS]   attn={chosen_attn_label}")
    log.info(f"[TTS]   compiled={_compiled}, use_cache={_use_cache}, static_cache={_has_static_cache}")
    log.info(f"[TTS]   model={TTS_MODEL_ID}, sample_rate={_sample_rate}, dtype=float16")
    log.info("=" * 60)

# ─────────────────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BATCH_SENTENCES = 8

class Req(BaseModel):
    text: str
    batch: bool = False

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": "cuda",
        "compiled": _compiled,
        "use_cache": _use_cache,
        "static_cache": _has_static_cache,
    }

# ─────────────────────────────────────────────────────────
# CORE SYNTHESIS
# ─────────────────────────────────────────────────────────

def synth(text: str):
    txt_enc = _tts_tokenizer(text, return_tensors="pt").to(_desc_enc_cached.input_ids.device)

    # Hindi needs ~55 audio codec steps per word. No hard cap.
    n_words = len(text.split())
    max_tokens = max(500, n_words * 55)

    # Reset static cache before each generation to avoid stale state
    if _has_static_cache:
        try:
            del _model._cache
        except AttributeError:
            pass

    t0 = time.time()
    gen = _model.generate(
        input_ids=_desc_enc_cached.input_ids,
        attention_mask=_desc_enc_cached.attention_mask,
        prompt_input_ids=txt_enc.input_ids,
        prompt_attention_mask=txt_enc.attention_mask,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        use_cache=_use_cache,
    )

    elapsed = round(time.time() - t0, 2)
    tokens_generated = gen.shape[-1]
    audio = gen.cpu().numpy().squeeze().astype("float32")

    # Guard against 0-dim tensor from degenerate generation
    if audio.ndim == 0:
        log.warning(f"[synth] Degenerate output (0-dim) for: {text[:60]}")
        audio = np.zeros(int(_sample_rate * 0.1), dtype="float32")

    audio_secs = round(len(audio) / _sample_rate, 1) if _sample_rate else 0
    log.info(f"[synth] {n_words} words → {tokens_generated} tokens in {elapsed}s → {audio_secs}s audio")

    del gen, txt_enc
    torch.cuda.empty_cache()

    return audio


def trim_trailing_silence(audio, threshold=0.01, min_silence_samples=2000):
    """Trim trailing silence from audio, keeping a small tail for natural decay."""
    abs_audio = np.abs(audio)
    above = np.where(abs_audio > threshold)[0]
    if len(above) == 0:
        return audio
    last_loud = above[-1]
    end = min(last_loud + min_silence_samples, len(audio))
    return audio[:end]


def to_wav_b64(audio):
    buf = io.BytesIO()
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype('int16')
    sf.write(buf, audio_int16, _sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.post("/tts")
async def tts(req: Req):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    import re

    normalized = normalize_for_tts(req.text)

    if req.batch:
        sentences = [s.strip() for s in re.split(r'(?<=[।.!?])\s+', normalized.strip()) if s.strip()]
        if not sentences:
            sentences = [normalized]
        if len(sentences) > MAX_BATCH_SENTENCES:
            sentences = sentences[:MAX_BATCH_SENTENCES]

        async with _sem:
            t0 = time.time()
            parts: list[np.ndarray] = []
            for i, sent in enumerate(sentences):
                log.info("[batch] Sentence %d/%d: %s", i + 1, len(sentences), sent[:60])
                audio = trim_trailing_silence(synth(sent))
                parts.append(audio)
                if i < len(sentences) - 1:
                    parts.append(np.zeros(int(_sample_rate * 0.1), dtype="float32"))

            if not parts:
                raise HTTPException(500, "No audio generated")

            combined = np.concatenate(parts)
            elapsed = round(time.time() - t0, 2)
            return {
                "audio_base64": to_wav_b64(combined),
                "elapsed": elapsed,
                "sentence_count": len(sentences),
            }
    else:
        async with _sem:
            t0 = time.time()
            audio = trim_trailing_silence(synth(normalized))
            return {
                "audio_base64": to_wav_b64(audio),
                "elapsed": round(time.time() - t0, 2)
            }


# /v1 alias
@app.post("/v1/tts")
async def tts_v1(req: Req):
    return await tts(req)

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    if not args.no_preload:
        load()

    uvicorn.run(app, host="0.0.0.0", port=TTS_PORT, workers=1)
