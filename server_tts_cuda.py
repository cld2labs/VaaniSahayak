"""Indic Parler-TTS — CUDA-optimized server for Kubernetes / GPU deployment.

Optimizations applied (per HF INFERENCE.md):
  1. bfloat16 precision                          — halves VRAM vs float32
  2. SDPA attention      (attn_implementation)   — 1.4× speedup over eager
  3. flash_attention_2   (optional, via env)      — further speedup if FA2 installed
  4. torch.compile + static cache               — up to 4.5× speedup (reduce-overhead)
  5. ParlerTTSStreamer                            — true token-level streaming, first audio <500ms
  6. Warmup passes on startup                    — JIT compile before first real request
  7. Adaptive token budget                       — prevents phantom speech

Environment variables:
  HF_TOKEN              : HuggingFace token (required for gated / rate-limited models)
  TTS_MODEL_ID          : model ID (default: ai4bharat/indic-parler-tts)
  TTS_PORT              : server port (default: 8003)
  TTS_COMPILE_MODE      : torch.compile mode — "reduce-overhead" (default) | "default" | "none"
  TTS_ATTN_IMPL         : attention backend — "sdpa" (default) | "flash_attention_2" | "eager"
  TTS_MAX_CONCURRENT    : max concurrent synthesis requests (default: 1)
  TTS_WORKERS           : uvicorn worker count (default: 1 — keep at 1 per GPU)

API (same contract as server_tts.py so tts_model.py works unchanged):
  POST /tts             → blocking, returns {audio_base64, elapsed_s}
  POST /tts/stream      → SSE streaming, chunks: {type, index, audio_base64, sample_rate}
  GET  /health          → {status, model_loaded, compiled, device, dtype}

Start:
  python server_tts_cuda.py
  python server_tts_cuda.py --port 8003 --compile-mode reduce-overhead
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import time
from threading import Thread

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tts-cuda")

# ── Config from environment ────────────────────────────────────────────────────

HF_TOKEN        = os.getenv("HF_TOKEN", "")
TTS_MODEL_ID    = os.getenv("TTS_MODEL_ID", "ai4bharat/indic-parler-tts")
TTS_PORT        = int(os.getenv("TTS_PORT", "8003"))
COMPILE_MODE    = os.getenv("TTS_COMPILE_MODE", "reduce-overhead")   # or "default" | "none"
ATTN_IMPL       = os.getenv("TTS_ATTN_IMPL", "sdpa")                 # or "flash_attention_2"
MAX_CONCURRENT  = int(os.getenv("TTS_MAX_CONCURRENT", "1"))

# Audio chunking
TTS_MAX_CHARS   = 1000
CHUNK_MAX_CHARS = 80
CHUNK_MIN_CHARS = 25

# Token budget (hindi speech: ~41 tokens/word, 86 tokens/sec codec rate)
_TOKENS_PER_WORD = 41
MAX_AUDIO_TOKENS = 600
MAX_AUDIO_TOKENS_RETRY = 400

# Voice description for Hindi government information
TTS_DESCRIPTION_HI = (
    "Divya speaks with a calm, clear voice at a moderate pace. "
    "The recording is of very high quality, with very clear audio and no background noise. "
    "Her voice sounds close up and is suitable for government information delivery."
)
TTS_DESCRIPTION_TE = (
    "A calm and clear female Telugu voice with very clear audio, "
    "speaking slowly and distinctly, suitable for government information delivery."
)
TTS_DESCRIPTIONS = {"hi": TTS_DESCRIPTION_HI, "te": TTS_DESCRIPTION_TE}

# ── Model state ───────────────────────────────────────────────────────────────

_model          = None
_tts_tokenizer  = None
_desc_tokenizer = None
_compiled       = False
_sample_rate    = None
_frame_rate     = None   # for ParlerTTSStreamer


def _authenticate_hf():
    """Log in to HuggingFace Hub if HF_TOKEN is set."""
    if not HF_TOKEN:
        log.warning("[HF] HF_TOKEN not set — using anonymous access (may fail for private models)")
        return
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        log.info("[HF] Authenticated with HuggingFace Hub")
    except Exception as e:
        log.error("[HF] Authentication failed: %s", e)
        sys.exit(1)


def load():
    """Load model, tokenizers, optionally compile, then warm up."""
    global _model, _tts_tokenizer, _desc_tokenizer, _compiled, _sample_rate, _frame_rate

    if _model is not None:
        return

    _authenticate_hf()

    if not torch.cuda.is_available():
        log.error("[TTS] CUDA not available. This server requires a GPU.")
        sys.exit(1)

    device = "cuda"
    dtype  = torch.bfloat16   # bfloat16: half VRAM, stable numerics on CUDA

    log.info("[TTS] Loading %s on %s (%s, attn=%s)", TTS_MODEL_ID, device, dtype, ATTN_IMPL)

    # Lazy imports so startup error messages are clear
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    # ── Load model ──────────────────────────────────────────────────────────
    _model = ParlerTTSForConditionalGeneration.from_pretrained(
        TTS_MODEL_ID,
        torch_dtype=dtype,
        attn_implementation=ATTN_IMPL,    # "sdpa" (default) or "flash_attention_2"
        token=HF_TOKEN or None,
    ).to(device)
    _model.eval()

    # ── Tokenizers ──────────────────────────────────────────────────────────
    # Prompt tokenizer: the TTS model's own tokenizer
    _tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID, token=HF_TOKEN or None)

    # Description tokenizer: must come from the text encoder (google/flan-t5-large)
    # Per HF docs: AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    _desc_tokenizer = AutoTokenizer.from_pretrained(
        _model.config.text_encoder._name_or_path,
        token=HF_TOKEN or None,
    )
    log.info("[TTS] Tokenizers loaded (desc: %s)", _model.config.text_encoder._name_or_path)

    _sample_rate = _model.config.sampling_rate
    # codec downsampling factor for ParlerTTSStreamer
    _frame_rate  = _model.audio_encoder.config.frame_rate

    # ── torch.compile ───────────────────────────────────────────────────────
    # "reduce-overhead" uses CUDA graphs → 3-4× speedup, requires static shapes.
    # Requires transformers static cache to be working (transformers >= 4.41).
    if COMPILE_MODE and COMPILE_MODE != "none":
        try:
            _model.generation_config.cache_implementation = "static"
            _model.forward = torch.compile(_model.forward, mode=COMPILE_MODE)
            _compiled = True
            log.info("[TTS] torch.compile enabled (mode=%s)", COMPILE_MODE)
        except Exception as e:
            log.warning("[TTS] torch.compile failed (%s) — continuing without it", e)
            _compiled = False
    else:
        log.info("[TTS] torch.compile disabled")

    # ── Warmup ──────────────────────────────────────────────────────────────
    _warmup(device)

    log.info("[TTS] Ready on %s | compiled=%s | sample_rate=%s", device, _compiled, _sample_rate)


def _warmup(device: str):
    """Trigger JIT compilation with dummy inputs (2 passes for reduce-overhead)."""
    log.info("[TTS] Running warmup passes...")
    n_steps = 2 if _compiled and COMPILE_MODE == "reduce-overhead" else 1
    dummy_desc = _desc_tokenizer(
        "A clear voice.", return_tensors="pt", padding="max_length", max_length=20
    ).to(device)
    dummy_text = _tts_tokenizer(
        "Hello.", return_tensors="pt", padding="max_length", max_length=10
    ).to(device)
    for i in range(n_steps):
        with torch.no_grad():
            _model.generate(
                input_ids=dummy_desc.input_ids,
                attention_mask=dummy_desc.attention_mask,
                prompt_input_ids=dummy_text.input_ids,
                prompt_attention_mask=dummy_text.attention_mask,
                max_new_tokens=30,
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log.info("[TTS]   Warmup pass %d/%d done", i + 1, n_steps)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Vaani TTS Server (CUDA)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sem = asyncio.Semaphore(MAX_CONCURRENT)


class TTSRequest(BaseModel):
    text: str
    description: str = TTS_DESCRIPTION_HI
    language: str = "hi"   # "hi" | "te" | etc.


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "compiled": _compiled,
        "compile_mode": COMPILE_MODE,
        "attn_impl": ATTN_IMPL,
        "device": "cuda" if torch.cuda.is_available() else "unavailable",
        "dtype": "bfloat16",
        "model_id": TTS_MODEL_ID,
    }


# ── Text chunking ─────────────────────────────────────────────────────────────

_CONJUNCTION_RE = re.compile(r"(?<=\s)(और|तथा|लेकिन|या|एवं|किन्तु|परन्तु)(?=\s)")


def _split_into_chunks(text: str) -> list[str]:
    """Split text into chunks ≤ CHUNK_MAX_CHARS for clean TTS synthesis."""
    raw_sentences = re.split(r"(?<=[।.!?])\s+", text.strip())
    chunks: list[str] = []

    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= CHUNK_MAX_CHARS:
            chunks.append(sent)
            continue
        sub_parts = re.split(r"(?<=,)\s+(?!\d)", sent)
        tier2: list[str] = []
        for part in sub_parts:
            if len(part) > CHUNK_MAX_CHARS:
                conj_parts = _CONJUNCTION_RE.split(part)
                i = 0
                while i < len(conj_parts):
                    piece = conj_parts[i].strip()
                    if i + 1 < len(conj_parts) and conj_parts[i + 1].strip() in (
                        "और", "तथा", "लेकिन", "या", "एवं", "किन्तु", "परन्तु"
                    ):
                        if piece:
                            tier2.append(piece)
                        i += 1
                        if i + 1 < len(conj_parts):
                            tier2.append(conj_parts[i].strip() + " " + conj_parts[i + 1].strip())
                            i += 2
                        else:
                            if conj_parts[i].strip():
                                tier2.append(conj_parts[i].strip())
                            i += 1
                    else:
                        if piece:
                            tier2.append(piece)
                        i += 1
            else:
                tier2.append(part.strip())
        for part in tier2:
            if len(part) <= CHUNK_MAX_CHARS:
                chunks.append(part)
            else:
                _force_split(part, chunks)

    # Merge tiny chunks into neighbour
    merged: list[str] = []
    for c in chunks:
        if c:
            if merged and len(c) < CHUNK_MIN_CHARS:
                candidate = merged[-1] + " " + c
                if len(candidate) <= CHUNK_MAX_CHARS:
                    merged[-1] = candidate
                    continue
            merged.append(c)
    return merged


def _force_split(text: str, out: list[str]) -> None:
    while len(text) > CHUNK_MAX_CHARS:
        idx = text.rfind(" ", 0, CHUNK_MAX_CHARS)
        if idx == -1:
            idx = CHUNK_MAX_CHARS
        out.append(text[:idx].strip())
        text = text[idx:].strip()
    if text:
        out.append(text)


# ── Audio post-processing ─────────────────────────────────────────────────────

def _trim_silence(audio: np.ndarray, sr: int, threshold: float = 0.005, window_ms: int = 50) -> np.ndarray:
    if audio.ndim == 0 or audio.size < 2:
        return audio
    win = int(sr * window_ms / 1000)
    if len(audio) < win:
        return audio
    last_active = 0
    for i in range(0, len(audio) - win, win):
        if np.sqrt(np.mean(audio[i:i + win] ** 2)) > threshold:
            last_active = i + win
    end = min(last_active + int(sr * 0.2), len(audio))
    trimmed = audio[:end] if end > 0 else audio
    fade = int(sr * 0.01)
    if len(trimmed) > fade:
        trimmed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return trimmed


def _trim_phantom_speech(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove codec phantom speech that appears after real speech ends."""
    SILENCE_TH, SPEECH_TH, WIN_MS = 0.02, 0.05, 100
    MIN_SIL, MIN_BEFORE, MIN_AFTER = 2, 2, 5
    win = int(sr * WIN_MS / 1000)
    n = len(audio) // win
    if n < MIN_SIL + MIN_BEFORE + MIN_AFTER + 2:
        return audio
    rms = np.array([np.sqrt(np.mean(audio[i * win:(i + 1) * win] ** 2)) for i in range(n)])
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if rms[i] < SILENCE_TH:
            j = i + 1
            while j < n and rms[j] < SILENCE_TH:
                j += 1
            if j - i >= MIN_SIL:
                runs.append((i, j))
            i = j
        else:
            i += 1
    for s_start, s_end in reversed(runs):
        if s_start / n < 0.45:
            continue
        speech_before = sum(1 for k in range(max(0, s_start - 15), s_start) if rms[k] > SPEECH_TH)
        if speech_before >= MIN_BEFORE and n - s_end >= MIN_AFTER:
            end_sample = min((s_start + 1) * win + int(sr * 0.15), len(audio))
            trimmed = audio[:end_sample].copy()
            fade = int(sr * 0.05)
            if len(trimmed) > fade:
                trimmed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            log.info("[TTS]   Phantom trim: %.2fs → %.2fs", len(audio) / sr, len(trimmed) / sr)
            return trimmed
    return audio


def _audio_quality_ok(audio: np.ndarray, sr: int, label: str = "") -> bool:
    if len(audio) / sr < 0.15:
        log.info("[TTS]   %s FAIL: too short", label)
        return False
    win = int(sr * 0.1)
    if len(audio) < win:
        return False
    silent = sum(1 for i in range(0, len(audio) - win, win) if np.sqrt(np.mean(audio[i:i + win] ** 2)) < 0.005)
    total  = max(1, (len(audio) - win) // win)
    ok = silent / total <= 0.85
    log.info("[TTS]   %s quality %s: %.2fs, %.0f%% silent", label, "OK" if ok else "FAIL", len(audio) / sr, 100 * silent / total)
    return ok


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak normalize to 85% full scale."""
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.85
    return audio


def _to_wav_b64(audio: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Synthesis — blocking (chunk-by-chunk) ─────────────────────────────────────

def _max_tokens(text: str) -> int:
    return min(MAX_AUDIO_TOKENS, max(150, len(text.split()) * _TOKENS_PER_WORD))


def _synthesize_chunk(text: str, description: str, max_tokens: int | None = None) -> np.ndarray:
    """Synthesize one short chunk; returns float32 audio array."""
    if max_tokens is None:
        max_tokens = _max_tokens(text)
    device = next(_model.parameters()).device

    desc_enc   = _desc_tokenizer(description, return_tensors="pt").to(device)
    prompt_enc = _tts_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = _model.generate(
            input_ids=desc_enc.input_ids,
            attention_mask=desc_enc.attention_mask,
            prompt_input_ids=prompt_enc.input_ids,
            prompt_attention_mask=prompt_enc.attention_mask,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=max_tokens,
        )

    audio = gen.cpu().numpy().squeeze().astype("float32")
    del gen, desc_enc, prompt_enc
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    sr = _sample_rate
    audio = _trim_silence(audio, sr)
    audio = _trim_phantom_speech(audio, sr)
    audio = _normalize_audio(audio)
    return audio


# ── Synthesis — streaming via ParlerTTSStreamer ───────────────────────────────

def _stream_chunks_via_streamer(text: str, description: str, play_steps_s: float = 0.5):
    """Use ParlerTTSStreamer for true token-level streaming → first audio <500ms.

    Yields raw numpy float32 arrays as they become available.
    """
    from parler_tts import ParlerTTSStreamer

    device = next(_model.parameters()).device
    play_steps = int(_frame_rate * play_steps_s)

    streamer = ParlerTTSStreamer(_model, device=device, play_steps=play_steps)

    desc_enc   = _desc_tokenizer(description, return_tensors="pt").to(device)
    prompt_enc = _tts_tokenizer(text, return_tensors="pt").to(device)

    gen_kwargs = dict(
        input_ids=desc_enc.input_ids,
        attention_mask=desc_enc.attention_mask,
        prompt_input_ids=prompt_enc.input_ids,
        prompt_attention_mask=prompt_enc.attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=0.7,
        min_new_tokens=10,
    )

    thread = Thread(target=_model.generate, kwargs=gen_kwargs)
    thread.start()

    for audio_chunk in streamer:
        if audio_chunk.shape[0] == 0:
            break
        chunk = audio_chunk.astype("float32")
        chunk = _normalize_audio(chunk)
        yield chunk

    thread.join()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/tts")
async def synthesize(req: TTSRequest):
    """Blocking TTS — returns full audio once all chunks are synthesized."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not _sem._value:
        raise HTTPException(status_code=503, detail="TTS busy — try again shortly.")

    description = TTS_DESCRIPTIONS.get(req.language, TTS_DESCRIPTION_HI)
    if req.description != TTS_DESCRIPTION_HI:
        description = req.description   # caller override

    try:
        from text_normalize import normalize_for_tts
        text = normalize_for_tts(req.text.strip())[:TTS_MAX_CHARS]
    except ImportError:
        text = req.text.strip()[:TTS_MAX_CHARS]

    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")

    chunks = _split_into_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No synthesizable chunks.")

    async with _sem:
        t0      = time.time()
        sr      = _sample_rate
        silence = np.zeros(int(sr * 0.3), dtype=np.float32)
        parts: list[np.ndarray] = []

        for i, chunk in enumerate(chunks):
            label = f"chunk {i + 1}/{len(chunks)}"
            log.info("[TTS] %s: %s", label, chunk[:60])
            audio = _synthesize_chunk(chunk, description)

            if not _audio_quality_ok(audio, sr, label):
                log.info("[TTS]   Retrying %s with fewer tokens", label)
                audio = _synthesize_chunk(chunk, description, max_tokens=MAX_AUDIO_TOKENS_RETRY)
                if not _audio_quality_ok(audio, sr, f"{label} retry"):
                    log.warning("[TTS]   Skipping %s", label)
                    continue

            if parts:
                parts.append(silence)
            parts.append(audio)

        elapsed = round(time.time() - t0, 2)

    if not parts:
        raise HTTPException(status_code=500, detail="All chunks failed quality check.")

    audio_arr = np.concatenate(parts)
    log.info("[TTS] Done: %d chunks, %.2fs, %.1fs audio", len(chunks), elapsed, len(audio_arr) / sr)
    return {"audio_base64": _to_wav_b64(audio_arr, sr), "elapsed_s": elapsed}


@app.post("/tts/stream")
async def synthesize_stream(req: TTSRequest):
    """Streaming TTS via ParlerTTSStreamer — first audio chunk in <500ms on GPU.

    SSE events:
      data: {"type":"chunk","index":0,"audio_base64":"...","sample_rate":44100}
      data: {"type":"done","elapsed_s":1.4,"total_chunks":6}
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not _sem._value:
        raise HTTPException(status_code=503, detail="TTS busy — try again shortly.")

    description = TTS_DESCRIPTIONS.get(req.language, TTS_DESCRIPTION_HI)
    if req.description != TTS_DESCRIPTION_HI:
        description = req.description

    try:
        from text_normalize import normalize_for_tts
        text = normalize_for_tts(req.text.strip())[:TTS_MAX_CHARS]
    except ImportError:
        text = req.text.strip()[:TTS_MAX_CHARS]

    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")

    chunks = _split_into_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No synthesizable chunks.")

    async def event_stream():
        async with _sem:
            t0   = time.time()
            sr   = _sample_rate
            sent = 0

            for i, chunk in enumerate(chunks):
                log.info("[TTS] Streaming chunk %d/%d: %s", i + 1, len(chunks), chunk[:60])
                # Use ParlerTTSStreamer for each chunk → true low-latency streaming
                stream_parts: list[np.ndarray] = []
                for audio_piece in _stream_chunks_via_streamer(chunk, description):
                    stream_parts.append(audio_piece)
                    # Emit each streamer sub-chunk immediately
                    event = {
                        "type": "chunk",
                        "index": sent,
                        "audio_base64": _to_wav_b64(audio_piece, sr),
                        "sample_rate": sr,
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                    sent += 1

            elapsed = round(time.time() - t0, 2)
            log.info("[TTS] Stream done: %d pieces, %.2fs", sent, elapsed)
            yield f"data: {json.dumps({'type': 'done', 'elapsed_s': elapsed, 'total_chunks': sent})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indic Parler-TTS CUDA server")
    parser.add_argument("--port", type=int, default=TTS_PORT)
    parser.add_argument("--compile-mode", default=COMPILE_MODE,
                        help="torch.compile mode: reduce-overhead | default | none")
    parser.add_argument("--attn-impl", default=ATTN_IMPL,
                        help="Attention backend: sdpa | flash_attention_2 | eager")
    parser.add_argument("--no-preload", action="store_true",
                        help="Skip model load at startup (lazy load on first request)")
    args = parser.parse_args()

    COMPILE_MODE = args.compile_mode
    ATTN_IMPL    = args.attn_impl

    if not args.no_preload:
        load()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
        workers=1,     # always 1 worker per GPU to avoid model duplication
    )
