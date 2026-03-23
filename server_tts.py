"""Indic Parler-TTS server — runs natively on Mac for MPS acceleration.

Optimizations applied (per HF inference guide):
  1. SDPA attention (1.4× speedup)
  2. float16 precision (MPS; bfloat16 on CUDA)
  3. torch.compile + static cache (up to 4.5× speedup)
  4. Correct description tokenizer (google/flan-t5-large)
  5. Streaming audio via ParlerTTSStreamer (first audio in <1s)

Start with:
    python server_tts.py --preload --port 8003
"""
import argparse
import asyncio
import base64
import io
import json
import re
import threading
import time

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSConfig
from transformers import AutoTokenizer
import uvicorn

from text_normalize import normalize_for_tts

TTS_MAX_CHARS = 1000  # overall hard cap across all sentences
CHUNK_TARGET_CHARS = 50   # ideal chunk size for clean audio
CHUNK_MAX_CHARS = 80      # hard cap before forced split
CHUNK_MIN_CHARS = 25      # merge chunks shorter than this with neighbour

TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
TTS_DESCRIPTION = (
    "A calm and clear female Hindi voice with very clear audio, "
    "speaking slowly and distinctly, suitable for government information delivery."
)

# torch.compile mode: "default" is safest on MPS.
# "reduce-overhead" uses CUDA graphs — CUDA only.
COMPILE_MODE = "default"

_model = None
_tts_tokenizer = None
_desc_tokenizer = None
_compiled = False


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load():
    global _model, _tts_tokenizer, _desc_tokenizer, _compiled
    if _model is not None:
        return

    device = _get_device()

    # Pick dtype: bfloat16 on CUDA, float32 on MPS/CPU.
    # float16 on MPS causes severe degradation — many ops silently fall back to CPU.
    if device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"[TTS] Loading {TTS_MODEL_ID} on {device} ({dtype})...")

    # SDPA on MPS causes progressive memory leak / degradation across sequential
    # generations. Disabled until PyTorch MPS SDPA support matures.
    # On CUDA, enable via: config.decoder._attn_implementation = "sdpa"
    _model = ParlerTTSForConditionalGeneration.from_pretrained(
        TTS_MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    _tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    _desc_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    print("[TTS] Tokenizers loaded (both from TTS model)")

    # torch.compile: disabled — without static cache (parler_tts/transformers version mismatch on
    # StaticCache API), the autoregressive loop hits recompile_limit on every generation step.
    # Re-enable when parler_tts updates to match transformers' StaticCache.batch_size API.
    _compiled = False

    print(f"[TTS] Ready on {device}. Compiled: {_compiled}")


def _warmup(device: str):
    """Two dummy generations to trigger torch.compile JIT compilation."""
    dummy_desc = _desc_tokenizer("A clear voice.", return_tensors="pt")
    dummy_text = _tts_tokenizer("Hello.", return_tensors="pt")
    for i in range(2):
        with torch.no_grad():
            _model.generate(
                input_ids=dummy_desc.input_ids.to(device),
                attention_mask=dummy_desc.attention_mask.to(device),
                prompt_input_ids=dummy_text.input_ids.to(device),
                prompt_attention_mask=dummy_text.attention_mask.to(device),
                max_new_tokens=50,
            )
        if str(device) == "mps":
            torch.mps.empty_cache()
        print(f"[TTS]   Warmup pass {i + 1}/2 done")


app = FastAPI(title="Vaani TTS Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sem = asyncio.Semaphore(1)  # one synthesis at a time


class TTSRequest(BaseModel):
    text: str
    description: str = TTS_DESCRIPTION


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "compiled": _compiled}


# ── Chunking ─────────────────────────────────────────────────────────

_CONJUNCTION_RE = re.compile(r"(?<=\s)(और|तथा|लेकिन|या|एवं|किन्तु|परन्तु)(?=\s)")


def _split_into_chunks(text: str) -> list[str]:
    """Split text into chunks of ~50 chars (max 80) for clean TTS synthesis."""
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
                            next_piece = conj_parts[i].strip() + " " + conj_parts[i + 1].strip()
                            tier2.append(next_piece)
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

    # Merge tiny chunks with neighbours to avoid per-chunk synthesis overhead
    merged: list[str] = []
    for c in chunks:
        if c:
            if merged and len(c) < CHUNK_MIN_CHARS:
                # Append to previous chunk if it still fits
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


# ── Audio processing ─────────────────────────────────────────────────

def _trim_silence(audio: np.ndarray, sr: int, threshold: float = 0.005, window_ms: int = 50) -> np.ndarray:
    if audio.ndim == 0 or audio.size < 2:
        return audio
    window_samples = int(sr * window_ms / 1000)
    if len(audio) < window_samples:
        return audio

    last_active = 0
    for i in range(0, len(audio) - window_samples, window_samples):
        chunk = audio[i:i + window_samples]
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms > threshold:
            last_active = i + window_samples

    end = min(last_active + int(sr * 0.2), len(audio))
    trimmed = audio[:end] if end > 0 else audio

    fade_samples = int(sr * 0.01)
    if len(trimmed) > fade_samples:
        fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        trimmed[-fade_samples:] *= fade

    return trimmed


def _trim_phantom_speech(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove codec 'phantom speech' — the gibberish regeneration that occurs
    after real speech ends when the model hits max_new_tokens without EOS.

    Pattern: real speech → silence gap (RMS < 0.02 for 200ms+) → phantom resurgence.
    We scan silence runs from the end backwards and trim at the boundary where
    a silence gap is followed by substantial audio (≥ 1s), indicating a resurgence.
    Two passes handle nested silence-resurgence patterns.
    """
    SILENCE_THRESHOLD = 0.02   # higher than _trim_silence to catch near-silence
    SPEECH_THRESHOLD  = 0.05   # "active" window
    WIN_MS = 100
    MIN_SILENCE_WIN = 2        # 200ms minimum silence run
    MIN_BEFORE_SPEECH = 2      # need 200ms of speech before the boundary
    MIN_AFTER_WIN = 5          # need 500ms of audio after boundary to bother trimming

    win = int(sr * WIN_MS / 1000)
    n = len(audio) // win
    if n < MIN_SILENCE_WIN + MIN_BEFORE_SPEECH + MIN_AFTER_WIN + 2:
        return audio

    rms = np.array([np.sqrt(np.mean(audio[i*win:(i+1)*win]**2)) for i in range(n)])

    # Collect all silence runs (start_idx, end_idx)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if rms[i] < SILENCE_THRESHOLD:
            j = i + 1
            while j < n and rms[j] < SILENCE_THRESHOLD:
                j += 1
            if j - i >= MIN_SILENCE_WIN:
                runs.append((i, j))
            i = j
        else:
            i += 1

    # Scan runs from end backwards — find last silence that is:
    #  1. After 45% of audio (not an internal mid-sentence pause near the start)
    #  2. Preceded by active speech
    #  3. Followed by substantial audio (indicating phantom resurgence worth trimming)
    for s_start, s_end in reversed(runs):
        if s_start / n < 0.45:          # too early — likely natural pause, not end of speech
            continue
        speech_before = sum(
            1 for k in range(max(0, s_start - 15), s_start)
            if rms[k] > SPEECH_THRESHOLD
        )
        after_win = n - s_end
        if speech_before >= MIN_BEFORE_SPEECH and after_win >= MIN_AFTER_WIN:
            end_sample = min((s_start + 1) * win + int(sr * 0.15), len(audio))
            trimmed = audio[:end_sample].copy()
            fade = int(sr * 0.05)
            if len(trimmed) > fade:
                trimmed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            print(f"[TTS]   Phantom trim: {len(audio)/sr:.2f}s → {len(trimmed)/sr:.2f}s "
                  f"(silence boundary at {s_start * WIN_MS / 1000:.1f}s)")
            return trimmed
    return audio


def _audio_quality_ok(audio: np.ndarray, sr: int, label: str = "") -> bool:
    duration = len(audio) / sr
    if duration < 0.15:
        print(f"[TTS]   {label} quality FAIL: too short ({duration:.2f}s)")
        return False
    window_samples = int(sr * 0.1)
    if len(audio) < window_samples:
        print(f"[TTS]   {label} quality FAIL: fewer samples than one window")
        return False
    silent_windows = 0
    total_windows = 0
    for i in range(0, len(audio) - window_samples, window_samples):
        chunk = audio[i:i + window_samples]
        rms = np.sqrt(np.mean(chunk ** 2))
        total_windows += 1
        if rms < 0.005:
            silent_windows += 1
    if total_windows == 0:
        print(f"[TTS]   {label} quality FAIL: zero windows")
        return False
    silent_ratio = silent_windows / total_windows
    ok = silent_ratio <= 0.85
    print(f"[TTS]   {label} quality {'OK' if ok else 'FAIL'}: {duration:.2f}s, "
          f"{silent_ratio:.0%} silent ({silent_windows}/{total_windows} windows)")
    return ok


# ── Synthesis ────────────────────────────────────────────────────────

MAX_AUDIO_TOKENS = 600
MAX_AUDIO_TOKENS_RETRY = 400

# Tokens/second the model generates (44100 Hz ÷ ~512 codec frame = ~86 tok/s)
_TOKENS_PER_SEC = 86
# Observed Hindi speech rate: ~34 tokens/word + 20% safety margin = 41
_TOKENS_PER_WORD = 41


def _max_tokens_for_text(text: str) -> int:
    """Estimate max_new_tokens based on word count so the model can't generate
    more than ~1.5× the expected speech duration. Caps phantom speech length."""
    words = len(text.split())
    estimated = max(150, words * _TOKENS_PER_WORD)
    return min(MAX_AUDIO_TOKENS, estimated)


def _synthesize_one(text: str, description: str, max_tokens: int | None = None) -> np.ndarray:
    if max_tokens is None:
        max_tokens = _max_tokens_for_text(text)
    """Synthesize a single short chunk and return the raw audio array."""
    device = next(_model.parameters()).device
    desc_enc = _desc_tokenizer(description, return_tensors="pt")
    prompt_enc = _tts_tokenizer(text, return_tensors="pt")

    desc_ids = desc_enc.input_ids.to(device)
    desc_mask = desc_enc.attention_mask.to(device)
    prompt_ids = prompt_enc.input_ids.to(device)
    prompt_mask = prompt_enc.attention_mask.to(device)

    with torch.no_grad():
        generation = _model.generate(
            input_ids=desc_ids,
            attention_mask=desc_mask,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=max_tokens,
        )

    audio_arr = generation.cpu().numpy().squeeze().astype("float32")
    del generation, desc_ids, desc_mask, prompt_ids, prompt_mask
    if str(device) == "mps":
        # MPS queues operations asynchronously — sync before clearing to ensure
        # all GPU work is complete, then empty cache to release allocations.
        torch.mps.synchronize()
        torch.mps.empty_cache()
        import gc; gc.collect()

    sr = _model.config.sampling_rate
    audio_arr = _trim_silence(audio_arr, sr)
    audio_arr = _trim_phantom_speech(audio_arr, sr)

    # Normalize to 85% of full scale so audio is audible regardless of model output level
    peak = np.max(np.abs(audio_arr))
    if peak > 1e-6:
        audio_arr = audio_arr / peak * 0.85

    return audio_arr


def _chunk_to_wav_b64(audio_arr: np.ndarray, sample_rate: int) -> str:
    """Encode a single audio array as base64 WAV."""
    buf = io.BytesIO()
    sf.write(buf, audio_arr, sample_rate, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Endpoints ────────────────────────────────────────────────────────

@app.post("/tts")
async def synthesize(req: TTSRequest):
    """Synchronous TTS — returns full audio when all chunks are done."""
    if _model is None:
        load()

    if not _sem._value:
        raise HTTPException(status_code=503, detail="TTS busy — try again shortly.")

    text = normalize_for_tts(req.text.strip())[:TTS_MAX_CHARS]
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")

    chunks = _split_into_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks to synthesize.")

    async with _sem:
        t0 = time.time()
        sample_rate = _model.config.sampling_rate
        silence = np.zeros(int(sample_rate * 0.3), dtype=np.float32)

        audio_parts = []
        for i, chunk in enumerate(chunks):
            label = f"chunk {i + 1}/{len(chunks)}"
            print(f"[TTS] Synthesizing {label}: {chunk[:60]}...")
            chunk_audio = _synthesize_one(chunk, req.description)

            if not _audio_quality_ok(chunk_audio, sample_rate, label):
                print(f"[TTS]   Retrying {label} with fewer tokens...")
                chunk_audio = _synthesize_one(chunk, req.description, max_tokens=MAX_AUDIO_TOKENS_RETRY)
                if not _audio_quality_ok(chunk_audio, sample_rate, f"{label} retry"):
                    print(f"[TTS]   Skipping {label}")
                    continue

            if audio_parts:
                audio_parts.append(silence)
            audio_parts.append(chunk_audio)

        elapsed = round(time.time() - t0, 2)

        if not audio_parts:
            raise HTTPException(status_code=500, detail="All chunks failed quality check.")

        audio_arr = np.concatenate(audio_parts)
        print(f"[TTS] Done: {len(chunks)} chunks, {elapsed}s, "
              f"{len(audio_arr) / sample_rate:.1f}s audio")

    audio_b64 = _chunk_to_wav_b64(audio_arr, sample_rate)
    return {"audio_base64": audio_b64, "elapsed_s": elapsed}


@app.post("/tts/stream")
async def synthesize_stream(req: TTSRequest):
    """Streaming TTS — sends each chunk's audio as an SSE event as soon as it's ready.

    SSE events:
      data: {"type":"chunk","index":0,"audio_base64":"...","sample_rate":44100}
      data: {"type":"chunk","index":1,"audio_base64":"...","sample_rate":44100}
      data: {"type":"done","elapsed_s":12.3,"total_chunks":3}
    """
    if _model is None:
        load()

    if not _sem._value:
        raise HTTPException(status_code=503, detail="TTS busy — try again shortly.")

    text = normalize_for_tts(req.text.strip())[:TTS_MAX_CHARS]
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")

    chunks = _split_into_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks to synthesize.")

    async def event_stream():
        async with _sem:
            t0 = time.time()
            sample_rate = _model.config.sampling_rate
            sent = 0

            for i, chunk in enumerate(chunks):
                label = f"chunk {i + 1}/{len(chunks)}"
                print(f"[TTS] Streaming {label}: {chunk[:60]}...")
                chunk_audio = _synthesize_one(chunk, req.description)

                if not _audio_quality_ok(chunk_audio, sample_rate, label):
                    print(f"[TTS]   Retrying {label} with fewer tokens...")
                    chunk_audio = _synthesize_one(chunk, req.description, max_tokens=MAX_AUDIO_TOKENS_RETRY)
                    if not _audio_quality_ok(chunk_audio, sample_rate, f"{label} retry"):
                        print(f"[TTS]   Skipping {label}")
                        continue

                audio_b64 = _chunk_to_wav_b64(chunk_audio, sample_rate)
                event = {"type": "chunk", "index": sent, "audio_base64": audio_b64, "sample_rate": sample_rate}
                yield f"data: {json.dumps(event)}\n\n"
                sent += 1

            elapsed = round(time.time() - t0, 2)
            print(f"[TTS] Stream done: {sent}/{len(chunks)} chunks, {elapsed}s")
            yield f"data: {json.dumps({'type': 'done', 'elapsed_s': elapsed, 'total_chunks': sent})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (useful for debugging)")
    args = parser.parse_args()

    if args.no_compile:
        COMPILE_MODE = None

    if args.preload:
        load()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
