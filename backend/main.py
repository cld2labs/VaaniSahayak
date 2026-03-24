"""Vaani Sahayak — FastAPI backend."""
import json
import queue
import re
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import CORS_ORIGINS
from backend.retrieval.embeddings import load_schemes_and_embeddings, get_schemes
from backend.retrieval.retriever import retrieve
from backend.models.param_model import load_model, build_prompt, generate, stream_generate, suspend_llm, resume_llm
from backend.models.tts_model import load_tts, text_to_audio_base64, stream_audio_chunks, synth_sentence


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load scheme data (non-fatal — run download/precompute scripts to populate)
    try:
        load_schemes_and_embeddings()
    except FileNotFoundError as e:
        print(f"[Startup] WARNING: {e}")
        print("[Startup] Run inside container: python scripts/download_data.py && python scripts/precompute_embeddings.py")
    load_model()   # warms up the OpenAI client + checks param1-server reachable
    try:
        load_tts()     # in-process — vLLM doesn't support TTS models
    except Exception as e:
        print(f"[Startup] WARNING: TTS failed to load — {e}")
        print("[Startup] Set HF_TOKEN env var and ensure access to ai4bharat/indic-parler-tts")
        print("[Startup] /ask endpoint will work but audio will not be generated.")
    yield


app = FastAPI(
    title="Vaani Sahayak API",
    description="Sovereign AI-powered Hindi voice assistant for Indian government schemes",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str
    language: str = "hi"  # "hi" or "en"
    tts: bool = True       # Whether to generate audio


class SchemeSnippet(BaseModel):
    name: str
    category: str | None
    state: str | None
    official_link: str | None
    score: float


class AskResponse(BaseModel):
    text: str
    audio_base64: str | None
    schemes: list[SchemeSnippet]
    latency_ms: int


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Patterns that indicate the model is echoing prompt structure
# Keep in sync with STOP_PATTERNS in server_param1.py
_STOP_PATS = [
    "<|im_start|>", "<|im_end|>",
    "[ASSISTANT", "[USER", "[SYSTEM",
    "[QUESTION", "[ANSWER",
    "\n\n\n",
    "नागरिक का प्रश्न", "పౌరుని ప్రశ్న",
    "\n--- Scheme",
]

# Sentence-ending delimiters — negative lookaround avoids splitting decimals like "1.5"
_SENTENCE_RE = re.compile(r'(?<!\d)[।.!?](?!\d)')


def _safe_boundary(full_text: str, sent_up_to: int) -> int:
    """Return the furthest index into full_text we can safely send.

    Only inspects unsent text (from sent_up_to onwards) so old text
    doesn't cause unnecessary hold-backs.
    """
    unsent = full_text[sent_up_to:]
    for n in range(min(len(unsent), 20), 0, -1):
        tail = unsent[-n:]
        if any(pat.startswith(tail) for pat in _STOP_PATS):
            return sent_up_to + len(unsent) - n
    return len(full_text)


def _check_stop(full_text: str, sent_up_to: int) -> int:
    """Return index of earliest stop pattern, or -1 if none found."""
    stop_idx = -1
    for pat in _STOP_PATS:
        idx = full_text.find(pat, max(0, sent_up_to - len(pat)))
        if idx != -1:
            stop_idx = idx if stop_idx == -1 else min(stop_idx, idx)
    return stop_idx


def _make_scheme_snippets(schemes: list[dict]) -> list[dict]:
    return [
        {
            "name": s.get("name", ""),
            "category": s.get("category"),
            "state": s.get("state"),
            "official_link": s.get("official_link"),
            "score": round(s.get("_score", 0.0), 3),
        }
        for s in schemes
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    schemes = get_schemes()
    return {
        "status": "ok",
        "schemes_loaded": len(schemes),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    t0 = time.time()

    # 1. Retrieve relevant schemes
    schemes = retrieve(req.query)

    # 2. Build prompt + generate text
    prompt = build_prompt(req.query, schemes, language=req.language)
    answer_text = generate(prompt)

    # 3. TTS (optional)
    audio_b64 = None
    if req.tts and answer_text:
        audio_b64 = text_to_audio_base64(answer_text, language=req.language)

    latency = int((time.time() - t0) * 1000)

    return AskResponse(
        text=answer_text,
        audio_base64=audio_b64,
        schemes=[
            SchemeSnippet(
                name=s.get("name", ""),
                category=s.get("category"),
                state=s.get("state"),
                official_link=s.get("official_link"),
                score=round(s.get("_score", 0.0), 3),
            )
            for s in schemes
        ],
        latency_ms=latency,
    )


@app.get("/schemes")
def list_schemes(page: int = 1, page_size: int = 20, category: str | None = None):
    schemes = get_schemes()
    if category:
        schemes = [s for s in schemes if s.get("category", "").lower() == category.lower()]
    total = len(schemes)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "schemes": schemes[start:end],
    }


@app.get("/schemes/suggestions")
def scheme_suggestions(count: int = 30, language: str = "hi"):
    """Return auto-generated Hindi queries from a random subset of schemes."""
    import random

    schemes = get_schemes()
    if not schemes:
        return {"suggestions": []}

    templates_hi = [
        "{name} योजना क्या है?",
        "{name} के लिए कौन पात्र है?",
        "{name} में क्या लाभ मिलते हैं?",
        "{name} के लिए आवेदन कैसे करें?",
        "{name} में कौन से दस्तावेज़ चाहिए?",
    ]
    templates_te = [
        "{name} పథకం ఏమిటి?",
        "{name} కి అర్హత ఎవరికి ఉంది?",
        "{name} లో ఏమేమి లాభాలు ఉన్నాయి?",
        "{name} కోసం ఎలా దరఖాస్తు చేయాలి?",
    ]
    templates = templates_te if language == "te" else templates_hi

    sample_size = min(count, len(schemes))
    sampled = random.sample(schemes, sample_size)

    suggestions = []
    for scheme in sampled:
        name = scheme.get("name", "").strip()
        if not name:
            continue
        tmpl = random.choice(templates)
        suggestions.append({
            "query": tmpl.format(name=name),
            "scheme_name": name,
        })

    return {"suggestions": suggestions}


@app.get("/schemes/{scheme_id}")
def get_scheme(scheme_id: int):
    schemes = get_schemes()
    if scheme_id < 0 or scheme_id >= len(schemes):
        raise HTTPException(status_code=404, detail="Scheme not found.")
    return schemes[scheme_id]


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """Streaming endpoint — SSE events:
      data: {"type":"token","text":"..."}
      data: {"type":"done","schemes":[...],"latency_ms":...}
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    t0 = time.time()
    schemes = retrieve(req.query)
    prompt = build_prompt(req.query, schemes, language=req.language)
    scheme_snippets = _make_scheme_snippets(schemes)

    def event_stream():
        full_text = ""
        sent_up_to = 0

        try:
            for chunk in stream_generate(prompt):
                full_text += chunk.replace("\ufffd", "")

                stop_idx = _check_stop(full_text, sent_up_to)
                if stop_idx != -1:
                    safe = full_text[sent_up_to:stop_idx]
                    if safe:
                        yield f"data: {json.dumps({'type': 'token', 'text': safe}, ensure_ascii=False)}\n\n"
                    break

                boundary = _safe_boundary(full_text, sent_up_to)
                new_text = full_text[sent_up_to:boundary]
                if new_text:
                    yield f"data: {json.dumps({'type': 'token', 'text': new_text}, ensure_ascii=False)}\n\n"
                    sent_up_to = boundary
            else:
                remaining = full_text[sent_up_to:]
                if remaining:
                    yield f"data: {json.dumps({'type': 'token', 'text': remaining}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'scope': 'llm', 'detail': str(exc)}, ensure_ascii=False)}\n\n"

        latency = int((time.time() - t0) * 1000)
        yield f"data: {json.dumps({'type': 'done', 'schemes': scheme_snippets, 'latency_ms': latency}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ask/speak")
def ask_speak(req: AskRequest):
    """Interleaved LLM + TTS — single SSE stream.

    SSE events:
      data: {"type":"token",    "text":"..."}
      data: {"type":"sentence", "index":0, "text":"..."}
      data: {"type":"audio",    "index":0, "audio_base64":"...", "sample_rate":44100}
      data: {"type":"done",     "schemes":[...], "latency_ms":...}
      data: {"type":"error",    "scope":"tts", "detail":"...", "sentence_index":0}
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    t0 = time.time()
    schemes = retrieve(req.query)
    prompt = build_prompt(req.query, schemes, language=req.language)
    scheme_snippets = _make_scheme_snippets(schemes)
    lang = req.language

    SENTINEL = object()

    def event_stream():
        # Queues for producer-consumer between main thread and TTS thread
        sentence_q: queue.Queue = queue.Queue()
        audio_q: queue.Queue = queue.Queue()

        def tts_worker():
            """Background thread: synthesize sentences as they arrive."""
            while True:
                item = sentence_q.get()
                if item is SENTINEL:
                    audio_q.put(SENTINEL)
                    break
                idx, text = item
                try:
                    b64 = synth_sentence(text, language=lang)
                    audio_q.put((idx, b64, None))
                except Exception as e:
                    audio_q.put((idx, None, str(e)))

        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

        full_text = ""
        sent_up_to = 0
        sentence_index = 0
        last_sentence_end = 0
        seen_sentences: set[str] = set()

        def detect_sentences(up_to: int):
            """Find complete sentences, skipping duplicates."""
            nonlocal sentence_index, last_sentence_end
            found = []
            segment = full_text[last_sentence_end:up_to]
            for m in _SENTENCE_RE.finditer(segment):
                end_pos = last_sentence_end + m.end()
                sentence_text = full_text[last_sentence_end:end_pos].strip()
                last_sentence_end = end_pos
                if not sentence_text or len(sentence_text) <= 5:
                    continue
                if sentence_text in seen_sentences:
                    continue
                seen_sentences.add(sentence_text)
                found.append((sentence_index, sentence_text))
                sentence_index += 1
            return found

        def drain_audio():
            """Non-blocking drain — yield audio SSE events as they complete."""
            events = []
            while True:
                try:
                    item = audio_q.get_nowait()
                except queue.Empty:
                    break
                if item is SENTINEL:
                    events.append(SENTINEL)
                    break
                idx, b64, err = item
                if err:
                    events.append(f"data: {json.dumps({'type': 'error', 'scope': 'tts', 'detail': err, 'sentence_index': idx}, ensure_ascii=False)}\n\n")
                elif b64:
                    events.append(f"data: {json.dumps({'type': 'audio', 'index': idx, 'audio_base64': b64, 'sample_rate': 44100}, ensure_ascii=False)}\n\n")
            return events

        # ── Stream LLM tokens, queue sentences for TTS ──
        stopped = False
        try:
            for chunk in stream_generate(prompt):
                full_text += chunk.replace("\ufffd", "")

                stop_idx = _check_stop(full_text, sent_up_to)
                if stop_idx != -1:
                    safe = full_text[sent_up_to:stop_idx]
                    if safe:
                        yield f"data: {json.dumps({'type': 'token', 'text': safe}, ensure_ascii=False)}\n\n"
                    for si, st in detect_sentences(stop_idx):
                        yield f"data: {json.dumps({'type': 'sentence', 'index': si, 'text': st}, ensure_ascii=False)}\n\n"
                        sentence_q.put((si, st))
                    stopped = True
                    break

                boundary = _safe_boundary(full_text, sent_up_to)
                new_text = full_text[sent_up_to:boundary]
                if new_text:
                    yield f"data: {json.dumps({'type': 'token', 'text': new_text}, ensure_ascii=False)}\n\n"
                    sent_up_to = boundary

                for si, st in detect_sentences(sent_up_to):
                    yield f"data: {json.dumps({'type': 'sentence', 'index': si, 'text': st}, ensure_ascii=False)}\n\n"
                    sentence_q.put((si, st))

                # Stream any completed audio chunks during LLM generation
                for evt in drain_audio():
                    if evt is SENTINEL:
                        break
                    yield evt
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'scope': 'llm', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
            stopped = True

        if not stopped:
            remaining = full_text[sent_up_to:]
            if remaining:
                yield f"data: {json.dumps({'type': 'token', 'text': remaining}, ensure_ascii=False)}\n\n"
            trailing = full_text[last_sentence_end:].strip()
            if trailing and len(trailing) > 5:
                yield f"data: {json.dumps({'type': 'sentence', 'index': sentence_index, 'text': trailing}, ensure_ascii=False)}\n\n"
                sentence_q.put((sentence_index, trailing))

        # Signal TTS thread: no more sentences
        sentence_q.put(SENTINEL)

        # ── Stream remaining audio chunks as they complete ──
        while True:
            try:
                item = audio_q.get(timeout=120)
            except queue.Empty:
                break
            if item is SENTINEL:
                break
            idx, b64, err = item
            if err:
                yield f"data: {json.dumps({'type': 'error', 'scope': 'tts', 'detail': err, 'sentence_index': idx}, ensure_ascii=False)}\n\n"
            elif b64:
                yield f"data: {json.dumps({'type': 'audio', 'index': idx, 'audio_base64': b64, 'sample_rate': 44100}, ensure_ascii=False)}\n\n"

        latency = int((time.time() - t0) * 1000)
        yield f"data: {json.dumps({'type': 'done', 'schemes': scheme_snippets, 'latency_ms': latency}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class NarrateRequest(BaseModel):
    text: str
    language: str = "hi"


@app.post("/narrate")
def narrate(req: NarrateRequest):
    """Convert text to speech — streams audio chunks as they're generated.

    SSE events:
      data: {"type":"chunk","index":0,"audio_base64":"...","sample_rate":44100}
      data: {"type":"chunk","index":1,"audio_base64":"...","sample_rate":44100}
      data: {"type":"done","elapsed_s":12.3,"total_chunks":3}
      data: {"type":"error","detail":"..."}
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")

    def event_stream():
        suspend_llm()
        try:
            chunk_count = 0
            for event in stream_audio_chunks(text, language=req.language):
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "chunk":
                    chunk_count += 1

            if chunk_count == 0:
                yield f"data: {json.dumps({'type': 'error', 'detail': 'TTS unavailable — check servers/server_tts.py is running.'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        finally:
            threading.Thread(target=resume_llm, daemon=True).start()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/categories")
def list_categories():
    schemes = get_schemes()
    cats = sorted({s.get("category", "Other") for s in schemes if s.get("category")})
    return {"categories": cats}
