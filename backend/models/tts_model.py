"""Indic Parler-TTS wrapper.

Routing priority (highest → lowest):
  1. Enterprise Inference (EI) stack — GPU-accelerated, Keycloak-authenticated
     Active when EI_TTS_URL + Keycloak creds are set in env.
  2. Local server_tts.py — MPS-accelerated Mac server on port 8003
  3. Unavailable — returns None / empty iterator

Both EI and local expose the same SSE streaming API:
  POST /tts/stream   → SSE chunks: {type, index, audio_base64, sample_rate}
  POST /tts          → blocking:   {audio_base64, elapsed_s}
"""
import io
import json
import base64
import logging
from collections.abc import Iterator
from typing import Optional

import httpx
import psutil

from backend.config import (
    TTS_SERVER_URL,
    EI_TTS_ENABLED,
    EI_TTS_URL,
    EI_TTS_TOKEN,
    EI_KEYCLOAK_URL,
    EI_KEYCLOAK_REALM,
    EI_CLIENT_ID,
    EI_CLIENT_SECRET,
    EI_VERIFY_SSL,
)

logger = logging.getLogger(__name__)

# Voice descriptions per language for Indic Parler-TTS
_TTS_DESCRIPTIONS = {
    "hi": (
        "A calm and clear female Hindi voice with very clear audio, "
        "speaking slowly and distinctly, suitable for government information delivery."
    ),
    "te": (
        "A calm and clear female Telugu voice with very clear audio, "
        "speaking slowly and distinctly, suitable for government information delivery."
    ),
}

_MIN_FREE_RAM_GB = 5.0
_TTS_MAX_CHARS = 800

# Local server state
_local_server_available: bool | None = None

# EI token manager (initialized lazily)
_ei_token_manager: Optional[object] = None


# ── Text prep ──────────────────────────────────────────────────────────

def _prepare_for_speech(text: str) -> str:
    """Strip markdown/bullets and truncate to _TTS_MAX_CHARS."""
    import re
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", " ", text)
    sentences = re.split(r'(?<=[।.!?])\s+', text.strip())
    result, length = [], 0
    for s in sentences:
        if length + len(s) > _TTS_MAX_CHARS and result:
            break
        result.append(s)
        length += len(s)
    return " ".join(result) if result else text[:_TTS_MAX_CHARS]


def _has_enough_memory() -> bool:
    free_gb = psutil.virtual_memory().available / 1e9
    if free_gb < _MIN_FREE_RAM_GB:
        logger.warning("[TTS] Skipping — only %.1fGB free (need %.1fGB)", free_gb, _MIN_FREE_RAM_GB)
        return False
    return True


# ── SSE stream parser (shared by EI and local) ─────────────────────────

def _parse_sse_stream(response: httpx.Response) -> Iterator[dict]:
    buffer = ""
    for raw_chunk in response.iter_text():
        buffer += raw_chunk
        while "\n\n" in buffer:
            event_str, buffer = buffer.split("\n\n", 1)
            for line in event_str.split("\n"):
                if line.startswith("data: "):
                    yield json.loads(line[6:])


# ══════════════════════════════════════════════════════════════════════ #
#  EI (Enterprise Inference) path                                        #
# ══════════════════════════════════════════════════════════════════════ #

def _ei_headers() -> dict[str, str]:
    """Return auth headers — static token takes priority over Keycloak."""
    if EI_TTS_TOKEN:
        return {"Authorization": f"Bearer {EI_TTS_TOKEN}"}
    return _get_ei_token_manager().auth_headers()


def _get_ei_token_manager():
    """Lazily initialise the Keycloak token manager (only used when no static token)."""
    global _ei_token_manager
    if _ei_token_manager is None:
        from backend.services.keycloak_auth import KeycloakTokenManager
        _ei_token_manager = KeycloakTokenManager(
            keycloak_url=EI_KEYCLOAK_URL,
            realm=EI_KEYCLOAK_REALM,
            client_id=EI_CLIENT_ID,
            client_secret=EI_CLIENT_SECRET,
            verify_ssl=EI_VERIFY_SSL,
        )
        logger.info("[TTS/EI] Keycloak token manager initialised (realm=%s)", EI_KEYCLOAK_REALM)
    return _ei_token_manager


def _ei_stream_audio_chunks(tts_text: str, description: str) -> Iterator[dict]:
    """Stream audio chunks from the EI GPU stack.

    If the server has a /tts/stream SSE endpoint, uses it.
    Otherwise falls back to blocking /tts and yields as a single chunk.
    """
    _base        = EI_TTS_URL.rstrip('/')
    blocking_url = _base if _base.endswith('/tts') else f"{_base}/tts"
    stream_url   = f"{blocking_url}/stream"

    # Try SSE streaming first
    try:
        with httpx.stream(
            "POST",
            stream_url,
            json={"text": tts_text, "description": description},
            headers=_ei_headers(),
            verify=EI_VERIFY_SSL,
            timeout=httpx.Timeout(connect=15, read=600, write=15, pool=15),
        ) as r:
            if r.status_code == 404:
                raise httpx.HTTPStatusError("404", request=r.request, response=r)
            r.raise_for_status()
            logger.info("[TTS/EI] Streaming from %s", stream_url)
            yield from _parse_sse_stream(r)
            return
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
        logger.info("[TTS/EI] /tts/stream not available — falling back to blocking /tts")

    # Fallback: blocking call → single chunk
    audio_b64 = _ei_text_to_audio_base64(tts_text, description)
    if audio_b64:
        yield {"type": "chunk", "index": 0, "audio_base64": audio_b64, "sample_rate": 44100}
        yield {"type": "done", "elapsed_s": 0, "total_chunks": 1}


def _ei_synth_one(sentence: str) -> Optional[str]:
    """Send a single sentence to EI /tts, return audio_base64."""
    _base = EI_TTS_URL.rstrip('/')
    url   = _base if _base.endswith('/tts') else f"{_base}/tts"

    for attempt in range(2):
        try:
            r = httpx.post(
                url,
                json={"text": sentence},
                headers=_ei_headers(),
                verify=EI_VERIFY_SSL,
                timeout=90,   # 90s per sentence — allows headroom for longer Hindi sentences
            )
            if r.status_code == 401 and attempt == 0 and not EI_TTS_TOKEN:
                _get_ei_token_manager().invalidate()
                continue
            r.raise_for_status()
            return r.json()["audio_base64"]
        except Exception as exc:
            logger.error("[TTS/EI] Sentence synth failed: %s", exc)
            return None
    return None


def _ei_batch_synth(tts_text: str) -> Optional[str]:
    """Send full text to /tts with batch=true — server normalizes, splits, and concatenates."""
    _base = EI_TTS_URL.rstrip('/')
    url = _base if _base.endswith('/tts') else f"{_base}/tts"

    try:
        r = httpx.post(
            url,
            json={"text": tts_text, "batch": True},
            headers=_ei_headers(),
            verify=EI_VERIFY_SSL,
            timeout=120,  # batch may take longer
        )
        r.raise_for_status()
        data = r.json()
        logger.info("[TTS/EI] Batch synth: %d sentences in %ss",
                     data.get("sentence_count", "?"), data.get("elapsed", "?"))
        return data["audio_base64"]
    except Exception as exc:
        logger.warning("[TTS/EI] Batch synth failed: %s", exc)
        return None


def _ei_parallel_synth(tts_text: str) -> Optional[str]:
    """Send sentences concurrently via ThreadPoolExecutor to /tts."""
    import re, base64, io
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np
    import soundfile as sf

    # Server-side /tts now normalizes, so skip client-side normalization

    sentences = [s.strip() for s in re.split(r'(?<=[।.!?])\s+', tts_text.strip()) if s.strip()]
    if not sentences:
        return None

    # Submit all sentences concurrently; server semaphore serializes GPU work
    # but we overlap network I/O with GPU compute
    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_ei_synth_one, sent): i for i, sent in enumerate(sentences)}
        for fut in as_completed(futures):
            idx = futures[fut]
            b64 = fut.result()
            if b64:
                results[idx] = b64

    if not results:
        return None

    parts: list[np.ndarray] = []
    sample_rate = None
    for i in range(len(sentences)):
        b64 = results.get(i)
        if not b64:
            continue
        wav_bytes = base64.b64decode(b64)
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        sample_rate = sr
        parts.append(audio.astype("float32"))
        if i < len(sentences) - 1:
            parts.append(np.zeros(int(sr * 0.25), dtype="float32"))

    if not parts or sample_rate is None:
        return None

    combined = np.concatenate(parts)
    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _ei_sequential_synth(tts_text: str) -> Optional[str]:
    """Original sequential sentence-by-sentence synthesis."""
    import re, base64, io
    import numpy as np
    import soundfile as sf

    # Server-side /tts now normalizes, so skip client-side normalization

    sentences = [s.strip() for s in re.split(r'(?<=[।.!?])\s+', tts_text.strip()) if s.strip()]
    if not sentences:
        return None

    parts: list[np.ndarray] = []
    sample_rate = None

    for sent in sentences:
        logger.info("[TTS/EI] Synthesizing: %s", sent[:60])
        b64 = _ei_synth_one(sent)
        if not b64:
            continue
        wav_bytes = base64.b64decode(b64)
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        sample_rate = sr
        parts.append(audio.astype("float32"))
        parts.append(np.zeros(int(sr * 0.25), dtype="float32"))

    if not parts or sample_rate is None:
        return None

    combined = np.concatenate(parts)
    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _ei_text_to_audio_base64(tts_text: str, description: str) -> Optional[str]:
    """Batch-first with parallel fallback, then sequential as last resort."""
    # 1. Try batch endpoint (server handles normalization + splitting)
    result = _ei_batch_synth(tts_text)
    if result:
        return result

    # 2. Fallback: parallel sentence synthesis
    logger.info("[TTS/EI] Falling back to parallel sentence synthesis")
    result = _ei_parallel_synth(tts_text)
    if result:
        return result

    # 3. Last resort: sequential
    logger.info("[TTS/EI] Falling back to sequential sentence synthesis")
    return _ei_sequential_synth(tts_text)


# ══════════════════════════════════════════════════════════════════════ #
#  Local server path                                                      #
# ══════════════════════════════════════════════════════════════════════ #

def _check_local_server() -> bool:
    global _local_server_available
    if _local_server_available is True:
        return True
    try:
        r = httpx.get(f"{TTS_SERVER_URL}/health", timeout=3)
        if r.status_code == 200 and r.json().get("model_loaded", False):
            _local_server_available = True
            logger.info("[TTS/local] Using local TTS server at %s", TTS_SERVER_URL)
    except Exception:
        pass
    return _local_server_available is True


def _local_stream_audio_chunks(tts_text: str, description: str) -> Iterator[dict]:
    global _local_server_available
    try:
        with httpx.stream(
            "POST",
            f"{TTS_SERVER_URL}/tts/stream",
            json={"text": tts_text, "description": description},
            timeout=httpx.Timeout(connect=10, read=600, write=10, pool=10),
        ) as r:
            r.raise_for_status()
            yield from _parse_sse_stream(r)
            return
    except Exception as e:
        logger.warning("[TTS/local] Stream failed: %s", e)
        _local_server_available = False
        raise


def _local_text_to_audio_base64(tts_text: str, description: str) -> Optional[str]:
    global _local_server_available
    try:
        r = httpx.post(
            f"{TTS_SERVER_URL}/tts",
            json={"text": tts_text, "description": description},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["audio_base64"]
    except Exception as e:
        logger.warning("[TTS/local] Blocking call failed: %s", e)
        _local_server_available = False
        return None


# ══════════════════════════════════════════════════════════════════════ #
#  Public API                                                             #
# ══════════════════════════════════════════════════════════════════════ #

def load_tts() -> None:
    """Called at startup to verify TTS availability and log which backend is active."""
    if EI_TTS_ENABLED:
        if EI_TTS_TOKEN:
            logger.info("[TTS] EI GPU stack ready at %s (static token)", EI_TTS_URL)
        else:
            try:
                _get_ei_token_manager().get_token()
                logger.info("[TTS] EI GPU stack ready at %s (Keycloak)", EI_TTS_URL)
            except Exception as e:
                logger.warning("[TTS] EI stack token fetch failed: %s — will retry per request", e)
        return

    if _check_local_server():
        return
    logger.warning(
        "[TTS] No TTS backend available. "
        "Start servers/server_tts.py locally or configure EI_TTS_URL + Keycloak env vars."
    )


def synth_sentence(text: str, language: str = "hi") -> Optional[str]:
    """Synthesize a single sentence — returns base64 WAV or None.

    Used by /ask/speak for per-sentence TTS. Unlike text_to_audio_base64,
    this does NOT split into sub-sentences or concatenate.
    """
    if EI_TTS_ENABLED:
        return _ei_synth_one(text)
    description = _TTS_DESCRIPTIONS.get(language, _TTS_DESCRIPTIONS["hi"])
    if _check_local_server():
        return _local_text_to_audio_base64(text, description)
    return None


def text_to_audio_base64(text: str, language: str = "hi") -> Optional[str]:
    """Convert text to speech — blocking, returns base64 WAV."""
    if not _has_enough_memory():
        return None

    tts_text = _prepare_for_speech(text)
    description = _TTS_DESCRIPTIONS.get(language, _TTS_DESCRIPTIONS["hi"])

    if EI_TTS_ENABLED:
        return _ei_text_to_audio_base64(tts_text, description)

    if _check_local_server():
        return _local_text_to_audio_base64(tts_text, description)

    return None


def stream_audio_chunks(text: str, language: str = "hi") -> Iterator[dict]:
    """Stream audio chunks as SSE events.

    Yields:
      {"type": "chunk", "index": int, "audio_base64": str, "sample_rate": int}
      {"type": "done",  "elapsed_s": float, "total_chunks": int}
    """
    if not _has_enough_memory():
        return

    tts_text = _prepare_for_speech(text)
    description = _TTS_DESCRIPTIONS.get(language, _TTS_DESCRIPTIONS["hi"])

    # ── EI GPU path ──────────────────────────────────────────────────
    if EI_TTS_ENABLED:
        try:
            yield from _ei_stream_audio_chunks(tts_text, description)
            return
        except Exception as e:
            logger.error("[TTS] EI stream failed: %s — no local fallback when EI is configured", e)
            return

    # ── Local server path ────────────────────────────────────────────
    if not _check_local_server():
        return

    try:
        yield from _local_stream_audio_chunks(tts_text, description)
        return
    except Exception:
        pass

    # Fallback: blocking local call → single chunk
    _local_server_available = None  # allow re-check
    audio_b64 = text_to_audio_base64(text, language)
    if audio_b64:
        yield {"type": "chunk", "index": 0, "audio_base64": audio_b64, "sample_rate": 44100}
        yield {"type": "done", "elapsed_s": 0, "total_chunks": 1}
