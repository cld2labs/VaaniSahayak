"""
Lightweight OpenAI-compatible inference server for Param-1-2.9B-Instruct.
Uses transformers + MPS (Apple Silicon Metal) directly.

Start with:
    python server_param1.py           # port 8001
    python server_param1.py --port 9001

Compatible with the openai Python client — exposes:
    POST /v1/chat/completions
    GET  /v1/models
"""
import argparse
import asyncio
import json
import threading
import time
import uuid
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

MODEL_ID = "bharatgenai/Param-1-2.9B-Instruct"
MAX_NEW_TOKENS = 512   # match backend/config.py — enough for detailed Hindi answers
TEMPERATURE = 0.6      # model card recommendation
TOP_K = 50             # model card recommendation
TOP_P = 0.95           # model card recommendation
TOKEN_IDLE_TIMEOUT_S = 15  # stop if no new token for this many seconds (catches runaway/stuck gen)
# Stop generation when the model starts echoing prompt structure.
# NOTE: "[ASSISTANT" (no closing ]) catches both "[ASSISTANT]" and "[ASSISTANT]:" variants.
STOP_PATTERNS = [
    # ChatML turn boundaries — primary stop mechanism
    "<|im_start|>",         # model trying to start a new turn
    "<|im_end|>",           # model ending its turn
    # Prompt structure leaking
    "[ASSISTANT",           # catches [ASSISTANT], [ASSISTANT]: etc.
    "[USER",                # catches [USER], [USER]: etc.
    "[SYSTEM",              # catches [SYSTEM], [SYSTEM]: etc.
    "[QUESTION",            # model generating fake Q&A pairs
    "[ANSWER",              # model generating fake Q&A pairs
    "\n\n\n",               # triple newline = runaway generation
    "नागरिक का प्रश्न",      # Hindi prompt template leaking
    "పౌరుని ప్రశ్న",          # Telugu prompt template leaking
    "\n--- Scheme",         # scheme context header leaking (not bare "Scheme")
]

app = FastAPI(title="Param-1 Inference Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_tokenizer = None
_model = None
_sem = asyncio.Semaphore(1)  # one generation at a time to protect MPS memory


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load():
    global _tokenizer, _model
    device = _get_device()
    print(f"[Param-1] Loading {MODEL_ID} on {device}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Ensure ChatML template is set (Param-1 uses <|im_start|>/<|im_end|>)
    if not getattr(_tokenizer, 'chat_template', None):
        _tokenizer.chat_template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
            "<|im_start|>assistant\n"
        )
        print("[Param-1] Set ChatML chat_template on tokenizer")

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    _model.eval()
    print(f"[Param-1] EOS token: {_tokenizer.eos_token!r} (id={_tokenizer.eos_token_id})")
    print(f"[Param-1] Ready on {device}.")


# ---------- Pydantic schemas ----------

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    max_tokens: int = MAX_NEW_TOKENS
    temperature: float = TEMPERATURE
    stream: bool = False


# ---------- Sampling helpers ----------

def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Apply temperature + top-k + top-p (nucleus) sampling and return next token id."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    # top-k
    if top_k > 0:
        top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_k_vals[:, -1:]] = float("-inf")

    probs = torch.softmax(logits, dim=-1)

    # top-p (nucleus)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens beyond nucleus (shift right so we keep the first token over threshold)
        mask = cum_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
        total = probs.sum(dim=-1, keepdim=True)
        probs = probs / total.clamp(min=1e-8)

    return torch.multinomial(probs, num_samples=1)


# ---------- Manual generation (bypasses DynamicCache incompatibility) ----------

def _generate(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> list[int]:
    """
    Token-by-token generation loop.
    Passes past_key_values=None on the first step so the model\'s own code
    initialises the cache in tuple-of-tuples format. Subsequent steps pass
    the cache returned by the model — bypassing the DynamicCache incompatibility
    introduced in transformers >=4.36.
    """
    eos_id = _tokenizer.eos_token_id
    device = _model.device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    past_key_values = None
    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = _model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]  # (1, vocab)
            past_key_values = outputs.past_key_values

            next_id = _sample_next_token(logits, temperature if do_sample else 0, TOP_K, TOP_P)
            tok = next_id.item()
            generated.append(tok)

            if tok == eos_id:
                break

            # Repetition guard: if last 20 tokens repeat a 5-token cycle, stop
            if len(generated) >= 25:
                tail = generated[-20:]
                pattern = generated[-5:]
                if tail[:15].count(pattern[0]) >= 4 and generated[-10:] == generated[-20:-10]:
                    break

            # Sentence-level repetition guard
            if len(generated) >= 60 and len(generated) % 20 == 0:
                import re as _re
                _decoded = _tokenizer.decode(generated, skip_special_tokens=True)
                _sents = [s.strip() for s in _re.split(r'(?<=[।.!?])\s+', _decoded.strip()) if s.strip()]
                if len(_sents) >= 3 and _sents[-1] == _sents[-2]:
                    _trimmed = _decoded[:_decoded.rfind(_sents[-1])].strip()
                    generated = _tokenizer.encode(_trimmed, add_special_tokens=False)
                    break

            # Stop-pattern guard: check every token (not every 10) to catch partial patterns
            partial = _tokenizer.decode(generated, skip_special_tokens=True)
            if any(pat in partial for pat in STOP_PATTERNS):
                # Trim the generated tokens up to where the stop pattern appears
                generated = _trim_at_stop(generated)
                break

            # Next step: only pass the new token; KV cache holds the rest
            input_ids = next_id
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)],
                    dim=-1,
                )

    # Free MPS cache after generation
    if device == "mps":
        torch.mps.empty_cache()

    return generated


def _trim_at_stop(token_ids: list[int]) -> list[int]:
    """Binary-search trim: remove trailing tokens that caused a stop pattern match."""
    full_text = _tokenizer.decode(token_ids, skip_special_tokens=True)
    for pat in STOP_PATTERNS:
        idx = full_text.find(pat)
        if idx != -1:
            clean = full_text[:idx].rstrip()
            # Re-encode the clean prefix to get the right token count
            return _tokenizer.encode(clean, add_special_tokens=False)
    return token_ids


def _build_input_ids(messages: list[Message]) -> torch.Tensor:
    """Use tokenizer.apply_chat_template (matches training format)."""
    conversation = [{"role": msg.role, "content": msg.content} for msg in messages]
    try:
        ids = _tokenizer.apply_chat_template(
            conversation=conversation,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        return ids
    except Exception as e:
        print(f"[Param-1] apply_chat_template failed ({e}), using manual ChatML")
        # Fallback: manual ChatML template matching model's training format
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return _tokenizer(prompt, return_tensors="pt")["input_ids"]


def _clean(text: str) -> str:
    # Strip ChatML markers
    for marker in ["<|im_start|>assistant\n", "<|im_start|>assistant", "<|im_end|>"]:
        text = text.replace(marker, "")
    # Legacy markers (shouldn't appear with correct template, but just in case)
    for marker in ["[ASSISTANT]:", "[ASSISTANT]", "[ASSISTANT"]:
        if marker in text:
            text = text.split(marker)[-1].strip()
            break
    # Trim at any remaining stop patterns
    for pat in STOP_PATTERNS:
        idx = text.find(pat)
        if idx != -1:
            text = text[:idx].rstrip()
    # Remove Unicode replacement characters (broken byte-level tokens)
    text = text.replace("\ufffd", "")
    return text.strip()


# ---------- Routes ----------

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "bharatgenai"}],
    }


@app.post("/suspend")
async def suspend():
    """Offload model weights to CPU, freeing MPS memory for TTS.
    Call /resume before the next generation request."""
    global _model
    if _model is None:
        return {"status": "not_loaded"}
    async with _sem:
        device = _model.device
        if str(device) != "cpu":
            _model = _model.to("cpu")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"[Param-1] Suspended — moved weights to CPU, MPS freed")
    return {"status": "suspended"}


@app.post("/resume")
async def resume():
    """Move model weights back to MPS after TTS is done."""
    global _model
    if _model is None:
        return {"status": "not_loaded"}
    device = _get_device()
    if str(_model.device) == "cpu" and device != "cpu":
        _model = _model.to(device)
        print(f"[Param-1] Resumed — weights back on {device}")
    return {"status": "resumed"}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if _model is None:
        load()

    # Reject if another generation is already running
    if not _sem._value:
        raise HTTPException(status_code=503, detail="Model busy — try again shortly.")

    input_ids = _build_input_ids(req.messages).to(_model.device)
    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    max_tokens = min(req.max_tokens, MAX_NEW_TOKENS)  # never exceed hard cap
    temperature = req.temperature

    if req.stream:
        def token_stream():
            t0 = time.time()
            generated = []
            prev_text = ""
            eos_id = _tokenizer.eos_token_id
            device = _model.device
            cur_ids = input_ids.clone()
            attention_mask = torch.ones_like(cur_ids)

            past_key_values = None
            last_token_t = time.time()
            with torch.no_grad():
                for _ in range(max_tokens):
                    outputs = _model(
                        input_ids=cur_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values

                    next_id = _sample_next_token(logits, temperature, TOP_K, TOP_P)
                    tok = next_id.item()
                    if tok == eos_id:
                        break
                    generated.append(tok)

                    # Repetition guard: 10-token cycle detection
                    if len(generated) >= 20 and generated[-10:] == generated[-20:-10]:
                        print("[Param-1] Repetition detected (token cycle) — stopping")
                        break

                    # Sentence-level repetition: if the decoded text contains the same
                    # sentence repeated, stop generation immediately
                    if len(generated) >= 60 and len(generated) % 20 == 0:
                        _decoded = _tokenizer.decode(generated, skip_special_tokens=True)
                        import re as _re
                        _sents = [s.strip() for s in _re.split(r'(?<=[।.!?])\s+', _decoded.strip()) if s.strip()]
                        if len(_sents) >= 3 and _sents[-1] == _sents[-2]:
                            print(f"[Param-1] Sentence repetition detected — stopping")
                            # Trim to just before the repeated sentence
                            _trimmed = _decoded[:_decoded.rfind(_sents[-1])].strip()
                            generated = _tokenizer.encode(_trimmed, add_special_tokens=False)
                            prev_text = _trimmed
                            break

                    # Decode full sequence and diff to get proper word-boundary spaces
                    full_text = _tokenizer.decode(generated, skip_special_tokens=True)
                    token_text = full_text[len(prev_text):]
                    prev_text = full_text

                    if not token_text:
                        # Idle timeout: if stuck producing empty tokens too long, stop
                        if time.time() - last_token_t > TOKEN_IDLE_TIMEOUT_S:
                            print(f"[Param-1] Idle timeout — stopping generation")
                            break
                        cur_ids = next_id
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)],
                            dim=-1,
                        )
                        continue

                    # Stop-pattern guard: check every token
                    hit_stop = False
                    for pat in STOP_PATTERNS:
                        if pat in full_text:
                            print(f"[Param-1] Stop pattern \'{pat}\' detected — stopping")
                            # Trim token_text if the stop pattern started mid-token
                            pat_idx = full_text.find(pat)
                            if pat_idx < len(prev_text) - len(token_text):
                                token_text = ""
                            else:
                                trim_at = pat_idx - (len(prev_text) - len(token_text))
                                token_text = token_text[:trim_at]
                            hit_stop = True
                            break

                    last_token_t = time.time()

                    # Strip Unicode replacement chars from streamed text
                    if token_text:
                        token_text = token_text.replace("\ufffd", "")
                    if token_text:
                        chunk = {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "model": req.model,
                            "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    if hit_stop:
                        break

                    cur_ids = next_id
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)],
                        dim=-1,
                    )

            # Free MPS memory
            if device == "mps":
                torch.mps.empty_cache()

            stop_chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "elapsed_s": round(time.time() - t0, 2),
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_sync_generation():
            """Runs in a background thread — feeds tokens into the queue."""
            try:
                for chunk_str in token_stream():
                    asyncio.run_coroutine_threadsafe(queue.put(chunk_str), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

        async def async_stream():
            async with _sem:
                thread = threading.Thread(target=_run_sync_generation, daemon=True)
                thread.start()
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item

        return StreamingResponse(async_stream(), media_type="text/event-stream")

    # Non-streaming path
    async with _sem:
        t0 = time.time()
        new_ids = _generate(
            input_ids=input_ids,
            attention_mask=None,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        elapsed = time.time() - t0
    text = _clean(_tokenizer.decode(new_ids, skip_special_tokens=True).strip())

    return {
        "id": cid,
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": len(new_ids),
            "total_tokens": int(input_ids.shape[1]) + len(new_ids),
        },
        "elapsed_s": round(elapsed, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--preload", action="store_true", help="Load model at startup")
    args = parser.parse_args()

    if args.preload:
        load()

    uvicorn.run(app, host=args.host, port=args.port)
