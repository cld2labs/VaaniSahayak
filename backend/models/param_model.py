"""Param-1 Instruct inference via vLLM's OpenAI-compatible chat API."""
from collections.abc import Iterator
from openai import OpenAI
from backend.config import VLLM_LLM_URL, PARAM_MODEL_ID, MAX_NEW_TOKENS, TEMPERATURE, MAX_SCHEMES_IN_PROMPT

_client: OpenAI | None = None


def load_model():
    """Initialize the OpenAI client pointing at the vLLM server."""
    global _client
    if _client is not None:
        return
    _client = OpenAI(base_url=VLLM_LLM_URL, api_key="dummy")
    try:
        models = _client.models.list()
        names = [m.id for m in models.data]
        print(f"[Param-1] vLLM server ready. Available models: {names}")
    except Exception as e:
        print(f"[Param-1] WARNING: vLLM server not reachable at {VLLM_LLM_URL} — {e}")
        print("  Start it with: python server_param1.py --preload --port 8001")


def generate(prompt: str) -> str:
    """Non-streaming: return full response text."""
    if _client is None:
        load_model()
    response = _client.chat.completions.create(
        model=PARAM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def stream_generate(prompt: str) -> Iterator[str]:
    """Streaming: yield text chunks as they arrive from the param1 server."""
    if _client is None:
        load_model()
    with _client.chat.completions.create(
        model=PARAM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def suspend_llm():
    """Ask Param-1 server to offload weights to CPU, freeing MPS for TTS."""
    try:
        import httpx
        r = httpx.post(f"{VLLM_LLM_URL.replace('/v1', '')}/suspend", timeout=10)
        print(f"[Param-1] Suspended: {r.json()}")
    except Exception as e:
        print(f"[Param-1] Suspend failed (non-fatal): {e}")


def resume_llm():
    """Ask Param-1 server to move weights back to MPS."""
    try:
        import httpx
        r = httpx.post(f"{VLLM_LLM_URL.replace('/v1', '')}/resume", timeout=30)
        print(f"[Param-1] Resumed: {r.json()}")
    except Exception as e:
        print(f"[Param-1] Resume failed (non-fatal): {e}")


def build_prompt(query: str, schemes: list[dict], language: str = "hi") -> str:
    """Build the instruction prompt injecting retrieved schemes.

    Retrieval may return more schemes than fit in the context window.
    MAX_SCHEMES_IN_PROMPT enforces the token budget here, keeping retrieval agnostic.
    """
    scheme_context = ""
    for i, scheme in enumerate(schemes[:MAX_SCHEMES_IN_PROMPT], 1):
        eligibility = scheme.get("eligibility", "")
        benefits = scheme.get("benefits", "")
        documents = scheme.get("documents", "")
        link = scheme.get("official_link", "")
        scheme_context += (
            f"\n--- Scheme {i}: {scheme['name']} ---\n"
            f"Eligibility: {eligibility[:300]}\n"
            f"Benefits: {benefits[:200]}\n"
            f"Documents: {documents[:150]}\n"
            f"Link: {link}\n"
        )

    if language == "te":
        return (
            "మీరు వాణీ సహాయక్ — భారత ప్రభుత్వ సంక్షేమ పథకాల సహాయకుడు.\n"
            "మీరు ఎల్లప్పుడూ సరళమైన తెలుగులో సమాధానం ఇస్తారు.\n\n"
            "క్రింది పథకాల సమాచారం ఆధారంగా ప్రశ్నకు సమాధానం ఇవ్వండి:\n"
            f"{scheme_context}\n"
            f"పౌరుని ప్రశ్న: {query}\n\n"
            "దయచేసి సంక్షిప్తంగా తెలుగులో సమాధానం ఇవ్వండి (గరిష్ఠంగా 5-6 వాక్యాలు). "
            "పైన ఇచ్చిన సమాచారాన్ని మాత్రమే ఉపయోగించండి. అదే విషయాన్ని పునరావృతం చేయకండి."
        )
    else:
        return (
            "आप वाणी सहायक हैं — भारत सरकार की कल्याण योजनाओं के लिए एक सहायक।\n"
            "आप हमेशा सरल और स्पष्ट हिंदी में उत्तर देते हैं।\n"
            "अंग्रेज़ी शब्दों का प्रयोग न करें — सभी अंग्रेज़ी शब्दों को हिंदी में लिखें।\n"
            "केवल एक उत्तर दें — प्रश्न-उत्तर की सूची न बनाएं।\n\n"
            "निम्नलिखित योजनाओं की जानकारी के आधार पर प्रश्न का उत्तर दें:\n"
            f"{scheme_context}\n"
            f"नागरिक का प्रश्न: {query}\n\n"
            "कृपया संक्षिप्त और स्पष्ट हिंदी में उत्तर दें (अधिकतम 5-6 वाक्य)। "
            "केवल ऊपर दी गई जानकारी का उपयोग करें। बार-बार एक ही बात न दोहराएं। "
            "यदि जानकारी उपलब्ध नहीं है तो विनम्रता से बताएं।"
        )
