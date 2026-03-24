"""
Day 1 smoke test — verify vLLM servers and TTS produce correct output.

Prerequisites:
  Terminal 1: vllm serve iitm-prakalp/param-1-2.9B --port 8001
  Terminal 2: vllm serve sentence-transformers/all-MiniLM-L6-v2 --task embed --port 8002

Run from project root:
    python scripts/test_day1.py [param|embed|tts]   # one test
    python scripts/test_day1.py                      # all tests
"""
import sys
import numpy as np


def test_param():
    print("\n=== Testing Param-1 via vLLM (port 8001) ===")
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

    models = client.models.list()
    print(f"Available models: {[m.id for m in models.data]}")

    prompt = "PM-KISAN योजना के बारे में हिंदी में बताएं।"
    response = client.chat.completions.create(
        model="bharatgenai/Param-1-2.9B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )
    text = response.choices[0].message.content.strip()
    print(f"[PARAM] {text[:300]}")


def test_embed():
    print("\n=== Testing Embeddings via vLLM (port 8002) ===")
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

    query = "PM-KISAN के लिए कौन पात्र है?"
    response = client.embeddings.create(
        model="sentence-transformers/all-MiniLM-L6-v2",
        input=query,
    )
    emb = np.array(response.data[0].embedding)
    print(f"[EMBED] Shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")


def test_tts():
    print("\n=== Testing Indic Parler-TTS (in-process) ===")
    import torch
    import soundfile as sf
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    model_id = "ai4bharat/indic-parler-tts"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    desc_tok = AutoTokenizer.from_pretrained(model_id)

    description = "A calm Hindi female voice speaking clearly."
    text = "नमस्ते, मैं वाणी सहायक हूँ।"

    desc_ids = desc_tok(description, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        gen = model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids)

    audio = gen.cpu().numpy().squeeze()
    sf.write("test_tts_output.wav", audio, model.config.sampling_rate)
    import os
    print(f"[TTS] test_tts_output.wav written ({os.path.getsize('test_tts_output.wav')} bytes)")


if __name__ == "__main__":
    tests = {"param": test_param, "embed": test_embed, "tts": test_tts}

    args = sys.argv[1:]
    targets = [tests[a] for a in args if a in tests] if args else list(tests.values())

    for fn in targets:
        try:
            fn()
        except Exception as e:
            print(f"FAILED: {e}")
