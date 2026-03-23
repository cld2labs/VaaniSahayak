"""Test remote TTS inference over the internet.

Usage:
  python scripts/test_remote_tts.py --token YOUR_TOKEN
  python scripts/test_remote_tts.py --token YOUR_TOKEN --text "नमस्ते, आप कैसे हैं?"
  python scripts/test_remote_tts.py --token YOUR_TOKEN --output /tmp/out.wav --no-play

Env vars (alternative to flags):
  TOKEN    : Bearer token
  TTS_URL  : override base URL (default: http://api.example.com:32237)
"""

import argparse
import base64
import os
import subprocess
import sys
import tempfile
import time

import httpx
from pathlib import Path

# Load .env from project root
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

TTS_URL = os.getenv("TTS_URL", "http://api.example.com:32237")
DEFAULT_TEXT = "नमस्ते। प्रधानमंत्री किसान सम्मान निधि योजना के तहत किसानों को प्रति वर्ष छह हजार रुपये दिए जाते हैं।"


def synth_one(sentence: str, token: str) -> bytes | None:
    """Synthesize a single sentence, return WAV bytes."""
    resp = httpx.post(
        f"{TTS_URL}/v1/tts",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"text": sentence},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    b64 = data.get("audio_base64") or data.get("audio")
    return base64.b64decode(b64) if b64 else None


def call_tts_sequential(text: str, token: str) -> tuple[bytes, float]:
    """Split text into sentences, synthesize each sequentially, concatenate."""
    import re
    import io
    import numpy as np
    import soundfile as sf

    try:
        from text_normalize import normalize_for_tts
        text = normalize_for_tts(text)
    except ImportError:
        pass
    sentences = [s.strip() for s in re.split(r'(?<=[।.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        sentences = [text]

    t0 = time.time()
    parts: list[np.ndarray] = []
    sample_rate = None

    for i, sent in enumerate(sentences):
        print(f"  Sentence {i+1}/{len(sentences)}: {sent[:60]}")
        wav = synth_one(sent, token)
        if not wav:
            continue
        audio, sr = sf.read(io.BytesIO(wav))
        sample_rate = sr
        parts.append(audio.astype("float32"))
        if i < len(sentences) - 1:
            parts.append(np.zeros(int(sr * 0.25), dtype="float32"))

    elapsed = round(time.time() - t0, 2)

    if not parts or sample_rate is None:
        print("[ERROR] No audio generated")
        sys.exit(1)

    combined = np.concatenate(parts)
    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV")
    wav_bytes = buf.getvalue()

    print(f"  Round-trip  : {elapsed}s")
    print(f"  Audio size  : {len(wav_bytes) / 1024:.1f} KB")
    return wav_bytes, elapsed


def call_tts_parallel(text: str, token: str) -> tuple[bytes, float]:
    """Synthesize sentences concurrently via ThreadPoolExecutor."""
    import re
    import io
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np
    import soundfile as sf

    try:
        from text_normalize import normalize_for_tts
        text = normalize_for_tts(text)
    except ImportError:
        pass
    sentences = [s.strip() for s in re.split(r'(?<=[।.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        sentences = [text]

    print(f"  Parallel mode: {len(sentences)} sentences, max_workers=4")
    t0 = time.time()
    results: dict[int, bytes] = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(synth_one, sent, token): i for i, sent in enumerate(sentences)}
        for fut in as_completed(futures):
            idx = futures[fut]
            wav = fut.result()
            if wav:
                results[idx] = wav
                print(f"  Sentence {idx+1}/{len(sentences)} done")

    parts: list[np.ndarray] = []
    sample_rate = None
    for i in range(len(sentences)):
        wav = results.get(i)
        if not wav:
            continue
        audio, sr = sf.read(io.BytesIO(wav))
        sample_rate = sr
        parts.append(audio.astype("float32"))
        if i < len(sentences) - 1:
            parts.append(np.zeros(int(sr * 0.25), dtype="float32"))

    elapsed = round(time.time() - t0, 2)

    if not parts or sample_rate is None:
        print("[ERROR] No audio generated")
        sys.exit(1)

    combined = np.concatenate(parts)
    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV")
    wav_bytes = buf.getvalue()

    print(f"  Round-trip  : {elapsed}s")
    print(f"  Audio size  : {len(wav_bytes) / 1024:.1f} KB")
    return wav_bytes, elapsed


def call_tts_batch(text: str, token: str) -> tuple[bytes, float]:
    """Send full text to /tts with batch=true — server normalizes, splits, concatenates."""
    print("  Batch mode: single request to /v1/tts with batch=true")
    t0 = time.time()
    resp = httpx.post(
        f"{TTS_URL}/v1/tts",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"text": text, "batch": True},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    elapsed = round(time.time() - t0, 2)

    b64 = data.get("audio_base64")
    if not b64:
        print("[ERROR] No audio in batch response")
        sys.exit(1)

    wav_bytes = base64.b64decode(b64)
    server_elapsed = data.get("elapsed", "?")
    sentence_count = data.get("sentence_count", "?")
    print(f"  Server time : {server_elapsed}s ({sentence_count} sentences)")
    print(f"  Round-trip  : {elapsed}s")
    print(f"  Audio size  : {len(wav_bytes) / 1024:.1f} KB")
    return wav_bytes, elapsed


def call_tts(text: str, token: str, mode: str = "sequential") -> tuple[bytes, float]:
    """Dispatch to the appropriate synthesis mode."""
    if mode == "batch":
        return call_tts_batch(text, token)
    elif mode == "parallel":
        return call_tts_parallel(text, token)
    else:
        return call_tts_sequential(text, token)


def play(wav_bytes: bytes, output_path: str | None):
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.write(wav_bytes)
        tmp.close()
    else:
        with open(output_path, "wb") as f:
            f.write(wav_bytes)

    print(f"  Saved to    : {output_path}")
    if sys.platform == "darwin":
        subprocess.run(["afplay", output_path], check=False)
    else:
        for player in ["aplay", "paplay"]:
            if subprocess.run(["which", player], capture_output=True).returncode == 0:
                subprocess.run([player, output_path], check=False)
                break


def run_benchmark(text: str, token: str):
    """Run all three modes and compare latency."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Comparing sequential / parallel / batch")
    print("=" * 60)

    results = {}
    for mode in ("sequential", "parallel", "batch"):
        print(f"\n── {mode.upper()} ──")
        try:
            _, elapsed = call_tts(text, token, mode=mode)
            results[mode] = elapsed
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[mode] = None

    print("\n" + "=" * 60)
    print("RESULTS:")
    for mode, elapsed in results.items():
        status = f"{elapsed}s" if elapsed is not None else "FAILED"
        print(f"  {mode:12s}: {status}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",      default=DEFAULT_TEXT)
    parser.add_argument("--token",     default=os.getenv("EI_TTS_TOKEN", os.getenv("TTS_TOKEN", os.getenv("TOKEN", ""))))
    parser.add_argument("--output",    default=None, help="Save WAV to path")
    parser.add_argument("--no-play",   action="store_true")
    parser.add_argument("--mode",      choices=["sequential", "parallel", "batch"], default="sequential",
                        help="Synthesis mode (default: sequential)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run all three modes and compare latency")
    args = parser.parse_args()

    if not args.token:
        print("[ERROR] Token required. Pass --token or set TOKEN env var.")
        sys.exit(1)

    print(f"\n→ {TTS_URL}/v1/tts")
    print(f"  Text: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")

    if args.benchmark:
        run_benchmark(args.text, args.token)
        return

    wav_bytes, _ = call_tts(args.text, args.token, mode=args.mode)

    if args.no_play:
        if args.output:
            with open(args.output, "wb") as f:
                f.write(wav_bytes)
            print(f"  Saved to    : {args.output}")
    else:
        play(wav_bytes, args.output)


if __name__ == "__main__":
    main()
