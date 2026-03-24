"""Local TTS test — calls server_tts.py, saves chunks + combined WAV, plays it.

Usage:
    python scripts/test_tts_local.py
    python scripts/test_tts_local.py --text "आपका कस्टम टेक्स्ट"
    python scripts/test_tts_local.py --no-play   # skip afplay, just save files
"""
import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import subprocess

TTS_URL = "http://localhost:8003"
OUT_DIR = Path("/tmp/vaani_tts_test")

DEFAULT_TEXT = (
    "आयुष्मान भारत एक राष्ट्रीय स्वास्थ्य सुरक्षा योजना है जो 2018 में शुरू की गई थी। "
    "यह योजना गरीब और आर्थिक रूप से कमजोर वर्गों की स्वास्थ्य सेवाओं की मदद करती है। "
    "इस योजना में लाभ शामिल हैं अस्पताल में भर्ती, आउटपेशेंट उपचार, दवाओं, और निदान की सुविधाएं। "
    "इसके अलावा, यह योजना गरीब परिवारों की दीर्घकालिक स्वास्थ्य देखभाल की जरूरत को पूरा करने "
    "वाली चिकित्सा सुनिश्चित करती है।"
)


def b64_to_array(b64: str) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    arr, sr = sf.read(buf, dtype="float32")
    return arr, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--no-play", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # Check health
    h = requests.get(f"{TTS_URL}/health", timeout=5).json()
    print(f"[health] {h}")
    if not h.get("model_loaded"):
        print("ERROR: TTS model not loaded. Start: python server_tts.py --preload")
        return

    print(f"\nText ({len(args.text)} chars):\n  {args.text[:120]}...\n")
    print("Streaming from TTS server...\n")

    chunks: list[np.ndarray] = []
    sample_rate = None
    t0 = time.time()

    with requests.post(
        f"{TTS_URL}/tts/stream",
        json={"text": args.text},
        stream=True,
        timeout=600,
    ) as r:
        r.raise_for_status()
        buffer = ""
        for raw in r.iter_content(chunk_size=None, decode_unicode=True):
            buffer += raw
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                for line in event_str.split("\n"):
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])

                    if event["type"] == "chunk":
                        idx = event["index"]
                        arr, sr = b64_to_array(event["audio_base64"])
                        sample_rate = sr
                        chunks.append(arr)
                        duration = len(arr) / sr
                        elapsed = time.time() - t0
                        print(f"  chunk {idx}: {duration:.2f}s audio  (arrived at {elapsed:.1f}s)")

                        # Save individual chunk
                        chunk_path = OUT_DIR / f"chunk_{idx:02d}.wav"
                        sf.write(chunk_path, arr, sr)

                    elif event["type"] == "done":
                        elapsed = round(time.time() - t0, 1)
                        print(f"\nDone: {event['total_chunks']} chunks in {elapsed}s")

    if not chunks:
        print("ERROR: no audio chunks received.")
        return

    # Combine all chunks with 300ms silence between them
    silence = np.zeros(int(sample_rate * 0.3), dtype=np.float32)
    parts = []
    for i, c in enumerate(chunks):
        if i > 0:
            parts.append(silence)
        parts.append(c)
    combined = np.concatenate(parts)

    combined_path = OUT_DIR / "combined.wav"
    sf.write(combined_path, combined, sample_rate)
    total_audio_s = len(combined) / sample_rate
    print(f"\nSaved {len(chunks)} chunks → {combined_path}")
    print(f"Total audio: {total_audio_s:.1f}s  |  Synthesis time: {time.time()-t0:.1f}s")

    if not args.no_play:
        print("\nPlaying combined audio (afplay)...")
        subprocess.run(["afplay", str(combined_path)])
    else:
        print(f"\nSkipped playback. To listen:\n  afplay {combined_path}")


if __name__ == "__main__":
    main()
