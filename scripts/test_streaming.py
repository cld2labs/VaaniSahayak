#!/usr/bin/env python3
"""
Test SSE streaming at each layer of the Vaani stack.

Measures time-between-chunks to prove tokens arrive incrementally
(not buffered until completion).

Usage:
    python scripts/test_streaming.py              # test all layers
    python scripts/test_streaming.py param1       # test Param-1 server only
    python scripts/test_streaming.py backend      # test FastAPI backend only
    python scripts/test_streaming.py nginx        # test through nginx only
"""
import sys
import time
import json
import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARAM1_URL = "http://localhost:8001/v1/chat/completions"
BACKEND_URL = "http://localhost:8000/ask/stream"
NGINX_URL = "http://localhost:3002/ask/stream"

PARAM1_PAYLOAD = {
    "model": "bharatgenai/Param-1-2.9B-Instruct",
    "messages": [{"role": "user", "content": "Hello, what is PM-KISAN?"}],
    "stream": True,
    "max_tokens": 30,
}

BACKEND_PAYLOAD = {
    "query": "PM-KISAN योजना क्या है?",
    "language": "hi",
}


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def test_layer(name: str, url: str, payload: dict):
    print(f"\n{'=' * 60}")
    print(f" {name}")
    print(f" {url}")
    print(f"{'=' * 60}")

    try:
        httpx.get(url.rsplit("/", 1)[0] if "/v1/" in url else url.rsplit("/", 1)[0] + "/health", timeout=3)
    except Exception:
        pass  # health check is optional

    chunks = []   # list of (timestamp, line_text)
    t0 = time.time()

    try:
        with httpx.stream(
            "POST", url, json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                print(f"  ERROR: HTTP {resp.status_code}")
                print(f"  {resp.read().decode()[:200]}")
                return False

            buf = ""
            for raw in resp.iter_bytes():
                buf += raw.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: "):
                        chunks.append((time.time() - t0, line))
    except httpx.ReadTimeout:
        print("  ERROR: Read timeout (120s)")
        return False
    except httpx.ConnectError as e:
        print(f"  ERROR: Cannot connect — {e}")
        print(f"  Is the server running?")
        return False

    total = time.time() - t0

    if not chunks:
        print("  ERROR: No SSE chunks received")
        return False

    # --- Report ---
    print(f"\n  Chunks received: {len(chunks)}")
    print(f"  Total time:      {total:.2f}s")
    print(f"  First chunk at:  {chunks[0][0]:.3f}s")
    if len(chunks) > 1:
        print(f"  Last chunk at:   {chunks[-1][0]:.3f}s")

    # Show first 8 chunks with timestamps
    print(f"\n  First chunks (with timestamps):")
    for i, (t, line) in enumerate(chunks[:8]):
        # Truncate long lines
        display = line[:100] + ("..." if len(line) > 100 else "")
        print(f"    [{t:7.3f}s] {display}")
    if len(chunks) > 8:
        print(f"    ... ({len(chunks) - 8} more chunks)")
        # Show last chunk
        t, line = chunks[-1]
        display = line[:100] + ("..." if len(line) > 100 else "")
        print(f"    [{t:7.3f}s] {display}")

    # Compute inter-chunk gaps
    if len(chunks) >= 3:
        gaps = [chunks[i + 1][0] - chunks[i][0] for i in range(len(chunks) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        min_gap = min(gaps)

        print(f"\n  Inter-chunk timing:")
        print(f"    Average gap: {avg_gap:.3f}s")
        print(f"    Min gap:     {min_gap:.3f}s")
        print(f"    Max gap:     {max_gap:.3f}s")

        # Verdict
        # If average gap < 5ms and total > 1s, chunks arrived all at once (buffered)
        if avg_gap < 0.005 and total > 1.0:
            print(f"\n  VERDICT: NOT STREAMING")
            print(f"    Chunks arrived in a burst (avg {avg_gap*1000:.1f}ms apart)")
            print(f"    but total generation took {total:.1f}s.")
            print(f"    This means responses are buffered, not streamed.")
            return False
        else:
            print(f"\n  VERDICT: STREAMING OK")
            print(f"    Chunks arrive with real delays between them.")
            return True
    else:
        print(f"\n  VERDICT: Too few chunks ({len(chunks)}) to assess streaming")
        return len(chunks) > 0

    # Check for ASSISTANT tag leaking
    all_text = ""
    for _, line in chunks:
        raw = line[6:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            evt = json.loads(raw)
            # Backend format
            if "text" in evt:
                all_text += evt["text"]
            # OpenAI format
            elif "choices" in evt:
                delta = evt.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    all_text += delta["content"]
        except json.JSONDecodeError:
            pass

    leaked = []
    for tag in ["[ASSISTANT]", "[USER]", "[SYSTEM]"]:
        if tag in all_text:
            leaked.append(tag)
    if leaked:
        print(f"\n  WARNING: Leaked tags in output: {leaked}")
        print(f"  Full text: {all_text[:200]}...")
    else:
        print(f"\n  Tag leak check: CLEAN")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["param1", "backend", "nginx"]

    results = {}
    for target in targets:
        if target == "param1":
            results[target] = test_layer("Param-1 Server (direct)", PARAM1_URL, PARAM1_PAYLOAD)
        elif target == "backend":
            results[target] = test_layer("FastAPI Backend (direct)", BACKEND_URL, BACKEND_PAYLOAD)
        elif target == "nginx":
            results[target] = test_layer("Through Nginx (frontend path)", NGINX_URL, BACKEND_PAYLOAD)
        else:
            print(f"Unknown target: {target}. Use: param1, backend, nginx")

    print(f"\n{'=' * 60}")
    print(f" SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:10s} : {status}")


if __name__ == "__main__":
    main()
