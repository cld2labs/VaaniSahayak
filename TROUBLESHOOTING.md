# Troubleshooting

Common issues and solutions for Vaani Sahayak.

## Table of Contents

- [Backend Issues](#backend-issues)
- [Model Server Issues](#model-server-issues)
- [Frontend Issues](#frontend-issues)
- [Docker Issues](#docker-issues)

---

## Backend Issues

### FastAPI won't start

**Symptom:** `ModuleNotFoundError` or import errors when running `uvicorn backend.main:app`

**Fix:**
```bash
# Ensure you're in the project root (not inside backend/)
cd /path/to/VaaniSahayak
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Missing scheme data

**Symptom:** `FileNotFoundError: schemes.json not found` or empty retrieval results

**Fix:**
```bash
python scripts/download_data.py
python scripts/precompute_embeddings.py
```

Verify files exist:
```bash
ls -la backend/data/schemes.json backend/data/scheme_embeddings.npy
```

### Embedding dimension mismatch

**Symptom:** `ValueError` during cosine similarity computation

**Fix:** Re-run the embedding script to regenerate `scheme_embeddings.npy`:
```bash
python scripts/precompute_embeddings.py
```

---

## Model Server Issues

### Param-1 server crashes or runs out of memory

**Symptom:** `torch.mps.OutOfMemoryError` or server becomes unresponsive

**Fix:**
- Ensure at least 16 GB RAM is available
- Close other memory-heavy applications
- Use the `/suspend` endpoint to offload weights when TTS needs memory:
  ```bash
  curl -X POST http://localhost:8001/suspend
  ```
- Resume when ready:
  ```bash
  curl -X POST http://localhost:8001/resume
  ```

### Param-1 model not found on HuggingFace

**Symptom:** `404` or authentication error when downloading the model

**Fix:**
1. Ensure `HF_TOKEN` is set in your `.env` file
2. Accept the model license at [bharatgenai/Param-1-2.9B-Instruct](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct)
3. Login via CLI:
   ```bash
   huggingface-cli login
   ```

### TTS server slow or producing garbled audio

**Symptom:** Audio takes >15 seconds or sounds distorted

**Fix:**
- Ensure `servers/server_tts.py` is using MPS:
  ```bash
  python servers/server_tts.py --preload --port 8003
  # Should log: "Using device: mps"
  ```
- If MPS is unavailable, it falls back to CPU (much slower)
- Keep input text under 800 characters for reliable synthesis

### TTS server won't load the model

**Symptom:** `OSError` or `RuntimeError` when loading Indic Parler-TTS

**Fix:**
```bash
# Ensure parler-tts is installed from source
pip install git+https://github.com/huggingface/parler-tts.git
```

---

## Frontend Issues

### API connection refused

**Symptom:** `Network Error` or `ERR_CONNECTION_REFUSED` in the browser console

**Fix:**
1. Verify the backend is running: `curl http://localhost:8000/health`
2. Check that `vite.config.js` proxies `/api` to `http://localhost:8000`
3. Ensure no firewall is blocking localhost ports

### Build fails

**Symptom:** `npm run build` fails with errors

**Fix:**
```bash
cd frontend
rm -rf node_modules
npm install
npm run build
```

### Audio not playing

**Symptom:** Response text appears but no audio plays

**Fix:**
1. Check that the TTS server is running: `curl http://localhost:8003/health`
2. Ensure the backend can reach the TTS server (check `TTS_SERVER_URL` in `.env`)
3. Check browser console for audio decoding errors
4. Try the `/narrate` endpoint directly to isolate the issue

### Streaming response stops mid-way

**Symptom:** Text appears partially, then stops

**Fix:**
- Check `servers/server_param1.py` logs for timeout or memory errors
- Ensure the Param-1 server hasn't been suspended (`/suspend`)
- Reduce query complexity to stay within the 2,048-token context window

---

## Docker Issues

### Backend can't reach model servers

**Symptom:** `Connection refused` errors for port 8001 or 8003

**Fix:** Model servers run natively (not in Docker). Ensure Docker services use `host.docker.internal`:
```yaml
# docker-compose.yml
environment:
  VLLM_LLM_URL: http://host.docker.internal:8001/v1
  TTS_SERVER_URL: http://host.docker.internal:8003
```

### Container runs out of memory

**Symptom:** Container killed by OOM or exits with code 137

**Fix:**
- Increase Docker Desktop memory allocation (Settings → Resources → Memory)
- The backend container loads embeddings in-process — allocate at least 4 GB

---

Still stuck? Open an [issue](https://github.com/cld2labs/VaaniSahayak/issues) with your error logs and environment details.
