# Indic Parler-TTS — CUDA GPU Deployment

Deploy [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) on a Kubernetes cluster with NVIDIA GPU for production-grade Hindi/Telugu text-to-speech.

## What's Inside

```
Cuda/
├── server_tts_cuda.py      # FastAPI TTS server (optimized for CUDA)
├── text_normalize.py       # Hindi text normalization (numbers, currency, English → Hindi)
├── requirements.txt        # Pinned Python deps
├── Dockerfile              # NVIDIA CUDA 12.1 + PyTorch 2.3.1 image
└── k8s/
    ├── tts-deployment.yaml   # K8s Deployment (1 GPU, env vars, HF token)
    ├── tts-service.yaml      # ClusterIP Service on port 8003
    └── tts-apisix-route.yaml # APISIX gateway route (/v1/tts → pod)
```

## Prerequisites

- Kubernetes cluster with at least **1 NVIDIA GPU node** (V100, A10, A100, or H100)
- **NVIDIA device plugin** installed (`nvidia.com/gpu` resource available)
- **APISIX Ingress Controller** (for external routing) or any ingress
- **HuggingFace token** with access to `ai4bharat/indic-parler-tts`
- `kubectl` and `docker` CLI tools

## Quick Deploy

### 1. Create the HuggingFace token secret

```bash
kubectl create secret generic hf-token \
  --from-literal=HUGGINGFACEHUB_API_TOKEN=hf_YOUR_TOKEN_HERE
```

### 2. Build and load the Docker image

```bash
# Build on the GPU machine (or wherever Docker runs)
cd Cuda/
docker build -t tts:local .

# If using a remote registry:
# docker tag tts:local your-registry.com/tts:latest
# docker push your-registry.com/tts:latest
# Then update tts-deployment.yaml image + imagePullPolicy
```

**Build time:** ~10-15 minutes (downloads PyTorch + parler-tts + dependencies).

Model weights are **not** baked into the image — they download from HuggingFace on first startup and cache in the container. For faster cold starts, mount a PersistentVolumeClaim at `/root/.cache/huggingface`.

### 3. Apply Kubernetes manifests

```bash
kubectl apply -f k8s/tts-deployment.yaml
kubectl apply -f k8s/tts-service.yaml
kubectl apply -f k8s/tts-apisix-route.yaml   # optional: exposes via APISIX gateway
```

### 4. Verify startup

```bash
# Watch pod logs — look for the READY banner
kubectl logs -f deployment/tts-deployment

# Expected startup log:
# [TTS] GPU: Tesla V100-SXM2-16GB (16.9 GB)
# [TTS] Trying attention backend: split (decoder=sdpa, encoder=eager)
# [TTS] ✓ split (decoder=sdpa, encoder=eager) loaded successfully
# [TTS] ✓ use_cache=True probe PASSED — KV cache enabled
# [TTS] ✓ Static cache enabled and verified
# [TTS] READY
# [TTS]   attn=split (decoder=sdpa, encoder=eager)
# [TTS]   compiled=False, use_cache=True, static_cache=True

# Health check
kubectl exec deployment/tts-deployment -- curl -s localhost:8003/health
```

### 5. Test synthesis

```bash
# Direct pod test
kubectl exec deployment/tts-deployment -- curl -s -X POST localhost:8003/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते, यह एक परीक्षण है।"}'

# Via APISIX gateway (if configured)
curl -X POST https://api.example.com/v1/tts \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते।"}'

# Batch mode (server splits, synthesizes, and stitches)
curl -X POST https://api.example.com/v1/tts \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "पहला वाक्य। दूसरा वाक्य।", "batch": true}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tts` | Synthesize text → `{"audio_base64": "...", "elapsed": 5.2}` |
| `POST` | `/tts` (batch=true) | Split + synthesize + stitch → single WAV |
| `POST` | `/v1/tts` | Alias for `/tts` (matches APISIX route) |
| `GET`  | `/health` | `{"model_loaded": true, "compiled": false, "use_cache": true, "static_cache": true}` |

### Request Schema

```json
{
  "text": "Hindi text to synthesize",
  "batch": false
}
```

- `text` (required): Hindi/Telugu text. Automatically normalized (numbers→words, ₹→rupees, English→Hindi).
- `batch` (optional, default `false`): When `true`, server splits text on sentence boundaries (`।.!?`), synthesizes each sentence, and returns a single concatenated WAV.

### Response Schema

```json
{
  "audio_base64": "UklGR...",
  "elapsed": 5.23,
  "sentence_count": 3
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | **Required.** HuggingFace access token |
| `TTS_MODEL_ID` | `ai4bharat/indic-parler-tts` | Model to load |
| `TTS_PORT` | `8003` | Server port |
| `TTS_ATTN_IMPL` | `auto` | Attention backend: `auto` (recommended), `eager`, `sdpa`, `flash_attention_2` |
| `TTS_COMPILE_MODE` | `default` | torch.compile mode: `default`, `reduce-overhead`, `none` |
| `TTS_MAX_CONCURRENT` | `1` | Max concurrent synthesis requests (GPU memory limited) |
| `TOKENIZERS_PARALLELISM` | `false` | Set `false` to suppress tokenizer fork warnings |

### Attention Backend Selection (`TTS_ATTN_IMPL=auto`)

When set to `auto`, the server tries backends in order and logs results:

```
[TTS] Trying attention backend: split (decoder=sdpa, encoder=eager)
[TTS] ✓ split loaded successfully             ← best for V100
  — or —
[TTS] ✗ split FAILED: ValueError: T5EncoderModel does not support SDPA
[TTS] Trying attention backend: flash_attention_2
[TTS] ✗ flash_attention_2 FAILED: flash_attn not installed
[TTS] Trying attention backend: eager
[TTS] ✓ eager loaded successfully              ← fallback
```

| GPU | Best Backend | Notes |
|-----|-------------|-------|
| V100 | split (decoder=sdpa, encoder=eager) | SDPA on decoder, eager on T5 encoder |
| A10/A100/H100 | flash_attention_2 | Install `flash-attn` package |

### Static Cache + torch.compile

The server probes static cache at startup. If available (parler-tts git HEAD), it enables fixed-shape KV cache. On Ampere+ GPUs, this unlocks `torch.compile` without recompilation penalty. On V100, compile is automatically skipped.

## Optimization Summary

| Optimization | V100 | A10/A100/H100 |
|---|---|---|
| Split attention (SDPA decoder) | Yes | Yes |
| Flash Attention 2 | No (Volta) | Yes |
| KV cache (`use_cache=True`) | Yes | Yes |
| Static cache | Yes | Yes |
| torch.compile | No (recompilation) | Yes |
| float16 | Yes | Yes |
| Pre-encoded description | Yes | Yes |
| Warmup inference | Yes | Yes |

## Text Normalization

All text is automatically normalized before synthesis:

| Input | Output |
|-------|--------|
| `₹10,000` | `दस हज़ार रुपये` |
| `2023` | `दो हज़ार तेईस` |
| `PM-KISAN` | `पीएम किसान` |
| `firstname@gmail.com` | *(removed)* |
| `50%` | `पचास प्रतिशत` |
| `PMAY` | `पीएमएवाई` |
| `Online` | `ऑनलाइन` |
| `DRDO` | `डीआरडीओ` |

100+ English terms have Hindi transliterations. Remaining uppercase acronyms are spelled letter-by-letter. Unknown English words are removed.

## Connecting to Vaani Backend

On the Vaani backend machine, set these in `.env`:

```bash
# Direct access (if pod is reachable)
EI_TTS_URL=http://<cluster-ip>:8003/tts

# Via APISIX gateway (recommended for production)
EI_TTS_URL=http://api.example.com:32237/v1/tts

# Authentication (pick one)
EI_TTS_TOKEN=<static-bearer-token>
# — or —
EI_KEYCLOAK_URL=https://api.example.com
EI_KEYCLOAK_REALM=master
EI_CLIENT_ID=api
EI_CLIENT_SECRET=<secret>
```

The Vaani backend automatically detects EI availability and routes TTS requests to the GPU server.

## Troubleshooting

**Pod stuck in `Pending`:**
```bash
kubectl describe pod -l app=tts
# Look for: "Insufficient nvidia.com/gpu" → no GPU node available
```

**Model download slow/failing:**
```bash
# Check HF token is valid
kubectl exec deployment/tts-deployment -- python3 -c "from huggingface_hub import HfApi; HfApi().whoami()"

# Mount a PVC for model cache to avoid re-downloading
# Add to tts-deployment.yaml:
# volumeMounts:
#   - name: hf-cache
#     mountPath: /root/.cache/huggingface
```

**500 errors on synthesis:**
```bash
kubectl logs deployment/tts-deployment --tail=50
# Look for: OOM, CUDA errors, static cache issues
```

**Scale down to save GPU costs:**
```bash
kubectl scale deployment tts-deployment --replicas=0
# Scale back up:
kubectl scale deployment tts-deployment --replicas=1
```
