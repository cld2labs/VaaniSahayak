FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# ─────────────────────────────────────────
# SYSTEM DEPENDENCIES
# ─────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────
# PYTHON OPTIMIZATION
# ─────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─────────────────────────────────────────
# INSTALL TORCH WITH CUDA (CRITICAL FIX)
# ─────────────────────────────────────────
RUN pip3 install --upgrade pip

# Install CUDA-compatible torch FIRST
RUN pip3 install --no-cache-dir \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ─────────────────────────────────────────
# INSTALL OTHER DEPENDENCIES
# ─────────────────────────────────────────
COPY requirements.txt .

RUN pip3 install --no-cache-dir \
    --use-deprecated=legacy-resolver \
    -r requirements.txt

# ─────────────────────────────────────────
# APP CODE
# ─────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────
# ENV
# ─────────────────────────────────────────
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

ENV TTS_PORT=8003

# ─────────────────────────────────────────
# START
# ─────────────────────────────────────────
CMD ["python3", "server_tts_cuda.py"]
