FROM python:3.11-slim

WORKDIR /app

# System deps for soundfile, sentencepiece, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (context root = project root)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (project root as build context)
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY servers/text_normalize.py ./

ENV PYTHONPATH=/app

# param1-server runs on the host (MPS); backend calls it via host.docker.internal
ENV VLLM_LLM_URL=http://host.docker.internal:8001/v1

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
