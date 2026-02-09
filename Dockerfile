# CUDA runtime base; good default for inference services
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip3 install -U pip \
 && pip3 install --no-cache-dir -r /app/requirements.txt

# Copy code
COPY . /app

# HuggingFace cache (mount this as a volume at runtime)
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_CACHE=/hf_cache

EXPOSE 8000

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
