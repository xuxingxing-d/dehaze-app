# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY backend/requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r /app/requirements.txt

# Optionally install PyTorch in image (choose one matching your GPU/driver)
# Uncomment one of the following lines or install at runtime:
# RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy app code
COPY backend/app.py /app/app.py
COPY backend/models /app/models

EXPOSE 3000

ENV MODEL_PATH=/app/models/model.pth \
    MODEL_TYPE=torchscript \
    USE_HALF=0 \
    PORT=3000

CMD ["python3", "app.py"]


