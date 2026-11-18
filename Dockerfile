FROM python:3.12-slim-bookworm AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    WHISPERX_DEVICE=cpu \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt \
    && python -m pip check
COPY app.py model_config.py run_logging.py runtime_setup.py transcribe_worker.py ./
COPY templates/ ./templates/
EXPOSE 8000
CMD ["python", "app.py"]

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    HOST=0.0.0.0 \
    WHISPERX_DEVICE=cuda \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install -r requirements.txt \
    && python -m pip check
COPY app.py model_config.py run_logging.py runtime_setup.py transcribe_worker.py ./
COPY templates/ ./templates/
EXPOSE 8000
CMD ["python", "app.py"]
