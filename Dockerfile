FROM python:3.10-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY glacier_mapping/ glacier_mapping/
RUN uv pip install --system --no-cache-dir -e .

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY glacier_mapping/ glacier_mapping/
COPY scripts/ scripts/

# Install deps (gradio for demo, others for training/preprocessing)
RUN uv pip install --system --no-cache-dir -e ".[dev]" gradio
# Force CPU-only torch (uv resolves CUDA from PyPI, so we override)
RUN pip install --no-cache-dir --force-reinstall torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "typing-extensions>=4.12.0"

# Demo data (baked in for self-contained image)
COPY demo_data/ demo_data/

ENV PYTHONUNBUFFERED=1
ENV GRADIO_HOST=0.0.0.0
ENV GRADIO_PORT=7860
ENV GRADIO_SHARE=0

# Mount checkpoints at /checkpoints for production
# or override PROCESSED_DIR for full dataset
ENV CKPT_DIR=/checkpoints

EXPOSE 7860

ENTRYPOINT ["python", "scripts/app_gradio.py"]
