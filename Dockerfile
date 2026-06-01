# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# PII-Safe LLM Chat — application image
#
# spaCy models (esp. zh_core_web_trf, a ~400MB BERT) are baked into the image
# at build time so the container starts offline and fast. boto3 reads AWS
# credentials from a read-only mount of the host's ~/.aws (see compose file).
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libs occasionally needed by torch / tokenizers wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download spaCy models into the image (no network needed at runtime).
RUN python -m spacy download zh_core_web_trf \
    && python -m spacy download en_core_web_sm

# Application code.
COPY . .

# Runtime data dirs (also mounted as volumes in compose for persistence).
RUN mkdir -p sessions uploads

EXPOSE 8000

# Single worker by default. With REDIS_URL set (see compose), you can safely
# raise the worker count because job state is shared via Redis.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
