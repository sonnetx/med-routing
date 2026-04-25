FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# build-essential is needed by sentencepiece on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY README.md ./

# Install with the [nli] extra so semantic_entropy can use DeBERTa-v3-MNLI.
# torch + transformers + sentencepiece add ~2GB but the model itself is fetched
# at runtime into the mounted hf_cache volume.
RUN pip install --upgrade pip && pip install -e ".[nli]"

EXPOSE 8000

CMD ["python", "-m", "med_routing.server"]
