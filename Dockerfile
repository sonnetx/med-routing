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
COPY demo_data ./demo_data
COPY README.md ./

# Base install only. The legacy semantic_entropy router (NLI / DeBERTa) is
# gated on ENABLE_NLI=true and would require pip install -e ".[nli]" — but the
# semantic_entropy_embed router uses OpenAI embeddings instead and needs only
# the base deps, so we skip the ~2GB torch+CUDA install.
RUN pip install --upgrade pip && pip install -e "."

EXPOSE 8000

# Start uvicorn directly. Going through `python -m med_routing.server` →
# uvicorn.run("med_routing.server:app") double-imports the module (once as
# __main__, once as med_routing.server), which can desync the FastAPI app's
# route registry from what tests expect.
CMD ["uvicorn", "med_routing.server:app", "--host", "0.0.0.0", "--port", "8000"]
