FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

COPY pyproject.toml ./
COPY src ./src
COPY README.md ./

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000

CMD ["python", "-m", "med_routing.server"]
