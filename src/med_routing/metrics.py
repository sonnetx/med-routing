from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "medr_requests_total",
    "Total requests handled by the cascade.",
    ["router", "escalated"],
    registry=REGISTRY,
)

MODEL_CALLS_TOTAL = Counter(
    "medr_model_calls_total",
    "Underlying model API calls.",
    ["model", "tier"],
    registry=REGISTRY,
)

TOKENS_TOTAL = Counter(
    "medr_tokens_total",
    "Tokens consumed.",
    ["model", "kind"],
    registry=REGISTRY,
)

COST_USD_TOTAL = Counter(
    "medr_cost_usd_total",
    "Cumulative spend (USD) per model.",
    ["model"],
    registry=REGISTRY,
)

UNCERTAINTY = Histogram(
    "medr_uncertainty",
    "Distribution of router uncertainty scores in [0,1].",
    ["router"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

LATENCY_SECONDS = Histogram(
    "medr_latency_seconds",
    "End-to-end latency by router and tier reached.",
    ["router", "tier"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
    registry=REGISTRY,
)

ACCURACY = Gauge(
    "medr_accuracy",
    "Rolling accuracy from the eval runner.",
    ["router"],
    registry=REGISTRY,
)

ESCALATION_RATE = Gauge(
    "medr_escalation_rate",
    "Fraction of requests escalated to the strong model.",
    ["router"],
    registry=REGISTRY,
)


def record_usage(model: str, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
    TOKENS_TOTAL.labels(model=model, kind="prompt").inc(prompt_tokens)
    TOKENS_TOTAL.labels(model=model, kind="completion").inc(completion_tokens)
    COST_USD_TOTAL.labels(model=model).inc(cost)
