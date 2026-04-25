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

ACTUAL_COST_BY_ROUTER = Counter(
    "medr_actual_cost_usd_total",
    "Cumulative actual spend per router (weak + any strong escalation).",
    ["router"],
    registry=REGISTRY,
)

COUNTERFACTUAL_COST_BY_ROUTER = Counter(
    "medr_counterfactual_cost_usd_total",
    "What the request would have cost if every query went straight to the strong model. "
    "Difference vs actual is the routing savings.",
    ["router"],
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

# Calibration: bin every observation by uncertainty score and track correctness.
# Reliability per bin = bin_correct / bin_total. A good router has high weak-accuracy
# in low-uncertainty bins and low weak-accuracy in high-uncertainty bins.
CALIBRATION_BINS = (
    "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
    "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0",
)

CALIBRATION_BIN_TOTAL = Counter(
    "medr_calibration_bin_total",
    "Observations in each uncertainty bin (for reliability diagram).",
    ["router", "bin"],
    registry=REGISTRY,
)

CALIBRATION_BIN_CORRECT = Counter(
    "medr_calibration_bin_correct",
    "Correct observations in each uncertainty bin.",
    ["router", "bin"],
    registry=REGISTRY,
)

ECE = Gauge(
    "medr_ece",
    "Expected calibration error: weighted |(1 - bin_midpoint_uncertainty) - bin_accuracy|.",
    ["router"],
    registry=REGISTRY,
)

KEPT_WEAK_ACCURACY = Gauge(
    "medr_kept_weak_accuracy",
    "Accuracy on questions where the router did NOT escalate (weak model only).",
    ["router"],
    registry=REGISTRY,
)

ESCALATED_ACCURACY = Gauge(
    "medr_escalated_accuracy",
    "Accuracy on questions where the router escalated to the strong model.",
    ["router"],
    registry=REGISTRY,
)

ACCURACY_BY_SUBJECT = Gauge(
    "medr_accuracy_by_subject",
    "Rolling accuracy partitioned by MedMCQA subject.",
    ["router", "subject"],
    registry=REGISTRY,
)


def score_bin(score: float) -> str:
    """Map a score in [0,1] to one of CALIBRATION_BINS. Right-edge inclusive on 1.0."""
    s = max(0.0, min(1.0, float(score)))
    idx = min(int(s * 10), 9)
    return CALIBRATION_BINS[idx]


def bin_midpoint(bin_label: str) -> float:
    lo, hi = bin_label.split("-")
    return (float(lo) + float(hi)) / 2.0


def record_usage(model: str, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
    TOKENS_TOTAL.labels(model=model, kind="prompt").inc(prompt_tokens)
    TOKENS_TOTAL.labels(model=model, kind="completion").inc(completion_tokens)
    COST_USD_TOTAL.labels(model=model).inc(cost)


# GDPR / data-residency observability. Every model call is one egress event from
# the wrapper to a third-party processor; cross-border counts when the call
# leaves the configured home region.
PROCESSOR_CALLS_TOTAL = Counter(
    "medr_processor_calls_total",
    "Calls to each underlying processor, by region (Art. 30 records of processing).",
    ["processor", "entity", "region"],
    registry=REGISTRY,
)

CROSS_BORDER_TOTAL = Counter(
    "medr_cross_border_total",
    "Requests where data left the configured home region.",
    ["router", "home_region", "foreign_region"],
    registry=REGISTRY,
)
