from __future__ import annotations

import pytest

from med_routing.fhir import decision_to_audit_event, to_bundle


def _row(**overrides):
    base = {
        "ts": "2026-04-25T18:00:00+00:00",
        "prompt_sha": "deadbeef" * 5,
        "router": "self_consistency",
        "score": 0.42,
        "threshold": 0.4,
        "escalated": True,
        "weak_model": "gpt-4o-mini",
        "weak_processor": "openai-gpt-4o-mini",
        "weak_entity": "OpenAI, Inc.",
        "weak_region": "US",
        "weak_dpa_ref": "OpenAI DPA 2024",
        "strong_model": "gpt-4o",
        "strong_processor": "openai-gpt-4o",
        "strong_region": "US",
        "final_model": "gpt-4o",
        "final_region": "US",
        "home_region": "US",
        "regions_touched": ["US"],
        "cross_border": False,
        "tokens_prompt": 100,
        "tokens_completion": 5,
        "cost_usd": 0.0001,
        "counterfactual_usd": 0.00012,
        "latency_ms": 540,
    }
    base.update(overrides)
    return base


def test_audit_event_required_r4_fields():
    ev = decision_to_audit_event(_row())
    # FHIR R4 AuditEvent required cardinality.
    assert ev["resourceType"] == "AuditEvent"
    assert "type" in ev and "code" in ev["type"]
    assert "recorded" in ev and ev["recorded"].startswith("2026-04-25")
    assert isinstance(ev["agent"], list) and len(ev["agent"]) >= 1
    assert "source" in ev and "observer" in ev["source"]


def test_audit_event_outcome_reflects_escalation():
    kept = decision_to_audit_event(_row(escalated=False))
    esc = decision_to_audit_event(_row(escalated=True))
    assert kept["outcomeDesc"] == "kept_weak"
    assert esc["outcomeDesc"] == "escalated_to_strong"


def test_audit_event_does_not_leak_raw_prompt():
    """The whole point: the FHIR resource references the prompt by SHA only."""
    row = _row()
    ev = decision_to_audit_event(row)
    serialized = str(ev)
    assert "deadbeefdeadbeefdeadbeef" in serialized  # the SHA itself is fine
    # If we ever start putting the raw prompt in the row, this guards against
    # accidentally leaking it through the FHIR shape.
    assert "raw_prompt" not in serialized


def test_audit_event_includes_processor_agent_with_dpa():
    ev = decision_to_audit_event(_row())
    weak_agent = next(a for a in ev["agent"] if a.get("name") == "openai-gpt-4o-mini")
    assert weak_agent["altId"] == "OpenAI DPA 2024"
    region_ext = next(e for e in weak_agent["extension"] if e["url"] == "urn:med-routing:region")
    assert region_ext["valueString"] == "US"


def test_audit_event_strong_agent_only_when_escalated():
    kept = decision_to_audit_event(_row(escalated=False, strong_processor=None))
    esc = decision_to_audit_event(_row(escalated=True))
    kept_names = {a.get("name") for a in kept["agent"]}
    esc_names = {a.get("name") for a in esc["agent"]}
    assert "openai-gpt-4o" not in kept_names
    assert "openai-gpt-4o" in esc_names


@pytest.mark.parametrize("key,expected", [
    ("router", "self_consistency"),
    ("escalated", "true"),
    ("cost_usd", "0.0001"),
    ("counterfactual_usd", "0.00012"),
    ("regions_touched", "US"),
])
def test_audit_event_entity_detail_fields(key, expected):
    ev = decision_to_audit_event(_row())
    details = {d["type"]: d["valueString"] for d in ev["entity"][0]["detail"]}
    assert details[key] == expected


def test_audit_event_id_is_deterministic_for_same_decision():
    """Same prompt+router+ts → same UUID. Useful for idempotent EHR ingestion."""
    a = decision_to_audit_event(_row())
    b = decision_to_audit_event(_row())
    assert a["id"] == b["id"]


def test_bundle_wraps_events():
    events = [decision_to_audit_event(_row(router=f"r{i}")) for i in range(3)]
    bundle = to_bundle(events)
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert bundle["total"] == 3
    assert len(bundle["entry"]) == 3
    assert bundle["entry"][0]["resource"]["resourceType"] == "AuditEvent"
