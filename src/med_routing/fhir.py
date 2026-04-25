from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

# HL7 FHIR R4 AuditEvent (https://www.hl7.org/fhir/auditevent.html).
#
# Each cascade decision is converted to an AuditEvent on read — we don't store
# a third copy. This shape drops directly into a FHIR Bulk Data export
# (`application/fhir+ndjson`) that any conformant EHR (Epic, Cerner, OpenEMR)
# will ingest into its audit module unchanged.

_AUDIT_TYPE = {
    "system": "http://terminology.hl7.org/CodeSystem/audit-event-type",
    "code": "rest",
    "display": "RESTful Operation",
}

_SUBTYPE_AI_INFERENCE = {
    "system": "urn:med-routing:audit-subtype",
    "code": "llm-cascade-decision",
    "display": "LLM cascade routing decision",
}

# RoleClass codes from HL7 v3.
_ROLE_PROCESSOR = {
    "coding": [
        {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ParticipationType",
            "code": "RESP",
            "display": "responsible party",
        }
    ]
}
_ROLE_DATAPROCESSOR = {
    "coding": [
        {
            "system": "urn:med-routing:audit-role",
            "code": "DATA-PROCESSOR",
            "display": "GDPR Article 28 data processor",
        }
    ]
}


def _detail(name: str, value: Any) -> dict[str, Any]:
    """An entity.detail entry. Use valueString — broadly supported across EHRs."""
    return {"type": name, "valueString": "" if value is None else str(value)}


def decision_to_audit_event(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one cascade decision into an HL7 FHIR R4 AuditEvent resource."""
    recorded = row.get("ts") or dt.datetime.now(dt.timezone.utc).isoformat()
    escalated = bool(row.get("escalated"))
    cross_border = bool(row.get("cross_border"))

    agents: list[dict[str, Any]] = [
        {
            "type": _ROLE_PROCESSOR,
            "who": {"display": "med-routing cascade controller"},
            "requestor": False,
            "name": "med-routing",
        },
    ]

    # Prefer the 3-tier `tier_chain` JSON. Fall back to legacy weak/strong
    # columns so AuditEvents continue to render for rows written by the older
    # 2-tier code.
    chain = row.get("tier_chain") or []
    if chain:
        for visit in chain:
            agents.append({
                "type": _ROLE_DATAPROCESSOR,
                "who": {"display": f"{visit.get('entity','')} — {visit.get('processor','')}".strip(" —")},
                "requestor": False,
                "name": str(visit.get("processor") or ""),
                "altId": str(visit.get("dpa_ref") or ""),
                "extension": [
                    {"url": "urn:med-routing:region", "valueString": str(visit.get("region") or "")},
                    {"url": "urn:med-routing:tier", "valueString": str(visit.get("tier_name") or "")},
                    {"url": "urn:med-routing:tier-index", "valueString": str(visit.get("tier_index"))},
                    {"url": "urn:med-routing:model", "valueString": str(visit.get("model") or "")},
                ],
            })
    else:
        agents.append({
            "type": _ROLE_DATAPROCESSOR,
            "who": {"display": f"{row.get('weak_entity', '')} — {row.get('weak_processor', '')}".strip(" —")},
            "requestor": False,
            "name": str(row.get("weak_processor") or ""),
            "altId": str(row.get("weak_dpa_ref") or ""),
            "extension": [
                {"url": "urn:med-routing:region", "valueString": str(row.get("weak_region") or "")},
                {"url": "urn:med-routing:tier", "valueString": "weak"},
            ],
        })
        if escalated and row.get("strong_processor"):
            agents.append({
                "type": _ROLE_DATAPROCESSOR,
                "who": {"display": f"strong-tier — {row.get('strong_processor')}"},
                "requestor": False,
                "name": str(row.get("strong_processor")),
                "extension": [
                    {"url": "urn:med-routing:region", "valueString": str(row.get("strong_region") or "")},
                    {"url": "urn:med-routing:tier", "valueString": "strong"},
                ],
            })

    entity = {
        "what": {
            "identifier": {
                "system": "urn:med-routing:prompt-sha1",
                "value": str(row.get("prompt_sha") or ""),
            }
        },
        "type": {
            "system": "http://terminology.hl7.org/CodeSystem/audit-entity-type",
            "code": "2",
            "display": "System Object",
        },
        "role": {
            "system": "http://terminology.hl7.org/CodeSystem/object-role",
            "code": "4",
            "display": "Domain Resource",
        },
        "name": "prompt",
        "description": "Prompt content recorded by SHA-1 only (data minimization, GDPR Art. 5(1)(c)).",
        "detail": [
            _detail("router", row.get("router")),
            _detail("score", row.get("score")),
            _detail("threshold", row.get("threshold")),
            _detail("escalated", "true" if escalated else "false"),
            _detail("final_tier_index", row.get("final_tier_index")),
            _detail("final_model", row.get("final_model")),
            _detail("home_region", row.get("home_region")),
            _detail("regions_touched", ",".join(row.get("regions_touched") or [])),
            _detail("cross_border", "true" if cross_border else "false"),
            _detail("tokens_prompt", row.get("tokens_prompt")),
            _detail("tokens_completion", row.get("tokens_completion")),
            _detail("cost_usd", row.get("cost_usd")),
            _detail("counterfactual_usd", row.get("counterfactual_usd")),
            _detail("latency_ms", row.get("latency_ms")),
        ],
    }

    return {
        "resourceType": "AuditEvent",
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{recorded}|{row.get('prompt_sha','')}|{row.get('router','')}")),
        "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/AuditEvent"]},
        "type": _AUDIT_TYPE,
        "subtype": [_SUBTYPE_AI_INFERENCE],
        "action": "E",  # Execute
        "recorded": recorded,
        "outcome": "0",  # success — failures would be 4 (minor) / 8 (serious) / 12 (major)
        "outcomeDesc": (
            f"committed_at_tier_{row.get('final_tier_index')}"
            if row.get("final_tier_index") is not None
            else ("escalated_to_strong" if escalated else "kept_weak")
        ),
        "purposeOfEvent": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason",
                        "code": "TREAT",
                        "display": "treatment",
                    }
                ]
            }
        ],
        "agent": agents,
        "source": {
            "site": str(row.get("home_region") or ""),
            "observer": {"display": "med-routing v0.1.0"},
            "type": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/security-source-type",
                    "code": "4",
                    "display": "Application Server",
                }
            ],
        },
        "entity": [entity],
    }


def to_bundle(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap AuditEvents in a FHIR Bundle of type 'collection'."""
    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "collection",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total": len(events),
        "entry": [{"resource": e} for e in events],
    }
