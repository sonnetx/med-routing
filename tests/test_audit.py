from __future__ import annotations

import json
from pathlib import Path

import pytest

from med_routing.audit import AuditLogger
from med_routing.cache import CompletionCache
from med_routing.cascade import CascadeController
from med_routing.metrics import CROSS_BORDER_TOTAL, PROCESSOR_CALLS_TOTAL
from med_routing.processors import get_processor, is_cross_border
from med_routing.routers.self_consistency import SelfConsistencyRouter
from tests.conftest import FakeOpenAIClient, make_completion


def test_processor_registry_known_models():
    assert get_processor("gpt-4o-mini").entity == "OpenAI, Inc."
    assert get_processor("gpt-4o-mini").region == "US"
    assert get_processor("local-llama").region == "EU"


def test_processor_registry_unknown_model_marks_unknown():
    p = get_processor("totally-made-up-model")
    assert p.region == "unknown"
    assert p.entity == "unknown"


@pytest.mark.parametrize(
    "region,home,expected",
    [("US", "US", False), ("EU", "US", True), ("unknown", "US", True), ("EU", "EU", False)],
)
def test_is_cross_border(region, home, expected):
    assert is_cross_border(region, home) is expected


def test_audit_logger_writes_jsonl_and_keeps_recent(tmp_path: Path):
    audit = AuditLogger(root=tmp_path, recent_size=10)
    audit.log({"router": "r", "score": 0.3, "escalated": False})
    audit.log({"router": "r", "score": 0.6, "escalated": True})

    files = sorted(tmp_path.glob("decisions-*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert rows[0]["score"] == 0.3
    assert rows[1]["escalated"] is True

    assert len(audit.recent()) == 2
    assert audit.recent(limit=1)[-1]["score"] == 0.6
    audit.close()


def _messages():
    return [{"role": "user", "content": "Q"}]


async def test_cascade_writes_audit_row(tmp_path: Path):
    client = FakeOpenAIClient()
    weak = make_completion("B", model="gpt-4o-mini")
    client.script(model="gpt-4o-mini", n=1, completions=[weak])
    client.script(model="gpt-4o-mini", n=5, completions=[make_completion("B") for _ in range(5)])

    audit = AuditLogger(root=tmp_path)
    controller = CascadeController(
        client=client,
        routers={"self_consistency": SelfConsistencyRouter()},
        cache=CompletionCache(maxsize=64),
        audit=audit,
    )
    await controller.handle(_messages(), "self_consistency")

    rows = audit.recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["router"] == "self_consistency"
    assert "prompt_sha" in row and len(row["prompt_sha"]) == 40  # sha1 hex
    assert "content" not in json.dumps(row)  # raw prompt MUST NOT be in audit
    assert row["weak_processor"] == "openai-gpt-4o-mini"
    assert row["weak_region"] == "US"
    assert row["escalated"] is False
    assert row["regions_touched"] == ["US"]
    audit.close()


async def test_cross_border_counter_increments_when_home_region_is_eu(tmp_path: Path, monkeypatch):
    # Force home region to EU; gpt-4o-mini is in US, so every call is cross-border.
    monkeypatch.setenv("HOME_REGION", "EU")
    from med_routing.config import get_settings as gs
    gs.cache_clear()  # force re-read

    before = CROSS_BORDER_TOTAL.labels(
        router="self_consistency", home_region="EU", foreign_region="US"
    )._value.get()

    client = FakeOpenAIClient()
    client.script(model="gpt-4o-mini", n=1, completions=[make_completion("B")])
    client.script(model="gpt-4o-mini", n=5, completions=[make_completion("B") for _ in range(5)])

    audit = AuditLogger(root=tmp_path)
    controller = CascadeController(
        client=client,
        routers={"self_consistency": SelfConsistencyRouter()},
        cache=CompletionCache(maxsize=64),
        audit=audit,
    )
    await controller.handle(_messages(), "self_consistency")

    after = CROSS_BORDER_TOTAL.labels(
        router="self_consistency", home_region="EU", foreign_region="US"
    )._value.get()
    assert after - before == 1

    row = audit.recent()[-1]
    assert row["cross_border"] is True
    assert row["home_region"] == "EU"

    monkeypatch.delenv("HOME_REGION", raising=False)
    gs.cache_clear()
    audit.close()


async def test_processor_calls_counter_increments(tmp_path: Path):
    """Every model call should bump PROCESSOR_CALLS_TOTAL with the right region."""
    before = PROCESSOR_CALLS_TOTAL.labels(
        processor="openai-gpt-4o-mini", entity="OpenAI, Inc.", region="US"
    )._value.get()

    client = FakeOpenAIClient()
    client.script(model="gpt-4o-mini", n=1, completions=[make_completion("B")])
    client.script(model="gpt-4o-mini", n=5, completions=[make_completion("B") for _ in range(5)])

    audit = AuditLogger(root=tmp_path)
    controller = CascadeController(
        client=client,
        routers={"self_consistency": SelfConsistencyRouter()},
        cache=CompletionCache(maxsize=64),
        audit=audit,
    )
    await controller.handle(_messages(), "self_consistency")

    after = PROCESSOR_CALLS_TOTAL.labels(
        processor="openai-gpt-4o-mini", entity="OpenAI, Inc.", region="US"
    )._value.get()
    # 1 weak call + 5 sample calls = 6
    assert after - before == 6
    audit.close()
