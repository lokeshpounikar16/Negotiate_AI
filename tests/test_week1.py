import asyncio

import pytest

from Negotiate_AI.core.schema import (
    NegotiationMessage,
    ProposalPayload,
    QualityTier,
    TaskAnnouncement,
    TaskConstraints,
    MessageType,
)


@pytest.fixture
def task():
    return TaskAnnouncement(
        title="Hardware procurement",
        description="Buy 100 servers for a lab workload.",
        constraints=TaskConstraints(
            max_budget_usd=500000,
            max_delivery_days=60,
            min_quality_tier=QualityTier.B,
            max_vendor_risk=0.5,
            min_warranty_months=12,
            quantity=100,
        ),
        max_rounds=6,
    )


def test_task_constraints_reject_over_budget(task):
    proposal = ProposalPayload(
        vendor="Test Vendor",
        unit_price_usd=6000,
        quantity=100,
        total_cost_usd=600000,
        delivery_days=40,
        quality_tier=QualityTier.B,
        vendor_risk_score=0.20,
    )
    ok, violations = task.constraints.validate_proposal(proposal)
    assert ok is False
    assert any("Cost" in v for v in violations)


def test_task_announcement_converts_to_message(task):
    msg = task.to_negotiation_message()
    assert msg.msg_type == MessageType.TASK_ANNOUNCEMENT
    assert msg.task_id == task.task_id
    assert msg.round == 0


def test_proposal_message_requires_payload():
    with pytest.raises(ValueError):
        NegotiationMessage(
            round=1,
            sender="cost_agent",
            msg_type=MessageType.PROPOSAL,
            task_id="demo-task",
            public_reason="We need a proposal to discuss.",
        )


def test_task_constraints_accept_valid_proposal(task):
    proposal = ProposalPayload(
        vendor="Valid Vendor",
        unit_price_usd=4000,
        quantity=100,
        total_cost_usd=400000,
        delivery_days=45,
        quality_tier=QualityTier.B,
        vendor_risk_score=0.3,
    )
    ok, violations = task.constraints.validate_proposal(proposal)
    assert ok is True
    assert violations == []


def test_gemini_health_check_degrades_without_api_key(monkeypatch):
    from Negotiate_AI.core.nebius_client import GeminiClient

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    client = GeminiClient(api_key=None)
    result = asyncio.run(client.health_check())

    assert client.model == "gemini-flash-latest"
    assert result["status"] == "degraded"


def test_gemini_client_reads_environment_configuration(monkeypatch):
    from Negotiate_AI.core.nebius_client import GeminiClient, NebiusClient

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    client = GeminiClient()

    assert client.api_key == "test-key"
    assert client.base_url == "https://example.test/v1"
    assert client.model == "test-model"
    assert NebiusClient is GeminiClient


def test_gemini_client_calls_generate_content_api(monkeypatch):
    from Negotiate_AI.core.nebius_client import GeminiClient
    import httpx

    captured = {}

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [
                    {"content": {"parts": [{"text": "Gemini response"}]}}
                ]
            }

    class MockAsyncClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            pass

        async def post(self, url, headers, json):
            captured.update(url=url, headers=headers, body=json)
            return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    client = GeminiClient(
        api_key="test-key",
        model="gemini-flash-latest",
    )

    result = asyncio.run(
        client.complete(
            system_prompt="System prompt",
            user_prompt="User prompt",
            response_mime_type="application/json",
        )
    )

    assert result == "Gemini response"
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-flash-latest:generateContent"
    )
    assert captured["headers"] == {"X-goog-api-key": "test-key"}
    assert captured["body"]["systemInstruction"]["parts"][0]["text"] == "System prompt"
    assert captured["body"]["contents"][0]["parts"][0]["text"] == "User prompt"
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
