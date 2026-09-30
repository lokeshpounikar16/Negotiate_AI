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
