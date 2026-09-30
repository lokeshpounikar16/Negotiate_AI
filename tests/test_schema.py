"""
Unit tests for NegotiationMessage protocol schema.
These run without any external dependencies (no LLM, no Redis).
Schema correctness is the foundation everything else builds on.
"""

import pytest
from pydantic import ValidationError

from negotiateai.core.schema import (
    AgentID,
    Concession,
    MessageType,
    NegotiationMessage,
    ProposalPayload,
    QualityTier,
    TaskAnnouncement,
    TaskConstraints,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_proposal() -> ProposalPayload:
    return ProposalPayload(
        vendor="Dell Technologies",
        unit_price_usd=3800.0,
        quantity=500,
        total_cost_usd=1_900_000.0,
        delivery_days=45,
        quality_tier=QualityTier.A_MINUS,
        vendor_risk_score=0.15,
        warranty_months=36,
        sla_uptime_pct=99.9,
        source_country="USA",
    )


@pytest.fixture
def valid_task() -> TaskAnnouncement:
    return TaskAnnouncement(
        title="Procure 500 servers for DC expansion",
        description="We need 500 rack servers for the new EU data centre. "
                    "Budget is $2M, must arrive within 60 days.",
        constraints=TaskConstraints(
            max_budget_usd=2_000_000,
            quantity=500,
            max_delivery_days=60,
            min_quality_tier=QualityTier.B_PLUS,
            max_vendor_risk=0.40,
            min_warranty_months=12,
        ),
        max_rounds=10,
    )


@pytest.fixture
def valid_message(valid_proposal) -> NegotiationMessage:
    return NegotiationMessage(
        round=1,
        sender=AgentID.COST,
        msg_type=MessageType.PROPOSAL,
        task_id="test-task-001",
        proposal=valid_proposal,
        public_reason="Opening with Dell at lowest viable price point to anchor the negotiation.",
        utility_score=0.72,
    )


# ── ProposalPayload tests ─────────────────────────────────────────────────────

class TestProposalPayload:

    def test_valid_proposal_creates_correctly(self, valid_proposal):
        assert valid_proposal.vendor == "Dell Technologies"
        assert valid_proposal.total_cost_usd == 1_900_000.0
        assert valid_proposal.quality_tier == QualityTier.A_MINUS

    def test_total_cost_mismatch_raises(self):
        """total_cost_usd must equal unit_price * quantity (within $1 tolerance)."""
        with pytest.raises(ValidationError, match="total_cost_usd"):
            ProposalPayload(
                vendor="Acme",
                unit_price_usd=1000.0,
                quantity=100,
                total_cost_usd=50_000.0,  # wrong: should be 100,000
                delivery_days=30,
                quality_tier=QualityTier.B,
                vendor_risk_score=0.2,
            )

    def test_total_cost_within_tolerance_passes(self):
        """Small floating-point differences within $1 should be accepted."""
        p = ProposalPayload(
            vendor="Acme",
            unit_price_usd=1000.0,
            quantity=3,
            total_cost_usd=3000.00,  # exact
            delivery_days=30,
            quality_tier=QualityTier.B,
            vendor_risk_score=0.2,
        )
        assert p.total_cost_usd == 3000.00

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            ProposalPayload(
                vendor="Acme",
                unit_price_usd=-500.0,
                quantity=10,
                total_cost_usd=-5000.0,
                delivery_days=30,
                quality_tier=QualityTier.B,
                vendor_risk_score=0.2,
            )

    def test_risk_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ProposalPayload(
                vendor="Acme",
                unit_price_usd=1000.0,
                quantity=10,
                total_cost_usd=10_000.0,
                delivery_days=30,
                quality_tier=QualityTier.B,
                vendor_risk_score=1.5,  # > 1.0 invalid
            )

    def test_all_quality_tiers_accepted(self):
        for tier in QualityTier:
            p = ProposalPayload(
                vendor="Acme",
                unit_price_usd=1000.0,
                quantity=1,
                total_cost_usd=1000.0,
                delivery_days=30,
                quality_tier=tier,
                vendor_risk_score=0.2,
            )
            assert p.quality_tier == tier


# ── NegotiationMessage tests ──────────────────────────────────────────────────

class TestNegotiationMessage:

    def test_valid_proposal_message(self, valid_message):
        assert valid_message.sender == AgentID.COST
        assert valid_message.msg_type == MessageType.PROPOSAL
        assert valid_message.proposal is not None
        assert valid_message.utility_score == 0.72

    def test_msg_id_auto_generated(self, valid_message):
        assert valid_message.msg_id is not None
        assert len(valid_message.msg_id) == 36  # UUID v4

    def test_two_messages_have_different_ids(self, valid_proposal):
        m1 = NegotiationMessage(
            round=1, sender=AgentID.COST, msg_type=MessageType.PROPOSAL,
            task_id="t1", proposal=valid_proposal,
            public_reason="First message for testing purposes.",
        )
        m2 = NegotiationMessage(
            round=1, sender=AgentID.COST, msg_type=MessageType.PROPOSAL,
            task_id="t1", proposal=valid_proposal,
            public_reason="Second message for testing purposes.",
        )
        assert m1.msg_id != m2.msg_id

    def test_proposal_required_for_proposal_type(self):
        """PROPOSAL messages without a payload should be rejected."""
        with pytest.raises(ValidationError, match="proposal payload is required"):
            NegotiationMessage(
                round=1,
                sender=AgentID.COST,
                msg_type=MessageType.PROPOSAL,
                task_id="t1",
                proposal=None,  # missing!
                public_reason="This should fail because no proposal payload.",
            )

    def test_counter_proposal_requires_payload(self):
        with pytest.raises(ValidationError, match="proposal payload is required"):
            NegotiationMessage(
                round=2,
                sender=AgentID.QUALITY,
                msg_type=MessageType.COUNTER_PROPOSAL,
                task_id="t1",
                proposal=None,
                public_reason="Counter proposal without payload should fail.",
            )

    def test_reject_requires_rejection_reason(self):
        with pytest.raises(ValidationError, match="rejection_reason is required"):
            NegotiationMessage(
                round=3,
                sender=AgentID.QUALITY,
                msg_type=MessageType.REJECT,
                task_id="t1",
                public_reason="Rejecting this proposal outright.",
                # rejection_reason missing!
            )

    def test_accept_message_valid_without_payload(self):
        """ACCEPT messages don't need a new proposal payload."""
        msg = NegotiationMessage(
            round=4,
            sender=AgentID.TIMELINE,
            msg_type=MessageType.ACCEPT,
            task_id="t1",
            public_reason="Accepting the current proposal — meets all timeline constraints.",
        )
        assert msg.msg_type == MessageType.ACCEPT

    def test_to_agent_view_strips_utility_score(self, valid_message):
        """utility_score is private and must not appear in broadcast view."""
        view = valid_message.to_agent_view()
        assert "utility_score" not in view

    def test_public_reason_minimum_length(self):
        """public_reason must be at least 10 characters."""
        with pytest.raises(ValidationError):
            NegotiationMessage(
                round=1,
                sender=AgentID.COST,
                msg_type=MessageType.ACCEPT,
                task_id="t1",
                public_reason="Too short",  # < 10 chars
            )

    def test_task_announcement_no_proposal_needed(self):
        """TASK_ANNOUNCEMENT from system doesn't need a proposal."""
        msg = NegotiationMessage(
            round=0,
            sender=AgentID.SYSTEM,
            msg_type=MessageType.TASK_ANNOUNCEMENT,
            task_id="t1",
            public_reason="Initiating procurement negotiation for 500 servers.",
        )
        assert msg.round == 0
        assert msg.sender == AgentID.SYSTEM

    def test_json_round_trip(self, valid_message):
        """Messages must survive JSON serialization/deserialization."""
        json_str = valid_message.model_dump_json()
        restored = NegotiationMessage.model_validate_json(json_str)
        assert restored.msg_id     == valid_message.msg_id
        assert restored.sender     == valid_message.sender
        assert restored.msg_type   == valid_message.msg_type
        assert restored.round      == valid_message.round
        assert restored.task_id    == valid_message.task_id
        assert restored.proposal.vendor == valid_message.proposal.vendor

    def test_concessions_attached_correctly(self, valid_proposal):
        """Concession records should be attached and validated."""
        concession = Concession(
            dimension="quality_tier",
            from_value="A+",
            to_value="A",
            utility_cost=0.15,
        )
        msg = NegotiationMessage(
            round=2,
            sender=AgentID.QUALITY,
            msg_type=MessageType.COUNTER_PROPOSAL,
            task_id="t1",
            proposal=valid_proposal,
            public_reason="Conceding on quality tier to move the negotiation forward.",
            concessions_made=[concession],
        )
        assert len(msg.concessions_made) == 1
        assert msg.concessions_made[0].dimension == "quality_tier"


# ── TaskConstraints validation tests ─────────────────────────────────────────

class TestTaskConstraints:

    @pytest.fixture
    def constraints(self) -> TaskConstraints:
        return TaskConstraints(
            max_budget_usd=2_000_000,
            quantity=500,
            max_delivery_days=60,
            min_quality_tier=QualityTier.B_PLUS,
            max_vendor_risk=0.40,
            min_warranty_months=12,
        )

    def test_valid_proposal_passes_constraints(self, constraints, valid_proposal):
        is_valid, violations = constraints.validate_proposal(valid_proposal)
        assert is_valid, f"Expected valid but got violations: {violations}"
        assert violations == []

    def test_over_budget_flagged(self, constraints):
        expensive = ProposalPayload(
            vendor="Cisco",
            unit_price_usd=5000.0,
            quantity=500,
            total_cost_usd=2_500_000.0,  # over $2M budget
            delivery_days=30,
            quality_tier=QualityTier.A_PLUS,
            vendor_risk_score=0.1,
        )
        is_valid, violations = constraints.validate_proposal(expensive)
        assert not is_valid
        assert any("budget" in v.lower() or "Cost" in v for v in violations)

    def test_slow_delivery_flagged(self, constraints):
        slow = ProposalPayload(
            vendor="Cheap Vendor",
            unit_price_usd=3000.0,
            quantity=500,
            total_cost_usd=1_500_000.0,
            delivery_days=90,  # over 60-day limit
            quality_tier=QualityTier.A,
            vendor_risk_score=0.2,
        )
        is_valid, violations = constraints.validate_proposal(slow)
        assert not is_valid
        assert any("delivery" in v.lower() or "Delivery" in v for v in violations)

    def test_below_quality_tier_flagged(self, constraints):
        low_quality = ProposalPayload(
            vendor="Budget Vendor",
            unit_price_usd=2000.0,
            quantity=500,
            total_cost_usd=1_000_000.0,
            delivery_days=45,
            quality_tier=QualityTier.C,  # below B+ minimum
            vendor_risk_score=0.3,
        )
        is_valid, violations = constraints.validate_proposal(low_quality)
        assert not is_valid
        assert any("quality" in v.lower() or "Quality" in v for v in violations)

    def test_multiple_violations_reported(self, constraints):
        bad = ProposalPayload(
            vendor="Terrible Vendor",
            unit_price_usd=5000.0,
            quantity=500,
            total_cost_usd=2_500_000.0,  # over budget
            delivery_days=90,             # over time
            quality_tier=QualityTier.C,   # below quality
            vendor_risk_score=0.9,        # too risky
        )
        is_valid, violations = constraints.validate_proposal(bad)
        assert not is_valid
        assert len(violations) >= 3  # at least: cost, delivery, quality

    def test_boundary_conditions_pass(self, constraints):
        """Proposals exactly at the limit should be valid."""
        boundary = ProposalPayload(
            vendor="Edge Case Vendor",
            unit_price_usd=4000.0,
            quantity=500,
            total_cost_usd=2_000_000.0,  # exactly at budget
            delivery_days=60,             # exactly at deadline
            quality_tier=QualityTier.B_PLUS,  # exactly at minimum
            vendor_risk_score=0.40,       # exactly at max risk
            warranty_months=12,           # exactly at minimum
        )
        is_valid, violations = constraints.validate_proposal(boundary)
        assert is_valid, f"Boundary proposal should be valid but got: {violations}"


# ── TaskAnnouncement tests ────────────────────────────────────────────────────

class TestTaskAnnouncement:

    def test_task_id_auto_generated(self, valid_task):
        assert valid_task.task_id is not None
        assert len(valid_task.task_id) > 0

    def test_to_negotiation_message(self, valid_task):
        msg = valid_task.to_negotiation_message()
        assert msg.msg_type == MessageType.TASK_ANNOUNCEMENT
        assert msg.sender == AgentID.SYSTEM
        assert msg.round == 0
        assert msg.task_id == valid_task.task_id
        assert "Budget" in msg.public_reason or "budget" in msg.public_reason

    def test_two_tasks_different_ids(self, valid_task):
        constraints = TaskConstraints(
            max_budget_usd=1_000_000,
            quantity=100,
            max_delivery_days=30,
        )
        other_task = TaskAnnouncement(
            title="Another Task",
            description="A different task entirely.",
            constraints=constraints,
        )
        assert valid_task.task_id != other_task.task_id
