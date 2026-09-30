"""
Cost Agent tests — mocks Nebius and Tavily so no API keys needed.
Tests the utility function, proposal logic, and accept/reject decisions.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from negotiateai.core.schema import (
    AgentID,
    MessageType,
    NegotiationMessage,
    ProposalPayload,
    QualityTier,
    TaskAnnouncement,
    TaskConstraints,
)
from negotiateai.agents.cost_agent import CostAgent, ProposalDecision, AcceptRejectDecision
from negotiateai.core.nebius_client import NebiusClient
from negotiateai.core.message_bus import MessageBus


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def task() -> TaskAnnouncement:
    return TaskAnnouncement(
        title="Procure 500 servers",
        description="500 rack servers for EU data centre expansion.",
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
def mock_nebius() -> NebiusClient:
    """A NebiusClient where complete_structured is fully mocked."""
    client = MagicMock(spec=NebiusClient)
    client.complete_structured = AsyncMock()
    return client


@pytest.fixture
def mock_bus() -> MessageBus:
    """A MessageBus where publish is mocked."""
    bus = MagicMock(spec=MessageBus)
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def agent(mock_nebius, mock_bus) -> CostAgent:
    agent = CostAgent(nebius_client=mock_nebius, message_bus=mock_bus)
    agent._tavily = None  # Disable Tavily — use stub data
    return agent


@pytest.fixture
def cheap_proposal() -> ProposalPayload:
    """A proposal that the Cost Agent should love (very cheap)."""
    return ProposalPayload(
        vendor="Lenovo ThinkSystem",
        unit_price_usd=3200.0,
        quantity=500,
        total_cost_usd=1_600_000.0,
        delivery_days=52,
        quality_tier=QualityTier.B_PLUS,
        vendor_risk_score=0.22,
        warranty_months=24,
    )


@pytest.fixture
def expensive_proposal() -> ProposalPayload:
    """A proposal the Cost Agent should dislike (expensive, over budget)."""
    return ProposalPayload(
        vendor="Cisco UCS",
        unit_price_usd=5000.0,
        quantity=500,
        total_cost_usd=2_500_000.0,
        delivery_days=30,
        quality_tier=QualityTier.A_PLUS,
        vendor_risk_score=0.10,
        warranty_months=60,
    )


# ── Utility function tests ────────────────────────────────────────────────────

class TestCostAgentUtility:

    def test_cheap_fast_proposal_high_utility(self, agent, task, cheap_proposal):
        agent.task = task
        utility = agent.compute_utility(cheap_proposal)
        # $1.6M out of $2M budget = cost_score = 0.20, delivery = 52/60 low
        # Should be moderate-to-high from cost agent's perspective
        assert utility > 0.20, f"Expected utility > 0.20, got {utility}"

    def test_expensive_over_budget_low_utility(self, agent, task, expensive_proposal):
        agent.task = task
        utility = agent.compute_utility(expensive_proposal)
        # Cost score = 0 (over budget), utility should be very low
        assert utility < 0.20, f"Expected utility < 0.20 for over-budget proposal, got {utility}"

    def test_utility_range_is_zero_to_one(self, agent, task):
        agent.task = task
        proposals = [
            ProposalPayload(
                vendor="V", unit_price_usd=p, quantity=500,
                total_cost_usd=p * 500,
                delivery_days=d, quality_tier=QualityTier.B,
                vendor_risk_score=0.2,
            )
            for p, d in [(1000, 10), (2000, 30), (3500, 50), (4500, 70)]
        ]
        for prop in proposals:
            u = agent.compute_utility(prop)
            assert 0.0 <= u <= 1.0, f"Utility {u} out of [0, 1] range for {prop.vendor}"

    def test_lower_cost_higher_utility(self, agent, task):
        """All else equal, a cheaper proposal should have higher utility."""
        agent.task = task

        cheap = ProposalPayload(
            vendor="Cheap", unit_price_usd=2000, quantity=500,
            total_cost_usd=1_000_000, delivery_days=40,
            quality_tier=QualityTier.B_PLUS, vendor_risk_score=0.3,
        )
        expensive = ProposalPayload(
            vendor="Expensive", unit_price_usd=3800, quantity=500,
            total_cost_usd=1_900_000, delivery_days=40,
            quality_tier=QualityTier.B_PLUS, vendor_risk_score=0.3,
        )

        assert agent.compute_utility(cheap) > agent.compute_utility(expensive)

    def test_no_task_returns_zero(self, agent):
        """Without a task, utility should return 0 (no constraints to evaluate against)."""
        proposal = ProposalPayload(
            vendor="Any", unit_price_usd=1000, quantity=10,
            total_cost_usd=10_000, delivery_days=30,
            quality_tier=QualityTier.A, vendor_risk_score=0.1,
        )
        agent.task = None
        assert agent.compute_utility(proposal) == 0.0


# ── BATNA tests ───────────────────────────────────────────────────────────────

class TestCostAgentBATNA:

    def test_batna_triggers_on_low_utility(self, agent, task, expensive_proposal):
        agent.task = task
        # Expensive proposal is over budget → utility near 0 → below BATNA threshold
        assert agent.should_activate_batna(expensive_proposal)

    def test_batna_not_triggered_on_good_proposal(self, agent, task, cheap_proposal):
        agent.task = task
        assert not agent.should_activate_batna(cheap_proposal)

    def test_concession_rate_increases_over_rounds(self, agent, task):
        agent.task = task
        rates = []
        for r in [1, 3, 5, 8, 10]:
            agent.current_round = r
            rates.append(agent.concession_rate())
        assert rates == sorted(rates), "Concession rate should increase monotonically"

    def test_concession_rate_caps_at_one(self, agent, task):
        agent.task = task
        agent.current_round = 999
        assert agent.concession_rate() == 1.0


# ── Opening proposal tests ────────────────────────────────────────────────────

class TestCostAgentOpeningProposal:

    @pytest.mark.asyncio
    async def test_generates_valid_proposal_message(self, agent, task, mock_nebius, cheap_proposal):
        """The LLM returns a ProposalDecision → agent wraps it in NegotiationMessage."""
        mock_nebius.complete_structured.return_value = ProposalDecision(
            proposal=cheap_proposal,
            public_reason="Opening with Lenovo at lowest viable cost to anchor the negotiation low.",
            utility_estimate=0.68,
            concessions_from_prior=[],
        )

        agent.task = task
        agent.current_round = 1
        msg = await agent.generate_opening_proposal(task)

        assert isinstance(msg, NegotiationMessage)
        assert msg.msg_type == MessageType.PROPOSAL
        assert msg.sender == AgentID.COST
        assert msg.proposal is not None
        assert msg.proposal.vendor == "Lenovo ThinkSystem"
        assert msg.utility_score == 0.68

    @pytest.mark.asyncio
    async def test_proposal_message_has_public_reason(self, agent, task, mock_nebius, cheap_proposal):
        mock_nebius.complete_structured.return_value = ProposalDecision(
            proposal=cheap_proposal,
            public_reason="Opening with Lenovo at lowest viable cost to anchor the negotiation.",
            utility_estimate=0.7,
        )
        agent.task = task
        msg = await agent.generate_opening_proposal(task)
        assert len(msg.public_reason) >= 10

    @pytest.mark.asyncio
    async def test_proposal_has_correct_task_id(self, agent, task, mock_nebius, cheap_proposal):
        mock_nebius.complete_structured.return_value = ProposalDecision(
            proposal=cheap_proposal,
            public_reason="Opening with Lenovo at lowest viable cost to anchor the negotiation.",
            utility_estimate=0.65,
        )
        agent.task = task
        msg = await agent.generate_opening_proposal(task)
        assert msg.task_id == task.task_id


# ── Response to proposals tests ───────────────────────────────────────────────

class TestCostAgentRespond:

    def _make_incoming(self, task, proposal, sender=AgentID.QUALITY, round_=2) -> NegotiationMessage:
        return NegotiationMessage(
            round=round_,
            sender=sender,
            msg_type=MessageType.PROPOSAL,
            task_id=task.task_id,
            proposal=proposal,
            public_reason="Proposing this vendor for quality and delivery considerations.",
        )

    @pytest.mark.asyncio
    async def test_fast_accept_on_high_utility(self, agent, task, cheap_proposal):
        """If utility ≥ 0.80, agent should fast-accept without calling LLM."""
        # Make a proposal that gives max utility to cost agent (very cheap)
        great_deal = ProposalPayload(
            vendor="BargainBox",
            unit_price_usd=200.0,
            quantity=500,
            total_cost_usd=100_000.0,  # 5% of budget — huge utility
            delivery_days=10,
            quality_tier=QualityTier.A_PLUS,
            vendor_risk_score=0.1,
        )
        agent.task = task
        agent.current_round = 2
        incoming = self._make_incoming(task, great_deal)

        response = await agent.respond_to_proposal(incoming, task)

        assert response.msg_type == MessageType.ACCEPT
        # LLM should NOT have been called (fast path)
        agent.client.complete_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_counter_proposal_on_medium_utility(
        self, agent, task, cheap_proposal, mock_nebius
    ):
        """For mid-range utility, agent asks LLM and may counter."""
        counter = ProposalPayload(
            vendor="Dell",
            unit_price_usd=3500.0,
            quantity=500,
            total_cost_usd=1_750_000.0,
            delivery_days=48,
            quality_tier=QualityTier.A_MINUS,
            vendor_risk_score=0.15,
        )
        mock_nebius.complete_structured.return_value = AcceptRejectDecision(
            action="COUNTER_PROPOSAL",
            public_reason="Can we push the unit price down to $3,400? Lenovo offers comparable specs cheaper.",
            utility_score=0.55,
            counter_proposal=counter,
            concessions_made=[],
        )

        agent.task = task
        agent.current_round = 2
        incoming = self._make_incoming(task, cheap_proposal)

        response = await agent.respond_to_proposal(incoming, task)

        assert response.msg_type == MessageType.COUNTER_PROPOSAL
        assert response.proposal is not None
        mock_nebius.complete_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_payload_incoming_returns_reject(self, agent, task):
        """A message with no proposal payload should get a reject response."""
        no_payload_msg = NegotiationMessage(
            round=2,
            sender=AgentID.QUALITY,
            msg_type=MessageType.ACCEPT,
            task_id=task.task_id,
            public_reason="Accepting with no payload attached to this message.",
        )
        agent.task = task
        agent.current_round = 2

        response = await agent.respond_to_proposal(no_payload_msg, task)
        assert response.msg_type == MessageType.REJECT

    @pytest.mark.asyncio
    async def test_batna_activated_in_response(self, agent, task, expensive_proposal, mock_nebius):
        """After round 3, very bad proposals should trigger BATNA flag."""
        mock_nebius.complete_structured.return_value = AcceptRejectDecision(
            action="COUNTER_PROPOSAL",
            public_reason="This is far over budget. Proposing a much cheaper alternative.",
            utility_score=0.1,
            counter_proposal=ProposalPayload(
                vendor="Alt Vendor", unit_price_usd=3000, quantity=500,
                total_cost_usd=1_500_000, delivery_days=55,
                quality_tier=QualityTier.B_PLUS, vendor_risk_score=0.3,
            ),
        )

        agent.task = task
        agent.current_round = 5  # past round 3
        incoming = self._make_incoming(task, expensive_proposal, round_=4)
        response = await agent.respond_to_proposal(incoming, task)

        # BATNA should be flagged for over-budget proposal at round 5
        assert response.batna_activated is True
