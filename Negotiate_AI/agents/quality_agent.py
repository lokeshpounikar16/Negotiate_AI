"""Quality-focused procurement agent for the multi-agent negotiation loop."""

from __future__ import annotations

from typing import Optional

from .base_agent import BaseAgent
from ..core.schema import (
    AgentID,
    MessageType,
    NegotiationMessage,
    ProposalPayload,
    QualityTier,
    TaskAnnouncement,
)


class QualityAgent(BaseAgent):
    """Prefers premium vendors with strong quality and SLA performance."""

    def __init__(
        self,
        min_quality_tier: str = "A-",
        weight_rating: float = 0.5,
        weight_sla: float = 0.3,
        weight_tier: float = 0.2,
        nebius_client=None,
        message_bus=None,
    ):
        super().__init__(
            agent_id=AgentID.QUALITY,
            nebius_client=nebius_client,
            message_bus=message_bus,
            batna_threshold=0.35,
        )
        self.min_quality_tier = min_quality_tier
        self.weight_rating = weight_rating
        self.weight_sla = weight_sla
        self.weight_tier = weight_tier
        self.time_pressure_coefficient = 0.3

    @property
    def system_prompt(self) -> str:
        return "You are the Quality Agent. Protect quality, SLA, and vendor reputation."

    def compute_utility(self, proposal: ProposalPayload) -> float:
        vendor_ratings = {
            "Dell Technologies": 4.2 / 5.0,
            "HPE ProLiant": 4.5 / 5.0,
            "Lenovo ThinkSystem": 3.8 / 5.0,
            "IBM Power Systems": 4.8 / 5.0,
        }
        rating_score = vendor_ratings.get(proposal.vendor, 3.5 / 5.0)
        sla_score = 0.9 if proposal.delivery_days <= 60 else 0.5
        tier_scores = {
            QualityTier.A_PLUS: 1.0,
            QualityTier.A: 1.0,
            QualityTier.A_MINUS: 0.9,
            QualityTier.B_PLUS: 0.7,
            QualityTier.B: 0.5,
            QualityTier.B_MINUS: 0.3,
            QualityTier.C: 0.1,
        }
        tier_score = tier_scores.get(proposal.quality_tier, 0.5)
        utility = (
            rating_score * self.weight_rating
            + sla_score * self.weight_sla
            + tier_score * self.weight_tier
        )
        return max(0.0, min(1.0, utility))

    def evaluate_proposal(self, proposal: ProposalPayload) -> bool:
        tier_scores = {
            QualityTier.A_PLUS: 1.0,
            QualityTier.A: 1.0,
            QualityTier.A_MINUS: 0.9,
            QualityTier.B_PLUS: 0.7,
            QualityTier.B: 0.5,
        }
        min_score = tier_scores.get(QualityTier(self.min_quality_tier), 0.9)
        proposal_score = tier_scores.get(proposal.quality_tier, 0.0)
        if proposal_score < min_score:
            return False
        return self.compute_utility(proposal) >= 0.60

    def generate_proposal(self) -> ProposalPayload:
        return ProposalPayload(
            vendor="IBM Power Systems",
            unit_price_usd=4500,
            quantity=500,
            total_cost_usd=2250000,
            delivery_days=60,
            quality_tier=QualityTier.A,
            vendor_risk_score=0.15,
            warranty_months=36,
            sla_uptime_pct=99.9,
            source_country="USA",
            notes="Premium vendor for high-quality enterprise hardware.",
        )

    def generate_counter_proposal(self, received: ProposalPayload) -> ProposalPayload:
        if received.quality_tier in {QualityTier.A, QualityTier.A_PLUS}:
            return received
        return ProposalPayload(
            vendor="HPE ProLiant",
            unit_price_usd=4100,
            quantity=received.quantity,
            total_cost_usd=4100 * received.quantity,
            delivery_days=50,
            quality_tier=QualityTier.A_MINUS,
            vendor_risk_score=0.22,
            warranty_months=36,
            sla_uptime_pct=99.9,
            source_country="USA",
            notes="Counter-offer that preserves quality while reducing cost.",
        )

    async def generate_opening_proposal(self, task: TaskAnnouncement) -> NegotiationMessage:
        proposal = self.generate_proposal()
        return NegotiationMessage(
            round=1,
            sender=AgentID.QUALITY,
            msg_type=MessageType.PROPOSAL,
            task_id=task.task_id,
            proposal=proposal,
            public_reason="Quality-first opening proposal from the Quality Agent.",
            utility_score=self.compute_utility(proposal),
        )

    async def respond_to_proposal(
        self, incoming: NegotiationMessage, task: TaskAnnouncement
    ) -> NegotiationMessage:
        if incoming.proposal is None:
            return NegotiationMessage(
                round=incoming.round + 1,
                sender=AgentID.QUALITY,
                msg_type=MessageType.REJECT,
                task_id=task.task_id,
                public_reason="Incoming message did not contain a proposal payload.",
                rejection_reason="Missing proposal payload.",
            )

        if self.evaluate_proposal(incoming.proposal):
            return NegotiationMessage(
                round=incoming.round + 1,
                sender=AgentID.QUALITY,
                msg_type=MessageType.ACCEPT,
                task_id=task.task_id,
                proposal=incoming.proposal,
                public_reason="This proposal satisfies the quality thresholds and maintains acceptable utility.",
                utility_score=self.compute_utility(incoming.proposal),
            )

        counter = self.generate_counter_proposal(incoming.proposal)
        return NegotiationMessage(
            round=incoming.round + 1,
            sender=AgentID.QUALITY,
            msg_type=MessageType.COUNTER_PROPOSAL,
            task_id=task.task_id,
            proposal=counter,
            public_reason="The quality minimum is not met; a premium-but-feasible counter is being offered.",
            utility_score=self.compute_utility(counter),
        )


__all__ = ["QualityAgent"]
