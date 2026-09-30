"""Risk-focused procurement agent for the multi-agent negotiation loop."""

from __future__ import annotations

from .base_agent import BaseAgent
from ..core.schema import (
    AgentID,
    MessageType,
    NegotiationMessage,
    ProposalPayload,
    QualityTier,
    TaskAnnouncement,
)


class RiskAgent(BaseAgent):
    """Seeks lower dependency risk and vendor diversification."""

    def __init__(self, nebius_client=None, message_bus=None):
        super().__init__(
            agent_id=AgentID.RISK,
            nebius_client=nebius_client,
            message_bus=message_bus,
            batna_threshold=0.40,
        )
        self.require_multiple_vendors = True
        self.max_single_vendor_pct = 0.6
        self.time_pressure_coefficient = 0.2

    @property
    def system_prompt(self) -> str:
        return "You are the Risk Agent. Minimize vendor concentration and protect supply-chain resilience."

    def compute_utility(self, proposal: ProposalPayload) -> float:
        risk_score = proposal.vendor_risk_score
        utility = max(0.0, 1.0 - risk_score)
        return max(0.0, min(1.0, utility))

    def evaluate_proposal(self, proposal: ProposalPayload) -> bool:
        utility = self.compute_utility(proposal)
        return utility >= 0.65

    def generate_proposal(self) -> ProposalPayload:
        return ProposalPayload(
            vendor="Dell + HPE",
            unit_price_usd=3900,
            quantity=500,
            total_cost_usd=1950000,
            delivery_days=50,
            quality_tier=QualityTier.A_MINUS,
            vendor_risk_score=0.18,
            warranty_months=36,
            sla_uptime_pct=99.9,
            source_country="USA",
            notes="Split-order strategy to lower concentration risk.",
        )

    def generate_counter_proposal(self, received: ProposalPayload) -> ProposalPayload:
        if received.vendor_risk_score < 0.20:
            return received
        return ProposalPayload(
            vendor="Dell + HPE (split)",
            unit_price_usd=3950,
            quantity=received.quantity,
            total_cost_usd=3950 * received.quantity,
            delivery_days=52,
            quality_tier=received.quality_tier,
            vendor_risk_score=0.18,
            warranty_months=received.warranty_months,
            sla_uptime_pct=received.sla_uptime_pct,
            source_country=received.source_country,
            notes="Diversifying across two vendors reduces single-source dependency.",
        )

    async def generate_opening_proposal(self, task: TaskAnnouncement) -> NegotiationMessage:
        proposal = self.generate_proposal()
        return NegotiationMessage(
            round=1,
            sender=AgentID.RISK,
            msg_type=MessageType.PROPOSAL,
            task_id=task.task_id,
            proposal=proposal,
            public_reason="Risk-first opening proposal using diversified sourcing.",
            utility_score=self.compute_utility(proposal),
        )

    async def respond_to_proposal(
        self, incoming: NegotiationMessage, task: TaskAnnouncement
    ) -> NegotiationMessage:
        if incoming.proposal is None:
            return NegotiationMessage(
                round=incoming.round + 1,
                sender=AgentID.RISK,
                msg_type=MessageType.REJECT,
                task_id=task.task_id,
                public_reason="Incoming message did not contain a proposal payload.",
                rejection_reason="Missing proposal payload.",
            )

        if self.evaluate_proposal(incoming.proposal):
            return NegotiationMessage(
                round=incoming.round + 1,
                sender=AgentID.RISK,
                msg_type=MessageType.ACCEPT,
                task_id=task.task_id,
                proposal=incoming.proposal,
                public_reason="This proposal keeps vendor concentration within acceptable risk limits.",
                utility_score=self.compute_utility(incoming.proposal),
            )

        counter = self.generate_counter_proposal(incoming.proposal)
        return NegotiationMessage(
            round=incoming.round + 1,
            sender=AgentID.RISK,
            msg_type=MessageType.COUNTER_PROPOSAL,
            task_id=task.task_id,
            proposal=counter,
            public_reason="This proposal is too concentrated; a split-order counter is being proposed.",
            utility_score=self.compute_utility(counter),
        )


__all__ = ["RiskAgent"]
