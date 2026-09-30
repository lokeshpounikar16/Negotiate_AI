"""Timeline-focused procurement agent for the multi-agent negotiation loop."""

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


class TimelineAgent(BaseAgent):
    """Prioritizes faster delivery and aggressive schedule compression."""

    def __init__(
        self,
        deadline: int = 60,
        weight_speed: float = 0.7,
        nebius_client=None,
        message_bus=None,
    ):
        super().__init__(
            agent_id=AgentID.TIMELINE,
            nebius_client=nebius_client,
            message_bus=message_bus,
            batna_threshold=0.30,
        )
        self.deadline = deadline
        self.weight_speed = weight_speed
        self.time_pressure_coefficient = 0.5

    @property
    def system_prompt(self) -> str:
        return "You are the Timeline Agent. Compress the procurement schedule and keep delivery dates aggressive."

    def compute_utility(self, proposal: ProposalPayload) -> float:
        utility = max(0, (self.deadline - proposal.delivery_days) / self.deadline)
        return max(0.0, min(1.0, utility))

    def evaluate_proposal(self, proposal: ProposalPayload) -> bool:
        if proposal.delivery_days > self.deadline:
            return False
        return self.compute_utility(proposal) >= 0.60

    def generate_proposal(self) -> ProposalPayload:
        return ProposalPayload(
            vendor="HPE ProLiant",
            unit_price_usd=4200,
            quantity=500,
            total_cost_usd=2100000,
            delivery_days=35,
            quality_tier=QualityTier.A,
            vendor_risk_score=0.25,
            warranty_months=36,
            sla_uptime_pct=99.9,
            source_country="USA",
            notes="Fastest reasonable delivery option to compress the schedule.",
        )

    def generate_counter_proposal(self, received: ProposalPayload) -> ProposalPayload:
        if received.delivery_days <= 45:
            return received
        return ProposalPayload(
            vendor="HPE ProLiant",
            unit_price_usd=4300,
            quantity=received.quantity,
            total_cost_usd=4300 * received.quantity,
            delivery_days=40,
            quality_tier=received.quality_tier,
            vendor_risk_score=received.vendor_risk_score,
            warranty_months=received.warranty_months,
            sla_uptime_pct=received.sla_uptime_pct,
            source_country=received.source_country,
            notes="Offer a faster alternative while keeping the proposal close to current terms.",
        )

    async def generate_opening_proposal(self, task: TaskAnnouncement) -> NegotiationMessage:
        proposal = self.generate_proposal()
        return NegotiationMessage(
            round=1,
            sender=AgentID.TIMELINE,
            msg_type=MessageType.PROPOSAL,
            task_id=task.task_id,
            proposal=proposal,
            public_reason="Timeline-first opening proposal to compress delivery timing.",
            utility_score=self.compute_utility(proposal),
        )

    async def respond_to_proposal(
        self, incoming: NegotiationMessage, task: TaskAnnouncement
    ) -> NegotiationMessage:
        if incoming.proposal is None:
            return NegotiationMessage(
                round=incoming.round + 1,
                sender=AgentID.TIMELINE,
                msg_type=MessageType.REJECT,
                task_id=task.task_id,
                public_reason="Incoming message did not contain a proposal payload.",
                rejection_reason="Missing proposal payload.",
            )

        if self.evaluate_proposal(incoming.proposal):
            return NegotiationMessage(
                round=incoming.round + 1,
                sender=AgentID.TIMELINE,
                msg_type=MessageType.ACCEPT,
                task_id=task.task_id,
                proposal=incoming.proposal,
                public_reason="This delivery schedule satisfies the timeline requirement.",
                utility_score=self.compute_utility(incoming.proposal),
            )

        counter = self.generate_counter_proposal(incoming.proposal)
        return NegotiationMessage(
            round=incoming.round + 1,
            sender=AgentID.TIMELINE,
            msg_type=MessageType.COUNTER_PROPOSAL,
            task_id=task.task_id,
            proposal=counter,
            public_reason="This plan misses the deadline; a faster schedule is offered to close the gap.",
            utility_score=self.compute_utility(counter),
        )


__all__ = ["TimelineAgent"]
