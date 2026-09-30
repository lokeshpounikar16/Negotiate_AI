"""Mediator agent for Week 3 deadlock detection and compromise proposals."""

from __future__ import annotations

from typing import Any, Sequence

from .base_agent import BaseAgent
from ..core.schema import (
    AgentID,
    MessageType,
    NegotiationMessage,
    ProposalPayload,
    QualityTier,
    TaskAnnouncement,
)


class MediatorAgent(BaseAgent):
    """Breaks deadlock by proposing a balanced compromise between agents."""

    def __init__(
        self,
        deadlock_threshold: int = 2,
        nebius_client=None,
        message_bus=None,
    ):
        super().__init__(
            agent_id=AgentID.MEDIATOR,
            nebius_client=nebius_client,
            message_bus=message_bus,
            batna_threshold=1.0,
        )
        self.deadlock_threshold = max(1, deadlock_threshold)
        self.rounds_without_progress = 0

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Mediator Agent. Detect deadlock and propose a balanced "
            "compromise that keeps the negotiation moving while protecting fairness."
        )

    def compute_utility(self, proposal: ProposalPayload) -> float:
        if not proposal:
            return 0.0
        # The mediator is not a commercial participant; it values fairness and progress.
        return 0.75

    def _utility_for_message(self, message: Any) -> float:
        if hasattr(message, "utility_score"):
            return float(message.utility_score or 0.0)
        if isinstance(message, dict):
            return float(message.get("utility_score") or 0.0)
        return 0.0

    def detect_deadlock(self, messages: Sequence[Any]) -> bool:
        """Return True if recent message windows repeatedly show the same utility trend."""
        recent = list(messages)
        if len(recent) < self.deadlock_threshold:
            return False

        window_size = max(1, self.deadlock_threshold)
        recent_window = recent[-max(4, window_size * 2):]
        if len(recent_window) < window_size * 2:
            return False

        averages = []
        for i in range(0, len(recent_window) - window_size + 1, window_size):
            chunk = recent_window[i : i + window_size]
            if len(chunk) < window_size:
                continue
            avg = sum(self._utility_for_message(m) for m in chunk) / len(chunk)
            averages.append(avg)

        if len(averages) < 2:
            return False

        stagnation = 0
        for i in range(1, len(averages)):
            if abs(averages[i] - averages[i - 1]) < 1e-3:
                stagnation += 1
            else:
                stagnation = 0
            if stagnation >= self.deadlock_threshold - 1:
                self.rounds_without_progress = stagnation + 1
                return True

        self.rounds_without_progress = 0
        return False

    def _average_proposal(self, proposals: list[ProposalPayload]) -> ProposalPayload:
        if not proposals:
            raise ValueError("No proposals available to create a compromise")

        last = proposals[-1]
        avg_price = sum(p.unit_price_usd for p in proposals) / len(proposals)
        avg_cost = sum(p.total_cost_usd for p in proposals) / len(proposals)
        avg_days = sum(p.delivery_days for p in proposals) / len(proposals)
        avg_risk = sum(p.vendor_risk_score for p in proposals) / len(proposals)

        quality_order = [
            QualityTier.C,
            QualityTier.B_MINUS,
            QualityTier.B,
            QualityTier.B_PLUS,
            QualityTier.A_MINUS,
            QualityTier.A,
            QualityTier.A_PLUS,
        ]

        ordered = sorted(
            {p.quality_tier for p in proposals},
            key=lambda q: quality_order.index(q),
        )
        compromise_tier = ordered[len(ordered) // 2] if ordered else last.quality_tier

        return ProposalPayload(
            vendor="Compromise Vendor",
            unit_price_usd=round(avg_price, 2),
            quantity=last.quantity,
            total_cost_usd=round(avg_cost, 2),
            delivery_days=max(1, round(avg_days)),
            quality_tier=compromise_tier,
            vendor_risk_score=round(avg_risk, 2),
            warranty_months=last.warranty_months,
            sla_uptime_pct=last.sla_uptime_pct,
            source_country=last.source_country,
            notes="Mediator compromise to restore progress in a deadlocked negotiation.",
        )

    def propose_compromise(
        self,
        messages: Sequence[Any],
        agents: dict[str, Any] | None = None,
        round_num: int = 1,
        task_id: str = "default-task",
    ) -> NegotiationMessage | None:
        """Create a compromise message when the negotiation stalls."""
        proposals = []
        for message in reversed(list(messages)):
            proposal = getattr(message, "proposal", None)
            if proposal is not None:
                proposals.append(proposal)
            elif isinstance(message, dict):
                proposal = message.get("proposal")
                if proposal is not None:
                    proposals.append(proposal)
            if len(proposals) >= 4:
                break

        if not proposals:
            return None

        compromise = self._average_proposal(proposals[:4])
        return NegotiationMessage(
            round=round_num,
            sender=AgentID.MEDIATOR,
            msg_type=MessageType.MEDIATOR_PROPOSAL,
            task_id=task_id,
            proposal=compromise,
            public_reason="Deadlock detected; the mediator is proposing a balanced compromise to keep the negotiation moving.",
            utility_score=self.compute_utility(compromise),
        )

    def generate_proposal(self) -> ProposalPayload:
        return ProposalPayload(
            vendor="Balanced Vendor",
            unit_price_usd=3900,
            quantity=500,
            total_cost_usd=1950000,
            delivery_days=50,
            quality_tier=QualityTier.A_MINUS,
            vendor_risk_score=0.20,
            warranty_months=36,
            sla_uptime_pct=99.9,
            source_country="USA",
            notes="Initial neutral compromise position.",
        )

    def generate_counter_proposal(self, received: ProposalPayload) -> ProposalPayload:
        return received

    async def generate_opening_proposal(self, task: TaskAnnouncement) -> NegotiationMessage:
        proposal = self.generate_proposal()
        return NegotiationMessage(
            round=1,
            sender=AgentID.MEDIATOR,
            msg_type=MessageType.MEDIATOR_PROPOSAL,
            task_id=task.task_id,
            proposal=proposal,
            public_reason="The mediator opens with a balanced compromise that protects fairness and progress.",
            utility_score=self.compute_utility(proposal),
        )

    async def respond_to_proposal(
        self, incoming: NegotiationMessage, task: TaskAnnouncement
    ) -> NegotiationMessage:
        if incoming.proposal is None:
            return NegotiationMessage(
                round=incoming.round + 1,
                sender=AgentID.MEDIATOR,
                msg_type=MessageType.REJECT,
                task_id=task.task_id,
                public_reason="The incoming message lacked a proposal for the mediator to evaluate.",
                rejection_reason="Missing proposal payload.",
            )

        compromise = self.generate_proposal()
        return NegotiationMessage(
            round=incoming.round + 1,
            sender=AgentID.MEDIATOR,
            msg_type=MessageType.MEDIATOR_PROPOSAL,
            task_id=task.task_id,
            proposal=compromise,
            public_reason="The mediator offers a balanced compromise when the parties are far apart.",
            utility_score=self.compute_utility(compromise),
        )


__all__ = ["MediatorAgent"]
