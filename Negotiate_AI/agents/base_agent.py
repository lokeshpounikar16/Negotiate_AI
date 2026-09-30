"""
Base Agent
Abstract class all negotiation agents inherit from.
Defines the lifecycle: receive task → evaluate → propose → respond to counter-proposals.
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from ..core.schema import (
    AgentID,
    MessageType,
    NegotiationMessage,
    ProposalPayload,
    TaskAnnouncement,
)

if TYPE_CHECKING:
    from ..core.message_bus import MessageBus
    from ..core.nebius_client import NebiusClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all negotiation agents.

    Each agent has:
    - An identity (AgentID)
    - A utility function (how it scores proposals)
    - A BATNA threshold (minimum utility to continue negotiating)
    - A system prompt (defines its personality and goals for the LLM)

    Agents communicate ONLY through NegotiationMessages via the MessageBus.
    """

    def __init__(
        self,
        agent_id: AgentID,
        nebius_client: "NebiusClient | None",
        message_bus: "MessageBus | None",
        batna_threshold: float = 0.3,  # Walk away if utility < this
    ):
        self.agent_id        = agent_id
        self.client          = nebius_client
        self.bus             = message_bus
        self.batna_threshold = batna_threshold

        # Session state (reset per task)
        self.task: Optional[TaskAnnouncement] = None
        self.current_round   = 0
        self.my_last_proposal: Optional[NegotiationMessage] = None
        self.best_seen_proposal: Optional[NegotiationMessage] = None
        self.best_seen_utility: float = 0.0
        self.accepted        = False

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        The LLM system prompt that defines this agent's role, goals, and constraints.
        This is the agent's 'personality' — it shapes every LLM call.
        """

    @abstractmethod
    def compute_utility(self, proposal: ProposalPayload) -> float:
        """
        Compute this agent's utility score for a proposal (0.0 → 1.0).
        Each agent has a different utility function reflecting its goals.
        Higher = better for this agent.
        """

    @abstractmethod
    async def generate_opening_proposal(self, task: TaskAnnouncement) -> NegotiationMessage:
        """Generate the agent's first proposal after seeing the task."""

    @abstractmethod
    async def respond_to_proposal(
        self, incoming: NegotiationMessage, task: TaskAnnouncement
    ) -> NegotiationMessage:
        """
        Generate a response to an incoming proposal.
        Can return COUNTER_PROPOSAL, CONCESSION, ACCEPT, or REJECT.
        """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self, task: TaskAnnouncement) -> None:
        """
        Main agent loop.
        1. Connect to message bus
        2. Listen for TASK_ANNOUNCEMENT
        3. Generate opening proposal
        4. Listen and respond until CONSENSUS or BATNA
        """
        self.task = task
        self.current_round = 0

        logger.info(f"[{self.agent_id.value}] Starting for task: {task.title}")

        # Send opening proposal in round 1
        proposal_msg = await self.generate_opening_proposal(task)
        await self.bus.publish(task.task_id, proposal_msg)
        self.my_last_proposal = proposal_msg
        self.current_round = 1

        # Listen for messages from other agents
        async for incoming in self.bus.subscribe(task.task_id, agent_id=self.agent_id):

            # Stop listening if negotiation is over
            if incoming.msg_type == MessageType.CONSENSUS:
                logger.info(f"[{self.agent_id.value}] Consensus reached — stopping.")
                break

            # Only respond to proposals directed at negotiation space
            if incoming.msg_type not in {
                MessageType.PROPOSAL,
                MessageType.COUNTER_PROPOSAL,
                MessageType.CONCESSION,
                MessageType.MEDIATOR_PROPOSAL,
            }:
                continue

            # Check if we've exceeded max rounds
            if incoming.round >= task.max_rounds:
                logger.info(f"[{self.agent_id.value}] Max rounds reached — stopping.")
                break

            # Track the best proposal we've seen
            if incoming.proposal:
                utility = self.compute_utility(incoming.proposal)
                if utility > self.best_seen_utility:
                    self.best_seen_utility = utility
                    self.best_seen_proposal = incoming
                    logger.debug(
                        f"[{self.agent_id.value}] New best proposal from "
                        f"{incoming.sender.value} with utility {utility:.3f}"
                    )

            # Generate response
            self.current_round = incoming.round + 1
            response = await self.respond_to_proposal(incoming, task)
            await self.bus.publish(task.task_id, response)

            if response.msg_type == MessageType.ACCEPT:
                self.accepted = True
                logger.info(
                    f"[{self.agent_id.value}] Accepted proposal from "
                    f"{incoming.sender.value} (utility={self.best_seen_utility:.3f})"
                )
                break

            if response.batna_activated:
                logger.warning(
                    f"[{self.agent_id.value}] BATNA activated — walking away."
                )
                break

    # ── Helpers ───────────────────────────────────────────────────────────────

    def concession_rate(self) -> float:
        """
        Time-pressure concession factor [0, 1].
        As rounds progress, agents become more willing to concede.
        Rises from 0 at round 1 to 1 at max_rounds.
        """
        if not self.task:
            return 0.5
        return min(1.0, self.current_round / self.task.max_rounds)

    def should_activate_batna(self, proposal: ProposalPayload) -> bool:
        """Check if this proposal is below the agent's walk-away threshold."""
        return self.compute_utility(proposal) < self.batna_threshold

    def _base_message(
        self,
        msg_type: MessageType,
        public_reason: str,
        in_reply_to: Optional[str] = None,
        proposal: Optional[ProposalPayload] = None,
        utility_score: Optional[float] = None,
        rejection_reason: Optional[str] = None,
    ) -> NegotiationMessage:
        """Helper to construct a NegotiationMessage with boilerplate filled in."""
        return NegotiationMessage(
            round=self.current_round,
            sender=self.agent_id,
            msg_type=msg_type,
            task_id=self.task.task_id if self.task else "unknown",
            in_reply_to=in_reply_to,
            proposal=proposal,
            public_reason=public_reason,
            utility_score=utility_score,
            rejection_reason=rejection_reason,
        )
