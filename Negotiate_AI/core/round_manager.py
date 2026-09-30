"""Simple round manager for multi-agent procurement negotiation."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..core.schema import AgentID, MessageType, NegotiationMessage, ProposalPayload


class RoundManager:
    """Coordinates a single negotiation round across multiple agents."""

    def __init__(self, agents: Dict[str, object], max_rounds: int = 10, task_id: str = "default-task"):
        self.agents = agents
        self.max_rounds = max_rounds
        self.task_id = task_id
        self.current_round = 0
        self.messages: List[NegotiationMessage] = []
        self.agent_order = list(agents.keys())
        self.agent_utilities: Dict[str, List[float]] = {
            agent_id: [] for agent_id in self.agent_order
        }

    def _coerce_agent_id(self, agent_id: str | AgentID) -> AgentID:
        if isinstance(agent_id, AgentID):
            return agent_id
        return AgentID(agent_id)

    def _record_message(self, msg: NegotiationMessage) -> None:
        self.messages.append(msg)
        self.agent_utilities.setdefault(msg.sender.value, []).append(msg.utility_score or 0.0)

    async def execute_round(self, round_num: int) -> List[NegotiationMessage]:
        self.current_round = round_num
        round_messages = []

        if round_num == 1:
            for agent_id in self.agent_order:
                agent = self.agents[agent_id]
                proposal = agent.generate_proposal() if hasattr(agent, "generate_proposal") else None
                if proposal is None:
                    continue
                msg = NegotiationMessage(
                    round=round_num,
                    sender=self._coerce_agent_id(agent_id),
                    msg_type=MessageType.PROPOSAL,
                    task_id=self.task_id,
                    proposal=proposal,
                    public_reason=f"{agent.__class__.__name__} opening proposal",
                    utility_score=getattr(agent, "compute_utility", lambda _: 0.0)(proposal),
                )
                round_messages.append(msg)
                self._record_message(msg)
            return round_messages

        if not self.messages:
            return round_messages

        last_proposal: Optional[ProposalPayload] = self.messages[-1].proposal
        for agent_id in self.agent_order:
            if last_proposal is None:
                continue
            agent = self.agents[agent_id]
            evaluate = getattr(agent, "evaluate_proposal", None)
            if callable(evaluate) and evaluate(last_proposal):
                msg = NegotiationMessage(
                    round=round_num,
                    sender=self._coerce_agent_id(agent_id),
                    msg_type=MessageType.ACCEPT,
                    task_id=self.task_id,
                    proposal=last_proposal,
                    public_reason=f"{agent.__class__.__name__} accepts the current proposal.",
                    utility_score=getattr(agent, "compute_utility", lambda _: 0.0)(last_proposal),
                )
            else:
                counter = agent.generate_counter_proposal(last_proposal)
                msg = NegotiationMessage(
                    round=round_num,
                    sender=self._coerce_agent_id(agent_id),
                    msg_type=MessageType.COUNTER_PROPOSAL,
                    task_id=self.task_id,
                    proposal=counter,
                    public_reason=f"{agent.__class__.__name__} counter-proposes a revised offer.",
                    utility_score=getattr(agent, "compute_utility", lambda _: 0.0)(counter),
                )
            round_messages.append(msg)
            self._record_message(msg)
        return round_messages

    def has_converged(self) -> bool:
        if len(self.messages) < len(self.agents):
            return False
        recent = self.messages[-len(self.agents):]
        return all(msg.msg_type == MessageType.ACCEPT for msg in recent)

    def get_pareto_frontier(self) -> List[NegotiationMessage]:
        frontier: List[NegotiationMessage] = []
        for msg in self.messages:
            if msg.utility_score is None:
                continue
            dominated = False
            for other in self.messages:
                if other.msg_id == msg.msg_id:
                    continue
                if other.utility_score is not None and other.utility_score > msg.utility_score:
                    dominated = True
                    break
            if not dominated:
                frontier.append(msg)
        return frontier


__all__ = ["RoundManager"]
