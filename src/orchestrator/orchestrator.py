from __future__ import annotations

from typing import Dict, List

from Negotiate_AI.agents.cost_agent import CostAgent
from Negotiate_AI.agents.quality_agent import QualityAgent
from Negotiate_AI.agents.risk_agent import RiskAgent
from Negotiate_AI.agents.timeline_agent import TimelineAgent
from Negotiate_AI.core.round_manager import RoundManager
from Negotiate_AI.core.schema import NegotiationMessage, TaskAnnouncement


class NegotiationOrchestrator:
    """Workflow-oriented facade that matches the implementation guide structure."""

    def __init__(self, max_rounds: int = 10):
        self.max_rounds = max_rounds
        self.agents: Dict[str, object] = {}
        self.round_manager: RoundManager | None = None
        self.history: List[NegotiationMessage] = []

    def initialize(self, task: TaskAnnouncement):
        self.agents = {
            "cost_agent": CostAgent(nebius_client=None, message_bus=None),
            "quality_agent": QualityAgent(nebius_client=None, message_bus=None),
            "timeline_agent": TimelineAgent(nebius_client=None, message_bus=None),
            "risk_agent": RiskAgent(nebius_client=None, message_bus=None),
        }
        for agent in self.agents.values():
            agent.task = task
        self.round_manager = RoundManager(self.agents, max_rounds=self.max_rounds, task_id=task.task_id)

    async def run(self, task: TaskAnnouncement):
        if self.round_manager is None:
            self.initialize(task)
        messages = await self.round_manager.execute_round(1)
        self.history.extend(messages)
        return messages
