import asyncio

from Negotiate_AI.agents.cost_agent import CostAgent
from Negotiate_AI.agents.quality_agent import QualityAgent
from Negotiate_AI.agents.risk_agent import RiskAgent
from Negotiate_AI.agents.timeline_agent import TimelineAgent
from Negotiate_AI.core.round_manager import RoundManager
from Negotiate_AI.core.schema import QualityTier, TaskAnnouncement, TaskConstraints


def test_full_workflow_round_manager_integration():
    task = TaskAnnouncement(
        title="Procure 500 servers",
        description="High-priority deployment with strict cost and quality constraints.",
        constraints=TaskConstraints(
            max_budget_usd=2_000_000,
            quantity=500,
            max_delivery_days=60,
            min_quality_tier=QualityTier.B_PLUS,
            max_vendor_risk=0.40,
            min_warranty_months=12,
        ),
        max_rounds=5,
    )

    agents = {
        "cost_agent": CostAgent(nebius_client=None, message_bus=None),
        "quality_agent": QualityAgent(nebius_client=None, message_bus=None),
        "timeline_agent": TimelineAgent(nebius_client=None, message_bus=None),
        "risk_agent": RiskAgent(nebius_client=None, message_bus=None),
    }
    for agent in agents.values():
        agent.task = task

    manager = RoundManager(agents, max_rounds=5, task_id=task.task_id)
    messages = asyncio.run(manager.execute_round(1))

    assert len(messages) == 4
    assert {msg.sender.value for msg in messages} == {
        "cost_agent",
        "quality_agent",
        "timeline_agent",
        "risk_agent",
    }
