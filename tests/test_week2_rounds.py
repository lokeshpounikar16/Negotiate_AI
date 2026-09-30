import asyncio

from fastapi.testclient import TestClient

from Negotiate_AI.agents.cost_agent import CostAgent
from Negotiate_AI.agents.mediator_agent import MediatorAgent
from Negotiate_AI.agents.quality_agent import QualityAgent
from Negotiate_AI.agents.risk_agent import RiskAgent
from Negotiate_AI.agents.timeline_agent import TimelineAgent
from Negotiate_AI.core.round_manager import RoundManager
from Negotiate_AI.core.schema import (
    QualityTier,
    TaskAnnouncement,
    TaskConstraints,
)
from main import app


def test_negotiation_round_manager_runs_all_four_agents():
    task = TaskAnnouncement(
        title="Procure 500 servers",
        description="Need 500 servers for new data center.",
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
    assert all(msg.utility_score is not None for msg in messages)
    assert all(0.0 <= msg.utility_score <= 1.0 for msg in messages)


def test_week2_negotiation_endpoint_is_registered():
    client = TestClient(app)
    routes = [route.path for route in client.app.routes]
    assert "/negotiation/run" in routes


def test_mediator_detects_deadlock_after_stagnation():
    mediator = MediatorAgent(deadlock_threshold=2)
    messages = []

    for round_num in range(1, 5):
        for agent_name in ["cost_agent", "quality_agent", "timeline_agent", "risk_agent"]:
            messages.append(
                {
                    "round": round_num,
                    "sender": agent_name,
                    "msg_type": "PROPOSAL",
                    "task_id": "task-123",
                    "proposal": None,
                    "public_reason": "stagnant",
                    "utility_score": 0.40,
                }
            )

    assert mediator.detect_deadlock(messages) is True
