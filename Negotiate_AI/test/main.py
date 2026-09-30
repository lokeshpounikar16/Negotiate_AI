"""
NegotiateAI — FastAPI Entrypoint
Week 1: Health checks + task creation + message log retrieval.
Week 2+ will add WebSocket streaming and full negotiation orchestration.
"""

from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from Negotiate_AI.core.message_bus import get_message_bus
from Negotiate_AI.core.nebius_client import get_nebius_client
from Negotiate_AI.core.schema import (
    MessageType,
    QualityTier,
    TaskAnnouncement,
    TaskConstraints,
)
from Negotiate_AI.agents.cost_agent import CostAgent

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("negotiateai.main")


# ── Lifespan: connect/disconnect Redis on startup/shutdown ───────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    bus = get_message_bus()
    await bus.connect()
    logger.info("✅ Redis message bus connected")
    yield
    await bus.disconnect()
    logger.info("👋 Redis message bus disconnected")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NegotiateAI",
    description="Agent-to-Agent Negotiation Protocol — Nebius × NVIDIA Hackathon",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class CreateTaskRequest(BaseModel):
    title: str
    description: str
    max_budget_usd: float
    quantity: int
    max_delivery_days: int
    min_quality_tier: QualityTier = QualityTier.B_PLUS
    max_vendor_risk: float = 0.40
    min_warranty_months: int = 12
    max_rounds: int = 10


class CreateTaskResponse(BaseModel):
    task_id: str
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

def _build_task(req: CreateTaskRequest) -> TaskAnnouncement:
    constraints = TaskConstraints(
        max_budget_usd=req.max_budget_usd,
        quantity=req.quantity,
        max_delivery_days=req.max_delivery_days,
        min_quality_tier=req.min_quality_tier,
        max_vendor_risk=req.max_vendor_risk,
        min_warranty_months=req.min_warranty_months,
    )
    return TaskAnnouncement(
        title=req.title,
        description=req.description,
        constraints=constraints,
        max_rounds=req.max_rounds,
    )


async def _run_cost_agent(task: TaskAnnouncement, bus):
    client = get_nebius_client()
    agent = CostAgent(nebius_client=client, message_bus=bus)
    agent.task = task
    agent.current_round = 1

    logger.info("Running Cost Agent for task %s", task.task_id)
    try:
        proposal_msg = await agent.generate_opening_proposal(task)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await bus.publish(task.task_id, proposal_msg)
    return proposal_msg

@app.get("/", tags=["meta"])
async def root():
    return {
        "project": "NegotiateAI",
        "version": "0.1.0",
        "week": 1,
        "status": "Foundation — Cost Agent + Message Bus",
    }


@app.get("/health", tags=["meta"])
async def health():
    """Check all dependencies: Nebius API + Redis message bus."""
    nebius_status = await get_nebius_client().health_check()
    bus_status    = await get_message_bus().health_check()
    ok = nebius_status.get("status") == "ok" and bus_status.get("status") == "ok"
    return {
        "status": "ok" if ok else "degraded",
        "nebius": nebius_status,
        "redis":  bus_status,
    }


@app.post("/tasks", response_model=CreateTaskResponse, tags=["tasks"])
async def create_task(req: CreateTaskRequest):
    """
    Create a new negotiation task.
    Returns a task_id that all subsequent calls use.
    """
    task = _build_task(req)

    bus = get_message_bus()
    await bus.store_task(task)
    await bus.publish(task.task_id, task.to_negotiation_message())

    logger.info(f"✅ Task created: {task.task_id} — {task.title}")
    return CreateTaskResponse(
        task_id=task.task_id,
        message=f"Task '{task.title}' created. "
                f"Budget: ${req.max_budget_usd:,.0f}, "
                f"Qty: {req.quantity}, "
                f"Deadline: {req.max_delivery_days}d",
    )


@app.post("/tasks/{task_id}/run-cost-agent", tags=["agents"])
async def run_cost_agent(task_id: str):
    """
    Week 1 endpoint: Run the Cost Agent for a given task.
    The agent researches vendors via Tavily and publishes an opening proposal.
    Returns the proposal message so you can inspect it immediately.
    """
    bus = get_message_bus()
    task = await bus.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"No task found for task_id={task_id}")

    existing_messages = await bus.get_log(task_id)
    if any(message.msg_type == MessageType.PROPOSAL for message in existing_messages):
        raise HTTPException(status_code=409, detail="A Cost Agent proposal already exists for this task")

    proposal_msg = await _run_cost_agent(task, bus)
    return {
        "task_id": task.task_id,
        "agent": "cost_agent",
        "round": proposal_msg.round,
        "proposal": proposal_msg.model_dump(mode="json"),
        "utility_score": proposal_msg.utility_score,
        "log_count": await bus.get_log_count(task.task_id),
    }


@app.post("/negotiate", tags=["agents"])
async def negotiate(req: CreateTaskRequest):
    """
    Week 1 combined endpoint:
    1. Creates the task
    2. Runs the Cost Agent to generate an opening proposal
    3. Returns the full proposal for inspection

    This is the primary Week 1 demo endpoint.
    """
    # Build task
    task = _build_task(req)

    bus = get_message_bus()
    await bus.store_task(task)
    await bus.publish(task.task_id, task.to_negotiation_message())

    proposal_msg = await _run_cost_agent(task, bus)

    # Return full context
    return {
        "task_id":       task.task_id,
        "task_title":    task.title,
        "agent":         "cost_agent",
        "round":         1,
        "proposal":      proposal_msg.model_dump(mode="json"),
        "utility_score": proposal_msg.utility_score,
        "log_count":     await bus.get_log_count(task.task_id),
        "next_steps":    "POST /tasks/{task_id}/log to see full message history",
    }


@app.get("/tasks/{task_id}/log", tags=["tasks"])
async def get_task_log(task_id: str):
    """
    Retrieve the full audit log for a negotiation session.
    Returns all messages in chronological order.
    """
    bus = get_message_bus()
    log = await bus.get_log(task_id)

    if not log:
        raise HTTPException(status_code=404, detail=f"No log found for task_id={task_id}")

    return {
        "task_id":    task_id,
        "msg_count":  len(log),
        "messages":   [m.to_agent_view() for m in log],
    }


@app.delete("/tasks/{task_id}", tags=["tasks"])
async def clear_task(task_id: str):
    """Clear all Redis data for a task (useful during development/testing)."""
    bus = get_message_bus()
    await bus.clear_task(task_id)
    return {"status": "cleared", "task_id": task_id}
