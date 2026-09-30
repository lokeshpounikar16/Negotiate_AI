"""
NegotiateAI — FastAPI Entrypoint
Week 1: Health checks + task creation + message log retrieval.
Week 2+ will add WebSocket streaming and full negotiation orchestration.
"""

from __future__ import annotations
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure the project root is importable regardless of where Python is launched from.
_pkg_root = str(Path(__file__).resolve().parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

load_dotenv(Path(__file__).resolve().parent / ".env")

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    bus = get_message_bus()
    await bus.connect()
    logger.info("✅ Redis message bus connected")
    yield
    await bus.disconnect()
    logger.info("👋 Redis message bus disconnected")


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


class RunCostAgentRequest(BaseModel):
    task_id: str


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
    bus_status = await get_message_bus().health_check()
    ok = nebius_status.get("status") == "ok" and bus_status.get("status") == "ok"
    return {
        "status": "ok" if ok else "degraded",
        "nebius": nebius_status,
        "redis": bus_status,
    }


@app.post("/tasks", response_model=CreateTaskResponse, tags=["tasks"])
async def create_task(req: CreateTaskRequest):
    constraints = TaskConstraints(
        max_budget_usd=req.max_budget_usd,
        quantity=req.quantity,
        max_delivery_days=req.max_delivery_days,
        min_quality_tier=req.min_quality_tier,
        max_vendor_risk=req.max_vendor_risk,
        min_warranty_months=req.min_warranty_months,
    )
    task = TaskAnnouncement(
        title=req.title,
        description=req.description,
        constraints=constraints,
        max_rounds=req.max_rounds,
    )

    bus = get_message_bus()
    announcement = task.to_negotiation_message()
    await bus.publish(task.task_id, announcement)

    logger.info(f"✅ Task created: {task.task_id} — {task.title}")
    return CreateTaskResponse(
        task_id=task.task_id,
        message=f"Task '{task.title}' created. "
        f"Budget: ${req.max_budget_usd:,.0f}, "
        f"Qty: {req.quantity}, "
        f"Deadline: {req.max_delivery_days}d",
    )


@app.post("/tasks/{task_id}/run-cost-agent", tags=["agents"])
async def run_cost_agent(task_id: str, body: Optional[RunCostAgentRequest] = None):
    bus = get_message_bus()
    log = await bus.get_log(task_id)

    if not log:
        raise HTTPException(status_code=404, detail=f"No messages found for task_id={task_id}")

    announcement_msg = next(
        (m for m in log if m.msg_type == MessageType.TASK_ANNOUNCEMENT),
        None,
    )
    if not announcement_msg:
        raise HTTPException(status_code=400, detail="No TASK_ANNOUNCEMENT found in log")

    # Week 1: use the combined /negotiate endpoint to generate a proposal inline.
    raise HTTPException(
        status_code=501,
        detail="Week 1 stub: Use the /negotiate endpoint instead, which takes the full task inline.",
    )


@app.post("/negotiate", tags=["agents"])
async def negotiate(req: CreateTaskRequest):
    constraints = TaskConstraints(
        max_budget_usd=req.max_budget_usd,
        quantity=req.quantity,
        max_delivery_days=req.max_delivery_days,
        min_quality_tier=req.min_quality_tier,
        max_vendor_risk=req.max_vendor_risk,
        min_warranty_months=req.min_warranty_months,
    )
    task = TaskAnnouncement(
        title=req.title,
        description=req.description,
        constraints=constraints,
        max_rounds=req.max_rounds,
    )

    bus = get_message_bus()
    client = get_nebius_client()
    await bus.publish(task.task_id, task.to_negotiation_message())

    agent = CostAgent(nebius_client=client, message_bus=bus)
    agent.task = task
    agent.current_round = 1

    logger.info(f"🤖 Running Cost Agent for task {task.task_id}...")
    proposal_msg = await agent.generate_opening_proposal(task)
    await bus.publish(task.task_id, proposal_msg)

    return {
        "task_id": task.task_id,
        "task_title": task.title,
        "agent": "cost_agent",
        "round": 1,
        "proposal": proposal_msg.model_dump(mode="json"),
        "utility_score": proposal_msg.utility_score,
        "log_count": await bus.get_log_count(task.task_id),
        "next_steps": "POST /tasks/{task_id}/log to see full message history",
    }


@app.get("/tasks/{task_id}/log", tags=["tasks"])
async def get_task_log(task_id: str):
    bus = get_message_bus()
    log = await bus.get_log(task_id)

    if not log:
        raise HTTPException(status_code=404, detail=f"No log found for task_id={task_id}")

    return {
        "task_id": task_id,
        "msg_count": len(log),
        "messages": [m.model_dump(mode="json") for m in log],
    }


@app.delete("/tasks/{task_id}", tags=["tasks"])
async def clear_task(task_id: str):
    bus = get_message_bus()
    await bus.clear_task(task_id)
    return {"status": "cleared", "task_id": task_id}
