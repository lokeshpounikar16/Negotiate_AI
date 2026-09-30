"""
NegotiationMessage Protocol Schema — v1
The typed message format all agents use to communicate.
This is the core technical contribution of the project.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ────────────────────────────────────────────────────────────────────

class AgentID(str, Enum):
    COST      = "cost_agent"
    QUALITY   = "quality_agent"
    TIMELINE  = "timeline_agent"
    RISK      = "risk_agent"
    MEDIATOR  = "mediator_agent"
    SYSTEM    = "system"          # for task announcements


class MessageType(str, Enum):
    TASK_ANNOUNCEMENT = "TASK_ANNOUNCEMENT"  # system → all agents
    PROPOSAL          = "PROPOSAL"           # opening bid
    COUNTER_PROPOSAL  = "COUNTER_PROPOSAL"   # counter to a prior proposal
    CONCESSION        = "CONCESSION"         # explicit concession on a dimension
    ACCEPT            = "ACCEPT"             # agent accepts current proposal
    REJECT            = "REJECT"             # agent rejects, no counter offered
    MEDIATOR_PROPOSAL = "MEDIATOR_PROPOSAL"  # mediator's compromise package
    CONSENSUS         = "CONSENSUS"          # system signals full agreement


class QualityTier(str, Enum):
    A_PLUS  = "A+"
    A       = "A"
    A_MINUS = "A-"
    B_PLUS  = "B+"
    B       = "B"
    B_MINUS = "B-"
    C       = "C"


# ── Proposal payload ─────────────────────────────────────────────────────────

class ProposalPayload(BaseModel):
    """
    The actual procurement proposal being negotiated.
    All dimensions that agents care about live here.
    """
    vendor: str                  = Field(..., description="Vendor name")
    unit_price_usd: float        = Field(..., gt=0, description="Price per unit in USD")
    quantity: int                = Field(..., gt=0, description="Number of units")
    total_cost_usd: float        = Field(..., gt=0, description="Total cost (unit_price × quantity)")
    delivery_days: int           = Field(..., gt=0, description="Estimated delivery in calendar days")
    quality_tier: QualityTier    = Field(..., description="Vendor quality tier rating")
    vendor_risk_score: float     = Field(..., ge=0.0, le=1.0, description="Risk score: 0=safe, 1=max risk")
    warranty_months: int         = Field(default=12, ge=0, description="Warranty duration in months")
    sla_uptime_pct: float        = Field(default=99.0, ge=0.0, le=100.0, description="SLA uptime guarantee %")
    source_country: str          = Field(default="Unknown", description="Country of manufacture")
    notes: Optional[str]         = Field(default=None, description="Free-text notes on this proposal")

    @model_validator(mode="after")
    def check_total_cost(self) -> "ProposalPayload":
        expected = round(self.unit_price_usd * self.quantity, 2)
        if abs(self.total_cost_usd - expected) > 1.0:
            raise ValueError(
                f"total_cost_usd ({self.total_cost_usd}) doesn't match "
                f"unit_price_usd × quantity ({expected})"
            )
        return self


# ── Concession record ────────────────────────────────────────────────────────

class Concession(BaseModel):
    """Structured record of what an agent gave up in this round."""
    dimension: str       = Field(..., description="What was conceded, e.g. 'quality_tier'")
    from_value: str      = Field(..., description="Previous value, e.g. 'A+'")
    to_value: str        = Field(..., description="New conceded value, e.g. 'A'")
    utility_cost: float  = Field(..., ge=0.0, le=1.0, description="How much utility this concession costs the agent")


# ── The core message ──────────────────────────────────────────────────────────

class NegotiationMessage(BaseModel):
    """
    The core typed message in the A2A negotiation protocol.
    Every agent communicates exclusively through this format.
    """
    # Protocol metadata
    msg_id: str              = Field(default_factory=lambda: str(uuid.uuid4()))
    protocol_version: str    = Field(default="1.0")
    round: int               = Field(..., ge=0, description="Negotiation round number (0 = task announcement)")
    sender: AgentID          = Field(..., description="Which agent sent this")
    msg_type: MessageType    = Field(..., description="Type of message in the protocol")
    timestamp: datetime      = Field(default_factory=lambda: datetime.now(timezone.utc))

    # References
    in_reply_to: Optional[str]  = Field(default=None, description="msg_id of the message this responds to")
    task_id: str                = Field(..., description="The task/negotiation session this belongs to")

    # Content
    proposal: Optional[ProposalPayload]    = Field(default=None, description="The proposal payload (required for PROPOSAL/COUNTER/MEDIATOR types)")
    concessions_made: list[Concession]     = Field(default_factory=list, description="Explicit concessions from prior position")
    public_reason: str                     = Field(..., min_length=10, description="Public justification for this message (visible to all agents)")
    rejection_reason: Optional[str]        = Field(default=None, description="Why the agent rejects (REJECT messages only)")

    # Private agent state (logged but not broadcast to other agents)
    utility_score: Optional[float]         = Field(default=None, ge=0.0, le=1.0, description="Agent's private utility score for this proposal")
    batna_activated: bool                  = Field(default=False, description="True if agent has activated its BATNA")
    is_final_position: bool                = Field(default=False, description="True if agent cannot concede further")

    @model_validator(mode="after")
    def validate_message_contract(self):
        needs_proposal = {
            MessageType.PROPOSAL,
            MessageType.COUNTER_PROPOSAL,
            MessageType.MEDIATOR_PROPOSAL,
        }
        if self.msg_type in needs_proposal and self.proposal is None:
            raise ValueError(f"proposal payload is required for msg_type={self.msg_type}")
        if self.msg_type == MessageType.REJECT and not self.rejection_reason:
            raise ValueError("rejection_reason is required for REJECT messages")
        return self

    def to_agent_view(self) -> dict:
        """
        Returns a version of the message safe to broadcast to other agents.
        Strips private agent state before broadcasting.
        """
        d = self.model_dump(mode="json")
        for field in ("utility_score", "batna_activated", "is_final_position"):
            d.pop(field, None)
        return d

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Task announcement (system → all agents) ───────────────────────────────────

class TaskConstraints(BaseModel):
    """
    Hard constraints the final proposal must satisfy.
    Any proposal violating these is automatically rejected by the system.
    """
    max_budget_usd: float        = Field(..., gt=0)
    max_delivery_days: int       = Field(..., gt=0)
    min_quality_tier: QualityTier = Field(default=QualityTier.B)
    max_vendor_risk: float       = Field(default=0.5, ge=0.0, le=1.0)
    min_warranty_months: int     = Field(default=12, ge=0)
    quantity: int                = Field(..., gt=0)

    def validate_proposal(self, p: ProposalPayload) -> tuple[bool, list[str]]:
        """Returns (is_valid, list_of_violations)."""
        violations = []
        if p.quantity != self.quantity:
            violations.append(f"Quantity {p.quantity} does not match required quantity {self.quantity}")
        if p.total_cost_usd > self.max_budget_usd:
            violations.append(f"Cost ${p.total_cost_usd:,.0f} exceeds budget ${self.max_budget_usd:,.0f}")
        if p.delivery_days > self.max_delivery_days:
            violations.append(f"Delivery {p.delivery_days}d exceeds max {self.max_delivery_days}d")
        if p.vendor_risk_score > self.max_vendor_risk:
            violations.append(f"Risk {p.vendor_risk_score:.2f} exceeds max {self.max_vendor_risk:.2f}")
        if p.warranty_months < self.min_warranty_months:
            violations.append(f"Warranty {p.warranty_months}mo below min {self.min_warranty_months}mo")
        # Quality tier comparison (A+ > A > A- > B+ > B > B- > C)
        tier_order = [t.value for t in QualityTier]
        if tier_order.index(p.quality_tier.value) > tier_order.index(self.min_quality_tier.value):
            violations.append(f"Quality tier {p.quality_tier.value} below minimum {self.min_quality_tier.value}")
        return len(violations) == 0, violations


class TaskAnnouncement(BaseModel):
    """The task that kicks off a negotiation session."""
    task_id: str            = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str              = Field(..., description="Human-readable task title")
    description: str        = Field(..., description="Full task description")
    constraints: TaskConstraints
    max_rounds: int         = Field(default=10, ge=1, le=50)
    created_at: datetime    = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_negotiation_message(self) -> NegotiationMessage:
        """Convert the task into the first message in the negotiation."""
        return NegotiationMessage(
            round=0,
            sender=AgentID.SYSTEM,
            msg_type=MessageType.TASK_ANNOUNCEMENT,
            task_id=self.task_id,
            public_reason=f"Task announced: {self.title}. "
                          f"Budget ≤ ${self.constraints.max_budget_usd:,.0f}, "
                          f"delivery ≤ {self.constraints.max_delivery_days}d, "
                          f"quality ≥ {self.constraints.min_quality_tier.value}.",
        )
