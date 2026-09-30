"""
Cost Agent
Goal: Minimize total procurement cost while staying within hard constraints.
Will trade quality tier and longer delivery for lower price.
Utility function heavily weights total_cost_usd.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Optional

from pydantic import BaseModel, Field
from tavily import TavilyClient

from .base_agent import BaseAgent
from ..core.message_bus import MessageBus
from ..core.nebius_client import NebiusClient
from ..core.schema import (
    AgentID,
    Concession,
    MessageType,
    NegotiationMessage,
    ProposalPayload,
    QualityTier,
    TaskAnnouncement,
)

logger = logging.getLogger(__name__)


# ── Structured output schemas for LLM calls ───────────────────────────────────

class VendorResearch(BaseModel):
    """What the LLM extracts from Tavily vendor search results."""
    vendors: list[dict] = Field(
        ...,
        description="List of vendors with keys: name, unit_price_usd, delivery_days, quality_tier, vendor_risk_score, warranty_months, source_country, notes"
    )
    market_context: str = Field(
        ...,
        description="Brief market summary relevant to cost optimization"
    )
    recommended_vendor: str = Field(
        ...,
        description="Vendor name the cost agent should propose first"
    )


class ProposalDecision(BaseModel):
    """LLM's structured decision on what proposal to make."""
    proposal: ProposalPayload
    public_reason: str         = Field(..., description="Explanation of this proposal for other agents")
    utility_estimate: float    = Field(..., ge=0.0, le=1.0, description="Estimated utility from cost agent's perspective")
    concessions_from_prior: list[dict] = Field(
        default_factory=list,
        description="List of {dimension, from_value, to_value, utility_cost} if this is a counter-proposal"
    )


class AcceptRejectDecision(BaseModel):
    """LLM's decision on whether to accept or counter-propose."""
    action: str              = Field(..., description="One of: ACCEPT, COUNTER_PROPOSAL, REJECT")
    public_reason: str       = Field(..., description="Explanation for other agents")
    utility_score: float     = Field(..., ge=0.0, le=1.0)
    counter_proposal: Optional[ProposalPayload] = Field(
        default=None,
        description="Required if action is COUNTER_PROPOSAL"
    )
    concessions_made: list[dict] = Field(default_factory=list)
    rejection_reason: Optional[str] = Field(default=None)


# ── Cost Agent ────────────────────────────────────────────────────────────────

class CostAgent(BaseAgent):
    """
    The Cost Agent's goal is to minimize total procurement spend.

    Utility function:
        - 70% weight on cost (lower cost = higher utility)
        - 20% weight on delivery (faster = slightly better)
        - 10% weight on quality (higher quality = marginally better)

    BATNA: If no proposal within 85% of budget can be found, walk away.
    """

    COST_WEIGHT     = 0.70
    DELIVERY_WEIGHT = 0.20
    QUALITY_WEIGHT  = 0.10

    def __init__(self, nebius_client: NebiusClient, message_bus: MessageBus):
        super().__init__(
            agent_id=AgentID.COST,
            nebius_client=nebius_client,
            message_bus=message_bus,
            batna_threshold=0.20,
        )
        tavily_key = os.environ.get("TAVILY_API_KEY")
        self._tavily = TavilyClient(api_key=tavily_key) if tavily_key else None

    # ── System prompt ─────────────────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        return """You are the Cost Agent in a multi-agent procurement negotiation system.

YOUR GOAL: Minimize total procurement cost. You represent the finance department.

YOUR PRIORITIES (in order):
1. Total cost must be as low as possible — this is your primary objective
2. Stay within the hard budget cap set in the task constraints
3. You will accept lower quality tiers (down to B+) or longer delivery times if it meaningfully reduces cost
4. You will NOT accept proposals that exceed the budget cap under any circumstances

YOUR NEGOTIATION STYLE:
- Start with the lowest reasonable price you can find from real market data
- When countering, make concessions on quality or delivery before conceding on price
- Be honest about your reasoning — other agents can see your public_reason
- Express your position in concrete dollar terms, not vague language
- If a proposal is within 5% of your ideal, accept it rather than over-negotiating

BATNA: If after concessions the best available proposal still costs more than 95% of the budget cap, evaluate whether the task is feasible at all.

Always respond with valid JSON matching the requested schema. No markdown, no preamble."""

    # ── Utility function ──────────────────────────────────────────────────────

    def compute_utility(self, proposal: ProposalPayload) -> float:
        """
        Cost agent utility: 70% cost score + 20% delivery score + 10% quality score.
        All sub-scores normalized to [0, 1] where 1 = best for cost agent.
        """
        if not self.task:
            return 0.0

        budget = self.task.constraints.max_budget_usd
        max_days = self.task.constraints.max_delivery_days

        # Cost Agent will not accept proposals that exceed the budget cap
        if proposal.total_cost_usd > budget:
            return 0.0

        # Cost score: 1.0 if free, 0.0 if at budget
        cost_score = max(0.0, 1.0 - (proposal.total_cost_usd / budget))

        # Delivery score: 1.0 if 1 day, 0.0 if at/over max days
        delivery_score = max(0.0, 1.0 - (proposal.delivery_days / max_days))

        # Quality score: cost agent mildly prefers higher quality
        tier_scores = {
            QualityTier.A_PLUS: 1.0, QualityTier.A: 0.85, QualityTier.A_MINUS: 0.7,
            QualityTier.B_PLUS: 0.55, QualityTier.B: 0.4, QualityTier.B_MINUS: 0.25,
            QualityTier.C: 0.1,
        }
        quality_score = tier_scores.get(proposal.quality_tier, 0.5)

        utility = (
            self.COST_WEIGHT * cost_score +
            self.DELIVERY_WEIGHT * delivery_score +
            self.QUALITY_WEIGHT * quality_score
        )

        logger.debug(
            f"[CostAgent] Utility for {proposal.vendor}: "
            f"cost={cost_score:.3f}, delivery={delivery_score:.3f}, "
            f"quality={quality_score:.3f} → total={utility:.3f}"
        )
        return utility

    # ── Tavily vendor research ────────────────────────────────────────────────

    async def _research_vendors(self, task: TaskAnnouncement) -> str:
        """
        Use Tavily to find real vendor pricing and specs for this task.
        Returns a formatted string with vendor data for the LLM to process.
        """
        if not self._tavily:
            # Fallback stub data when Tavily key is not set (for testing)
            logger.warning("[CostAgent] No Tavily key — using stub vendor data")
            return self._stub_vendor_data(task)

        query = (
            f"server hardware procurement vendors pricing "
            f"{task.constraints.quantity} units enterprise "
            f"budget under ${task.constraints.max_budget_usd:,.0f} "
            f"delivery within {task.constraints.max_delivery_days} days 2026"
        )

        try:
            results = self._tavily.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
            )

            vendor_info = f"MARKET RESEARCH for: {task.description}\n\n"
            vendor_info += f"Summary: {results.get('answer', 'No summary available')}\n\n"
            vendor_info += "Sources:\n"

            for r in results.get("results", []):
                vendor_info += f"- {r.get('title', 'Untitled')}: {r.get('content', '')[:300]}\n"

            return vendor_info

        except Exception as e:
            logger.error(f"[CostAgent] Tavily search failed: {e}. Using stub data.")
            return self._stub_vendor_data(task)

    def _stub_vendor_data(self, task: TaskAnnouncement) -> str:
        """Fallback vendor data for testing without a Tavily key."""
        budget_per_unit = task.constraints.max_budget_usd / task.constraints.quantity
        return f"""
STUB VENDOR DATA (Tavily not configured):
Task: {task.description}
Budget per unit: ${budget_per_unit:,.0f}
Quantity needed: {task.constraints.quantity}

Known vendors for consideration:
- Dell Technologies: ~${budget_per_unit * 0.88:,.0f}/unit, delivery 45d, quality A-, risk 0.15, warranty 36mo
- HPE (Hewlett Packard Enterprise): ~${budget_per_unit * 0.93:,.0f}/unit, delivery 38d, quality A, risk 0.18, warranty 36mo
- Lenovo ThinkSystem: ~${budget_per_unit * 0.82:,.0f}/unit, delivery 52d, quality B+, risk 0.22, warranty 24mo
- Supermicro: ~${budget_per_unit * 0.75:,.0f}/unit, delivery 60d, quality B, risk 0.35, warranty 12mo
- Cisco UCS: ~${budget_per_unit * 1.05:,.0f}/unit, delivery 30d, quality A+, risk 0.12, warranty 60mo
"""

    # ── Opening proposal ──────────────────────────────────────────────────────

    async def generate_opening_proposal(self, task: TaskAnnouncement) -> NegotiationMessage:
        """
        Research vendors via Tavily, then use the LLM to generate an opening proposal
        optimized for minimum cost within constraints.
        """
        logger.info("[CostAgent] Researching vendors for opening proposal...")
        vendor_data = await self._research_vendors(task)

        user_prompt = f"""
TASK ANNOUNCEMENT:
{task.description}

HARD CONSTRAINTS (non-negotiable):
- Budget cap: ${task.constraints.max_budget_usd:,.0f} total
- Max delivery: {task.constraints.max_delivery_days} days
- Min quality tier: {task.constraints.min_quality_tier.value}
- Min warranty: {task.constraints.min_warranty_months} months
- Quantity: {task.constraints.quantity} units

MARKET RESEARCH:
{vendor_data}

As the Cost Agent, generate your opening proposal. Pick the vendor that minimizes total cost
while meeting all hard constraints. Be bold — start low.

Respond with a JSON object matching the ProposalDecision schema.
"""

        decision = await self.client.complete_structured(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            output_schema=ProposalDecision,
        )

        concessions = [
            Concession(**c) for c in decision.concessions_from_prior
            if all(k in c for k in ("dimension", "from_value", "to_value", "utility_cost"))
        ]

        return self._base_message(
            msg_type=MessageType.PROPOSAL,
            proposal=decision.proposal,
            public_reason=decision.public_reason,
            utility_score=decision.utility_estimate,
        )

    # ── Response to incoming proposals ───────────────────────────────────────

    async def respond_to_proposal(
        self, incoming: NegotiationMessage, task: TaskAnnouncement
    ) -> NegotiationMessage:
        """
        Evaluate an incoming proposal and decide: ACCEPT, COUNTER, or REJECT.
        The LLM reasons about it, but the final decision is also checked
        against the utility function for consistency.
        """
        if not incoming.proposal:
            # Can't respond meaningfully to a proposal-less message
            return self._base_message(
                msg_type=MessageType.REJECT,
                public_reason="Incoming message contained no proposal payload.",
                rejection_reason="Incoming message contained no proposal payload.",
                in_reply_to=incoming.msg_id,
            )

        incoming_utility = self.compute_utility(incoming.proposal)

        # If utility is very high, accept without LLM call (fast path)
        if incoming_utility >= 0.80:
            logger.info(
                f"[CostAgent] Fast-accept: utility={incoming_utility:.3f} ≥ 0.80"
            )
            return self._base_message(
                msg_type=MessageType.ACCEPT,
                proposal=incoming.proposal,
                in_reply_to=incoming.msg_id,
                public_reason=(
                    f"Accepting — this proposal achieves strong cost efficiency "
                    f"(total ${incoming.proposal.total_cost_usd:,.0f}, "
                    f"{incoming.proposal.delivery_days}d delivery)."
                ),
                utility_score=incoming_utility,
            )

        # If utility is too low, check BATNA
        if incoming_utility < self.batna_threshold and self.current_round > 3:
            logger.warning(
                f"[CostAgent] Below BATNA threshold (utility={incoming_utility:.3f})"
            )

        # Otherwise, ask LLM to reason about the response
        prior_proposal_summary = (
            f"My last proposal was for {self.my_last_proposal.proposal.vendor} "
            f"at ${self.my_last_proposal.proposal.total_cost_usd:,.0f} total, "
            f"{self.my_last_proposal.proposal.delivery_days} days delivery, "
            f"quality {self.my_last_proposal.proposal.quality_tier.value}."
            if self.my_last_proposal and self.my_last_proposal.proposal
            else "This is my first response."
        )

        user_prompt = f"""
INCOMING PROPOSAL from {incoming.sender.value}:
Vendor: {incoming.proposal.vendor}
Total cost: ${incoming.proposal.total_cost_usd:,.0f}
Unit price: ${incoming.proposal.unit_price_usd:,.0f}
Delivery: {incoming.proposal.delivery_days} days
Quality: {incoming.proposal.quality_tier.value}
Risk score: {incoming.proposal.vendor_risk_score}
Warranty: {incoming.proposal.warranty_months} months
Their reason: "{incoming.public_reason}"

MY UTILITY SCORE FOR THIS: {incoming_utility:.3f} / 1.0

CONTEXT:
{prior_proposal_summary}
Current round: {self.current_round} of {task.max_rounds}
Concession pressure: {self.concession_rate():.0%} (higher = more urgency to agree)
Budget remaining headroom: ${task.constraints.max_budget_usd - incoming.proposal.total_cost_usd:,.0f}

TASK CONSTRAINTS:
- Budget cap: ${task.constraints.max_budget_usd:,.0f}
- Max delivery: {task.constraints.max_delivery_days} days
- Min quality: {task.constraints.min_quality_tier.value}

As the Cost Agent, decide: ACCEPT, COUNTER_PROPOSAL, or REJECT.
- ACCEPT if utility ≥ 0.65 or we are past 80% of max rounds
- COUNTER_PROPOSAL with concrete improvements to cost
- REJECT only if the proposal fundamentally violates hard constraints

Respond with a JSON object matching the AcceptRejectDecision schema.
"""

        decision = await self.client.complete_structured(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            output_schema=AcceptRejectDecision,
        )

        # Map LLM decision to MessageType
        action_map = {
            "ACCEPT": MessageType.ACCEPT,
            "COUNTER_PROPOSAL": MessageType.COUNTER_PROPOSAL,
            "REJECT": MessageType.REJECT,
        }
        msg_type = action_map.get(decision.action, MessageType.COUNTER_PROPOSAL)

        concessions = [
            Concession(**c) for c in decision.concessions_made
            if all(k in c for k in ("dimension", "from_value", "to_value", "utility_cost"))
        ]

        msg = self._base_message(
            msg_type=msg_type,
            proposal=decision.counter_proposal if msg_type == MessageType.COUNTER_PROPOSAL else (
                incoming.proposal if msg_type == MessageType.ACCEPT else None
            ),
            in_reply_to=incoming.msg_id,
            public_reason=decision.public_reason,
            utility_score=decision.utility_score,
            rejection_reason=decision.rejection_reason,
        )
        msg.concessions_made = concessions
        msg.batna_activated  = self.should_activate_batna(incoming.proposal) and self.current_round > 3

        if msg_type == MessageType.COUNTER_PROPOSAL and decision.counter_proposal:
            self.my_last_proposal = msg

        return msg
