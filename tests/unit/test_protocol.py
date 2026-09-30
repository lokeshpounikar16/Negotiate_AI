from src.protocol.message import MessageType, NegotiationMessage, ProposalDetail


def test_message_creation_from_workflow_package():
    message = NegotiationMessage(
        round=1,
        sender="cost_agent",
        msg_type=MessageType.PROPOSAL,
        task_id="workflow-task",
        proposal=ProposalDetail(
            vendor="Dell",
            unit_price_usd=3800,
            quantity=500,
            total_cost_usd=1_900_000,
            delivery_days=55,
            quality_tier="B+",
            vendor_risk_score=0.24,
        ),
        public_reason="Test workflow proposal",
        utility_score=0.75,
    )

    assert message.sender == "cost_agent"
    assert message.proposal.vendor == "Dell"
    assert message.utility_score == 0.75
