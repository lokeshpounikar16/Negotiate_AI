# negotiateai/core/__init__.py
from pathlib import Path

from .schema import (
    NegotiationMessage,
    TaskAnnouncement,
    TaskConstraints,
    ProposalPayload,
    AgentID,
    MessageType,
    QualityTier,
    Concession,
)
from .nebius_client import NebiusClient, get_nebius_client
from .message_bus import MessageBus, get_message_bus

_NESTED_PACKAGE = Path(__file__).resolve().parent.parent / "Negotiate_AI" / "core"
__path__ = []
if _NESTED_PACKAGE.exists():
    __path__.append(str(_NESTED_PACKAGE))
__path__.append(str(Path(__file__).resolve().parent))

__all__ = [
    "NegotiationMessage", "TaskAnnouncement", "TaskConstraints",
    "ProposalPayload", "AgentID", "MessageType", "QualityTier", "Concession",
    "NebiusClient", "get_nebius_client",
    "MessageBus", "get_message_bus",
]
