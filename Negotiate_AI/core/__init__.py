# negotiateai/core/__init__.py
from __future__ import annotations

from importlib import import_module

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

__all__ = [
    "NegotiationMessage", "TaskAnnouncement", "TaskConstraints",
    "ProposalPayload", "AgentID", "MessageType", "QualityTier", "Concession",
    "NebiusClient", "get_nebius_client",
    "MessageBus", "get_message_bus",
    "RoundManager",
]


def __getattr__(name: str):
    if name == "RoundManager":
        return import_module(".round_manager", __name__).RoundManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
