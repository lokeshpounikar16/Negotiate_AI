# negotiateai/agents/__init__.py
from pathlib import Path

from .base_agent import BaseAgent
from .cost_agent import CostAgent

_NESTED_PACKAGE = Path(__file__).resolve().parent.parent / "Negotiate_AI" / "agents"
__path__ = []
if _NESTED_PACKAGE.exists():
    __path__.append(str(_NESTED_PACKAGE))
__path__.append(str(Path(__file__).resolve().parent))

__all__ = ["BaseAgent", "CostAgent"]
