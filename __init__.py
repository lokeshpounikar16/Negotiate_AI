"""Project root package shim for NegotiateAI imports."""

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_NESTED_PACKAGE = _PACKAGE_ROOT / "Negotiate_AI"
__path__ = []
if _NESTED_PACKAGE.exists():
    __path__.append(str(_NESTED_PACKAGE))
__path__.append(str(_PACKAGE_ROOT))
