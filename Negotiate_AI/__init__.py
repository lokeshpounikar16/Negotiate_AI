"""NegotiateAI Week 1 application package."""

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PARENT_ROOT = _PACKAGE_ROOT.parent
__path__ = [str(_PACKAGE_ROOT), str(_PARENT_ROOT)]


from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
__path__ = [str(_PACKAGE_ROOT), str(_PACKAGE_ROOT.parent)]
