from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent

filtered = [p for p in sys.path if p not in {"", str(ROOT), str(PARENT)}]
filtered.insert(0, str(PARENT))
filtered.insert(1, str(ROOT))
sys.path[:] = filtered
