from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_event(path: str | Path, event: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as handle:
        handle.write(json.dumps(event, default=str, sort_keys=True) + "\n")
