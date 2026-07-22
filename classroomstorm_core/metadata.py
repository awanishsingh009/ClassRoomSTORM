from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_metadata(path: str | Path, data: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(dict(data), handle, indent=2, sort_keys=True)


def read_metadata(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
