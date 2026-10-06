import json
from pathlib import Path
from typing import Any


def last_successful_checkpoint(checkpoints_dir: str = "checkpoints") -> dict[str, Any] | None:
    root = Path(checkpoints_dir)
    if not root.exists():
        return None
    checkpoint_paths = sorted(root.glob("checkpoint_*.json"))
    if not checkpoint_paths:
        return None
    last_path = checkpoint_paths[-1]
    try:
        return json.loads(last_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_checkpoint(checkpoints_dir: str, state: dict[str, Any], name: str = "checkpoint") -> Path:
    root = Path(checkpoints_dir)
    root.mkdir(parents=True, exist_ok=True)
    file_path = root / f"{name}.json"
    file_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return file_path
