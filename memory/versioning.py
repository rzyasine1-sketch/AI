import json
from pathlib import Path
from typing import Any


class MemoryVersioning:
    def __init__(self, base_path: str = "memory_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _history_file(self, key: str) -> Path:
        history_dir = self.base_path / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir / f"{key}.json"

    def snapshot(self, key: str, old: Any, new: Any, agent: str, cycle: int, reason: str, confidence: float = 0.0) -> dict:
        from datetime import datetime, timezone

        entry = {
            "type": "update",
            "key": key,
            "old": old,
            "new": new,
            "reason": reason,
            "agent": agent,
            "cycle": cycle,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        history = self.load_history(key)
        history.append(entry)
        self._history_file(key).write_text(json.dumps(history, indent=2), encoding="utf-8")
        return entry

    def load_history(self, key: str) -> list[dict]:
        file_path = self._history_file(key)
        if not file_path.exists():
            return []
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            return []
        return []

    def restore(self, key: str, version_index: int = -1) -> Any:
        history = self.load_history(key)
        if not history:
            return None
        item = history[version_index]
        return item.get("old")
