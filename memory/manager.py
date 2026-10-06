import json
from pathlib import Path
from typing import Any

from memory.versioning import MemoryVersioning


class MemoryManager:
    def __init__(self, base_path: str = "memory_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.versioning = MemoryVersioning(base_path)

    def _file_path(self, key: str) -> Path:
        return self.base_path / f"{key}.json"

    def read_memory(self, key: str, default: Any | None = None) -> Any:
        file_path = self._file_path(key)
        if not file_path.exists():
            return default
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return default

    def write_memory(self, key: str, value: Any, agent: str = "system", cycle: int = 0, reason: str = "persisted", confidence: float = 0.0) -> Any:
        file_path = self._file_path(key)
        old_value = self.read_memory(key)
        if old_value != value:
            self.versioning.snapshot(key, old_value, value, agent=agent, cycle=cycle, reason=reason, confidence=confidence)
        file_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        return value

    def append_list(self, key: str, item: Any) -> Any:
        current = self.read_memory(key, [])
        if not isinstance(current, list):
            current = [current]
        current.append(item)
        return self.write_memory(key, current)

    def load_all(self) -> dict[str, Any]:
        items: dict[str, Any] = {}
        for path in sorted(self.base_path.glob("*.json")):
            try:
                items[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
        return items
