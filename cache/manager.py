import hashlib
import json
from pathlib import Path
from typing import Any


class CacheManager:
    def __init__(self, base_path: str = "cache", enabled: bool = True):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    @staticmethod
    def hash_payload(payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        return self.base_path / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        file_path = self._path_for(key)
        if not file_path.exists():
            return None
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: Any):
        if not self.enabled:
            return value
        file_path = self._path_for(key)
        file_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        return value

    def create_hash_key(self, prompt: str, agent_name: str, config: dict[str, Any] | None = None) -> str:
        payload = {"prompt": prompt, "agent": agent_name, "config": config or {}}
        return self.hash_payload(payload)
