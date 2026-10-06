from typing import Any


class ModelBackend:
    def __init__(self, name: str = "base", config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}

    def is_available(self) -> bool:
        return True

    def generate(self, agent_name: str, prompt: str, system: str | None = None, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement generate().")
