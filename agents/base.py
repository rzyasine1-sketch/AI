from __future__ import annotations

from typing import Any


class Agent:
    def __init__(self, name: str, backend):
        self.name = name
        self.backend = backend

    def generate(self, prompt: str, cycle: int = 1, **kwargs) -> dict[str, Any]:
        payload = self.backend.generate(agent_name=self.name, prompt=prompt, cycle=cycle, **kwargs)
        if not isinstance(payload, dict):
            raise TypeError(f"Agent {self.name} returned non-dict output.")
        return payload

    def run(self, cycle: int = 1, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("Each agent must implement run().")
