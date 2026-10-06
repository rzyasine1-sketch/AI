from agents.base import Agent
from memory.schemas import repair_payload, validate_agent_payload


class QwenAgent(Agent):
    def __init__(self, backend):
        super().__init__("qwen", backend)

    def run(self, cycle: int = 1, deepseek_output=None, **kwargs):
        prompt = (
            "Critique each proposed visual-style inference against the separately supplied raw observations and "
            "visual analyses. Keep raw evidence immutable. Flag unsupported "
            "claims about characters, backgrounds, camera language, and scene meaning; check palette, composition, "
            "lighting, framing, recurring elements, and consistency for contradictions. Return JSON fields: cycle "
            "(integer), validated (array), errors (array), contradictions (array), weak_claims (array), "
            "corrections (array), confidence (number from 0 to 1)."
        )
        payload = self.generate(prompt, cycle=cycle, deepseek_output=deepseek_output, **kwargs)
        payload = repair_payload("qwen", payload)
        if not validate_agent_payload("qwen", payload):
            raise ValueError("Qwen produced invalid output for schema validation")
        return payload
