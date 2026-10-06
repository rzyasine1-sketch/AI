from agents.base import Agent
from memory.schemas import repair_payload, validate_agent_payload


class LlamaAgent(Agent):
    def __init__(self, backend):
        super().__init__("llama", backend)

    def run(self, cycle: int = 1, deepseek_output=None, qwen_output=None, memory=None, **kwargs):
        prompt = (
            "Review the evidence-backed visual-style analysis and critique. Keep, update, or reject persistent "
            "rules for visual style, palette, composition, character appearance, backgrounds, lighting, framing, "
            "camera language, recurring visual elements, visual consistency, and scene structure. Do not add audio rules. "
            "Return validated_style_rules only for rules that survive Qwen's critique, and keep them separate from "
            "the direct raw_observation records. Never store inferred rules as observations. "
            "Return JSON fields: cycle (integer), kept_knowledge (array), updated_knowledge (array), "
            "rejected_knowledge (array), reasons (array), next_cycle_instructions (array), memory_changes (array), "
            "confidence (number from 0 to 1), and visual rule arrays when supported."
        )
        payload = self.generate(prompt, cycle=cycle, deepseek_output=deepseek_output, qwen_output=qwen_output, memory=memory, **kwargs)
        payload = repair_payload("llama", payload)
        if not validate_agent_payload("llama", payload):
            raise ValueError("Llama produced invalid output for schema validation")
        return payload
