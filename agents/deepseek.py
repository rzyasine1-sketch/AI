from agents.base import Agent
from memory.schemas import repair_payload, validate_agent_payload


class DeepSeekAgent(Agent):
    def __init__(self, backend):
        super().__init__("deepseek", backend)

    def run(self, cycle: int = 1, source_data=None, memory=None, **kwargs):
        prompt = (
            "Analyze channel visual STYLE from the supplied frames and pixel evidence, not only what happens. "
            "Extract visual_style, color palette, composition, character appearance (only if visually supported), "
            "backgrounds, lighting, framing, camera language, recurring visual elements, visual consistency, and "
            "scene structure. Keep direct image facts in observations; put cross-frame style inferences only in "
            "style_patterns and new_rules. Never rewrite an inference as raw evidence. Return JSON fields: cycle "
            "(integer), observations (array), style_patterns (array), video_patterns (array), image_patterns (array), "
            "confidence (number from 0 to 1), new_rules (array), unknowns (array), next_tasks (array)."
        )
        payload = self.generate(prompt, cycle=cycle, source_data=source_data, memory=memory, **kwargs)
        payload = repair_payload("deepseek", payload)
        if not validate_agent_payload("deepseek", payload):
            raise ValueError("DeepSeek produced invalid output for schema validation")
        return payload
