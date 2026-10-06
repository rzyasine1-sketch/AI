from typing import Any

from models.base import ModelBackend


class MockModelBackend(ModelBackend):
    def __init__(self, name: str = "mock-model", config: dict[str, Any] | None = None):
        super().__init__(name=name, config=config)

    def is_available(self) -> bool:
        return True

    def generate(self, agent_name: str, prompt: str, system: str | None = None, **kwargs) -> Any:
        if agent_name == "deepseek":
            return {
                "cycle": kwargs.get("cycle", 1),
                "observations": ["consistent pacing", "clear narrative arc", "strong visual consistency"],
                "style_patterns": ["educational voice", "tight pacing", "recurring calls to action"],
                "video_patterns": ["intro hook", "short cuts", "summary end card"],
                "image_patterns": ["high contrast thumbnails", "close-up faces"],
                "confidence": 0.88,
                "new_rules": ["keep introduction under 8 seconds", "maintain narrative coherence"],
                "unknowns": ["exact audience segment"],
                "next_tasks": ["validate thumbnail strategy", "audit retention spikes"],
            }
        if agent_name == "qwen":
            return {
                "cycle": kwargs.get("cycle", 1),
                "validated": ["intro hook remains consistent", "summary pattern is useful"],
                "errors": ["audience segment is still speculative"],
                "contradictions": [],
                "weak_claims": ["retention spikes are not yet proven"],
                "corrections": ["tighten claim around audience segment"],
                "confidence": 0.81,
            }
        if agent_name == "llama":
            return {
                "cycle": kwargs.get("cycle", 1),
                "kept_knowledge": ["strong narrative arc", "intro hook matters"],
                "updated_knowledge": ["audience claims require proof"],
                "rejected_knowledge": ["thumbnail strategy without evidence"],
                "reasons": ["unverified claims require evidence"],
                "next_cycle_instructions": ["focus on retention behavior and evidence-based updates"],
                "memory_changes": [{"key": "visual_style_rules", "type": "update", "agent": "llama"}],
                "visual_style_rules": ["muted palette", "soft contrast", "character-first framing"],
                "validated_style_rules": ["muted palette", "soft contrast", "character-first framing"],
                "character_rules": ["clean silhouette", "simple expressive features"],
                "composition_rules": ["left-third framing for character emphasis", "balanced negative space"],
                "color_rules": ["muted blue and warm beige dominate frames"],
                "lighting_rules": ["soft diffused lighting", "limited harsh shadows"],
                "background_rules": ["simple uncluttered backgrounds"],
                "camera_rules": ["close-up and medium close-up compositions"],
                "motion_visual_rules": ["small transitions between frames, preserved visual continuity"],
                "confidence": 0.91,
            }
        return {"status": "ok", "response": prompt}
