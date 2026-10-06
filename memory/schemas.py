from copy import deepcopy

AGENT_SCHEMAS = {
    "deepseek": {
        "type": "object",
        "required": [
            "cycle",
            "observations",
            "style_patterns",
            "video_patterns",
            "image_patterns",
            "confidence",
            "new_rules",
            "unknowns",
            "next_tasks",
        ],
        "properties": {
            "cycle": {"type": "integer"},
            "observations": {"type": "array"},
            "style_patterns": {"type": "array"},
            "video_patterns": {"type": "array"},
            "image_patterns": {"type": "array"},
            "confidence": {"type": "number"},
            "new_rules": {"type": "array"},
            "unknowns": {"type": "array"},
            "next_tasks": {"type": "array"},
        },
        "additionalProperties": True,
    },
    "qwen": {
        "type": "object",
        "required": [
            "cycle",
            "validated",
            "errors",
            "contradictions",
            "weak_claims",
            "corrections",
            "confidence",
        ],
        "properties": {
            "cycle": {"type": "integer"},
            "validated": {"type": "array"},
            "errors": {"type": "array"},
            "contradictions": {"type": "array"},
            "weak_claims": {"type": "array"},
            "corrections": {"type": "array"},
            "confidence": {"type": "number"},
        },
        "additionalProperties": True,
    },
    "llama": {
        "type": "object",
        "required": [
            "cycle",
            "kept_knowledge",
            "updated_knowledge",
            "rejected_knowledge",
            "reasons",
            "next_cycle_instructions",
            "memory_changes",
        ],
        "properties": {
            "cycle": {"type": "integer"},
            "kept_knowledge": {"type": "array"},
            "updated_knowledge": {"type": "array"},
            "rejected_knowledge": {"type": "array"},
            "reasons": {"type": "array"},
            "next_cycle_instructions": {"type": "array"},
            "memory_changes": {"type": "array"},
        },
        "additionalProperties": True,
    },
}


def validate_agent_payload(agent_name: str, payload: dict) -> bool:
    schema = AGENT_SCHEMAS.get(agent_name)
    if schema is None:
        raise ValueError(f"Unknown agent schema: {agent_name}")

    if not isinstance(payload, dict):
        return False

    for key in schema["required"]:
        if key not in payload:
            return False

    if payload.get("cycle") is not None and not isinstance(payload["cycle"], int):
        return False

    for field_name in ["confidence"]:
        if field_name in payload and not isinstance(payload[field_name], (int, float)):
            return False

    return True


def repair_payload(agent_name: str, payload: dict) -> dict:
    repaired = deepcopy(payload)
    if not isinstance(repaired, dict):
        raise TypeError("Agent payload must be a dict.")

    if "cycle" not in repaired:
        repaired["cycle"] = 0
    if "confidence" not in repaired:
        repaired["confidence"] = 0.0

    for key in ["observations", "style_patterns", "video_patterns", "image_patterns", "new_rules", "unknowns", "next_tasks"]:
        if key in repaired and repaired[key] is None:
            repaired[key] = []
    for key in ["validated", "errors", "contradictions", "weak_claims", "corrections"]:
        if key in repaired and repaired[key] is None:
            repaired[key] = []
    for key in ["kept_knowledge", "updated_knowledge", "rejected_knowledge", "reasons", "next_cycle_instructions", "memory_changes"]:
        if key in repaired and repaired[key] is None:
            repaired[key] = []

    return repaired
