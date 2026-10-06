from typing import Any


def calculate_quality_score(metrics: dict[str, Any]) -> float:
    consistency = float(metrics.get("consistency", 0.0))
    correctness = float(metrics.get("correctness", 0.0))
    confidence = float(metrics.get("confidence", 0.0))
    duplicate_rate = float(metrics.get("duplicate_rate", 0.0))
    contradiction_rate = float(metrics.get("contradiction_rate", 0.0))
    memory_stability = float(metrics.get("memory_stability", 0.0))

    score = (
        0.25 * consistency
        + 0.25 * correctness
        + 0.20 * confidence
        + 0.15 * (1.0 - duplicate_rate)
        + 0.10 * (1.0 - contradiction_rate)
        + 0.05 * memory_stability
    )
    return max(0.0, min(1.0, score))
