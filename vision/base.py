from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2

VISUAL_ANALYSIS_FIELDS = (
    "characters",
    "character_appearance",
    "pose",
    "background",
    "composition",
    "framing",
    "lighting",
    "color_palette",
    "drawing_rendering_style",
    "textures",
    "visual_effects",
    "on_screen_text",
    "scene_type",
    "camera_framing",
    "style_confidence",
)


class VisionBackend(ABC):
    name = "vision-backend"

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def analyze(self, image_path: Path) -> dict[str, Any]:
        raise NotImplementedError


class VisionAnalyzer:
    def __init__(self, backend: VisionBackend):
        self.backend = backend

    def is_available(self) -> bool:
        return self.backend.is_available()

    @property
    def availability_error(self) -> str:
        return str(getattr(self.backend, "availability_error", ""))

    def analyze(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Image is missing or empty: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"OpenCV could not decode image: {path}")

        result = self.backend.analyze(path)
        if not isinstance(result, dict):
            raise TypeError("Vision backend must return an object")
        raw_observation = result.get("raw_observation")
        visual_analysis = result.get("visual_analysis")
        if not isinstance(raw_observation, dict):
            raise TypeError("Vision backend must return raw_observation as an object")
        if not isinstance(visual_analysis, dict):
            raise TypeError("Vision backend must return visual_analysis as an object")
        missing = [field for field in VISUAL_ANALYSIS_FIELDS if field not in visual_analysis]
        if missing:
            raise ValueError(f"Vision analysis is missing fields: {', '.join(missing)}")
        confidence = visual_analysis["style_confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("style_confidence must be a number between 0 and 1")
        return {
            **result,
            "image_path": str(path),
            "analysis_backend": getattr(self.backend, "name", type(self.backend).__name__),
        }
