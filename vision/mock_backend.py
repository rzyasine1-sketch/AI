from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from ingestion.frame_extractor import FrameExtractor
from vision.base import VisionBackend


class MockVisionBackend(VisionBackend):
    name = "mock-vision"

    def analyze(self, image_path: Path) -> dict[str, Any]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Mock vision backend could not decode image: {image_path}")
        height, width = image.shape[:2]
        measured = FrameExtractor.analyze_frame(image)
        raw_observation = {
            "image_dimensions": {"width": width, "height": height},
            "measured_palette": measured["palette"],
            "mean_brightness": measured["brightness_mean"],
            "contrast_stddev": measured["contrast_stddev"],
            "edge_density": measured["edge_density"],
            "measured_composition": measured["composition"],
            "observation_source": "OpenCV pixel measurements; no semantic model used",
        }
        visual_analysis = {
            "characters": [],
            "character_appearance": None,
            "pose": None,
            "background": None,
            "composition": measured["composition"],
            "framing": measured["framing"],
            "lighting": measured["lighting"],
            "color_palette": measured["palette"],
            "drawing_rendering_style": None,
            "textures": [],
            "visual_effects": [],
            "on_screen_text": [],
            "scene_type": None,
            "camera_framing": None,
            "style_confidence": 0.0,
            "analysis_status": "mock_no_vlm",
            "unavailable_semantics": [
                "characters",
                "character appearance",
                "pose",
                "background semantics",
                "rendering style",
                "textures",
                "visual effects",
                "on-screen text",
                "scene type",
                "camera language",
            ],
        }
        return {"raw_observation": raw_observation, "visual_analysis": visual_analysis}
