from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2

from vision.base import VisionBackend


class LocalVisionBackend(VisionBackend):
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.base_url = str(self.config.get("base_url") or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434")
        if "://" not in self.base_url:
            self.base_url = f"http://{self.base_url}"
        self.base_url = self.base_url.rstrip("/")
        self.model_name = str(self.config.get("model_name") or "qwen3-vl:2b")
        self.name = f"ollama-vision:{self.model_name}"
        self.timeout = float(self.config.get("timeout", 300))
        self.keep_alive = self.config.get("keep_alive", "5m")
        self.availability_error = "Vision runtime has not been checked"

    @staticmethod
    def _request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        if payload is None:
            request = Request(url, method="GET")
        else:
            request = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise TypeError("Vision runtime response must be a JSON object")
        return result

    def is_available(self) -> bool:
        try:
            tags = self._request_json(f"{self.base_url}/api/tags", None, timeout=3)
            models = tags.get("models", [])
            installed_names = {
                str(value)
                for model in models
                if isinstance(model, dict)
                for value in (model.get("name"), model.get("model"))
                if value
            }
            if self.model_name not in installed_names:
                self.availability_error = f"Vision model {self.model_name!r} is not installed in Ollama at {self.base_url}"
                return False

            details = self._request_json(
                f"{self.base_url}/api/show",
                {"model": self.model_name},
                timeout=3,
            )
            capabilities = {str(value).lower() for value in details.get("capabilities") or []}
            model_details = details.get("details", {})
            families = (model_details.get("families") or []) if isinstance(model_details, dict) else []
            model_info = details.get("model_info", {})
            architecture = " ".join(str(value) for value in model_info.values()) if isinstance(model_info, dict) else ""
            vision_markers = " ".join([*(str(value) for value in families), architecture]).lower()
            known_vision_architectures = ("vision", "-vl", "_vl", "qwen3vl", "llava", "moondream", "minicpmv", "bakllava")
            supports_vision = "vision" in capabilities or any(marker in vision_markers for marker in known_vision_architectures)
            if not supports_vision:
                self.availability_error = f"Installed Ollama model {self.model_name!r} does not declare Vision capability"
                return False
            self.availability_error = "available"
            return True
        except (OSError, URLError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self.availability_error = f"Local vision runtime unavailable at {self.base_url}: {exc}"
            return False

    def analyze(self, image_path: Path) -> dict[str, Any]:
        if not self.is_available():
            raise RuntimeError(self.availability_error)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Vision model input is not a readable image: {image_path}")
        encoded_ok, encoded_image = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not encoded_ok:
            raise RuntimeError(f"Could not encode image for vision model: {image_path}")
        image_base64 = base64.b64encode(encoded_image.tobytes()).decode("ascii")
        prompt = """Analyze the attached image itself. Return a single JSON object with exactly these top-level keys:
{
  "raw_observation": {
    "visible_entities": [], "visible_attributes": [], "spatial_relationships": [],
    "visible_text": [], "direct_visual_evidence": []
  },
  "visual_analysis": {
    "characters": [], "character_appearance": null, "pose": null, "background": null,
    "composition": null, "framing": null, "lighting": null, "color_palette": [],
    "drawing_rendering_style": null, "textures": [], "visual_effects": [],
    "on_screen_text": [], "scene_type": null, "camera_framing": null, "style_confidence": 0.0
  }
}
Keep raw_observation to directly visible evidence only. Put stylistic interpretation only in visual_analysis. Use null or [] when uncertain; do not invent characters, text, camera movement, or scene meaning. style_confidence must be between 0 and 1."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_base64],
            "format": "json",
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.1},
        }
        try:
            response = self._request_json(f"{self.base_url}/api/generate", payload, timeout=self.timeout)
            output = response.get("response")
            if not isinstance(output, str):
                raise TypeError("Vision model returned no JSON response text")
            result = json.loads(output)
            if not isinstance(result, dict):
                raise TypeError("Vision model output must be a JSON object")
            return result
        except (OSError, URLError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Vision model {self.model_name!r} failed to analyze {image_path}: {exc}") from exc

    def unload(self) -> None:
        self._request_json(
            f"{self.base_url}/api/generate",
            {"model": self.model_name, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=30,
        )
