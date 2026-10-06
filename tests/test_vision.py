import base64
import json
from pathlib import Path

import cv2
import numpy as np

from vision.base import VisionAnalyzer
from vision.local_backend import LocalVisionBackend
from vision.mock_backend import MockVisionBackend


def _write_test_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((48, 80, 3), dtype=np.uint8)
    image[:, :40] = (30, 120, 210)
    image[:, 40:] = (180, 190, 200)
    assert cv2.imwrite(str(path), image)
    return path


def test_mock_analyzer_reads_real_image_and_returns_structured_fields(tmp_path):
    image_path = _write_test_image(tmp_path / "frame.jpg")

    result = VisionAnalyzer(MockVisionBackend()).analyze(image_path)

    assert result["analysis_backend"] == "mock-vision"
    assert result["raw_observation"]["image_dimensions"] == {"width": 80, "height": 48}
    analysis = result["visual_analysis"]
    assert analysis["color_palette"]
    assert analysis["composition"]
    assert "characters" in analysis
    assert analysis["analysis_status"] == "mock_no_vlm"


def test_local_backend_sends_actual_image_and_requires_vlm_capability(tmp_path, monkeypatch):
    image_path = _write_test_image(tmp_path / "frame.png")
    response_payload = {
        "raw_observation": {"visible_items": ["blue and gray regions"]},
        "visual_analysis": {
            "characters": [],
            "character_appearance": "none detected",
            "pose": "not applicable",
            "background": "flat color regions",
            "composition": "two vertical color fields",
            "framing": "landscape",
            "lighting": "even",
            "color_palette": ["blue", "gray"],
            "drawing_rendering_style": "digital graphic",
            "textures": [],
            "visual_effects": [],
            "on_screen_text": [],
            "scene_type": "graphic",
            "camera_framing": "static wide view",
            "style_confidence": 0.82,
        },
    }
    requests_seen = []

    class Response:
        def __init__(self, value):
            self.value = json.dumps(value).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.value

    def fake_urlopen(request, timeout):
        requests_seen.append(request)
        if request.full_url.endswith("/api/tags"):
            return Response({"models": [{"name": "qwen3-vl:2b"}]})
        payload = json.loads(request.data.decode())
        if request.full_url.endswith("/api/show"):
            return Response({"capabilities": ["vision", "completion"], "details": {"families": ["qwen3vl"]}})
        if request.full_url.endswith("/api/generate"):
            assert payload["model"] == "qwen3-vl:2b"
            assert payload["images"] == [base64.b64encode(cv2.imencode(".jpg", cv2.imread(str(image_path)))[1]).decode()]
            assert "raw_observation" in payload["prompt"]
            return Response({"response": json.dumps(response_payload)})
        raise AssertionError(f"Unexpected local model request: {request.full_url}")

    monkeypatch.setattr("vision.local_backend.urlopen", fake_urlopen)
    backend = LocalVisionBackend({"base_url": "http://local-model", "model_name": "qwen3-vl:2b"})

    result = VisionAnalyzer(backend).analyze(image_path)

    assert result["analysis_backend"] == "ollama-vision:qwen3-vl:2b"
    assert result["visual_analysis"]["style_confidence"] == 0.82
    assert any(request.full_url.endswith("/api/generate") for request in requests_seen)


def test_local_backend_blocks_without_runtime():
    backend = LocalVisionBackend({"base_url": "http://127.0.0.1:1", "model_name": "qwen3-vl:2b"})

    assert backend.is_available() is False
    assert "unavailable" in backend.availability_error.lower()


def test_local_backend_rejects_installed_text_only_model(monkeypatch):
    class Response:
        def __init__(self, value):
            self.value = json.dumps(value).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.value

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/api/tags"):
            return Response({"models": [{"name": "qwen3-vl:2b"}]})
        return Response({"capabilities": ["completion"], "details": {"families": ["qwen3"]}})

    monkeypatch.setattr("vision.local_backend.urlopen", fake_urlopen)
    backend = LocalVisionBackend({"base_url": "http://local-model", "model_name": "qwen3-vl:2b"})

    assert backend.is_available() is False
    assert "does not declare Vision capability" in backend.availability_error