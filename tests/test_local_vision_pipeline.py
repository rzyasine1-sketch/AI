import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.local_vision import run_local_vision_pipeline


def _create_frame(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((48, 80, 3), dtype=np.uint8)
    image[:, :40] = (15, 80, 180)
    image[:, 40:] = (190, 195, 200)
    assert cv2.imwrite(str(path), image)
    return path


def test_mock_vision_pipeline_preserves_evidence_and_inferences(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame_path = _create_frame(tmp_path / "data" / "frames" / "frame_000000.jpg")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("models: {}\nmemory:\n  store_path: memory_store\n", encoding="utf-8")

    report = run_local_vision_pipeline([frame_path], mock_mode=True, config_path=config_path)

    assert report["status"] == "mock_complete"
    assert report["mock_used"] is True
    assert report["frame_count"] == report["dataset_record_count"] == 1
    dataset_path = Path(report["dataset_path"])
    sample = json.loads(dataset_path.read_text().strip())
    image_path = tmp_path / "data" / sample["image"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None and image.size > 0
    assert sample["analysis_backend"] == "mock-vision"
    assert sample["raw_observation"]["image_dimensions"] == {"width": 80, "height": 48}
    assert sample["visual_analysis"]["analysis_status"] == "mock_no_vlm"
    assert sample["validated_style_rules"]
    assert "validated_style_rules" not in sample["raw_observation"]
    memory = json.loads(Path(report["memory_path"]).read_text())
    assert memory["validated_style_rules"] == sample["validated_style_rules"]


def test_real_vision_pipeline_blocks_without_vlm_and_never_uses_mock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame_path = _create_frame(tmp_path / "data" / "frames" / "frame_000000.jpg")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "vision:\n  base_url: http://127.0.0.1:1\n  model_name: qwen3-vl:2b\nmodels: {}\n",
        encoding="utf-8",
    )

    report = run_local_vision_pipeline([frame_path], mock_mode=False, config_path=config_path)

    assert report["status"] == "blocked"
    assert report["mock_used"] is False
    assert "unavailable" in report["blocked_by"].lower()
    assert report["dataset_record_count"] == 0
    assert not Path(report["dataset_path"]).exists()