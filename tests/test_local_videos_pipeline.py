import json
from pathlib import Path

import cv2
import numpy as np

from ingestion.local_videos import discover_local_mp4s
from pipeline.local_videos import run_local_videos_pipeline
from training.dataset import TrainingDataset


def _write_mp4(path: Path, color_shift: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (96, 64))
    assert writer.isOpened()
    for index in range(10):
        frame = np.full((64, 96, 3), (index * 12 + color_shift, 90, 170), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None
    return path


def test_discover_local_mp4s_recurses_and_ignores_other_files(tmp_path):
    first = _write_mp4(tmp_path / "first.mp4")
    nested = _write_mp4(tmp_path / "subdir" / "second.MP4", 8)
    (tmp_path / "ignore.avi").write_bytes(b"not an mp4")
    (tmp_path / "notes.txt").write_text("ignore")

    assert discover_local_mp4s(tmp_path) == [first.resolve(), nested.resolve()]


def test_local_video_pipeline_uses_real_files_checkpoints_and_resume(tmp_path, monkeypatch):
    from pipeline import local_videos

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "models: {}\nvision: {model_name: test-vlm}\nmemory:\n  store_path: memory_store\n",
        encoding="utf-8",
    )
    input_dir = tmp_path / "input_videos"
    first = _write_mp4(input_dir / "first.mp4")
    second = _write_mp4(input_dir / "nested" / "second.mp4", 6)
    original_sizes = {path: path.stat().st_size for path in (first, second)}
    vision_reads = []
    agent_order = []

    class AvailableVisionBackend:
        name = "test-real-vlm"
        availability_error = "available"
        model_name = "test-vlm"

        def __init__(self, config):
            pass

        def is_available(self):
            return True

        def analyze(self, image_path):
            image = cv2.imread(str(image_path))
            assert image is not None and image.size > 0
            vision_reads.append(Path(image_path))
            return {
                "raw_observation": {"direct_visual_evidence": ["visible color regions"]},
                "visual_analysis": {
                    "characters": [],
                    "character_appearance": None,
                    "pose": None,
                    "background": "simple colored background",
                    "composition": "balanced split composition",
                    "framing": "landscape",
                    "lighting": "even",
                    "color_palette": ["blue", "gray"],
                    "drawing_rendering_style": "flat graphic",
                    "textures": [],
                    "visual_effects": [],
                    "on_screen_text": [],
                    "scene_type": "graphic",
                    "camera_framing": "wide static frame",
                    "style_confidence": 0.86,
                },
            }

        def unload(self):
            pass

    class Agent:
        def __init__(self, name):
            self.name = name

        def run(self, cycle=1, **kwargs):
            agent_order.append((cycle, self.name))
            if self.name == "deepseek":
                return {"cycle": cycle, "observations": [], "style_patterns": [], "video_patterns": [], "image_patterns": [], "confidence": 0.9, "new_rules": ["consistent split palette"], "unknowns": [], "next_tasks": []}
            if self.name == "qwen":
                return {"cycle": cycle, "validated": [], "errors": [], "contradictions": [], "weak_claims": [], "corrections": [], "confidence": 0.9}
            return {"cycle": cycle, "kept_knowledge": [], "updated_knowledge": [], "rejected_knowledge": [], "reasons": [], "next_cycle_instructions": [], "memory_changes": [], "validated_style_rules": ["consistent split palette"], "confidence": 0.9}

    monkeypatch.setattr(local_videos, "LocalVisionBackend", AvailableVisionBackend)
    monkeypatch.setattr(
        local_videos,
        "_prepare_real_agents",
        lambda config: (
            {name: Agent(name) for name in ("deepseek", "qwen", "llama")},
            {name: {"backend": "test-real", "available": True, "detail": "available"} for name in ("deepseek", "qwen", "llama")},
        ),
    )

    report = run_local_videos_pipeline(input_dir, frames_per_video=2, cycles=2, resume=True)

    assert report["status"] == "complete"
    assert report["source_file_count"] == 2
    assert report["processed_video_count"] == 2
    assert report["extracted_frame_count"] == report["dataset_record_count"] == 4
    assert len(vision_reads) == 4
    assert [name for _, name in agent_order] == ["deepseek", "qwen", "llama"] * 2
    assert {path: path.stat().st_size for path in (first, second)} == original_sizes
    checkpoints = Path(report["checkpoints_dir"])
    assert (checkpoints / "inventory.json").is_file()
    assert (checkpoints / "videos").is_dir()
    assert (checkpoints / "agent_cycles" / "cycle_002.json").is_file()
    assert (checkpoints / "memory.json").is_file()
    assert (checkpoints / "dataset.json").is_file()
    records = TrainingDataset(storage_path="training/local_videos", image_root="data").load_visual_records()
    assert len(records) == 4
    assert all(record["raw_observation"] and record["visual_analysis"] for record in records)
    assert all(record["validated_style_rules"] == ["consistent split palette"] for record in records)

    resumed = run_local_videos_pipeline(input_dir, frames_per_video=2, cycles=2, resume=True)
    assert resumed["status"] == "complete"
    assert len(vision_reads) == 4
    assert len(agent_order) == 6
    assert resumed["dataset_record_count"] == 4
    assert {path: path.stat().st_size for path in (first, second)} == original_sizes


def test_local_video_real_mode_blocks_without_vision_or_agents(tmp_path, monkeypatch):
    from pipeline import local_videos

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("models: {}\nvision: {base_url: http://127.0.0.1:1}\n", encoding="utf-8")
    input_dir = tmp_path / "input"
    _write_mp4(input_dir / "source.mp4")

    report = local_videos.run_local_videos_pipeline(input_dir, frames_per_video=1, cycles=1)

    assert report["status"] == "blocked"
    assert report["mock_used"] is False
    assert report["dataset_record_count"] == 0
    assert not Path("training/local_videos/golden_visual_dataset.jsonl").exists()


def test_local_videos_cli_ignores_channel_environment_variable(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "models: {}\nvision: {base_url: http://127.0.0.1:1}\nmemory:\n  store_path: memory_store\n",
        encoding="utf-8",
    )
    input_dir = tmp_path / "input"
    _write_mp4(input_dir / "source.mp4")
    monkeypatch.setenv("CHANNEL_URL", "https://youtube.com/@unrelated/videos")
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--local-videos", str(input_dir), "--frames-per-video", "1", "--cycles", "1"],
    )

    from run import main

    result = main()

    report = json.loads((tmp_path / "data" / "local_videos_run_report.json").read_text())
    assert result == 2
    assert report["status"] == "blocked"
    assert report["mock_used"] is False
    assert report["source_file_count"] == 1
    assert "Vision:" in capsys.readouterr().out
