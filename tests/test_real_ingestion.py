import json
from pathlib import Path

import cv2
import numpy as np

from ingestion.frame_extractor import FrameExtractor
from ingestion.youtube import YouTubeIngestion
from training.dataset import TrainingDataset


def test_smart_sampling_covers_timeline_and_high_performance():
    inventory = [
        {"id": f"video-{index}", "playlist_position": index, "view_count": index * 100, "duration": 60}
        for index in range(12)
    ]

    selected = YouTubeIngestion.select_videos(inventory, target_videos=4)

    assert len(selected) == 4
    assert {item["selection_reason"] for item in selected} >= {
        "recent",
        "middle",
        "old",
        "high_performance",
    }


def test_smart_sampling_includes_distinct_content_types_before_popularity_fill():
    titles = ["Recent launch trailer", "How to photograph an eclipse", "Crew interview", "Mission recap", "Video"]
    inventory = [
        {"id": f"video-{index}", "title": title, "playlist_position": index, "view_count": 1000 - index, "duration": 60}
        for index, title in enumerate(titles)
    ]

    selected = YouTubeIngestion.select_videos(inventory, target_videos=5)

    reasons = {item["selection_reason"] for item in selected}
    assert "content_type:educational" in reasons
    assert "Crew interview" in {item["title"] for item in selected}


def test_extract_video_frames_writes_openable_images(tmp_path):
    video_path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (96, 64))
    assert writer.isOpened()
    for index in range(10):
        frame = np.full((64, 96, 3), (index * 20, 80, 160), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    records = FrameExtractor().extract_video_frames(video_path, tmp_path / "frames", frame_count=4)

    assert len(records) == 4
    for record in records:
        image_path = Path(record["image_path"])
        image = cv2.imread(str(image_path))
        assert image_path.is_file()
        assert image_path.stat().st_size > 0
        assert image is not None and image.size > 0
        assert record["visual_analysis"]["palette"]


def test_video_inventory_and_selection_are_persisted(tmp_path, monkeypatch):
    ingestion = YouTubeIngestion(data_dir=tmp_path)
    monkeypatch.setattr(
        ingestion,
        "_extract_channel_metadata",
        lambda url, max_entries=None: {
            "id": "channel-id",
            "title": "Channel",
            "entries": [
                {"id": f"video-{index}", "title": f"Video {index}", "view_count": index, "duration": 60}
                for index in range(8)
            ],
        },
    )

    inventory = ingestion.inventory("https://www.youtube.com/@channel/videos")
    selected = ingestion.select_and_save(inventory, target_videos=3)

    saved_inventory = json.loads((tmp_path / "channel_inventory.json").read_text())
    saved_selection = json.loads((tmp_path / "selected_videos.json").read_text())
    assert saved_inventory["channel_id"] == "channel-id"
    assert len(saved_inventory["videos"]) == 8
    assert len(selected) == len(saved_selection["videos"]) == 3


def test_pixel_analysis_dataset_requires_a_real_image(tmp_path):
    image_path = tmp_path / "videos" / "video-id" / "frames" / "frame_000001.jpg"
    image_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image_path), np.full((32, 32, 3), 128, dtype=np.uint8))
    dataset = TrainingDataset(storage_path=str(tmp_path / "training"), image_root=str(tmp_path))
    sample = {
        "image": "videos/video-id/frames/frame_000001.jpg",
        "description": "Pixel-derived palette and composition measurements.",
        "visual_style": "muted soft-contrast low-edge-detail",
        "composition": "edge detail concentrated in center-middle third",
        "lighting": "mid-range",
        "colors": ["#808080"],
        "characters": [],
        "scene_type": "unclassified",
        "source_video": "video-id",
        "source_timestamp": 1.0,
        "timestamp": 1.0,
        "cycle": 1,
        "validated_by": ["opencv-pixel-analysis"],
        "analysis_backend": "opencv-pixel-analysis",
        "confidence": 1.0,
    }

    assert dataset.add_visual_sample(sample) is True
    assert dataset.add_visual_sample({**sample, "image": "videos/missing.jpg"}) is False


def test_channel_run_persists_download_blocker_without_mock_fallback(tmp_path, monkeypatch):
    from pipeline import real_channel

    class AvailableVisionBackend:
        name = "test-vlm"
        availability_error = "available"

        def __init__(self, config):
            pass

        def is_available(self):
            return True

        def unload(self):
            pass

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("models: {}\nmemory: {}\n", encoding="utf-8")
    video = {"id": "blocked-video", "title": "Video", "selection_reason": "recent"}
    inventory = {"channel_id": "channel-id", "videos": [video]}
    monkeypatch.setattr(real_channel.YouTubeIngestion, "inventory", lambda self, url, max_entries: inventory)
    monkeypatch.setattr(real_channel.YouTubeIngestion, "select_and_save", lambda self, inv, target_videos: [video])
    monkeypatch.setattr(real_channel, "LocalVisionBackend", AvailableVisionBackend)
    monkeypatch.setattr(
        real_channel.YouTubeIngestion,
        "download_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("authentication required")),
    )
    monkeypatch.setattr(
        real_channel,
        "_prepare_real_agents",
        lambda config: (
            {name: object() for name in ("deepseek", "qwen", "llama")},
            {name: {"available": True, "detail": "available"} for name in ("deepseek", "qwen", "llama")},
        ),
    )

    report = real_channel.run_channel_pipeline("https://youtube.com/@channel/videos", target_videos=1)

    assert report["status"] == "blocked"
    assert report["downloaded_and_analyzed_count"] == 0
    assert "authentication required" in report["blocked_by"][0]["error"]
    assert (tmp_path / "data" / "real_run_report.json").is_file()
    assert not (tmp_path / "training" / "real_visual" / "golden_visual_dataset.jsonl").exists()


def test_channel_pipeline_uses_vlm_checkpoints_and_resumes_without_redownload(tmp_path, monkeypatch):
    import shutil

    from pipeline import real_channel

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "models: {}\nvision: {model_name: test-vlm}\nmemory:\n  store_path: memory_store\n",
        encoding="utf-8",
    )
    source_video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source_video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (96, 64))
    assert writer.isOpened()
    for index in range(10):
        writer.write(np.full((64, 96, 3), (index * 15, 90, 170), dtype=np.uint8))
    writer.release()

    video = {"id": "video-id", "title": "Video", "selection_reason": "recent"}
    inventory = {"channel_id": "channel-id", "videos": [video]}
    download_count = 0
    vision_count = 0
    agent_order = []
    temporary_downloads = []

    class AvailableVisionBackend:
        name = "test-vlm"
        availability_error = "available"

        def __init__(self, config):
            pass

        def is_available(self):
            return True

        def analyze(self, image_path):
            nonlocal vision_count
            image = cv2.imread(str(image_path))
            assert image is not None and image.size
            vision_count += 1
            return {
                "raw_observation": {"direct_visual_evidence": ["two visible color fields"]},
                "visual_analysis": {
                    "characters": [],
                    "character_appearance": None,
                    "pose": None,
                    "background": "simple color background",
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
                    "style_confidence": 0.85,
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
                return {"cycle": cycle, "observations": [], "style_patterns": [], "video_patterns": [], "image_patterns": [], "confidence": 0.9, "new_rules": ["consistent color split"], "unknowns": [], "next_tasks": []}
            if self.name == "qwen":
                return {"cycle": cycle, "validated": [], "errors": [], "contradictions": [], "weak_claims": [], "corrections": [], "confidence": 0.9}
            return {"cycle": cycle, "kept_knowledge": [], "updated_knowledge": [], "rejected_knowledge": [], "reasons": [], "next_cycle_instructions": [], "memory_changes": [], "validated_style_rules": ["consistent color split"], "confidence": 0.9}

    monkeypatch.setattr(real_channel, "LocalVisionBackend", AvailableVisionBackend)
    monkeypatch.setattr(real_channel.YouTubeIngestion, "inventory", lambda self, url, max_entries: inventory)
    monkeypatch.setattr(real_channel.YouTubeIngestion, "select_and_save", lambda self, inv, target_videos: [video])

    def download(self, selected_video, videos_dir):
        nonlocal download_count
        download_count += 1
        temporary_path = Path(videos_dir) / selected_video["id"] / "video.avi"
        temporary_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_video, temporary_path)
        temporary_downloads.append(temporary_path)
        return temporary_path

    monkeypatch.setattr(real_channel.YouTubeIngestion, "download_video", download)
    monkeypatch.setattr(
        real_channel,
        "_prepare_real_agents",
        lambda config: (
            {name: Agent(name) for name in ("deepseek", "qwen", "llama")},
            {name: {"available": True, "detail": "available"} for name in ("deepseek", "qwen", "llama")},
        ),
    )

    report = real_channel.run_channel_pipeline(
        "https://youtube.com/@channel/videos",
        target_videos=1,
        frames_per_video=2,
        cycles=2,
        resume=True,
    )

    assert report["status"] == "complete"
    assert report["extracted_frame_count"] == report["dataset_record_count"] == 2
    assert download_count == 1 and vision_count == 2
    assert [name for cycle, name in agent_order] == ["deepseek", "qwen", "llama"] * 2
    assert not temporary_downloads[0].exists()
    checkpoint_root = tmp_path / report["checkpoints_dir"]
    assert (checkpoint_root / "videos" / "video-id.json").exists()
    assert (checkpoint_root / "agent_cycles" / "cycle_002.json").exists()
    rows = TrainingDataset(storage_path="training/real_visual", image_root="data").load_visual_records()
    assert len(rows) == 2
    assert rows[0]["raw_observation"] and rows[0]["visual_analysis"]
    assert rows[0]["validated_style_rules"] == ["consistent color split"]

    resumed = real_channel.run_channel_pipeline(
        "https://youtube.com/@channel/videos",
        target_videos=1,
        frames_per_video=2,
        cycles=2,
        resume=True,
    )
    assert resumed["status"] == "complete"
    assert download_count == 1 and vision_count == 2
    assert resumed["dataset_record_count"] == 2


def test_lora_imagefolder_keeps_only_approved_high_confidence_records(tmp_path):
    from training.lora_finetune import prepare_imagefolder

    image_root = tmp_path / "data"
    image_root.mkdir()
    samples = []
    for name, confidence, rules in (
        ("accepted.jpg", 0.9, ["consistent muted palette"]),
        ("low-confidence.jpg", 0.4, ["uncertain rule"]),
        ("no-approved-rule.jpg", 0.95, []),
    ):
        image_path = image_root / name
        assert cv2.imwrite(str(image_path), np.full((16, 16, 3), 128, dtype=np.uint8))
        samples.append({
            "image": name,
            "description": "direct visual evidence",
            "source_video": "video-id",
            "style_confidence": confidence,
            "validated_style_rules": rules,
        })
    dataset_path = tmp_path / "golden_visual_dataset.jsonl"
    dataset_path.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")

    sample_count = prepare_imagefolder(dataset_path, image_root, tmp_path / "imagefolder")

    rows = [json.loads(line) for line in (tmp_path / "imagefolder" / "metadata.jsonl").read_text().splitlines()]
    assert sample_count == len(rows) == 1
    assert rows[0]["text"].startswith("consistent muted palette")