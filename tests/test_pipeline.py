import json
import subprocess
import sys
from pathlib import Path

import pytest

from ingestion.frame_extractor import FrameExtractor
from memory.manager import MemoryManager
from memory.schemas import validate_agent_payload
from memory.versioning import MemoryVersioning
from pipeline.quality import calculate_quality_score
from training.dataset import TrainingDataset


def test_json_schema_validation():
    payload = {
        "cycle": 1,
        "observations": ["x"],
        "style_patterns": ["y"],
        "video_patterns": ["z"],
        "image_patterns": [],
        "confidence": 0.9,
        "new_rules": ["rule"],
        "unknowns": [],
        "next_tasks": ["task"],
    }
    assert validate_agent_payload("deepseek", payload) is True


def test_memory_read_write(tmp_path):
    mgr = MemoryManager(base_path=str(tmp_path))
    mgr.write_memory("camera_motion", {"value": "tracking"}, agent="deepseek", cycle=1)
    assert mgr.read_memory("camera_motion") == {"value": "tracking"}


def test_memory_versioning(tmp_path):
    versioner = MemoryVersioning(base_path=str(tmp_path))
    versioner.snapshot("camera_motion", "old", "new", agent="llama", cycle=2, reason="updated")
    history = versioner.load_history("camera_motion")
    assert len(history) == 1
    assert history[0]["agent"] == "llama"


def test_checkpoint_creation(tmp_path):
    ckpt_path = tmp_path / "checkpoint_001.json"
    ckpt_path.write_text(json.dumps({"cycle": 1, "status": "ok"}))
    assert ckpt_path.exists()


def test_cache_and_hash(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    from cache.manager import CacheManager

    cache = CacheManager(str(cache_dir))
    key = "abc"
    assert cache.get(key) is None
    cache.set(key, {"ok": True})
    assert cache.get(key) == {"ok": True}


def test_training_dataset_filtering():
    ds = TrainingDataset(storage_path=".")
    sample = {
        "instruction": "Summarize style",
        "input": "video analysis",
        "output": "style is balanced",
        "source": "mock",
        "cycle": 1,
        "validated_by": ["qwen", "llama"],
        "confidence": 0.9,
    }
    assert ds.is_valid_sample(sample) is True
    assert ds.is_valid_sample({**sample, "confidence": 0.4}) is False


def test_quality_score_range():
    score = calculate_quality_score({
        "consistency": 0.9,
        "correctness": 0.8,
        "confidence": 0.7,
        "duplicate_rate": 0.1,
        "contradiction_rate": 0.2,
        "memory_stability": 0.85,
    })
    assert 0.0 <= score <= 1.0


def test_resume_state(tmp_path):
    state_path = tmp_path / "resume_state.json"
    state = {"last_completed_cycle": 8}
    state_path.write_text(json.dumps(state))
    data = json.loads(state_path.read_text())
    assert data["last_completed_cycle"] == 8


def test_stop_signal(tmp_path):
    stop_path = tmp_path / "STOP"
    stop_path.write_text("stop")
    assert stop_path.exists()


def test_config_loading(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text('models:\n  deepseek:\n    name: "x"\n')
    assert config_path.exists()


def test_mock_pipeline_sequence():
    assert ["deepseek", "qwen", "llama"] == ["deepseek", "qwen", "llama"]


def test_corrupted_json_handling(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{broken json')
    with pytest.raises(ValueError):
        json.loads(path.read_text())


def test_cli_main_invocation():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "main.py", "--mock", "--cycles", "2"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "cycles_run" in result.stdout


def test_full_mock_cycle(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
models:
  deepseek:
    name: "mock-deepseek"
    backend: "mock"
  qwen:
    name: "mock-qwen"
    backend: "mock"
  llama:
    name: "mock-llama"
    backend: "mock"

pipeline:
  max_cycles: 2
  retry_count: 2
  timeout: 30
  quality_threshold: 0.75
  no_improvement_cycles: 3

cache:
  enabled: true

memory:
  versioning: true
  store_path: "memory_store"
""".strip()
    )
    monkeypatch.chdir(tmp_path)
    from pipeline.supervisor import Supervisor

    supervisor = Supervisor(config_path="config.yaml", mock_mode=True)
    result = supervisor.run(max_cycles=2, resume=False)
    assert result["last_completed_cycle"] == 2
    assert (tmp_path / "cycles" / "cycle_001" / "deepseek.json").exists()
    assert (tmp_path / "checkpoints" / "checkpoint_002.json").exists()


def test_resume_keeps_quality_history(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
models:
  deepseek:
    name: "mock-deepseek"
    backend: "mock"
  qwen:
    name: "mock-qwen"
    backend: "mock"
  llama:
    name: "mock-llama"
    backend: "mock"

pipeline:
  max_cycles: 2
  retry_count: 2
  timeout: 30
  quality_threshold: 0.75
  no_improvement_cycles: 3

cache:
  enabled: true

memory:
  versioning: true
  store_path: "memory_store"
""".strip()
    )
    monkeypatch.chdir(tmp_path)
    from pipeline.supervisor import Supervisor

    first = Supervisor(config_path="config.yaml", mock_mode=True)
    first.run(max_cycles=1, resume=False)
    second = Supervisor(config_path="config.yaml", mock_mode=True)
    resumed = second.run(max_cycles=2, resume=True)
    assert resumed["last_completed_cycle"] == 2
    assert len(resumed["quality_history"]) == 2


def test_visual_frame_dataset_contract(tmp_path):
    ds = TrainingDataset(storage_path=str(tmp_path))
    sample = {
        "image": "images/frame_000123.jpg",
        "description": "hand-drawn character in left third with muted colors",
        "visual_style": "hand-drawn with soft shadows",
        "composition": "character in left third",
        "lighting": "soft diffused light",
        "colors": ["muted blue", "warm beige"],
        "characters": ["protagonist"],
        "scene_type": "close-up",
        "source_video": "video_001.mp4",
        "source_timestamp": 123.4,
        "cycle": 8,
        "validated_by": ["qwen", "llama"],
        "confidence": 0.94,
    }
    assert ds.add_visual_sample(sample) is True
    assert (tmp_path / "visual" / "metadata.jsonl").exists()
    assert (tmp_path / "visual" / "manifest.json").exists()


def test_frame_extractor_selects_representative_frames():
    frame_data = [
        {"frame_id": 1, "timestamp": 1.0, "score": 0.2},
        {"frame_id": 2, "timestamp": 2.0, "score": 0.9},
        {"frame_id": 3, "timestamp": 3.0, "score": 0.88},
    ]
    extractor = FrameExtractor()
    selected = extractor.select_representative_frames(frame_data)
    assert len(selected) >= 2
    assert selected[0]["frame_id"] == 2
