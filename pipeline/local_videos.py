from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2

from ingestion.frame_extractor import FrameExtractor
from ingestion.local_videos import describe_local_video, discover_local_mp4s
from memory.manager import MemoryManager
from pipeline.real_channel import (
    _describe_video,
    _load_config,
    _load_stage_checkpoint,
    _prepare_real_agents,
    _save_stage_checkpoint,
)
from training.dataset import TrainingDataset
from vision.base import VisionAnalyzer
from vision.local_backend import LocalVisionBackend


def _write_report(data_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = data_root / "local_videos_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "report_path": str(report_path)}


def run_local_videos_pipeline(
    input_dir: str | Path,
    frames_per_video: int = 8,
    cycles: int = 1,
    resume: bool = True,
    config_path: str | Path = "config.yaml",
) -> dict[str, Any]:
    if frames_per_video < 1 or cycles < 1:
        raise ValueError("frames_per_video and cycles must be at least 1")
    input_root = Path(input_dir).expanduser().resolve()
    video_paths = discover_local_mp4s(input_root)
    if not video_paths:
        raise ValueError(f"No MP4 files found under {input_root}")

    source_videos = [describe_local_video(path, input_root) for path in video_paths]
    config = _load_config(config_path)
    vision_analyzer = VisionAnalyzer(LocalVisionBackend(config.get("vision", {})))
    vision_available = vision_analyzer.is_available()
    agents, agent_status = _prepare_real_agents(config)
    data_root = Path("data").resolve()
    checkpoint_id = hashlib.sha256(str(input_root).encode("utf-8")).hexdigest()[:12]
    checkpoint_dir = Path("checkpoints") / "local_videos" / checkpoint_id
    _save_stage_checkpoint(checkpoint_dir / "inventory.json", "inventory", {
        "input_dir": str(input_root),
        "source_count": len(source_videos),
        "videos": source_videos,
    })

    blockers = []
    if not vision_available:
        blockers.append(f"Vision: {vision_analyzer.availability_error}")
    if not agents:
        blockers.extend(
            f"{name}: {details['detail']}"
            for name, details in agent_status.items()
            if not details["available"]
        )
    if blockers:
        return _write_report(data_root, {
            "status": "blocked",
            "input_dir": str(input_root),
            "source_file_count": len(source_videos),
            "frames_per_video": frames_per_video,
            "cycles": cycles,
            "processed_video_count": 0,
            "extracted_frame_count": 0,
            "dataset_record_count": 0,
            "dataset_path": "training/local_videos/golden_visual_dataset.jsonl",
            "vision_status": "available" if vision_available else vision_analyzer.availability_error,
            "agent_status": agent_status,
            "mock_used": False,
            "blocked_by": blockers,
            "checkpoints_dir": str(checkpoint_dir),
        })

    extractor = FrameExtractor()
    processed_videos: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for video in source_videos:
        video_id = video["id"]
        video_checkpoint = checkpoint_dir / "videos" / f"{video_id}.json"
        saved_video = _load_stage_checkpoint(video_checkpoint) if resume else None
        if (
            saved_video
            and saved_video.get("stage") == "vision_complete"
            and saved_video.get("source_size_bytes") == video["size_bytes"]
            and saved_video.get("source_mtime_ns") == video["mtime_ns"]
            and saved_video.get("frames_per_video") == frames_per_video
            and saved_video.get("vision_model") == vision_analyzer.backend.model_name
        ):
            saved_frames = saved_video.get("frames", [])
            if saved_frames and all(
                Path(frame["image_path"]).is_file() and Path(frame["image_path"]).stat().st_size > 0
                for frame in saved_frames
            ):
                analysis = saved_video["analysis"]
                processed_videos.append({"video": video, "analysis": analysis, **analysis})
                continue

        try:
            frame_dir = data_root / "local_videos" / video_id / "frames"
            frames = extractor.extract_video_frames(
                video["source_path"],
                frame_dir,
                frame_count=frames_per_video,
            )
            for frame in frames:
                vision_result = vision_analyzer.analyze(frame["image_path"])
                frame["pixel_analysis"] = frame["visual_analysis"]
                frame["raw_observation"] = vision_result["raw_observation"]
                frame["visual_analysis"] = vision_result["visual_analysis"]
                frame["analysis_backend"] = vision_result["analysis_backend"]
            analysis = _describe_video(video, frames)
            processed = {"video": video, "analysis": analysis, **analysis}
            processed_videos.append(processed)
            _save_stage_checkpoint(video_checkpoint, "vision_complete", {
                "video_id": video_id,
                "source_path": video["source_path"],
                "source_size_bytes": video["size_bytes"],
                "source_mtime_ns": video["mtime_ns"],
                "frames_per_video": frames_per_video,
                "vision_model": vision_analyzer.backend.model_name,
                "frames": frames,
                "analysis": analysis,
            })
        except (OSError, RuntimeError, TypeError, ValueError, cv2.error) as exc:
            failures.append({"source_path": video["source_path"], "error": f"{type(exc).__name__}: {exc}"})

    try:
        vision_analyzer.backend.unload()
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    if not processed_videos:
        return _write_report(data_root, {
            "status": "blocked",
            "input_dir": str(input_root),
            "source_file_count": len(source_videos),
            "frames_per_video": frames_per_video,
            "cycles": cycles,
            "processed_video_count": 0,
            "extracted_frame_count": 0,
            "dataset_record_count": 0,
            "dataset_path": "training/local_videos/golden_visual_dataset.jsonl",
            "vision_status": "available" if vision_available else vision_analyzer.availability_error,
            "agent_status": agent_status,
            "mock_used": False,
            "video_failures": failures,
            "checkpoints_dir": str(checkpoint_dir),
        })

    selected_ids = [video["video"]["id"] for video in processed_videos]
    previous_memory: dict[str, Any] = {}
    agent_outputs: dict[str, Any] = {}
    agent_history: list[dict[str, Any]] = []
    evidence = [
        {
            "source_video": video["video"]["id"],
            "source_path": video["video"]["relative_path"],
            "frames": [
                {
                    "image": Path(frame["image_path"]).resolve().relative_to(data_root).as_posix(),
                    "timestamp": frame["timestamp"],
                    "raw_observation": frame["raw_observation"],
                    "visual_analysis": frame["visual_analysis"],
                }
                for frame in video["frames"]
            ],
        }
        for video in processed_videos
    ]
    try:
        for cycle_number in range(1, cycles + 1):
            cycle_checkpoint = checkpoint_dir / "agent_cycles" / f"cycle_{cycle_number:03d}.json"
            saved_cycle = _load_stage_checkpoint(cycle_checkpoint) if resume else None
            if (
                saved_cycle
                and saved_cycle.get("stage") == "agents_cycle_complete"
                and saved_cycle.get("source_ids") == selected_ids
            ):
                cycle_outputs = saved_cycle["outputs"]
            else:
                cycle_outputs = {
                    "deepseek": agents["deepseek"].run(
                        cycle=cycle_number,
                        source_data={"source": "local_videos", "videos": evidence},
                        memory=previous_memory,
                    ),
                }
                cycle_outputs["qwen"] = agents["qwen"].run(
                    cycle=cycle_number,
                    deepseek_output=cycle_outputs["deepseek"],
                    source_data={"videos": evidence},
                )
                cycle_outputs["llama"] = agents["llama"].run(
                    cycle=cycle_number,
                    deepseek_output=cycle_outputs["deepseek"],
                    qwen_output=cycle_outputs["qwen"],
                    memory=previous_memory,
                    source_data={"videos": evidence},
                )
                _save_stage_checkpoint(cycle_checkpoint, "agents_cycle_complete", {
                    "source_ids": selected_ids,
                    "cycle": cycle_number,
                    "outputs": cycle_outputs,
                })
            agent_history.append({"cycle": cycle_number, **cycle_outputs})
            agent_outputs = cycle_outputs
            previous_memory = {
                "validated_style_rules": agent_outputs["llama"].get("validated_style_rules", []),
                "kept_knowledge": agent_outputs["llama"].get("kept_knowledge", []),
            }
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return _write_report(data_root, {
            "status": "blocked",
            "input_dir": str(input_root),
            "source_file_count": len(source_videos),
            "processed_video_count": len(processed_videos),
            "extracted_frame_count": sum(len(video["frames"]) for video in processed_videos),
            "dataset_record_count": 0,
            "mock_used": False,
            "blocked_by": f"{type(exc).__name__}: {exc}",
            "checkpoints_dir": str(checkpoint_dir),
        })

    validated_style_rules = agent_outputs["llama"].get("validated_style_rules", [])
    if not isinstance(validated_style_rules, list):
        validated_style_rules = []
    validated_style_rules = [rule for rule in validated_style_rules if isinstance(rule, str) and rule.strip()]
    memory_manager = MemoryManager(config.get("memory", {}).get("store_path", "memory_store"))
    memory_state = {
        "raw_visual_observations": [
            {
                "source_video": video["video"]["relative_path"],
                "frames": [
                    {"image": frame["image_path"], "raw_observation": frame["raw_observation"]}
                    for frame in video["frames"]
                ],
            }
            for video in processed_videos
        ],
        "validated_style_rules": validated_style_rules,
        "rule_candidates": agent_outputs["deepseek"].get("new_rules", []),
        "critic_review": agent_outputs["qwen"],
        "accepted_knowledge": agent_outputs["llama"].get("kept_knowledge", []),
        "updated_knowledge": agent_outputs["llama"].get("updated_knowledge", []),
        "analysis_backend": f"{vision_analyzer.backend.name} -> DeepSeek -> Qwen -> Llama",
    }
    confidence = sum(float(agent_outputs[name].get("confidence", 0.0)) for name in ("deepseek", "qwen", "llama")) / 3
    memory_manager.write_memory(
        "visual_knowledge_local_videos",
        memory_state,
        agent="llama",
        cycle=cycles,
        reason="persisted local MP4 observations separately from validated visual style rules",
        confidence=confidence,
    )
    _save_stage_checkpoint(checkpoint_dir / "memory.json", "memory_complete", {
        "memory_path": str(memory_manager._file_path("visual_knowledge_local_videos")),
        "validated_style_rules": len(validated_style_rules),
    })

    dataset = TrainingDataset(storage_path="training/local_videos", image_root=data_root)
    existing_images = {sample.get("image") for sample in dataset.load_visual_records()}
    added_samples = 0
    for video in processed_videos:
        for frame in video["frames"]:
            image_path = Path(frame["image_path"]).resolve().relative_to(data_root).as_posix()
            if image_path in existing_images:
                continue
            analysis = frame["visual_analysis"]
            raw_observation = frame["raw_observation"]
            direct_evidence = raw_observation.get("direct_visual_evidence", [])
            sample = {
                "image": image_path,
                "description": "; ".join(str(item) for item in direct_evidence) or json.dumps(raw_observation, ensure_ascii=True),
                "visual_style": analysis.get("drawing_rendering_style"),
                "composition": analysis.get("composition"),
                "lighting": analysis.get("lighting"),
                "colors": analysis.get("color_palette", []),
                "characters": analysis.get("characters", []),
                "character_appearance": analysis.get("character_appearance"),
                "pose": analysis.get("pose"),
                "background": analysis.get("background"),
                "framing": analysis.get("framing"),
                "textures": analysis.get("textures", []),
                "visual_effects": analysis.get("visual_effects", []),
                "on_screen_text": analysis.get("on_screen_text", []),
                "scene_type": analysis.get("scene_type"),
                "camera_framing": analysis.get("camera_framing"),
                "style_confidence": analysis.get("style_confidence", 0.0),
                "source_video": video["video"]["id"],
                "source_path": video["video"]["relative_path"],
                "source_timestamp": frame["timestamp"],
                "timestamp": frame["timestamp"],
                "frame_id": frame["frame_id"],
                "raw_observation": raw_observation,
                "visual_analysis": analysis,
                "validated_style_rules": validated_style_rules,
                "agent_analysis": agent_outputs,
                "cycle": cycles,
                "validated_by": ["deepseek", "qwen", "llama"],
                "analysis_backend": vision_analyzer.backend.name,
                "agent_backend": "ollama",
                "confidence": round(confidence, 4),
                "confidence_basis": "mean confidence returned by the sequential real agents",
                "validation_state": "agent_validated",
            }
            if not dataset.add_visual_sample(sample):
                raise RuntimeError(f"Dataset rejected local frame record: {image_path}")
            existing_images.add(image_path)
            added_samples += 1

    _save_stage_checkpoint(checkpoint_dir / "dataset.json", "dataset_complete", {
        "record_count": len(dataset.load_visual_records()),
        "added_this_run": added_samples,
        "dataset_path": str(dataset.golden_visual_path),
    })
    _save_stage_checkpoint(checkpoint_dir / "agents.json", "agents_complete", {
        "cycles": cycles,
        "source_ids": selected_ids,
        "history": agent_history,
        "outputs": agent_outputs,
    })
    report = {
        "status": "complete" if not failures else "partial",
        "input_dir": str(input_root),
        "source_file_count": len(source_videos),
        "processed_video_count": len(processed_videos),
        "frames_per_video": frames_per_video,
        "cycles": cycles,
        "extracted_frame_count": sum(len(video["frames"]) for video in processed_videos),
        "dataset_record_count": len(dataset.load_visual_records()),
        "dataset_records_added_this_run": added_samples,
        "dataset_path": str(dataset.golden_visual_path),
        "memory_path": str(memory_manager._file_path("visual_knowledge_local_videos")),
        "memory_key": "visual_knowledge_local_videos",
        "validated_style_rules": len(validated_style_rules),
        "agent_status": agent_status,
        "vision_backend": vision_analyzer.backend.name,
        "mock_used": False,
        "source_videos_preserved": all(Path(video["source_path"]).is_file() for video in source_videos),
        "video_failures": failures,
        "checkpoints_dir": str(checkpoint_dir),
    }
    return _write_report(data_root, report)
