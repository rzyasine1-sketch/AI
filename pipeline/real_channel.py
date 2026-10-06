from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from agents.deepseek import DeepSeekAgent
from agents.llama import LlamaAgent
from agents.qwen import QwenAgent
from ingestion.frame_extractor import FrameExtractor
from ingestion.youtube import YouTubeIngestion
from memory.manager import MemoryManager
from models.ollama_backend import OllamaBackend
from training.dataset import TrainingDataset
from vision.base import VisionAnalyzer
from vision.local_backend import LocalVisionBackend


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _save_stage_checkpoint(path: Path, stage: str, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"stage": stage, **state}, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_stage_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _describe_video(video: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
    palette_counts = Counter(
        color
        for frame in frames
        for color in frame.get("visual_analysis", {}).get("color_palette", [])
    )
    if not palette_counts:
        palette_counts = Counter(
            color
            for frame in frames
            for color in frame.get("pixel_analysis", {}).get("palette", [])
        )
    style_tags = Counter(
        tag
        for frame in frames
        for tag in (
            frame["visual_analysis"].get("drawing_rendering_style"),
            frame["visual_analysis"].get("lighting"),
        )
        if tag
    )
    compositions = Counter(frame["visual_analysis"].get("composition") for frame in frames if frame["visual_analysis"].get("composition"))
    if not compositions:
        compositions = Counter(frame.get("pixel_analysis", {}).get("composition") for frame in frames)
    common_palette = [color for color, _ in palette_counts.most_common(5)]
    palette_sets = [
        set(frame["visual_analysis"].get("color_palette") or frame.get("pixel_analysis", {}).get("palette", []))
        for frame in frames
    ]
    consistency = (
        sum(len(palette_sets[0] & palette) / max(1, len(palette_sets[0] | palette)) for palette in palette_sets[1:])
        / (len(palette_sets) - 1)
        if len(palette_sets) > 1
        else 1.0
    )
    change_scores = []
    for previous, current in pairwise(frames):
        previous_image = cv2.imread(previous["image_path"])
        current_image = cv2.imread(current["image_path"])
        if previous_image is None or current_image is None:
            continue
        previous_image = cv2.resize(previous_image, (96, 64))
        current_image = cv2.resize(current_image, (96, 64))
        change_scores.append(float(np.mean(cv2.absdiff(previous_image, current_image))) / 255.0)
    mean_change = sum(change_scores) / len(change_scores) if change_scores else 0.0
    scene_structure = "stable sampled appearance" if mean_change < 0.08 else "appearance changes between sampled frames"
    lighting = Counter(
        frame["visual_analysis"].get("lighting") for frame in frames if frame["visual_analysis"].get("lighting")
    ).most_common(1)
    framing = Counter(
        frame["visual_analysis"].get("framing") for frame in frames if frame["visual_analysis"].get("framing")
    ).most_common(1)
    camera_framing = Counter(
        frame["visual_analysis"].get("camera_framing")
        for frame in frames
        if frame["visual_analysis"].get("camera_framing")
    ).most_common(1)
    recurring_elements = Counter(
        str(item)
        for frame in frames
        for item in (
            frame["visual_analysis"].get("characters", [])
            + frame["visual_analysis"].get("visual_effects", [])
        )
    )
    return {
        "video_id": str(video["id"]),
        "visual_style": ", ".join(style_tags.keys()),
        "colors": common_palette,
        "composition": compositions.most_common(1)[0][0] if compositions else None,
        "lighting": lighting[0][0] if lighting else None,
        "framing": framing[0][0] if framing else None,
        "camera_framing": camera_framing[0][0] if camera_framing else None,
        "recurring_visual_elements": [
            item for item, count in recurring_elements.items() if count > 1
        ] + [f"recurring measured color {color}" for color, count in palette_counts.items() if count > 1],
        "visual_consistency": round(consistency, 4),
        "scene_structure": scene_structure,
        "sampled_frame_change": round(mean_change, 5),
        "camera_language": "inferred from vision model per-frame framing; motion not inferred from still images",
        "character_appearance": None,
        "background_description": None,
        "raw_observations": [frame["raw_observation"] for frame in frames],
        "visual_analyses": [frame["visual_analysis"] for frame in frames],
        "frames": frames,
    }


def _prepare_real_agents(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_config = config.get("models", {})
    backends = {
        name: OllamaBackend(
            name=name,
            config={
                "model_name": model_config.get(name, {}).get("name", name),
                "base_url": model_config.get(name, {}).get("base_url"),
                "timeout": config.get("pipeline", {}).get("timeout", 300),
            },
        )
        for name in ("deepseek", "qwen", "llama")
    }
    status = {
        name: {
            "backend": "ollama",
            "model": backend.model_name,
            "available": backend.is_available(),
            "detail": backend.availability_error,
        }
        for name, backend in backends.items()
    }
    if not all(item["available"] for item in status.values()):
        return {}, status
    agents = {
        "deepseek": DeepSeekAgent(backends["deepseek"]),
        "qwen": QwenAgent(backends["qwen"]),
        "llama": LlamaAgent(backends["llama"]),
    }
    return agents, status


def run_channel_pipeline(
    channel_url: str,
    target_videos: int = 20,
    frames_per_video: int = 8,
    inventory_limit: int = 500,
    cycles: int = 1,
    resume: bool = True,
    cookies_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
) -> dict[str, Any]:
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    data_dir = Path("data")
    videos_dir = data_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    vision_config = config.get("vision", {})
    vision_analyzer = VisionAnalyzer(LocalVisionBackend(vision_config))
    vision_available = vision_analyzer.is_available()
    agents, agent_status = _prepare_real_agents(config)
    ingestion = YouTubeIngestion(data_dir=data_dir, cookies_path=cookies_path)
    inventory = ingestion.inventory(channel_url, max_entries=inventory_limit)
    selected = ingestion.select_and_save(inventory, target_videos=target_videos)
    if not selected:
        raise RuntimeError("The channel inventory contained no eligible videos to sample")

    checkpoint_dir = Path("checkpoints") / "overnight" / str(inventory.get("channel_id", "channel"))
    _save_stage_checkpoint(checkpoint_dir / "inventory.json", "inventory", {
        "channel_url": channel_url,
        "inventory_count": len(inventory["videos"]),
        "channel_id": inventory.get("channel_id"),
    })
    _save_stage_checkpoint(checkpoint_dir / "sampling.json", "sampling", {
        "target_videos": target_videos,
        "selected_ids": [str(video["id"]) for video in selected],
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
        report = {
            "status": "blocked",
            "channel_url": channel_url,
            "channel_id": inventory.get("channel_id"),
            "target_videos": target_videos,
            "frames_per_video": frames_per_video,
            "cycles": cycles,
            "inventory_count": len(inventory["videos"]),
            "selected_count": len(selected),
            "downloaded_and_analyzed_count": 0,
            "extracted_frame_count": 0,
            "dataset_record_count": 0,
            "vision_status": "available" if vision_available else vision_analyzer.availability_error,
            "agent_status": agent_status,
            "blocked_by": blockers,
        }
        (data_dir / "real_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    extractor = FrameExtractor()
    processed_videos: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for video in selected:
        video_id = str(video["id"])
        video_checkpoint = checkpoint_dir / "videos" / f"{video_id}.json"
        saved_video = _load_stage_checkpoint(video_checkpoint) if resume else None
        if saved_video and saved_video.get("status") == "analysis_complete":
            saved_frames = saved_video.get("frames", [])
            if saved_frames and all(
                Path(frame["image_path"]).is_file()
                and Path(frame["image_path"]).stat().st_size > 0
                for frame in saved_frames
            ):
                processed_videos.append({"video": video, **saved_video["analysis"]})
                continue
        try:
            frame_dir = videos_dir / video_id / "frames"
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            with tempfile.TemporaryDirectory(prefix=f"{video_id}_", dir=videos_dir) as temporary_dir:
                video_path = ingestion.download_video(video, videos_dir=temporary_dir)
                frames = extractor.extract_video_frames(video_path, frame_dir, frame_count=frames_per_video)
            for frame in frames:
                vision_result = vision_analyzer.analyze(frame["image_path"])
                frame["pixel_analysis"] = frame["visual_analysis"]
                frame["raw_observation"] = vision_result["raw_observation"]
                frame["visual_analysis"] = vision_result["visual_analysis"]
                frame["analysis_backend"] = vision_result["analysis_backend"]
            analysis = _describe_video(video, frames)
            processed_videos.append({"video": video, "analysis": analysis, **analysis})
            _save_stage_checkpoint(video_checkpoint, "analysis_complete", {
                "status": "analysis_complete",
                "video_id": video_id,
                "frames": frames,
                "analysis": analysis,
            })
        except (OSError, RuntimeError, TypeError, ValueError, cv2.error) as exc:
            failures.append({"video_id": video_id, "error": f"{type(exc).__name__}: {exc}"})
    try:
        vision_analyzer.backend.unload()
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    if not processed_videos:
        report = {
            "status": "blocked",
            "channel_url": channel_url,
            "channel_id": inventory.get("channel_id"),
            "youtube_authentication": "cookies file supplied" if cookies_path else "anonymous",
            "inventory_count": len(inventory["videos"]),
            "target_videos": target_videos,
            "frames_per_video": frames_per_video,
            "cycles": cycles,
            "selected_count": len(selected),
            "downloaded_and_analyzed_count": 0,
            "extracted_frame_count": 0,
            "dataset_record_count": 0,
            "dataset_path": "training/real_visual/golden_visual_dataset.jsonl",
            "vision_status": "available" if vision_available else vision_analyzer.availability_error,
            "agent_status": agent_status,
            "blocked_by": failures,
            "video_failures": failures,
        }
        (data_dir / "real_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    selected_ids = [str(video["id"]) for video in selected]
    agents_checkpoint = checkpoint_dir / "agents.json"
    agent_outputs: dict[str, Any] = {}
    agent_history: list[dict[str, Any]] = []
    agent_error = None
    previous_memory: dict[str, Any] = {}
    evidence = [
        {
            "source_video": video["video"]["id"],
            "title": video["video"].get("title"),
            "selection_reason": video["video"].get("selection_reason"),
            "frames": [
                {
                    "image": Path(frame["image_path"]).resolve().relative_to(data_dir.resolve()).as_posix(),
                    "timestamp": frame["timestamp"],
                    "raw_observation": frame["raw_observation"],
                    "visual_analysis": frame["visual_analysis"],
                }
                for frame in video["frames"]
            ],
        }
        for video in processed_videos
    ]
    if agents:
        try:
            for cycle_number in range(1, cycles + 1):
                cycle_checkpoint = checkpoint_dir / "agent_cycles" / f"cycle_{cycle_number:03d}.json"
                saved_cycle = _load_stage_checkpoint(cycle_checkpoint) if resume else None
                if (
                    saved_cycle
                    and saved_cycle.get("stage") == "agents_cycle_complete"
                    and saved_cycle.get("selected_ids") == selected_ids
                ):
                    cycle_outputs = saved_cycle["outputs"]
                else:
                    cycle_outputs = {
                        "deepseek": agents["deepseek"].run(
                            cycle=cycle_number,
                            source_data={"source": "youtube", "channel_id": inventory["channel_id"], "videos": evidence},
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
                        "selected_ids": selected_ids,
                        "cycle": cycle_number,
                        "outputs": cycle_outputs,
                    })
                agent_history.append({"cycle": cycle_number, **cycle_outputs})
                agent_outputs = cycle_outputs
                previous_memory = {
                    "validated_style_rules": agent_outputs["llama"].get("validated_style_rules", []),
                    "kept_knowledge": agent_outputs["llama"].get("kept_knowledge", []),
                }
            _save_stage_checkpoint(agents_checkpoint, "agents_complete", {
                "selected_ids": selected_ids,
                "cycles": cycles,
                "history": agent_history,
                "outputs": agent_outputs,
            })
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            agent_error = f"{type(exc).__name__}: {exc}"
            agent_outputs = {}

    memory_manager = MemoryManager(config.get("memory", {}).get("store_path", "memory_store"))
    validated_style_rules = agent_outputs.get("llama", {}).get("validated_style_rules", [])
    if not isinstance(validated_style_rules, list):
        validated_style_rules = []
    if agent_outputs:
        memory_state = {
            "raw_visual_observations": [
                {
                    "video_id": video["video_id"],
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
        agent_confidence = sum(float(agent_outputs[name].get("confidence", 0.0)) for name in agent_outputs) / 3
        dataset_validators = list(agent_outputs)
        confidence_basis = "mean confidence returned by the three configured agent models"
        validation_state = "agent_validated"
    else:
        memory_state = {
            "raw_visual_observations": [
                {"video_id": video["video_id"], "frames": [frame["raw_observation"] for frame in video["frames"]]}
                for video in processed_videos
            ],
            "validated_style_rules": [],
            "analysis_backend": vision_analyzer.backend.name,
            "agent_status": agent_status,
        }
        agent_confidence = 0.0
        dataset_validators = [vision_analyzer.backend.name]
        confidence_basis = "vision model style confidence; agent validation unavailable"
        validation_state = "vision_analyzed_agent_validation_blocked"
    memory_manager.write_memory(
        "visual_knowledge",
        memory_state,
        agent="llama" if agent_outputs else vision_analyzer.backend.name,
        cycle=1,
        reason="persisted raw frame observations separately from critic-reviewed visual style rules",
        confidence=agent_confidence,
    )
    _save_stage_checkpoint(checkpoint_dir / "memory.json", "memory_complete", {
        "memory_path": str(memory_manager._file_path("visual_knowledge")),
        "validated_style_rules": len(validated_style_rules),
        "agent_validated": bool(agent_outputs),
    })

    dataset = TrainingDataset(storage_path="training/real_visual", image_root=data_dir)
    existing_images = {sample.get("image") for sample in dataset.load_visual_records()}
    added_samples = 0
    for video in processed_videos:
        for frame in video["frames"]:
            image_path = Path(frame["image_path"]).resolve().relative_to(data_dir.resolve()).as_posix()
            if image_path in existing_images:
                continue
            analysis = frame["visual_analysis"]
            raw_observation = frame["raw_observation"]
            direct_evidence = raw_observation.get("direct_visual_evidence", [])
            description = "; ".join(str(item) for item in direct_evidence) or json.dumps(raw_observation, ensure_ascii=True)
            sample = {
                "image": image_path,
                "description": description,
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
                "recurring_visual_elements": video["recurring_visual_elements"],
                "visual_consistency": video["visual_consistency"],
                "scene_structure": video["scene_structure"],
                "source_video": video["video_id"],
                "source_timestamp": frame["timestamp"],
                "timestamp": frame["timestamp"],
                "raw_observation": raw_observation,
                "visual_analysis": analysis,
                "validated_style_rules": validated_style_rules,
                "agent_analysis": agent_outputs,
                "cycle": 1,
                "validated_by": dataset_validators,
                "analysis_backend": vision_analyzer.backend.name,
                "confidence": round(agent_confidence, 4),
                "confidence_basis": confidence_basis,
                "validation_state": validation_state,
            }
            if dataset.add_visual_sample(sample):
                added_samples += 1
                existing_images.add(image_path)
            else:
                raise RuntimeError(f"Dataset rejected frame with image path {image_path}")

    _save_stage_checkpoint(checkpoint_dir / "dataset.json", "dataset_complete", {
        "record_count": len(dataset.load_visual_records()),
        "added_this_run": added_samples,
        "dataset_path": str(dataset.golden_visual_path),
    })

    if agent_outputs:
        (data_dir / "agent_analysis.json").write_text(json.dumps(agent_outputs, indent=2), encoding="utf-8")
    report = {
        "status": "complete" if not failures and agent_outputs else "partial",
        "channel_url": channel_url,
        "channel_id": inventory.get("channel_id"),
        "youtube_authentication": "cookies file supplied" if cookies_path else "anonymous",
        "inventory_count": len(inventory["videos"]),
        "target_videos": target_videos,
        "frames_per_video": frames_per_video,
        "cycles": cycles,
        "selected_count": len(selected),
        "downloaded_and_analyzed_count": len(processed_videos),
        "extracted_frame_count": sum(len(video["frames"]) for video in processed_videos),
        "dataset_record_count": len(dataset.load_visual_records()),
        "dataset_records_added_this_run": added_samples,
        "dataset_path": str(dataset.golden_visual_path),
        "visual_manifest": str(dataset.visual_manifest_path),
        "memory_path": str(memory_manager._file_path("visual_knowledge")),
        "agent_status": agent_status,
        "agent_error": agent_error,
        "validation_state": validation_state,
        "video_failures": failures,
        "checkpoints_dir": str(checkpoint_dir),
        "downloaded_video_retention": "temporary files deleted after frame extraction",
    }
    (data_dir / "real_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
