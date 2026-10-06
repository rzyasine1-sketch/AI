from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agents.deepseek import DeepSeekAgent
from agents.llama import LlamaAgent
from agents.qwen import QwenAgent
from memory.manager import MemoryManager
from models.mock_backend import MockModelBackend
from models.ollama_backend import OllamaBackend
from training.dataset import TrainingDataset
from vision.base import VisionAnalyzer
from vision.local_backend import LocalVisionBackend
from vision.mock_backend import MockVisionBackend


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _create_agents(config: dict[str, Any], mock_mode: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    names = ("deepseek", "qwen", "llama")
    if mock_mode:
        backends = {name: MockModelBackend(f"mock-{name}") for name in names}
        return {
            "deepseek": DeepSeekAgent(backends["deepseek"]),
            "qwen": QwenAgent(backends["qwen"]),
            "llama": LlamaAgent(backends["llama"]),
        }, {name: {"backend": "mock", "available": True} for name in names}

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
        for name in names
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
    return {
        "deepseek": DeepSeekAgent(backends["deepseek"]),
        "qwen": QwenAgent(backends["qwen"]),
        "llama": LlamaAgent(backends["llama"]),
    }, status


def _report_path(data_root: Path, mock_mode: bool) -> Path:
    return data_root / ("local_vision_mock_report.json" if mock_mode else "local_vision_real_report.json")


def run_local_vision_pipeline(
    frame_paths: list[str | Path],
    mock_mode: bool = True,
    config_path: str | Path = "config.yaml",
    frame_dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_root = Path("data").resolve()
    resolved_frames = [Path(frame).resolve() for frame in frame_paths]
    if not resolved_frames:
        raise ValueError("At least one extracted frame image is required")
    for image_path in resolved_frames:
        try:
            image_path.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(f"Frame must be inside the data directory to create a safe dataset reference: {image_path}") from exc
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise ValueError(f"Frame is missing or empty: {image_path}")

    config = _load_config(config_path)
    vision_config = config.get("vision", {})
    if mock_mode:
        analyzer = VisionAnalyzer(MockVisionBackend())
    else:
        analyzer = VisionAnalyzer(LocalVisionBackend(vision_config))
        if not analyzer.is_available():
            report = {
                "status": "blocked",
                "vision_mode": "real",
                "mock_used": False,
                "vision_backend": "ollama-vision",
                "model": vision_config.get("model_name", "qwen3-vl:2b"),
                "frame_count": len(resolved_frames),
                "dataset_record_count": 0,
                "dataset_path": "training/vision_real/golden_visual_dataset.jsonl",
                "blocked_by": analyzer.availability_error,
            }
            report_path = _report_path(data_root, mock_mode)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return {**report, "report_path": str(report_path)}

    frame_results = []
    try:
        for image_path in resolved_frames:
            analysis = analyzer.analyze(image_path)
            frame_results.append({"path": image_path, **analysis})
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        report = {
            "status": "blocked",
            "vision_mode": "mock" if mock_mode else "real",
            "mock_used": mock_mode,
            "vision_backend": analyzer.backend.name,
            "frame_count": len(resolved_frames),
            "analyzed_frame_count": len(frame_results),
            "dataset_record_count": 0,
            "blocked_by": f"{type(exc).__name__}: {exc}",
        }
        report_path = _report_path(data_root, mock_mode)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {**report, "report_path": str(report_path)}

    agents, agent_status = _create_agents(config, mock_mode)
    agent_outputs: dict[str, Any] = {}
    agent_error = None
    if agents:
        evidence = [
            {
                "image": result["path"].relative_to(data_root).as_posix(),
                "raw_observation": result["raw_observation"],
                "visual_analysis": result["visual_analysis"],
            }
            for result in frame_results
        ]
        memory_manager = MemoryManager(config.get("memory", {}).get("store_path", "memory_store"))
        previous_memory = memory_manager.read_memory("visual_knowledge", {}) if not mock_mode else {}
        try:
            agent_outputs["deepseek"] = agents["deepseek"].run(
                cycle=1,
                source_data={"source": "local_frames", "frames": evidence},
                memory=previous_memory,
            )
            agent_outputs["qwen"] = agents["qwen"].run(
                cycle=1,
                deepseek_output=agent_outputs["deepseek"],
                source_data={"frames": evidence},
            )
            agent_outputs["llama"] = agents["llama"].run(
                cycle=1,
                deepseek_output=agent_outputs["deepseek"],
                qwen_output=agent_outputs["qwen"],
                memory=previous_memory,
                source_data={"frames": evidence},
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            agent_error = f"{type(exc).__name__}: {exc}"
            agent_outputs = {}
    else:
        memory_manager = MemoryManager(config.get("memory", {}).get("store_path", "memory_store"))
        agent_error = "One or more configured real analysis agents are unavailable"

    llama_output = agent_outputs.get("llama", {})
    validated_rules = llama_output.get("validated_style_rules", [])
    if not isinstance(validated_rules, list):
        validated_rules = []
    validated_rules = [rule for rule in validated_rules if isinstance(rule, str) and rule.strip()]
    dataset_validators = (
        ["vision-vlm", "deepseek", "qwen", "llama"]
        if agent_outputs and not mock_mode
        else ["mock-vision", "mock-deepseek", "mock-qwen", "mock-llama"]
        if mock_mode and agent_outputs
        else [analyzer.backend.name]
    )
    style_confidences = [float(result["visual_analysis"]["style_confidence"]) for result in frame_results]
    dataset_confidence = sum(style_confidences) / len(style_confidences)
    if mock_mode:
        dataset_confidence = 0.0

    visual_memory = {
        "raw_visual_observations": [
            {
                "image": result["path"].relative_to(data_root).as_posix(),
                "raw_observation": result["raw_observation"],
            }
            for result in frame_results
        ],
        "validated_style_rules": validated_rules,
        "rule_candidates": agent_outputs.get("deepseek", {}).get("new_rules", []),
        "critic_review": agent_outputs.get("qwen", {}),
        "accepted_knowledge": llama_output.get("kept_knowledge", []),
        "updated_knowledge": llama_output.get("updated_knowledge", []),
        "analysis_backend": analyzer.backend.name,
        "agent_backend": "mock" if mock_mode else "ollama" if agent_outputs else "unavailable",
    }
    memory_key = "visual_knowledge_mock" if mock_mode else "visual_knowledge"
    memory_manager.write_memory(
        memory_key,
        visual_memory,
        agent="llama" if agent_outputs else analyzer.backend.name,
        cycle=1,
        reason="separate direct frame observations from critic-reviewed visual style rules",
        confidence=dataset_confidence,
    )

    dataset_dir = Path(frame_dataset_dir or ("training/vision_mock" if mock_mode else "training/vision_real"))
    dataset = TrainingDataset(storage_path=str(dataset_dir), image_root=str(data_root))
    record_count = 0
    for result in frame_results:
        analysis = result["visual_analysis"]
        image_relative = result["path"].relative_to(data_root).as_posix()
        raw_observation = result["raw_observation"]
        direct_evidence = raw_observation.get("direct_visual_evidence", [])
        if not direct_evidence:
            dimensions = raw_observation.get("image_dimensions", {})
            direct_evidence = [
                f"Decoded {dimensions.get('width')}x{dimensions.get('height')} image"
            ]
        sample = {
            "image": image_relative,
            "description": "; ".join(str(item) for item in direct_evidence),
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
            "raw_observation": raw_observation,
            "visual_analysis": analysis,
            "validated_style_rules": validated_rules,
            "agent_analysis": agent_outputs,
            "cycle": 1,
            "source_video": "test_video",
            "source_timestamp": result.get("timestamp"),
            "timestamp": result.get("timestamp"),
            "frame_id": result.get("frame_id"),
            "analysis_backend": analyzer.backend.name,
            "agent_backend": "mock" if mock_mode else "ollama" if agent_outputs else "unavailable",
            "validation_state": "mock_pipeline_only" if mock_mode else "agent_validated" if agent_outputs else "vision_analyzed_agent_validation_blocked",
            "validated_by": dataset_validators,
            "confidence": round(dataset_confidence, 4),
            "confidence_basis": "mock backend; not a real vision confidence" if mock_mode else "style_confidence returned by the vision model",
        }
        if dataset.add_visual_sample(sample):
            record_count += 1
        else:
            raise RuntimeError(f"Visual dataset rejected image-backed sample: {image_relative}")

    report = {
        "status": "mock_complete" if mock_mode else "real_complete" if agent_outputs else "real_vision_partial",
        "vision_mode": "mock" if mock_mode else "real",
        "mock_used": mock_mode,
        "vision_backend": analyzer.backend.name,
        "vision_model_invoked": not mock_mode,
        "frame_count": len(frame_results),
        "dataset_record_count": record_count,
        "dataset_path": str(dataset.golden_visual_path),
        "metadata_path": str(dataset.visual_metadata_path),
        "validated_style_rules": len(validated_rules),
        "agent_status": agent_status,
        "agent_error": agent_error,
        "memory_path": str(memory_manager._file_path(memory_key)),
        "memory_key": memory_key,
    }
    report_path = _report_path(data_root, mock_mode)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "report_path": str(report_path)}
