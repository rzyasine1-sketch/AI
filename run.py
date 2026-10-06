from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from pipeline.real_channel import run_channel_pipeline
from training.lora_finetune import train_lora


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable overnight visual style pipeline")
    parser.add_argument("--overnight", action="store_true", required=True, help="Run real channel ingestion through the gated LoRA stage")
    parser.add_argument("--channel", default=os.environ.get("CHANNEL_URL"), help="YouTube channel URL; may be set with CHANNEL_URL")
    parser.add_argument("--cookies", help="Temporary local Netscape-format cookies file; never commit it")
    parser.add_argument("--target-videos", type=int, default=3)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--inventory-limit", type=int, default=500)
    parser.add_argument("--min-dataset-samples", type=int, default=100)
    parser.add_argument("--min-style-confidence", type=float, default=0.75)
    parser.add_argument("--lora-epochs", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--base-image-model", default=os.environ.get("BASE_IMAGE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.channel:
        parser.error("provide --channel or set CHANNEL_URL")
    if min(args.target_videos, args.frames_per_video, args.cycles, args.inventory_limit, args.min_dataset_samples) < 1:
        parser.error("video, frame, cycle, inventory, and dataset limits must be positive")
    if not 0 <= args.min_style_confidence <= 1:
        parser.error("--min-style-confidence must be between 0 and 1")

    try:
        report = run_channel_pipeline(
            args.channel,
            target_videos=args.target_videos,
            frames_per_video=args.frames_per_video,
            inventory_limit=args.inventory_limit,
            cycles=args.cycles,
            resume=not args.no_resume,
            cookies_path=args.cookies,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        report = {"status": "blocked", "blocked_by": f"{type(exc).__name__}: {exc}"}
        _save_report(Path("data/real_run_report.json"), report)
        print(json.dumps(report, indent=2))
        return 2

    report_path = Path("data/real_run_report.json")
    if report.get("status") != "complete":
        report["training_status"] = "blocked_upstream_pipeline_not_complete"
        report["training_blocked_by"] = report.get("blocked_by") or report.get("agent_error") or "real pipeline incomplete"
        _save_report(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    dataset_path = Path(report["dataset_path"])
    records = _load_jsonl(dataset_path)
    image_root = Path("data")
    quality_records = [
        sample
        for sample in records
        if sample.get("validated_style_rules")
        and float(sample.get("style_confidence", 0.0)) >= args.min_style_confidence
        and (image_root / str(sample.get("image", ""))).is_file()
    ]
    mean_confidence = (
        sum(float(sample["style_confidence"]) for sample in quality_records) / len(quality_records)
        if quality_records
        else 0.0
    )
    report["training_gate"] = {
        "valid_samples": len(quality_records),
        "minimum_samples": args.min_dataset_samples,
        "mean_style_confidence": round(mean_confidence, 4),
        "minimum_style_confidence": args.min_style_confidence,
        "base_model": args.base_image_model,
    }
    checkpoints_dir = Path(report.get("checkpoints_dir", "checkpoints/overnight"))
    if len(quality_records) < args.min_dataset_samples or mean_confidence < args.min_style_confidence:
        report["training_status"] = "skipped_quality_or_size_gate"
        _save_report(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    try:
        training_result = train_lora(
            dataset_path,
            image_root,
            "training/lora",
            base_model=args.base_image_model,
            epochs=args.lora_epochs,
            resume=not args.no_resume,
            minimum_style_confidence=args.min_style_confidence,
        )
        report["training_status"] = training_result["status"]
        report["training_result"] = training_result
        checkpoint = checkpoints_dir / "training.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"stage": "training", **training_result}, indent=2), encoding="utf-8")
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        report["training_status"] = "blocked"
        report["training_blocked_by"] = f"{type(exc).__name__}: {exc}"
        _save_report(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    _save_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_status"] in {"complete", "skipped_below_minimum"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
