from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def prepare_imagefolder(
    dataset_path: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    minimum_style_confidence: float = 0.75,
) -> int:
    dataset_path = Path(dataset_path)
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    rows = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        rules = sample.get("validated_style_rules", [])
        if not isinstance(rules, list) or not rules:
            continue
        try:
            confidence = float(sample.get("style_confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < minimum_style_confidence:
            continue
        source = image_root / sample["image"]
        if not source.is_file() or source.stat().st_size == 0:
            continue
        safe_name = f"{sample.get('source_video', 'video')}_{Path(sample['image']).name}"
        shutil.copy2(source, output_dir / safe_name)
        caption = "; ".join(str(value) for value in rules)
        evidence = sample.get("description", "")
        rows.append({"file_name": safe_name, "text": f"{caption}. Visual evidence: {evidence}"})
    metadata_path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def train_lora(
    dataset_path: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    base_model: str,
    epochs: int = 1,
    resume: bool = True,
    minimum_style_confidence: float = 0.75,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    imagefolder = output_dir / "imagefolder"
    sample_count = prepare_imagefolder(dataset_path, image_root, imagefolder, minimum_style_confidence)
    if sample_count < 100:
        return {"status": "skipped_below_minimum", "valid_training_samples": sample_count, "minimum_samples": 100}

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "diffusers[torch]", "transformers", "accelerate", "peft", "datasets", "bitsandbytes"],
        check=True,
    )
    diffusers_dir = output_dir / "diffusers"
    if not diffusers_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/huggingface/diffusers.git", str(diffusers_dir)],
            check=True,
        )
    training_script = diffusers_dir / "examples" / "text_to_image" / "train_text_to_image_lora.py"
    if not training_script.is_file():
        raise RuntimeError(f"Diffusers LoRA training entrypoint is missing: {training_script}")

    gradient_accumulation = 4
    steps_per_epoch = max(1, math.ceil(sample_count / gradient_accumulation))
    command = [
        "accelerate", "launch", str(training_script),
        "--pretrained_model_name_or_path", base_model,
        "--train_data_dir", str(imagefolder),
        "--image_column", "image",
        "--caption_column", "text",
        "--resolution", "512",
        "--center_crop",
        "--random_flip",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", str(gradient_accumulation),
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--learning_rate", "1e-4",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--num_train_epochs", str(epochs),
        "--checkpointing_steps", str(steps_per_epoch),
        "--checkpointing_limit", "3",
        "--mixed_precision", "fp16",
        "--output_dir", str(output_dir / "adapter"),
        "--report_to", "none",
    ]
    checkpoint_root = output_dir / "adapter"
    if resume and checkpoint_root.exists() and any(checkpoint_root.glob("checkpoint-*")):
        command.extend(["--resume_from_checkpoint", "latest"])
    (output_dir / "training_command.json").write_text(
        json.dumps({"command": command, "epochs": epochs, "steps_per_epoch": steps_per_epoch}, indent=2),
        encoding="utf-8",
    )
    subprocess.run(command, check=True)
    checkpoints = sorted(path.name for path in checkpoint_root.glob("checkpoint-*"))
    return {
        "status": "complete",
        "base_model": base_model,
        "valid_training_samples": sample_count,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "adapter_dir": str(checkpoint_root),
        "epoch_checkpoints": checkpoints,
    }
