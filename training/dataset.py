import json
from pathlib import Path
from typing import Any


class TrainingDataset:
    def __init__(self, storage_path: str = "training", image_root: str = "data"):
        self.storage_path = Path(storage_path)
        self.image_root = Path(image_root)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.visual_dir = self.storage_path / "visual"
        self.visual_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_path / "golden_dataset.jsonl"
        self.golden_visual_path = self.storage_path / "golden_visual_dataset.jsonl"
        self.visual_metadata_path = self.visual_dir / "metadata.jsonl"
        self.visual_captions_path = self.visual_dir / "captions.jsonl"
        self.visual_manifest_path = self.visual_dir / "manifest.json"
        self.visual_images_dir = self.visual_dir / "images"
        self.visual_images_dir.mkdir(parents=True, exist_ok=True)

    def is_valid_sample(self, sample: dict[str, Any]) -> bool:
        required = ["instruction", "input", "output", "source", "cycle", "validated_by", "confidence"]
        for key in required:
            if key not in sample:
                return False
        if not isinstance(sample["validated_by"], list):
            return False
        if not isinstance(sample["confidence"], (int, float)):
            return False
        if sample["confidence"] < 0.7:
            return False
        return not len(sample["validated_by"]) < 2

    def is_valid_visual_sample(self, sample: dict[str, Any]) -> bool:
        required = [
            "image",
            "description",
            "visual_style",
            "composition",
            "lighting",
            "colors",
            "characters",
            "scene_type",
            "source_video",
            "source_timestamp",
            "cycle",
            "validated_by",
            "confidence",
        ]
        for key in required:
            if key not in sample:
                return False
        if not isinstance(sample["validated_by"], list):
            return False
        if not isinstance(sample["confidence"], (int, float)):
            return False
        if not 0 <= sample["confidence"] <= 1:
            return False
        analysis_backend = str(sample.get("analysis_backend", ""))
        if (
            analysis_backend == "opencv-pixel-analysis"
            or analysis_backend == "mock-vision"
            or analysis_backend.startswith("ollama-vision:")
        ):
            image = Path(str(sample["image"]))
            if image.is_absolute() or ".." in image.parts:
                return False
            image_path = self.image_root / image
            return image_path.is_file() and image_path.stat().st_size > 0 and bool(sample["validated_by"])
        if sample["confidence"] < 0.7:
            return False
        return len(sample["validated_by"]) >= 2

    def add_sample(self, sample: dict[str, Any], auto_filter: bool = True) -> bool:
        if auto_filter and not self.is_valid_sample(sample):
            return False
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        return True

    def add_visual_sample(self, sample: dict[str, Any], auto_filter: bool = True) -> bool:
        if auto_filter and not self.is_valid_visual_sample(sample):
            return False
        if not self.visual_metadata_path.exists():
            self.visual_metadata_path.write_text("", encoding="utf-8")
        with self.visual_metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        with self.golden_visual_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        caption = {
            "image": sample.get("image"),
            "caption": sample.get("description", ""),
            "visual_style": sample.get("visual_style", ""),
        }
        with self.visual_captions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(caption, sort_keys=True) + "\n")
        manifest = {
            "visual_dataset": "golden_visual_dataset",
            "count": len(self.load_visual_records()),
            "target_model": "visual-frame-generation",
            "scope": "visual style learning only",
            "image_root": str(self.image_root),
            "agent_validation": "required for semantic style claims; pixel-only records are marked provisional",
        }
        self.visual_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return True

    def export_jsonl(self, records: list[dict[str, Any]]) -> Path:
        output_path = self.storage_path / "golden_dataset.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        return output_path

    def export_visual_dataset(self, records: list[dict[str, Any]]) -> Path:
        output_path = self.storage_path / "golden_visual_dataset.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        return output_path

    def load_records(self) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            return []
        rows = []
        for line in self.file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows

    def load_visual_records(self) -> list[dict[str, Any]]:
        if not self.visual_metadata_path.exists():
            return []
        rows = []
        for line in self.visual_metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows
