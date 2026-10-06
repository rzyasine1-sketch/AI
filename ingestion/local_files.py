import json
from pathlib import Path
from typing import Any

from ingestion.frame_extractor import FrameExtractor


class LocalFilesIngestion:
    def __init__(self, root_path: str = "data/raw"):
        self.root_path = Path(root_path)
        self.frame_extractor = FrameExtractor()

    def read_files(self) -> list[dict[str, Any]]:
        records = []
        if not self.root_path.exists():
            return records
        for path in sorted(self.root_path.iterdir()):
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix.lower() == ".json":
                try:
                    parsed = json.loads(content)
                    records.append({"path": str(path), "type": "json", "content": parsed})
                except json.JSONDecodeError:
                    records.append({"path": str(path), "type": "json", "content": content})
            elif path.suffix.lower() in {".txt", ".md"}:
                records.append({"path": str(path), "type": path.suffix.lower().lstrip("."), "content": content})
            elif path.suffix.lower() == ".jsonl":
                rows = []
                for line in content.splitlines():
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            rows.append({"raw": line})
                records.append({"path": str(path), "type": "jsonl", "content": rows})

            if path.suffix.lower() in {".json", ".txt", ".md", ".jsonl"}:
                representative_frames = self.frame_extractor.select_representative_frames([
                    {"frame_id": f"{path.stem}_1", "timestamp": 1.0, "score": 0.9},
                    {"frame_id": f"{path.stem}_2", "timestamp": 2.0, "score": 0.8},
                ])
                records[-1]["representative_frames"] = representative_frames
        return records
