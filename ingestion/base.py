from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class IngestionSource(ABC):
    @abstractmethod
    def load(self) -> list[Any]:
        raise NotImplementedError


class LocalDataSource(IngestionSource):
    def __init__(self, root_path: str = "data/raw"):
        self.root_path = Path(root_path)

    def load(self) -> list[Any]:
        items = []
        if not self.root_path.exists():
            return items
        for file_path in sorted(self.root_path.iterdir()):
            if file_path.is_file():
                items.append({"path": str(file_path), "type": file_path.suffix.lower(), "content": file_path.read_text(encoding="utf-8", errors="ignore")})
        return items
