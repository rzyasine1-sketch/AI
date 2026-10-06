from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


def discover_local_mp4s(root_path: str | Path) -> list[Path]:
    root = Path(root_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Local video directory does not exist: {root}")
    return sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )


def describe_local_video(path: Path, root: Path) -> dict[str, Any]:
    relative_path = path.relative_to(root).as_posix()
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "video"
    stat = path.stat()
    identity = f"{relative_path}:{stat.st_size}:{stat.st_mtime_ns}"
    identifier = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"{safe_stem}_{identifier}",
        "source_path": str(path),
        "relative_path": relative_path,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
