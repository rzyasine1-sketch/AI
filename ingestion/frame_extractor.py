from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FrameExtractor:
    """Selects representative frames for visual-style learning."""

    def __init__(self, min_frames: int = 2, max_frames: int = 8):
        self.min_frames = min_frames
        self.max_frames = max_frames

    def select_representative_frames(self, frame_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not frame_data:
            return []
        ranked = sorted(frame_data, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        selected = []
        seen_timestamps: set[float] = set()
        for frame in ranked:
            timestamp = float(frame.get("timestamp", 0.0))
            if timestamp in seen_timestamps:
                continue
            selected.append(frame)
            seen_timestamps.add(timestamp)
            if len(selected) >= self.max_frames:
                break
        if len(selected) < self.min_frames and ranked:
            selected = ranked[: self.min_frames]
        return selected

    def build_frame_metadata(self, source_video: str, frame_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = self.select_representative_frames(frame_records)
        for index, frame in enumerate(selected, start=1):
            frame.setdefault("frame_id", f"{source_video}_{index:04d}")
            frame.setdefault("selected", True)
            frame.setdefault("source_video", source_video)
        return selected

    def extract_video_frames(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        frame_count: int = 8,
    ) -> list[dict[str, Any]]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise ValueError(f"Video file is missing or empty: {video_path}")
        if frame_count < 1:
            raise ValueError("frame_count must be at least 1")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total_frames < 1 or not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise RuntimeError(f"Video has invalid frame count or frame rate: {video_path}")

        safe_end = max(0, total_frames - max(2, int(fps * 0.5)))
        frame_indices = np.linspace(0, safe_end, min(frame_count, safe_end + 1), dtype=int)
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        try:
            for frame_index in np.unique(frame_indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                success, frame = capture.read()
                if not success or frame is None or frame.size == 0:
                    raise RuntimeError(f"Could not decode frame {frame_index} from {video_path}")
                image_path = output_dir / f"frame_{int(frame_index):06d}.jpg"
                if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(f"Could not write extracted frame: {image_path}")
                verified = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if verified is None or verified.size == 0 or image_path.stat().st_size == 0:
                    raise RuntimeError(f"Extracted frame failed image verification: {image_path}")
                records.append(
                    {
                        "frame_id": int(frame_index),
                        "timestamp": round(int(frame_index) / fps, 3),
                        "image_path": str(image_path),
                        "score": self._frame_score(verified),
                        "visual_analysis": self.analyze_frame(verified),
                    }
                )
        finally:
            capture.release()
        if not records:
            raise RuntimeError(f"No frames were extracted from {video_path}")
        return records

    @staticmethod
    def _frame_score(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4)

    @staticmethod
    def analyze_frame(frame: np.ndarray) -> dict[str, Any]:
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        cell_height = max(1, height // 3)
        cell_width = max(1, width // 3)
        composition_cells = []
        for row in range(3):
            for column in range(3):
                cell = edges[row * cell_height : (row + 1) * cell_height, column * cell_width : (column + 1) * cell_width]
                composition_cells.append(float(np.mean(cell)) if cell.size else 0.0)
        dominant_cell = int(np.argmax(composition_cells))
        regions = ["left", "center", "right"]
        verticals = ["upper", "middle", "lower"]
        most_detailed_region = f"{verticals[dominant_cell // 3]}-{regions[dominant_cell % 3]} third"

        sampled = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3)
        quantized = (sampled // 32) * 32 + 16
        colors, counts = np.unique(quantized, axis=0, return_counts=True)
        dominant_colors = colors[np.argsort(counts)[-3:][::-1]]
        palette = ["#" + "".join(f"{int(channel):02x}" for channel in color[::-1]) for color in dominant_colors]
        lighting = "bright" if brightness >= 170 else "dark" if brightness <= 75 else "mid-range"
        contrast_style = "high-contrast" if contrast >= 55 else "soft-contrast"
        detail_style = "high-edge-detail" if edge_density >= 0.12 else "low-edge-detail"
        saturation = float(np.mean(hsv[:, :, 1]))
        palette_style = "saturated" if saturation >= 110 else "muted"
        return {
            "analysis_backend": "opencv-pixel-analysis",
            "palette": palette,
            "palette_style": palette_style,
            "brightness_mean": round(brightness, 2),
            "contrast_stddev": round(contrast, 2),
            "lighting": lighting,
            "contrast_style": contrast_style,
            "detail_style": detail_style,
            "edge_density": round(edge_density, 5),
            "composition": f"edge detail concentrated in {most_detailed_region}",
            "framing": "landscape" if width >= height else "portrait",
            "character_appearance": None,
            "background_description": None,
            "semantic_analysis_status": "requires a vision-language model",
        }
