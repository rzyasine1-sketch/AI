from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class YouTubeIngestion:
    def __init__(self, data_dir: str | Path = "data", cookies_path: str | Path | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cookies_path = Path(cookies_path) if cookies_path else None

    def _cookie_options(self) -> dict[str, str]:
        if self.cookies_path is None:
            return {}
        if not self.cookies_path.is_file():
            raise FileNotFoundError(f"YouTube cookies file does not exist: {self.cookies_path}")
        return {"cookiefile": str(self.cookies_path)}

    def _extract_channel_metadata(self, channel_url: str, max_entries: int | None = None) -> dict[str, Any]:
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError("yt-dlp is required for real YouTube ingestion; install requirements.txt") from exc
        options: dict[str, Any] = {
            "extract_flat": "in_playlist",
            "skip_download": True,
            "ignoreerrors": False,
            "quiet": True,
            "no_warnings": True,
        }
        options.update(self._cookie_options())
        if max_entries is not None:
            options["playlistend"] = max_entries
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                metadata = downloader.extract_info(channel_url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"YouTube channel inventory failed: {exc}") from exc
        if not isinstance(metadata, dict):
            raise TypeError("YouTube did not return channel metadata")
        return metadata

    def inventory(self, channel_url: str, max_entries: int | None = None) -> dict[str, Any]:
        metadata = self._extract_channel_metadata(channel_url, max_entries=max_entries)
        videos = []
        for position, entry in enumerate(metadata.get("entries") or []):
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            video_id = str(entry["id"])
            videos.append(
                {
                    "id": video_id,
                    "title": entry.get("title") or "",
                    "url": entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
                    "playlist_position": position,
                    "view_count": entry.get("view_count"),
                    "duration": entry.get("duration"),
                    "timestamp": entry.get("timestamp"),
                    "upload_date": entry.get("upload_date"),
                    "live_status": entry.get("live_status"),
                    "availability": entry.get("availability"),
                }
            )
        result = {
            "channel_url": channel_url,
            "channel_id": metadata.get("channel_id") or metadata.get("id"),
            "channel_title": metadata.get("channel") or metadata.get("title"),
            "inventory_limit": max_entries,
            "videos": videos,
        }
        (self.data_dir / "channel_inventory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    @staticmethod
    def select_videos(inventory: list[dict[str, Any]], target_videos: int = 20) -> list[dict[str, Any]]:
        if target_videos < 1:
            raise ValueError("target_videos must be at least 1")
        eligible = [
            video for video in inventory
            if video.get("id")
            and video.get("live_status") not in {"is_live", "is_upcoming"}
            and (video.get("duration") is None or 0 < video["duration"] <= 1800)
        ]
        if not eligible:
            return []
        chronological = sorted(eligible, key=lambda video: int(video.get("playlist_position", 0)))
        temporal_candidates = [
            (chronological[0], "recent"),
            (chronological[len(chronological) // 2], "middle"),
            (chronological[-1], "old"),
        ]
        ranked = sorted(eligible, key=lambda video: int(video.get("view_count") or 0), reverse=True)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def add(video: dict[str, Any], reason: str) -> bool:
            video_id = str(video["id"])
            if video_id in selected_ids:
                return False
            selected.append({**video, "selection_reason": reason})
            selected_ids.add(video_id)
            return len(selected) >= target_videos

        for video, reason in temporal_candidates:
            if add(video, reason):
                return selected

        content_types = {
            "interview": ("interview", "conversation", "crew"),
            "educational": ("how to", "explained", "learn", "why"),
            "event": ("live", "launch", "recap", "mission"),
            "trailer": ("trailer", "teaser", "preview"),
        }
        seen_types: set[str] = set()
        for video in chronological:
            title = str(video.get("title", "")).lower()
            kind = next((name for name, terms in content_types.items() if any(term in title for term in terms)), None)
            if kind is None or kind in seen_types or str(video["id"]) in selected_ids:
                continue
            seen_types.add(kind)
            if add(video, f"content_type:{kind}"):
                return selected
        for video in ranked:
            if add(video, "high_performance"):
                return selected
        return selected

    def select_and_save(self, inventory: dict[str, Any], target_videos: int = 20) -> list[dict[str, Any]]:
        selected = self.select_videos(inventory.get("videos", []), target_videos=target_videos)
        selection = {
            "channel_id": inventory.get("channel_id"),
            "target_videos": target_videos,
            "selected_count": len(selected),
            "selection_method": "chronological_strata + views + title-derived content type",
            "timeline_basis": "playlist position; upload timestamps were unavailable in the flat inventory",
            "videos": selected,
        }
        (self.data_dir / "selected_videos.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        return selected

    def download_video(self, video: dict[str, Any], videos_dir: str | Path = "data/videos") -> Path:
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError("yt-dlp is required to download videos; install requirements.txt") from exc
        video_id = str(video["id"])
        output_dir = Path(videos_dir) / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        url = video.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        options: dict[str, Any] = {
            "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
            "outtmpl": str(output_dir / "video.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "max_filesize": 250_000_000,
        }
        options.update(self._cookie_options())
        try:
            import imageio_ffmpeg

            options["ffmpeg_location"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                result = downloader.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"YouTube video download failed for {video_id}: {exc}") from exc
        requested = result.get("requested_downloads") or []
        paths = [Path(item["filepath"]) for item in requested if item.get("filepath")]
        paths.extend(sorted(output_dir.glob("video.*")))
        path = next((item for item in paths if item.is_file() and item.stat().st_size > 0), None)
        if path is None:
            raise RuntimeError(f"yt-dlp did not produce a non-empty video file for {video_id}")
        return path