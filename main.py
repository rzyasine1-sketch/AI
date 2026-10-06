import argparse
import json

from pipeline.local_vision import run_local_vision_pipeline
from pipeline.real_channel import run_channel_pipeline
from pipeline.supervisor import Supervisor


def build_parser():
    parser = argparse.ArgumentParser(description="Visual style learning pipeline")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="Use mock agents for pipeline-only testing")
    mode.add_argument("--real", action="store_true", help="Require configured real Ollama models; never fall back to mock")
    parser.add_argument("--channel", help="YouTube channel URL; runs real inventory, sampling, download, and frame analysis")
    parser.add_argument("--frames", nargs="+", help="One or more extracted local frame image paths for visual analysis")
    parser.add_argument("--cookies", help="Path to a local Netscape-format YouTube cookies file when authentication is required")
    parser.add_argument("--target-videos", type=int, default=20, help="Maximum videos to sample and download from a channel")
    parser.add_argument("--frames-per-video", type=int, default=8, help="Maximum frames to extract from each downloaded video")
    parser.add_argument("--inventory-limit", type=int, default=500, help="Maximum channel entries to inventory without downloading")
    parser.add_argument("--cycles", type=int, default=None, help="Maximum number of cycles (defaults to 1 for a channel, 15 for the mock/agent loop)")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint")
    parser.add_argument("--status", action="store_true", help="Display pipeline status")
    parser.add_argument("--validate", action="store_true", help="Validate the current configuration and setup")
    parser.add_argument("--export-training", action="store_true", help="Export training dataset")
    parser.add_argument("--no-cache", action="store_true", help="Disable the application cache")
    return parser


def _backend_info(supervisor, mock_mode: bool) -> dict:
    if mock_mode:
        return {
            "agent_mode": "mock",
            "agent_backends": {name: "mock" for name in ("deepseek", "qwen", "llama")},
        }
    models = supervisor.config.get("models", {})
    return {
        "agent_mode": "real",
        "agent_backends": {
            name: {"backend": "ollama", "model": models.get(name, {}).get("name", name)}
            for name in ("deepseek", "qwen", "llama")
        },
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.target_videos < 1 or args.frames_per_video < 1 or args.inventory_limit < 1:
        parser.error("--target-videos, --frames-per-video, and --inventory-limit must be positive")

    if args.frames:
        if args.channel:
            parser.error("--frames and --channel are separate ingestion paths and cannot be combined")
        if not args.mock and not args.real:
            parser.error("--frames requires an explicit --mock or --real mode")
        try:
            result = run_local_vision_pipeline(args.frames, mock_mode=args.mock)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(json.dumps({"status": "blocked", "mock_used": args.mock, "blocked_by": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "mock_complete" or result["status"] == "real_complete" else 2

    if args.channel:
        if args.mock:
            parser.error("--channel always performs real video ingestion and cannot be combined with --mock")
        try:
            result = run_channel_pipeline(
                args.channel,
                target_videos=args.target_videos,
                frames_per_video=args.frames_per_video,
                inventory_limit=args.inventory_limit,
                cycles=args.cycles or 1,
                resume=args.resume,
                cookies_path=args.cookies,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(json.dumps({"status": "blocked", "blocked_by": f"{type(exc).__name__}: {exc}"}, indent=2))
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["downloaded_and_analyzed_count"] else 2

    if not args.mock and not args.real:
        parser.error("choose --mock or --real, or provide --channel for real video ingestion")

    supervisor = Supervisor(mock_mode=args.mock)

    if args.status:
        print(json.dumps({**supervisor.status(), **_backend_info(supervisor, args.mock)}, indent=2, sort_keys=True))
        return 0

    if args.validate:
        print(json.dumps({**supervisor.validate(), **_backend_info(supervisor, args.mock)}, indent=2, sort_keys=True))
        return 0

    if args.export_training:
        export_path = supervisor.export_training()
        print(json.dumps({"path": str(export_path)}, indent=2, sort_keys=True))
        return 0

    try:
        result = supervisor.run(max_cycles=args.cycles, resume=args.resume, no_cache=args.no_cache)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", **_backend_info(supervisor, args.mock), "blocked_by": str(exc)}, indent=2))
        return 2
    print(json.dumps({**result, **_backend_info(supervisor, args.mock)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
