# Visual Style Learning Pipeline

This project analyzes real channel videos to learn visual style and build an image-backed dataset for a future visual-generation model. It does not generate audio, train audio models, or train the target model's weights.

## Install

```bash
python3 -m pip install -r requirements.txt
```

The video workflow uses yt-dlp and OpenCV. yt-dlp supplies FFmpeg through imageio-ffmpeg when format merging is required.

## Kaggle GPU Overnight Run

The Kaggle setup and first-run notebook are in `kaggle/`. Before leaving a GPU session unattended:

1. Publish the current project source and notebook to the configured GitHub branch. The checked `origin/main` currently contains only five JSON context files, not this application source, so a Kaggle clone will intentionally stop until the source is published.
2. Create a Kaggle Notebook, enable Internet and a GPU accelerator (T4x2 or better), and add an optional Kaggle Secret named `YOUTUBE_COOKIES` containing Netscape-format cookies if the channel requires authentication.
3. Set `CHANNEL_URL` in the notebook and run its cells from top to bottom.

The first run defaults to 3 videos, 8 frames per video, and 1 sequential analysis cycle. It inspects actual VRAM before selecting the configured small Qwen3-VL model, installs Ollama, checks every model, and never switches to mocks. Video downloads live in temporary storage and are removed after frame extraction. Inventory, frame-analysis, agent-cycle, memory, dataset, and LoRA-epoch checkpoints are kept separately.

LoRA fine-tuning uses a Stable Diffusion 1.5 base model; it is gated behind at least 100 image-backed records with non-empty approved visual rules and mean VLM style confidence of 0.75. The initial 3x8 trial should normally skip training. Nothing trains a model from scratch.

Kaggle requires manual GPU/Internet enablement, GitHub source publication, a channel URL, and optionally the YouTube cookies Secret (and Hugging Face authorization if the selected base model requires it). See `kaggle/README.md` for the manual setup and BLOCKED conditions.

## Run a channel

```bash
python3 main.py --channel "https://www.youtube.com/@CHANNEL/videos"
```

The default is at most 20 selected videos from a metadata inventory capped at 500 entries, with at most 8 frames extracted per video. It does not download every channel video. For a small trial:

```bash
python3 main.py --channel "https://www.youtube.com/@CHANNEL/videos" --target-videos 3 --frames-per-video 4
```

Inventory, selection, downloads, extracted JPEGs, and the real run report are written beneath `data/`:

- `data/channel_inventory.json`
- `data/selected_videos.json`
- `data/videos/<video_id>/video.*`
- `data/videos/<video_id>/frames/`
- `data/real_run_report.json`

The real image-backed dataset is isolated at `training/real_visual/golden_visual_dataset.jsonl`; its image paths resolve under `data/`. Pixel-only observations are marked provisional. Character identity, semantic backgrounds, and camera intent are not guessed without a vision-language model.

If YouTube requires authentication, provide a local Netscape-format cookie file:

```bash
python3 main.py --channel "https://www.youtube.com/@CHANNEL/videos" --target-videos 3 --cookies /path/to/youtube-cookies.txt
```

Keep that cookie file private. Cookie contents are passed only to yt-dlp and are not sent to agents or stored in run reports.

## Agent backends

Use `--real` to require the configured Ollama models. Real mode never falls back to mock:

```bash
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5:0.5b
ollama pull llama3.2:1b
python3 main.py --real --cycles 3
```

A channel command also uses real mode for agents; if the runtime or models are unavailable, the report says so. When downloads work but agents do not, OpenCV pixel measurements can still be recorded, explicitly marked provisional and not semantically validated.

Use `--mock` only for the existing pipeline architecture tests; it cannot be combined with `--channel`:

```bash
python3 main.py --mock --cycles 3
python3 main.py --mock --cycles 3 --resume
```

## Local Vision Analyzer

Analyze already-extracted local frame images without YouTube:

```bash
python3 main.py --mock --frames data/videos/test_video/frames/frame_000000.jpg data/videos/test_video/frames/frame_041706.jpg
```

This runs the frame images through the explicitly labelled mock vision backend and mock DeepSeek/Qwen/Llama agents. Outputs go to `training/vision_mock/` and `memory_store/visual_knowledge_mock.json`. Mock results are not real VLM analysis.

For genuine image understanding, run:

```bash
python3 main.py --real --frames data/videos/test_video/frames/frame_000000.jpg
```

Real vision mode requires an already-running local Ollama runtime and an already-installed vision-capable model configured under `vision:` in `config.yaml` (default: `qwen3-vl:2b`). The pipeline checks Ollama's installed models and vision capability, then sends the image bytes to the model. It does not download models and never falls back to mock. If the VLM is unavailable, it writes a `blocked` report and no real dataset.

Each visual dataset row keeps `raw_observation`, `visual_analysis`, and Qwen-reviewed/Llama-approved `validated_style_rules` in separate fields. A mock run keeps the same contract but clearly marks its backend and confidence as mock.

## Other commands

```bash
python3 main.py --mock --status
python3 main.py --mock --validate
python3 main.py --mock --export-training
python3 -m pytest -q
```

## Visual evidence and scope

The local analyzer measures actual decoded images for dominant colors, brightness, contrast, edge distribution, and frame-to-frame appearance changes. These pixel measurements do not identify people, objects, or scene meaning. DeepSeek, Qwen, and Llama are prompted to focus on visual style rules and distinguish evidence from interpretation.

Audio generation and audio-model training are outside this project's scope. No audio dataset is created. Transcripts are not required by the visual pipeline.
