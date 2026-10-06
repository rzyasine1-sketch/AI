# Kaggle GPU First Run

## One-time manual setup

1. Publish the current project source and this notebook to the GitHub branch configured in the notebook. The checked `origin/main` currently contains only JSON context files; Kaggle must not run that stale branch.
2. In Kaggle, create a Notebook and enable **Internet** plus a **T4x2 or better GPU** accelerator.
3. Set `CHANNEL_URL` in the notebook settings cell.
4. If YouTube requires authentication, create a Kaggle Secret named `YOUTUBE_COOKIES` containing Netscape-format cookies. The notebook writes it only to a mode-0600 temporary file under `/tmp`, then deletes it. Do not place cookies in Git, notebook cells, outputs, or Kaggle input datasets.
5. Run cells top to bottom. Default first run: 3 videos, 8 frames per video, 1 analysis cycle. The 20-video/15-cycle mode is configurable only after the first run is confirmed.

Once the environment, project source, channel URL, and optional cookies are in place, the equivalent single command is:

```bash
python3 run.py --overnight --channel "$CHANNEL_URL" --target-videos 3 --frames-per-video 8 --cycles 1
```

The notebook checks available VRAM before selecting a Qwen3-VL model (2B on T4-class devices, 4B only when one GPU has at least 24 GiB), starts Ollama, pulls models sequentially, and verifies each installed model/capability. Agent calls unload their text model after use. The vision model stays loaded for its frame batch and is explicitly unloaded before the text-agent sequence.

Downloaded video files live under a temporary directory and are removed after frame extraction. Selected frame images, JSON metadata, memory, and stage checkpoints are retained. Checkpoints are written beneath `checkpoints/overnight/<channel-id>/`; preserve Kaggle working outputs/checkpoints when restarting a session. Resume verifies frame files before skipping completed analysis.

## Training gate

The first-stage target is LoRA on a pretrained Stable Diffusion 1.5 base model, never training from scratch. Training starts only with at least 100 image-backed samples, approved style rules, mean VLM style confidence >= 0.75, valid dataset paths, and a real GPU. The initial 3x8 run is expected to skip training. Diffusers creates a checkpoint at each configured epoch boundary and resumes from `latest` when available. Base-model access may require accepting the model license and setting a Kaggle Secret named `HF_TOKEN`.

## BLOCKED conditions

The notebook stops with `BLOCKED` if Git is stale, Internet is unavailable, GPU/VRAM/disk is insufficient, model installation/capability verification fails, YouTube denies downloads, agent models are missing, dataset paths fail validation, or training fails after passing its gate. It never switches to mock. Mock tests are run as a separate validation cell before the real pipeline.
