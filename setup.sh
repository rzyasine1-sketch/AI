#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 - <<'PY'
import importlib.util
import json
import shutil

required = ["cv2", "numpy", "yaml", "jsonschema", "psutil", "yt_dlp", "imageio_ffmpeg"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
result = {
    "python_dependencies": "blocked" if missing else "ready",
    "missing_modules": missing,
    "ffmpeg": shutil.which("ffmpeg"),
    "ollama": shutil.which("ollama"),
}
print(json.dumps(result, indent=2))
if missing:
    raise SystemExit("BLOCKED: project dependencies are missing")
PY

if [[ "${INSTALL_OLLAMA:-0}" == "1" ]] && ! command -v ollama >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || { printf '%s\n' 'BLOCKED: curl is required to install Ollama'; exit 2; }
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    curl -fsSL https://ollama.com/install.sh -o "$installer"
    sh "$installer"
fi