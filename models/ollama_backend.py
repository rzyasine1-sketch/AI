import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from models.base import ModelBackend


class OllamaBackend(ModelBackend):
    def __init__(self, name: str = "ollama", config: dict[str, Any] | None = None):
        super().__init__(name=name, config=config or {})
        self.base_url = str(self.config.get("base_url") or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434")
        if "://" not in self.base_url:
            self.base_url = f"http://{self.base_url}"
        self.model_name = str(self.config.get("model_name") or name)
        self.availability_error = "Ollama runtime has not been checked"

    def is_available(self) -> bool:
        try:
            with urlopen(f"{self.base_url.rstrip('/')}/api/tags", timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            installed = {item.get("name") for item in payload.get("models", [])}
            installed.update(item.get("model") for item in payload.get("models", []))
            if self.model_name not in installed:
                self.availability_error = f"Model {self.model_name!r} is not installed in Ollama at {self.base_url}"
                return False
            self.availability_error = "available"
            return True
        except (OSError, URLError, json.JSONDecodeError) as exc:
            self.availability_error = f"Ollama runtime unavailable at {self.base_url}: {exc}"
            return False

    def generate(self, agent_name: str, prompt: str, system: str | None = None, **kwargs) -> Any:
        if not self.is_available():
            raise RuntimeError(self.availability_error)
        context = {key: value for key, value in kwargs.items() if key not in {"cycle"}}
        full_prompt = (
            f"{prompt}\nAnalyze visual STYLE, not merely events. Return one JSON object only.\n"
            f"Evidence and context:\n{json.dumps(context, ensure_ascii=True, default=str)}"
        )
        body = {
            "model": self.model_name,
            "prompt": full_prompt,
            "system": system or "Return schema-compatible JSON. Do not claim visual details unsupported by the provided evidence.",
            "format": "json",
            "stream": False,
            "keep_alive": self.config.get("keep_alive", 0),
            "options": {"temperature": 0.2},
        }
        request = Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=float(self.config.get("timeout", 300))) as response:
                result = json.loads(response.read().decode("utf-8"))
            generated = result.get("response")
            if not isinstance(generated, str):
                raise TypeError(f"Ollama returned no text from model {self.model_name!r}")
            payload = json.loads(generated)
            if not isinstance(payload, dict):
                raise TypeError(f"Ollama model {self.model_name!r} returned non-object JSON")
            return payload
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama generation failed for {self.model_name!r}: {exc}") from exc
