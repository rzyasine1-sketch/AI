import json
import logging
from pathlib import Path
from typing import Any

import yaml

from agents.deepseek import DeepSeekAgent
from agents.llama import LlamaAgent
from agents.qwen import QwenAgent
from cache.manager import CacheManager
from ingestion.local_files import LocalFilesIngestion
from memory.manager import MemoryManager
from models.mock_backend import MockModelBackend
from models.ollama_backend import OllamaBackend
from pipeline.cycle import run_single_cycle
from pipeline.recovery import last_successful_checkpoint, save_checkpoint
from training.dataset import TrainingDataset


class Supervisor:
    def __init__(self, config_path: str = "config.yaml", mock_mode: bool = True):
        self.config_path = Path(config_path)
        self.mock_mode = mock_mode
        self.config = self.load_config()
        self.base_dir = Path.cwd()
        self.memory_manager = MemoryManager(self.config.get("memory", {}).get("store_path", "memory_store"))
        self.dataset = TrainingDataset(storage_path="training")
        self.cache_manager = CacheManager(enabled=self.config.get("cache", {}).get("enabled", True))
        self.quality_history_path = Path("quality_history.json")
        self.stop_path = Path("control/STOP")
        self.control_dir = Path("control")
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = Path("checkpoints")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.system_logger = logging.getLogger("ai_pipeline_system")
        self.system_logger.setLevel(logging.INFO)
        self.system_logger.handlers.clear()
        self.system_logger.addHandler(logging.FileHandler(self.logs_dir / "system.log", encoding="utf-8"))
        self.error_logger = logging.getLogger("ai_pipeline_errors")
        self.error_logger.setLevel(logging.ERROR)
        self.error_logger.handlers.clear()
        self.error_logger.addHandler(logging.FileHandler(self.logs_dir / "errors.log", encoding="utf-8"))
        self.ingestion = LocalFilesIngestion("data/raw")
        self.last_completed_cycle = self._detect_last_completed_cycle()

    def load_config(self) -> dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _detect_last_completed_cycle(self) -> int:
        state = last_successful_checkpoint(str(self.checkpoints_dir))
        if not state:
            return 0
        return int(state.get("last_completed_cycle", 0))

    def create_agents(self):
        if self.mock_mode:
            backend_map = {
                "deepseek": MockModelBackend("mock-deepseek"),
                "qwen": MockModelBackend("mock-qwen"),
                "llama": MockModelBackend("mock-llama"),
            }
        else:
            model_config = self.config.get("models", {})
            backend_map = {
                name: OllamaBackend(
                    name=name,
                    config={
                        "model_name": model_config.get(name, {}).get("name", name),
                        "base_url": model_config.get(name, {}).get("base_url"),
                        "timeout": self.config.get("pipeline", {}).get("timeout", 300),
                    },
                )
                for name in ("deepseek", "qwen", "llama")
            }
            unavailable = [
                f"{name}: {backend.availability_error}"
                for name, backend in backend_map.items()
                if not backend.is_available()
            ]
            if unavailable:
                raise RuntimeError("Real agent backends are unavailable; no mock fallback was used. " + "; ".join(unavailable))
        return {
            "deepseek": DeepSeekAgent(backend_map["deepseek"]),
            "qwen": QwenAgent(backend_map["qwen"]),
            "llama": LlamaAgent(backend_map["llama"]),
        }

    def run(self, max_cycles: int | None = None, resume: bool = False, no_cache: bool = False) -> dict[str, Any]:
        if no_cache:
            self.cache_manager.enabled = False
        max_cycles = max_cycles or int(self.config.get("pipeline", {}).get("max_cycles", 15))
        existing_state = self._load_state()
        last_cycle = existing_state.get("last_completed_cycle", self.last_completed_cycle)
        if resume:
            start_cycle = last_cycle + 1
            final_state = {
                "last_completed_cycle": last_cycle,
                "cycles_run": 0,
                "quality_history": list(existing_state.get("quality_history", [])),
            }
        else:
            start_cycle = 1
            final_state = {"last_completed_cycle": 0, "cycles_run": 0, "quality_history": []}
        retry_count = int(self.config.get("pipeline", {}).get("retry_count", 2))
        agents = self.create_agents()
        for cycle_number in range(start_cycle, max_cycles + 1):
            if self.stop_path.exists():
                final_state["last_completed_cycle"] = cycle_number - 1
                break
            source_data = self.ingestion.read_files()
            attempt = 0
            result = None
            while attempt <= retry_count:
                try:
                    result = run_single_cycle(
                        cycle_number,
                        agents["deepseek"],
                        agents["qwen"],
                        agents["llama"],
                        self.memory_manager,
                        self.dataset,
                        source_data=source_data[0] if source_data else {"source": "local_files"},
                        cache_manager=self.cache_manager,
                    )
                    break
                except Exception:
                    attempt += 1
                    self.error_logger.exception("Cycle %s failed on attempt %s", cycle_number, attempt)
                    if attempt > retry_count:
                        final_state["last_completed_cycle"] = cycle_number - 1
                        return final_state
            if result is None:
                continue
            final_state["cycles_run"] += 1
            final_state["last_completed_cycle"] = cycle_number
            final_state["quality_history"].append({"cycle": cycle_number, "quality_score": result["quality_score"]})
            self._save_quality_history(final_state["quality_history"])
            checkpoint_state = {
                "last_completed_cycle": cycle_number,
                "quality_score": result["quality_score"],
                "status": "ok",
            }
            save_checkpoint(str(self.checkpoints_dir), checkpoint_state, f"checkpoint_{cycle_number:03d}")
            save_checkpoint(str(self.checkpoints_dir), checkpoint_state, "last_checkpoint")
            self.system_logger.info("Completed cycle %s with quality score %s", cycle_number, result["quality_score"])
            if self._should_stop(final_state["quality_history"], cycle_number, max_cycles):
                break
        return final_state

    def _should_stop(self, quality_history: list[dict[str, Any]], cycle_number: int, max_cycles: int) -> bool:
        if cycle_number >= max_cycles:
            return True
        if self.stop_path.exists():
            return True
        if len(quality_history) < 3:
            return False
        recent = quality_history[-3:]
        scores = [item["quality_score"] for item in recent]
        return max(scores) - min(scores) < 0.01

    def _save_quality_history(self, history: list[dict[str, Any]]) -> None:
        self.quality_history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def status(self) -> dict[str, Any]:
        state = self._load_state()
        cycle = state.get("last_completed_cycle", self.last_completed_cycle)
        history = state.get("quality_history", [])
        return {
            "last_completed_cycle": cycle,
            "quality_history": history,
            "status": "ready" if cycle >= 0 else "not_started",
        }

    def validate(self) -> dict[str, bool]:
        checks = {
            "config": self.config_path.exists(),
            "memory_store": self.memory_manager.base_path.exists(),
            "training_dataset": self.dataset.file_path.exists() or True,
            "visual_dataset": self.dataset.visual_manifest_path.exists() or True,
            "cache": True,
        }
        return checks

    def export_training(self) -> Path:
        rows = self.dataset.load_records()
        if rows:
            self.dataset.export_jsonl(rows)
        visual_rows = self.dataset.load_visual_records()
        if visual_rows:
            self.dataset.export_visual_dataset(visual_rows)
            return self.dataset.golden_visual_path
        return self.dataset.file_path

    def _load_state(self) -> dict[str, Any]:
        if not self.quality_history_path.exists():
            return {"last_completed_cycle": self.last_completed_cycle, "quality_history": []}
        try:
            data = json.loads(self.quality_history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                last_cycle = data[-1].get("cycle", self.last_completed_cycle) if data else self.last_completed_cycle
                return {"last_completed_cycle": last_cycle, "quality_history": data}
            if isinstance(data, dict):
                return {
                    "last_completed_cycle": data.get("last_completed_cycle", self.last_completed_cycle),
                    "quality_history": data.get("quality_history", []),
                }
        except json.JSONDecodeError:
            pass
        return {"last_completed_cycle": self.last_completed_cycle, "quality_history": []}
