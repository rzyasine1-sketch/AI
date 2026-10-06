import os
from typing import Any

import psutil


class ResourceManager:
    def __init__(self, max_ram_percent: int = 80, max_cpu_percent: int = 85, max_disk_percent: int = 90):
        self.max_ram_percent = max_ram_percent
        self.max_cpu_percent = max_cpu_percent
        self.max_disk_percent = max_disk_percent

    def snapshot(self) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=None)
        process_state = {
            'pid': os.getpid(),
            'rss_mb': round(psutil.Process().memory_info().rss / (1024 * 1024), 2),
            'cpu_percent': psutil.Process().cpu_percent(interval=None),
        }
        return {
            'ram_percent': round(memory.percent, 2),
            'cpu_percent': round(cpu_percent, 2),
            'disk_percent': round((disk.used / disk.total) * 100, 2),
            'process_state': process_state,
        }

    def ensure_limits(self) -> bool:
        snapshot = self.snapshot()
        if snapshot['ram_percent'] > self.max_ram_percent:
            return False
        if snapshot['cpu_percent'] > self.max_cpu_percent:
            return False
        return not snapshot['disk_percent'] > self.max_disk_percent
