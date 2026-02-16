import src.trainer.stats.base as base
import src.config as config
import torch
import time
import logging
import pynvml
import psutil
import os
import csv
from pathlib import Path

logger = logging.getLogger(__name__)

trainer_stats_name="basic_resources_stats"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to basic resource trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    return BasicResourcesStats(device=device, csv_path=conf.trainer_stats_configs.basic_resources.output_dir)

pynvml.nvmlInit()

class BasicResourcesStats(base.TrainerStats):
    """Stats class that tracks GPU utilization, memory consumption, and I/O."""

    def __init__(self, device: torch.device, csv_path: str = ".", csv_name: str ="basic_resources_stats.csv") -> None:
        super().__init__()

        self.process = psutil.Process(os.getpid())        
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
            device.index if device.index is not None else 0
        )

        self.csv_path = Path(f"{csv_path}/{csv_name}")
        self.step_idx = 0
        self.csv_file = None
        self.csv_writer = None

    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get current GPU statistics."""
        stats = {
            "gpu_utilization": 0.0,
            "gpu_memory_used_mb": 0.0,
            "gpu_memory_total_mb": 0.0,
            "gpu_memory_percent": 0.0,
        }
        
        if torch.cuda.is_available():
            # PyTorch GPU memory
            stats["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / 1024**2
            stats["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
            stats["gpu_memory_percent"] = (stats["gpu_memory_used_mb"] / stats["gpu_memory_total_mb"]) * 100
            
            # NVML stats if available
            if self.gpu_handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    stats["gpu_utilization"] = util.gpu
                except:
                    pass
        
        return stats
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get system memory statistics."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "system_memory_used_mb": mem.used / 1024**2,
                "system_memory_total_mb": mem.total / 1024**2,
                "system_memory_percent": mem.percent,
            }
        except ImportError:
            return {
                "system_memory_used_mb": 0.0,
                "system_memory_total_mb": 0.0,
                "system_memory_percent": 0.0,
            }

    def _reset_system_counters(self):
        self.mem_before = None
        self.io_before = None
        self.gpu_mem_before = None
        self.gpu_util_before = None

    def _capture_before(self):
        self.mem_before = self.process.memory_info().rss
        self.io_before = self.process.io_counters()
        self.gpu_mem_before = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        self.gpu_util_before = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)

    def _capture_after(self):
        mem_after = self.process.memory_info().rss
        io_after = self.process.io_counters()
        gpu_mem_after = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        gpu_util_after = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        gpu_stats = self._get_gpu_stats()
        mem_stats = self._get_memory_stats()

        stats = {
            "ram_mb_abs": (mem_after) / 1e6,
            "io_read_mb_abs": (io_after.read_bytes) / 1e6,
            "io_write_mb_abs": (io_after.write_bytes) / 1e6,
            "ram_delta_mb": (mem_after - self.mem_before) / 1e6,
            "io_read_mb": (io_after.read_bytes - self.io_before.read_bytes) / 1e6,
            "io_write_mb": (io_after.write_bytes - self.io_before.write_bytes) / 1e6,
            "gpu_util_before": self.gpu_util_before.gpu,
            "gpu_util_after": gpu_util_after.gpu,
            "gpu_mem_before_mb": self.gpu_mem_before.used / 1e6,
            "gpu_mem_after_mb": gpu_mem_after.used / 1e6,
        }
        stats.update(gpu_stats)
        stats.update(mem_stats)

        return stats

    def start_train(self) -> None:
        """Initialize CSV logging."""
        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.csv_writer = None

    def stop_train(self) -> None:
        """Close CSV file."""
        if self.csv_file is not None:
            self.csv_file.close()

    def start_step(self) -> None:
        """Start a training step."""
        self._capture_before()

    def stop_step(self) -> None:
        """Stop a training step."""
        self.last_step_system = self._capture_after()

    def start_forward(self) -> None:
        """Start the forward pass."""
        pass

    def stop_forward(self) -> None:
        """Stop the forward pass."""
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        """Logs the loss of the current step by passing it to the stats."""
        pass

    def start_backward(self) -> None:
        """Start the backward pass."""
        pass

    def stop_backward(self) -> None:
        """Stop the backward pass"""
        pass

    def start_optimizer_step(self) -> None:
        """Start the optimizer step."""
        pass

    def stop_optimizer_step(self) -> None:
        """Stop the optimizer step."""
        pass

    def start_save_checkpoint(self) -> None:
        """Start checkpointing."""
        pass

    def stop_save_checkpoint(self) -> None:
        """Stop checkpointing."""
        pass

    def log_step(self) -> None:
        """Logs information about the previous step."""
        sys = getattr(self, "last_step_system", {})

        print(
            f"RAM abs {sys.get('ram_mb_abs', 0):.1f} MB -- "
            f"I/O R abs {sys.get('io_read_mb_abs', 0):.1f} MB W {sys.get('io_write_mb_abs', 0):.1f} MB -- "
            f"RAM Δ {sys.get('ram_delta_mb', 0):.1f} MB -- "
            f"I/O R {sys.get('io_read_mb', 0):.1f} MB W {sys.get('io_write_mb', 0):.1f} MB -- "
            f"GPU util {sys.get('gpu_util_after', 0)}% -- "
            f"GPU mem {sys.get('gpu_mem_after_mb', 0):.1f} MB"
        )

        row = {
            "step": self.step_idx,
            "ram_mb_abs": sys.get("ram_mb_abs", 0),
            "io_read_mb_abs": sys.get("io_read_mb_abs", 0),
            "io_write_mb_abs": sys.get("io_write_mb_abs", 0),
            "ram_delta_mb": sys.get("ram_delta_mb", 0),
            "io_read_mb": sys.get("io_read_mb", 0),
            "io_write_mb": sys.get("io_write_mb", 0),
            "gpu_util_after": sys.get("gpu_util_after", 0),
            "gpu_mem_after_mb": sys.get("gpu_mem_after_mb", 0),
            "gpu_util_stephanie": sys.get("gpu_utilization", 0.0),
            "gpu_mem_used_stephanie": sys.get("gpu_memory_used_mb", 0.0),
            "gpu_mem_total_stephanie": sys.get("gpu_memory_total_mb", 0.0),
            "gpu_mem_percent_stephanie": sys.get("gpu_memory_percent", 0.0),
            "mem_used_stephanie": sys.get("system_memory_used_mb", 0.0),
            "mem_total_stephanie": sys.get("system_memory_total_mb", 0.0),
            "mem_percent_stephanie": sys.get("system_memory_percent", 0.0),
        }

        # Initialize writer with header after first row
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=row.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow(row)
        self.csv_file.flush()  # important for long runs

        self.step_idx += 1

    def log_stats(self) -> None:
        """Logs information about the data accumulated so far."""
        pass

