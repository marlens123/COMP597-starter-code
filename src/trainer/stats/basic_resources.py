import src.trainer.stats.base as base
import torch
import time
import logging
import psutil
import os
import subprocess
import src.config as config

logger = logging.getLogger(__name__)

trainer_stats_name="basic_resource_stats"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    return BasicResourcesStats()

class BasicResourcesStats(base.TrainerStats):
    """Stats class that tracks GPU utilization, memory consumption, and I/O."""


    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.process = psutil.Process(os.getpid())
        self.last_step_system = {}

    def _capture_system_metrics(self):
        """Capture CPU RAM, I/O, and GPU metrics using nvidia-smi."""
        ram_used = self.process.memory_info().rss / 1e6  # MB

        io_counters = self.process.io_counters()
        io_read = io_counters.read_bytes / 1e6
        io_write = io_counters.write_bytes / 1e6

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ]
            )
            util, mem = map(int, out.decode().strip().split(", "))
        except Exception:
            util, mem = 0, 0

        return {
            "ram_mb": ram_used,
            "io_read_mb": io_read,
            "io_write_mb": io_write,
            "gpu_util_percent": util,
            "gpu_mem_mb": mem,
        }

    def start_train(self) -> None:
        """Start training."""
        pass

    def stop_train(self) -> None:
        """Stop training."""
        pass

    def start_step(self) -> None:
        """Start a training step."""
        self._capture_system_metrics()

    def stop_step(self) -> None:
        """Stop a training step."""
        self.last_step_system = self._capture_system_metrics()

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
            f"RAM Δ {sys.get('ram_delta_mb', 0):.1f} MB -- "
            f"I/O R {sys.get('io_read_mb', 0):.1f} MB W {sys.get('io_write_mb', 0):.1f} MB -- "
            f"GPU util {sys.get('gpu_util_before', 0)}%→{sys.get('gpu_util_after', 0)}% -- "
            f"GPU mem {sys.get('gpu_mem_before_mb', 0):.1f}→{sys.get('gpu_mem_after_mb', 0):.1f} MB"
        )

    def log_stats(self) -> None:
        """Logs information about the data accumulated so far."""
        pass

