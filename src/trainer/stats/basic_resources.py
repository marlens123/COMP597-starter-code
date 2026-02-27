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

class BasicResourcesStats(base.TrainerStats):
    """Stats class that tracks GPU utilization, memory consumption, and I/O."""

    def __init__(self, device: torch.device, csv_path: str, csv_name: str ="basic_resources_stats") -> None:
        super().__init__()

        self.process = psutil.Process(os.getpid())   

        if torch.cuda.is_available():
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                device.index if device.index is not None else torch.cuda.current_device()
            )
        else:
            self.gpu_handle = None

        self.nvml_log_interval = 15

        self._nvml_accumulator = {
            "gpu_util": [],
        }

        Path(csv_path).mkdir(parents=True, exist_ok=True)

        self.step_csv_path = Path(f"{csv_path}/{csv_name}_{int(time.time())}.csv")
        self.substeps_csv_path = Path(f"{csv_path}/{csv_name}_substeps_{int(time.time())}.csv")
        self.step_idx = 0
        self.step_csv_file = None
        self.step_csv_writer = None
        self.substep_csv_file = None
        self.substep_csv_writer = None

        self.batch_size = None
        self.world_size = 1
        self.samples_processed = 0

    def _capture_before_step(self):
        self.mem_before_step = self.process.memory_info().rss
        self.io_before_step = self.process.io_counters()
        self.cpu_times_before_step = self.process.cpu_times()
        self.time_before_step = time.time()

    def _capture_before_substep(self):
        self.mem_before_substep = self.process.memory_info().rss
        self.io_before_substep = self.process.io_counters()
        self.cpu_times_before_substep = self.process.cpu_times()
        self.time_before_substep = time.time()

    def _capture_after_step(self):
        mem_after = self.process.memory_info().rss
        io_after = self.process.io_counters()

        if self.gpu_handle is not None:
            gpu_util_after = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        else:
            gpu_util_after = None

        if torch.cuda.is_available():
            # PyTorch GPU memory
            gpu_memory_used_mb = torch.cuda.memory_allocated() / 1024**2
        else:
            gpu_memory_used_mb = 0

        time_after = time.time()

        cpu_times_after = self.process.cpu_times()

        cpu_time_delta = (
            (cpu_times_after.user - self.cpu_times_before_step.user)
            + (cpu_times_after.system - self.cpu_times_before_step.system)
        )

        wall_time_delta = time_after - self.time_before_step

        cpu_util_percent = 100 * cpu_time_delta / wall_time_delta


        stats = {
            "time_sec": time_after - self.time_before_step,
            "cpu_util_percent": cpu_util_percent,
            "ram_mb_abs": (mem_after) / 1024**2,
            "ram_delta_mb": (mem_after - self.mem_before_step) / 1024**2,
            "io_read_mb_abs": (io_after.read_bytes) / 1e6,
            "io_write_mb_abs": (io_after.write_bytes) / 1e6,
            "io_read_mb": (io_after.read_bytes - self.io_before_step.read_bytes) / 1e6,
            "io_write_mb": (io_after.write_bytes - self.io_before_step.write_bytes) / 1e6,
            "gpu_util_moment": gpu_util_after.gpu,
            "gpu_memory_used_mb": gpu_memory_used_mb,
        }

        return stats
    
    def _capture_after_substep(self):
        mem_after = self.process.memory_info().rss
        io_after = self.process.io_counters()

        if self.gpu_handle is not None:
            gpu_util_after = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        else:
            gpu_util_after = None

        if torch.cuda.is_available():
            # PyTorch GPU memory
            gpu_memory_used_mb = torch.cuda.memory_allocated() / 1024**2
        else:
            gpu_memory_used_mb = 0

        time_after = time.time()

        cpu_times_after = self.process.cpu_times()

        cpu_time_delta = (
            (cpu_times_after.user - self.cpu_times_before_substep.user)
            + (cpu_times_after.system - self.cpu_times_before_substep.system)
        )

        wall_time_delta = time_after - self.time_before_substep

        cpu_util_percent = 100 * cpu_time_delta / wall_time_delta

        stats = {
            "time_sec": time_after - self.time_before_substep,
            "cpu_util_percent": cpu_util_percent,
            "ram_mb_abs": (mem_after) / 1e6,
            "ram_delta_mb": (mem_after - self.mem_before_substep) / 1e6,
            "io_read_mb_abs": (io_after.read_bytes) / 1e6,
            "io_write_mb_abs": (io_after.write_bytes) / 1e6,
            "io_read_mb": (io_after.read_bytes - self.io_before_substep.read_bytes) / 1e6,
            "io_write_mb": (io_after.write_bytes - self.io_before_substep.write_bytes) / 1e6,
            "gpu_util_moment": gpu_util_after.gpu,
            "gpu_memory_used_mb": gpu_memory_used_mb,
        }

        return stats

    def start_train(self) -> None:
        """Initialize CSV logging."""
        self.training_time_start = time.time()

        self.step_csv_file = open(self.step_csv_path, mode="w", newline="")
        self.step_csv_writer = None
        self.substep_csv_file = open(self.substeps_csv_path, mode="w", newline="")
        self.substep_csv_writer = None

    def stop_train(self) -> None:
        """Close CSV file."""
        if self.step_csv_file is not None:
            self.step_csv_file.close()
        if self.substep_csv_file is not None:
            self.substep_csv_file.close()

        if torch.cuda.is_available():
            pynvml.nvmlShutdown()

        self.training_time_end = time.time()

        print(f"End-to-end training time with logging: {self.training_time_end - self.training_time_start}.")

    def start_step(self, batch_size: int = None) -> None:
        if batch_size is not None:
            self.batch_size = batch_size
        self._capture_before_step()

    def stop_step(self) -> None:
        """Stop a training step."""
        self.last_step_system = self._capture_after_step()

    def start_forward(self) -> None:
        self._capture_before_substep()

    def stop_forward(self) -> None:
        self._log_substep("forward")

    def start_backward(self) -> None:
        self._capture_before_substep()

    def stop_backward(self) -> None:
        self._log_substep("backward")

    def start_optimizer_step(self) -> None:
        self._capture_before_substep()

    def stop_optimizer_step(self) -> None:
        self._log_substep("optimizer")

    def log_loss(self, loss: torch.Tensor) -> None:
        """Logs the loss of the current step by passing it to the stats."""
        pass

    def start_save_checkpoint(self) -> None:
        """Start checkpointing."""
        pass

    def stop_save_checkpoint(self) -> None:
        """Stop checkpointing."""
        pass

    def _get_nvml_average(self):
        if not self._nvml_accumulator["gpu_util"]:
            return

        avg_gpu_util = sum(self._nvml_accumulator["gpu_util"]) / len(self._nvml_accumulator["gpu_util"])

        # Reset accumulator
        self._nvml_accumulator["gpu_util"].clear()
        return avg_gpu_util

    def _generate_plots(self):
        pass

    def log_step(self) -> None:
        """Logs information about the previous step."""
        sys = getattr(self, "last_step_system", {})
        step_time = sys.get("time_sec", 0)
        global_batch = (self.batch_size or 0) * self.world_size

        throughput = 0.0
        if step_time > 0 and global_batch > 0:
            throughput = global_batch / step_time

        row = {
            "step": self.step_idx,
            "time_sec": sys.get("time_sec", 0),
            "ram_mb_abs": sys.get("ram_mb_abs", 0),
            "ram_delta_mb": sys.get("ram_delta_mb", 0),
            "io_read_mb_abs": sys.get("io_read_mb_abs", 0),
            "io_write_mb_abs": sys.get("io_write_mb_abs", 0),
            "io_read_mb": sys.get("io_read_mb", 0),
            "io_write_mb": sys.get("io_write_mb", 0),
            "gpu_mem_used_mb": sys.get("gpu_memory_used_mb", 0),
            "cpu_util_percent": sys.get("cpu_util_percent", 0),
            "throughput_samples_per_sec": throughput,
            "global_batch_size": global_batch,
        }
        self._nvml_accumulator["gpu_util"].append(sys.get("gpu_util_moment", 0))

        if self.step_idx % self.nvml_log_interval == 0:
            avg_gpu_util = self._get_nvml_average()
            row["gpu_util_moment"] = avg_gpu_util
        else:
            row["gpu_util_moment"] = None

        # Initialize writer with header after first row
        if self.step_csv_writer is None:
            self.step_csv_writer = csv.DictWriter(self.step_csv_file, fieldnames=row.keys())
            self.step_csv_writer.writeheader()

        self.step_csv_writer.writerow(row)
        self.step_csv_file.flush()  # important for long runs

        self.step_idx += 1

    def _log_substep(self, name: str):
        stats = self._capture_after_substep()

        row = {
            "step": self.step_idx,
            "substep": name,
            "time_sec": stats.get("time_sec", 0),
            "ram_mb_abs": stats.get("ram_mb_abs", 0),
            "io_read_mb_abs": stats.get("io_read_mb_abs", 0),
            "io_write_mb_abs": stats.get("io_write_mb_abs", 0),
            "gpu_mem_used_mb": stats.get("gpu_memory_used_mb", 0),
            "cpu_util_percent": stats.get("cpu_util_percent", 0),
        }

        if self.substep_csv_writer is None:
            self.substep_csv_writer = csv.DictWriter(
                self.substep_csv_file,
                fieldnames=row.keys()
            )
            self.substep_csv_writer.writeheader()

        self.substep_csv_writer.writerow(row)
        self.substep_csv_file.flush()

    def log_stats(self) -> None:
        """Logs information about the data accumulated so far."""
        pass

