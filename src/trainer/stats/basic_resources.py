import psutil

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
import numpy as np
import matplotlib.pyplot as plt
import threading
import multiprocessing as mp

logger = logging.getLogger(__name__)

trainer_stats_name="basic_resources_stats"

# aim for CPU logging interval of around 100 ms
pre_computed_cpu_interval = {
    "batch_size_32": 1,
    "batch_size_64": 1,
    "batch_size_128": 1,
}

# aim for GPU logging interval of around 160 ms
pre_computed_gpu_interval = {
    "batch_size_32": 1, # step time is around 170 ms, so log every step
    "batch_size_64": 1, # step time is around 330 ms, so log every step
    "batch_size_128": 1,    # step time is around 660 ms, so log every step
}

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to basic resource trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    return BasicResourcesStats(device=device, output_path=conf.trainer_stats_configs.basic_resources.output_dir, batch_size=conf.batch_size)

class BasicResourcesStats(base.TrainerStats):
    """Stats class that tracks GPU utilization, memory consumption, and I/O."""

    def __init__(self, device: torch.device, output_path: str, csv_name: str ="basic_resources_stats", batch_size: int = None) -> None:
        super().__init__()

        self.process = psutil.Process(os.getpid())  

        self.device = device 

        if torch.cuda.is_available():
            self.nvml_initialized = False
            if torch.cuda.is_available():
                pynvml.nvmlInit()
                self.nvml_initialized = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.device.index if self.device.index is not None else torch.cuda.current_device()
            )
        else:
            self.gpu_handle = None

        self.output_path = output_path
        self.logging_timestamp = int(time.perf_counter())   # used to differentiate logs from different runs

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.step_idx = 0
        self.batch_size = batch_size if batch_size is not None else 1
        self.world_size = 1

        self.step_csv_path = Path(f"{self.output_path}/{csv_name}_steps_{self.logging_timestamp}_batch_size_{self.batch_size}.csv")
        self.step_file = None
        self.step_writer = None

        self.substeps_csv_path = Path(f"{self.output_path}/{csv_name}_substeps_{self.logging_timestamp}_batch_size_{self.batch_size}.csv")
        self.substep_file = None
        self.substep_writer = None

        self.timeline_csv_path = Path(
            f"{self.output_path}/{csv_name}_counter_{self.logging_timestamp}_batch_size_{self.batch_size}.csv"
        )
        self.timeline_file = None
        self.timeline_writer = None

        ctx = mp.get_context("spawn")
        self.sampling_flag = ctx.Value('b', True)

        self.cpu_interval = 0.1     # 100 ms
        self.gpu_interval = 0.166   # 166 ms

        # to ensure that logging from the sampler thread doesn't interfere with the main training thread
        self.lock = threading.Lock()

    def _cuda_sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def start_train(self) -> None:
        self._cuda_sync()
        self.training_time_start = time.perf_counter()

        self.step_file = open(self.step_csv_path, mode="w", newline="")
        self.step_writer = csv.DictWriter(
            self.step_file, 
            fieldnames=["step", "step_end_timestamp", "time_sec", "throughput_samples_per_sec", "ram_mb", "gpu_mem_used_mb", "gpu_util", "cpu_util"]
        )
        self.step_writer.writeheader()

        self.substep_file = open(self.substeps_csv_path, mode="w", newline="")
        self.substep_writer = csv.DictWriter(
            self.substep_file,
            fieldnames=["step", "substep_end_timestamp", "substep", "time_sec"]
        )
        self.substep_writer.writeheader()

        self.timeline_file = open(self.timeline_csv_path, mode="w", newline="")
        self.timeline_writer = csv.DictWriter(
            self.timeline_file, 
            fieldnames=["timestamp", "cpu_util_percent", "gpu_util_percent"]
        )
        self.timeline_writer.writeheader()



    def stop_train(self) -> None:
        self._cuda_sync()

        self.sampling_flag.value = False
        self.sampler_thread.join()

        if self.step_file:
            self.step_file.close()
        if self.substep_file:
            self.substep_file.close()
        if self.timeline_file:
            self.timeline_file.close()

        if torch.cuda.is_available():
            if self.nvml_initialized:
                pynvml.nvmlShutdown()

        total_time = time.perf_counter() - self.training_time_start

        print(f"End-to-end training time with logging: {total_time}.")
        print("Saved stats to:", self.output_path)

    def start_step(self) -> None:
        self.time_before_step = time.perf_counter()
        self.cpu_before_step = self.process.cpu_times()

    def stop_step(self) -> None:
        gpu_mem = (
            torch.cuda.memory_allocated(self.device) / 1024**2
            if torch.cuda.is_available() else 0
        )
        global_batch = self.batch_size * self.world_size
        throughput = global_batch / step_time if step_time > 0 else 0

        time_after = time.perf_counter()
        ram_mem = psutil.virtual_memory().used / 1024**2

        step_time = time_after - self.time_before_step

        self.step_writer.writerow({
            "step": self.step_idx,
            "step_end_timestamp": time.perf_counter(),
            "time_sec": step_time,
            "throughput_samples_per_sec": throughput,
            "ram_mb": ram_mem,
            "gpu_mem_used_mb": gpu_mem,
        })

        # log GPU util every 4th step to avoid overhead of NVML calls, which can be slow
        if self.step_idx % pre_computed_gpu_interval[f"batch_size_{self.batch_size}"] == 0:
            gpu_util = (
                pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                if self.gpu_handle else None
            )
            self.timeline_writer.writerow({
                "timestamp": step_time,
                "cpu_util_percent": None,
                "gpu_util_percent": gpu_util,
            })

        if self.step_idx % pre_computed_cpu_interval[f"batch_size_{self.batch_size}"] == 0:
            cpu_util = psutil.cpu_percent(interval=None)
            self.timeline_writer.writerow({
                "timestamp": step_time,
                "cpu_util_percent": cpu_util,
                "gpu_util_percent": None,
            })

        self.step_idx += 1

    # -------------------------------
    # -------------------------------
    # Substep Monitoring
    # -------------------------------
    # -------------------------------

    def start_forward(self) -> None:
        self._cuda_sync()
        self.substep_start = time.perf_counter()

    def stop_forward(self) -> None:
        self._log_substep("forward")

    def start_backward(self) -> None:
        self._cuda_sync()
        self.substep_start = time.perf_counter()

    def stop_backward(self) -> None:
        self._log_substep("backward")

    def start_optimizer_step(self) -> None:
        self._cuda_sync()
        self.substep_start = time.perf_counter()

    def stop_optimizer_step(self) -> None:
        self._log_substep("optimizer")

    def _log_substep(self, name: str):
        self._cuda_sync()

        duration = time.perf_counter() - self.substep_start

        row = {
            "step": self.step_idx,
            "substep_end_timestamp": time.perf_counter(),
            "substep": name,
            "time_sec": duration,
        }

        with self.lock:
            self.substep_writer.writerow(row)

    def log_loss(self, loss: torch.Tensor) -> None:
        """Logs the loss of the current step by passing it to the stats."""
        pass

    def start_save_checkpoint(self) -> None:
        """Start checkpointing."""
        pass

    def stop_save_checkpoint(self) -> None:
        """Stop checkpointing."""
        pass

    def log_stats(self) -> None:
        """Logs information about the data accumulated so far."""
        pass

    def log_step(self) -> None:
        """Logs information about the data accumulated so far."""
        pass