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

logger = logging.getLogger(__name__)

trainer_stats_name="basic_resources_stats"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to basic resource trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    return BasicResourcesStats(device=device, output_path=conf.trainer_stats_configs.basic_resources.output_dir)

class BasicResourcesStats(base.TrainerStats):
    """Stats class that tracks GPU utilization, memory consumption, and I/O."""

    def __init__(self, device: torch.device, output_path: str, csv_name: str ="basic_resources_stats") -> None:
        super().__init__()

        self.process = psutil.Process(os.getpid())  

        self.device = device 

        if torch.cuda.is_available():
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.device.index if self.device.index is not None else torch.cuda.current_device()
            )
        else:
            self.gpu_handle = None

        self.nvml_log_interval = 15

        self._nvml_accumulator = {
            "gpu_util": [],
        }

        self.output_path = output_path
        self.logging_timestamp = int(time.time())

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.step_csv_path = Path(f"{self.output_path}/{csv_name}_{self.logging_timestamp}.csv")
        self.substeps_csv_path = Path(f"{self.output_path}/{csv_name}_substeps_{self.logging_timestamp}.csv")

        self.step_file = None
        self.step_writer = None
        self.substep_file = None
        self.substep_writer = None

        self.step_idx = 0
        self.batch_size = None
        self.world_size = 1

    def _cuda_sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def start_train(self) -> None:
        """Initialize CSV logging."""
        self.training_time_start = time.time()

        self.step_file = open(self.step_csv_path, mode="w", newline="")
        self.substep_file = open(self.substeps_csv_path, mode="w", newline="")

    def stop_train(self) -> None:
        """Close CSV file."""
        if self.step_file:
            self.step_file.close()
        if self.substep_file:
            self.substep_file.close()

        if torch.cuda.is_available():
            pynvml.nvmlShutdown()

        total_time = time.time() - self.training_time_start

        print(f"End-to-end training time with logging: {total_time}.")
        self._generate_timeline_plots()
        self._generate_substep_plot()

        print("Saved stats & plots to:", self.output_path)

    def start_step(self, batch_size: int = None) -> None:
        if batch_size is not None:
            self.batch_size = batch_size

        self._cuda_sync()

        self.time_before_step = time.time()
        self.cpu_before_step = self.process.cpu_times()

    def stop_step(self) -> None:
        self._cuda_sync()

        time_after = time.time()
        ram_mem = psutil.virtual_memory().used / 1024**2

        step_time = time_after - self.time_before_step

        if self.gpu_handle:
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
        else:
            gpu_util = 0

        gpu_mem = (
            torch.cuda.memory_allocated() / 1024**2
            if torch.cuda.is_available() else 0
        )

        cpu_util = psutil.cpu_percent(interval=None)

        global_batch = self.batch_size * self.world_size
        throughput = global_batch / step_time if step_time > 0 else 0

        row = {
            "step": self.step_idx,
            "step_end_timestamp": time.time(),
            "time_sec": step_time,
            "throughput_samples_per_sec": throughput,
            "cpu_util_percent": cpu_util,
            "ram_mb": ram_mem,
            "gpu_util_percent": gpu_util,
            "gpu_mem_used_mb": gpu_mem,
        }

        self._nvml_accumulator["gpu_util"].append(gpu_util)

        if self.step_idx % self.nvml_log_interval == 0:
            avg_gpu_util = self._get_nvml_average()
            row["gpu_util_percent"] = avg_gpu_util
        else:
            row["gpu_util_percent"] = None

        if self.step_writer is None:
            self.step_writer = csv.DictWriter(self.step_file, fieldnames=row.keys())
            self.step_writer.writeheader()

        self.step_writer.writerow(row)

        self.step_idx += 1

    def _get_nvml_average(self):
        if not self._nvml_accumulator["gpu_util"]:
            return

        avg_gpu_util = sum(self._nvml_accumulator["gpu_util"]) / len(self._nvml_accumulator["gpu_util"])

        # Reset accumulator
        self._nvml_accumulator["gpu_util"].clear()
        return avg_gpu_util

    # -------------------------------
    # -------------------------------
    # Substep Monitoring
    # -------------------------------
    # -------------------------------

    def start_forward(self) -> None:
        self._cuda_sync()
        self.substep_start = time.time()

    def stop_forward(self) -> None:
        self._log_substep("forward")

    def start_backward(self) -> None:
        self._cuda_sync()
        self.substep_start = time.time()

    def stop_backward(self) -> None:
        self._log_substep("backward")

    def start_optimizer_step(self) -> None:
        self._cuda_sync()
        self.substep_start = time.time()

    def stop_optimizer_step(self) -> None:
        self._log_substep("optimizer")

    def _log_substep(self, name: str):
        self._cuda_sync()

        duration = time.time() - self.substep_start

        gpu_mem = (
            torch.cuda.memory_allocated() / 1024**2
            if torch.cuda.is_available() else 0
        )

        row = {
            "step": self.step_idx,
            "substep": name,
            "time_sec": duration,
            "gpu_mem_used_mb": gpu_mem,
        }

        if self.substep_writer is None:
            self.substep_writer = csv.DictWriter(
                self.substep_file,
                fieldnames=row.keys()
            )
            self.substep_writer.writeheader()

        self.substep_writer.writerow(row)

    def _generate_timeline_plots(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path

        df = pd.read_csv(self.step_csv_path)

        if df.empty:
            logger.warning("No timeline data available.")
            return

        # Convert timestamp to relative seconds
        df["t"] = df["step_end_timestamp"] - df["step_end_timestamp"].iloc[0]

        # only plot the first 5 minutes
        df = df[df["t"] <= 300]

        fig, axes = plt.subplots(
            2, 3,
            figsize=(18, 8),
            sharex=True
        )
        fig.suptitle('ResNet152 Basic Training Metrics', fontsize=16, fontweight='bold')

        for ax in axes.flat:
            ax.tick_params(labelbottom=True)

        # GPU Util
        y = pd.to_numeric(df["gpu_util_percent"], errors="coerce").fillna(0)

        axes[0,0].plot(df["t"], y, linewidth=1.5)
        axes[0,0].set_ylabel("GPU Util (%)")
        axes[0,0].grid(alpha=0.3)

        avg = np.mean(y)
        ax.set_title(f"GPU Utilization (Average: {avg:.2f})")
        ax.set_xlabel("Time (seconds)")

        axes[0,0].set_ylim(bottom=0)
        if df["gpu_util_percent"].max() > 0:
            axes[0,0].set_ylim(top=df["gpu_util_percent"].max() * 1.1)

        # CPU Util
        y = pd.to_numeric(df["cpu_util_percent"], errors="coerce").fillna(0)

        axes[0,1].plot(df["t"], y, linewidth=1.5)
        axes[0,1].set_ylabel("CPU Util (%)")
        axes[0,1].grid(alpha=0.3)

        avg = np.mean(y)
        ax.set_title(f"CPU Utilization (Average: {avg:.2f})")
        ax.set_xlabel("Time (seconds)")

        axes[0,1].set_ylim(bottom=0)
        if df["cpu_util_percent"].max() > 0:
            axes[0,1].set_ylim(top=df["cpu_util_percent"].max() * 1.1)

        # GPU Memory
        y = pd.to_numeric(df["gpu_mem_used_mb"], errors="coerce").fillna(0)

        axes[0,2].plot(df["t"], y, linewidth=1.5)
        axes[0,2].set_ylabel("GPU Mem (MB)")
        axes[0,2].grid(alpha=0.3)

        avg = np.mean(y)
        ax.set_title(f"GPU Memory (Average: {avg:.2f})")
        ax.set_xlabel("Time (seconds)")

        axes[0,2].set_ylim(bottom=0)
        if df["gpu_mem_used_mb"].max() > 0:
            axes[0,2].set_ylim(top=df["gpu_mem_used_mb"].max() * 1.1)

        # RAM
        y = pd.to_numeric(df["ram_mb"], errors="coerce").fillna(0)

        axes[1,0].plot(df["t"], y, linewidth=1.5)
        axes[1,0].set_ylabel("RAM (MB)")
        axes[1,0].grid(alpha=0.3)

        avg = np.mean(y)
        ax.set_title(f"RAM Usage (Average: {avg:.2f})")
        ax.set_xlabel("Time (seconds)")

        axes[1,0].set_ylim(bottom=0)
        if df["ram_mb"].max() > 0:
            axes[1,0].set_ylim(top=df["ram_mb"].max() * 1.1)

        # Throughput
        y = pd.to_numeric(df["throughput_samples_per_sec"], errors="coerce").fillna(0)

        axes[1,1].plot(df["t"], y, linewidth=1.5)
        axes[1,1].set_ylabel("Samples / sec")
        axes[1,1].grid(alpha=0.3)

        avg = np.mean(y)
        ax.set_title(f"Throughput (Average: {avg:.2f})")
        ax.set_xlabel("Time (seconds)")

        axes[1,1].set_ylim(bottom=0)
        if df["throughput_samples_per_sec"].max() > 0:
            axes[1,1].set_ylim(top=df["throughput_samples_per_sec"].max() * 1.1)

        # Time per step
        y = pd.to_numeric(df["time_sec"], errors="coerce").fillna(0)

        axes[1,2].plot(df["step"], y, linewidth=1.5)
        axes[1,2].set_ylabel("Time (sec)")
        axes[1,2].grid(alpha=0.3)

        avg = np.mean(y)
        ax.set_title(f"Step Time")
        ax.set_xlabel("Step")

        axes[1,2].set_ylim(bottom=0)
        if df["time_sec"].max() > 0:
            axes[1,2].set_ylim(top=df["time_sec"].max() * 1.1)

        plt.tight_layout()
        output = Path(self.output_path) / f"timeline_{self.logging_timestamp}.png"
        plt.savefig(output, dpi=150)
        plt.close()

        logger.info(f"Saved timeline plot to {output}")

    def _generate_substep_plot(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path

        sub_df = pd.read_csv(self.substeps_csv_path)

        if sub_df.empty:
            logger.warning("No substep data available.")
            return

        # Only phase timing
        phase_df = sub_df[sub_df["substep"].isin(["forward", "backward", "optimizer"])]

        grouped = phase_df.groupby("substep")["time_sec"]

        means = grouped.mean()
        stds = grouped.std()

        phases = means.index

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle('ResNet152 Phase Metrics', fontsize=16, fontweight='bold')

        ax.bar(phases, means, yerr=stds, capsize=5)

        ax.set_ylabel("Time per Phase (sec)")
        ax.set_title("Mean Phase Time ± Std Dev")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        output = Path(self.output_path) / f"phase_bars_{self.logging_timestamp}.png"
        plt.savefig(output, dpi=150)
        plt.close()

        logger.info(f"Saved phase bar plot to {output}")

    def _generate_plots(self):
        import pandas as pd
        import matplotlib.pyplot as plt

        # =========================
        # Step duration plots
        # =========================
        df = pd.read_csv(self.step_csv_path)

        if df.empty:
            logger.warning("No data available for plotting.")
            return

        x = df["step"]

        fig, axes = plt.subplots(
            2, 3,
            figsize=(18, 8),
            sharex=True
        )
        fig.suptitle('ResNet152 Basic Training Metrics', fontsize=16, fontweight='bold')

        for ax in axes.flat:
            ax.tick_params(labelbottom=True)

        def _single_plot(ax, x, y_column, ylabel, title):

            y = pd.to_numeric(df[y_column], errors="coerce").fillna(0)

            ax.plot(x, y, linewidth=1.5)

            avg = np.mean(y)

            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} (Average: {avg:.2f})")
            ax.grid(alpha=0.3)
            ax.set_xlabel("Step")

            ax.set_ylim(bottom=0)
            if y.max() > 0:
                ax.set_ylim(top=y.max() * 1.1)

        # Row 1
        _single_plot(axes[0,0], x, "time_sec", "Time (sec)", "Step Time")
        _single_plot(axes[0,0], x, "gpu_util_percent", "GPU Util (%)", "GPU Utilization")
        _single_plot(axes[0,1], x, "gpu_mem_used_mb", "GPU Mem (MB)", "GPU Memory")

        _single_plot(axes[1,0], x, "throughput_samples_per_sec", "Samples / sec", "Throughput")
        _single_plot(axes[1,1], x, "cpu_util_percent", "CPU Util (%)", "CPU Utilization")
        _single_plot(axes[0,2], x, "ram_mb", "RAM (MB)", "RAM Usage")

        # Remove unused subplot instead of leaving it empty
        fig.delaxes(axes[1,2])

        plt.tight_layout()

        # Save figure
        output_file = Path(self.output_path) / f"basic_training_metrics_{self.logging_timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        # =========================
        # Substep duration plots
        # =========================

        sub_df = pd.read_csv(self.substeps_csv_path)

        if sub_df.empty:
            logger.warning("No substep data available for plotting.")
            return

        # Ensure numeric
        sub_df["time_sec"] = pd.to_numeric(sub_df["time_sec"], errors="coerce").fillna(0)

        # Pivot to get forward/backward/optimizer per step
        pivot = sub_df.pivot_table(
            index="step",
            columns="substep",
            values="time_sec",
            aggfunc="sum"
        ).fillna(0)

        steps = pivot.index

        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

        # --------------------------------
        # (1) Substep duration over steps
        # --------------------------------
        for col in pivot.columns:
            axes2[0].plot(steps, pivot[col], label=col)

        axes2[0].set_title("Substep Duration per Step")
        axes2[0].set_xlabel("Step")
        axes2[0].set_ylabel("Time (sec)")
        axes2[0].grid(alpha=0.3)
        axes2[0].legend()

        # --------------------------------
        # (2) Total accumulated time
        # --------------------------------
        total_times = pivot.sum()

        axes2[1].bar(total_times.index, total_times.values)

        axes2[1].set_title("Total Time Spent per Substep")
        axes2[1].set_ylabel("Total Time (sec)")
        axes2[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()

        substep_output = Path(self.output_path) / f"substep_durations_{self.logging_timestamp}.png"
        plt.savefig(substep_output, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved substep plots to {substep_output}")


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