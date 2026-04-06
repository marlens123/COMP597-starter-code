import src.config as config
import src.trainer.stats.base as base
import torch
import time
from pathlib import Path
import csv

trainer_stats_name="end_to_end_time_stats"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to basic resource trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    return EndToEndTrainerStats(device=device, batch_size=conf.batch_size, seed=conf.seed)

class EndToEndTrainerStats(base.TrainerStats):
    """
    Tracks the time needed for an end-to-end run.
    """

    def __init__(self, device, csv_name: str ="time_baseline", batch_size: int=None, seed: int=0) -> None:
        super().__init__()

        self.seed = seed
        self.batch_size = batch_size
        self.device = device

        self.logging_timestamp = int(time.perf_counter_ns())   # used to differentiate logs from different runs

        self.total_csv_path = Path(f"{csv_name}_{self.logging_timestamp}_batch_size_{self.batch_size}_seed_{self.seed}.csv")

        self.total_file = open(self.total_csv_path, mode="w", newline="")
        self.total_writer = csv.DictWriter(
            self.total_file, 
            fieldnames=["total_training_time"]
        )
        self.total_writer.writeheader()

    def _cuda_sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def start_train(self) -> None:
        self._cuda_sync()
        self.training_time_start = time.perf_counter_ns()

    def stop_train(self) -> None:
        self._cuda_sync()
        total_time = time.perf_counter_ns() - self.training_time_start
        self.total_writer.writerow({
            "total_training_time": total_time
        })
        self.total_file.close()

    def start_step(self) -> None:
        pass

    def stop_step(self) -> None:
        pass

    def start_dataloading(self) -> None:
        pass

    def stop_dataloading(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_forward(self) -> None:
        pass

    def stop_forward(self) -> None:
        pass

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass
    
    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass
