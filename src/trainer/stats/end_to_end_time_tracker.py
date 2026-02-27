import src.config as config
import src.trainer.stats.base as base
import torch
from time import time

trainer_stats_name="end_to_end_stats"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    return EndToEndTrainerStats()

class EndToEndTrainerStats(base.TrainerStats):
    """
    Tracks the time needed for an end-to-end run.
    """

    def __init__(self) -> None:
        super().__init__()

        self.start_time = None
        self.end_time = None

    def start_train(self) -> None:
        self.start_time = time()

    def stop_train(self) -> None:
        self.end_time = time()
        print(f"End-to-end time without logging: {self.end_time - self.start_time}.")

    def start_step(self, batch_size: int = None) -> None:
        pass

    def stop_step(self) -> None:
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
