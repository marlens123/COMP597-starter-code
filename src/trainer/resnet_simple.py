from src.trainer.simple import SimpleTrainer
from typing import Any, Dict
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as data
import src.trainer.stats as stats
from typing import Optional, override
import src.config as config
import tqdm
import time

# Pre-computed approximations of the number of steps needed to train for 5 minutes (including logging overhead)
# The first step is always much slower than the rest, so we exclude it from the calculations.
pre_computed_num_steps = {
    "batch_size_32": 3750,
    "batch_size_64": 1250,
    "batch_size_128": 1000,
}

class ResNetSimpleTrainer(SimpleTrainer):
    """Wrapper around SimpleTrainer for ResNet-specific training."""
    def __init__(self, 
                 loader : data.DataLoader, 
                 model : nn.Module, 
                 optimizer : optim.Optimizer, 
                 lr_scheduler : optim.lr_scheduler.LRScheduler, 
                 device : torch.device, 
                 stats : stats.TrainerStats,
                 conf: Optional[config.Config] = None):
        super().__init__(loader=loader, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, stats=stats, conf=conf)

        self.criterion = nn.CrossEntropyLoss().to(self.model.device)

    @override
    def process_batch(self, i : int, batch : Any) -> Any:
        if isinstance(batch, (list, tuple)):
            return [v.to(self.device) for v in batch]
        else:
            raise TypeError(f"Unsupported batch type {type(batch)}")
        
    @override
    def forward(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Defines loss function and makes forward pass applicable to tuple/list inputs.
        """
        self.optimizer.zero_grad() # Zero the gradients
        input, target = batch
        outputs = self.model(input, **model_kwargs)
        return self.criterion(outputs, target)
    
    @override
    def train(self, model_kwargs : Optional[Dict[str, Any]]) -> None:
        """Training loop for the model.
        
        This will execute a training step on each batch provided by the 
        dataloader. The number of iterations is defined by the dataloader 
        provided when the object was constructed. 

        A progress bar is updated after every iteration with the iteration 
        number and the most recent loss.

        Training statistics will be logged after every iteration if the `stats` 
        attribute implements the method `log_step`. Additionally, more 
        statistics will be displayed at the end of training if the `stats` 
        attribute implements the method `log_stats`.

        Parameters
        ----------
        model_kwargs
            Additional arguments that need to be provided to the model during 
            the forward pass.

        Notes
        -----
            This does not support multi-epoch training. If you need training on 
            multiple epochs, you should implement a class that inherits 
            `Trainer` and overrides the `train` method.

        """
        progress_bar = tqdm.auto.tqdm(desc="loss: N/A")
        steps = 0

        self.stats.start_train()

        # loop to enable restarting dataloader if we haven't reached the pre-computed number of steps for 5 minutes of training
        while steps < pre_computed_num_steps.get(f"batch_size_{self.loader.batch_size}", float('inf')):
            for i, batch in enumerate(self.loader):
                self.stats.start_step()
                loss, descr = self.step(i, batch, model_kwargs)
                self.stats.stop_step()

                if self.enable_checkpointing and self.should_save_checkpoint(i):
                    self.stats.start_save_checkpoint()
                    self.save_checkpoint(i)
                    self.stats.stop_save_checkpoint()

                # logging
                self.stats.log_loss(loss)
                self.stats.log_step()

                if descr is not None:
                    progress_bar.clear()
                progress_bar.clear()
                progress_bar.update(1)

                steps += 1

        self.stats.stop_train()
        progress_bar.close()
        self.stats.log_stats()

    def checkpoint_dict(self, i: int) -> Dict[str, Any]:
        super_dict = super().checkpoint_dict(i)
        super_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        return super_dict

    def optimizer_step(self, i: int) -> None:
        self.optimizer.step()