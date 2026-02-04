from src.trainer.simple import SimpleTrainer
from typing import Any, Dict
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as data
import src.trainer.stats as stats
from typing import Optional
import src.config as config

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

    def process_batch(self, i : int, batch : Any) -> Any:
        if isinstance(batch, (list, tuple)):
            return [v.to(self.device) for v in batch]
        else:
            raise TypeError(f"Unsupported batch type {type(batch)}")
        
    def forward(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Defines loss function and makes forward pass applicable to tuple/list inputs.
        """
        self.optimizer.zero_grad() #Zero the gradients
        criterion = nn.CrossEntropyLoss().to(self.model.device)
        input, target = batch
        outputs = self.model(input, **model_kwargs)
        return criterion(outputs, target)