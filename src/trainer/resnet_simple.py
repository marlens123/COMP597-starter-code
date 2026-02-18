from typing import Optional, Dict, Any
import tqdm
import torch

def train(self, model_kwargs: Optional[Dict[str, Any]] = None) -> None:
    """
    Memory-safe training loop for a single epoch.
    Logs only scalars and avoids accumulating tensors.
    """
    model_kwargs = model_kwargs or {}
    self.stats.start_train()
    
    # Wrap dataloader with tqdm
    progress_bar = tqdm.auto.tqdm(self.loader, desc="Training", unit="batch")
    
    for i, batch in enumerate(progress_bar):
        # Determine batch size safely
        if isinstance(batch, (list, tuple)):
            batch_size = batch[0].size(0)
        else:
            batch_size = len(batch)
        self.stats.start_step(batch_size=batch_size)
        
        # Process batch (moves tensors to device)
        batch = self.process_batch(i, batch)
        
        # Forward + loss
        loss = self.forward(i, batch, model_kwargs)
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)  # frees old grad buffers
        
        # Detach and log scalar loss only
        loss_scalar = loss.item()
        self.stats.log_loss(loss_scalar)
        self.stats.log_step()
        
        # Optional description for progress bar
        progress_bar.set_postfix({"loss": f"{loss_scalar:.4f}"})
        
        # Checkpointing
        if self.enable_checkpointing and self.should_save_checkpoint(i):
            self.stats.start_save_checkpoint()
            self.save_checkpoint(i)
            self.stats.stop_save_checkpoint()
        
        self.stats.stop_step()
    
    self.stats.stop_train()
    progress_bar.close()
    self.stats.log_stats()
