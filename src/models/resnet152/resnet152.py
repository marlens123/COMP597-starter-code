# === import necessary modules ===
import src.config as config # Configurations
import src.trainer as trainer # Trainer base class
import src.trainer.stats as trainer_stats # Trainer statistics module
from src.data.dataset.dataloader import imagenet_dataloader

# === import necessary external modules ===
from typing import Dict, Optional, Tuple
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import resnet152
import argparse
from pathlib import Path
import torch

"""
This file contains the code to train a ResNet152 model using Simple trainer (src/trainer/simple.py).
It is based on the ResNet152 model from TorchVision.
https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.resnet152.html
"""

def init_resnet152_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    """
    Initializes the optimizer for the ResNet152 model.
    Args:
        conf (config.Config): The configuration object.
        model (nn.Module): The ResNet152 model.
    Returns:
        optim.Optimizer: The initialized optimizer.
    """
    # Note: The learning rate is taken from the configuration object. Adjust it as needed for different models and training setups based on the loss function.
    # Here we use SGD as in the Mila Benchmark
    return optim.SGD(model.parameters(), lr=conf.learning_rate)

def pre_init_resnet152() -> resnet152:
    """
    Prepares the ResNet152 model, dataset, tokenizer and data collator for training.
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
    Returns:
        Tuple[transformers.PreTrainedModel, data.Dataset, transformers.PreTrainedTokenizer, transformers.DataCollatorForLanguageModeling]: The GPT-2 model, dataset, tokenizer and data collator.
    """
    model = resnet152(weights="DEFAULT")
    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(model.device)
    return model

################################################################################
#################################    Simple    #################################
################################################################################

def resnet_simple_trainer(conf : config.Config, model : resnet152, dataset : data.Dataset, dataloader_args) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Simple trainer for ResNet152 model. Uses the SimpleTrainer from src/trainer/simple.py.
    Args:
        conf (config.Config): The configuration object.
        model (transformers.GPT2LMHeadModel): The ResNet152 model to train.
        dataset (data.Dataset): The dataset to train on.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        data_collator (transformers.DataCollatorForLanguageModeling): The data collator to use.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The simple trainer and a dictionary with additional options.
    """ 
    loader = imagenet_dataloader(dataset, dataloader_args, model)

    model = model.cuda() # Move the model to GPU
    optimizer = init_resnet152_optim(conf, model) # Initialize the optimizer for ResNet152 
    scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer) # linear scheduler

    # Return the SimpleTrainer with the initialized components
    return trainer.ResNetSimpleTrainer(loader=loader, model=model, optimizer=optimizer, lr_scheduler=scheduler, device=model.device, stats=trainer_stats.init_from_conf(conf=conf, device=model.device, num_train_steps=len(loader))), None

################################################################################
##################################    Init    ##################################
################################################################################

def resnet152_init(conf: config.Config, dataset: torch.utils.data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Initializes the ResNet152 model and returns the appropriate trainer based on the configuration.
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The initialized trainer and a dictionary with additional options.
    """

    class DataloaderArgs:
        batch_size = conf.batch_size
        loader = "pytorch"
        num_workers = 0

    dataloader_args = DataloaderArgs()

    model = pre_init_resnet152()
    # Note: Currently, only ResNetSimple trainer is implemented for ResNet152. Add more trainers as needed.
    if conf.trainer == "resnet_simple": 
        return resnet_simple_trainer(conf, model, dataset, dataloader_args)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")

