"""
The main training script for training on synthetic data
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
from src.acc_modules.acc_utils import LOGGER


def test_epoch(hl_module, test_loader) -> float:
    """
    Evaluate the network.
    """
    hl_module.eval()
    
    test_loss = 0
    num_elements = 0

    if LOGGER.rank == 0:
        num_batches = len(test_loader)
        pbar = tqdm.tqdm(total=num_batches)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):            
            loss, B, sample = hl_module.validation_step(batch, batch_idx)
            
            test_loss += (loss.item() * B)
            num_elements += B

            if LOGGER.rank == 0:
                pbar.set_postfix(loss='%.05f'%(loss.item()) )
                pbar.update()

        return test_loss / num_elements

def train_epoch(hl_module, train_loader) -> float:
    """
    Train a single epoch.
    """    
    # Set the model to training.
    hl_module.train()
    
    # Training loop
    train_loss = 0
    num_elements = 0

    
    if LOGGER.rank == 0:
        num_batches = len(train_loader)
        pbar = tqdm.tqdm(total=num_batches)
    
    for batch_idx, batch in enumerate(train_loader):
        # Reset grad
        hl_module.reset_grad()
        
        # Forward pass
        loss, B, sample = hl_module.training_step(batch, batch_idx)

        # Backpropagation
        hl_module.backprop(loss)

        # Save losses
        loss = loss.detach() 
        train_loss += (loss.item() * B)
        num_elements += B

        if LOGGER.rank == 0:
            pbar.set_postfix(loss='%.05f'%(loss.item()) )
            pbar.update()

    return train_loss / num_elements
