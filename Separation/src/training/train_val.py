"""
The main training script for training on synthetic data
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm


def to_device(batch, device):
    if type(batch) == torch.Tensor:
        return batch.to(device)
    elif type(batch) == dict:
        for k in batch:
            batch[k] = to_device(batch[k], device)
        return batch
    elif type(batch) in [list, tuple]:
        batch = [to_device(x, device) for x in batch]
        return batch
    else:
        return batch

def test_epoch(hl_module, test_loader, device, early_quit = -1) -> float:
    """
    Evaluate the network.
    """
    hl_module.eval()
    
    test_loss = 0
    num_elements = 0

    if early_quit > 0:
        num_batches =  early_quit
    else:
        num_batches = len(test_loader)

    pbar = tqdm.tqdm(total=num_batches)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
            batch = to_device(batch, device)
            loss, B = hl_module.validation_step(batch, batch_idx)
            #print(loss.item(), B)
            test_loss += (loss.item() * B)
            num_elements += B

            pbar.set_postfix(loss='%.05f'%(loss.item()) )
            pbar.update()

        return test_loss / num_elements

def train_epoch(hl_module, train_loader, device, early_quit = -1) -> float:
    """
    Train a single epoch.
    """    
    # Set the model to training.
    hl_module.train()
    
    # Training loop
    train_loss = 0
    num_elements = 0
    if early_quit > 0:
        num_batches =  early_quit
    else:
        num_batches = len(train_loader)
    pbar = tqdm.tqdm(total=num_batches)
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        batch = to_device(batch, device)

        # Reset grad
        hl_module.reset_grad()
        
        # Forward pass
        loss, B = hl_module.training_step(batch, batch_idx)

        # Backpropagation
        loss.backward(retain_graph=False)
        hl_module.backprop()

        # Save losses
        loss = loss.detach() 
        train_loss += (loss.item() * B)
        num_elements += B
#        if batch_idx % 20 == 0:
#            print(loss.item(), B)
#            print('{}/{}'.format(batch_idx, num_batches))
        pbar.set_postfix(loss='%.05f'%(loss.item()) )
        pbar.update()

    return train_loss / num_elements
